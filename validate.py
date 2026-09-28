#!/usr/bin/env python3
"""
Backtest validation, run by CI on every push. Three kinds of check against a frozen snapshot
of real prices (validation/prices.csv.gz), so results only change when the code does:

  1. Calculations  every number is sane: probabilities in [0, 1], scores in [0, 100], no inf,
                   grades consistent with their criteria, confidence intervals around estimates.
  2. Guidance      the claims the pages make about their own method still hold: planted bounce
                   patterns are detected, random walks aren't, regime notes match the multipliers.
  3. Findings      screener + dip-research results on the snapshot match the approved baseline
                   (validation/baseline.json). Any change fails CI and prints what moved.

  python validate.py            run all checks
  python validate.py --update   run them, then approve the current findings as the new baseline
                                (do this when a change is SUPPOSED to move the findings; commit it)
  python validate.py --fetch    re-download the price snapshot (then --update)
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import dip_backtest as D
import screener as S

FIXTURE = Path("validation/prices.csv.gz")
BASELINE = Path("validation/baseline.json")
# Holdings, some peers, and one or two names from each other sector, plus the VIX pair.
TICKERS = D.POSITIONS + ["AMD", "AAPL", "TSLA", "CEG", "IWM", "SH", "JPM", "XOM", "JNJ", "PG", "KO",
                         "CAT", "NEE", "HD", "UNH", "LIN", "AMT"] + ["^VIX", "^VIX3M"]
SCREEN_BARS = 1260   # production screens 5 years; the dip research uses the full 10
SIM_SERIES = 200     # simulated series per guidance check (60 was too noisy: +/-6 pts run to run)
SCREEN_FIELDS = ["trend", "weekly_trend", "support_level", "resist_level", "broken_level", "long_score",
                 "fade_score", "support_rate", "support_base", "resist_rate", "resist_base", "respect_profile"]
HEADLINE = {"grade", "best_rule", "status", "support_level", "resist_level", "broken_level", "trend", "vix_regime"}
DIP_FIELDS = ["grade", "best_rule", "score", "status", "trades", "win", "edge", "t_stat", "edge_oos",
              "win_lo", "win_hi", "edge_lo", "edge_hi", "kelly_frac"]


# --------------------------------------------------------------------------- #
# Snapshot
# --------------------------------------------------------------------------- #
def fetch() -> None:
    prices = S.download_prices(TICKERS, "10y")
    missing = sorted(set(TICKERS) - set(prices))
    if missing:
        raise SystemExit(f"Could not download {missing}; not writing a partial snapshot.")
    long = pd.concat({t: d[["Open", "High", "Low", "Close", "Volume"]] for t, d in prices.items()},
                     names=["ticker", "date"]).reset_index()
    FIXTURE.parent.mkdir(exist_ok=True)
    long.to_csv(FIXTURE, index=False, float_format="%.4f")
    print(f"Wrote {FIXTURE}: {len(prices)} tickers, {long['date'].min()} to {long['date'].max()}")


def load() -> dict[str, pd.DataFrame]:
    df = pd.read_csv(FIXTURE, parse_dates=["date"])
    return {t: g.drop(columns="ticker").set_index("date").sort_index() for t, g in df.groupby("ticker")}


# --------------------------------------------------------------------------- #
# Run the real pipeline code on the snapshot
# --------------------------------------------------------------------------- #
def run_pipeline(prices: dict[str, pd.DataFrame]):
    vix = S.vix_context(prices["^VIX"], prices.get("^VIX3M"))
    spy = prices["SPY"]["Close"]
    stocks = {t: d for t, d in prices.items() if not t.startswith("^")}

    recent = {t: d.iloc[-SCREEN_BARS:] for t, d in stocks.items()}
    screen = pd.DataFrame([r for t, d in recent.items() if (r := S.analyze_stock(t, d, spy.iloc[-SCREEN_BARS:]))])
    screen["long_score"] = (screen["long_raw"] * vix["long"]).clip(upper=100)
    screen["fade_score"] = (screen["fade_raw"] * vix["fade"]).clip(upper=100)
    conviction = S.conviction_score(S.market_breadth(screen, recent), vix, screen)

    dip = pd.DataFrame([r[0] for t, d in stocks.items() if (r := D.analyze(t, d, spy.pct_change()))])
    return vix, screen, dip, conviction


def measure_guidance() -> dict:
    """Re-run the simulations behind the pages' claims about their own method."""
    detect = lambda hl: np.mean([D.passes(D.evaluate(D.synth_series("ou", 7000 + i, half_life=hl),
                                                     horizons=(D.PRIMARY_H,))[2]) for i in range(SIM_SERIES)])
    flagged = tested = 0
    for i in range(SIM_SERIES):
        r = S.analyze_stock("RW", D.synth_series("rw", 9000 + i, n=SCREEN_BARS))
        for k in [k[:-2] for k in (r or {}) if k.endswith("_n") and k[:-2] + "_rate" in r]:
            if r[k + "_n"] >= S.MIN_EVENTS:
                tested += 1
                flagged += r[k + "_rate"] - r[k + "_base"] >= S.EXCESS_MIN
    return dict(detect={hl: float(detect(hl)) for hl in {**D.CLAIM_DETECT_MIN, **D.CLAIM_DETECT_MAX}},
                null_any=float(D.null_pass_rate(SIM_SERIES, seed=3)["any"]),
                rw_false_respect=flagged / max(tested, 1))


# --------------------------------------------------------------------------- #
# Checks
# --------------------------------------------------------------------------- #
def _in(x, lo, hi) -> bool:
    return isinstance(x, (int, float, np.number)) and np.isfinite(x) and lo - 1e-9 <= x <= hi + 1e-9


def check_calculations(vix: dict, screen: pd.DataFrame, dip: pd.DataFrame, conviction: dict) -> list[str]:
    bad = []
    for name, df in (("screen", screen), ("dip", dip)):
        num = df.select_dtypes("number")
        for col in num.columns[np.isinf(num.to_numpy(float)).any(axis=0)]:
            bad.append(f"{name}.{col}: contains inf")

    for r in screen.itertuples():
        t = f"screen {r.ticker}"
        for col in ("long_score", "fade_score", "support_score", "resist_score"):
            if not _in(getattr(r, col), 0, 100):
                bad.append(f"{t}: {col}={getattr(r, col)} outside [0, 100]")
        if not _in(r.rsi, 0, 100):
            bad.append(f"{t}: rsi={r.rsi} outside [0, 100]")
        if r.trend not in ("Up", "Mixed", "Down") or r.weekly_trend not in ("Up", "Mixed", "Down"):
            bad.append(f"{t}: unknown trend {r.trend}/{r.weekly_trend}")
        for role, lvl in (("support", r.support_level), ("resist", r.resist_level)):
            if lvl and lvl not in S.LEVELS:
                bad.append(f"{t}: unknown {role} level {lvl!r}")
            if lvl:   # a named level must actually meet the "respected" rule
                rate, base, n = getattr(r, f"{role}_rate"), getattr(r, f"{role}_base"), getattr(r, f"{role}_n")
                if n < S.MIN_EVENTS or rate - base < S.EXCESS_MIN - 1e-9:
                    bad.append(f"{t}: {role} level {lvl} shown but not respected (rate {rate:.2f}, base {base:.2f}, n {n})")
    for col in [c for c in screen.columns if c.endswith(("_rate", "_base"))]:
        v = screen[col].dropna()
        if not v.between(0, 1).all():
            bad.append(f"screen.{col}: values outside [0, 1]")

    for r in dip.itertuples():
        t = f"dip {r.ticker}"
        if r.grade not in ("Consistent", "Mostly", "No", "n/a (decay)"):
            bad.append(f"{t}: unknown grade {r.grade!r}")
        if r.grade != "n/a (decay)" and ((r.grade == "Consistent") != (r.score == 5)
                                         or (r.grade == "Mostly" and r.score != 4)):
            bad.append(f"{t}: grade {r.grade} inconsistent with score {r.score}/5")
        if r.status not in ("Signal today", "Signal in last 3 sessions", "Trend filter off", "Armed"):
            bad.append(f"{t}: unknown status {r.status!r}")
        if r.trades < 0 or (r.trades > 0 and not _in(r.win, 0, 1)):
            bad.append(f"{t}: trades={r.trades}, win={r.win}")
        if np.isfinite(r.win_lo) and not (r.win_lo - 1e-9 <= r.win <= r.win_hi + 1e-9):
            bad.append(f"{t}: win {r.win:.3f} outside its bootstrap interval [{r.win_lo:.3f}, {r.win_hi:.3f}]")
        if np.isfinite(r.edge_lo) and not (r.edge_lo - 1e-9 <= r.edge <= r.edge_hi + 1e-9):
            bad.append(f"{t}: edge {r.edge:.4f} outside its bootstrap interval [{r.edge_lo:.4f}, {r.edge_hi:.4f}]")
        if np.isfinite(r.kelly_frac) and not _in(r.kelly_frac, 0, D.KELLY_CAP):
            bad.append(f"{t}: kelly_frac={r.kelly_frac} outside [0, {D.KELLY_CAP}]")

    if vix["key"] not in S.REGIMES or not _in(vix["pct"], 0, 1):
        bad.append(f"vix: key={vix['key']}, pct={vix['pct']}")
    sc = conviction["score"]
    want = ("Bullish" if sc >= 75 else "Constructive" if sc >= 60 else "Neutral" if sc >= 40
            else "Cautious" if sc >= 25 else "Bearish")
    if not _in(sc, 0, 100) or conviction["label"] != want:
        bad.append(f"conviction: score {sc}, label {conviction['label']}")
    return bad


def check_guidance(g: dict) -> list[tuple[str, str, bool]]:
    """(claim, measured, holds) for every claim the pages make about their own method."""
    rows = []
    for hl, m in D.CLAIM_DETECT_MIN.items():
        rows.append((f"Dip gate detects a planted {hl}-day bounce in >= {m:.0%} of series",
                     f"{g['detect'][hl]:.0%}", g["detect"][hl] >= m))
    for hl, m in D.CLAIM_DETECT_MAX.items():
        rows.append((f"...but a {hl}-day reversion in <= {m:.0%}", f"{g['detect'][hl]:.0%}", g["detect"][hl] <= m))
    rows.append((f"Random walks pass the dip gate in <= {D.CLAIM_NULL_MAX:.0%}",
                 f"{g['null_any']:.0%}", g["null_any"] <= D.CLAIM_NULL_MAX))
    rows.append((f"Random walks show <= {S.CLAIM_RW_RESPECT_MAX:.0%} of tested levels as respected",
                 f"{g['rw_false_respect']:.1%}", g["rw_false_respect"] <= S.CLAIM_RW_RESPECT_MAX))
    # Each VIX regime's note must state the multipliers it actually applies.
    for key, reg in S.REGIMES.items():
        note = reg["note"].lower()
        stated = {side: (1 - int(p) / 100 if verb == "cut" else 1 + int(p) / 100)
                  for side, verb, p in re.findall(r"(long|fade)[\w -]*?scores? (?:are )?(cut|raised) (\d+)%", note)}
        if "no vix adjustment" in note:
            stated = {"long": 1.0, "fade": 1.0}
        ok = all(math.isclose(reg[side], v) for side, v in stated.items())
        rows.append((f"'{reg['label']}' note matches its multipliers",
                     f"long x{reg['long']:.2f}, fade x{reg['fade']:.2f}", ok))
    return rows


# --------------------------------------------------------------------------- #
# Findings vs baseline
# --------------------------------------------------------------------------- #
def _clean(v):
    if isinstance(v, (np.bool_, bool)):
        return bool(v)
    if isinstance(v, (np.integer, int)):
        return int(v)
    if isinstance(v, (np.floating, float)):
        return None if not np.isfinite(v) else float(v)
    return v


def findings(vix, screen, dip, conviction, guidance) -> dict:
    return {
        "market": {"vix_regime": vix["key"], "conviction": _clean(conviction["score"])},
        "guidance": {"detect_" + str(k): v for k, v in guidance["detect"].items()}
                    | {"null_any": guidance["null_any"], "rw_false_respect": guidance["rw_false_respect"]},
        "screen": {r["ticker"]: {f: _clean(r[f]) for f in SCREEN_FIELDS} for _, r in screen.iterrows()},
        "dip": {r["ticker"]: {f: _clean(r[f]) for f in DIP_FIELDS} for _, r in dip.iterrows()},
    }


def diff(old: dict, new: dict, path: str = "") -> list[tuple[str, object, object]]:
    out = []
    for k in sorted(set(old) | set(new), key=str):
        a, b, p = old.get(k), new.get(k), f"{path}{k}"
        if isinstance(a, dict) and isinstance(b, dict):
            out += diff(a, b, p + ".")
        elif isinstance(a, float) and isinstance(b, float):
            if not math.isclose(a, b, rel_tol=1e-6, abs_tol=1e-9):
                out.append((p, a, b))
        elif a != b:
            out.append((p, a, b))
    return out


def _fmt(v) -> str:
    return "-" if v is None else f"{v:.4g}" if isinstance(v, float) else str(v)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--update", action="store_true", help="approve current findings as the new baseline")
    ap.add_argument("--fetch", action="store_true", help="re-download the price snapshot")
    args = ap.parse_args()
    if args.fetch:
        fetch()

    vix, screen, dip, conviction = run_pipeline(load())
    guidance = measure_guidance()
    calc = check_calculations(vix, screen, dip, conviction)
    claims = check_guidance(guidance)
    now = findings(vix, screen, dip, conviction, guidance)
    old = json.loads(BASELINE.read_text()) if BASELINE.exists() else None
    changes = diff(old, now) if old else []
    # Headline changes (a grade, rule, level or status flipping) first; the stats under them follow.
    changes.sort(key=lambda c: (c[0].rsplit(".", 1)[-1] not in HEADLINE, c[0]))

    lines = ["## Backtest validation", "",
             f"Snapshot: {len(screen)} stocks screened, {len(dip)} dip-researched, "
             f"VIX regime {vix['label']}, conviction {conviction['score']:.0f}/100.", "",
             "### Calculations", ""]
    lines += [f"- ❌ {b}" for b in calc] or ["- ✅ all values within valid ranges"]
    lines += ["", "### Guidance claims", "", "| Claim | Measured | Holds |", "|---|---|---|"]
    lines += [f"| {c} | {m} | {'✅' if ok else '❌'} |" for c, m, ok in claims]
    lines += ["", "### Findings vs approved baseline", ""]
    if old is None:
        lines.append("No baseline yet: run `python validate.py --update` and commit it.")
    elif not changes:
        lines.append("✅ unchanged")
    else:
        lines += [f"{len(changes)} value(s) moved. If the change is intended, run `python validate.py --update` "
                  "and commit validation/baseline.json with it.", "", "| Finding | Baseline | Now |", "|---|---|---|"]
        lines += [f"| {p} | {_fmt(a)} | {_fmt(b)} |" for p, a, b in changes[:200]]
    graded = dip[dip["grade"].isin(["Consistent", "Mostly"])]
    lines += ["", "### Current findings on the snapshot", "",
              "Dip-bouncers graded Consistent/Mostly: "
              + (", ".join(f"{r.ticker} ({r.grade}, {r.best_rule}, {r.win:.0%} win)" for r in graded.itertuples()) or "none"),
              "", "Screen setups: "
              + (", ".join(f"{r.ticker} long {r.long_score:.0f} at {r.support_level}"
                           for r in screen[screen["support_level"] != ""].itertuples()) or "none")]
    report = "\n".join(lines)
    print(report)
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as f:
            f.write(report + "\n")

    failed = bool(calc) or not all(ok for _, _, ok in claims)
    if args.update and not failed:
        BASELINE.write_text(json.dumps(now, indent=1, sort_keys=True) + "\n")
        print(f"\nApproved: wrote {BASELINE}")
    elif failed or old is None or changes:
        sys.exit(1)


if __name__ == "__main__":
    main()
