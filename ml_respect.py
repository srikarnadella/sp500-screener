#!/usr/bin/env python3
"""
Experimental: does a learned P(hold) model beat screener.py's placebo-baseline
heuristic at predicting whether a level test holds or breaks?

screener.py's "respect" measurement (level_respect + shrunk_rate) is a hand-built
rule: count held vs. broke, shrink toward a placebo baseline, require it to beat
that baseline by EXCESS_MIN. This trains a gradient-boosted classifier on features
available at each test event (RSI, %B, relvol, trend context, day of week, which
level/role/family) pooled across every stock, and benchmarks it against the
heuristic OUT OF TIME: train on the earlier events, evaluate on the most recent
slice neither model has seen.

This does NOT change screener.py's live scoring. It's a standalone benchmark --
wire it in only if the printed numbers say the model actually wins. The heuristic
side of the comparison is reconstructed per stock, per level/role, using only
training-period data and the same two-pass shrinkage analyze_stock does -- including
rebuilding the PLACEBO levels (shifted moving averages, off-multiple Bollinger bands)
that Pass 1 actually shrinks toward, not the real levels' own pooled rate, which
would silently reintroduce the "any price line holds by construction" bias the
placebo mechanism exists to cancel. So it's not a straw man: it's what the current
pipeline would have believed at the time.

Run:  python ml_respect.py --demo     synthetic data (screener.demo_data), no network, fast
      python ml_respect.py            live S&P 500 history, writes data/ml_respect_report.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import screener as S

FEATURES = ["rsi", "pctb", "relvol", "atr_pct", "macd_pct", "bandwidth", "chg1d",
           "above_sma200", "sma50_above_sma200", "weekday", "is_support", "is_ma", "level_rank"]
# `dist_atr` (price's distance from the level at the moment of the test) is deliberately excluded:
# it mechanically predicts which threshold (held/broke) gets hit first from the test's own
# geometry -- the same "any price line holds by construction" problem the placebo baseline exists
# to correct for, just visible to the model at finer resolution. Confirmed by benchmarking against
# pure-noise synthetic data (see test_ml_respect.py): including it made the model "beat" the
# heuristic on data with zero real structure, by the same margin as on planted bounce structure.
LEVEL_RANK = {"SMA20": 0, "SMA50": 1, "SMA100": 2, "SMA200": 3, "BB lower": -1, "BB upper": -1}
OOS_FRAC = 0.30    # most recent slice of events held out, by date, for both the model and heuristic
MIN_TRAIN = 500    # skip the benchmark if there's not enough training data for the split to mean anything


# --------------------------------------------------------------------------- #
# Build the training table: one row per resolved test event, pooled across stocks
# --------------------------------------------------------------------------- #
def _events_for_stock(ticker: str, df: pd.DataFrame) -> list[dict]:
    d = S.add_indicators(df)
    if len(d) < S.MIN_BARS:
        return []
    close = d["Close"].to_numpy(float)
    high = d["High"].to_numpy(float)
    low = d["Low"].to_numpy(float)
    atr = d["atr"].to_numpy(float)
    sma200 = d["sma200"].to_numpy(float)
    sma50 = d["sma50"].to_numpy(float)
    # sma200/sma50 need up to 200/50 bars, well after MIN_BARS gates the whole series -- a test
    # can otherwise fire before either exists, which would corrupt above_sma200/sma50_above_sma200
    # to a false 0.0 (px > NaN is False in numpy, not NaN). Folded into the condition mask itself,
    # not filtered per-event after the fact, so a suppressed early test doesn't also eat the
    # cooldown window a later, valid test would otherwise get.
    has_trend = np.isfinite(sma200) & np.isfinite(sma50)

    out = []
    for name, (col, fam, roles) in S.LEVELS.items():
        lvl = d[col].to_numpy(float)
        for role in roles:
            cond = S._test_condition(high, low, close, lvl, atr, role) & has_trend
            for t in S._test_indices(cond, S.COOLDOWN):
                label = S._resolve_outcome(close, lvl, atr, t, role)
                if label is None:      # unresolved within HORIZON bars -- ignored, same as level_respect
                    continue
                row, a, px = d.iloc[t], atr[t], close[t]
                out.append(dict(
                    ticker=ticker, date=d.index[t], level=name, role=role, label=label,
                    rsi=S._f(row["rsi"], 50.0), pctb=S._f(row["pctb"], 0.5),
                    relvol=S._f(row["relvol"], 1.0), atr_pct=a / px if px else np.nan,
                    macd_pct=S._f(row["macd_hist"], 0.0) / px if px else np.nan,
                    bandwidth=S._f(row["bandwidth"]), dist_atr=(px - lvl[t]) / a if a else np.nan,
                    chg1d=S._f(row["chg1d"], 0.0), above_sma200=float(px > sma200[t]),
                    sma50_above_sma200=float(sma50[t] > sma200[t]),
                    weekday=float(d.index[t].weekday()),
                    is_support=float(role == "support"), is_ma=float(fam == "ma"),
                    level_rank=float(LEVEL_RANK[name]),
                ))
    return out


def build_events(prices: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Pooled, resolved test events across every stock and level -- the ML training table."""
    rows = []
    for t, df in prices.items():
        try:
            rows += _events_for_stock(t, df)
        except Exception as exc:
            print(f"[warn] {t}: {exc}", file=sys.stderr)
    if not rows:
        return pd.DataFrame(columns=["ticker", "date", "level", "role", "label", *FEATURES])
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


# --------------------------------------------------------------------------- #
# The heuristic's own out-of-time prediction, for a fair comparison
# --------------------------------------------------------------------------- #
def _placebo_family_rates(prices: dict[str, pd.DataFrame], train: pd.DataFrame) -> dict[tuple, float]:
    """Per (ticker, family, role): the placebo-baseline hold rate analyze_stock's Pass 1 actually
    computes -- from PLACEBO levels (moving averages shifted by PLACEBO_ATR, Bollinger bands at
    other PLACEBO_SIGMA multiples), NEVER from the real levels' own outcomes (pooling real levels
    together would just reintroduce the "any price line holds by construction" bias the placebo
    mechanism exists to cancel -- an earlier version of this function did exactly that). Restricted
    to each ticker's own bars at or before the last date it contributed to `train`, so this can't
    see anything the model/heuristic being benchmarked wouldn't have."""
    cutoff_by_ticker = train.groupby("ticker")["date"].max()
    out = {}
    for ticker, cutoff in cutoff_by_ticker.items():
        df = prices.get(ticker)
        if df is None:
            continue
        d = S.add_indicators(df)
        d = d[d.index <= cutoff]
        if len(d) < S.MIN_BARS:
            continue
        close, high, low = d["Close"].to_numpy(float), d["High"].to_numpy(float), d["Low"].to_numpy(float)
        atr, sma20, bb_sd = d["atr"].to_numpy(float), d["sma20"].to_numpy(float), d["bb_sd"].to_numpy(float)
        for name, (col, fam, roles) in S.LEVELS.items():
            lvl = d[col].to_numpy(float)
            for role in roles:
                if fam == "ma":
                    placebos = [lvl + m * atr for m in S.PLACEBO_ATR]
                else:
                    sign = -1 if role == "support" else 1
                    placebos = [sma20 + sign * m * bb_sd for m in S.PLACEBO_SIGMA]
                h = b = 0
                for pl in placebos:
                    ph, pb = S.level_respect(high, low, close, pl, atr, role)
                    h, b = h + ph, b + pb
                out[(ticker, fam, role)] = S.shrunk_rate(h, b, 0.5)
    return out


def heuristic_oos_probs(prices: dict[str, pd.DataFrame], train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    """What screener.py's placebo-baseline heuristic would have predicted for each held-out
    event: this stock's own shrunk hold-rate for this level/role (from training-period events
    only), shrunk toward that stock's own PLACEBO-based family baseline -- the same two-pass math
    analyze_stock does, just restricted to data available before the held-out period."""
    fam_of = {name: S.LEVELS[name][1] for name in S.LEVELS}
    fam_rate = _placebo_family_rates(prices, train)
    market_base = float(train["label"].mean()) if len(train) else 0.5

    lvl_g = train.assign(fam=train["level"].map(fam_of)).groupby(["ticker", "level", "role", "fam"])["label"] \
        .agg(held="sum", n="count").reset_index()
    lvl_g["broke"] = lvl_g["n"] - lvl_g["held"]
    lvl_g["fam_rate"] = lvl_g.apply(lambda r: fam_rate.get((r["ticker"], r["fam"], r["role"]), market_base), axis=1)
    lvl_g["lvl_rate"] = lvl_g.apply(lambda r: S.shrunk_rate(r["held"], r["broke"], r["fam_rate"]), axis=1)

    t = test.assign(fam=test["level"].map(fam_of))
    t = t.merge(lvl_g[["ticker", "level", "role", "lvl_rate"]], on=["ticker", "level", "role"], how="left")
    t["fam_fallback"] = t.apply(lambda r: fam_rate.get((r["ticker"], r["fam"], r["role"]), market_base), axis=1)
    return t["lvl_rate"].fillna(t["fam_fallback"]).to_numpy(float)


def _brier(y: np.ndarray, p: np.ndarray) -> float:
    """Mean squared error between predicted probability and outcome. Lower is better; rewards
    calibration, not just which side of 0.5 the prediction falls on."""
    return float(np.mean((y - p) ** 2))


def train_model(train: pd.DataFrame):
    from sklearn.ensemble import HistGradientBoostingClassifier
    clf = HistGradientBoostingClassifier(max_depth=4, max_iter=300, learning_rate=0.05, random_state=0)
    clf.fit(train[FEATURES], train["label"])
    return clf


def score(prices: dict[str, pd.DataFrame], train: pd.DataFrame, test: pd.DataFrame) -> tuple[dict, object | None]:
    """Train on `train`, score on `test` (assumed disjoint and unseen by the model). Reports
    Brier score / accuracy for three things: the ML model, the reconstructed heuristic, and a
    naive constant baseline (always predict the training set's overall hold rate).

    The constant baseline matters more than it looks: a shuffled-label sanity check (see
    test_ml_respect.py) showed the heuristic reconstruction can score WORSE than this trivial
    constant, because shrinking toward hundreds of per-ticker/level/role rates has more variance
    than a single global rate when there's little true structure to find. Without this baseline,
    "the model beats the heuristic" can just mean "the heuristic is overfit," not that the model
    found anything real -- see verdict()."""
    if train["label"].nunique() < 2 or len(test) < 50:
        return dict(ok=False, note="not enough class variety or held-out events for a fair benchmark"), None

    clf = train_model(train)
    model_p = clf.predict_proba(test[FEATURES])[:, 1]
    heur_p = heuristic_oos_probs(prices, train, test)
    const_p = np.full(len(test), float(train["label"].mean()))
    y = test["label"].to_numpy(float)

    def _stats(p):
        return _brier(y, p), float(((p >= 0.5) == y).mean())

    model_brier, model_acc = _stats(model_p)
    heur_brier, heur_acc = _stats(heur_p)
    const_brier, const_acc = _stats(const_p)
    report = dict(
        ok=True, n_train=len(train), n_test=len(test), base_rate=float(y.mean()),
        model_brier=model_brier, model_acc=model_acc,
        heuristic_brier=heur_brier, heuristic_acc=heur_acc,
        constant_brier=const_brier, constant_acc=const_acc,
    )
    return report, clf


def evaluate(prices: dict[str, pd.DataFrame], events: pd.DataFrame) -> tuple[dict, object | None]:
    """Out-of-time benchmark: train on the earlier (1 - OOS_FRAC) of events by date, score on the
    later slice neither the model nor the heuristic reconstruction has seen."""
    events = events.dropna(subset=FEATURES + ["label"]).sort_values("date")
    if len(events) < MIN_TRAIN:
        return dict(ok=False, note=f"only {len(events)} resolved events, need >= {MIN_TRAIN}"), None
    split = int(len(events) * (1 - OOS_FRAC))
    return score(prices, events.iloc[:split], events.iloc[split:])


def evaluate_by_ticker(prices: dict[str, pd.DataFrame], events: pd.DataFrame, seed: int = 0) -> tuple[dict, object | None]:
    """Stricter generalization check: hold out entire TICKERS instead of just later dates of
    tickers already seen in training. Answers 'does this transfer to a stock the model has never
    seen a single event from' -- catching entity-specific fingerprinting the time-based split
    can't, since under a time split both sides still get to see a stock's own earlier history."""
    events = events.dropna(subset=FEATURES + ["label"])
    tickers = sorted(events["ticker"].unique())
    if len(tickers) < 10:
        return dict(ok=False, note=f"only {len(tickers)} tickers, need >= 10 for a ticker holdout"), None
    rng = np.random.default_rng(seed)
    held_out = set(rng.choice(tickers, size=max(1, int(len(tickers) * OOS_FRAC)), replace=False))
    train = events[~events["ticker"].isin(held_out)]
    test = events[events["ticker"].isin(held_out)]
    if len(train) < MIN_TRAIN:
        return dict(ok=False, note=f"only {len(train)} training events after the ticker split"), None
    return score(prices, train, test)


def verdict(report: dict) -> str:
    """The model has to clear two bars: meaningfully beat doing nothing (the constant baseline),
    and meaningfully beat the current heuristic. Beating a heuristic that itself doesn't clear the
    constant baseline doesn't count -- that's a weak heuristic, not a strong model."""
    if not report.get("ok"):
        return "n/a"
    cb, mb, hb = report["constant_brier"], report["model_brier"], report["heuristic_brier"]
    heur_vs_const = (cb - hb) / cb if cb else 0.0
    model_vs_const = (cb - mb) / cb if cb else 0.0
    model_vs_heur = (hb - mb) / hb if hb else 0.0
    if heur_vs_const < 0.005:
        return "heuristic reconstruction doesn't clear the constant baseline here -- treat with caution"
    if model_vs_const < 0.01:
        return "no real skill detected (doesn't clear the constant baseline)"
    if model_vs_heur > 0.02:
        return "beats"
    if model_vs_heur < -0.02:
        return "loses to"
    return "roughly ties"


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="data")
    ap.add_argument("--cache", default="data/sp500_constituents.csv")
    ap.add_argument("--period", default=S.PERIOD)
    ap.add_argument("--demo", action="store_true", help="synthetic data, no network")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if args.demo:
        _, prices, _, _ = S.demo_data()
    else:
        print("Loading S&P 500 constituents...")
        universe = S.load_universe(Path(args.cache))
        print(f"Downloading {len(universe)} tickers ({args.period})...")
        prices = S.download_prices(universe["ticker"].tolist(), args.period)
        if len(prices) < 0.8 * len(universe):
            raise SystemExit(f"Only {len(prices)}/{len(universe)} tickers downloaded; try again later.")

    print(f"Building the test-event table from {len(prices)} tickers...")
    events = build_events(prices)
    print(f"{len(events)} resolved test events.")

    def _show(name, report):
        if not report["ok"]:
            print(f"[skip {name}] {report['note']}")
            return None
        v = verdict(report)
        print(f"\n{name} ({report['n_train']} train / {report['n_test']} test events, "
              f"base hold rate {report['base_rate']:.0%}):")
        print(f"  constant   Brier {report['constant_brier']:.4f}  accuracy {report['constant_acc']:.1%}"
              f"  (always predict the training rate)")
        print(f"  heuristic  Brier {report['heuristic_brier']:.4f}  accuracy {report['heuristic_acc']:.1%}")
        print(f"  ML model   Brier {report['model_brier']:.4f}  accuracy {report['model_acc']:.1%}")
        print(f"  -> {v}")
        report["verdict"] = v
        return report

    time_report, clf = evaluate(prices, events)
    time_report = _show("Out-of-time benchmark (same stocks, later dates)", time_report)
    ticker_report, _ = evaluate_by_ticker(prices, events)
    ticker_report = _show("Out-of-stock benchmark (never-seen tickers, stricter)", ticker_report)
    print("\nNot wired into screener.py's live scoring -- see the module docstring.")

    combined = dict(time_split=time_report, ticker_split=ticker_report)
    (out / "ml_respect_report.json").write_text(json.dumps(combined, indent=2, default=float))
    print(f"\nReport: {out / 'ml_respect_report.json'}")
    if clf is not None and not args.demo:
        import joblib
        joblib.dump(clf, out / "ml_respect_model.joblib")
        print(f"Model: {out / 'ml_respect_model.joblib'}")


if __name__ == "__main__":
    main()
