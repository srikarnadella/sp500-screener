#!/usr/bin/env python3
"""
S&P 500 "level respect" screener.

For every S&P 500 stock it asks two questions:

  1. Which technical levels does THIS stock actually respect?
     (20/50/100/200-day moving averages, lower/upper Bollinger Band)
     Measured by backtesting 5 years of history: each time price tested a level,
     did it bounce/reject by >= 1 ATR before slicing through by >= 1 ATR,
     and how does that compare with placebo levels (the stock's own baseline)?

  2. Is the stock at one of those levels right now, and does the trend,
     momentum and VIX backdrop support acting on it?

Outputs (in --out, default ./docs so GitHub Pages can serve it):
  index.html                  the daily report
  data/screen_latest.csv      every stock, every metric
  data/signals_history.csv    top picks logged daily, for forward-testing

Run:  python screener.py            (live data, run after the US close)
      python screener.py --demo     (synthetic data, no network needed)
"""
from __future__ import annotations

import argparse
import html
import io
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Configuration. Everything you might want to tune lives here.
# --------------------------------------------------------------------------- #
PERIOD = "5y"          # history used for the respect backtest
MIN_BARS = 300         # skip stocks with less history than this
TOL_ATR = 0.5          # a "test" = bar reaches within 0.5 ATR of the level
MOVE_ATR = 1.0         # bounce / break = close 1 ATR away from the level
HORIZON = 10           # bars allowed for the bounce / break to resolve
COOLDOWN = 5           # bars to ignore after a test (avoids double counting)
MIN_EVENTS = 6         # need at least this many resolved tests to trust a level
PRIOR_WEIGHT = 4       # shrinks small samples toward the stock's baseline hold-rate
EXCESS_MIN = 0.10      # hold-rate must beat the placebo baseline by 10 pts to be "respected"
EXCESS_FULL = 0.20     # ...and beating it by 20 pts earns full credit
PLACEBO_ATR = (-6, -4, 4, 6)          # moving-average placebos: level shifted by this many ATRs
PLACEBO_SIGMA = (1.0, 1.5, 2.5, 3.0)  # Bollinger placebos: same band at other std-dev multiples
IN_PLAY_ATR = 1.5      # how close (in ATRs) price must be for a level to count

# level name -> (column, family, roles it can play).  Each family gets its own placebo baseline.
LEVELS = {
    "SMA20":    ("sma20",  "ma", ("support", "resistance")),
    "SMA50":    ("sma50",  "ma", ("support", "resistance")),
    "SMA100":   ("sma100", "ma", ("support", "resistance")),
    "SMA200":   ("sma200", "ma", ("support", "resistance")),
    "BB lower": ("bb_lo",  "bb", ("support",)),
    "BB upper": ("bb_up",  "bb", ("resistance",)),
}

# VIX regime -> multipliers applied to the two scores. These are judgment-based
# heuristics, not fitted values. Tune them, and check them against the
# signals_history.csv the script builds up.
REGIMES = {
    "rolling_over": dict(
        label="Fear rolling over", long=1.20, fade=0.80,
        note="VIX spiked and is falling fast. Bounces off support have tended "
             "to work better as fear unwinds."),
    "stress": dict(
        label="Stress", long=0.60, fade=1.00,
        note="VIX at 30 or above, or the VIX curve is inverted. Support levels "
             "break more often, so long-side scores are cut 40%."),
    "rising": dict(
        label="Rising fear", long=0.80, fade=1.00,
        note="VIX is climbing more than 15% above its 20-day average. "
             "Long-side scores are cut 20% until it settles."),
    "complacent": dict(
        label="Complacent", long=0.90, fade=1.15,
        note="VIX is very low, leaving little cushion if sentiment turns. "
             "Fade scores are raised 15%, long scores cut 10%."),
    "normal": dict(
        label="Normal", long=1.00, fade=1.00,
        note="No VIX adjustment applied."),
}

WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"


# --------------------------------------------------------------------------- #
# Universe and data
# --------------------------------------------------------------------------- #
def load_universe(cache: Path) -> pd.DataFrame:
    """Current S&P 500 constituents from Wikipedia, cached to disk."""
    import requests

    try:
        r = requests.get(WIKI_URL, timeout=30,
                         headers={"User-Agent": "Mozilla/5.0 (sp500-screener)"})
        r.raise_for_status()
        tbl = pd.read_html(io.StringIO(r.text), match="Symbol")[0]
        df = tbl[["Symbol", "Security", "GICS Sector"]].rename(
            columns={"Symbol": "ticker", "Security": "name", "GICS Sector": "sector"})
        df["ticker"] = df["ticker"].str.replace(".", "-", regex=False)  # BRK.B -> BRK-B
        cache.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(cache, index=False)
        return df
    except Exception as exc:  # network, layout change, etc.
        if cache.exists():
            print(f"[warn] could not refresh S&P 500 list ({exc}); using cache",
                  file=sys.stderr)
            return pd.read_csv(cache)
        raise SystemExit(f"Could not fetch the S&P 500 list and no cache exists: {exc}")


def _clean(df: pd.DataFrame) -> pd.DataFrame | None:
    need = ["High", "Low", "Close", "Volume"]
    if df is None or any(c not in df.columns for c in need):
        return None
    df = df.dropna(subset=["Close"]).copy()
    idx = pd.DatetimeIndex(df.index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    df.index = idx.normalize()
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return df if len(df) else None


def _split_download(data: pd.DataFrame, chunk: list[str]) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    if data is None or data.empty:
        return out
    if isinstance(data.columns, pd.MultiIndex):
        top = set(data.columns.get_level_values(0))
        for t in chunk:
            if t in top:
                d = _clean(data[t])
                if d is not None:
                    out[t] = d
    elif len(chunk) == 1:
        d = _clean(data)
        if d is not None:
            out[chunk[0]] = d
    return out


def download_prices(tickers: list[str], period: str = PERIOD,
                    batch: int = 100) -> dict[str, pd.DataFrame]:
    """Daily, split/dividend-adjusted OHLCV from Yahoo Finance via yfinance."""
    import yfinance as yf

    def fetch(chunk: list[str]) -> dict[str, pd.DataFrame]:
        # An empty/short frame (rate limit) isn't an exception, so it needs its own retry check.
        for attempt in range(3):
            try:
                data = yf.download(chunk, period=period, interval="1d",
                                   auto_adjust=True, group_by="ticker",
                                   threads=True, progress=False)
                out = _split_download(data, chunk)
                if len(out) >= 0.5 * len(chunk) or attempt == 2:
                    return out
                print(f"[warn] only {len(out)}/{len(chunk)} came back, retrying",
                      file=sys.stderr)
            except Exception as exc:
                print(f"[warn] download attempt {attempt + 1} failed: {exc}",
                      file=sys.stderr)
            time.sleep(3 * (attempt + 1))
        return {}

    result: dict[str, pd.DataFrame] = {}
    for i in range(0, len(tickers), batch):
        chunk = tickers[i:i + batch]
        result.update(fetch(chunk))
        print(f"  downloaded {min(i + batch, len(tickers))}/{len(tickers)}")
        time.sleep(1)
    missing = [t for t in tickers if t not in result]
    if missing:  # one retry pass for stragglers, chunked the same as the main pass
        for i in range(0, len(missing), batch):
            result.update(fetch(missing[i:i + batch]))
    return result


# --------------------------------------------------------------------------- #
# Indicators
# --------------------------------------------------------------------------- #
def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    c, h, l, v = d["Close"], d["High"], d["Low"], d["Volume"]

    for n in (20, 50, 100, 200):
        d[f"sma{n}"] = c.rolling(n).mean()

    sd = c.rolling(20).std(ddof=0)
    d["bb_sd"] = sd
    d["bb_up"] = d["sma20"] + 2 * sd
    d["bb_lo"] = d["sma20"] - 2 * sd
    width = d["bb_up"] - d["bb_lo"]
    d["pctb"] = (c - d["bb_lo"]) / width.replace(0, np.nan)
    d["bandwidth"] = width / d["sma20"]

    prev = c.shift(1)
    tr = pd.concat([h - l, (h - prev).abs(), (l - prev).abs()], axis=1).max(axis=1)
    d["atr"] = tr.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()

    delta = c.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    d["rsi"] = 100 - 100 / (1 + gain / loss)

    macd = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
    d["macd_hist"] = macd - macd.ewm(span=9, adjust=False).mean()

    d["relvol"] = v / v.rolling(20).mean()
    d["chg1d"] = c.pct_change()
    d["hi52"] = h.rolling(252).max()
    d["lo52"] = l.rolling(252).min()
    return d


# --------------------------------------------------------------------------- #
# The core idea: how much does this stock respect this level?
# --------------------------------------------------------------------------- #
def level_respect(high, low, close, level, atr, role,
                  tol=TOL_ATR, move=MOVE_ATR, horizon=HORIZON, cooldown=COOLDOWN):
    """
    Count how often price held vs. broke a level over the whole history.

    A "test" happens when a bar reaches within `tol` ATRs of the level, coming
    from the correct side (from above for support, from below for resistance).
    Starting from that same bar's close, over the next `horizon` bars the test resolves as:
        held   - close moves `move` ATRs away from the level on the near side
                 (bounce off support / rejection at resistance) first
        broke  - close moves `move` ATRs through the level first
    Tests that resolve neither way are ignored.  Returns (held, broke).
    """
    close = np.asarray(close, float)
    high = np.asarray(high, float)
    low = np.asarray(low, float)
    level = np.asarray(level, float)
    atr = np.asarray(atr, float)

    cond = _test_condition(high, low, close, level, atr, role, tol)
    held = broke = 0
    for t in _test_indices(cond, cooldown):
        outcome = _resolve_outcome(close, level, atr, t, role, move, horizon)
        if outcome is None:
            continue
        held += outcome
        broke += 1 - outcome
    return held, broke


def _test_condition(high: np.ndarray, low: np.ndarray, close: np.ndarray, level: np.ndarray,
                    atr: np.ndarray, role: str, tol: float = TOL_ATR) -> np.ndarray:
    """Boolean mask: bar reaches within `tol` ATRs of `level`, coming from the correct side
    (from above for support, from below for resistance)."""
    prev_close = np.roll(close, 1)
    prev_close[0] = np.nan
    ok = np.isfinite(level) & np.isfinite(atr) & np.isfinite(prev_close)
    with np.errstate(invalid="ignore"):
        if role == "support":
            return ok & (prev_close > level) & (low <= level + tol * atr)
        return ok & (prev_close < level) & (high >= level - tol * atr)


def _test_indices(cond: np.ndarray, cooldown: int) -> list[int]:
    """Bar indices where `cond` (a test-condition mask) fires, at most one per `cooldown` bars."""
    out, last = [], -10 ** 9
    for t in np.flatnonzero(cond):
        if t - last < cooldown:
            continue
        last = t
        out.append(t)
    return out


def _resolve_outcome(close: np.ndarray, level: np.ndarray, atr: np.ndarray, t: int, role: str,
                     move: float = MOVE_ATR, horizon: int = HORIZON) -> int | None:
    """1 if the test at bar `t` resolved as held, 0 if broke, None if neither happened in time."""
    seg = close[t:t + horizon + 1]
    lv, a = level[t], atr[t]
    away = np.flatnonzero(seg >= lv + move * a)
    through = np.flatnonzero(seg <= lv - move * a)
    good, bad = (away, through) if role == "support" else (through, away)
    g = good[0] if good.size else np.inf
    b = bad[0] if bad.size else np.inf
    if np.isinf(g) and np.isinf(b):
        return None
    return 1 if g < b else 0


def shrunk_rate(held: int, broke: int, base: float = 0.5, k: int = PRIOR_WEIGHT) -> float:
    """Hold-rate pulled toward the baseline when there are few observations."""
    return (held + k * base) / (held + broke + k)


def _edge(rate: float, base: float) -> float:
    """0 when the level holds no better than a placebo, 1 at EXCESS_FULL better."""
    return float(np.clip((rate - base) / EXCESS_FULL, 0, 1))


def _f(x, default=np.nan):
    try:
        x = float(x)
        return x if np.isfinite(x) else default
    except (TypeError, ValueError):
        return default


def rel_strength(close: pd.Series, spy: pd.Series | None, n: int) -> float:
    """n-session return minus SPY's return over the same dates (NaN without SPY)."""
    if spy is None or len(close) <= n:
        return np.nan
    s = spy.reindex(close.index, method="ffill")
    a, b = s.iloc[-1], s.iloc[-1 - n]
    if not (np.isfinite(a) and np.isfinite(b)) or b <= 0:
        return np.nan
    return float(close.iloc[-1] / close.iloc[-1 - n] - a / b)


# --------------------------------------------------------------------------- #
# VIX regime
# --------------------------------------------------------------------------- #
def vix_context(vix: pd.DataFrame, vix3m: pd.DataFrame | None) -> dict:
    v = vix["Close"].dropna()
    level = float(v.iloc[-1])
    yr = v.iloc[-252:]
    pct = float((yr <= level).mean())
    ma20 = float(v.iloc[-20:].mean())
    chg5 = float(level / v.iloc[-6] - 1) if len(v) > 6 else 0.0
    hi10 = float(v.iloc[-10:].max())
    term = np.nan
    if vix3m is not None:
        v3 = vix3m["Close"].dropna()
        if len(v3):
            term = level / float(v3.iloc[-1])

    stressed = level >= 30 or (np.isfinite(term) and term >= 1.05)
    if stressed:
        key = "stress"
    elif hi10 >= 25 and level <= hi10 * 0.85 and chg5 < 0:
        key = "rolling_over"
    elif level > ma20 * 1.15 and chg5 > 0:
        key = "rising"
    elif level < 15 or pct < 0.15:
        key = "complacent"
    else:
        key = "normal"

    return dict(key=key, level=level, pct=pct, ma20=ma20, chg5=chg5, term=term,
                lo=float(yr.min()), hi=float(yr.max()), **REGIMES[key])


FRED_HY_OAS_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=BAMLH0A0HYM2"


def credit_spread_context(cache: Path) -> dict:
    """High-yield vs Treasury option-adjusted spread (FRED series BAMLH0A0HYM2, no API key).
    An early stress signal: spreads widen when credit markets get nervous, often before equities
    react. Cached to disk like the S&P constituent list, so a FRED outage doesn't block the report."""
    import requests

    try:
        r = requests.get(FRED_HY_OAS_URL, timeout=20)
        r.raise_for_status()
        df = pd.read_csv(io.StringIO(r.text)).rename(
            columns={"observation_date": "date", "BAMLH0A0HYM2": "spread"})
        df["spread"] = pd.to_numeric(df["spread"], errors="coerce")
        df = df.dropna(subset=["spread"])
        cache.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(cache, index=False)
    except Exception as exc:
        if not cache.exists():
            return dict(ok=False, level=np.nan, chg5=np.nan, chg20=np.nan, note=f"FRED unavailable ({exc})")
        print(f"[warn] could not refresh credit spread ({exc}); using cache", file=sys.stderr)
        df = pd.read_csv(cache)
    s = df["spread"].to_numpy(float)
    if len(s) < 25:
        return dict(ok=False, level=np.nan, chg5=np.nan, chg20=np.nan, note="not enough FRED history")
    return dict(ok=True, level=float(s[-1]), chg5=float(s[-1] - s[-6]), chg20=float(s[-1] - s[-21]),
                note="High-yield vs Treasury spread (FRED BAMLH0A0HYM2). Widening = credit stress.")


def breadth_series(prices: dict[str, pd.DataFrame]) -> dict:
    """Advance/decline breadth across every downloaded ticker: the McClellan Oscillator
    (EMA19 - EMA39 of net advances) and a Zweig-style breadth thrust (the 10-day EMA of the
    advance ratio flipping from under 40% to over 61.5% within 10 sessions -- a rare, historically
    bullish "everyone buys at once" signal)."""
    closes = pd.concat({t: d["Close"] for t, d in prices.items() if len(d) > 45}, axis=1)
    closes = closes.sort_index().ffill(limit=3)
    chg = closes.diff()
    adv, decl = (chg > 0).sum(axis=1), (chg < 0).sum(axis=1)
    net = (adv - decl).astype(float)
    mcclellan = net.ewm(span=19, adjust=False).mean() - net.ewm(span=39, adjust=False).mean()
    ratio10 = (adv / (adv + decl).replace(0, np.nan)).ewm(span=10, adjust=False).mean()
    thrust = bool(len(ratio10) > 11 and ratio10.iloc[-11:].min() < 0.40 and ratio10.iloc[-1] > 0.615)
    return dict(mcclellan=float(mcclellan.iloc[-1]) if len(mcclellan) else np.nan,
                zweig_thrust=thrust, adv=int(adv.iloc[-1]) if len(adv) else 0,
                decl=int(decl.iloc[-1]) if len(decl) else 0)


def sector_strength(res: pd.DataFrame) -> pd.DataFrame:
    """Which sectors are oversold vs SPY right now, not just in absolute terms."""
    g = res.groupby("sector").agg(n=("ticker", "size"), rs_1m=("rs_1m", "mean"), rs_3m=("rs_3m", "mean"),
                                  vs_sma50=("vs_sma50", "mean"), vs_sma200=("vs_sma200", "mean"))
    return g.sort_values("rs_1m").reset_index()


def conviction_score(breadth: dict, ctx: dict, res: pd.DataFrame) -> dict:
    """Blend breadth, the VIX regime and how many stocks currently flash a real long setup into
    one 0-100 read on whether this is a market where dip-buying is favored. A heuristic overlay
    for the report header -- it does not feed back into individual stock scores."""
    regime_pts = {"stress": 10, "rising": 35, "normal": 60, "complacent": 65, "rolling_over": 85}
    b = 50 * np.clip(breadth["above50"] / 100, 0, 1) + 50 * np.clip(breadth["above200"] / 100, 0, 1)
    density = 100 * (res["long_score"] >= 50).mean()
    parts = [("Breadth (%>50d / %>200d)", float(b)),
             ("VIX regime", float(regime_pts.get(ctx["key"], 50))),
             ("Share of stocks with a live long setup", float(density))]
    if breadth.get("zweig_thrust"):
        parts.append(("Zweig breadth thrust firing", 100.0))
    score = float(np.mean([p[1] for p in parts]))
    label = ("Bullish" if score >= 75 else "Constructive" if score >= 60 else "Neutral"
             if score >= 40 else "Cautious" if score >= 25 else "Bearish")
    return dict(score=score, label=label, parts=parts)


# --------------------------------------------------------------------------- #
# Per-stock analysis
# --------------------------------------------------------------------------- #
def analyze_stock(ticker: str, df: pd.DataFrame, spy_close: pd.Series | None = None) -> dict | None:
    d = add_indicators(df)
    if len(d) < MIN_BARS:
        return None

    close = d["Close"].to_numpy(float)
    high = d["High"].to_numpy(float)
    low = d["Low"].to_numpy(float)
    atr = d["atr"].to_numpy(float)
    last = d.iloc[-1]
    a_now = atr[-1]
    if not np.isfinite(a_now) or a_now <= 0 or not np.isfinite(last["sma200"]):
        return None

    # Pass 1: real levels, plus placebo levels to learn this stock's baseline hold-rate.
    # Without a baseline every level looks "respected": a test starts slightly on the near
    # side of the level, so the bounce target is closer than the break target.
    #   moving averages -> the level shifted several ATRs away
    #   Bollinger bands -> the same band at other std-dev multiples
    raw: dict[tuple[str, str], tuple[int, int]] = {}
    lvls: dict[str, np.ndarray] = {}
    pool: dict[tuple[str, str], list[int]] = {}
    dist: dict[str, float] = {}
    sma20 = d["sma20"].to_numpy(float)
    bb_sd = d["bb_sd"].to_numpy(float)
    for name, (col, fam, roles) in LEVELS.items():
        lvl = d[col].to_numpy(float)
        lvls[name] = lvl
        dist[name] = (close[-1] - lvl[-1]) / a_now
        for role in roles:
            raw[(name, role)] = level_respect(high, low, close, lvl, atr, role)
            if fam == "ma":
                placebos = [lvl + m * atr for m in PLACEBO_ATR]
            else:
                sign = -1 if role == "support" else 1
                placebos = [sma20 + sign * m * bb_sd for m in PLACEBO_SIGMA]
            acc = pool.setdefault((fam, role), [0, 0])
            for pl in placebos:
                ph, pb = level_respect(high, low, close, pl, atr, role)
                acc[0] += ph
                acc[1] += pb
    base = {k: (shrunk_rate(h, b, 0.5) if h + b else 0.5) for k, (h, b) in pool.items()}

    # Pass 2: rate vs. baseline
    stats: dict[tuple[str, str], dict] = {}
    broken: list[tuple[str, float, int, float, float]] = []
    for (name, role), (h, b) in raw.items():
        bs = base[(LEVELS[name][1], role)]
        rate = shrunk_rate(h, b, bs)
        stats[(name, role)] = dict(held=h, broke=b, n=h + b, rate=rate, base=bs)
        # A respected support that gave way in the last 5 sessions
        if role == "support" and h + b >= MIN_EVENTS and rate - bs >= EXCESS_MIN:
            recent = (close[-6:] - lvls[name][-6:]) / a_now
            prior = recent[:-1]
            if np.isfinite(prior).any() and np.nanmax(prior) >= 0 and recent[-1] <= -MOVE_ATR:
                broken.append((name, rate, h + b, recent[-1], bs))

    def best(role: str):
        top = (0.0, None)
        for (name, r), s in stats.items():
            if r != role or s["n"] < MIN_EVENTS:
                continue
            if s["rate"] - s["base"] < EXCESS_MIN:   # not distinguishable from a placebo
                continue
            dd = dist[name]
            if role == "support" and not (-0.75 <= dd <= IN_PLAY_ATR):
                continue
            if role == "resistance" and not (-IN_PLAY_ATR <= dd <= 0.75):
                continue
            score = 100 * (1 - abs(dd) / IN_PLAY_ATR) * _edge(s["rate"], s["base"])
            if score > top[0]:
                top = (score, name)
        return top

    sup_score, sup_name = best("support")
    res_score, res_name = best("resistance")

    px = _f(last["Close"])
    sma50, sma200 = _f(last["sma50"]), _f(last["sma200"])
    sma200_prev = _f(d["sma200"].iloc[-21], sma200)
    rsi = _f(last["rsi"], 50.0)
    pctb = _f(last["pctb"], 0.5)
    macd_h = _f(last["macd_hist"], 0.0)

    if px > sma200 and sma50 > sma200:
        trend = "Up"
    elif px < sma200 and sma50 < sma200:
        trend = "Down"
    else:
        trend = "Mixed"

    # Multi-timeframe check: does the weekly trend agree with the daily setup? Weekly close vs.
    # a 10-week (~50-day) average, requiring the average itself to be sloping the same way.
    relvol_now = _f(last["relvol"], 1.0)
    wk = d["Close"].resample("W-FRI").last().dropna()
    weekly_up = weekly_down = None
    if len(wk) >= 12:
        wk_sma = wk.rolling(10).mean()
        weekly_up = bool(wk.iloc[-1] > wk_sma.iloc[-1] > wk_sma.iloc[-2])
        weekly_down = bool(wk.iloc[-1] < wk_sma.iloc[-1] < wk_sma.iloc[-2])
    weekly_trend = "Up" if weekly_up else "Down" if weekly_down else "Mixed"
    # unknown (too little weekly history) is treated as neutral, not a penalty
    mtf_long_mult = 1.0 if weekly_up is None else (1.0 if weekly_up else 0.85)
    mtf_fade_mult = 1.0 if weekly_down is None else (1.0 if weekly_down else 0.85)

    # Volume confirmation: today's dip/stretch matters more on above-average volume.
    vol_pts = 5 * np.clip((relvol_now - 1) / 0.5, 0, 1)

    # Long setup: pullback to a respected support inside an intact uptrend
    trend_pts = 10 * (px > sma200) + 8 * (sma50 > sma200) + 7 * (sma200 > sma200_prev)
    pull_pts = (10 * np.clip((55 - rsi) / 25, 0, 1)
                + 10 * np.clip((0.35 - pctb) / 0.35, 0, 1))
    long_raw = (0.55 * sup_score + trend_pts + pull_pts + vol_pts) * mtf_long_mult

    # Fade setup: stretched into a respected resistance, with weak trend/momentum
    weak_pts = 8 * (px < sma200) + 6 * (sma50 < sma200) + 6 * (macd_h < 0)
    ext_pts = (15 * np.clip((rsi - 55) / 25, 0, 1)
               + 10 * np.clip((pctb - 0.7) / 0.3, 0, 1))
    fade_raw = (0.55 * res_score + weak_pts + ext_pts + vol_pts) * mtf_fade_mult

    # Compact "what does this stock respect" summary
    prof = []
    for name in LEVELS:
        cands = [z for (n_, _), z in stats.items() if n_ == name and z["n"] >= MIN_EVENTS]
        if cands:
            z = max(cands, key=lambda q: q["rate"] - q["base"])
            prof.append((name, z["rate"], z["base"], z["n"]))
    prof.sort(key=lambda q: -(q[1] - q[2]))
    profile = " | ".join(f"{n} {r:.0%} vs {b:.0%} (n={k})"
                         for n, r, b, k in prof[:2] if r - b >= EXCESS_MIN)

    row = dict(
        ticker=ticker, close=px, chg1d=_f(last["chg1d"], 0.0), rsi=rsi, pctb=pctb,
        bandwidth=_f(last["bandwidth"]), atr_pct=a_now / px, relvol=_f(last["relvol"]),
        pos52=(px - _f(last["lo52"])) / max(_f(last["hi52"]) - _f(last["lo52"]), 1e-9),
        vs_sma20=px / _f(last["sma20"]) - 1, vs_sma50=px / sma50 - 1,
        vs_sma200=px / sma200 - 1, macd_hist=macd_h, trend=trend,
        support_level=sup_name or "", support_dist_atr=dist.get(sup_name, np.nan) if sup_name else np.nan,
        support_rate=stats[(sup_name, "support")]["rate"] if sup_name else np.nan,
        support_n=stats[(sup_name, "support")]["n"] if sup_name else 0,
        support_base=stats[(sup_name, "support")]["base"] if sup_name else np.nan,
        support_score=sup_score,
        resist_level=res_name or "", resist_dist_atr=dist.get(res_name, np.nan) if res_name else np.nan,
        resist_base=stats[(res_name, "resistance")]["base"] if res_name else np.nan,
        resist_rate=stats[(res_name, "resistance")]["rate"] if res_name else np.nan,
        resist_n=stats[(res_name, "resistance")]["n"] if res_name else 0,
        resist_score=res_score,
        long_raw=long_raw, fade_raw=fade_raw, respect_profile=profile,
        rs_1m=rel_strength(d["Close"], spy_close, 21), rs_3m=rel_strength(d["Close"], spy_close, 63),
        weekly_trend=weekly_trend, vol_confirm=bool(relvol_now >= 1.2),
    )
    if broken:
        name, rate, n, dd, bs = max(broken, key=lambda z: z[1] - z[4])
        row.update(broken_level=name, broken_rate=rate, broken_n=n, broken_dist_atr=dd, broken_base=bs)
    else:
        row.update(broken_level="", broken_rate=np.nan, broken_n=0, broken_dist_atr=np.nan,
                   broken_base=np.nan)
    for (name, role), s in stats.items():
        key = f"{name.lower().replace(' ', '_')}_{role[:3]}"
        row[f"{key}_rate"] = s["rate"]
        row[f"{key}_base"] = s["base"]
        row[f"{key}_n"] = s["n"]
    return row


# --------------------------------------------------------------------------- #
# HTML report
# --------------------------------------------------------------------------- #
CSS = """
:root{--ink:#18212b;--muted:#5d6b7a;--paper:#f2f5f8;--panel:#fff;--line:#d8dfe6;
--up:#0d7a5f;--down:#b3352d;--accent:#2557d6;--warn:#a86a00;--bar:#c6d3f3}
@media (prefers-color-scheme:dark){:root{--ink:#e6ebf1;--muted:#93a1b1;--paper:#10161d;
--panel:#171f28;--line:#2a3541;--up:#3ccf9f;--down:#f0766d;--accent:#7ea2ff;--warn:#e0a23a;--bar:#2b3f73}}
*{box-sizing:border-box}
body{margin:0;background:var(--paper);color:var(--ink);
font:15px/1.5 "IBM Plex Sans",system-ui,-apple-system,"Segoe UI",sans-serif;
font-variant-numeric:tabular-nums}
main{max-width:1180px;margin:0 auto;padding:28px 20px 64px}
nav.pages{display:flex;gap:6px;margin:0 0 20px;flex-wrap:wrap}
nav.pages a{color:var(--muted);text-decoration:none;font-size:13px;padding:6px 12px;
border:1px solid var(--line);border-radius:4px;background:var(--panel)}
nav.pages a[aria-current]{color:var(--ink);border-color:var(--accent);font-weight:600}
h1{font-size:26px;line-height:1.2;margin:0 0 4px;font-weight:600;letter-spacing:-.01em}
h2{font-size:18px;margin:0 0 4px;font-weight:600}
.sub{color:var(--muted);margin:0 0 24px}
.demo{background:var(--warn);color:#fff;padding:8px 14px;border-radius:4px;margin-bottom:20px;font-weight:600}
.regime{background:var(--panel);border:1px solid var(--line);border-left:4px solid var(--accent);
border-radius:4px;padding:20px 22px;margin-bottom:36px}
.regime h2{font-size:22px}
.regime .note{margin:2px 0 4px;max-width:70ch}
.regime .mult{color:var(--muted);margin:0 0 18px}
.gauge{max-width:560px;margin:0 0 4px}
.track{position:relative;height:8px;border-radius:4px;background:var(--bar)}
.mk{position:absolute;top:-5px;width:4px;height:18px;border-radius:2px;background:var(--accent);
transform:translateX(-2px)}
.ends{display:flex;justify-content:space-between;color:var(--muted);font-size:13px;margin-top:6px}
dl.stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:14px 24px;margin:18px 0 0}
dl.stats div{border-top:1px solid var(--line);padding-top:8px}
dt{color:var(--muted);font-size:13px}
dd{margin:0;font-size:20px;font-weight:600}
section.block{margin-bottom:40px}
.desc{color:var(--muted);margin:0 0 12px;max-width:75ch}
.wrap{overflow-x:auto;background:var(--panel);border:1px solid var(--line);border-radius:4px}
table{border-collapse:collapse;width:100%;min-width:760px}
th,td{padding:9px 12px;text-align:right;white-space:nowrap;border-bottom:1px solid var(--line)}
th:first-child,td:first-child,th.l,td.l{text-align:left}
tbody tr:last-child td{border-bottom:0}
th{font-weight:500;color:var(--muted);font-size:13px;background:var(--panel);position:sticky;top:0}
th button{all:unset;cursor:pointer;padding:2px 0}
th button:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
th[aria-sort] button{color:var(--ink)}
th[aria-sort="ascending"] button::after{content:" \\25B2";font-size:10px}
th[aria-sort="descending"] button::after{content:" \\25BC";font-size:10px}
td small{display:block;color:var(--muted);font-size:12px;line-height:1.2}
td b{font-weight:600}
.pos{color:var(--up)}.neg{color:var(--down)}
.score{font-weight:600;font-size:16px}
.rbar{display:inline-block;width:44px;height:6px;border-radius:3px;background:var(--bar);
margin-right:8px;vertical-align:middle;overflow:hidden}
.rbar i{display:block;height:100%;background:var(--accent)}
.empty{padding:18px;color:var(--muted)}
input[type=search]{font:inherit;padding:8px 12px;border:1px solid var(--line);border-radius:4px;
background:var(--panel);color:var(--ink);width:min(320px,100%)}
input[type=search]:focus-visible{outline:2px solid var(--accent);outline-offset:1px}
.filters{display:flex;flex-wrap:wrap;gap:10px 16px;align-items:center;margin-bottom:12px}
.filters select,.filters input[type=number]{font:inherit;padding:7px 10px;border:1px solid var(--line);
border-radius:4px;background:var(--panel);color:var(--ink)}
.filters input[type=number]{width:4em}
.filters label{display:inline-flex;align-items:center;gap:5px;color:var(--muted);font-size:14px}
footer{color:var(--muted);font-size:13px;max-width:80ch;border-top:1px solid var(--line);padding-top:16px}
footer p{margin:0 0 8px}
"""

JS = """
document.querySelectorAll('table.sortable').forEach(function(t){
  t.querySelectorAll('th button').forEach(function(btn){
    btn.addEventListener('click',function(){
      var th=btn.parentElement, i=Array.prototype.indexOf.call(th.parentElement.children,th);
      var asc=th.getAttribute('aria-sort')!=='ascending';
      t.querySelectorAll('th').forEach(function(x){x.removeAttribute('aria-sort')});
      th.setAttribute('aria-sort',asc?'ascending':'descending');
      var tb=t.tBodies[0], rows=Array.prototype.slice.call(tb.rows);
      rows.sort(function(a,b){
        var av=a.cells[i].getAttribute('data-v'), bv=b.cells[i].getAttribute('data-v');
        if(av===null){av=a.cells[i].textContent} if(bv===null){bv=b.cells[i].textContent}
        var an=parseFloat(av), bn=parseFloat(bv), r;
        if(!isNaN(an)&&!isNaN(bn)){r=an-bn}else{r=String(av).localeCompare(String(bv))}
        return asc?r:-r;
      });
      rows.forEach(function(r){tb.appendChild(r)});
    });
  });
});
(function(){
  var q=document.getElementById('filter');
  if(!q) return;
  var fs=document.getElementById('f-sector'), ft=document.getElementById('f-trend'),
      fv=document.getElementById('f-vol'), fl=document.getElementById('f-long'),
      ff=document.getElementById('f-fade');
  function apply(){
    var s=q.value.toLowerCase(), sec=fs.value, tr=ft.value, vol=fv.checked,
        minL=parseFloat(fl.value)||0, minF=parseFloat(ff.value)||0;
    document.querySelectorAll('#all tbody tr').forEach(function(r){
      var ok=r.textContent.toLowerCase().indexOf(s)>-1
        && (!sec||r.getAttribute('data-sector')===sec)
        && (!tr||r.getAttribute('data-trend')===tr)
        && (!vol||r.getAttribute('data-vol')==='1')
        && parseFloat(r.getAttribute('data-long'))>=minL
        && parseFloat(r.getAttribute('data-fade'))>=minF;
      r.style.display=ok?'':'none';
    });
  }
  [q,fs,ft,fv,fl,ff].forEach(function(el){el.addEventListener('input',apply);el.addEventListener('change',apply);});
})();
"""


def _esc(x) -> str:
    return html.escape(str(x))


PAGES = [("index.html", "S&P 500 screen"), ("dip.html", "Dip research"), ("dashboard.html", "Dashboard")]


def _nav(active: str) -> str:
    """Shared nav bar so the three reports (screener, dip research, dashboard) link to each
    other -- otherwise each is only reachable if you already know its exact URL."""
    links = "".join(
        f'<a href="{href}"{" aria-current=\"page\"" if href == active else ""}>{_esc(label)}</a>'
        for href, label in PAGES)
    return f'<nav class="pages" aria-label="Reports">{links}</nav>'


def _pct(x, d=1, sign=False) -> str:
    if x is None or not np.isfinite(x):
        return "-"
    return f"{x * 100:+.{d}f}%" if sign else f"{x * 100:.{d}f}%"


def _cls(x) -> str:
    return "pos" if x > 0 else "neg" if x < 0 else ""


def _th(label: str, left: bool = False) -> str:
    return f'<th class="{"l" if left else ""}"><button type="button">{_esc(label)}</button></th>'


def _td(content: str, sort=None, cls: str = "") -> str:
    v = f' data-v="{sort}"' if sort is not None else ""
    c = f' class="{cls}"' if cls else ""
    return f"<td{v}{c}>{content}</td>"


def _stock_cell(r) -> str:
    return _td(f"<b>{_esc(r['ticker'])}</b><small>{_esc(str(r.get('name', ''))[:28])}</small>",
               sort=_esc(r["ticker"]))


def _chg_cell(r) -> str:
    return _td(_pct(r["chg1d"], 1, True), sort=f"{r['chg1d']:.5f}", cls=_cls(r["chg1d"]))


def _respect_cell(rate, n, base) -> str:
    if not np.isfinite(rate):
        return _td("-", sort=-1)
    w = int(round(np.clip((rate - base) / EXCESS_FULL, 0, 1) * 100))
    return _td(f'<span class="rbar"><i style="width:{w}%"></i></span>{rate:.0%}'
               f'<small>vs {base:.0%} baseline, {int(n)} tests</small>', sort=f"{rate - base:.4f}")


def _dist_cell(name, dist) -> str:
    if not name:
        return _td("-", sort="")
    return _td(f"{_esc(name)}<small>{dist:+.1f} ATR</small>", sort=_esc(name))


def _rows(df: pd.DataFrame, cells, attrs=None) -> list[str]:
    """Build <tr> strings, one per row, from a list of `(row) -> <td>...</td>` callables.
    `attrs`, if given, is `(row) -> ' data-x="y" ...'` for client-side filtering."""
    a = attrs or (lambda r: "")
    return [f"<tr{a(r)}>" + "".join(fn(r) for fn in cells) + "</tr>" for _, r in df.iterrows()]


def _table(headers: list[str], rows: list[str], tid: str = "") -> str:
    if not rows:
        return '<div class="wrap"><p class="empty">Nothing meets the criteria today.</p></div>'
    ths = "".join(_th(h, left=(i == 0 or h in ("Sector", "Most-respected levels", "Status", "Rule", "Grade")))
                  for i, h in enumerate(headers))
    tid_attr = f' id="{tid}"' if tid else ""
    return (f'<div class="wrap"><table class="sortable"{tid_attr}><thead><tr>{ths}</tr></thead>'
            f'<tbody>{"".join(rows)}</tbody></table></div>')


def render_html(res: pd.DataFrame, ctx: dict, breadth: dict, credit: dict, conviction: dict,
                sector_tbl: pd.DataFrame, asof: pd.Timestamp, n_screened: int, n_universe: int,
                top: int, demo: bool) -> str:
    # ---- regime strip
    span = max(ctx["hi"] - ctx["lo"], 1e-9)
    pos = float(np.clip((ctx["level"] - ctx["lo"]) / span, 0, 1)) * 100
    term = f'{ctx["term"]:.2f}' if np.isfinite(ctx["term"]) else "n/a"
    regime = f"""
<section class="regime" aria-label="VIX regime">
  <h2>{_esc(ctx['label'])}</h2>
  <p class="note">{_esc(ctx['note'])}</p>
  <p class="mult">Score adjustment: long setups x{ctx['long']:.2f}, fade setups x{ctx['fade']:.2f}</p>
  <div class="gauge" role="img" aria-label="VIX at {ctx['level']:.1f}, one-year range {ctx['lo']:.1f} to {ctx['hi']:.1f}">
    <div class="track"><span class="mk" style="left:{pos:.1f}%"></span></div>
    <div class="ends"><span>1-year low {ctx['lo']:.1f}</span><span>1-year high {ctx['hi']:.1f}</span></div>
  </div>
  <dl class="stats">
    <div><dt>VIX</dt><dd>{ctx['level']:.2f}</dd></div>
    <div><dt>1-year percentile</dt><dd>{ctx['pct'] * 100:.0f}</dd></div>
    <div><dt>vs 20-day average</dt><dd>{(ctx['level'] / ctx['ma20'] - 1) * 100:+.0f}%</dd></div>
    <div><dt>5-day change</dt><dd>{ctx['chg5'] * 100:+.0f}%</dd></div>
    <div><dt>VIX / VIX3M</dt><dd>{term}</dd></div>
  </dl>
</section>"""

    # ---- market conviction / breadth strip
    credit_dd = (f"{credit['level']:.2f}<small>{credit['chg20']:+.2f} pts / 20d</small>"
                if credit["ok"] else "n/a")
    conv_parts = "".join(f"<div><dt>{_esc(name)}</dt><dd>{val:.0f}</dd></div>" for name, val in conviction["parts"])
    conviction_sec = f"""
<section class="regime" aria-label="Market conviction">
  <h2>Conviction: {conviction['score']:.0f}/100, {_esc(conviction['label'])}</h2>
  <p class="note">Breadth, the VIX regime and the share of stocks with a live long setup, blended into one read
  on whether this is a market where dip-buying is favored. Heuristic overlay, not a per-stock multiplier.</p>
  <dl class="stats">
    {conv_parts}
    <div><dt>Above 50-day / 200-day</dt><dd>{breadth['above50']:.0f}% / {breadth['above200']:.0f}%</dd></div>
    <div><dt>RSI under 30 / over 70</dt><dd>{breadth['oversold']} / {breadth['overbought']}</dd></div>
    <div><dt>New 52-week highs / lows</dt><dd>{breadth['new_highs']} / {breadth['new_lows']}</dd></div>
    <div><dt>McClellan Oscillator</dt><dd>{breadth['mcclellan']:+.0f}</dd></div>
    <div><dt>Zweig breadth thrust</dt><dd>{"Firing" if breadth['zweig_thrust'] else "No"}</dd></div>
    <div><dt>High-yield credit spread</dt><dd>{credit_dd}</dd></div>
  </dl>
</section>"""

    sector_rows = _rows(sector_tbl, [
        lambda r: _td(_esc(r["sector"]), cls="l", sort=_esc(r["sector"])),
        lambda r: _td(str(int(r["n"])), sort=r["n"]),
        lambda r: _td(_pct(r["rs_1m"], 1, True), sort=f"{r['rs_1m']:.4f}" if np.isfinite(r["rs_1m"]) else -9,
                     cls=_cls(r["rs_1m"]) if np.isfinite(r["rs_1m"]) else ""),
        lambda r: _td(_pct(r["rs_3m"], 1, True), sort=f"{r['rs_3m']:.4f}" if np.isfinite(r["rs_3m"]) else -9,
                     cls=_cls(r["rs_3m"]) if np.isfinite(r["rs_3m"]) else ""),
        lambda r: _td(_pct(r["vs_sma50"], 1, True), sort=f"{r['vs_sma50']:.4f}", cls=_cls(r["vs_sma50"])),
        lambda r: _td(_pct(r["vs_sma200"], 1, True), sort=f"{r['vs_sma200']:.4f}", cls=_cls(r["vs_sma200"])),
    ])
    sector_tbl_html = _table(["Sector", "Stocks", "1m vs SPY", "3m vs SPY", "vs 50d", "vs 200d"], sector_rows)

    def sector_cell(r):
        return _td(_esc(r["sector"]), cls="l", sort=_esc(r["sector"]))

    def rsi_cell(r):
        return _td(f"{r['rsi']:.0f}", sort=f"{r['rsi']:.1f}")

    def pctb_cell(r):
        return _td(f"{r['pctb']:.2f}", sort=f"{r['pctb']:.3f}")

    def trend_cell(r):
        return _td(_esc(r["trend"]), sort=_esc(r["trend"]))

    def close_cell(r):
        return _td(f"{r['close']:.2f}", sort=f"{r['close']:.2f}")

    # ---- long table
    lg = res[res["support_score"] > 0].sort_values("long_score", ascending=False).head(top)
    long_tbl = _table(
        ["Stock", "Sector", "Close", "1d", "Score", "Support", "Respect", "RSI", "%B", "Trend"],
        _rows(lg, [_stock_cell, sector_cell, close_cell, _chg_cell,
                  lambda r: _td(f'<span class="score">{r["long_score"]:.0f}</span>', sort=f"{r['long_score']:.2f}"),
                  lambda r: _dist_cell(r["support_level"], r["support_dist_atr"]),
                  lambda r: _respect_cell(r["support_rate"], r["support_n"], r["support_base"]),
                  rsi_cell, pctb_cell, trend_cell]))

    # ---- fade table
    fd = res[res["resist_score"] > 0].sort_values("fade_score", ascending=False).head(top)
    fade_tbl = _table(
        ["Stock", "Sector", "Close", "1d", "Score", "Resistance", "Respect", "RSI", "%B", "Trend"],
        _rows(fd, [_stock_cell, sector_cell, close_cell, _chg_cell,
                  lambda r: _td(f'<span class="score">{r["fade_score"]:.0f}</span>', sort=f"{r['fade_score']:.2f}"),
                  lambda r: _dist_cell(r["resist_level"], r["resist_dist_atr"]),
                  lambda r: _respect_cell(r["resist_rate"], r["resist_n"], r["resist_base"]),
                  rsi_cell, pctb_cell, trend_cell]))

    # ---- broken table
    bk = res[res["broken_level"] != ""].assign(_x=lambda x: x["broken_rate"] - x["broken_base"]) \
        .sort_values("_x", ascending=False).head(top)
    broke_tbl = _table(
        ["Stock", "Sector", "Close", "1d", "Level lost", "Respect", "RSI", "Trend"],
        _rows(bk, [_stock_cell, sector_cell, close_cell, _chg_cell,
                  lambda r: _dist_cell(r["broken_level"], r["broken_dist_atr"]),
                  lambda r: _respect_cell(r["broken_rate"], r["broken_n"], r["broken_base"]),
                  rsi_cell, trend_cell]))

    # ---- all stocks (tagged with data-* attributes for the filter bar below)
    def all_attrs(r):
        return (f' data-sector="{_esc(r["sector"])}" data-trend="{_esc(r["trend"])}"'
                f' data-long="{r["long_score"]:.1f}" data-fade="{r["fade_score"]:.1f}"'
                f' data-vol="{1 if r["vol_confirm"] else 0}"')

    all_tbl = _table(
        ["Stock", "Sector", "Close", "1d", "Long", "Fade", "RSI", "%B", "vs 50d", "vs 200d",
         "Trend", "Weekly", "Most-respected levels"],
        _rows(res.sort_values("long_score", ascending=False),
             [_stock_cell, sector_cell, close_cell, _chg_cell,
              lambda r: _td(f"{r['long_score']:.0f}", sort=f"{r['long_score']:.2f}"),
              lambda r: _td(f"{r['fade_score']:.0f}", sort=f"{r['fade_score']:.2f}"),
              rsi_cell, pctb_cell,
              lambda r: _td(_pct(r["vs_sma50"], 1, True), sort=f"{r['vs_sma50']:.4f}", cls=_cls(r["vs_sma50"])),
              lambda r: _td(_pct(r["vs_sma200"], 1, True), sort=f"{r['vs_sma200']:.4f}", cls=_cls(r["vs_sma200"])),
              trend_cell,
              lambda r: _td(_esc(r["weekly_trend"]), sort=_esc(r["weekly_trend"])),
              lambda r: _td(_esc(r["respect_profile"] or "none above 50%"), cls="l", sort=_esc(r["respect_profile"]))],
             attrs=all_attrs),
        tid="all")
    sector_opts = "".join(f'<option value="{_esc(s)}">{_esc(s)}</option>'
                          for s in sorted(res["sector"].dropna().unique()))
    filters = f"""
<div class="filters">
  <input id="filter" type="search" placeholder="Filter by ticker, name, sector" aria-label="Filter stocks">
  <select id="f-sector" aria-label="Filter by sector"><option value="">All sectors</option>{sector_opts}</select>
  <select id="f-trend" aria-label="Filter by trend"><option value="">Any trend</option>
    <option>Up</option><option>Mixed</option><option>Down</option></select>
  <label><input type="checkbox" id="f-vol"> Volume confirmed</label>
  <label>Min long <input type="number" id="f-long" min="0" max="100" value="0"></label>
  <label>Min fade <input type="number" id="f-fade" min="0" max="100" value="0"></label>
</div>"""

    banner = ('<div class="demo">Synthetic demo data. These are random price series, not real stocks.</div>'
              if demo else "")
    body = f"""
{_nav("index.html")}
{banner}
<h1>S&amp;P 500 level-respect screen</h1>
<p class="sub">Data through {asof:%A, %B %d, %Y}. {n_screened} of {n_universe} stocks screened.</p>
{regime}
{conviction_sec}
<section class="block">
  <h2>Sector relative strength</h2>
  <p class="desc">Each sector's average 1- and 3-month return minus SPY's, and average distance from the 50-/200-day
  average. Sorted most-oversold-vs-SPY first.</p>
  {sector_tbl_html}
</section>
<section class="block">
  <h2>Pullbacks to a respected support</h2>
  <p class="desc">Price is sitting on a level this stock has held historically, inside an intact uptrend and
  with a pulled-back RSI or Bollinger %B. Ranked by VIX-adjusted long score.</p>
  {long_tbl}
</section>
<section class="block">
  <h2>Stretched into a respected resistance</h2>
  <p class="desc">Price has run into a level this stock has rejected before, with overbought readings or a weak
  trend. Ranked by VIX-adjusted fade score. This is a watch list for trimming or avoiding, not a short signal.</p>
  {fade_tbl}
</section>
<section class="block">
  <h2>Respected support just lost</h2>
  <p class="desc">Closed at least 1 ATR below a level the stock usually holds, within the last 5 sessions.
  When a level that reliably held finally breaks, the setup has failed.</p>
  {broke_tbl}
</section>
<section class="block">
  <h2>All screened stocks</h2>
  {filters}
  {all_tbl}
</section>
<footer>
  <p><b>How respect is measured.</b> A test is a bar reaching within {TOL_ATR} ATR of a level from the right side.
  It counts as held if the close moves {MOVE_ATR} ATR back away from the level within {HORIZON} bars before moving
  {MOVE_ATR} ATR through it. Each stock is also tested against placebo levels (the same level shifted several ATRs
  away) to set a baseline hold rate, because any price line holds more than half the time by construction. A level
  counts as respected only if it beats that baseline by {EXCESS_MIN * 100:.0f} points over at least {MIN_EVENTS}
  resolved tests. Small samples are shrunk toward the baseline.</p>
  <p><b>Limits.</b> Scores are heuristics built for screening, not a validated trading edge. With six levels across
  500 stocks, some will look respected by chance. Levels tested only a handful of times are noisy, even after shrinkage. This is not
  investment advice.</p>
</footer>"""
    return (f'<!doctype html><html lang="en"><head><meta charset="utf-8">'
            f'<meta name="viewport" content="width=device-width,initial-scale=1">'
            f'<title>S&amp;P 500 level-respect screen, {asof:%Y-%m-%d}</title>'
            f'<link rel="preconnect" href="https://fonts.googleapis.com">'
            f'<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&display=swap" rel="stylesheet">'
            f'<style>{CSS}</style></head><body><main>{body}</main><script>{JS}</script></body></html>')


# --------------------------------------------------------------------------- #
# Demo data (offline testing only)
# --------------------------------------------------------------------------- #
def demo_data(n: int = 60, seed: int = 7):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=1300)
    sectors = ["Information Technology", "Financials", "Health Care", "Industrials",
               "Consumer Discretionary", "Energy", "Utilities"]
    prices, uni = {}, []
    for i in range(n):
        t = f"DEMO{i:02d}"
        sigma, drift = rng.uniform(0.010, 0.022), rng.normal(0.0003, 0.0003)
        k = rng.choice([0.0, 0.0, 0.03, 0.06])          # some names gravitate to their 50-day
        lp = np.empty(len(dates))
        lp[0] = np.log(rng.uniform(30, 300))
        window = [np.exp(lp[0])]
        for j in range(1, len(dates)):
            sma = np.mean(window[-50:])
            lp[j] = lp[j - 1] + drift + k * (np.log(sma) - lp[j - 1]) + sigma * rng.standard_normal()
            window.append(np.exp(lp[j]))
        c = np.exp(lp)
        spread = c * sigma * rng.uniform(0.5, 1.2, len(dates))
        h = c + spread * rng.uniform(0.2, 1.0, len(dates))
        l = c - spread * rng.uniform(0.2, 1.0, len(dates))
        prices[t] = pd.DataFrame({"Open": np.r_[c[0], c[:-1]], "High": h, "Low": l, "Close": c,
                                  "Volume": rng.lognormal(15, 0.4, len(dates))}, index=dates)
        prices[t].attrs["pull_to_sma50"] = float(k)
        uni.append((t, f"Demo Company {i}", sectors[i % len(sectors)]))
    lv = np.empty(len(dates))
    lv[0] = np.log(18)
    for j in range(1, len(dates)):
        lv[j] = lv[j - 1] + 0.05 * (np.log(18) - lv[j - 1]) + 0.06 * rng.standard_normal()
    vix = pd.DataFrame({"Close": np.exp(lv)}, index=dates)
    vix3m = pd.DataFrame({"Close": pd.Series(np.exp(lv), index=dates).rolling(10, min_periods=1).mean() * 1.05})
    return pd.DataFrame(uni, columns=["ticker", "name", "sector"]), prices, vix, vix3m


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="docs", help="output folder (default: docs)")
    ap.add_argument("--cache", default="data/sp500_constituents.csv")
    ap.add_argument("--top", type=int, default=25, help="rows per candidate table")
    ap.add_argument("--period", default=PERIOD)
    ap.add_argument("--demo", action="store_true", help="synthetic data, no network")
    args = ap.parse_args()

    out = Path(args.out)
    (out / "data").mkdir(parents=True, exist_ok=True)

    if args.demo:
        universe, prices, vix, vix3m = demo_data()
        spy = prices[next(iter(prices))]["Close"]  # any series; demo credit/conviction numbers are illustrative
        credit = dict(ok=False, level=np.nan, chg5=np.nan, chg20=np.nan, note="not fetched in --demo mode")
    else:
        print("Loading S&P 500 constituents...")
        universe = load_universe(Path(args.cache))
        print(f"Downloading {len(universe)} tickers ({args.period})...")
        prices = download_prices(universe["ticker"].tolist(), args.period)
        idx = download_prices(["^VIX", "^VIX3M", "SPY"], "2y")
        vix, vix3m = idx.get("^VIX"), idx.get("^VIX3M")
        spy = idx["SPY"]["Close"] if "SPY" in idx else None
        if vix is None or len(vix) < 30:
            raise SystemExit("Could not download ^VIX; refusing to publish a report without it.")
        if len(prices) < 0.8 * len(universe):
            raise SystemExit(f"Only {len(prices)}/{len(universe)} tickers downloaded; "
                             "Yahoo is probably rate-limiting this IP. Try again later.")
        credit = credit_spread_context(out / "data" / "credit_spread.csv")

    ctx = vix_context(vix, vix3m)
    print(f"VIX {ctx['level']:.2f}, regime: {ctx['label']}")

    # Per-ticker earnings-date lookups (500 extra network calls) are skipped here: they are slow
    # and Yahoo already rate-limits the daily/period download at this universe size. dip_backtest.py
    # covers earnings-date awareness for the much smaller position + peer list instead.
    rows = []
    for t, df in prices.items():
        try:
            r = analyze_stock(t, df, spy)
        except Exception as exc:
            print(f"[warn] {t}: {exc}", file=sys.stderr)
            continue
        if r:
            r["last_date"] = df.index[-1]
            rows.append(r)
    if not rows:
        raise SystemExit("No stocks could be analyzed.")

    res = pd.DataFrame(rows).merge(universe, on="ticker", how="left")
    asof = pd.Series(res["last_date"]).mode()[0]
    res = res[res["last_date"] >= asof - pd.Timedelta(days=4)].copy()  # drop stale/halted names
    res["long_score"] = (res["long_raw"] * ctx["long"]).clip(upper=100)
    res["fade_score"] = (res["fade_raw"] * ctx["fade"]).clip(upper=100)

    breadth = dict(
        above50=100 * (res["vs_sma50"] > 0).mean(),
        above200=100 * (res["vs_sma200"] > 0).mean(),
        oversold=int((res["rsi"] < 30).sum()),
        overbought=int((res["rsi"] > 70).sum()),
        new_highs=int((res["pos52"] >= 0.99).sum()),
        new_lows=int((res["pos52"] <= 0.01).sum()),
        **breadth_series(prices),
    )
    sector_tbl = sector_strength(res)
    conviction = conviction_score(breadth, ctx, res)

    # Outputs
    cols_first = ["ticker", "name", "sector", "close", "chg1d", "long_score", "fade_score", "trend",
                  "respect_profile"]
    csv = res.sort_values("long_score", ascending=False)
    csv = csv[cols_first + [c for c in csv.columns if c not in cols_first and c != "last_date"]]
    csv.to_csv(out / "data" / "screen_latest.csv", index=False, float_format="%.4f")

    # Snapshot of market-level context, so dashboard.py can build a merged page without
    # re-downloading anything.
    import json
    (out / "data" / "market_context.json").write_text(json.dumps(dict(
        asof=str(asof.date()), vix=ctx, breadth=breadth, credit=credit, conviction=conviction,
        sectors=sector_tbl.to_dict("records"),
    ), default=float), encoding="utf-8")

    page = render_html(res, ctx, breadth, credit, conviction, sector_tbl, asof, len(res), len(universe),
                       args.top, args.demo)
    (out / "index.html").write_text(page, encoding="utf-8")

    if not args.demo:  # keep a log of what we flagged, so the ideas can be forward-tested later
        log = []
        for lst, score_col, lvl_col, gate in [("long", "long_score", "support_level", "support_score"),
                                              ("fade", "fade_score", "resist_level", "resist_score")]:
            pick = res[res[gate] > 0].sort_values(score_col, ascending=False).head(args.top)
            for rank, (_, r) in enumerate(pick.iterrows(), 1):
                log.append(dict(date=asof.date(), list=lst, rank=rank, ticker=r["ticker"],
                                score=round(r[score_col], 1), close=round(r["close"], 2),
                                level=r[lvl_col], vix=round(ctx["level"], 2), regime=ctx["label"]))
        hist_path = out / "data" / "signals_history.csv"
        new = pd.DataFrame(log)
        if hist_path.exists():
            old = pd.read_csv(hist_path)
            old = old[old["date"] != str(asof.date())]
            new = pd.concat([old, new], ignore_index=True)
        new.to_csv(hist_path, index=False)

    print(f"\nScreened {len(res)} stocks as of {asof.date()}. Report: {out / 'index.html'}")
    show = res[res["support_score"] > 0].sort_values("long_score", ascending=False).head(5)
    print("Top long setups:", ", ".join(f"{r.ticker} ({r.long_score:.0f})" for r in show.itertuples()) or "none")
    show = res[res["resist_score"] > 0].sort_values("fade_score", ascending=False).head(5)
    print("Top fade setups:", ", ".join(f"{r.ticker} ({r.fade_score:.0f})" for r in show.itertuples()) or "none")


if __name__ == "__main__":
    main()
