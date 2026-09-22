#!/usr/bin/env python3
"""
Dip-buying research: which of your holdings (and their sector peers) bounce
back reliably after dropping, and are any of them "low" right now?

Three dip rules are fixed in advance (never tuned per stock, to avoid overfitting):

  BB dip           close at/below the lower Bollinger Band (%B <= 0.10), RSI < 40,
                   and still above the 200-day average
  50-day pullback  price within 0.75 ATR of the 50-day in an intact uptrend, RSI < 50
  10% drawdown     10%+ below its 60-day high while above the 200-day average

Each signal is traded the way a real screener would use it: the signal appears at
the close, you buy the NEXT session's open, and you sell H sessions later at the
close.  Trades never overlap, and a 0.1% round-trip cost is deducted.

A stock is called "consistent" only if it passes ALL of these on the primary
10-session horizon:
  1. at least 15 trades
  2. win rate >= 60%
  3. average return beats the stock's own unconditional 10-day return, t >= 2
  4. that edge is positive BOTH in the first 60% and the last 40% of history
  5. the edge is positive in at least 60% of calendar years that had signals

Because ~150 tests are run, some stocks pass by luck.  The script measures how often
pure random-walk series would pass the same gate, and prints it in the report.

Run:  python dip_backtest.py            live data (yfinance), writes docs/dip.html
      python dip_backtest.py --demo     synthetic data, no network
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import screener as S

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
POSITIONS = ["BE", "SDS", "SPY", "VOO", "MU", "QQQ", "NVDA", "AMZN", "GOOGL", "META", "MSFT"]

# Peers are grouped by the holding(s) they resemble.
PEER_GROUPS = {
    "Semiconductors and memory (NVDA, MU)":
        ["AMD", "AVGO", "TSM", "QCOM", "AMAT", "LRCX", "KLAC", "MRVL", "TXN", "INTC", "ASML", "SMH"],
    "Mega-cap platforms (AMZN, GOOGL, META, MSFT)":
        ["AAPL", "ORCL", "NFLX", "CRM", "ADBE", "NOW", "TSLA"],
    "Power and clean energy (BE)":
        ["VST", "CEG", "GEV", "NRG", "FSLR", "ENPH", "PLUG"],
    "Index ETFs (SPY, VOO, QQQ)":
        ["IVV", "DIA", "IWM", "XLK"],
    "Inverse ETFs (SDS)":
        ["SH", "SPXU", "SQQQ"],
}
POSITION_GROUP = {
    "BE": "Power and clean energy (BE)", "SDS": "Inverse ETFs (SDS)",
    "SPY": "Index ETFs (SPY, VOO, QQQ)", "VOO": "Index ETFs (SPY, VOO, QQQ)",
    "QQQ": "Index ETFs (SPY, VOO, QQQ)", "MU": "Semiconductors and memory (NVDA, MU)",
    "NVDA": "Semiconductors and memory (NVDA, MU)",
    "AMZN": "Mega-cap platforms (AMZN, GOOGL, META, MSFT)",
    "GOOGL": "Mega-cap platforms (AMZN, GOOGL, META, MSFT)",
    "META": "Mega-cap platforms (AMZN, GOOGL, META, MSFT)",
    "MSFT": "Mega-cap platforms (AMZN, GOOGL, META, MSFT)",
}
LEVERAGED_INVERSE = {"SDS", "SH", "SPXU", "SQQQ"}   # decay structurally; dip rules don't apply

HORIZONS = (5, 10, 20)     # sessions held
PRIMARY_H = 10
COST = 0.001               # 0.1% round trip
HIGH_COST = 0.003          # sensitivity check: does the edge survive 3x the assumed cost?
OOS_FRAC = 0.40            # last 40% of history is the out-of-sample check
WF_FOLDS = 5               # sequential folds for the walk-forward edge-sign check
MC_DRAWS = 1000            # bootstrap resamples for the win-rate / edge confidence interval
MIN_TRADES = 15
MIN_WIN = 0.60
MIN_T = 2.0
MIN_YEARLY = 0.60
MIN_YEARS = 4
STRATEGIES = ["BB dip", "50-day pullback", "10% drawdown"]

ACCOUNT_SIZE = 100_000     # edit to your actual account size; only used for the Kelly sizing suggestion
KELLY_CAP = 0.25           # hard cap on suggested position size, regardless of what Kelly says


# --------------------------------------------------------------------------- #
# Signals and trades
# --------------------------------------------------------------------------- #
def dip_signals(d: pd.DataFrame) -> dict[str, np.ndarray]:
    c = d["Close"]
    up = c > d["sma200"]
    bb = (d["pctb"] <= 0.10) & (d["rsi"] < 40) & up
    pull = ((c - d["sma50"]).abs() <= 0.75 * d["atr"]) & (d["sma50"] > d["sma200"]) & up & (d["rsi"] < 50)
    dd = (c <= 0.90 * c.rolling(60).max()) & up
    return {"BB dip": bb.fillna(False).to_numpy(bool),
            "50-day pullback": pull.fillna(False).to_numpy(bool),
            "10% drawdown": dd.fillna(False).to_numpy(bool)}


def forward_arrays(o: np.ndarray, l: np.ndarray, c: np.ndarray, H: int):
    """For every possible entry session e: return of buying the open of e and selling
    the close of e+H-1, and the worst low seen in between (max adverse excursion)."""
    n = len(c)
    base = np.full(n, np.nan)
    mae = np.full(n, np.nan)
    if n < H + 1:
        return base, mae
    e = np.arange(0, n - H + 1)
    base[e] = c[e + H - 1] / o[e] - 1
    lows = pd.Series(l).rolling(H).min().shift(-(H - 1)).to_numpy()
    mae[e] = lows[e] / o[e] - 1
    return base, mae


def pick_trades(sig: np.ndarray, H: int) -> np.ndarray:
    """Signal at close of t -> enter at open of t+1.  No overlapping positions."""
    n = len(sig)
    out, busy = [], -1
    for t in np.flatnonzero(sig):
        if t < busy:
            continue
        e = t + 1
        if e + H - 1 > n - 1:
            break
        out.append(e)
        busy = e + H - 1
    return np.array(out, dtype=int)


def fold_edges(dates: pd.DatetimeIndex, e_idx: np.ndarray, base: np.ndarray, cost: float,
              k: int = WF_FOLDS) -> list[float]:
    """Split the history into k sequential folds and compute this rule's edge within each --
    a stronger check than one static in/out-of-sample split. There is no walk-forward
    re-optimization here because the three dip rules are fixed in advance by design (see the
    module docstring); this instead checks the sign of the (untuned) edge holds up fold by fold."""
    n = len(dates)
    bounds = np.linspace(0, n, k + 1).astype(int)
    out = []
    for i in range(k):
        lo, hi = bounds[i], bounds[i + 1]
        e = e_idx[(e_idx >= lo) & (e_idx < hi)]
        b = base[lo:hi]
        if len(e) < 3 or not np.isfinite(b).any():
            continue
        out.append(float((base[e] - cost).mean() - np.nanmean(b)))
    return out


def summarize(dates: pd.DatetimeIndex, e_idx: np.ndarray, base: np.ndarray, mae: np.ndarray,
              cost: float = COST) -> dict:
    n = len(e_idx)
    empty = dict(n=0, win=np.nan, mean=np.nan, median=np.nan, base=np.nan, edge=np.nan, t=np.nan,
                 mae=np.nan, worst=np.nan, edge_is=np.nan, edge_oos=np.nan, n_oos=0,
                 yearly=np.nan, years=0, avg_win=np.nan, avg_loss=np.nan,
                 wf_folds=0, wf_positive_frac=np.nan, edge_lo=np.nan, edge_hi=np.nan,
                 win_lo=np.nan, win_hi=np.nan, edge_hc=np.nan)
    if n == 0:
        return empty
    r = base[e_idx] - cost
    bmean = float(np.nanmean(base))
    edge = float(r.mean() - bmean)
    sd = r.std(ddof=1) if n > 1 else np.nan
    t = edge / (sd / np.sqrt(n)) if n > 1 and sd > 0 else np.nan

    split = int(len(dates) * (1 - OOS_FRAC))
    is_e, oos_e = e_idx[e_idx < split], e_idx[e_idx >= split]
    b_is, b_oos = np.nanmean(base[:split]), np.nanmean(base[split:])
    edge_is = float((base[is_e] - cost).mean() - b_is) if len(is_e) else np.nan
    edge_oos = float((base[oos_e] - cost).mean() - b_oos) if len(oos_e) else np.nan

    years = dates.year.to_numpy()
    hits = []
    for y in np.unique(years[e_idx]):
        m = years[e_idx] == y
        by = base[years == y]
        if m.sum() >= 2 and np.isfinite(by).any():
            hits.append(r[m].mean() - np.nanmean(by) > 0)
    yearly = float(np.mean(hits)) if len(hits) >= MIN_YEARS else np.nan

    wins, losses = r[r > 0], r[r <= 0]
    avg_win = float(wins.mean()) if len(wins) else np.nan
    avg_loss = float(-losses.mean()) if len(losses) else np.nan

    folds = fold_edges(dates, e_idx, base, cost)
    wf_frac = float(np.mean([f > 0 for f in folds])) if len(folds) >= 3 else np.nan

    # Monte Carlo: bootstrap-resample the trades themselves to get a confidence interval on the
    # win rate and edge, instead of trusting the single point estimate above.
    if n >= 8:
        rng = np.random.default_rng(42)
        samp = r[rng.integers(0, n, size=(MC_DRAWS, n))]
        boot_edge = samp.mean(axis=1) - bmean
        boot_win = (samp > 0).mean(axis=1)
        edge_lo, edge_hi = float(np.percentile(boot_edge, 5)), float(np.percentile(boot_edge, 95))
        win_lo, win_hi = float(np.percentile(boot_win, 5)), float(np.percentile(boot_win, 95))
    else:
        edge_lo = edge_hi = win_lo = win_hi = np.nan

    edge_hc = float((base[e_idx] - HIGH_COST).mean() - bmean)   # does the edge survive 3x the cost?

    return dict(n=n, win=float((r > 0).mean()), mean=float(r.mean()), median=float(np.median(r)),
                base=bmean, edge=edge, t=float(t) if np.isfinite(t) else np.nan,
                mae=float(np.mean(mae[e_idx])), worst=float(r.min()), edge_is=edge_is,
                edge_oos=edge_oos, n_oos=len(oos_e), yearly=yearly, years=len(hits),
                avg_win=avg_win, avg_loss=avg_loss, wf_folds=len(folds), wf_positive_frac=wf_frac,
                edge_lo=edge_lo, edge_hi=edge_hi, win_lo=win_lo, win_hi=win_hi, edge_hc=edge_hc)


def buy_hold_stats(close: pd.Series) -> dict:
    """Total and annualized return of simply holding the stock over the same history used for
    the backtest -- the benchmark every dip rule has to beat, not just its own baseline."""
    years = (close.index[-1] - close.index[0]).days / 365.25
    total = float(close.iloc[-1] / close.iloc[0] - 1)
    cagr = float((1 + total) ** (1 / years) - 1) if years > 0.5 else np.nan
    return dict(bh_total=total, bh_cagr=cagr)


def kelly_fraction(win: float, avg_win: float, avg_loss: float, cap: float = KELLY_CAP) -> float:
    """Half-Kelly position size: f* = win - (1-win)/R, R = avg win / avg loss, halved and capped
    because the win-rate/payoff estimates behind it are themselves noisy."""
    if not (np.isfinite(win) and np.isfinite(avg_win) and np.isfinite(avg_loss)) or avg_loss <= 0:
        return np.nan
    f = win - (1 - win) / (avg_win / avg_loss)
    return float(np.clip(0.5 * f, 0, cap))


def portfolio_exposure(res: pd.DataFrame) -> pd.DataFrame:
    """Equal-weighted exposure by factor/sector group across your positions (reuses the existing
    POSITION_GROUP/PEER_GROUPS labels). Equal-weighted because this script doesn't know your actual
    dollar sizes per position -- edit ACCOUNT_SIZE and this if you want to weight it properly."""
    mine = res[res["is_position"]]
    g = mine.groupby("group").agg(n=("ticker", "size"), tickers=("ticker", lambda s: ", ".join(s)))
    g["weight"] = 100 * g["n"] / len(mine)
    return g.sort_values("weight", ascending=False).reset_index()


def grade(s: dict) -> tuple[int, str]:
    """Five pre-declared criteria -> (score 0-5, tier)."""
    ok = [
        s["n"] >= MIN_TRADES,
        s["win"] >= MIN_WIN if s["n"] else False,
        (s["edge"] > 0 and s["t"] >= MIN_T) if np.isfinite(s["t"]) else False,
        (s["edge_is"] > 0 and s["edge_oos"] > 0 and s["n_oos"] >= 5)
        if np.isfinite(s["edge_is"]) and np.isfinite(s["edge_oos"]) else False,
        (s["yearly"] >= MIN_YEARLY) if np.isfinite(s["yearly"]) else False,
    ]
    score = int(sum(ok))
    if score == 5:
        return score, "Consistent"
    if score == 4 and ok[0] and ok[2]:
        return score, "Mostly"
    return score, "No"


def evaluate(df: pd.DataFrame, horizons=HORIZONS):
    """Full backtest of the three dip rules on one price history."""
    d = S.add_indicators(df)
    o = (d["Open"] if "Open" in d else d["Close"].shift(1).fillna(d["Close"])).to_numpy(float)
    l, c = d["Low"].to_numpy(float), d["Close"].to_numpy(float)
    sigs = dip_signals(d)
    res = {}
    for H in horizons:
        base, mae = forward_arrays(o, l, c, H)
        for name, sig in sigs.items():
            idx = pick_trades(sig, H)
            res[(name, H)] = (summarize(d.index, idx, base, mae), idx, base)
    return d, sigs, res


# --------------------------------------------------------------------------- #
# Synthetic series (demo mode, tests, and the false-positive calibration)
# --------------------------------------------------------------------------- #
def synth_series(kind: str, seed: int, n: int = 2520, vol: float | None = None,
                 end: pd.Timestamp | None = None) -> pd.DataFrame:
    """kind='rw': random walk with drift.
    kind='ou': price snapping back toward a rising trend (4-day half-life), a planted 'dip-bouncer'."""
    rng = np.random.default_rng(seed)
    vol = vol or rng.uniform(0.012, 0.028)
    drift = 0.0004
    if kind == "ou":
        x = np.zeros(n)
        for j in range(1, n):
            x[j] = 0.85 * x[j - 1] + vol * rng.standard_normal()
        logp = np.arange(n) * drift + x
    else:
        logp = np.cumsum(drift + vol * rng.standard_normal(n))
    c = 100 * np.exp(logp)
    o = np.r_[c[0], c[:-1]] * (1 + 0.1 * vol * rng.standard_normal(n))
    spread = c * vol * rng.uniform(0.4, 0.9, n)
    h = np.maximum(c, o) + spread * rng.uniform(0.1, 1.0, n)
    l = np.minimum(c, o) - spread * rng.uniform(0.1, 1.0, n)
    end = end or pd.Timestamp.today().normalize()
    idx = pd.bdate_range(end=end, periods=n)
    return pd.DataFrame({"Open": o, "High": h, "Low": l, "Close": c,
                         "Volume": rng.lognormal(15, 0.4, n)}, index=idx)


def passes(res: dict) -> bool:
    return any(grade(res[(s, PRIMARY_H)][0])[1] != "No" for s in STRATEGIES)


def null_pass_rate(n_series: int = 150, seed: int = 1) -> dict:
    """How often would pure random walks earn a 'Consistent'/'Mostly' grade from this gate?"""
    hit = {s: 0 for s in STRATEGIES}
    any_hit = 0
    for i in range(n_series):
        _, _, res = evaluate(synth_series("rw", seed * 10_000 + i), horizons=(PRIMARY_H,))
        g = {s: grade(res[(s, PRIMARY_H)][0])[1] != "No" for s in STRATEGIES}
        for s in STRATEGIES:
            hit[s] += g[s]
        any_hit += any(g.values())
    out = {s: hit[s] / n_series for s in STRATEGIES}
    out["any"] = any_hit / n_series
    return out


# --------------------------------------------------------------------------- #
# Per-ticker analysis
# --------------------------------------------------------------------------- #
def analyze(ticker: str, df: pd.DataFrame, spy_ret: pd.Series | None) -> tuple[dict, list[dict]] | None:
    if len(df) < 500:
        return None
    d, sigs, res = evaluate(df)
    close = d["Close"]
    last = d.iloc[-1]

    # Grade each rule at the primary horizon; keep the best
    graded = []
    for s in STRATEGIES:
        st = res[(s, PRIMARY_H)][0]
        sc, tier = grade(st)
        graded.append((sc, st["t"] if np.isfinite(st["t"]) else -9, s, st, tier))
    graded.sort(key=lambda z: (z[0], z[1]), reverse=True)
    sc, _, best, st, tier = graded[0]
    if ticker in LEVERAGED_INVERSE:
        tier = "n/a (decay)"

    # Where is each rule's trigger right now?
    a, c0 = float(last["atr"]), float(last["Close"])
    up = c0 > float(last["sma200"])
    trend_ok = {"BB dip": up, "50-day pullback": bool(up and last["sma50"] > last["sma200"]),
                "10% drawdown": up}
    hi60 = float(close.iloc[-60:].max())
    # Price-only trigger; ignores the rule's RSI < 40 leg, so this price alone can be reached
    # without the signal actually firing.
    bb_trig = float(last["bb_lo"] + 0.10 * (last["bb_up"] - last["bb_lo"]))
    # 50-day pullback is a two-sided zone: price can be above it (needs to fall) or,
    # after a steeper drop, below it (needs to rise back up) -- pick the near edge either way.
    lo50, hi50 = float(last["sma50"] - 0.75 * a), float(last["sma50"] + 0.75 * a)
    pull_trig = hi50 if c0 > hi50 else lo50 if c0 < lo50 else c0
    trig = {"BB dip": bb_trig, "50-day pullback": pull_trig, "10% drawdown": 0.90 * hi60}
    active = bool(sigs[best][-1])
    recent = bool(sigs[best][-3:].any())
    # negative: price must fall to reach the trigger; positive: must rise; 0 if already there
    gap = 0.0 if trig[best] == c0 else trig[best] / c0 - 1
    if active:
        status = "Signal today"
    elif recent:
        status = "Signal in last 3 sessions"
    elif not trend_ok[best]:
        status = "Trend filter off"
    else:
        status = "Armed"

    r = close.pct_change()
    r1 = r.iloc[-252:]
    vol = float(r1.std() * np.sqrt(252))
    beta = corr = np.nan
    if spy_ret is not None:
        j = pd.concat([r1, spy_ret.reindex(r1.index)], axis=1).dropna()
        if len(j) > 60:
            beta = float(np.cov(j.iloc[:, 0], j.iloc[:, 1])[0, 1] / np.var(j.iloc[:, 1], ddof=1))
            corr = float(j.corr().iloc[0, 1])
    w = close.iloc[-252:]
    kelly = kelly_fraction(st["win"], st["avg_win"], st["avg_loss"])

    row = dict(
        ticker=ticker, close=c0, best_rule=best, grade=tier, score=sc, trades=st["n"], win=st["win"],
        avg_ret=st["mean"], baseline=st["base"], edge=st["edge"], t_stat=st["t"], edge_is=st["edge_is"],
        edge_oos=st["edge_oos"], n_oos=st["n_oos"], yearly=st["yearly"], years=st["years"],
        mae=st["mae"], worst=st["worst"], status=status, trigger_price=trig[best], gap_to_trigger=gap,
        rsi=float(last["rsi"]), pctb=float(last["pctb"]) if np.isfinite(last["pctb"]) else np.nan,
        vs_sma50=c0 / float(last["sma50"]) - 1, vs_sma200=c0 / float(last["sma200"]) - 1,
        off_52w_high=c0 / float(w.max()) - 1, ret_1m=c0 / float(close.iloc[-22]) - 1,
        ret_3m=c0 / float(close.iloc[-64]) - 1, ret_12m=c0 / float(close.iloc[-253]) - 1,
        vol=vol, beta=beta, corr_spy=corr,
        max_dd_1y=float((w / w.cummax() - 1).min()),
        n_bars=len(d),
        # backtest rigor: sequential-fold edge sign, bootstrap CI, cost sensitivity
        wf_folds=st["wf_folds"], wf_positive_frac=st["wf_positive_frac"],
        edge_lo=st["edge_lo"], edge_hi=st["edge_hi"], win_lo=st["win_lo"], win_hi=st["win_hi"],
        edge_hc=st["edge_hc"],
        # position sizing, once a signal fires
        kelly_frac=kelly, kelly_dollars=kelly * ACCOUNT_SIZE if np.isfinite(kelly) else np.nan,
    )
    row.update(buy_hold_stats(close))
    for s in STRATEGIES:                                          # per-rule detail for the CSV
        for H in HORIZONS:
            stt = res[(s, H)][0]
            k = f"{s.split()[0].lower()}_{H}d"
            row[f"{k}_n"], row[f"{k}_win"], row[f"{k}_edge"] = stt["n"], stt["win"], stt["edge"]

    # Events for the pooled VIX study (primary horizon, every rule)
    events = []
    for s in STRATEGIES:
        stt, idx, base = res[(s, PRIMARY_H)]
        for e in idx:
            events.append(dict(ticker=ticker, rule=s, date=d.index[max(e - 1, 0)],
                               excess=float(base[e] - COST - stt["base"])))
    return row, events


def vix_study(events: pd.DataFrame, vix: pd.Series) -> pd.DataFrame:
    """Do dips work better when the VIX is high, or once it starts falling?"""
    if events.empty:
        return pd.DataFrame()
    v = vix.reindex(events["date"], method="ffill").to_numpy()
    chg = (vix / vix.shift(5) - 1).reindex(events["date"], method="ffill").to_numpy()
    ev = events.assign(vix=v, vchg=chg).dropna(subset=["vix"])
    buckets = [("VIX under 15", ev["vix"] < 15), ("VIX 15 to 20", (ev["vix"] >= 15) & (ev["vix"] < 20)),
               ("VIX 20 to 25", (ev["vix"] >= 20) & (ev["vix"] < 25)), ("VIX 25 and above", ev["vix"] >= 25),
               ("VIX falling over 5 days", ev["vchg"] < 0), ("VIX rising over 5 days", ev["vchg"] > 0)]
    rows = []
    for name, m in buckets:
        sub = ev[m]
        if len(sub):
            rows.append(dict(bucket=name, signals=len(sub), days=sub["date"].nunique(),
                             excess=float(sub["excess"].mean()),
                             positive=float((sub["excess"] > 0).mean())))
    return pd.DataFrame(rows)


def _insider_sign(text: pd.Series) -> np.ndarray:
    """+1 open-market purchase, -1 sale, 0 for grants/gifts/option exercises (not a market view)."""
    t = text.astype(str)
    return np.where(t.str.contains("Sale", case=False, na=False), -1,
             np.where(t.str.contains("Purchase|Buy", case=False, na=False, regex=True), 1, 0))


def extra_signals(ticker: str) -> dict:
    """Per-ticker context that's too slow and flaky for the full 500-stock screener, but fine for
    this ~50-ticker watchlist: short interest (days to cover), net insider buying/selling from SEC
    Form 4 filings over the last 6 months, the near-term put/call volume ratio, and the next
    earnings date. Every lookup is independently wrapped, so one flaky ticker or a Yahoo hiccup
    just leaves that field blank instead of failing the run."""
    out = dict(short_pct_float=np.nan, days_to_cover=np.nan, insider_net_shares=np.nan,
               put_call=np.nan, earnings_date=pd.NaT)
    import yfinance as yf
    t = yf.Ticker(ticker)
    try:
        info = t.info or {}
        out["short_pct_float"] = S._f(info.get("shortPercentOfFloat"))
        out["days_to_cover"] = S._f(info.get("shortRatio"))
    except Exception:
        pass
    try:
        tx = t.insider_transactions
        if tx is not None and len(tx):
            tx = tx.copy()
            tx["Start Date"] = pd.to_datetime(tx["Start Date"], errors="coerce")
            recent = tx[tx["Start Date"] >= pd.Timestamp.today() - pd.Timedelta(days=180)]
            # yfinance's "Transaction" column is blank; the actual description ("Sale at price...",
            # "Stock Award(Grant)...") lives in "Text".
            txt = recent["Text"] if "Text" in recent else pd.Series(dtype=str)
            sign = _insider_sign(txt)
            shares = pd.to_numeric(recent.get("Shares"), errors="coerce").fillna(0).to_numpy()
            out["insider_net_shares"] = float((sign * shares).sum())
    except Exception:
        pass
    try:
        exps = t.options
        if exps:
            ch = t.option_chain(exps[0])
            cv = pd.to_numeric(ch.calls["volume"], errors="coerce").sum()
            pv = pd.to_numeric(ch.puts["volume"], errors="coerce").sum()
            if cv > 0:
                out["put_call"] = float(pv / cv)
    except Exception:
        pass
    try:
        cal = t.calendar
        ed = cal.get("Earnings Date") if isinstance(cal, dict) else None
        if ed:
            out["earnings_date"] = pd.Timestamp(ed[0] if isinstance(ed, (list, tuple)) else ed)
    except Exception:
        pass
    return out


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #
def _p(x, d=1, sign=True):
    return "-" if x is None or not np.isfinite(x) else (f"{x * 100:+.{d}f}%" if sign else f"{x * 100:.{d}f}%")


def _ticker_table(df: pd.DataFrame, extras: bool = False, sizing: bool = False,
                  empty_msg: str | None = None) -> str:
    rows = []
    for _, r in df.iterrows():
        g = r["grade"]
        grade_txt = f"{g}<small>{int(r['score'])} of 5 tests</small>"
        status = S._esc(r["status"])
        if r["status"] == "Armed" and np.isfinite(r["gap_to_trigger"]) and r["gap_to_trigger"] != 0:
            verb = "needs" if r["gap_to_trigger"] < 0 else "needs to rise"
            status += f"<small>{verb} {r['gap_to_trigger'] * 100:+.1f}% (${r['trigger_price']:.2f})</small>"
        elif r["status"] in ("Signal today", "Signal in last 3 sessions"):
            status = f"<b>{status}</b>"
        if r.get("earnings_in_hold"):
            status += ' <span class="warn">⚠ earnings in hold window</span>'
        cells = [
            S._td(f"<b>{S._esc(r['ticker'])}</b><small>{S._esc(str(r.get('group', '')).split(' (')[0])}</small>", sort=S._esc(r["ticker"])),
            S._td(f"{r['close']:.2f}", sort=f"{r['close']:.2f}"),
            S._td(status, cls="l", sort=S._esc(r["status"])),
            S._td(S._esc(r["best_rule"]), cls="l", sort=S._esc(r["best_rule"])),
            S._td(grade_txt, sort=f"{r['score']}", cls="l"),
            S._td(f"{_p(r['win'], 0, False)} won<small>{int(r['trades'])} trades</small>",
                  sort=f"{r['win']:.3f}" if np.isfinite(r["win"]) else -1),
            S._td(_p(r["avg_ret"]), sort=f"{r['avg_ret']:.4f}" if np.isfinite(r["avg_ret"]) else -9,
                  cls=S._cls(r["avg_ret"]) if np.isfinite(r["avg_ret"]) else ""),
            S._td(_p(r["edge"]), sort=f"{r['edge']:.4f}" if np.isfinite(r["edge"]) else -9,
                  cls=S._cls(r["edge"]) if np.isfinite(r["edge"]) else ""),
            S._td(_p(r["edge_oos"]), sort=f"{r['edge_oos']:.4f}" if np.isfinite(r["edge_oos"]) else -9,
                  cls=S._cls(r["edge_oos"]) if np.isfinite(r["edge_oos"]) else ""),
            S._td(_p(r["mae"]), sort=f"{r['mae']:.4f}" if np.isfinite(r["mae"]) else -9, cls="neg"),
        ]
        if sizing:
            ci = (f"{_p(r['edge_lo'], 1)} to {_p(r['edge_hi'], 1)}" if np.isfinite(r["edge_lo"]) else "-")
            wf = f"{int(round(r['wf_positive_frac'] * r['wf_folds']))}/{int(r['wf_folds'])}" \
                if np.isfinite(r.get("wf_positive_frac", np.nan)) else "-"
            kelly = (f"{r['kelly_frac'] * 100:.1f}%<small>${r['kelly_dollars']:,.0f} of ${ACCOUNT_SIZE:,.0f}</small>"
                    if np.isfinite(r.get("kelly_frac", np.nan)) and r["kelly_frac"] > 0
                    else "-<small>edge or sample too weak</small>")
            cells += [S._td(kelly, sort=f"{r.get('kelly_frac', 0):.4f}" if np.isfinite(r.get("kelly_frac", np.nan)) else -1),
                      S._td(ci, sort=f"{r['edge_lo']:.4f}" if np.isfinite(r["edge_lo"]) else -9),
                      S._td(wf, sort=f"{r.get('wf_positive_frac', 0):.3f}" if np.isfinite(r.get("wf_positive_frac", np.nan)) else -1),
                      S._td(_p(r["edge_hc"]), sort=f"{r['edge_hc']:.4f}" if np.isfinite(r["edge_hc"]) else -9,
                            cls=S._cls(r["edge_hc"]) if np.isfinite(r["edge_hc"]) else "")]
        if extras:
            cells += [S._td(_p(r["off_52w_high"]), sort=f"{r['off_52w_high']:.4f}", cls=S._cls(r["off_52w_high"])),
                      S._td(_p(r["vol"], 0, False), sort=f"{r['vol']:.3f}"),
                      S._td(f"{r['beta']:.2f}" if np.isfinite(r["beta"]) else "-", sort=f"{r['beta']:.3f}" if np.isfinite(r["beta"]) else -9),
                      S._td(f"{r['rsi']:.0f}", sort=f"{r['rsi']:.1f}"),
                      S._td(_p(r.get("bh_cagr", np.nan), 0), sort=f"{r.get('bh_cagr', np.nan):.4f}" if np.isfinite(r.get("bh_cagr", np.nan)) else -9),
                      S._td(f"{r['short_pct_float'] * 100:.1f}%" if np.isfinite(r.get("short_pct_float", np.nan)) else "-",
                            sort=f"{r.get('short_pct_float', 0):.4f}" if np.isfinite(r.get("short_pct_float", np.nan)) else -1),
                      S._td(f"{r['days_to_cover']:.1f}d" if np.isfinite(r.get("days_to_cover", np.nan)) else "-",
                            sort=f"{r.get('days_to_cover', 0):.2f}" if np.isfinite(r.get("days_to_cover", np.nan)) else -1),
                      S._td(f"{r['insider_net_shares']:+,.0f}" if np.isfinite(r.get("insider_net_shares", np.nan)) else "-",
                            sort=f"{r.get('insider_net_shares', 0):.0f}" if np.isfinite(r.get("insider_net_shares", np.nan)) else -1e12,
                            cls=S._cls(r["insider_net_shares"]) if np.isfinite(r.get("insider_net_shares", np.nan)) else ""),
                      S._td(f"{r['put_call']:.2f}" if np.isfinite(r.get("put_call", np.nan)) else "-",
                            sort=f"{r.get('put_call', 0):.3f}" if np.isfinite(r.get("put_call", np.nan)) else -1)]
        rows.append("<tr>" + "".join(cells) + "</tr>")
    heads = ["Stock", "Close", "Status", "Rule", "Grade", "Record", "Avg 10d", "Edge", "Edge, recent",
             "Dip after entry"]
    if sizing:
        heads += ["Suggested size (half-Kelly)", "Edge, 90% CI", "Walk-forward folds +", "Edge at 3x cost"]
    if extras:
        heads += ["From 52w high", "Volatility", "Beta", "RSI", "Buy & hold CAGR", "Short % float",
                  "Days to cover", "Insider net shares (6mo)", "Put/Call vol"]
    if not rows:
        return f'<div class="wrap"><p class="empty">{S._esc(empty_msg or "Nothing meets the criteria today.")}</p></div>'
    return S._table(heads, rows)


def render(res: pd.DataFrame, ctx: dict, vixtab: pd.DataFrame, null: dict, corr_pairs: list,
           asof: pd.Timestamp, demo: bool) -> str:
    graded = res[res["grade"].isin(["Consistent", "Mostly"])]
    catch = graded[graded["status"].isin(["Signal today", "Signal in last 3 sessions"])] \
        .sort_values(["score", "t_stat"], ascending=False)
    watch = graded[(graded["status"] == "Armed") & (graded["gap_to_trigger"] > -0.10)
                  & (graded["gap_to_trigger"] <= 0)] \
        .sort_values("gap_to_trigger", ascending=False)
    mine = res[res["is_position"]].copy()
    mine["_o"] = mine["ticker"].map({t: i for i, t in enumerate(POSITIONS)})
    mine = mine.sort_values("_o")
    peers = res[~res["is_position"]].sort_values(["group", "score", "t_stat"], ascending=[True, False, False])

    n_ok = int((res["grade"].isin(["Consistent", "Mostly"])).sum())
    exp_fp = null["any"] * len(res[res["grade"] != "n/a (decay)"])

    peer_blocks = ""
    for grp, sub in peers.groupby("group", sort=False):
        peer_blocks += f"<h3>{S._esc(grp)}</h3>{_ticker_table(sub, extras=True)}"

    overlap = ""
    if corr_pairs:
        items = ", ".join(f"{a} and {b} ({c:.2f})" for a, b, c in corr_pairs)
        overlap = f'<p class="desc"><b>Overlap in your holdings:</b> these pairs moved almost in lockstep over the last year: {items}. ' \
                  "Holding several of them adds less diversification than the ticker count suggests.</p>"

    exposure = portfolio_exposure(res)
    exp_rows = S._rows(exposure, [
        lambda r: S._td(S._esc(r["group"]), cls="l", sort=S._esc(r["group"])),
        lambda r: S._td(f"{r['weight']:.0f}%", sort=f"{r['weight']:.2f}"),
        lambda r: S._td(str(int(r["n"])), sort=r["n"]),
        lambda r: S._td(S._esc(r["tickers"]), cls="l"),
    ])
    exposure_tbl = S._table(["Factor / sector group", "Weight", "Positions", "Tickers"], exp_rows)

    vix_rows = []
    for _, r in vixtab.iterrows():
        vix_rows.append("<tr>" + S._td(S._esc(r["bucket"]), cls="l") + S._td(str(int(r["signals"])), sort=r["signals"]) \
            + S._td(str(int(r["days"])), sort=r["days"]) \
            + S._td(_p(r["excess"], 2), sort=f"{r['excess']:.4f}", cls=S._cls(r["excess"])) \
            + S._td(_p(r["positive"], 0, False), sort=f"{r['positive']:.3f}") + "</tr>")
    vix_tbl = S._table(["Condition when the dip signal fired", "Signals", "Distinct days",
                        "Avg 10-day return vs normal", "Share beating normal"], vix_rows)

    regime = f"""
<section class="regime"><h2>VIX: {S._esc(ctx['label'])}</h2>
<p class="note">Shown for context. The dip grades below are not adjusted for the VIX; the table near the bottom
shows how dips actually performed in each VIX condition over the sample.</p>
<dl class="stats">
 <div><dt>VIX</dt><dd>{ctx['level']:.2f}</dd></div>
 <div><dt>1-year percentile</dt><dd>{ctx['pct'] * 100:.0f}</dd></div>
 <div><dt>5-day change</dt><dd>{ctx['chg5'] * 100:+.0f}%</dd></div>
</dl></section>"""

    banner = ('<div class="demo">Synthetic demo data. These are random price series, not the real tickers. '
              'Every number below is meaningless except to show the layout.</div>' if demo else "")
    body = f"""{S._nav("dip.html")}
{banner}
<h1>Dip-buying research: your holdings and sector peers</h1>
<p class="sub">Data through {asof:%A, %B %d, %Y}. {len(res)} tickers, up to 10 years of daily history.
Analysis only, not investment advice.</p>
{regime}
<section class="block"><h2>Consistent dip-bouncers with a signal right now</h2>
<p class="desc">Stocks that passed all or nearly all five consistency tests and whose best dip rule fired today
or in the last three sessions. This is where a historically reliable bounce pattern is currently active.
Suggested size is half-Kelly (win rate and average win/loss from the backtest, capped at {KELLY_CAP:.0%} of the
account) against the {ACCOUNT_SIZE:,.0f} account size set at the top of the script -- edit it to your own.</p>
{_ticker_table(catch, sizing=True, empty_msg="No consistent dip-bouncer has an active signal today.")}</section>
<section class="block"><h2>Consistent dip-bouncers, armed and within 10% of their trigger</h2>
<p class="desc">Same stocks, not yet triggered. The status column shows how far the price must fall to hit the rule.
The trigger price is approximate because Bollinger Bands and the 60-day high move each day.</p>
{_ticker_table(watch, empty_msg="Nothing consistent is armed within 10% of its trigger.")}</section>
<section class="block"><h2>Your positions</h2>
{overlap}
<p class="desc">SDS is a 2x inverse S&amp;P 500 fund. It loses value to daily reset decay over time, so dip rules
do not apply to it the way they do to ordinary stocks.</p>
{_ticker_table(mine, extras=True)}</section>
<section class="block"><h2>Portfolio exposure by factor / sector group</h2>
<p class="desc">Equal-weighted by ticker count, not dollars -- this script doesn't know your actual position sizes.
Groups reuse the peer-group labels above (e.g. everything in "Mega-cap platforms" counts once each).</p>
{exposure_tbl}</section>
<section class="block"><h2>Sector peers</h2>
<p class="desc">Stocks in the same sectors as your holdings, graded by the same rules.</p>
{peer_blocks}</section>
<section class="block"><h2>Does the VIX change how dips perform?</h2>
<p class="desc">All dip signals across every ticker, grouped by the VIX on the signal day. Signals on the same day
across many stocks are not independent, so read the distinct-days column as the true sample size.</p>
{vix_tbl}</section>
<footer>
<p><b>How to read the grade.</b> A stock is graded on its best of three fixed dip rules at a 10-session hold.
Consistent means all five tests passed (at least {MIN_TRADES} trades, win rate at least {MIN_WIN:.0%}, beats its normal
10-day return with t of {MIN_T:.0f} or more, positive in both the older 60% and recent 40% of history, and positive in
at least {MIN_YEARLY:.0%} of calendar years). Mostly means four of five. "Edge" is the average trade minus the
stock's average 10-day return on any day, and "Edge, recent" is the same measure over the most recent 40% of history.
"Dip after entry" is the average worst point reached within the hold, so it shows how far under your purchase price
the position typically trades before it recovers.</p>
<p><b>False-positive check.</b> The same gate was run on 150 pure random-walk series. {null['any']:.0%} of them
earned Consistent or Mostly on at least one rule. Here {n_ok} of {len(res)} tickers earned it, so roughly
{exp_fp:.0f} of those would be expected by luck alone. Treat the grade as a filter to focus attention, not proof.</p>
<p><b>What this can and cannot detect.</b> In simulations, the gate found a planted bounce pattern with a 3 to 5 day
half-life in roughly 60% to 98% of series, but found almost none when reversion took 10 days or longer. So "no consistent
dip-bouncers" means no fast, reliable bounce pattern, not that no pattern exists.</p>
<p><b>Extra rigor.</b> "Edge, 90% CI" bootstrap-resamples the trades themselves ({MC_DRAWS} draws) to show a range on
the edge, not just the point estimate -- a wide range crossing zero means the win rate is noisier than it looks.
"Walk-forward folds+" splits history into {WF_FOLDS} sequential chunks and counts how many had a positive edge (out
of however many had enough trades to check); there is no re-optimization step because the three rules are fixed by
design. "Edge at 3x cost" reruns the same trades at a {HIGH_COST:.1%} round trip instead of {COST:.1%}, so you can see
whether the edge is a real inefficiency or just barely clearing a thin assumed cost. "Buy & hold CAGR" is what simply
holding the stock returned over the same history, the benchmark every dip rule has to beat.</p>
<p><b>Paper-trading log.</b> Every day this runs live, today's Consistent/Mostly signals are logged to
data/dip_signals_history.csv, and once {PRIMARY_H} sessions have passed for an older signal its actual outcome
(entry at the next open, exit {PRIMARY_H} sessions later) is recorded to data/paper_trades.csv. That file is the
honest check on whether these backtest numbers hold up in real time.</p>
<p><b>Limits.</b> Past bounce behavior can stop working, especially around earnings, rate shocks, and regime changes.
Buying at the next open ignores slippage on fast moves. Short interest, insider activity, put/call ratio and the next
earnings date come from Yahoo Finance and can be missing or stale for some tickers. This ignores taxes. Not investment
advice.</p>
</footer>"""
    return (f'<!doctype html><html lang="en"><head><meta charset="utf-8">'
            f'<meta name="viewport" content="width=device-width,initial-scale=1">'
            f'<title>Dip research, {asof:%Y-%m-%d}</title>'
            f'<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&display=swap" rel="stylesheet">'
            f'<style>{S.CSS}h3{{font-size:16px;margin:22px 0 8px;font-weight:600}}'
            f'.warn{{color:var(--warn);font-weight:600}}</style></head>'
            f'<body><main>{body}</main><script>{S.JS}</script></body></html>')


def reconcile_paper_trades(prices: dict[str, pd.DataFrame], hist_path: Path, out_path: Path,
                           H: int = PRIMARY_H, cost: float = COST) -> None:
    """Live paper-trading log: for every logged signal old enough that its H-session hold has
    actually elapsed, record what really happened (buy the next session's open, sell the close
    H sessions later) so the backtest's expectations can be checked against reality over time.
    Idempotent -- already-reconciled (date, ticker) pairs are skipped."""
    if not hist_path.exists():
        return
    hist = pd.read_csv(hist_path, parse_dates=["date"])
    done = pd.read_csv(out_path, parse_dates=["date"]) if out_path.exists() else pd.DataFrame(columns=["date", "ticker"])
    seen = set(zip(done["date"].astype(str), done["ticker"])) if len(done) else set()
    rows = []
    for _, r in hist.iterrows():
        key = (str(r["date"].date()), r["ticker"])
        if key in seen or r["ticker"] not in prices:
            continue
        d = prices[r["ticker"]]
        after = d.index[d.index > r["date"]]
        if len(after) < H:                # hold period hasn't fully elapsed yet
            continue
        entry_date, exit_date = after[0], after[H - 1]
        o = d.loc[entry_date, "Open"] if "Open" in d else d.loc[entry_date, "Close"]
        c = d.loc[exit_date, "Close"]
        ret = float(c / o - 1) - cost
        rows.append(dict(date=r["date"].date(), ticker=r["ticker"], best_rule=r.get("best_rule", ""),
                         grade=r.get("grade", ""), entry_date=entry_date.date(), exit_date=exit_date.date(),
                         entry_price=round(float(o), 2), exit_price=round(float(c), 2),
                         realized_return=round(ret, 4), win=bool(ret > 0)))
    if rows:
        new = pd.concat([done, pd.DataFrame(rows)], ignore_index=True) if len(done) else pd.DataFrame(rows)
        new.to_csv(out_path, index=False)
        print(f"Paper-trading log: reconciled {len(rows)} signal(s) whose {H}-session hold has elapsed.")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="docs")
    ap.add_argument("--period", default="10y")
    ap.add_argument("--extra", default="", help="comma-separated extra tickers to include")
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--null-series", type=int, default=150, help="random-walk series for the false-positive check")
    ap.add_argument("--skip-extra", action="store_true",
                    help="skip short interest / insider / options / earnings lookups (faster)")
    args = ap.parse_args()

    out = Path(args.out)
    (out / "data").mkdir(parents=True, exist_ok=True)
    extra = [t.strip().upper() for t in args.extra.split(",") if t.strip()]
    group_of = dict(POSITION_GROUP)
    for g, ts in PEER_GROUPS.items():
        for t in ts:
            group_of.setdefault(t, g)
    for t in extra:
        group_of.setdefault(t, "Added by you")
    tickers = list(dict.fromkeys(POSITIONS + [t for ts in PEER_GROUPS.values() for t in ts] + extra))

    if args.demo:
        end = pd.Timestamp.today().normalize()
        prices = {t: synth_series("ou" if i % 4 == 0 else "rw", 100 + i, end=end) for i, t in enumerate(tickers)}
        rng = np.random.default_rng(999)
        lv = np.full(2520, np.log(17.0))
        for j in range(1, 2520):
            lv[j] = lv[j - 1] + 0.05 * (np.log(17.0) - lv[j - 1]) + 0.06 * rng.standard_normal()
        vix_df = pd.DataFrame({"Close": np.exp(lv)}, index=pd.bdate_range(end=end, periods=2520))
        vix3m = None
        spy = prices["SPY"]["Close"].pct_change()
    else:
        print(f"Downloading {len(tickers)} tickers ({args.period})...")
        prices = S.download_prices(tickers, args.period)
        idx = S.download_prices(["^VIX", "^VIX3M"], "3y")
        vix_df, vix3m = idx.get("^VIX"), idx.get("^VIX3M")
        if vix_df is None:
            raise SystemExit("Could not download ^VIX; refusing to publish.")
        if len(prices) < 0.8 * len(tickers):
            raise SystemExit(f"Only {len(prices)}/{len(tickers)} tickers downloaded (Yahoo rate limit?). Try later.")
        # SPY only feeds the beta/correlation columns, so its absence shouldn't abort the run.
        spy = prices["SPY"]["Close"].pct_change() if "SPY" in prices else None
        missing = [t for t in tickers if t not in prices]
        if missing:
            print(f"[warn] no data for: {', '.join(missing)}", file=sys.stderr)

    ctx = S.vix_context(vix_df, vix3m)
    rows, events = [], []
    for t, df in prices.items():
        try:
            r = analyze(t, df, spy)
        except Exception as exc:
            print(f"[warn] {t}: {exc}", file=sys.stderr)
            continue
        if r:
            row, ev = r
            row["group"], row["is_position"] = group_of.get(t, ""), t in POSITIONS
            row["last_date"] = df.index[-1]
            rows.append(row)
            events += ev
    if not rows:
        raise SystemExit("No tickers could be analyzed.")
    res = pd.DataFrame(rows)
    asof = res["last_date"].mode()[0]
    res = res[res["last_date"] >= asof - pd.Timedelta(days=4)].copy()  # drop stale/halted names
    res = res.drop(columns="last_date")

    if not args.demo and not args.skip_extra:
        print(f"Fetching short interest, insider activity, options skew and earnings dates for "
              f"{len(res)} tickers...")
        extras = {}
        for i, t in enumerate(res["ticker"], 1):
            try:
                extras[t] = extra_signals(t)
            except Exception as exc:
                print(f"[warn] extra_signals({t}): {exc}", file=sys.stderr)
            if i % 15 == 0:
                print(f"  {i}/{len(res)}")
        ex_df = pd.DataFrame.from_dict(extras, orient="index").reset_index().rename(columns={"index": "ticker"})
        res = res.merge(ex_df, on="ticker", how="left")
    else:
        for c in ("short_pct_float", "days_to_cover", "insider_net_shares", "put_call"):
            res[c] = np.nan
        res["earnings_date"] = pd.NaT
    res["earnings_date"] = pd.to_datetime(res["earnings_date"])
    today = pd.Timestamp.today().normalize()
    res["earnings_in_hold"] = res["earnings_date"].apply(
        lambda dt: bool(pd.notna(dt) and 0 <= (dt - today).days <= PRIMARY_H * 1.4))

    ev_df = pd.DataFrame(events)
    vixtab = vix_study(ev_df[~ev_df["ticker"].isin(LEVERAGED_INVERSE)] if len(ev_df) else ev_df,
                       vix_df["Close"])

    print(f"Running false-positive calibration on {args.null_series} random walks...")
    null = null_pass_rate(args.null_series)

    # Overlap among the user's own holdings
    pos_close = pd.DataFrame({t: prices[t]["Close"] for t in POSITIONS if t in prices}).pct_change().iloc[-252:]
    cm = pos_close.corr()
    pairs = [(a, b, float(cm.loc[a, b])) for i, a in enumerate(cm.columns) for b in cm.columns[i + 1:]
             if cm.loc[a, b] >= 0.90]
    pairs.sort(key=lambda z: -z[2])

    res.sort_values(["is_position", "score"], ascending=[False, False]).to_csv(
        out / "data" / "dip_results.csv", index=False, float_format="%.4f")
    (out / "dip.html").write_text(render(res, ctx, vixtab, null, pairs[:8], asof, args.demo), encoding="utf-8")

    if not args.demo:
        cur = res[res["grade"].isin(["Consistent", "Mostly"]) &
                  res["status"].isin(["Signal today", "Signal in last 3 sessions"])]
        log = cur.assign(date=str(asof.date()), vix=round(ctx["level"], 2), regime=ctx["label"])[
            ["date", "ticker", "best_rule", "grade", "close", "vix", "regime"]]
        hp = out / "data" / "dip_signals_history.csv"
        if hp.exists():
            old = pd.read_csv(hp)
            log = pd.concat([old[old["date"] != str(asof.date())], log], ignore_index=True)
        log.to_csv(hp, index=False)
        reconcile_paper_trades(prices, hp, out / "data" / "paper_trades.csv")

    n_ok = int(res["grade"].isin(["Consistent", "Mostly"]).sum())
    print(f"\n{len(res)} tickers analyzed through {asof.date()}. {n_ok} graded Consistent/Mostly "
          f"(random-walk false-positive rate {null['any']:.0%}).")
    print(f"Report: {out / 'dip.html'}")
    for _, r in res[res["is_position"]].iterrows():
        print(f"  {r['ticker']:5s} {r['grade']:12s} {r['best_rule']:16s} {r['status']}")


if __name__ == "__main__":
    main()
