"""Run with:  python test_screener.py   (or, if you have pytest installed: python -m pytest -q)"""
import numpy as np
import pandas as pd

import screener as s


def _series(pattern, n_cycles):
    """Build OHLC around a flat level at 100 from a repeating close pattern."""
    close = np.array(pattern * n_cycles, float)
    high, low = close + 0.5, close - 0.5
    return high, low, close, np.full_like(close, 100.0), np.full_like(close, 2.0)


def test_support_that_always_holds():
    # drift down to the level, touch it, bounce >= 1 ATR (2 pts) above, repeat
    pattern = [106, 104, 102, 100.5, 103, 106, 108, 108]
    h, l, c, lvl, atr = _series(pattern, 12)
    held, broke = s.level_respect(h, l, c, lvl, atr, "support")
    assert held >= 8 and broke == 0


def test_support_that_always_breaks():
    # touches the level, then closes >= 1 ATR beneath it
    pattern = [106, 104, 102, 100.5, 97, 94, 94, 106]
    h, l, c, lvl, atr = _series(pattern, 12)
    held, broke = s.level_respect(h, l, c, lvl, atr, "support")
    assert broke >= 8 and held == 0


def test_resistance_that_always_rejects():
    pattern = [94, 96, 98, 99.5, 97, 94, 92, 92]
    h, l, c, lvl, atr = _series(pattern, 12)
    held, broke = s.level_respect(h, l, c, lvl, atr, "resistance")
    assert held >= 8 and broke == 0


def test_shrinkage_punishes_small_samples():
    assert s.shrunk_rate(3, 0) < s.shrunk_rate(15, 0) < 1.0
    assert abs(s.shrunk_rate(0, 0) - 0.5) < 1e-9


def _vix(values):
    idx = pd.bdate_range(end="2026-01-30", periods=len(values))
    return pd.DataFrame({"Close": values}, index=idx)


def test_vix_regimes():
    calm = _vix(list(np.linspace(14, 13, 260)))
    assert s.vix_context(calm, None)["key"] == "complacent"
    spike = _vix([18] * 250 + [22, 26, 32, 34, 33, 34, 35, 36, 36, 37])
    assert s.vix_context(spike, None)["key"] == "stress"
    unwind = _vix([18] * 250 + [22, 26, 32, 34, 33, 31, 29, 28, 27, 26])
    assert s.vix_context(unwind, None)["key"] == "rolling_over"
    normal = _vix([19] * 260)
    assert s.vix_context(normal, None)["key"] == "normal"


def _synthetic(reflect, seed):
    """Random walk; if `reflect`, price bounces off its own 50-day 85% of the time."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(end="2026-09-18", periods=1300)
    sig, lp, win = 0.014, [np.log(100.0)], [100.0]
    for _ in range(1, 1300):
        sma = np.mean(win[-50:])
        new = lp[-1] + 0.0004 + sig * rng.standard_normal()
        if reflect and lp[-1] > np.log(sma) + 0.004 and new < np.log(sma) + 0.5 * sig \
                and rng.random() < 0.85:
            new = np.log(sma) + sig * (1.2 + abs(rng.standard_normal()))
        lp.append(new)
        win.append(np.exp(new))
    c = np.exp(lp)
    sp = c * sig * rng.uniform(0.4, 0.9, 1300)
    return pd.DataFrame({"High": c + sp * rng.uniform(0.2, 1, 1300),
                         "Low": c - sp * rng.uniform(0.2, 1, 1300),
                         "Close": c, "Volume": 1e6}, index=dates)


def test_detects_planted_level_but_not_noise():
    """Positive and negative control for the whole respect + placebo-baseline pipeline."""
    planted = [s.analyze_stock("P", _synthetic(True, i)) for i in range(6)]
    noise = [s.analyze_stock("N", _synthetic(False, 100 + i)) for i in range(6)]
    ex = lambda rows: np.mean([r["sma50_sup_rate"] - r["sma50_sup_base"] for r in rows])
    assert ex(planted) > 0.15, ex(planted)   # real bounce level is clearly detected
    assert abs(ex(noise)) < 0.06, ex(noise)  # random walks show ~no excess


def test_rel_strength_beats_or_lags_spy():
    idx = pd.bdate_range("2026-01-01", periods=80)
    spy = pd.Series(np.linspace(100, 110, 80), index=idx)         # SPY up 10%
    winner = pd.Series(np.linspace(100, 130, 80), index=idx)      # up 30%: beats SPY
    loser = pd.Series(np.linspace(100, 102, 80), index=idx)       # up 2%: lags SPY
    assert s.rel_strength(winner, spy, 60) > 0.10
    assert s.rel_strength(loser, spy, 60) < -0.03
    assert np.isnan(s.rel_strength(winner, None, 60))             # no SPY series -> NaN, not a crash


def test_breadth_series_counts_advancers_and_flags_thrust():
    idx = pd.bdate_range("2026-01-01", periods=60)
    up = pd.DataFrame({"Close": np.linspace(100, 120, 60)}, index=idx)     # rises every day
    down = pd.DataFrame({"Close": np.linspace(100, 80, 60)}, index=idx)    # falls every day
    b = s.breadth_series({"UP": up, "DOWN": down})
    assert b["adv"] == 1 and b["decl"] == 1                        # one advancer, one decliner, every day
    assert b["mcclellan"] == 0                                     # net advances constant at 0 -> oscillator flat
    assert b["zweig_thrust"] is False                              # ratio never dips under 40% first


def test_conviction_score_ranks_strong_market_above_weak():
    breadth_strong = dict(above50=90, above200=90, zweig_thrust=True)
    breadth_weak = dict(above50=10, above200=10, zweig_thrust=False)
    strong = s.conviction_score(breadth_strong, {"key": "rolling_over"}, pd.DataFrame({"long_score": [80, 70, 60]}))
    weak = s.conviction_score(breadth_weak, {"key": "stress"}, pd.DataFrame({"long_score": [10, 20, 5]}))
    assert strong["score"] > weak["score"]
    assert strong["label"] in ("Constructive", "Bullish")
    assert weak["label"] in ("Bearish", "Cautious")


def test_parses_yfinance_shaped_frames():
    idx = pd.bdate_range("2026-01-01", periods=5, tz="America/New_York")
    def ohlcv(base):
        return pd.DataFrame({"Open": base, "High": base + 1, "Low": base - 1, "Close": base,
                             "Volume": 1000.0}, index=idx)
    # multi-ticker, group_by="ticker": columns are (Ticker, Field); one ticker has no data at all
    multi = pd.concat({"AAA": ohlcv(10.0), "BBB": ohlcv(20.0), "ZZZ": ohlcv(np.nan)}, axis=1)
    out = s._split_download(multi, ["AAA", "BBB", "ZZZ", "MISSING"])
    assert set(out) == {"AAA", "BBB"}
    assert out["AAA"].index.tz is None                      # tz stripped
    # single ticker comes back as a plain frame
    assert set(s._split_download(ohlcv(5.0), ["ONLY"])) == {"ONLY"}
    # empty download
    assert s._split_download(pd.DataFrame(), ["AAA"]) == {}


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
