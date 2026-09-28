"""Run with:  python test_validate.py   -- proves validate.py's checks actually catch broken numbers."""
import numpy as np
import pandas as pd

import screener as S
import validate as V

VIX = dict(key="normal", pct=0.5)
CONV = dict(score=50.0, label="Neutral")


def _screen(**over):
    row = dict(ticker="AAA", long_score=40.0, fade_score=10.0, support_score=30.0, resist_score=0.0, rsi=45.0,
               trend="Up", weekly_trend="Up", support_level="SMA50", support_rate=0.8, support_base=0.6, support_n=12,
               resist_level="", resist_rate=np.nan, resist_base=np.nan, resist_n=0)
    return pd.DataFrame([row | over])


def _dip(**over):
    row = dict(ticker="AAA", grade="Consistent", score=5, status="Armed", trades=30, win=0.7, win_lo=0.6, win_hi=0.8,
               edge=0.01, edge_lo=0.005, edge_hi=0.015, kelly_frac=0.1)
    return pd.DataFrame([row | over])


def test_clean_values_pass():
    assert V.check_calculations(VIX, _screen(), _dip(), CONV) == []


def test_broken_calculations_are_caught():
    cases = [
        (_screen(long_score=150.0), _dip()),                       # score out of range
        (_screen(long_score=np.inf), _dip()),                      # inf
        (_screen(support_rate=0.62), _dip()),                      # level shown but not beating its baseline
        (_screen(support_rate=1.3), _dip()),                       # probability > 1
        (_screen(), _dip(score=3)),                                # "Consistent" with 3/5 criteria
        (_screen(), _dip(win=0.95)),                               # estimate outside its own bootstrap CI
        (_screen(), _dip(kelly_frac=0.9)),                         # sizing above the hard cap
        (_screen(), _dip(win=np.nan)),                             # trades but no win rate
    ]
    for scr, dip in cases:
        assert V.check_calculations(VIX, scr, dip, CONV), (scr.iloc[0].to_dict(), dip.iloc[0].to_dict())
    assert V.check_calculations(VIX, _screen(), _dip(), dict(score=50.0, label="Bullish"))   # wrong label


def test_regime_note_contradicting_its_multiplier_is_caught():
    old = S.REGIMES["stress"]["long"]
    try:
        S.REGIMES["stress"]["long"] = 0.75                         # note still says "cut 40%"
        g = dict(detect={hl: 1.0 for hl in (3, 4, 5)} | {10: 0.0, 15: 0.0}, null_any=0.0, rw_false_respect=0.0)
        failed = [c for c, _, ok in V.check_guidance(g) if not ok]
        assert failed == ["'Stress' note matches its multipliers"], failed
    finally:
        S.REGIMES["stress"]["long"] = old


def test_diff_ignores_float_noise_but_not_real_changes():
    old = {"dip": {"MU": {"grade": "No", "win": 0.61}}}
    assert V.diff(old, {"dip": {"MU": {"grade": "No", "win": 0.61 + 1e-12}}}) == []
    assert V.diff(old, {"dip": {"MU": {"grade": "Mostly", "win": 0.64}}}) == [
        ("dip.MU.grade", "No", "Mostly"), ("dip.MU.win", 0.61, 0.64)]


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
