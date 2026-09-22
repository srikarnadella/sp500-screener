"""Run with:  python test_ml_respect.py

The scikit-learn dependency is skipped gracefully if it isn't installed (these tests are the
only thing in the repo that needs it)."""
import sys

import numpy as np
import pandas as pd

import screener as s

try:
    import sklearn  # noqa: F401
    import ml_respect as m
except ImportError:
    print("scikit-learn not installed -- skipping ml_respect tests")
    sys.exit(0)


def test_events_for_stock_matches_level_respect_counts():
    """build_events() must count held/broke exactly the way level_respect() does for every
    level/role, on the bars where a trend-context feature (sma200/sma50) is actually available --
    _events_for_stock() deliberately excludes earlier bars (see its comment), reproduced here by
    masking the level to NaN there too, which suppresses the test the same way at the test-
    condition stage rather than emulating the post-hoc skip (that would leave a discrepancy: a
    cooldown-suppressing "test" that's later discarded shouldn't also cost a nearby real one)."""
    _, prices, _, _ = s.demo_data(n=3, seed=11)
    for ticker, df in prices.items():
        d = s.add_indicators(df)
        close, high, low = d["Close"].to_numpy(float), d["High"].to_numpy(float), d["Low"].to_numpy(float)
        atr = d["atr"].to_numpy(float)
        has_trend = np.isfinite(d["sma200"].to_numpy(float)) & np.isfinite(d["sma50"].to_numpy(float))
        events = pd.DataFrame(m._events_for_stock(ticker, df))
        for name, (col, fam, roles) in s.LEVELS.items():
            lvl = np.where(has_trend, d[col].to_numpy(float), np.nan)
            for role in roles:
                held_direct, broke_direct = s.level_respect(high, low, close, lvl, atr, role)
                sub = events[(events["level"] == name) & (events["role"] == role)] if len(events) else events
                held_ev = int((sub["label"] == 1).sum()) if len(sub) else 0
                broke_ev = int((sub["label"] == 0).sum()) if len(sub) else 0
                assert (held_ev, broke_ev) == (held_direct, broke_direct), \
                    f"{ticker} {name}/{role}: events gave {(held_ev, broke_ev)}, level_respect gave {(held_direct, broke_direct)}"


def test_no_false_skill_on_shuffled_labels():
    """The decisive null check: destroy any real relationship between features and outcome by
    shuffling labels, then confirm the model collapses to (not beats) the constant baseline. If
    this ever fails, something in the pipeline is leaking the true label into the features."""
    _, prices, _, _ = s.demo_data()
    events = m.build_events(prices).dropna(subset=m.FEATURES + ["label"]).sort_values("date")
    rng = np.random.default_rng(0)
    shuffled = events.copy()
    shuffled["label"] = rng.permutation(shuffled["label"].to_numpy())

    for evaluator in (m.evaluate, m.evaluate_by_ticker):
        report, _ = evaluator(prices, shuffled)
        assert report["ok"], report
        rel = (report["constant_brier"] - report["model_brier"]) / report["constant_brier"]
        assert abs(rel) < 0.01, f"{evaluator.__name__}: model 'beat' shuffled labels by {rel:+.1%}"


def test_detects_real_structure_on_unshuffled_data():
    """Sanity check the other direction: on real (unshuffled) data, the model should show a real,
    non-trivial improvement over the constant baseline -- confirms the pipeline isn't just inert."""
    _, prices, _, _ = s.demo_data()
    events = m.build_events(prices)
    report, _ = m.evaluate(prices, events)
    assert report["ok"], report
    rel = (report["constant_brier"] - report["model_brier"]) / report["constant_brier"]
    assert rel > 0.03, f"model only improved on the constant baseline by {rel:+.1%}"


def test_verdict_flags_a_heuristic_weaker_than_doing_nothing():
    weak_heuristic = dict(ok=True, constant_brier=0.20, heuristic_brier=0.21, model_brier=0.15)
    assert "caution" in m.verdict(weak_heuristic)


def test_verdict_flags_no_real_skill():
    no_skill = dict(ok=True, constant_brier=0.20, heuristic_brier=0.19, model_brier=0.199)
    assert "no real skill" in m.verdict(no_skill)


def test_verdict_beats_when_model_clears_both_bars():
    clear_win = dict(ok=True, constant_brier=0.20, heuristic_brier=0.19, model_brier=0.16)
    assert m.verdict(clear_win) == "beats"


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
