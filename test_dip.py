"""Run with:  python test_dip.py"""
from pathlib import Path

import numpy as np
import pandas as pd

import dip_backtest as D


def test_entry_is_next_open_and_trades_do_not_overlap():
    n, H = 60, 10
    o = np.arange(100.0, 100.0 + n)          # open rises 1/day
    c = o + 0.5
    l = o - 0.2
    sig = np.zeros(n, bool)
    sig[[5, 6, 7, 30]] = True                # 6 and 7 fall inside the position opened after 5
    idx = D.pick_trades(sig, H)
    assert list(idx) == [6, 31]              # enter the session AFTER the signal
    base, mae = D.forward_arrays(o, l, c, H)
    assert abs(base[6] - (c[6 + H - 1] / o[6] - 1)) < 1e-12   # buy open of e, sell close of e+H-1
    assert mae[6] < 0                        # lowest low sits under the entry open


def test_trade_at_end_of_data_is_dropped():
    sig = np.zeros(30, bool)
    sig[27] = True                           # entry would be session 28, exit past the end
    assert len(D.pick_trades(sig, 10)) == 0


def test_grade_gate():
    good = dict(n=30, win=0.7, edge=0.02, t=3.0, edge_is=0.01, edge_oos=0.03, n_oos=12, yearly=0.8)
    assert D.grade(good) == (5, "Consistent")
    assert D.grade({**good, "edge_oos": -0.01})[1] != "Consistent"    # fails out-of-sample
    assert D.grade({**good, "n": 8})[1] == "No"                       # too few trades
    assert D.grade({**good, "yearly": 0.3})[1] == "Mostly"            # 4 of 5, core tests intact
    assert D.grade({**good, "win": 0.5, "yearly": 0.3})[1] == "No"    # 3 of 5


def test_random_walks_rarely_pass_and_planted_bouncers_usually_do():
    null = D.null_pass_rate(50, seed=3)
    assert null["any"] <= 0.12, null                                   # false-positive control
    hits = 0
    for i in range(20):
        _, _, res = D.evaluate(D.synth_series("ou", 40 + i), horizons=(D.PRIMARY_H,))
        hits += D.passes(res)
    assert hits >= 10, hits                                            # detects a real fast bounce


def test_random_walk_engine_has_no_hidden_bias():
    edges = []
    for i in range(40):
        _, _, res = D.evaluate(D.synth_series("rw", 300 + i), horizons=(D.PRIMARY_H,))
        st = res[("50-day pullback", D.PRIMARY_H)][0]
        if st["n"] >= 10:
            edges.append(st["edge"])
    assert abs(np.mean(edges)) < 0.006, np.mean(edges)                 # only ~the 0.1% cost


def test_summarize_rigor_fields_on_a_real_edge():
    """Backtest-rigor fields (walk-forward folds, bootstrap CI, cost sensitivity) on a series
    with a real planted edge: the CI shouldn't straddle zero, and most folds should agree."""
    _, _, res = D.evaluate(D.synth_series("ou", 7), horizons=(D.PRIMARY_H,))
    st = res[("50-day pullback", D.PRIMARY_H)][0]
    assert st["n"] >= 15, st["n"]
    assert st["wf_folds"] >= 3
    assert st["wf_positive_frac"] >= 0.5
    assert st["edge_lo"] < st["edge"] < st["edge_hi"]                  # point estimate inside its own CI
    assert st["edge_hi"] > 0                                            # a real edge: CI's upper bound is positive
    assert st["edge_hc"] < st["edge"]                                   # higher cost can only shrink the edge


def test_insider_sign_reads_sale_and_purchase_not_grants():
    text = pd.Series(["Sale at price 10 - 12 per share.", "Purchase at price 9 per share.",
                      "Stock Award(Grant) at price 0.00 per share.", "Stock Gift at price 0.00 per share."])
    assert list(D._insider_sign(text)) == [-1, 1, 0, 0]


def test_kelly_fraction_is_zero_without_an_edge_and_positive_with_one():
    assert D.kelly_fraction(0.5, 0.02, 0.02) == 0                       # 50/50 even-money: no edge, no size
    assert 0 < D.kelly_fraction(0.65, 0.03, 0.02) <= D.KELLY_CAP        # real edge -> capped positive size
    assert np.isnan(D.kelly_fraction(0.6, np.nan, 0.02))                # missing inputs -> NaN, not a crash


def test_buy_hold_matches_simple_total_return():
    idx = pd.bdate_range("2020-01-01", periods=756)                     # ~3 years
    close = pd.Series(np.linspace(100, 150, 756), index=idx)
    bh = D.buy_hold_stats(close)
    assert abs(bh["bh_total"] - 0.5) < 1e-9
    assert bh["bh_cagr"] > 0


def test_portfolio_exposure_sums_to_100_percent():
    res = pd.DataFrame({"ticker": ["A", "B", "C", "D"], "is_position": [True, True, True, False],
                        "group": ["G1", "G1", "G2", "G3"]})
    exp = D.portfolio_exposure(res)
    assert abs(exp["weight"].sum() - 100) < 1e-9
    assert set(exp["group"]) == {"G1", "G2"}                            # D is a peer, not a position


def test_reconcile_paper_trades_waits_for_the_hold_to_elapse(tmp_path=None):
    tmp_path = tmp_path or Path(__import__("tempfile").mkdtemp())
    idx = pd.bdate_range("2026-01-01", periods=40)
    prices = {"XYZ": pd.DataFrame({"Open": np.arange(100.0, 140.0), "Close": np.arange(100.5, 140.5)}, index=idx)}
    # idx[5] has 34 sessions after it (>= H=10: reconciles); idx[35] has only 4 (< H: waits)
    hist = pd.DataFrame({"date": [idx[5], idx[35]], "ticker": ["XYZ", "XYZ"],
                         "best_rule": ["BB dip", "BB dip"], "grade": ["Consistent", "Consistent"]})
    hist_path, out_path = tmp_path / "hist.csv", tmp_path / "paper.csv"
    hist.to_csv(hist_path, index=False)
    D.reconcile_paper_trades(prices, hist_path, out_path, H=10)
    assert out_path.exists()
    done = pd.read_csv(out_path)
    assert len(done) == 1 and done["ticker"].iloc[0] == "XYZ"           # only the earlier signal's hold has elapsed
    assert done["win"].iloc[0] == (done["realized_return"].iloc[0] > 0)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
