"""Run with:  python test_dip.py"""
import numpy as np

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


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
