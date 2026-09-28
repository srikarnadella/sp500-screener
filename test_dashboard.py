"""Run with:  python test_dashboard.py"""
import datetime as dt

import pandas as pd

import dashboard as B


def test_next_session_skips_weekend():
    assert B.next_session(dt.date(2026, 10, 2)) == dt.date(2026, 10, 5)   # Fri -> Mon
    assert B.next_session(dt.date(2026, 9, 29)) == dt.date(2026, 9, 30)


def test_alerts_only_for_what_needs_attention_next_session():
    dip = pd.DataFrame([
        dict(ticker="MU", grade="Consistent", best_rule="BB dip", status="Signal today", win=0.7),
        dict(ticker="QQQ", grade="Consistent", best_rule="BB dip", status="Signal in last 3 sessions", win=0.7),
        dict(ticker="XOM", grade="Weak", best_rule="BB dip", status="Signal today", win=0.4)])
    ev = lambda d, name: dict(date=d, time="08:30", event=name, category="", consensus="", previous="")
    cal = dict(econ=[ev("2026-10-02", "Nonfarm Payrolls"), ev("2026-10-02", "Initial Jobless Claims"),
                     ev("2026-10-05", "CPI")],
               fomc=[dict(ev("2026-10-02", "FOMC rate decision"), time="14:00")],
               earnings=[dict(date="2026-10-02", ticker="NVDA", name="NVIDIA", cap=4e12, time="After close", eps="-"),
                         dict(date="2026-10-02", ticker="JPM", name="JPMorgan", cap=9e11, time="-", eps="-")])
    items = B.alerts(dip, cal, dt.date(2026, 10, 1))
    text = "\n".join(items)
    assert "**MU** dip signal today" in text and "QQQ" not in text and "XOM" not in text
    assert "**NVDA** (you hold it) reports earnings Fri Oct 2, after close" in text and "JPM" not in text
    assert "Nonfarm Payrolls" in text and "FOMC rate decision" in text
    assert "Jobless" not in text and "CPI" not in text          # not top-tier / not next session
    assert B.alerts(dip.iloc[1:], None, dt.date(2026, 10, 1)) == []


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
