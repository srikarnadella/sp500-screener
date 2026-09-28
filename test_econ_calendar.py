"""Run with:  python test_econ_calendar.py   (or, if you have pytest installed: python -m pytest -q)"""
import datetime as dt

import econ_calendar as c

FOMC_PAGE = """
<h4><a id="1">2027 FOMC Meetings</a></h4>
<div class="fomc-meeting__month col-xs-5"><strong>January</strong></div>
<div class="fomc-meeting__date col-xs-4">26-27</div>
<div class="fomc-meeting--shaded fomc-meeting__month col-xs-5"><strong>April/May</strong></div>
<div class="fomc-meeting__date col-xs-4">30-1*</div>
<h4><a id="2">2025 FOMC Meetings</a></h4>
<div class="fomc-meeting__month col-xs-5"><strong>Aug</strong></div>
<div class="fomc-meeting__date col-xs-4">22 (notation vote)</div>
<div class="fomc-meeting__month col-xs-5"><strong>Sept</strong></div>
<div class="fomc-meeting__date col-xs-4">16-17*</div>
"""


def test_parse_fomc():
    got = [(e["date"], e["event"]) for e in c.parse_fomc(FOMC_PAGE)]
    assert got == [("2027-01-27", "FOMC rate decision"),
                   ("2027-05-01", "FOMC rate decision + projections"),   # cross-month meeting
                   ("2025-09-17", "FOMC rate decision + projections")]   # notation vote skipped


def test_filter_econ_keeps_key_us_releases_once():
    rows = [dict(country="United States", eventName="CPI", gmt="08:30", consensus="0.3%", previous="0.2%"),
            dict(country="United States", eventName="CPI", gmt="08:30", consensus="3.1%", previous="3.0%"),
            dict(country="United States", eventName="CPI Index, n.s.a.", gmt="08:30", consensus="", previous=""),
            dict(country="United States", eventName="Fed Waller Speaks", gmt="10:00", consensus="", previous=""),
            dict(country="Canada", eventName="Unemployment Rate", gmt="08:30", consensus="", previous=""),
            dict(country="United States", eventName="Nonfarm Payrolls", gmt="08:30", consensus="&nbsp;", previous="98K")]
    got = c.filter_econ(rows, "2026-10-14")
    assert [(e["event"], e["category"], e["consensus"]) for e in got] == [
        ("CPI", "Inflation", "0.3%"), ("Nonfarm Payrolls", "Jobs", "")]


def test_filter_earnings_major_sp500_only():
    rows = [dict(symbol="JPM", marketCap="$900,000,000,000", time="time-pre-market", epsForecast="$5.84"),
            dict(symbol="MKC", marketCap="$12,000,000,000", time="time-pre-market", epsForecast="$0.75"),
            dict(symbol="TSM", marketCap="$1,000,000,000,000", time="time-pre-market", epsForecast="$2"),
            dict(symbol="GS", marketCap="N/A", time="time-after-hours", epsForecast=""),
            dict(symbol="BRK/B", marketCap="$1,000,000,000,000", time="time-not-supplied", epsForecast="")]
    got = c.filter_earnings(rows, "2026-10-13", {"JPM": "JPMorgan Chase", "MKC": "McCormick", "GS": "Goldman",
                                                 "BRK-B": "Berkshire Hathaway"})
    assert [(e["ticker"], e["time"]) for e in got] == [("JPM", "Before open"), ("BRK-B", "-")]


def test_render_drops_past_events():
    data = dict(econ=[dict(date="2026-09-01", time="08:30", event="Old CPI", category="Inflation",
                           consensus="", previous="")],
                fomc=c.parse_fomc(FOMC_PAGE), earnings=[])
    page = c.render(data, dt.date(2026, 10, 1))
    # fixture lists 2027 before 2025; only 2027 is upcoming, and it must be sorted by date
    assert "Old CPI" not in page and "Jan 27" in page and "May 1" in page and 'aria-current="page">Calendar' in page


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
