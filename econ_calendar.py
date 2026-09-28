#!/usr/bin/env python3
"""
Upcoming market-moving events: key US economic releases (jobs, inflation, growth,
sentiment), FOMC rate decisions, and earnings from the largest S&P 500 companies.

Sources (no API keys):
  api.nasdaq.com economic calendar   US releases, ~2 weeks out, with consensus/previous
  federalreserve.gov FOMC calendar   scheduled meetings, a year+ out
  api.nasdaq.com earnings calendar   earnings dates, ~1 month out

Each source falls back to the last good copy in data/calendar.json if a fetch fails.

  python econ_calendar.py     writes docs/calendar.html
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import sys
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

import screener as S

UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/128 Safari/537.36"}
NASDAQ = "https://api.nasdaq.com/api/calendar/{kind}?date={date}"
FOMC_URL = "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm"
ECON_DAYS, EARN_DAYS, FOMC_AHEAD = 14, 21, 3
MAJOR_CAP = 100e9   # the page lists S&P 500 members worth $100B+; the cache keeps all of them
                    # so screener.py can flag any pick that reports soon

# First match wins. Anything that matches none of these (Fed speeches, bill auctions, oil
# inventories, regional surveys...) is dropped as noise.
CATEGORIES = [
    ("Jobs", r"^(Nonfarm Payrolls|Unemployment Rate|Average Hourly Earnings|JOLTS Job Openings"
             r"|Initial Jobless Claims|ADP Nonfarm Employment Change)$"),
    ("Inflation", r"^(Core )?(CPI|PPI)$|^Core PCE Price Index$"),
    ("Fed", r"^FOMC Minutes$"),   # rate decisions come from the Fed's own calendar (parse_fomc)
    ("Growth", r"^(GDP|Retail Sales|Core Retail Sales|ISM Manufacturing PMI|ISM Non-Manufacturing PMI)$"),
    ("Sentiment", r"^(Michigan Consumer Sentiment|CB Consumer Confidence)$"),
]


def categorize(name: str) -> str | None:
    name = name.strip()
    return next((cat for cat, pat in CATEGORIES if re.search(pat, name)), None)


def filter_econ(rows: list[dict], date: str) -> list[dict]:
    """Keep only the US releases worth knowing about, one row per event name. Nasdaq lists MoM
    before YoY under the same name (e.g. two "CPI" rows), so keeping the first keeps MoM."""
    out, seen = [], set()
    for r in rows:
        cat = categorize(r.get("eventName", ""))
        if r.get("country") != "United States" or not cat or r["eventName"] in seen:
            continue
        seen.add(r["eventName"])
        clean = lambda v: re.sub(r"&nbsp;|\s+", " ", str(v or "")).strip()
        out.append(dict(date=date, time=clean(r.get("gmt")), event=r["eventName"].strip(), category=cat,
                        consensus=clean(r.get("consensus")), previous=clean(r.get("previous"))))
    return out


def parse_fomc(page: str) -> list[dict]:
    """Scheduled meetings from the Fed's calendar page. The rate decision comes out at 2pm ET
    on the last day; '*' marks meetings that also publish the Summary of Economic Projections.
    Unscheduled entries ('22 (notation vote)') don't match the day-range pattern and are skipped."""
    out = []
    chunks = re.split(r"(\d{4}) FOMC Meetings", page)
    for year, chunk in zip(chunks[1::2], chunks[2::2]):
        months = re.findall(r"fomc-meeting__month[^>]*><strong>([^<]+)</strong>", chunk)
        days = re.findall(r"fomc-meeting__date[^>]*>([^<]+)<", chunk)
        for month, day in zip(months, days):
            m = re.fullmatch(r"\s*\d+-(\d+)(\*?)\s*", day)
            if not m:
                continue
            last_month = month.split("/")[-1].strip()[:3]   # "April/May" 30-1 -> May; old years use "Feb"
            d = dt.datetime.strptime(f"{last_month} {m.group(1)} {year}", "%b %d %Y").date()
            out.append(dict(date=d.isoformat(), time="14:00",
                            event="FOMC rate decision" + (" + projections" if m.group(2) else ""),
                            category="Fed", consensus="", previous=""))
    return out


def filter_earnings(rows: list[dict], date: str, members: dict[str, str]) -> list[dict]:
    by_sym = {t.replace("-", "."): t for t in members}   # BRK-B in the constituents file, BRK.B / BRK/B at Nasdaq
    out = []
    for r in rows:
        sym = by_sym.get(str(r.get("symbol", "")).replace("/", ".").replace("-", "."))
        try:
            cap = float(re.sub(r"[$,]", "", r.get("marketCap") or ""))
        except ValueError:
            continue
        if sym:
            out.append(dict(date=date, ticker=sym, name=members[sym], cap=cap,
                            time={"time-pre-market": "Before open", "time-after-hours": "After close"}
                                 .get(r.get("time"), "-"),
                            eps=(r.get("epsForecast") or "").strip() or "-"))
    return out


def _nasdaq(kind: str, date: dt.date) -> list[dict]:
    import requests

    r = requests.get(NASDAQ.format(kind=kind, date=date), headers=UA, timeout=20)
    r.raise_for_status()
    return (r.json().get("data") or {}).get("rows") or []   # data is null on weekends


def fetch(today: dt.date, members: dict[str, str], cached: dict) -> dict:
    import requests

    def attempt(key, fn):
        try:
            got = fn()
            if not got:   # weeks of real calendar are never empty; Nasdaq answers blocks with data: null
                raise ValueError("fetched nothing")
            return got
        except Exception as exc:
            print(f"[warn] {key}: {exc}; using cached copy", file=sys.stderr)
            return cached.get(key, [])

    days = lambda n: [today + dt.timedelta(i) for i in range(n)]
    econ = attempt("econ", lambda: [e for d in days(ECON_DAYS)
                                    for e in filter_econ(_nasdaq("economicevents", d), d.isoformat())])
    fomc = attempt("fomc", lambda: parse_fomc(requests.get(FOMC_URL, headers=UA, timeout=20).text))
    earnings = attempt("earnings", lambda: [e for d in days(EARN_DAYS)
                                            for e in filter_earnings(_nasdaq("earnings", d), d.isoformat(), members)])
    return dict(econ=econ, fomc=fomc, earnings=earnings)


def _when(date: str, time: str = "") -> str:
    d = dt.date.fromisoformat(date)
    return f"<b>{d:%a %b} {d.day}</b>" + (f"<small>{S._esc(time)} ET</small>" if time else "")


def render(data: dict, today: dt.date) -> str:
    t = today.isoformat()
    fomc = sorted((e for e in data["fomc"] if e["date"] >= t), key=lambda e: e["date"])[:FOMC_AHEAD]
    econ = pd.DataFrame([e for e in data["econ"] if e["date"] >= t] + fomc,
                        columns=["date", "time", "event", "category", "consensus", "previous"])
    econ = econ.sort_values(["date", "time"], kind="stable")
    earn = pd.DataFrame([e for e in data["earnings"] if e["date"] >= t and e["cap"] >= MAJOR_CAP],
                        columns=["date", "ticker", "name", "cap", "time", "eps"])
    earn = earn.sort_values(["date", "cap"], ascending=[True, False], kind="stable")

    econ_tbl = S._table(["When", "Event", "Category", "Consensus", "Previous"], S._rows(econ, [
        lambda r: S._td(_when(r["date"], r["time"]), sort=f'{r["date"]} {r["time"]}'),
        lambda r: S._td(S._esc(r["event"]), cls="l"),
        lambda r: S._td(S._esc(r["category"]), cls="l"),
        lambda r: S._td(S._esc(r["consensus"] or "-")),
        lambda r: S._td(S._esc(r["previous"] or "-")),
    ]))
    earn_tbl = S._table(["When", "Stock", "Timing", "Market cap", "EPS estimate"], S._rows(earn, [
        lambda r: S._td(_when(r["date"]), sort=r["date"]),
        S._stock_cell,
        lambda r: S._td(S._esc(r["time"]), cls="l"),
        lambda r: S._td(f'${r["cap"] / 1e9:,.0f}B', sort=f'{r["cap"]:.0f}'),
        lambda r: S._td(S._esc(r["eps"])),
    ]))

    body = f"""
{S._nav("calendar.html")}
<h1>Market calendar</h1>
<p class="sub" data-asof="{t}">Updated {today:%a %b} {today.day}, {today.year}. Times are US Eastern.</p>
<section class="block">
  <h2>Economic releases and Fed decisions</h2>
  <p class="desc">Jobs, inflation, growth and sentiment releases for the next {ECON_DAYS} days, plus the next
  {FOMC_AHEAD} scheduled FOMC rate decisions.</p>
  {econ_tbl}
</section>
<section class="block">
  <h2>Major earnings</h2>
  <p class="desc">S&amp;P 500 companies worth ${MAJOR_CAP / 1e9:.0f}B+ reporting in the next {EARN_DAYS} days,
  largest first each day.</p>
  {earn_tbl}
</section>
<footer><p>Sources: Nasdaq economic and earnings calendars, Federal Reserve FOMC calendar. Dates can
shift, so confirm against the source before trading around them.</p></footer>"""
    return (f'<!doctype html><html lang="en"><head><meta charset="utf-8">'
            f'<meta name="viewport" content="width=device-width,initial-scale=1">'
            f'<title>Market calendar</title>'
            f'<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&display=swap" rel="stylesheet">'
            f'<style>{S.CSS}</style></head><body><main>{body}</main><script>{S.JS}</script></body></html>')


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="docs")
    args = ap.parse_args()
    out = Path(args.out)
    cache = out / "data" / "calendar.json"
    cached = json.loads(cache.read_text()) if cache.exists() else {}
    members = pd.read_csv("data/sp500_constituents.csv").set_index("ticker")["name"].to_dict()
    today = dt.datetime.now(ZoneInfo("America/New_York")).date()

    data = fetch(today, members, cached)
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(data, indent=1))
    (out / "calendar.html").write_text(render(data, today), encoding="utf-8")
    print(f"Wrote {out / 'calendar.html'}: {len(data['econ'])} releases, "
          f"{len(data['fomc'])} FOMC meetings, {len(data['earnings'])} earnings")


if __name__ == "__main__":
    main()
