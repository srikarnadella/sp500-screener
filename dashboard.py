#!/usr/bin/env python3
"""
One merged landing page: the S&P 500 screen, the dip research, and the market
breadth/conviction gauge, instead of three separate reports.

Pure aggregator -- it re-downloads nothing. It reads the CSV/JSON files that
screener.py and dip_backtest.py already write, so run those first:

  python screener.py
  python dip_backtest.py
  python dashboard.py         writes docs/dashboard.html
  python dashboard.py --alert alert.md
                              also writes alert.md when something needs attention next session
                              (the daily workflow turns it into a GitHub issue, which emails you)
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from pandas.tseries.holiday import (AbstractHolidayCalendar, GoodFriday, Holiday, USLaborDay,
                                    USMartinLutherKingJr, USMemorialDay, USPresidentsDay,
                                    USThanksgivingDay, nearest_workday)
from pandas.tseries.offsets import CustomBusinessDay

import dip_backtest as D
import econ_calendar as C
import screener as S


def _need(path: Path, script: str) -> None:
    if not path.exists():
        raise SystemExit(f"{path} not found -- run `python {script}` first.")


ALERT_EVENTS = ("CPI", "Core CPI", "Nonfarm Payrolls", "Core PCE Price Index")   # plus every FOMC decision


class NYSEHolidays(AbstractHolidayCalendar):
    """NYSE full-day closures. Skipping them matters: the alert for a data date covers the NEXT
    session, and runs on a holiday see the same data date, so treating a holiday as a session
    would silently skip the day after it (e.g. a Tuesday CPI after Labor Day)."""
    rules = [Holiday("New Year's Day", month=1, day=1, observance=nearest_workday), USMartinLutherKingJr,
             USPresidentsDay, GoodFriday, USMemorialDay,
             Holiday("Juneteenth", month=6, day=19, start_date="2022-01-01", observance=nearest_workday),
             Holiday("Independence Day", month=7, day=4, observance=nearest_workday), USLaborDay,
             USThanksgivingDay, Holiday("Christmas", month=12, day=25, observance=nearest_workday)]


SESSION = CustomBusinessDay(calendar=NYSEHolidays())


def next_session(d: dt.date) -> dt.date:
    return (pd.Timestamp(d) + SESSION).date()


def alerts(dip: pd.DataFrame, cal: dict | None, asof: dt.date) -> list[str]:
    """Markdown bullets for what needs attention by the next session: a consistent dip-bouncer
    firing today, a holding reporting earnings, or a top-tier release/FOMC decision. Keyed to the
    data date, so the evening run and the next morning's run (same close) raise the same alert."""
    items = [f"- **{r.ticker}** dip signal today: {r.grade}, {r.best_rule}, {r.win:.0%} historical win rate"
             for r in dip[dip["grade"].isin(["Consistent", "Mostly"]) & (dip["status"] == "Signal today")].itertuples()]
    if cal:
        nxt = next_session(asof)
        econ, earn = C.upcoming(cal, nxt, days=1, also=tuple(D.POSITIONS))
        day = f"{nxt:%a %b} {nxt.day}"
        items += [f"- **{r.ticker}** (you hold it) reports earnings {day}" + (f", {r.time.lower()}" if r.time != "-" else "")
                  for r in earn[earn["ticker"].isin(D.POSITIONS)].itertuples()]
        items += [f"- **{r.event}** {day} at {r.time} ET" for r in econ.itertuples()
                  if r.event in ALERT_EVENTS or r.event.startswith("FOMC rate decision")]
    return items


def build(out: Path) -> str:
    _need(out / "data" / "market_context.json", "screener.py")
    _need(out / "data" / "screen_latest.csv", "screener.py")
    _need(out / "data" / "dip_results.csv", "dip_backtest.py")

    ctx = json.loads((out / "data" / "market_context.json").read_text())
    screen = pd.read_csv(out / "data" / "screen_latest.csv")
    dip = pd.read_csv(out / "data" / "dip_results.csv")

    vix, breadth, credit, conviction = ctx["vix"], ctx["breadth"], ctx["credit"], ctx["conviction"]

    conv_parts = "".join(f"<div><dt>{S._esc(name)}</dt><dd>{val:.0f}</dd></div>" for name, val in conviction["parts"])
    credit_dd = (f"{credit['level']:.2f}<small>{credit['chg20']:+.2f} pts / 20d</small>"
                if credit.get("ok") else "n/a")
    header = f"""
<section class="regime" aria-label="Market conviction">
  <h2>Conviction: {conviction['score']:.0f}/100, {S._esc(conviction['label'])}</h2>
  <p class="note">VIX regime: {S._esc(vix['label'])}. {S._esc(vix['note'])}</p>
  <dl class="stats">
    {conv_parts}
    <div><dt>VIX</dt><dd>{vix['level']:.2f}</dd></div>
    <div><dt>Above 50-day / 200-day</dt><dd>{breadth['above50']:.0f}% / {breadth['above200']:.0f}%</dd></div>
    <div><dt>New 52-week highs / lows</dt><dd>{breadth['new_highs']} / {breadth['new_lows']}</dd></div>
    <div><dt>McClellan Oscillator</dt><dd>{breadth['mcclellan']:+.0f}</dd></div>
    <div><dt>Zweig breadth thrust</dt><dd>{"Firing" if breadth['zweig_thrust'] else "No"}</dd></div>
    <div><dt>High-yield credit spread</dt><dd>{credit_dd}</dd></div>
  </dl>
</section>"""

    sec_rows = S._rows(pd.DataFrame(ctx["sectors"]).head(6), [
        lambda r: S._td(S._esc(r["sector"]), cls="l", sort=S._esc(r["sector"])),
        lambda r: S._td(S._pct(r["rs_1m"], 1, True), sort=f"{r['rs_1m']:.4f}" if np.isfinite(r["rs_1m"]) else -9,
                        cls=S._cls(r["rs_1m"]) if np.isfinite(r["rs_1m"]) else ""),
        lambda r: S._td(S._pct(r["rs_3m"], 1, True), sort=f"{r['rs_3m']:.4f}" if np.isfinite(r["rs_3m"]) else -9,
                        cls=S._cls(r["rs_3m"]) if np.isfinite(r["rs_3m"]) else ""),
    ])
    sector_tbl = S._table(["Most oversold sectors vs SPY", "1m", "3m"], sec_rows)

    top_long = screen[screen["long_score"] > 0].sort_values("long_score", ascending=False).head(8)
    long_rows = S._rows(top_long, [
        S._stock_cell,
        lambda r: S._td(f'<span class="score">{r["long_score"]:.0f}</span>', sort=f"{r['long_score']:.2f}"),
        lambda r: S._td(S._esc(r["support_level"]), cls="l", sort=S._esc(r["support_level"])),
        lambda r: S._td(S._esc(r["trend"]), sort=S._esc(r["trend"])),
    ])
    long_tbl = S._table(["Stock", "Long score", "Support", "Trend"], long_rows)

    catch = dip[dip["grade"].isin(["Consistent", "Mostly"])
               & dip["status"].isin(["Signal today", "Signal in last 3 sessions"])]
    catch_rows = S._rows(catch, [
        lambda r: S._td(f"<b>{S._esc(r['ticker'])}</b>", sort=S._esc(r["ticker"])),
        lambda r: S._td(S._esc(r["grade"]), cls="l", sort=S._esc(r["grade"])),
        lambda r: S._td(S._esc(r["best_rule"]), cls="l", sort=S._esc(r["best_rule"])),
        lambda r: S._td(S._esc(r["status"]), cls="l", sort=S._esc(r["status"])),
        lambda r: S._td(f"{r['win'] * 100:.0f}% win", sort=f"{r['win']:.3f}" if np.isfinite(r["win"]) else -1),
    ])
    catch_tbl = S._table(["Stock", "Grade", "Rule", "Status", "Record"], catch_rows)

    # Optional: econ_calendar.py's cache. Missing just means no "coming up" section.
    cal_path = out / "data" / "calendar.json"
    coming = ""
    if cal_path.exists():
        today = dt.datetime.now(ZoneInfo("America/New_York")).date()
        econ, earn = C.upcoming(json.loads(cal_path.read_text()), today, days=7, also=tuple(D.POSITIONS))
        coming = f"""
<section class="block">
  <h2>Coming up this week</h2>
  <p class="desc">Key releases, Fed decisions, and earnings from $100B+ names or your holdings over the next
  7 days. Full list on the <a href="calendar.html">calendar</a>.</p>
  {C.econ_table(econ, "No key releases in the next 7 days.")}
  <div style="height:12px"></div>
  {C.earnings_table(earn, "No major or held-stock earnings in the next 7 days.")}
</section>"""

    # Track record: the screen's logged picks (scored by screener.py) and the dip paper trades.
    sc = ctx.get("scorecard") or {}
    track = [(f"Screen: {lbl}", s.get("n", 0), s.get("pending", 0), s.get("avg_ret"), s.get("avg_excess"), s.get("beat"))
             for key, lbl in (("long", "long setups"), ("fade", "fade setups")) if (s := sc.get(key)) is not None]
    pt_path = out / "data" / "paper_trades.csv"
    if pt_path.exists():
        pt = pd.read_csv(pt_path)
        track.append(("Dip signals (paper trades)", len(pt), None, pt["realized_return"].mean(), None, pt["win"].mean()))
    num = lambda x: x is not None and np.isfinite(x)
    track_tbl = '<div class="wrap"><p class="empty">No track record yet: it starts once screener.py has run with scoring enabled.</p></div>' if not track else S._table(["Picks", "Resolved", "Still in hold", "Avg 10-session return", "Avg vs SPY", "Hit rate"], [
        "<tr>" + S._td(S._esc(name), cls="l") + S._td(str(n), sort=n)
        + S._td("-" if pending is None else str(pending))
        + S._td(S._pct(ret, 2, True) if num(ret) else "-", cls=S._cls(ret) if num(ret) else "")
        + S._td(S._pct(exc, 2, True) if num(exc) else "-", cls=S._cls(exc) if num(exc) else "")
        + S._td(S._pct(hit, 0) if num(hit) else "-") + "</tr>"
        for name, n, pending, ret, exc, hit in track])

    exposure = D.portfolio_exposure(dip)
    exp_rows = S._rows(exposure, [
        lambda r: S._td(S._esc(r["group"]), cls="l", sort=S._esc(r["group"])),
        lambda r: S._td(f"{r['weight']:.0f}%", sort=f"{r['weight']:.2f}"),
        lambda r: S._td(S._esc(r["tickers"]), cls="l"),
    ])
    exposure_tbl = S._table(["Factor / sector group", "Weight", "Tickers"], exp_rows)

    body = f"""
{S._nav("dashboard.html")}
<h1>Market dashboard</h1>
<p class="sub" data-asof="{S._esc(ctx['asof'])}">Data through {S._esc(ctx['asof'])}. Merges the S&amp;P 500 screen, the dip research and the
breadth/conviction gauge. Full detail in the two linked reports below.</p>
{header}
{coming}
<section class="block">
  <h2>Sector relative strength</h2>
  {sector_tbl}
</section>
<section class="block">
  <h2>Top long setups</h2>
  <p class="desc">From the <a href="index.html">S&amp;P 500 screen</a> -- full table there.</p>
  {long_tbl}
</section>
<section class="block">
  <h2>Consistent dip-bouncers with a live signal</h2>
  <p class="desc">From <a href="dip.html">dip research</a> -- sizing, confidence intervals and full detail there.</p>
  {catch_tbl}
</section>
<section class="block">
  <h2>Portfolio exposure by factor / sector group</h2>
  {exposure_tbl}
</section>
<section class="block">
  <h2>Track record</h2>
  <p class="desc">How past picks actually did, bought at the next open and sold 10 sessions later. Hit rate: share of
  long setups that beat SPY, share of fade setups that lagged it, and share of dip trades that made money. This is
  the live check on the backtests, and it needs months of picks before it means much.</p>
  {track_tbl}
</section>
<footer><p>Screening heuristics, not investment advice. See <a href="index.html">the S&amp;P 500 screen</a> and
<a href="dip.html">the dip research</a> for methodology and limits.</p></footer>"""
    return (f'<!doctype html><html lang="en"><head><meta charset="utf-8">'
            f'<meta name="viewport" content="width=device-width,initial-scale=1">'
            f'<title>Market dashboard, {S._esc(ctx["asof"])}</title>'
            f'<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&display=swap" rel="stylesheet">'
            f'<style>{S.CSS}</style></head><body><main>{body}</main><script>{S.JS}</script></body></html>')


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="docs")
    ap.add_argument("--alert", help="write an alert file here when something needs attention")
    args = ap.parse_args()
    out = Path(args.out)
    page = build(out)
    (out / "dashboard.html").write_text(page, encoding="utf-8")
    print(f"Wrote {out / 'dashboard.html'}")
    if args.alert:
        asof = dt.date.fromisoformat(json.loads((out / "data" / "market_context.json").read_text())["asof"])
        cal_path = out / "data" / "calendar.json"
        items = alerts(pd.read_csv(out / "data" / "dip_results.csv"),
                       json.loads(cal_path.read_text()) if cal_path.exists() else None, asof)
        if items:   # first line is the issue title, the rest its body
            Path(args.alert).write_text(f"Screener alerts for {asof}\n\n" + "\n".join(items)
                                        + "\n\nDashboard: https://srikarnadella.github.io/sp500-screener/dashboard.html\n")
            print(f"Wrote {args.alert}: {len(items)} alert(s)")


if __name__ == "__main__":
    main()
