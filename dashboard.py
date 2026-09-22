#!/usr/bin/env python3
"""
One merged landing page: the S&P 500 screen, the dip research, and the market
breadth/conviction gauge, instead of three separate reports.

Pure aggregator -- it re-downloads nothing. It reads the CSV/JSON files that
screener.py and dip_backtest.py already write, so run those first:

  python screener.py
  python dip_backtest.py
  python dashboard.py         writes docs/dashboard.html
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import dip_backtest as D
import screener as S


def _need(path: Path, script: str) -> None:
    if not path.exists():
        raise SystemExit(f"{path} not found -- run `python {script}` first.")


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
<p class="sub">Data through {S._esc(ctx['asof'])}. Merges the S&amp;P 500 screen, the dip research and the
breadth/conviction gauge. Full detail in the two linked reports below.</p>
{header}
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
    args = ap.parse_args()
    out = Path(args.out)
    page = build(out)
    (out / "dashboard.html").write_text(page, encoding="utf-8")
    print(f"Wrote {out / 'dashboard.html'}")


if __name__ == "__main__":
    main()
