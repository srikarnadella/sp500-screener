# S&P 500 level-respect screener

Screens every S&P 500 stock for setups at technical levels **that stock has actually respected**,
then adjusts the ranking for the VIX regime. Runs automatically each weekday after the close.

## What it does

1. **Universe** – current S&P 500 constituents (Wikipedia, cached in `data/`).
2. **Data** – 5 years of daily prices per stock, plus `^VIX` and `^VIX3M`, from Yahoo Finance via `yfinance`.
3. **Levels** – SMA 20/50/100/200 and the lower/upper Bollinger Band (20, 2). RSI, MACD, %B and ATR are computed for context.
4. **Respect backtest, per stock, per level** – every time price *tested* a level (came within 0.5 ATR from the correct side),
   did it move 1 ATR away (held) or 1 ATR through (broke) first, within 10 bars?
5. **Placebo baseline** – each stock is also tested against fake levels (the same level shifted several ATRs; Bollinger
   bands at other std-dev multiples). Any price line "holds" more than 50% of the time by construction, and bull markets
   inflate it further. A level counts as *respected* only if it beats the stock's own baseline by 10+ points over 6+ tests.
6. **Scores (0-100)**
   - **Long** = respected support in play (55%) + trend intact (25%) + pulled-back RSI / %B (20%)
   - **Fade** = respected resistance in play (55%) + weak trend / MACD (20%) + overbought RSI / %B (25%)
7. **VIX regime** multiplies the scores: *Stress* (VIX >= 30 or VIX/VIX3M >= 1.05) cuts long scores 40%;
   *Rising fear* cuts them 20%; *Fear rolling over* boosts them 20%; *Complacent* boosts fade 15% and trims long 10%.
8. **Volume confirmation and multi-timeframe check**: today's above-average volume adds up to 5 points to a
   stock's raw score; if the weekly trend (close vs. a sloping 10-week average) disagrees with the daily setup,
   the raw score is cut 15%.
9. **Relative strength**: each stock's 1- and 3-month return vs. SPY (`rs_1m`/`rs_3m`), and a sector relative-strength
   table showing which sectors are oversold vs. SPY, not just in absolute terms.
10. **Market conviction**: breadth (% above 50-/200-day, new 52-week highs/lows, McClellan Oscillator, a
    Zweig-style breadth thrust), the VIX regime, and the share of stocks with a live long setup are blended into
    one 0-100 conviction score shown at the top of the report. It's a heuristic overlay for context, not a
    multiplier on individual stock scores.
11. **Credit spreads**: the high-yield vs. Treasury option-adjusted spread (FRED `BAMLH0A0HYM2`, no API key,
    disk-cached) as an early stress signal -- widening spreads often precede equity weakness.
12. **Filters**: the "All screened stocks" table has sector/trend/volume-confirmed/min-score filters (combined
    with the existing text search), all client-side.

Per-ticker earnings-date lookups are *not* fetched for the full 500-stock universe -- 500 extra network calls
would worsen the Yahoo rate-limiting this script already retries around. That's covered in `dip_backtest.py`
instead, where the ticker list is small enough to afford it.

## Outputs

| File | What |
|---|---|
| `docs/index.html` | The daily report: conviction/breadth, sector relative strength, three candidate tables, filterable table of all stocks |
| `docs/data/screen_latest.csv` | Every stock and every metric, including per-level hold rates, relative strength and baselines |
| `docs/data/signals_history.csv` | Top picks logged daily, so you can forward-test the ideas later |
| `docs/data/market_context.json` | Snapshot of VIX/breadth/credit/conviction/sector-strength, so `dashboard.py` doesn't have to re-download anything |
| `docs/data/credit_spread.csv` | Cached FRED high-yield spread history |

## Merged dashboard

`dashboard.py` reads the outputs above (and `dip_backtest.py`'s) and writes `docs/dashboard.html`: conviction
score, breadth, sector relative strength, top long setups, live dip-bouncer signals and portfolio exposure on
one page, instead of three separate reports. It re-downloads nothing, so run it after the other two:

```bash
python screener.py && python dip_backtest.py && python dashboard.py
```

## Dip research for your holdings and their sector peers

`dip_backtest.py` answers a different question: which stocks bounce back reliably after a dip, and are any low right now?
It covers your 11 positions plus ~33 sector peers (edit the lists at the top of the file, or add tickers with
`--extra AAPL,TSLA`) and writes `docs/dip.html` and `docs/data/dip_results.csv`.

- **Three fixed dip rules**, never tuned per stock: lower Bollinger Band + RSI under 40 above the 200-day; a pullback to the
  50-day in an uptrend; and a 10% drop from the 60-day high above the 200-day.
- **Realistic trades**: signal at the close, buy the next open, sell 10 sessions later, 0.1% round-trip cost, no overlapping trades.
- **Five-test consistency gate**: 15+ trades, 60%+ win rate, beats the stock's own normal 10-day return (t of 2+),
  positive in both the older 60% and recent 40% of history, and positive in 60%+ of calendar years.
- **Self-calibrating**: the report runs the same gate on 150 random-walk series and tells you how many passes to expect by luck.
- **VIX study**: pooled dip results grouped by the VIX level and its 5-day direction on the signal day.
- **Trigger prices** for stocks that are armed but not yet triggered, so you can see how far each is from a signal.
- SDS and other leveraged inverse funds are flagged as structurally decaying and excluded from grading.
- **Extra backtest rigor**: a bootstrap (Monte Carlo) 90% confidence interval on the edge and win rate, not just the
  point estimate; a 5-fold walk-forward edge-sign check (no re-optimization -- the rules are fixed by design);
  a cost-sensitivity check at 3x the assumed round-trip cost; and buy-and-hold's own total/annualized return as
  the benchmark every rule has to beat.
- **Paper-trading log**: every live run logs today's signals to `docs/data/dip_signals_history.csv`, and once a
  signal's 10-session hold has actually elapsed, its real outcome (entry/exit price, realized return) is recorded
  to `docs/data/paper_trades.csv` -- the honest check on whether the backtest holds up in real time.
- **Position sizing**: a half-Kelly suggested size (capped, from the rule's own win rate and avg win/loss) against
  `ACCOUNT_SIZE` (edit it at the top of the file) for any Consistent/Mostly stock with a live signal.
- **Portfolio exposure**: equal-weighted exposure by factor/sector group across your positions (reuses the
  existing peer-group labels).
- **Short interest, insider activity, options skew, earnings dates** (your ~50 positions + peers only -- too slow
  for the full 500-stock screener): short % of float and days-to-cover, net insider buying/selling from SEC
  Form 4 filings over the last 6 months, the near-term put/call volume ratio, and a flag when the next earnings
  date falls inside a signal's hold window. Pass `--skip-extra` to skip these and run faster.

```bash
python dip_backtest.py               # live data, with short interest/insider/options/earnings
python dip_backtest.py --skip-extra  # live data, faster, without the extras above
python dip_backtest.py --demo        # synthetic data preview, no network
python test_dip.py                   # includes look-ahead, overlap and bias checks
```

Detection power: with a 10-session hold the gate finds a planted 3 to 5 day bounce pattern in ~60% to 98% of simulated
series but is nearly blind to reversion slower than about 10 days.

## Experimental: a learned P(hold) model (`ml_respect.py`)

**Not wired into the live report.** `level_respect()`'s "respect" measurement is a hand-built rule
(count held vs. broke, shrink toward a placebo baseline). `ml_respect.py` asks whether a gradient-
boosted classifier, trained on features at each test event (RSI, %B, relvol, trend context, which
level/role/family) pooled across every stock, predicts held/broke better than that heuristic does
**out of time** -- trained on the earlier events, scored on a later slice neither side has seen.

- **The heuristic side is reconstructed causally**: each stock's own shrunk rate, computed only
  from that stock's training-period events -- not a straw man built from pooled/global rates.
- **A constant baseline (always predict the training hold-rate) is reported alongside both.**
  This caught a real issue during development: the heuristic reconstruction scored *worse* than
  this trivial constant on a run with only ~70% of history to estimate from, because shrinking
  toward hundreds of per-ticker/level/role rates has more variance than one global rate when
  there's little data per group. Without this baseline, "the model beats the heuristic" could
  just mean the heuristic estimate was noisy, not that the model found anything real.
- **Two out-of-sample checks**: same stocks/later dates, and entirely held-out tickers (stricter --
  catches the model fingerprinting a stock's own quirks instead of learning something transferable).
- **The decisive test lives in `test_ml_respect.py`**: shuffle the labels so there's provably
  nothing left to learn, and confirm the model's Brier score collapses to the constant baseline
  (no false skill) rather than "beating" a benchmark that has nothing real to find.

```bash
python ml_respect.py --demo     # synthetic data, no network, prints the benchmark
python ml_respect.py            # live S&P 500 history, writes data/ml_respect_report.json
python test_ml_respect.py       # the shuffle-label null check + regression tests
```

Whether this is worth wiring into `screener.py`'s live scoring depends on what it reports on real
history, not synthetic data -- run it live and read the verdict before deciding.

## Set up the daily automation (GitHub Actions, free)

1. Create a new GitHub repo and push this folder to it.
2. **Settings > Pages**: deploy from branch `main`, folder `/docs`. Your report will live at
   `https://<you>.github.io/<repo>/`. (Public repos only on free plans. Use a private repo without Pages
   and just open `docs/index.html` from a clone if you want it private.)
3. **Actions tab > Daily S&P 500 screen > Run workflow** to run it once by hand and confirm it works.
4. From then on it runs Mon-Fri at 9:00am and 5:00pm EST (14:00/22:00 UTC) and commits the refreshed report.

### Or run it on your own machine

```bash
pip install -r requirements.txt
python screener.py                # writes docs/index.html
python screener.py --demo         # synthetic data, no network, to preview the report
python test_screener.py           # sanity tests for the screener
python dashboard.py               # after the two above: writes docs/dashboard.html
```

Schedule it with cron (Mac/Linux), e.g. weekdays at 4:45pm Eastern:
`45 16 * * 1-5  cd /path/to/sp500-screener && python screener.py`
On Windows use Task Scheduler with the same command. Run it after the close: a mid-session run
would treat a partial bar as the day's close.

## Tuning

Everything is in the config block at the top of `screener.py`: level list, test tolerance, bounce/break size,
minimum test count, VIX regime thresholds and multipliers. The VIX multipliers are judgment-based, not fitted.
After a few months, `signals_history.csv` lets you check whether high-scoring picks actually did better than the rest.

In `dip_backtest.py`: your `POSITIONS`/`PEER_GROUPS`, the dip-rule thresholds, `HORIZONS`/`COST`/`HIGH_COST`, the
five consistency-gate thresholds, and `ACCOUNT_SIZE`/`KELLY_CAP` for the position-sizing suggestion.

## Known limits

- **Yahoo is unofficial.** `yfinance` scrapes Yahoo Finance and can be rate-limited, especially from shared cloud IPs
  like GitHub's runners. The script refuses to publish if fewer than 80% of tickers download, so a bad day
  leaves yesterday's report in place. If it fails repeatedly, run it locally or swap in a paid data source.
  `dip_backtest.py`'s short interest/insider/options/earnings lookups add ~4 more Yahoo calls per ticker per run
  (twice daily in the scheduled workflow); pass `--skip-extra` if this starts tripping the rate limit.
- **Survivorship bias.** It uses today's constituents, so the per-stock history covers companies that are in the index now.
- **Multiple comparisons.** Six levels x 500 stocks means some levels will look respected by chance
  (about 10% of pure-noise stocks get flagged in testing). Read the test counts, not just the percentages.
- **Not investment advice.** These are screening heuristics, not a validated trading edge.
