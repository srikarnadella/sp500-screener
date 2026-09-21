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
   Market breadth (% above 50/200-day, RSI extremes) is shown alongside.

## Outputs

| File | What |
|---|---|
| `docs/index.html` | The daily report: VIX regime, three candidate tables, searchable table of all stocks |
| `docs/data/screen_latest.csv` | Every stock and every metric, including per-level hold rates and baselines |
| `docs/data/signals_history.csv` | Top picks logged daily, so you can forward-test the ideas later |

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

```bash
python dip_backtest.py            # live data
python dip_backtest.py --demo     # synthetic data preview
python test_dip.py                # includes look-ahead, overlap and bias checks
```

Detection power: with a 10-session hold the gate finds a planted 3 to 5 day bounce pattern in ~60% to 98% of simulated
series but is nearly blind to reversion slower than about 10 days.

## Set up the daily automation (GitHub Actions, free)

1. Create a new GitHub repo and push this folder to it.
2. **Settings > Pages**: deploy from branch `main`, folder `/docs`. Your report will live at
   `https://<you>.github.io/<repo>/`. (Public repos only on free plans. Use a private repo without Pages
   and just open `docs/index.html` from a clone if you want it private.)
3. **Actions tab > Daily S&P 500 screen > Run workflow** to run it once by hand and confirm it works.
4. From then on it runs Mon-Fri at 21:30 UTC and commits the refreshed report.

### Or run it on your own machine

```bash
pip install -r requirements.txt
python screener.py                # writes docs/index.html
python screener.py --demo         # synthetic data, no network, to preview the report
python test_screener.py           # sanity tests for the screener
```

Schedule it with cron (Mac/Linux), e.g. weekdays at 4:45pm Eastern:
`45 16 * * 1-5  cd /path/to/sp500-screener && python screener.py`
On Windows use Task Scheduler with the same command. Run it after the close: a mid-session run
would treat a partial bar as the day's close.

## Tuning

Everything is in the config block at the top of `screener.py`: level list, test tolerance, bounce/break size,
minimum test count, VIX regime thresholds and multipliers. The VIX multipliers are judgment-based, not fitted.
After a few months, `signals_history.csv` lets you check whether high-scoring picks actually did better than the rest.

## Known limits

- **Yahoo is unofficial.** `yfinance` scrapes Yahoo Finance and can be rate-limited, especially from shared cloud IPs
  like GitHub's runners. The script refuses to publish if fewer than 80% of tickers download, so a bad day
  leaves yesterday's report in place. If it fails repeatedly, run it locally or swap in a paid data source.
- **Survivorship bias.** It uses today's constituents, so the per-stock history covers companies that are in the index now.
- **Multiple comparisons.** Six levels x 500 stocks means some levels will look respected by chance
  (about 10% of pure-noise stocks get flagged in testing). Read the test counts, not just the percentages.
- **Not investment advice.** These are screening heuristics, not a validated trading edge.
