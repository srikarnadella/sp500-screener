# Working on this repo

Every change to screening, backtest, or report logic follows this loop before committing:

1. Run all tests: `for t in test_*.py; do python $t || break; done`
2. Run `python validate.py`. It backtests the real code on a frozen price snapshot and checks
   calculations, the claims the pages make, and findings vs `validation/baseline.json`.
3. If findings moved, show the user what moved and why before approving with
   `python validate.py --update` (commit the baseline in the same commit as the change).
   Never approve a baseline change just to make CI pass.
4. Build the pages offline to be sure they render:
   `python screener.py --demo --out /tmp/site && python dip_backtest.py --demo --null-series 20 --out /tmp/site && python dashboard.py --out /tmp/site`

If a page states a number about the method, make it a constant that `validate.py` checks
(see `CLAIM_*` in `dip_backtest.py` and `screener.py`), never a hardcoded figure in the text.
