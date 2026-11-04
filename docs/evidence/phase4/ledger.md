# Experiment ledger

Every variant tested, so selection stays visible.
An entry is added whenever a variant or an assumption changes.

## Pre-registered variants (16)

| Book | Spread multiple | Policies |
|---|---|---|
| Short strangle | 1 | unhedged, delta_daily, delta_band, opt_dg, opt_dgv, opt_dgv_cheap, opt_dgv_cvar |
| Short strangle | 3 | unhedged, delta_daily, delta_band, opt_dgv, opt_dgv_cheap |
| Long put overlay | 1 | unhedged, delta_daily, opt_dgv, opt_dgv_cheap |

## Runs

| Run | Code | Outcome | Status |
|---|---|---|---|
| 1 | `e973e3e` | Stopped by hand, because the solver was being changed | Discarded, no results read |
| 2 | `7dffb11` | `opt_dgv_cvar` bought 11.6 billion lots and lost $10 trillion, with 318 failed days. Others completed | **Rejected: exposed a defect.** CVaR could be farmed for profit |
| 3 | `062f33a` | All variants completed, with no failed days | Superseded: results depended on machine load |
| 4 | `e5d3a9d` | All variants completed, with no failed days. Delta policies identical to run 3, and optimizer policies differed | Superseded: this difference exposed the load dependence |
| 5 | With the node limit | All variants completed, with no failed days | Superseded by run 6, with identical values |
| 6 | After the gate, search limit, and repair fixes | All variants completed, with no failed days. Every value identical to run 5 | **Current:** `proxy_backtest.json` |

## Changes between runs 2 and 3

These were all fixes to defects, none aimed at improving results, and every variant was rerun:

- CVaR floored at zero, with a hard lot cap.
- Default gamma weight corrected from 0.015 to 0.045.
- Accounting, time, and quote gate fixes from independent review.

Reported variance reduction against delta for `opt_dgv` moved from 93.1% to 92.3%.

## Changes between runs 3 and 5

Fixes from a further review, plus a search that stops at a node count rather than a clock.
No change was aimed at improving results, and every variant was rerun.
Variance reduction against delta for `opt_dgv` moved from 92.3% to 91.1%.
That movement, between runs of nominally equal quality, is a useful sense of how much weight small differences here can bear.

## Judgment values, set once and not tuned

| Value | Setting | Source |
|---|---|---|
| Smile skew and curvature | -0.10, 0.02 | Stylized |
| ATM as a share of VIX | 0.90 | Stylized, since the VIX sits above ATM vol |
| Option half-spread | 0.15 + 0.2% of premium, scaled up with VIX | Near the SPX levels seen on 2026-10-30, and otherwise assumed |
| Risk aversion | 1.0, and 0.2 for `cheap` | Judgment |
| No-trade band | 0.5 bp cost, risk aversion 1e-6 | Judgment |

## Historical-chain run, pre-registered

Entered here before the first full run on real chains.

Smoke runs over parts of 2020 were read while the market build was debugged, and three handling changes followed, each for a data reason and none for a P&L figure:

- Spot comes from put-call parity rather than the close. With the close, 2020-03-16 and other fast days were skipped, because the source's chain snapshot and close are not simultaneous, and dropping the worst days would flatter every hedge.
- Books and hedge grids prefer listed monthly expiries, as the proxy's do, because the source keeps a monthly in its sample for weeks, while its other expiries change every day.
- A held-out mark audit and a mark-noise estimate were added, because the source almost never quotes the same contract twice, so most marks come from the calibrated surface.

Tuesdays and Thursdays were added when samples showed the source has some in later years.

| Item | Setting | Source |
|---|---|---|
| Data | SPY end-of-day option bid and ask, SPY closes and dividends, from DoltHub `post-no-preference/options` and `post-no-preference/stocks`; FRED DGS3MO for rates | Free, no account |
| Period | Every chain day the source has from 2019-01-01 to 2026-10-31, probing every weekday | The source's sampling: mostly Monday, Wednesday, and Friday, with other days in later years |
| Books | The proxy's two, at the same notional: short strangle of 100 SPY lots, and 100,000 SPY shares with a protective put | One SPX point is about ten SPY dollars |
| Policies | The proxy's 16 variants, built by `registered_policy`, with no parameter changed | `backtest/policies.py` |
| Spread multiples | 1, the quoted spread, and 3 as a stress | Same as the proxy |
| Fees | $0.50 per option contract; the stock trades at $0.01 a side with no fee | Assumed |
| Quote size | 500 contracts a side, since the source has none | Assumed, same as the proxy |
| Dividends | The last amount paid, projected onto quarterly third Fridays | No amount is read before it is paid |
| Warm-up | None, since the chain market has no trailing state | |
| Stress days | 2020-03-16 and 2024-08-05, the proxy's dates inside this period | Fixed in advance |
| Bootstrap block | 20 observations, about seven weeks at three a week | |

Gate decision, fixed in advance as the proxy README states it: **go** if option-hedge variance and tail reduction against daily delta clear zero at quoted spreads, and **no-go, fix the model** if they do not.

## Promotion rule

Nothing here is promoted to a recommended default.
That needs a rerun on historical chains, then a paper pilot.
