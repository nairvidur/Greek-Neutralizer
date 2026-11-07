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
| Stress days | 2020-03-16 and 2024-08-05, the proxy's dates inside this period | Fixed in advance. The source has no chain on 2024-08-05, or on any weekday from 2024-07-30 to 2024-08-09, so no period ends on it; the README reports the period that spans it |
| Bootstrap block | 20 observations, about seven weeks at three a week | |

Gate decision, fixed in advance as the proxy README states it: **go** if option-hedge variance and tail reduction against daily delta clear zero at quoted spreads, and **no-go, fix the model** if they do not.

## Historical-chain runs

| Run | Code | Outcome | Status |
|---|---|---|---|
| 1 | `9e27b7d` | All 16 variants completed. The machine was starved of CPU for hours, and solves hit the 30 second wall-clock backstop, which the backtest ignored | **Rejected: exposed a defect.** Results depended on machine load |
| 2 | `9e27b7d` plus per-period series | The 11 variants at quoted spreads, partly on a loaded machine. One variant had a timed-out day, and optimizer results matched run 1 only where run 1 was not starved | Discarded as evidence; it showed run 1 was not reproducible |
| 3 | `40a5b8c` | All 16 variants, with no failed days and no solve at the backstop. Every optimizer variant but `opt_dgv_cheap` at quoted spreads matches, to the dollar, run 1 or run 2 wherever that run had the machine to itself, and that one was load-affected in both | **Current:** `chain_backtest.json`, and `chain_horizons.json` from its series. Decision: no-go |

## Changes between chain runs 1 and 3

- The backtest policy refuses a solve that stopped at the wall-clock backstop, so that day is recorded as failed instead of traded.
- The registered configs raise that backstop from 30 to 600 seconds, so the node limit governs, as these runs were designed to. The proxy evidence was made with the old backstop, and was not rerun. Its runs 5 and 6 gave identical values, which says the backstop did not bind there.
- Each period records option contracts traded and their cost, and per-period series are written for the post-hoc horizon check.

None of these was aimed at a result, and every variant was rerun.

## Position-aware optimizer, pre-registered

Entered here, and committed, before any run on the held-out symbols.

### Diagnosis, made on SPY after the no-go

These findings were read from chain run 3's SPY series and from short SPY traces, so they are in-sample.

- Twenty of 1,163 periods carry 74% of the squared difference between `opt_dgv`'s period P&L and daily delta's.
- The objective priced only the day's trade, never the position left behind.
  Each day the grid offered new contracts, the policy added legs, and it never unwound offsetting ones, because unwinding paid a spread and holding cost nothing.
  On one March 2026 trace the hedges held 28 legs and 1,313 contracts against a 200-contract book, with seven expiries.
- That gross position is flat in net delta, gamma, and 30-day vega, but it carries skew and term-structure risk those Greeks do not see, and most of it is marked from the model.
- Separately, the source has one SPY chain day in 2019, 2019-05-10, then none until 2020-01-06, so chain run 3's first period spans eight months.
  It is about 40% of daily delta's sum of squared period P&L, so it inflated the baseline every policy was compared with, and the optimizer's no-go is stronger without it.

### The fix

- `HedgeConfig.price_exit` charges every candidate's half-spread and fee on the hedger's position in it after the trade, the cost of closing it later.
  Selling a held hedge then saves an exit, and opening an offsetting one adds a second.
- `HedgeConfig.max_gross_option_contracts` caps the absolute option contracts of the hedger's position after the trade.
- Both act on `HedgeProblem.held_hedges` plus the legs, never on the client's book, which a hedge does not close.
- `OptimizerPolicy.gross_multiple` sets that cap each day at a multiple of the book's own option contracts.
- `OptimizerPolicy.option_band` offers options only when residual gamma or vega is outside that share of the book's own, and otherwise hedges delta alone.
- `SURFACE_SHAPE_SHOCKS` are six moves of the surface's shape at about one period's size: skew steeper and flatter by 0.10 a unit of log-moneyness, front vol up and down 3 points at a term power of 1, a 3% selloff with vol up 5 and skew steeper, and a 2% rally with vol down 2 and skew flatter.
- Chain runs now start after the last gap of more than 15 calendar days between chain days, and the dropped days are listed in the output.

### Variants

| Policy | Targets | Risk aversion | Exit priced | Gross cap | Option band | CVaR |
|---|---|---|---|---|---|---|
| `opt_dgv_bounded` | delta, gamma, vega | 1.0 | yes | hedges up to 1x the book's option contracts | none | none |
| `opt_dgv_bounded_band` | delta, gamma, vega | 1.0 | yes | 1x | 0.5 | none |
| `opt_dgv_surface` | delta, gamma, vega | 1.0 | yes | 1x | 0.5 | weight 1.0 at alpha 0.8 on `SURFACE_SHAPE_SHOCKS` |

All other settings are the registered ones: up to 6 legs a trade, a 1% MIP gap, a 300-node limit, and the 600 second backstop.
Each runs on the short strangle at spread multiples 1 and 3, and on the long put overlay at 1, beside the 16 earlier variants, which are rerun unchanged: 25 variants per symbol.
The values 1x, 0.5, and the CVaR weight are judgment values, set before any run and not tuned.
If a cap is infeasible because the book shrank below its hedges by more than a day's displayed size can unwind, the policy re-solves with the cap at the hedges' current size, so they can shrink but not grow and the day's hedge is kept.

### Development run, disclosed

One run on SPY chain days from 2020-01-06 to 2021-06-30, at quoted spreads on the short strangle, with these exact settings, was read before this entry.

| Policy | Periods | Stdev | Variance reduction vs delta | Total P&L | Hedge cost |
|---|---|---|---|---|---|
| `delta_daily` | 213 | 9,085 | 0 | 114,755 | 2,034 |
| `opt_dgv` | 213 | 22,028 | -488% | -836,964 | 460,986 |
| `opt_dgv_bounded` | 213 | 1,913 | 95.6% | -65,699 | 36,789 |
| `opt_dgv_bounded_band` | 213 | 2,379 | 93.1% | -61,707 | 23,470 |
| `opt_dgv_surface` | 213 | 2,284 | 93.7% | -28,148 | 40,381 |

No setting was changed after it.
It shows the cost of the fix as well as its benefit: the option hedges remove most of the variance, and with it most of the strangle's premium.
It ran the first version of the fix, before the defect below was corrected.

### Correction after entry, before any held-out run

An independent review of the entered code found a defect, fixed before any held-out run.
The first version charged the exit and counted the cap on the whole book, the client's positions included, netted with the hedges by instrument.
A hedge that bought a client's short option was then priced as closing it, so it cost nothing, although the client still holds and closes its own leg; on the put overlay, whose 100,000 shares share a key with the stock hedge, selling shares was free.
Both terms now act only on the hedger's held hedges and its trade, as above.
The cap was 2x the whole hedged book's option contracts, netted; it is now 1x the book's for the hedges alone, which is the same 2x total without the netting.
SPY chain run 4, started on the first version, was stopped and not read.

### Held-out test

- Symbols: DIA, primary, and XLF, secondary: of the ETFs probed, the only two besides SPY that the free source carries.
  QQQ, IWM, EEM, GLD, and TLT have no chains there.
  Their chains were downloaded, but no backtest was run on them before this entry.
- Period: every chain day each source has from 2019-01-01 to 2026-10-31, after the 15-day gap rule.
  It is the same calendar period as SPY, so the market regime is shared, and only the instrument, its quotes, and its contracts are held out.
- Books, costs, fees, quote sizes, dividends, rates, stress days, and bootstrap: as for SPY, at the same contract counts, 100 lots and 100,000 shares.
  XLF trades at about a twelfth of SPY's price, so its book is smaller in dollars, and the $0.50 fee is larger against its premiums.
- Gate for each new variant, decided on DIA at quoted spreads on the short strangle: **go** if its variance reduction and its 97.5% ES improvement against daily delta both have 95% intervals above zero, and **no-go** otherwise.
  XLF is reported as a second test: a go on DIA that fails on XLF is reported as not confirmed.
- With three variants, one may pass by chance; every variant's result is reported.
- Not a gate, but stated with any decision: the mean P&L difference against daily delta, with its interval, which is the carry the hedge gives up.
  Removing a strangle's risk removes much of its premium, so a go says the hedge reduces risk at a measured cost, not that it pays.
- SPY is rerun with all 25 variants under the gap rule as chain run 4. Its results for the new variants are in-sample, and are labeled so.

## Promotion rule

Nothing here is promoted to a recommended default.
The chain run gave no-go, so promotion now needs a fixed policy, entered here as a new pre-registered variant before it runs, then a paper pilot.
The position-aware variants above are that entry: a go on the held-out test makes one eligible for a paper pilot, not a default.
