# Phase 4 evidence: does hedging with this work?

Gate: "a published out-of-sample report net of costs, including a go/no-go decision."

**Gate status: MET on free historical chains.**

- **The optimizer as first configured: NO-GO.** On SPY chains every variant had many times the variance of daily delta, worse expected shortfall, and far higher cost, so the proxy's 91% variance reduction did not survive real quotes.
- **The position-aware fix: GO on the held-out DIA test, NOT CONFIRMED on XLF.** On DIA, a symbol held out until the fix was entered in `ledger.md`, all three fixed variants cut the short strangle's variance against daily delta by 43% to 61%, and its 97.5% expected shortfall, with both intervals above zero at quoted spreads.
  On XLF, the second held-out symbol, variance fell 20% to 40%, but no tail improvement was established.
- **The hedge is not free.** On DIA it gave up $790 to $1,150 a period against daily delta, with intervals below zero: removing a strangle's risk removes much of its premium.
- **It is fragile.** At three times quoted spreads, or on the long put overlay, or over sums of 20 periods, the DIA result mostly does not hold.

Daily delta hedging remains the recommended default.
The fixed optimizer is eligible for a paper pilot, as the promotion rule allows, and is not a default.

## Position-aware optimizer: the held-out test

### What was fixed

Chain run 3, below, showed the optimizer's objective priced only each day's trade, never the position it left.
A daily re-hedge on a strike grid that moves with spot added new legs each day and never unwound offsetting ones: one SPY trace held 1,313 contracts in 28 legs against a 200-contract book.
That book was flat in net delta, gamma, and 30-day vega, but exposed to skew and term structure, and most of it was marked from the model.

The fix, in `HedgeConfig`, `HedgeProblem`, and `OptimizerPolicy` and described in `docs/conventions.md`:

- `price_exit` charges each candidate's half-spread and fee on the hedger's position after the trade, so unwinding a held hedge saves an exit and stacking a new one adds one.
- `max_gross_option_contracts` caps the hedger's option contracts after the trade, here at the book's own count.
- `option_band` trades options only when residual gamma or vega leaves half the book's own, and hedges delta alone otherwise.
- `SURFACE_SHAPE_SHOCKS`, six skew, front-vol, selloff, and rally moves, enter the third variant's CVaR.

Three variants were entered in `ledger.md` and committed before any run on the held-out symbols: `opt_dgv_bounded`, `opt_dgv_bounded_band`, and `opt_dgv_surface`.
An independent review then found that the first version charged exits on the client's own positions too, and that was fixed, and recorded, before those runs.

### The test

- SPY was rerun with all 25 variants as chain run 4. Its results for the fixed variants are in-sample, since the fix was designed on SPY.
- DIA, primary, and XLF, secondary, were held out: of the ETFs probed, they are the only two besides SPY that the free source carries.
  Their chains were downloaded before the fix was entered, and no backtest ran on them before it.
- They share SPY's calendar, 2020-01-06 to 2026-10-30, so the market regime is not held out, only the instrument, its quotes, and its contracts.
- Books, costs, and settings are SPY's, at the same contract counts.
- Each run now starts after the last gap of more than 15 days between chain days. The source has one chain day in 2019 for each symbol, then none until 2020, and in run 3 that made SPY's first period eight months long.

### Results: short strangle, 100 lots, quoted spreads

Against daily delta, with 95% block-bootstrap intervals.

| Symbol | Policy | Variance reduction | ES improvement | Mean P&L per period | Hedge cost |
|---|---|---|---|---|---:|
| DIA, held out | `opt_dgv_bounded` | **60.9% (44.3 to 73.4)** | **$11,799 (4,086 to 21,064)** | -$1,150 (-1,489 to -743) | 944,193 |
| DIA, held out | `opt_dgv_bounded_band` | **42.5% (25.6 to 65.1)** | **$9,933 (3,446 to 18,705)** | -$793 (-1,155 to -373) | 500,792 |
| DIA, held out | `opt_dgv_surface` | **59.1% (36.0 to 72.7)** | **$12,546 (3,473 to 22,860)** | -$935 (-1,301 to -524) | 555,989 |
| DIA, held out | `opt_dgv`, as first configured | -2,058% (-3,712 to -1,081) | -$119,034 | -$10,581 | 11,357,274 |
| XLF, held out | `opt_dgv_bounded` | 19.8% (-41.3 to 49.9) | $1,432 (-1,366 to 5,262) | -$119 (-226 to +42) | 145,810 |
| XLF, held out | `opt_dgv_bounded_band` | 28.2% (-15.5 to 51.6) | $2,015 (-710 to 5,777) | -$60 (-186 to +112) | 97,996 |
| XLF, held out | `opt_dgv_surface` | 40.3% (4.5 to 63.1) | $2,069 (-451 to 5,825) | -$103 (-203 to +34) | 98,566 |
| SPY, in-sample | `opt_dgv_bounded` | 88.2% (78.5 to 94.4) | $26,689 (17,856 to 36,438) | -$814 (-1,222 to -361) | 184,021 |
| SPY, in-sample | `opt_dgv_bounded_band` | 84.3% (75.6 to 91.8) | $25,534 (16,784 to 35,086) | -$766 (-1,174 to -319) | 126,923 |
| SPY, in-sample | `opt_dgv_surface` | 87.9% (84.7 to 91.2) | $27,365 (17,714 to 36,720) | -$643 (-1,099 to -132) | 127,516 |

Daily delta, for scale:

| Symbol | Periods | Total P&L | Period stdev | ES 97.5% | Hedge cost |
|---|---:|---:|---:|---:|---:|
| DIA | 1,180 | -331,748 | 8,531 | 36,831 | 10,391 |
| XLF | 967 | -249,545 | 2,301 | 9,598 | 8,588 |
| SPY | 1,162 | +543,586 | 8,824 | 38,255 | 10,329 |

### How to read this

- **By the pre-registered rule, each fixed variant is a go on DIA and not confirmed on XLF.** Out of sample the effect is smaller than in-sample on SPY, 43% to 61% against 84% to 88%, so the SPY figures should not be quoted as the result.
- **The data do not separate the three variants.** Their intervals overlap on every symbol, so none is picked here.
- **The price of the risk reduction is the strangle's carry.** On DIA the fixed hedges lost $1.27 to $1.69 million against delta's $0.33 million over the period. Whether that trade is worth making is the desk's call, and this shows its size.
- **Turnover fell twentyfold.** Across the three symbols the fixed variants traded 21 to 41 option contracts a period against 577 to 834 for `opt_dgv`, and on DIA their costs were $0.5 to $0.9 million against its $11.4 million.
- **XLF is the weakest test.** Its free chain has few strikes near the money, so 231 of its 1,199 days could not be calibrated and fail closed, against 34 for SPY and 26 for DIA, and its held-out mark error is an RMS of $0.50 a share against a $0.045 median half-spread. That includes 2020-03-16, so XLF has no stress-day figure.
- **It does not survive stress.** At three times quoted spreads on DIA, variance reduction is 16% to 20% and only `opt_dgv_bounded_band` clears the tail test, barely. On the long put overlay, no fixed variant's tail improvement clears zero on DIA or XLF, though variance falls 29% to 64%.
- **It fades with horizon on DIA.** In the post-hoc horizon check, `chain_holdout_dia_horizons.json`, the reduction holds at 5 and 10 periods, 45% to 68%, but over 20 periods, about two months, it is 13% to 34% with intervals that include zero. The lag-one autocorrelation of each fixed variant's P&L difference from delta is between -0.04 and +0.06, so reversing mark noise does not explain these results.
- **The stress day.** On the period ending 2020-03-16 the DIA strangle lost $106,879 delta hedged and $37,766 to $100,515 under the fixed variants. On SPY, in-sample, `opt_dgv_surface` lost $7,333 against delta's $59,031. One period each, so observations, not results.

Every variant on every symbol is in `chain_run4_spy.json`, `chain_holdout_dia.json`, and `chain_holdout_xlf.json`.

### What this does not establish

- That the fix works in another market regime: the held-out symbols share SPY's calendar, and DIA moves closely with SPY.
- That it works on a full listed chain. 92% to 99% of held option positions on DIA and SPY had a model mark at one end of a period, because the free source rarely quotes the same contract twice.
- That any of the three is better than the others.

## SPY chain run 3: the first test, and the no-go

This is the run the no-go was decided on, before the fix.
Run 4, above, corrected one flaw in it: its first period ran eight months, from the source's lone 2019 chain day, and was about 40% of daily delta's sum of squared period P&L.
Without it daily delta's period stdev is $8,824 rather than $11,512, and `opt_dgv`'s variance against delta is -1,988% rather than -1,738%, so the no-go is stronger.
The tables below are run 3's, as decided.

### What was run

Data, all free and with no account:

- SPY end-of-day option bid and ask from DoltHub `post-no-preference/options`: 1,198 days from 2019-05-10 to 2026-10-30, mostly three a week, with about four expiries and 13 to 27 strikes each.
- SPY closes and cash dividends from DoltHub `post-no-preference/stocks`, and FRED DGS3MO for rates.

`scripts/fetch_dolthub_chains.py` downloads them, and `scripts/run_chain_backtest.py` runs the proxy's 16 pre-registered variants on them, unchanged except for the mapping to SPY at the same notional.
Settings were entered in `ledger.md` before the full run, which also discloses the smoke runs read while the market build was debugged.

How each day is built, in `backtest/chains.py`:

- The phase 2 calibration on that day's real quotes, and nothing from later days, which is tested.
- Spot from put-call parity across the calibrated expiries, because the source's close and chain snapshot are not simultaneous. The median gap to the close is 0.07%, and the largest 2.1%.
- Dividends projected from the last amount already paid, on quarterly third Fridays.
- Trades at the real bid and ask, plus $0.50 per option contract. The stock trades at $0.01 a side. The source has no quote sizes, so 500 contracts a side are assumed, as in the proxy.

34 days were skipped, all because the calibrated surface showed arbitrage inside the quoted strikes, which fails closed by design.
The next P&L period spans each skipped day.
No policy failed to produce a hedge on any day.

### Results

Short strangle of 100 SPY lots, rolled monthly, over 1,163 P&L periods, in dollars at quoted spreads.
A period runs from one chain day to the next, two or three calendar days, so these are not daily figures.

| Policy | Total P&L | Period stdev | Worst period | ES 97.5% | Hedge cost |
|---|---:|---:|---:|---:|---:|
| Unhedged | +907,508 | 25,382 | -258,236 | 99,289 | 0 |
| Delta, daily | +291,804 | 11,512 | -251,794 | 46,208 | 10,329 |
| Delta, no-trade band | +268,438 | 12,316 | -213,657 | 50,119 | 4,416 |
| Optimizer, delta and gamma | -536,136 | 35,542 | -467,955 | 120,991 | 1,344,620 |
| Optimizer, delta, gamma, vega | -698,649 | 49,347 | -346,084 | 114,289 | 2,549,622 |
| Same, lower risk aversion | -837,440 | 55,050 | -1,037,486 | 147,766 | 2,635,268 |
| Same, with CVaR | -5,685,451 | 43,640 | -406,330 | 125,021 | 2,657,974 |

Against daily delta, with 95% block-bootstrap intervals:

| Policy | Variance reduction | ES improvement | Mean P&L per period |
|---|---|---|---|
| Optimizer, delta and gamma | -853% (-2,443 to -328) | -$74,784 (-109,947 to -40,640) | -$712 (-2,847 to +1,171) |
| Optimizer, delta, gamma, vega | -1,738% (-6,053 to -421) | -$68,081 (-105,385 to -37,167) | -$852 (-3,657 to +2,527) |
| Same, lower risk aversion | -2,187% (-6,650 to -736) | -$101,558 (-183,077 to -49,356) | -$971 (-4,416 to +2,207) |
| Same, with CVaR | -1,337% (-3,483 to -557) | -$78,814 (-114,704 to -48,256) | **-$5,140 (-7,961 to -2,114)** |
| Delta, no-trade band | -14.5% (-59.8 to +7.9) | -$3,911 (-8,854 to +755) | -$20 (-292 to +253) |

At three times the quoted spreads the picture is the same, with costs higher still: `chain_backtest.json` has every variant.
At 3x the widened bid is floored at zero, so cheap wing quotes are charged less than three times their half-spread.
That makes the 3x costs of the option hedges a lower bound, which flatters them.
It cannot change the no-go, which is decided at quoted spreads.

**The one place the option hedges did what they are for is the crash.**
On the period ending 2020-03-16 the strangle lost $209,329 unhedged and $59,031 delta hedged, while the delta, gamma, and vega optimizer lost $1,850 and the CVaR variant made $11,355.
That is one of the two pre-registered stress dates, so it is an observation, not a result.

The other, 2024-08-05, has no period of its own: the source has no SPY chain on any weekday from 2024-07-30 to 2024-08-09, so no period ends on it and `stress_days` in `chain_backtest.json` omits it.
It is not one of the 34 skipped days; it is a gap in the source.
As a stand-in, this is the period that spans it, from 2024-07-29 to 2024-08-12, about two weeks, in dollars at quoted spreads.
It is one two-week period, so it is an observation, not a result.

| Book | Policy | P&L, 2024-07-29 to 2024-08-12 |
|---|---|---:|
| Short strangle | Unhedged | -47,833 |
| Short strangle | Delta, daily | +7,996 |
| Short strangle | Delta, no-trade band | -5,405 |
| Short strangle | Optimizer, delta and gamma | -89,045 |
| Short strangle | Optimizer, delta, gamma, vega | +284,866 |
| Short strangle | Same, lower risk aversion | +193,438 |
| Short strangle | Same, with CVaR | +105,311 |
| Long put overlay | Unhedged | -1,225,578 |
| Long put overlay | Delta, daily | -246,079 |
| Long put overlay | Optimizer, delta, gamma, vega | -207,567 |
| Long put overlay | Same, lower risk aversion | -130,955 |

The long put overlay is reported because it was pre-registered, though, as in the proxy, it is a poor test.
There the lower-risk-aversion optimizer did reduce variance against delta, by 48% (2 to 66), with no established difference in tail or mean P&L, while the higher-risk-aversion one did not, at 25% (-84 to +68).
Over five periods or more, the lower-risk-aversion one's interval also includes zero.

### Why the optimizer failed here, and how sure that is

**Costs are real and large.**
The optimizer variants paid $1.3 to $2.7 million in spreads and fees against $10,329 for daily delta, on a book that made $907,508 unhedged, trading 460 to 820 option contracts a period on average.
That is not a marking artifact: each trade paid a real or modeled half-spread plus the fee.
The source quotes strikes on a grid that moves with spot, so a contract held today is rarely in tomorrow's grid, and the optimizer keeps opening new legs near its targets rather than adding to old ones.
On a full listed chain some of that churn would go, but the policy has no penalty for holding many small legs, and it rebalances every period.

**Marks are mostly from the model, and their error is measured.**
The source quotes the same contract on the next day only 8% of the time, so 98.5% of held option positions, counted period by period, had at least one of their two marks from that day's calibrated surface.
A held-out audit over 230 days put the error of those marks at an RMS of $0.08 a share when the expiry was quoted but the strike was not, and $0.20 when the expiry was missing, against a median quoted half-spread of $0.005 to $0.01.
On the hundreds of option contracts the optimizer trades each period, errors of that size add P&L noise that real mids would not.
Treating each mark's error as independent puts that noise at 12% to over 100% of the optimizer's variance, depending on which error level applies, so that estimate cannot settle it.
The next check does.

**The excess variance is real hedge P&L, not mark noise.**
This check was added after the results were read, so it explains them and does not change the decision.
A wrong mark at the end of one period is the next period's start mark, so its error enters as +e and then -e, and summed over k periods it stays the size of one mark error while real P&L variance grows with k.
If marks drove the result, the optimizer's variance against delta would fall as k grows, and its period-to-period P&L difference from delta would be autocorrelated near -0.5.
Neither happens, in `chain_horizons.json`:

| Policy | 1 period | 5 periods | 10 periods | 20 periods | Lag-one autocorrelation |
|---|---:|---:|---:|---:|---:|
| Optimizer, delta and gamma | -853% | -884% | -643% | -587% | -0.01 |
| Optimizer, delta, gamma, vega | -1,738% | -1,744% | -1,381% | -1,658% | +0.03 |
| Same, lower risk aversion | -2,187% | -2,054% | -1,600% | -1,966% | +0.03 |
| Same, with CVaR | -1,337% | -1,206% | -1,227% | -1,502% | +0.01 |

Variance reduction against daily delta on sums over k periods, for the short strangle at quoted spreads.
Every interval is below zero at every horizon.

**The likely mechanism is surface-shape risk.**
The targets bound the book's aggregate delta, gamma, and vega, but nothing bounds the hedge's gross size.
So the optimizer can build a large set of offsetting options that is flat on those three Greeks and exposed to changes in skew and term structure, which real surfaces have and the proxy's fixed-shape smile did not.
That fits the proxy's 91% reduction and the failure here, but this run does not measure it directly.

**The test itself was checked.**
An earlier run of this backtest was invalid: on a machine starved of CPU, solves hit the solver's 30 second wall-clock backstop and the backtest traded whatever the search had found, so results depended on machine load.
The backtest now refuses such a solve, and the registered backstop is 600 seconds, so the deterministic node limit governs.
In the current run no solve hit it.
Every optimizer variant but one matches, to the dollar, an earlier run made while that variant had the machine to itself, and the exception was load-affected in both earlier runs.
See `ledger.md`.

### What run 3 said would change the decision

- **Fix the policy, then rerun as a new pre-registered variant.** Bound the hedge's gross option notional or penalize it, add skew and term-structure scenarios to the risk the optimizer sees, penalize the number of held legs, and rebalance options on a band rather than every period.
  This was done, as above, except a penalty on the number of held legs: pricing each held leg's exit and capping gross contracts did that job.
- **Better data.** A licensed chain with every listed strike, such as Cboe DataShop, ORATS, or OptionMetrics, would let held positions be marked at observed mids and would remove the strike-grid churn. That would also test whether the crash-day protection holds up across more stress periods.
  This remains open.

## Reproduce the chain runs

```sh
curl -L 'https://fred.stlouisfed.org/graph/fredgraph.csv?id=DGS3MO' -o vendor_data/DGS3MO.csv
for sym in SPY DIA XLF; do
  dir=vendor_data/dolthub_$sym
  uv run python scripts/fetch_dolthub_chains.py $dir --symbol $sym
  uv run python scripts/run_chain_backtest.py $dir/chains/$sym $dir/${sym}_dividends.csv \
    vendor_data/DGS3MO.csv --symbol $sym --out $sym.json --series series_$sym/
  uv run python scripts/analyze_chain_series.py series_$sym/ --out ${sym}_horizons.json
done
```

Each download takes one to two hours, since each day needs its own query.
The registered runs probed every weekday, which is the fetch script's default.
Only a full 25-variant run is a registered result.
`--variants` runs a subset for diagnostics, and its results file records how many ran in `variants_run`.
A full run takes about two hours on 8 processes, most of it in the first optimizer variants.
Run 3 had 16 variants and no gap rule; `chain_backtest.json` and `chain_horizons.json` are its files.
DoltHub and FRED can revise their data, so a later download can differ.

## Proxy backtest, 2016 to 2026, with modeled option prices

This was the evidence before the chain run, and the chain run supersedes it.
It is kept because it shows how far a modeled market can mislead: the same policies that cut variance by 91% here increased it on real quotes.

### What this is, and is not

Spot, VIX, and rates are real: FRED SP500, VIXCLS, and DGS3MO, from 2016-10-31 to 2026-10-29, over 2,513 days.
The first 252 days are warm-up, which leaves 2,260 measured.

Option prices are **modeled**: a stylized smile anchored to the VIX, with spreads from an assumed model.
So this market has no quote noise, no skew dynamics, and no dislocations.
Real quotes have all three, and they hurt option hedges most.
**Results for option hedges are probably overstated.**
The results do not rule that bias out, and nothing here measures its size.

Every day's decision sees only that day's data, and this is tested.

### Main result: short strangle, 10 lots, rolled monthly

Dollars, over the 2,260 measured days, with spreads as modeled:

| Policy | Total P&L | Daily stdev | Worst day | ES 97.5% | Hedge cost |
|---|---|---:|---:|---:|---:|
| Unhedged | +1,533,830 | 17,981 | -214,972 | 68,224 | 0 |
| Delta, daily | +843,170 | 8,629 | -121,703 | 35,752 | 32,487 |
| Delta, no-trade band | +1,073,010 | 9,847 | -132,069 | 40,004 | 10,995 |
| Optimizer, delta and gamma | -487,373 | 4,928 | -70,315 | 18,856 | 181,495 |
| Optimizer, delta, gamma, vega | -697,259 | 2,575 | -47,612 | 8,577 | 286,031 |
| Same, lower risk aversion | -542,569 | 2,785 | -36,915 | 8,563 | 227,508 |
| Same, with CVaR | -113,247 | 2,991 | -14,241 | 4,916 | 234,715 |

Against daily delta hedging, with 95% block-bootstrap intervals:

| Policy | Variance reduction | ES improvement | Mean P&L per day |
|---|---|---|---|
| Optimizer, delta, gamma, vega | 91.1% (85.1 to 95.1) | +$27,175 (20,123 to 35,207) | **-$682 (-975 to -348)** |
| Same, lower risk aversion | 89.6% (82.6 to 94.4) | +$27,189 (20,359 to 34,479) | **-$613 (-909 to -281)** |
| Same, with CVaR | 88.0% (73.3 to 96.3) | +$30,836 (22,273 to 40,931) | -$423 (-801 to +19) |
| Delta, no-trade band | -30.2% (-42.6 to -19.4) | -$4,252 (-6,688 to -1,763) | +$102 (-38 to +233) |

### How to read this

**Risk reduction is large, and survives higher costs.**
At three times the assumed spreads, variance reduction against delta is still 84.9% (71.4 to 93.3).

**It costs real money.**
Unhedged, this book made money over the decade, because short options earn a volatility premium in this market.
A neutral hedge gives up that premium and pays spreads, so its total P&L is negative.
**This is not a way to make a short-vol book more profitable.**
It trades carry for a much narrower distribution, and whether that is worth it depends on the mandate.

**The no-trade band did not clearly beat daily delta.**
It saved cost, but it raised variance and tail loss, and its P&L gain is not statistically established.
Its parameters were set once, and were not tuned.

**Stress days**, in dollars, unhedged and then delta and then the CVaR policy:

| Day | Unhedged | Delta | Optimizer with CVaR |
|---|---:|---:|---:|
| 2018-02-05 | -65,013 | -65,516 | +7,928 |
| 2020-03-16 | -190,888 | -121,703 | +14,609 |
| 2024-08-05 | -141,567 | -49,453 | +8,602 |

These three days are in the data but were picked in advance as known stress dates.
They are not a test of a crash the policy has never seen, and CVaR's scenario list names them too: it contains these very moves.
**That makes the CVaR policy's stress-day results especially likely to flatter it.**

### Selection

All 16 variants were fixed in `scripts/run_proxy_backtest.py` before the first run, and are all in `proxy_backtest.json`.
The CVaR variant has the best tail figures, but it is the best of several, with no adjustment for that.
Its P&L difference from delta is not established at 95%, and its interval on variance reduction is the widest.
See `ledger.md`.

### The overlay book is not a good test

`long_put_overlay` holds 10,000 index units and a protective put.
Its unhedged P&L is almost entirely equity beta, which a delta-neutral hedge removes by design, so "variance reduction" there means little.
It is reported because it was pre-registered.
A real overlay would target the put's Greeks, and leave the equity exposure alone.

### Reproduce the proxy run

```sh
curl -L 'https://fred.stlouisfed.org/graph/fredgraph.csv?id=SP500' -o SP500.csv
curl -L 'https://fred.stlouisfed.org/graph/fredgraph.csv?id=VIXCLS' -o VIXCLS.csv
curl -L 'https://fred.stlouisfed.org/graph/fredgraph.csv?id=DGS3MO' -o DGS3MO.csv
uv run python scripts/run_proxy_backtest.py SP500.csv VIXCLS.csv DGS3MO.csv --out results.json
```

FRED revises data, so a later download can differ.
The search stops at a node count rather than a clock, so the same data and code give the same result.
