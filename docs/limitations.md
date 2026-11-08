# Known limitations

These matter as much as the features.
This is alpha software, and **it has not been used to hedge real money.**

## Evidence

- **The option-hedge optimizer has not shown it beats daily delta robustly.** As first configured it failed its historical-chain backtest: on free SPY chains it had many times the variance of daily delta hedging and paid far more in costs.
  With `price_exit` and a gross option cap it passed a pre-registered held-out test on DIA, but XLF did not confirm the tail improvement, and the DIA result weakens at three times quoted spreads, on the put overlay, and over two-month horizons.
  It also gives up most of a short strangle's carry.
  Use those settings for any repeated re-hedge, and treat it as a paper-pilot candidate, not a default.
  Most marks in these runs came from the calibrated surface, because the free source rarely quotes the same contract twice, and the audit of those marks is in `docs/evidence/phase4/`.
- **The position terms need the hedger's holdings, and nothing here tracks them.** `price_exit` and `max_gross_option_contracts` act on the held hedges and the trade.
  A request names its held hedges in `held_hedges`, apart from the client's `book`.
  The service keeps no positions, so a caller that leaves them out, or sends stale ones, gets terms on the new trade alone, and nothing flags it.
- **No live or paper pilot, and no reconciliation to broker or clearing statements.**
- **No independent validation.** Every review here was automated, and only one day's real chain has been reconciled.

## Models

- Parametric surfaces fit inside the quoted market for only about 35% of SPX and 29% of SPY contracts. Costs therefore come from live quotes, never the surface.
- Scenario P&L over a horizon includes discrete dividends, but not financing or continuous carry income.
- Volatility dynamics are simple: scenarios shift a surface, and do not simulate it. Heston, local vol, SLV, and Monte Carlo are **not implemented**, so path-dependent and barrier options are not supported.
- Dividends use the escrowed model. Desks using a spot-vol model will see differences of about 0.12 on the tested calls.
- American veta, speed, zomma, and color are European approximations. They are not in position exposures.
- The tree uses one flat rate per contract, with dividends restated to their curve PV.
- Sticky-strike shifts with dividends assume the default calendar clock.
- Expired options settle at the next snapshot's spot, so an AM-settled option uses that day's close in a backtest, rather than its opening settlement value.

## Asset classes

Phase 7 is **not started**.
US listed equity, ETF, and index options work today, with delta-one and futures as hedges.
Every instrument must be in USD, and any other currency is rejected at construction, because totals are in dollars and there is no FX conversion.

| Not implemented | Note |
|---|---|
| Options on futures | Black-76 pricing is not wired up as a product |
| FX | No delta conventions, or risk-reversal and butterfly quotes |
| Rates | No Bachelier, SABR, swaptions, or caps |
| Commodities and crypto | Coin-settled payoffs are not supported |
| Variance swaps | |

## Margin

Estimates only, for comparison. SPAN, SPAN 2, STANS, TIMS official values, and SIMM are not implemented: see `docs/risk/margin.md`.
FRTB sensitivity export is not implemented.

## Data

- The only vendor adapter parses Cboe's delayed public endpoint. It is for development, with no streaming, no licensed real-time feed, and no bitemporal store.
- Each quote gets the chain's timestamp, so per-row staleness cannot be detected from that source.

## Product

- Not implemented: Excel add-in, OMS and PMS position import, FIX drop copy, broker APIs, live or paper execution, post-trade slippage analysis, and hedge accounting export.
- Tickets are generated but have not been certified with any broker.
- Limit monitoring reports breaches when asked, against limits in the request. There is no stored limit set, continuous monitoring, alerting, acknowledgement, escalation, breach history, or blocking of a proposal.
- Liquidity estimates use the spread and daily volume the caller supplies. There is no volume feed or market impact model behind them.
- The expiry ladder, limit checks, liquidity estimates, and hedge decay forecasts are in the API and CLI only, not the web UI.
- Security gaps are listed in `docs/operations/security.md`, including SSO, RBAC, tenant isolation, and a durable store. No certification is claimed.
- `exchange_calendars` is optional, and the default session predicate does not know holidays.

## Numerics and performance

- The search stops at a fixed node count, so a hard problem gets a good answer rather than a proven best one. The gap to the LP bound is reported.
- Replay is exact only without a `SOLVER_TIME_LIMIT` issue, and on the same library versions, which the manifest records, since HiGHS and SciPy updates can change results. Other architectures are unverified.
- A search that reaches its limit with no hedge returns a `search_limit` error, with no answer about whether a hedge exists.
- SciPy's bundled HiGHS prints `HighsMipSolverData::transformNewIntegerFeasibleSolution tmpSolver.run();` to stdout on some integer solves, even with `disp=False`. It is an upstream debug print and does not affect results. It is not suppressed, because that would mean redirecting the process's stdout around a native call, which is unsafe in the threaded service.
- With `max_legs` or a ticket cost, the reported gap is against a weak bound and overstates the true shortfall.
- Inputs past what is plausible are refused rather than valued. A carry yield is limited to 100% a year either way, as rates are, because 2.0 is far more often a mistyped 2% than a real rate, so a real stock borrow fee above 100% a year cannot be entered. Spot is limited to 1e9, a position to 1e12 units, and an objective cost to 1e15.
- A proxy is marked at spot times its delta ratio, flagged by a `PROXY_MARK` issue on every valuation holding one, because a book position carries no quote. Reported value is wrong for a proxy whose price is not that, though its delta and scenario P&L are right. In a backtest that mark is also what a proxy is financed on, so a triple-leveraged fund is financed on three times its value.
- A proxy's quote is not compared with the market, since it has no reference to compare with. A wrong proxy quote gets through, flagged only by the `QUOTE_UNCHECKED` issue on a proposal using it.
- Hedges were measured reaching their caps with positions up to 1e12 units at spots from 100 to 1e9, on a stock-only sample and, at a spot of 100, an option universe. A CVaR request is refused once a scenario loss reaches 1e20, which took a spot of 1e8 at 1e12 units there. That is the sample's range, not a guarantee for every book.
- Budgets in `docs/evidence/performance/` are provisional targets. Few repeats were run, so `p99` there is not a real tail measurement.
