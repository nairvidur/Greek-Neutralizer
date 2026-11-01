# Methodology

This describes what each model does, what it assumes, and how it was checked.
Evidence is in `docs/evidence/`, and known gaps are in `docs/limitations.md`.

## European pricing

Generalized Black-Scholes on the forward, with `F = (S - PV(dividends)) * exp((r - q) T)`.

The premium is computed from the out-of-the-money side and adds discounted intrinsic if in the money.
Deep wings use the scaled complementary error function, which keeps relative accuracy where the textbook difference of two CDFs cancels to noise and can go negative.

Limits are explicit.
Zero vol gives the discounted forward intrinsic, and zero time gives spot intrinsic.
Invalid inputs raise rather than propagate.

## Implied volatility

Newton's method on the normalized OTM premium, which is monotone in total stdev with one inflection at `sqrt(2 |x|)`.
It starts from the inflection, works in log space for tiny premiums, and keeps a bracket with bisection as a fallback.
Each element reports a status, so an unattainable price comes back as such rather than as a wrong vol.

## American options

A Leisen-Reimer binomial tree on the escrowed spot, vectorized across options, with the European closed form as control variate.

- **No dividends:** the premium converges smoothly at O(1/n), measured against QuantLib's ALO engine, so premiums at n and n/2 are Richardson-extrapolated.
- **Discrete dividends:** the error is not smooth in n, since node placement around the exercise boundary dominates, so a single finer tree is used.
  Each dividend also gets the exact value of the strategy of exercising just before the ex-date, which usually falls between tree steps.
- **Routing:** where early exercise has no value, the closed form is exact and is used directly.

Dividends use the **escrowed** model: volatility applies to spot less the PV of dividends.
This is an assumption.
QuantLib's default `Spot` model prices the same test calls about 0.12 higher.

Delta, gamma, vega, theta, rho, vanna, volga, and charm come from bumps through the control-variate price.
Veta, speed, zomma, and color stay European approximations for American options.

## Surfaces

Raw SVI per expiry, fit by weighted least squares in vol, where each residual is weighted by vega over spread.
Between slices, total variance interpolates linearly at fixed log-moneyness.

Calibration penalizes butterfly arbitrage (Durrleman's condition) and calendar crossings, the latter only where both slices are quoted.
Arbitrage is then measured, because a penalty does not guarantee its absence.
`quoted_arbitrage_report` is the gate, and it covers only quoted strikes.
The book reports `EXTRAPOLATED_VOL` for anything beyond them.

## Forwards and rates

The forward at each expiry comes from put-call parity near the money.
The discount rate is an input, since regressing it from American quotes is biased by early exercise.
`box_implied_rate` derives a rate from European chains only.
Spot for a chain comes from the nearest parity forward, because the last index print goes stale against option quotes.

## Hedge optimizer

A mixed-integer linear program solved in stages:

1. **Filter.** Candidates failing the quote gates, expired, or beyond the surface's quotes are excluded, with reasons reported.
2. **Relax.** The LP relaxation gives a true lower bound on the objective and decides feasibility.
3. **Screen.** Reduced costs select the candidates that can matter, at most 40 beyond the LP's own picks.
4. **Search.** HiGHS searches integer lots in a window around the LP solution that can always drop a leg, stopping at a fixed node count so that machine load does not change the result. If rounding a share leg cannot meet a band, shares are branched on too. Reaching that count with no hedge is reported as its own error, and only a solve over whole lots and shares at full capacity can report infeasibility, and with an instrument unseen by the solver only for a band or budget short by more than all such instruments could reach.
5. **Polish.** A deterministic local search over lot moves takes only strict improvements.
6. **Verify.** The hedged book is repriced from scratch, and a mismatch raises instead of returning a proposal.

Fine-grained linear hedges with a multiplier of 1 are solved as continuous and then rounded.
The reported gap compares the final objective with the LP bound, so an incumbent at the time limit cannot pass as optimal.
With `max_legs` or a ticket cost, the LP bound is weak because binaries relax, and the gap overstates the true shortfall.

CVaR uses the Rockafellar-Uryasev linearization over full-revaluation scenarios, floored at zero, so protection earns nothing once the tail loss is gone.
Every contract-sized leg has a hard cap, `max_lots_per_leg`, regardless of the objective.

Before any of this, candidate quotes are compared with the model.
A stock or future quote is compared with what the market implies instead: spot times its delta ratio, or the forward for a future, using the market's own carry.
A stock or future's mid must be within 25% of that, or the candidate is excluded with the reason, and with nothing else usable the status is `no_usable_candidates`.
The 25% is judgment, not measurement.
It has to catch a wrong spot, a stale or wrong-symbol quote, or a unit mistake, such as a stock at 200 or 10 against a spot of 100, while a correct quote must never trip it.
Spot and quotes are near-simultaneous, since quotes older than the age gate are already out, so only a violent move inside that window separates them.
A single-stock gap or a stale index print could reach 5%, and the market-wide circuit breaker sits at 20%, so a tighter figure could refuse a correct quote.
A smaller error slips through, so this is a check on gross mistakes rather than a pricing control.
A proxy skips this, because it has no market reference: a leveraged fund at 200 with a delta ratio of 0.1 on a spot of 6000 is correctly priced, and its price says nothing about its exposure.
A proposal using a proxy carries a `QUOTE_UNCHECKED` issue naming it, so the approver knows.
A single far-off quote is excluded, and broad disagreement raises, since it points to bad market data such as a vol in the wrong units.
Repricing cannot catch that, because it uses the same market.

## Stability

Each perturbation re-solves in a slightly different market: spot +/-0.25%, vol +/-1 point, and skew +/-0.02.

- `lot_change_ratio`: how far the option trade moves, as a share of its contracts.
- `regret_ratio`: how much worse today's hedge is than the re-solved one in that market, as a share of the benefit of hedging.
- `condition_number`: for inspection only.

A proposal is `unstable` if the trade moves and the move costs real benefit, or if under any perturbation the held hedge is no longer feasible.
Near-substitutes can swap freely: on one expiry, gamma and vega are exactly proportional, so a sensible hedge can have a singular Greek matrix.
On the reconciled SPX hedge the trade moves 87% of its contracts, but holding it gives up only 5.9% of the benefit.

## Scenarios and risk

Every scenario reprices the book in full.
Vol shocks are set at a 30-day reference and scale with tenor as `(30d / tau)^power`, capped at 3 times.
`taylor_vs_full` shows how far Greeks are from the truth, using the vol change each scenario actually applies to each contract: on the SPX hedge the second-order expansion is wrong by up to $3.8M in the crash scenarios, which is why stress uses repricing.

VaR is historical simulation, ES averages the tail, and Kupiec's test backtests exceptions.
Attribution decomposes P&L by start-of-period Greeks, and unexplained P&L is the control metric to track daily.

## Backtesting

Decisions at day t see only day t's market, which is structural and tested.
Trades pay half the spread plus fees, marks are at mid, and each policy finances its own marked value.
Comparisons use a circular block bootstrap of 20 days, so clustered days are resampled together.
All variants are declared before running and are all reported.
