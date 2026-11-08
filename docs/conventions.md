# Conventions

Every number the system reports follows these definitions.
`pricing/book.py` holds the single implementation of position scaling.

## Signs and quantity

`Position.quantity` is a signed integer, and positive means long.
There is no separate side field, so sign cannot disagree with side.
All exposures are from the holder's perspective.

## Time

| Symbol | Meaning | Used for |
|---|---|---|
| `T` | Calendar years, ACT/365, from the snapshot to the expiry instant | Discounting and carry |
| `tau` | Variance years, from the market's `VarianceClock` | Total stdev, `sigma * sqrt(tau)` |

The default clock has `tau == T`, which is the quoting convention most vendors assume.
Implied vols and pricing must use the same clock, so the clock travels with the market snapshot.

The expiry instant is when the value is fixed: 16:00 ET for PM settlement and 09:30 ET for AM settlement.
All timestamps must be timezone-aware, and naive datetimes are rejected.

## Per-share Greeks

From `pricing.black.black_greeks`, in natural units:

| Greek | Definition |
|---|---|
| `delta` | dV/dS |
| `gamma` | d2V/dS2 |
| `vega` | dV/dsigma, per 1.00 of vol |
| `theta` | dV/dt, per calendar year, with spot held fixed |
| `rho` | dV/dr, per 1.00 of rate |
| `phi` | dV/dq, per 1.00 of carry yield |
| `vanna` | d2V/dS dsigma |
| `volga` | d2V/dsigma2 |
| `charm` | d(delta)/dt, per year |
| `veta` | d(vega)/dt, per year |
| `speed` | d(gamma)/dS |
| `zomma` | d(gamma)/dsigma |
| `color` | d(gamma)/dt, per year |

## Position exposures

From `pricing.book.Exposure`, for the whole position, with `size = quantity * multiplier`:

| Field | Definition |
|---|---|
| `value` | `size * price`. Zero for a future, which has no value at inception. A proxy's price here is spot times its delta ratio, which is its exposure and not its market price, and any valuation holding one carries a `PROXY_MARK` issue. |
| `delta_shares` | `size * delta` |
| `delta_dollars` | `delta_shares * spot` |
| `gamma_shares` | `size * gamma`: change in `delta_shares` per $1 of spot |
| `gamma_1pct` | `gamma_shares * spot^2 * 0.01`: change in `delta_dollars` for a +1% move |
| `vega_1pt` | `size * vega * 0.01`: P&L per +1 vol point |
| `theta_1d` | `size * theta / 365`: P&L per calendar day |
| `rho_1pct` | `size * rho * 0.01`: P&L per +1% rate |
| `vanna_1pt` | `size * vanna * 0.01`: change in `delta_shares` per +1 vol point |
| `volga_1pt` | `size * volga * 1e-4`: change in `vega_1pt` per +1 vol point |
| `charm_1d` | `size * charm / 365`: change in `delta_shares` per calendar day |

Instantaneous theta understates decay into the close on a 0DTE option.
Use a `days_forward` scenario to measure a real roll.

## Moneyness

Log-moneyness is `k = ln(K / F)`, against the forward for that expiry, and not against spot.

## Hedge objective

All in dollars:

```text
transaction cost
+ risk_aversion * sum(weight * |residual exposure|)
+ cvar_weight * CVaR(scenario loss)
+ risk_aversion * proxy basis risk
```

Default weights express a residual as the dollar P&L under a typical shock:

| Greek | Default weight | Meaning |
|---|---|---|
| `delta_dollars` | 0.01 | a 1% spot move |
| `gamma_1pct` | 0.045 | second-order P&L on a 3% move |
| `vega_1pt` | 2.0 | a 2 vol point move |
| `theta_1d` | 1.0 | one day of decay |
| `rho_1pct` | 0.25 | a 25 bp rate move |

These are judgment values and are meant to be set by the desk.

Premium paid at mid is not a cost, because it buys an asset worth that much.
The cost of a hedge is the half-spread, fees, and impact.

By default the objective prices only the trade, not the position it leaves.
Re-hedged every day, that stacks offsetting legs: on historical SPY chains it held 1,313 contracts against a 200-contract book, flat in net Greeks but exposed to skew and term structure.
Two position terms, off by default, address it:

- `price_exit` adds each candidate's half-spread and fee on the hedger's position in it after the trade, the cost of closing it later.
  Selling a held hedge then saves an exit, while opening an offsetting one adds a second.
- `max_gross_option_contracts` caps the absolute option contracts of the hedger's position after the trade.
  Held hedge options that are not candidates count against it as they stand, so a cap below them is infeasible and is reported as over the limit by that much.

Both act on the problem's `held_hedges`, the hedger's own positions, which the book also holds, plus the trade.
A hedge request lists them in `held_hedges`, apart from the client's `book`, and the problem holds both: the Greeks, scenarios, and verification see the whole position.
They never act on the rest of the book, the client's: a hedge that offsets a client position does not close it, so the two are not netted.
With no `held_hedges`, as in a one-off request, they act on the trade alone: the exit doubles the cost of what is opened, and the cap limits it.
The exit cost on held hedges that do not trade is a constant, so it shifts the objective and dilutes the relative MIP gap, but it does not change the hedge.
