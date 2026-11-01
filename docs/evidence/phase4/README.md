# Phase 4 evidence: does hedging with this work?

Gate: "a published out-of-sample report net of costs, including a go/no-go decision."

**Gate status: OPEN. It needs a backtest on historical option chains, which needs licensed data.**
The results below are proxy evidence only, and they do not satisfy the gate.

**Decision on proxy evidence: conditional go on risk reduction, and no claim on P&L.**

## What this is, and is not

Spot, VIX, and rates are real: FRED SP500, VIXCLS, and DGS3MO, from 2016-10-31 to 2026-10-29, over 2,513 days.
The first 252 days are warm-up, which leaves 2,260 measured.

Option prices are **modeled**: a stylized smile anchored to the VIX, with spreads from an assumed model.
So this market has no quote noise, no skew dynamics, and no dislocations.
Real quotes have all three, and they hurt option hedges most.
**Results for option hedges are probably overstated.**
The results do not rule that bias out, and nothing here measures its size.

Every day's decision sees only that day's data, and this is tested.

## Main result: short strangle, 10 lots, rolled monthly

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

## How to read this

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

## Selection

All 16 variants were fixed in `scripts/run_proxy_backtest.py` before the first run, and are all in `proxy_backtest.json`.
The CVaR variant has the best tail figures, but it is the best of several, with no adjustment for that.
Its P&L difference from delta is not established at 95%, and its interval on variance reduction is the widest.
See `ledger.md`.

## The overlay book is not a good test

`long_put_overlay` holds 10,000 index units and a protective put.
Its unhedged P&L is almost entirely equity beta, which a delta-neutral hedge removes by design, so "variance reduction" there means little.
It is reported because it was pre-registered.
A real overlay would target the put's Greeks, and leave the equity exposure alone.

## To close the gate

Run this same engine on historical chains.
Implement `DayMarket` from a licensed source, such as OptionMetrics, ORATS, Cboe DataShop, or Databento, and rerun the variants.
The engine and report need no changes.

The decision that follows from that run is:

- **Go** if option-hedge variance and tail reduction against delta still clear zero after real spreads.
- **No-go, fix the model** if they do not.

## Reproduce

```sh
curl -L 'https://fred.stlouisfed.org/graph/fredgraph.csv?id=SP500' -o SP500.csv
curl -L 'https://fred.stlouisfed.org/graph/fredgraph.csv?id=VIXCLS' -o VIXCLS.csv
curl -L 'https://fred.stlouisfed.org/graph/fredgraph.csv?id=DGS3MO' -o DGS3MO.csv
uv run python scripts/run_proxy_backtest.py SP500.csv VIXCLS.csv DGS3MO.csv --out results.json
```

FRED revises data, so a later download can differ.
The search stops at a node count rather than a clock, so the same data and code give the same result.
