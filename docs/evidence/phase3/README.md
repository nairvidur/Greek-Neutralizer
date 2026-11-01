# Phase 3 evidence: optimizer and risk

Gate: "independent full-revaluation verification of all proposals, and stress results reviewed."

**Status: met, with the limitations below.**

## Real SPX hedge

`hedge_SPX.json` holds the results for a short 50-lot strangle at about 35 days, hedged from 2,254 real quoted candidates at the 2026-10-30 close.

```sh
uv run python scripts/hedge_real_chain.py vendor_data/_SPX.json --rate 0.0482
```

| Check | Result |
|---|---|
| Book value of the hedged book against QuantLib | Difference about $5e-9 on $397,596 |
| Delta, gamma, and vega against QuantLib | Differences of 1e-11 or smaller |
| Internal repricing verification | Passed |
| Objective against the LP lower bound | 2,438 against 2,337, at most 4.1% from integer lots and leg limits |
| Trade movement under small input changes | 87% of contracts |
| Regret from holding it anyway | 5.9% of the hedge's benefit, so it is rated stable |
| Replay | Byte-identical across two runs, one with all 12 cores saturated by other work |
| Candidates excluded | 40, with reasons |

`hedge_SPX.json` carries a `GATE_SETTINGS` issue because this script allows a 30-day quote age, so that a saved chain can be replayed.
A live service refuses that setting by default.

QuantLib is a separate implementation, so the agreement checks arithmetic independently.
It uses the same vols and rates, so it cannot detect a wrong input or assumption.

## Stress results

Before and after, in dollars, for that book:

| Finding | Value |
|---|---|
| Worst scenario unhedged (1987) | -$6.81M |
| Worst scenario after the Greek-neutral hedge | -$2.79M |
| Delta-only baseline in the same scenario | -$6.04M |
| Largest Taylor error, after the hedge | $3.77M |

**Greek-neutral is not crash-proof.** The hedge is flat at spot, yet it still loses $2.79M in the worst scenario.
The Taylor error shows why stress must reprice: Greeks cannot describe moves of that size.

## What testing found

This is what the evidence process surfaced in the optimizer, in the order it was found:

1. A frontier that was not monotone in risk aversion. Screening with LP reduced costs fixed it.
2. Numerical warnings from badly scaled rows. Row scaling fixed it.
3. A constrained problem scoring better than the unconstrained one, which is impossible at an optimum. A local polish fixed it.
4. Stability reporting `0.0` because comparisons were skipped when candidate sets differed. The real figure was about 80%.
5. **Results that depended on machine load,** because the search stopped on a wall-clock limit. The same hedge reported 81% and then 132% trade movement. It now stops at a node count, which replays exactly, is about 7 times faster here, and costs about 0.8% in objective.
6. **CVaR farming scenario profit:** 11.6 billion lots and a modeled $10 trillion loss in the first backtest. CVaR is now floored at zero, and legs have a hard cap.

## Independent review

Two fresh-context reviews found 47 defects, all with reproductions.
`review_red.txt` lists the numerical tests failing before their fixes, and these are pinned in `tests/test_review_findings.py` and `tests/test_fail_closed.py`.
The review also corrected claims in this repository: a gamma weight 3x too small, and an inflated Taylor error.

## Limitations

- One book, one day, one instant.
- Stress vol and skew shocks are partly judgment: see `docs/risk/stress.md`.
- Quote sizes for SPX are absent from the source, so this run assumes 200 displayed per candidate.
- At 2,254 candidates with stability, this takes about 5 seconds: see `docs/evidence/performance/`.
- Margin before and after the hedge reads the same on the Reg T estimate, since it ignores offsets.
