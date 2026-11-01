# Performance

Measured by `scripts/benchmark.py`, in one process on a 12-core Apple Silicon laptop, with results in `benchmark.json`.
Budgets are provisional targets from the product brief, not requirements anyone has validated.

| Case | p50 | Budget | Result |
|---|---:|---|---|
| Vectorized kernel, 100k options | 9 ms | | |
| Value 100k European contracts | 463 ms | Under 1 second | **Met** |
| Value 5k European contracts | 24 ms | | |
| Value 1k American contracts, with bump Greeks | 310 ms | | |
| Value 5k American contracts, with bump Greeks | 1,463 ms | | See below |
| Stress 5k positions across 7 scenarios | 193 ms | | |
| Hedge over 100 candidates | 31 ms | | |
| Hedge over 1,000 candidates | 143 ms | | |
| Hedge over 5,000 candidates | 581 ms | Under 1 second at p99 | **Met on p50**, see below |
| Hedge over 100 candidates, with stability | 143 ms | | |

## What these do not show

- **No real tail.** Each case ran 2 to 20 times, so the `p99` in the JSON is close to the maximum, and it is not a true p99.
  The maximum on the 5,000-candidate hedge was 588 ms.
- **Real universes are harder.** On the real SPX chain, 2,254 candidates with `max_legs` and stability take about 5 seconds, against 41 seconds before the search moved to a node limit.
- **The "live book risk update under 250 ms at p99" budget has no direct measure,** since there is no streaming path yet. The 5k-position stress is only an indication.
- **Concurrent load has not been tested.**

## Decisions

A 5x gain came from walking only held legs in the solver's evaluator, found by profiling.
Compiling kernels is not justified by these results, under ADR 1.
Valuing 5k American contracts takes 1.5 seconds, so a large single-name book may be the first thing to justify it.
Deduplicating repeated contracts is the cheaper next step, and has not been done.
