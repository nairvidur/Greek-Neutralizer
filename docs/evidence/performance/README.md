# Performance

Measured by `scripts/benchmark.py`, in one process on a 12-core Apple Silicon laptop, with results in `benchmark.json`.
Budgets are provisional targets from the product brief, not requirements anyone has validated.

| Case | p50 | Budget | Result |
|---|---:|---|---|
| Vectorized kernel, 100k options | 10 ms | | |
| Value 100k European contracts | 496 ms | Under 1 second | **Met** |
| Value 5k European contracts | 25 ms | | |
| Value 1k American contracts, with bump Greeks | 337 ms | | |
| Value 5k American contracts, with bump Greeks | 1,669 ms | | See below |
| Stress 5k positions across 7 scenarios | 209 ms | | |
| Hedge over 100 candidates | 33 ms | | |
| Hedge over 1,000 candidates | 156 ms | | |
| Hedge over 5,000 candidates | 661 ms | Under 1 second at p99 | **Met on p50**, see below |
| Hedge over 100 candidates, with stability | 180 ms | | |

## What these do not show

- **No real tail.** Each case ran 2 to 20 times, so the `p99` in the JSON is close to the maximum, and it is not a true p99.
  The maximum on the 5,000-candidate hedge was 680 ms.
- **Synthetic universes are easier than real ones.** On the real SPX chain, 2,254 candidates with `max_legs` and stability took about 41 seconds, with the machine under load from a backtest, because each of the seven solves can reach its 1-second limit.
- **The "live book risk update under 250 ms at p99" budget has no direct measure,** since there is no streaming path yet. The 5k-position stress is only an indication.
- **Concurrent load has not been tested.**

## Decisions

A 5x gain came from walking only held legs in the solver's evaluator, found by profiling.
Compiling kernels is not justified by these results, under ADR 1.
Valuing 5k American contracts takes 1.7 seconds, so a large single-name book may be the first thing to justify it.
Deduplicating repeated contracts is the cheaper next step, and has not been done.
