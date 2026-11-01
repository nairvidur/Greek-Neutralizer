# Performance

Measured by `scripts/benchmark.py`, in one process on a 12-core Apple Silicon laptop, with results in `benchmark.json`.
Budgets are provisional targets from the product brief, not requirements anyone has validated.

| Case | p50 | Budget | Result |
|---|---:|---|---|
| Vectorized kernel, 100k options | 8 ms | | |
| Value 100k European contracts | 459 ms | Under 1 second | **Met** |
| Value 5k European contracts | 23 ms | | |
| Value 1k American contracts, with bump Greeks | 301 ms | | |
| Value 5k American contracts, with bump Greeks | 1,609 ms | | See below |
| Stress 5k positions across 7 scenarios | 195 ms | | |
| Hedge over 100 candidates | 30 ms | | |
| Hedge over 1,000 candidates | 148 ms | | |
| Hedge over 5,000 candidates | 594 ms | Under 1 second at p99 | **Met on p50**, see below |
| Hedge over 100 candidates, with stability | 145 ms | | |

## What these do not show

- **No real tail.** Each case ran 2 to 20 times, so the `p99` in the JSON is close to the maximum, and it is not a true p99.
  The maximum on the 5,000-candidate hedge was 608 ms.
- **Real universes are harder.** On the real SPX chain, 2,254 candidates with `max_legs` and stability take about 5 seconds, against 41 seconds before the search moved to a node limit.
- **The "live book risk update under 250 ms at p99" budget has no direct measure,** since there is no streaming path yet. The 5k-position stress is only an indication.
- **Concurrent load is measured separately,** over HTTP in `docs/evidence/phase5/load_test.json`, and only on a small sample request.

## Decisions

A 5x gain came from walking only held legs in the solver's evaluator, found by profiling.
Compiling kernels is not justified by these results, under ADR 1.
Valuing 5k American contracts takes 1.6 seconds, so a large single-name book may be the first thing to justify it.
Deduplicating repeated contracts is the cheaper next step, and has not been done.
