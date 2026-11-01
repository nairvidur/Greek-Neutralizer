# Phase 1 evidence: foundations

Gate: "a clean install from a built wheel works, and every listed defect has a regression test."

**Status: met.**

- `red_legacy.txt` is the output of 12 baseline defects failing against the original engine, before any fix.
- `tests/test_regressions.py` and `tests/test_hedging.py` pin each behavior against the new engine.
- The CI `wheel` job builds a wheel, installs it into a clean environment away from the source tree, and runs a verified hedge. The same steps were run by hand on Python 3.11.
