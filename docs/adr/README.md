# Architecture decisions

## 1. Python and NumPy core, compiled only where measured

**Decision:** a typed NumPy core, with kernels moved to C++ or Rust only where a benchmark misses its budget.
**Why:** batch pricing vectorizes well, and QuantLib and HiGHS are available for reference and solving.
**Downside:** a pure-Python hot loop remains in the integer polish, and in position-level valuation.
**Revisit when:** a measured budget in `docs/evidence/performance/` is missed in a way vectorizing cannot fix.

## 2. MILP through HiGHS, not MIQP

**Decision:** express risk as absolute residuals and CVaR, keeping the program linear.
**Why:** dollars per unit of residual are easy to explain to a desk, a linear program gives a true lower bound, and there is no commercial solver dependency.
**Downside:** no quadratic risk, so a large residual is not penalized more than proportionally.
**Alternative:** a commercial MIQP solver, which adds a license and a dependency for little practical gain here.

## 3. Marks and costs from quotes, risk from the surface

**Decision:** a hedge's cost and limit price come only from its live quote, and the surface only supplies risk and interpolation.
**Why:** on real SPX, only about 35% of parametric prices are inside the bid/ask, so a model price is not tradable.
**Downside:** a stale or wrong quote directly affects a proposal, which is why the quality gates fail closed.

## 4. Escrowed dividends

**Decision:** volatility applies to spot less the PV of dividends.
**Why:** it is internally consistent with the forward and has a clean reference in QuantLib.
**Downside:** it differs from the `Spot` model some desks use, by about 0.12 in the tested calls, so a desk on that convention will see a difference.

## 5. No order routing

**Decision:** the package stops at exported tickets.
**Why:** routing brings market-access obligations and needs a human decision, plus a broker, kill switch, and certification.
**Downside:** a person or another system has to act on each ticket.

## 6. Evidence before features

**Decision:** phase gates are measured against real data and independent references wherever possible, and any gate met only by proxy is reported as open.
**Why:** the most severe failures this session were all invisible to synthetic tests.
**Downside:** slower visible progress.
