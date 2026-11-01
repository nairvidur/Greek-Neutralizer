# Greeks Neutralizer

Options hedge decisions and risk.
Given a book, the tradable market right now, your costs, and your limits, it proposes the best hedge, states what risk remains, and shows why.

**Status: alpha. It has not been used to hedge real money.**
It is decision support and does not route orders.
Read `docs/limitations.md` before relying on it.

## What it does

- **Prices** European and American options with discrete dividends, with Greeks to third order, reconciled to QuantLib.
- **Calibrates** arbitrage-checked volatility surfaces from real chains, with quality gates that fail closed.
- **Optimizes hedges** across any number of options and delta-one instruments, using real bid/ask costs, integer lots, Greek bands, budgets, leg limits, and CVaR over stress scenarios.
- **Verifies** each proposal by repricing the hedged book from scratch, and raises rather than returning one that fails.
- **Tests stability** by re-solving under small market changes and measuring what being wrong would cost.
- **Stresses** books by full repricing, across historical and hypothetical scenarios.
- **Governs** hedges with four-eyes approval, expiry, and a hash-chained audit trail, ending in CSV or FIX tickets.
- **Backtests** policies point in time, with bootstrap confidence intervals.

## Install

```sh
uv sync                                  # development, from a clone
pip install greeks_neutralizer-*.whl     # from a built wheel
pip install "$(ls greeks_neutralizer-*.whl)[service]"   # with the HTTP service and UI
```

Requires Python 3.11 or later.
The package is not published to PyPI.

## Use

The CLI prints TOON for agents and people, and takes no interactive input:

```sh
gn example --kind hedge > request.json
gn hedge --request request.json
gn frontier --request request.json
gn example --kind book > book.json
gn example --kind market > market.json
gn value --book book.json --market market.json
gn stress --book book.json --market market.json --set historical
```

As a library:

```python
from datetime import UTC, datetime, timedelta

from greeks_neutralizer.domain import (
    ExerciseStyle, LinearInstrument, OptionContract, OptionRight, Position, Quote, SettlementKind,
)
from greeks_neutralizer.hedging.optimizer import solve_hedge
from greeks_neutralizer.hedging.problem import (
    GreekName, GreekTarget, HedgeCandidate, HedgeProblem,
)
from greeks_neutralizer.market import DiscountCurve, FlatVol, MarketSnapshot, UnderlyingMarket

now = datetime(2026, 1, 5, 15, 0, tzinfo=UTC)
market = MarketSnapshot(
    now, {"XYZ": UnderlyingMarket("XYZ", 100.0, DiscountCurve.flat(0.05), FlatVol(0.25))}
)
short_call = OptionContract(
    "XYZ", OptionRight.CALL, 100.0, now + timedelta(days=30),
    ExerciseStyle.EUROPEAN, settlement=SettlementKind.CASH,
)
stock = HedgeCandidate(
    LinearInstrument("XYZ", "XYZ"), Quote(99.99, 100.01, 100_000, 100_000, now)
)
problem = HedgeProblem(
    book=(Position(short_call, -10),),
    candidates=(stock,),
    targets=(GreekTarget("XYZ", GreekName.DELTA),),
)
proposal = solve_hedge(problem, market)
for leg in proposal.legs:
    print(leg.rationale)
```

The service and web UI:

```sh
export GN_TOKENS="token-a=alice,token-b=bob"
uv run uvicorn --factory greeks_neutralizer.service.app:factory --port 8000
```

## Evidence

Each claim links to what backs it, and to what it does not establish.

| Claim | Evidence |
|---|---|
| Pricing matches QuantLib | `docs/evidence/phase2/README.md`, `uv run pytest -m reference` |
| Fits real SPX and SPY chains, arbitrage-free over quoted strikes | `docs/evidence/phase2/` |
| Hedges real chains, with Greeks matching QuantLib to 1e-11 | `docs/evidence/phase3/` |
| Reduces hedging risk on real market paths with **modeled** option prices | `docs/evidence/phase4/` |
| Performance against provisional budgets | `docs/evidence/performance/` |
| The web workflow works in a real browser | `docs/evidence/phase5/` |

**It has no backtest on historical option chains yet.**
That needs licensed data, and until then the claim of real hedging value is not established.

## Docs

| Read | For |
|---|---|
| `docs/limitations.md` | What it does not do. Start here |
| `docs/conventions.md` | Units and signs of every number |
| `docs/methodology.md` | How each model works |
| `docs/evidence/` | Results, and how to reproduce them |
| `docs/risk/stress.md`, `docs/risk/margin.md` | Scenario provenance, and margin estimates |
| `docs/governance/model-pack.md` | Inputs to a model risk process |
| `docs/operations/runbook.md`, `docs/operations/security.md` | Running it, and its security posture |
| `docs/adr/README.md` | Why it is built this way |

## Develop

```sh
uv sync
uv run ruff check . && uv run ruff format --check .
uv run mypy
uv run pytest -q
```

Warnings are errors in tests.
Each bug fix starts from a failing test, and vendor market data is never committed.

## License

MIT. See `LICENSE`.
Whether a commercial license or open-core model fits is an open decision for the owner.
