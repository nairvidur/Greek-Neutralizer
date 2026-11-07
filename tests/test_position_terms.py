"""Position terms of the hedge optimizer: the exit cost and the gross option cap.

Without them the objective prices only the trade, so a daily re-hedge on historical chains stacked
offsetting legs to 1,300 contracts against a 200-contract book. The oracle throughout is the
definition: gross option contracts are the absolute net contracts per option of the hedged book,
and the exit cost is the half-spread plus the fee on each position the candidates hold after it.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.optimize import Bounds, milp

from greeks_neutralizer.domain import OptionContract, Position
from greeks_neutralizer.errors import ValidationError
from greeks_neutralizer.hedging.optimizer import _filter_candidates, _Program, solve_hedge
from greeks_neutralizer.hedging.problem import HedgeConfig, HedgeProblem
from greeks_neutralizer.pricing.book import value_book
from tests.conftest import flat_snapshot
from tests.test_hedging import BOOK, CANDIDATES, DGV

# A hedge already held on one candidate, as a daily re-hedge carries forward.
HELD = CANDIDATES[4].instrument  # put 90, 90 days
HELD_BOOK = (*BOOK, Position(HELD, 40))
BOOK_ONLY_CONTRACTS = 10 + 5 + 8  # BOOK's options, none of which is a candidate


def gross_option_contracts(positions):
    net: dict[str, int] = {}
    for p in positions:
        if isinstance(p.instrument, OptionContract):
            net[p.instrument.key] = net.get(p.instrument.key, 0) + p.quantity
    return sum(abs(v) for v in net.values())


def proposal(book, config):
    return solve_hedge(HedgeProblem(book, CANDIDATES, DGV, (), config), flat_snapshot())


def test_cap_binds_and_is_respected():
    """The cap must bind here, or the test proves nothing: uncapped, the hedge is larger."""
    free = proposal(BOOK, HedgeConfig(risk_aversion=10.0))
    uncapped = gross_option_contracts(free.hedged_book(BOOK))
    cap = BOOK_ONLY_CONTRACTS + (uncapped - BOOK_ONLY_CONTRACTS) // 2
    assert uncapped > cap
    capped = proposal(BOOK, HedgeConfig(risk_aversion=10.0, max_gross_option_contracts=cap))
    assert gross_option_contracts(capped.hedged_book(BOOK)) <= cap
    assert capped.verification.passed


def test_cap_at_the_book_forces_the_held_hedge_out():
    """A cap equal to the book's own options leaves no room for any hedge option, so the 40 held
    puts must be sold and no other option bought. A wrong reading of `held` (ignoring it, or
    counting the trade alone) would leave them on, or buy another option."""
    config = HedgeConfig(risk_aversion=10.0, max_gross_option_contracts=BOOK_ONLY_CONTRACTS)
    result = proposal(HELD_BOOK, config)
    options = {
        leg.instrument.key: leg.quantity
        for leg in result.legs
        if isinstance(leg.instrument, OptionContract)
    }
    assert options == {HELD.key: -40}
    assert gross_option_contracts(result.hedged_book(HELD_BOOK)) == BOOK_ONLY_CONTRACTS


def test_cap_below_the_untradable_options_is_infeasible_not_ignored():
    from greeks_neutralizer.errors import HedgeInfeasibleError

    config = HedgeConfig(risk_aversion=10.0, max_gross_option_contracts=BOOK_ONLY_CONTRACTS - 1)
    with pytest.raises(HedgeInfeasibleError) as raised:
        proposal(BOOK, config)
    # Short by exactly the one contract the cap is below the book's untradable options.
    assert raised.value.binding == ("gross option contracts over the limit by 1.00",)


def _program(book, config):
    problem = HedgeProblem(book, CANDIDATES, DGV, (), config)
    snapshot = flat_snapshot()
    usable, _ = _filter_candidates(problem, snapshot)
    valued = value_book(book, snapshot)
    return _Program(problem, usable, dict(valued.by_underlying), np.zeros(0), np.zeros((0, 6)))


def test_exit_cost_is_the_half_spread_and_fee_on_each_position_left():
    """Oracle: the quotes. For any trade, the objective with price_exit is the one without it plus
    sum over candidates of (half-spread * multiplier + fee) * |held + trade|."""
    base = HedgeConfig(risk_aversion=10.0)
    plain = _program(HELD_BOOK, base)
    priced = _program(HELD_BOOK, replace(base, price_exit=True))
    held = np.array([40 if c.instrument.key == HELD.key else 0 for c in CANDIDATES])
    for lots in (np.zeros(6, dtype=np.int64), np.array([-300, 0, 2, -3, -40, 1], dtype=np.int64)):
        expected = sum(
            (c.quote.half_spread * c.instrument.multiplier + c.fee_per_unit) * abs(h + q)
            for c, h, q in zip(CANDIDATES, held, lots, strict=True)
        )
        assert_allclose(priced.evaluate(lots) - plain.evaluate(lots), expected, rtol=1e-12)


def test_evaluator_reproduces_the_solver_with_position_terms():
    """The evaluator is an independent statement of the program. With both position terms on and
    a held leg, it must equal the solver's objective at the solver's own solution."""
    config = HedgeConfig(
        risk_aversion=10.0, price_exit=True, max_gross_option_contracts=BOOK_ONLY_CONTRACTS + 60
    )
    program = _program(HELD_BOOK, config)
    constraints, _ = program._constraints()
    lb, ub = program._variable_bounds()
    ub[: 4 * program.n] = np.minimum(ub[: 4 * program.n], 60.0)
    result = milp(
        c=program._costs(),
        constraints=constraints,
        bounds=Bounds(lb, ub),
        integrality=program._integrality(all_whole=True),
        options={"mip_rel_gap": 1e-9},
    )
    assert result.status == 0
    x = np.asarray(result.x)
    lots = np.round(program._net_lots(x)).astype(np.int64)
    # Exact optimum, so no lot is both bought and sold.
    for i in range(program.n):
        assert min(x[4 * i] + x[4 * i + 1], x[4 * i + 2] + x[4 * i + 3]) < 1e-9
    assert_allclose(program.evaluate(lots), result.fun, rtol=1e-7)


def test_pricing_the_exit_unwinds_rather_than_stacks():
    """Selling a held option to cut risk saves its exit, while selling a new one adds a second
    exit, so with price_exit the hedge leaves fewer gross contracts behind."""
    base = HedgeConfig(risk_aversion=10.0)
    stacked = proposal(HELD_BOOK, base)
    priced = proposal(HELD_BOOK, replace(base, price_exit=True))
    assert gross_option_contracts(priced.hedged_book(HELD_BOOK)) < gross_option_contracts(
        stacked.hedged_book(HELD_BOOK)
    )


@pytest.mark.parametrize(
    "kwargs",
    [{"max_gross_option_contracts": -1}, {"max_gross_option_contracts": True}, {"price_exit": 1}],
    ids=["negative-cap", "bool-cap", "non-bool-exit"],
)
def test_position_settings_are_validated(kwargs):
    with pytest.raises(ValidationError):
        HedgeConfig(**kwargs)


class _Recording:
    """Wraps a policy, keeping each day's book and hedged-book positions after its trades."""

    def __init__(self, inner):
        self.inner, self.name, self.days = inner, inner.name, []

    def trades(self, day, book, hedges):
        trades = self.inner.trades(day, book, hedges)
        self.days.append((tuple(book), (*book, *hedges, *trades), tuple(trades)))
        return trades


def _proxy_run(policy, n=40):
    from greeks_neutralizer.backtest.engine import run_backtest
    from greeks_neutralizer.backtest.proxy import ProxyMarket, ShortStrangleBook
    from tests.test_backtest import history

    rng = np.random.default_rng(17)
    spots = 4000.0 * np.exp(np.cumsum(rng.normal(0.0, 0.012, n)))
    recording = _Recording(policy)
    run_backtest(ProxyMarket(history(spots)), ShortStrangleBook(), recording, fail_fast=True)
    return recording.days


def _dgv_policy(**kwargs):
    from greeks_neutralizer.backtest.policies import OptimizerPolicy
    from greeks_neutralizer.hedging.problem import GreekName, GreekTarget

    targets = tuple(
        GreekTarget("SPX", g) for g in (GreekName.DELTA, GreekName.GAMMA, GreekName.VEGA)
    )
    config = HedgeConfig(risk_aversion=1.0, max_legs=6, price_exit=True)
    return OptimizerPolicy("dgv", targets, config, **kwargs)


def test_policy_holds_the_hedged_book_within_its_multiple_of_the_book():
    """Every day, gross option contracts after the trade are at most the multiple of the book's
    own. The uncapped policy goes past it on the same path, so the cap is what holds it."""
    uncapped = _proxy_run(_dgv_policy())
    capped = _proxy_run(_dgv_policy(gross_multiple=1.5))
    assert (
        max(
            gross_option_contracts(after) / gross_option_contracts(book)
            for book, after, _ in uncapped
        )
        > 1.5
    )
    for book, after, _ in capped:
        assert gross_option_contracts(after) <= int(1.5 * gross_option_contracts(book))


def test_band_of_the_whole_book_never_trades_options():
    """With option_band at 1, an unhedged book's residual equals its own Greeks, which is inside
    the band, so options are never offered and every trade is in the linear hedge. With no band
    the same path does trade options, so the band is what stops them."""
    banded = _proxy_run(_dgv_policy(option_band=1.0), n=15)
    free = _proxy_run(_dgv_policy(), n=15)

    def option_trades(days):
        return sum(
            isinstance(t.instrument, OptionContract) for _, _, trades in days for t in trades
        )

    assert option_trades(free) > 0
    assert option_trades(banded) == 0


@pytest.mark.parametrize(
    ("days", "start"),
    [
        # SPY's source: one day in 2019, then a gap of eight months.
        (["2019-05-10", "2020-01-06", "2020-01-08", "2020-01-10"], 1),
        # A 14-day gap, like SPY's in 2024, is kept: it is a period, not a separate history.
        (["2024-07-26", "2024-07-29", "2024-08-12", "2024-08-14"], 0),
        (["2024-01-02", "2024-01-03"], 0),
    ],
    ids=["isolated-leading-day", "fourteen-day-gap-kept", "no-gap"],
)
def test_chain_history_starts_after_the_last_long_gap(monkeypatch, days, start):
    """Oracle: the rule in the ledger, a history that starts after the last gap over 15 days."""
    import importlib
    from pathlib import Path

    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    script = importlib.import_module("run_chain_backtest")
    assert script.continuous_history([Path(f"{d}.csv") for d in days]) == start
