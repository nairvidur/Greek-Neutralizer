"""Hedge optimizer: exact neutralization, integer lots, costs, constraints, and verification."""

from __future__ import annotations

import contextlib
from dataclasses import replace
from datetime import timedelta

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from numpy.testing import assert_allclose
from scipy.optimize import linprog as real_linprog
from scipy.optimize import milp as real_milp

from greeks_neutralizer.domain import (
    LinearInstrument,
    LinearKind,
    OptionContract,
    Position,
    Quote,
)
from greeks_neutralizer.errors import HedgeInfeasibleError, SearchLimitError, ValidationError
from greeks_neutralizer.hedging import optimizer
from greeks_neutralizer.hedging.diagnostics import (
    hedge_decay_forecast,
    hedge_frontier,
    whalley_wilmott_band,
)
from greeks_neutralizer.hedging.optimizer import VerificationError, solve_hedge
from greeks_neutralizer.hedging.problem import (
    GreekName,
    GreekTarget,
    HedgeCandidate,
    HedgeConfig,
    HedgeProblem,
    ProposalStatus,
    VerificationReport,
)
from greeks_neutralizer.pricing.black import black_greeks
from greeks_neutralizer.pricing.book import value_book
from greeks_neutralizer.risk.scenarios import HISTORICAL_STRESSES, Scenario, shock_snapshot
from tests.conftest import AS_OF, euro, flat_snapshot

STOCK = LinearInstrument("XYZ", "XYZ")


def q(mid, half=0.02, size=1000, age=0.0):
    return Quote(max(mid - half, 0.0), mid + half, size, size, AS_OF - timedelta(seconds=age))


def option_candidate(right, strike, days, half=0.02, size=1000, **kw):
    contract = euro(right, strike, days)
    T = days / 365
    price = black_greeks(100.0, strike, T, 0.05, 0.0, 0.25, (right == "call" and 1) or -1).price[0]
    return HedgeCandidate(contract, q(price, half, size), **kw)


def stock_candidate(half=0.01, size=1_000_000, **kw):
    return HedgeCandidate(STOCK, q(100.0, half, size), **kw)


BOOK = (
    Position(euro("call", 100, 30), 10),
    Position(euro("put", 95, 30), -5),
    Position(euro("call", 110, 60), 8),
)
CANDIDATES = (
    stock_candidate(),
    option_candidate("put", 95, 60),
    option_candidate("call", 105, 60),
    option_candidate("call", 100, 90),
    option_candidate("put", 90, 90),
    option_candidate("call", 110, 90),
)
DGV = tuple(
    GreekTarget("XYZ", g, weight=w)
    for g, w in ((GreekName.DELTA, 1.0), (GreekName.GAMMA, 1.0), (GreekName.VEGA, 1.0))
)


def solve(**kw):
    config = kw.pop("config", HedgeConfig(risk_aversion=10.0))
    problem = HedgeProblem(
        kw.pop("book", BOOK),
        kw.pop("candidates", CANDIDATES),
        kw.pop("targets", DGV),
        kw.pop("scenarios", ()),
        config,
    )
    return solve_hedge(problem, flat_snapshot(), **kw)


def test_high_risk_aversion_reduces_all_greeks_within_one_contract():
    proposal = solve()
    for line in proposal.residuals:
        assert abs(line.after) < abs(line.before)
    after = proposal.after["XYZ"]
    # Shares trade singly, so one share is what is left of delta. Vega moves only by whole
    # contracts, so it is within the smallest vega on offer, valued on its own here.
    assert abs(after.delta_shares) < 1.0
    smallest_vega = min(
        value_book([Position(c.instrument, 1)], flat_snapshot()).by_underlying["XYZ"].vega_1pt
        for c in CANDIDATES[1:]
    )
    assert abs(after.vega_1pt) < smallest_vega


def test_quantities_are_integer_lots():
    proposal = solve()
    assert proposal.legs
    assert all(type(leg.quantity) is int and leg.quantity != 0 for leg in proposal.legs)
    tens = (
        CANDIDATES[0],
        *(replace(c, lot_size=10, max_buy=100, max_sell=100) for c in CANDIDATES[1:]),
    )
    options = [
        leg for leg in solve(candidates=tens).legs if isinstance(leg.instrument, OptionContract)
    ]
    assert options
    assert all(leg.quantity % 10 == 0 for leg in options)


def test_more_than_three_instruments_and_linear_hedge_are_supported():
    # Baseline defect: the old solver accepted exactly three option hedges and no underlying.
    proposal = solve()
    instruments = {leg.instrument.key for leg in proposal.legs}
    assert STOCK.key in instruments
    assert len(instruments) > 3


def test_premium_is_accounted_for_and_separate_from_cost():
    # Baseline defect: net value ignored the premium paid for the hedge.
    proposal = solve()
    assert proposal.net_premium != 0.0
    expected = sum(
        leg.quantity * leg.instrument.multiplier * c.quote.mid
        for leg in proposal.legs
        for c in CANDIDATES
        if c.instrument == leg.instrument
    )
    assert_allclose(proposal.net_premium, expected)
    assert proposal.transaction_cost > 0.0
    assert proposal.transaction_cost < abs(proposal.net_premium)


def test_transaction_cost_is_spread_times_size_plus_fees():
    candidates = (stock_candidate(half=0.05, fee_per_unit=0.01),)
    targets = (GreekTarget("XYZ", GreekName.DELTA, weight=1.0),)
    p = solve(candidates=candidates, targets=targets)
    leg = p.legs[0]
    assert_allclose(p.transaction_cost, abs(leg.quantity) * 0.06)


def test_low_risk_aversion_trades_less_than_high():
    cheap = solve(config=HedgeConfig(risk_aversion=0.001))
    keen = solve(config=HedgeConfig(risk_aversion=10.0))
    # Only cost is monotone. Gross lots is not: an option can replace many cheap shares.
    assert cheap.transaction_cost < keen.transaction_cost


def test_flat_book_needs_no_trade():
    empty = solve(book=())
    assert empty.status is ProposalStatus.NO_TRADE
    assert empty.legs == ()


def test_hard_band_is_met_and_verified():
    targets = (
        GreekTarget("XYZ", GreekName.DELTA, lower=-500.0, upper=500.0, weight=0.0),
        GreekTarget("XYZ", GreekName.VEGA, lower=-5.0, upper=5.0, weight=0.0),
    )
    p = solve(targets=targets)
    after = p.after["XYZ"]
    assert -500.0 <= after.delta_dollars <= 500.0
    assert -5.0 <= after.vega_1pt <= 5.0
    assert p.verification.passed


def test_infeasible_band_names_the_binding_constraint():
    targets = (GreekTarget("XYZ", GreekName.VEGA, lower=-1.0, upper=1.0),)
    candidates = (stock_candidate(),)  # stock has no vega, so vega cannot be hedged
    with pytest.raises(HedgeInfeasibleError, match="hard constraints") as err:
        solve(candidates=candidates, targets=targets)
    assert any("vega_1pt" in text for text in err.value.binding)


def test_premium_budget_is_enforced():
    # The unconstrained hedge sets the level, so that the budget is certain to bind.
    unconstrained = solve(candidates=CANDIDATES[1:])
    budget = unconstrained.net_premium - 300.0
    capped = solve(
        candidates=CANDIDATES[1:],
        config=HedgeConfig(risk_aversion=10.0, premium_budget=budget),
    )
    assert capped.net_premium <= budget + 1e-6


def test_a_constraint_cannot_improve_the_objective():
    # Found by this suite: at its time limit the solver returned an incumbent so poor that adding a
    # constraint appeared to improve the answer, which is impossible at an optimum.
    free = solve(candidates=CANDIDATES[1:])
    capped = solve(
        candidates=CANDIDATES[1:],
        config=HedgeConfig(risk_aversion=10.0, premium_budget=free.net_premium - 300.0),
    )
    assert capped.objective >= free.objective * (1.0 - 0.02)
    assert capped.objective_lower_bound >= free.objective_lower_bound - 1e-6


@pytest.mark.parametrize(
    "config",
    [
        HedgeConfig(risk_aversion=1.0),
        HedgeConfig(risk_aversion=1.0, cvar_weight=0.5, cvar_alpha=0.5),
        HedgeConfig(risk_aversion=1.0, cvar_weight=0.5, ticket_cost=5.0),
        HedgeConfig(risk_aversion=0.0, cvar_weight=1.0, cvar_alpha=0.8),
    ],
    ids=["greeks", "cvar", "cvar-ticket", "cvar-only"],
)
def test_exact_evaluator_reproduces_the_solver_objective(config):
    # Two independent statements of one objective must agree. Options only, since all lots are then
    # integer: a continuous share hedge would differ from its rounding, by design.
    from scipy.optimize import Bounds, milp

    from greeks_neutralizer.hedging.optimizer import _filter_candidates, _Program
    from greeks_neutralizer.risk.scenarios import book_scenario_pnl, scenario_pnl_matrix

    scenarios = HISTORICAL_STRESSES[:4]
    problem = HedgeProblem(BOOK, CANDIDATES[1:], DGV, scenarios, config)
    snapshot = flat_snapshot()
    usable, _ = _filter_candidates(problem, snapshot)
    book = value_book(problem.book, snapshot)
    program = _Program(
        problem,
        usable,
        dict(book.by_underlying),
        book_scenario_pnl(problem.book, snapshot, scenarios),
        scenario_pnl_matrix([u.candidate.instrument for u in usable], snapshot, scenarios),
    )
    constraints, _ = program._constraints()
    lb, ub = program._variable_bounds()
    # Bounded lots keep the exact solve fast. Agreement at this solution is what is being tested.
    ub[: 4 * program.n] = np.minimum(ub[: 4 * program.n], 30.0)
    result = milp(
        c=program._costs(),
        constraints=constraints,
        bounds=Bounds(lb, ub),
        integrality=program._integrality(),
        options={"mip_rel_gap": 1e-3, "node_limit": 2000},
    )
    assert result.x is not None
    x = np.asarray(result.x)
    lots = np.round(program._net_lots(x)).astype(np.int64)
    # An incumbent at a search limit can hold offsetting lots, which pay cost and net to nothing.
    # Reading back the net position is right, so the evaluator comes out lower by exactly that.
    costs = program._costs()
    wash = 0.0
    for i in range(program.n):
        buys, sells = x[4 * i] + x[4 * i + 1], x[4 * i + 2] + x[4 * i + 3]
        wash += min(buys, sells) * (costs[4 * i] + costs[4 * i + 2])
    assert_allclose(program.evaluate(lots), result.fun - wash, rtol=1e-6)


def _hard_option_problem(time_limit):
    # Options only, so that integrality is hard and the search stops at a limit, not a proof.
    return HedgeProblem(
        BOOK,
        CANDIDATES[1:],
        DGV,
        (),
        HedgeConfig(risk_aversion=10.0, mip_gap=1e-6, time_limit_seconds=time_limit),
    )


def test_the_search_stops_at_a_node_limit_and_the_clock_is_only_a_backstop(monkeypatch):
    # Found comparing evidence runs: stopping on the wall clock made results vary with machine load,
    # so a proposal could not be replayed. Every integer search must get the node limit, and the
    # configured clock only beside it. The clock is held back from HiGHS so that a slow machine
    # cannot change this test's outcome.
    seen = []

    def without_the_clock(real):
        def solver(*args, **kwargs):
            options = dict(kwargs["options"])
            seen.append(options.pop("time_limit"))
            return real(*args, **{**kwargs, "options": options})

        return solver

    node_limits = []

    def recording_milp(*args, **kwargs):
        node_limits.append(kwargs["options"]["node_limit"])
        return without_the_clock(real_milp)(*args, **kwargs)

    monkeypatch.setattr(optimizer, "milp", recording_milp)
    monkeypatch.setattr(optimizer, "linprog", without_the_clock(real_linprog))
    proposal = solve_hedge(_hard_option_problem(12.5), flat_snapshot())
    assert proposal.legs
    assert set(seen) == {12.5}
    assert node_limits
    assert set(node_limits) == {HedgeConfig().mip_node_limit}
    assert not any("SOLVER_TIME_LIMIT" in issue for issue in proposal.issues)


def test_a_real_search_stops_at_its_node_limit(monkeypatch):
    # HiGHS itself: this search explores past one node when allowed, and stops there when told to.
    counts = []

    def recording(*args, **kwargs):
        result = real_milp(*args, **kwargs)
        counts.append(int(result.mip_node_count))
        return result

    monkeypatch.setattr(optimizer, "milp", recording)
    problem = _hard_option_problem(30.0)
    solve_hedge(problem, flat_snapshot())
    assert max(counts) > 1
    counts.clear()
    limited = replace(problem.config, mip_node_limit=1)
    with contextlib.suppress(SearchLimitError):
        solve_hedge(replace(problem, config=limited), flat_snapshot())
    assert counts
    assert max(counts) <= 1


def test_hitting_the_wall_clock_backstop_is_reported(monkeypatch):
    # HiGHS stopping at the clock with an incumbent, which is status 1 from scipy, on demand.
    def at_the_clock(*args, **kwargs):
        result = real_milp(*args, **kwargs)
        result.status = 1
        result.message = "Time limit reached. (HiGHS Status 13: model_status is Time limit reached)"
        return result

    monkeypatch.setattr(optimizer, "milp", at_the_clock)
    p = solve_hedge(_hard_option_problem(30.0), flat_snapshot())
    assert p.legs
    assert any("SOLVER_TIME_LIMIT" in issue for issue in p.issues)


def test_node_limit_must_be_positive():
    with pytest.raises(ValidationError, match="at least 1"):
        HedgeConfig(mip_node_limit=0)


def test_objective_never_beats_its_continuous_lower_bound():
    for config in (HedgeConfig(risk_aversion=0.1), HedgeConfig(risk_aversion=10.0)):
        p = solve(candidates=CANDIDATES[1:], config=config)
        assert p.objective >= p.objective_lower_bound - 1e-6
        assert 0.0 <= p.solver_gap < 1.0


def _large_universe():
    out = [stock_candidate()]
    for days in (30, 45, 60, 75, 90, 120, 150, 180):
        for strike in np.arange(70.0, 131.0, 2.5):
            out.append(option_candidate("put" if strike < 100.0 else "call", float(strike), days))
    return tuple(out)


def test_frontier_is_monotone_on_a_large_universe():
    # Found on a real SPX chain: with 2,144 candidates and an unscreened integer search, the time
    # limit left higher risk aversion with a higher residual.
    universe = _large_universe()
    assert len(universe) > 150
    residuals = []
    for aversion in (0.05, 0.5, 5.0):
        p = solve(candidates=universe, config=HedgeConfig(risk_aversion=aversion, max_legs=6))
        residuals.append(
            sum(t.effective_weight * abs(getattr(p.after["XYZ"], t.greek.value)) for t in DGV)
        )
    # Integer lots leave a tolerance: one contract's residual, at most about $150 here.
    assert residuals[1] <= residuals[0] + 150.0
    assert residuals[2] <= residuals[1] + 150.0


def test_screen_on_a_large_universe_keeps_the_leg_that_offsets_the_book(monkeypatch):
    # Oracle: buying back the exact contract the book is short flattens every Greek, and it has the
    # tightest market here, so any other leg only adds cost. It is listed last, so that a screen
    # keeping an arbitrary subset would lose it.
    exact = option_candidate("call", 100, 60, half=0.01)
    others = [
        option_candidate("call", 80 + 0.5 * i, 60, half=0.5)
        for i in range(60)
        if 80 + 0.5 * i != 100
    ]
    universe = (*others, exact)
    assert len(universe) > optimizer._MAX_INTEGER_CANDIDATES
    book = (Position(exact.instrument, -20),)

    screened = solve(book=book, candidates=universe)
    assert screened.quantities == {exact.instrument.key: 20}

    monkeypatch.setattr(optimizer, "_MAX_INTEGER_CANDIDATES", len(universe))
    assert solve(book=book, candidates=universe).quantities == screened.quantities


def test_integrality_cost_is_reported_when_material():
    p = solve(candidates=CANDIDATES[1:], config=HedgeConfig(risk_aversion=10.0, mip_gap=1e-6))
    assert p.solver_gap > 1e-6
    assert p.status is ProposalStatus.WITHIN_GAP
    assert any("INTEGRALITY_GAP" in issue for issue in p.issues)


def test_max_legs_is_respected():
    p = solve(config=HedgeConfig(risk_aversion=10.0, max_legs=2))
    assert len(p.legs) <= 2


def test_ticket_cost_reduces_legs():
    free = solve(config=HedgeConfig(risk_aversion=0.5))
    dear = solve(config=HedgeConfig(risk_aversion=0.5, ticket_cost=200.0))
    assert len(dear.legs) < len(free.legs)


def test_max_buy_and_sell_caps_are_respected():
    capped = (
        replace(CANDIDATES[0], max_buy=50, max_sell=50),
        *(replace(c, max_buy=3, max_sell=3) for c in CANDIDATES[1:]),
    )
    # Without caps some leg goes past them, so the caps below have work to do.
    assert any(
        abs(leg.quantity) > (50 if isinstance(leg.instrument, LinearInstrument) else 3)
        for leg in solve().legs
    )
    p = solve(candidates=capped)
    assert any(isinstance(leg.instrument, OptionContract) for leg in p.legs)
    for leg in p.legs:
        limit = 50 if isinstance(leg.instrument, LinearInstrument) else 3
        assert abs(leg.quantity) <= limit


def test_lot_size_is_respected():
    candidates = (replace(CANDIDATES[0], lot_size=100),)
    targets = (GreekTarget("XYZ", GreekName.DELTA, weight=1.0),)
    p = solve(candidates=candidates, targets=targets)
    assert p.legs[0].quantity % 100 == 0


def test_impact_tier_is_used_only_past_displayed_size():
    deep = stock_candidate(
        half=0.01, size=100, impact_per_unit=0.5, max_buy=100_000, max_sell=100_000
    )
    targets = (GreekTarget("XYZ", GreekName.DELTA, weight=1.0),)
    p = solve(candidates=(deep,), targets=targets)
    size = abs(p.legs[0].quantity)
    assert size > 100
    expected = size * 0.01 + (size - 100) * 0.5
    assert_allclose(p.transaction_cost, expected, rtol=1e-9)


def test_zero_bid_cannot_be_sold():
    held = euro("call", 105, 60)
    fair_price = black_greeks(100.0, 105, 60 / 365, 0.05, 0.0, 0.25, 1).price[0]

    def traded(bid):
        market = Quote(bid, 2.0 * fair_price - bid, 100, 100, AS_OF)
        proposal = solve(
            book=(Position(held, 10),),
            candidates=(HedgeCandidate(held, market), option_candidate("call", 130, 60)),
            config=HedgeConfig(risk_aversion=1000.0),
        )
        return proposal.quantities

    # With any bid at all, selling this contract is the exact offset, so the book wants to sell it.
    assert traded(0.01)[held.key] < 0
    # With no bid, the book still wants to reduce its risk, and has to use the alternative.
    no_bid = traded(0.0)
    assert no_bid.get(held.key, 0) >= 0
    assert no_bid


@pytest.mark.parametrize(
    ("quote", "reason"),
    [
        (Quote(1.2, 1.1, 10, 10, AS_OF), "crossed"),
        (Quote(1.0, 1.1, 10, 10, AS_OF - timedelta(minutes=5)), "stale"),
        (Quote(1.0, 1.1, 0, 0, AS_OF), "no_size"),
    ],
    ids=["crossed", "stale", "no-size"],
)
def test_bad_quotes_are_excluded_with_reasons(quote, reason):
    bad = HedgeCandidate(euro("call", 105, 60), quote)
    p = solve(candidates=(bad, stock_candidate()))
    assert any(reason in item.reason for item in p.excluded)
    assert all(leg.instrument != bad.instrument for leg in p.legs)


def test_expired_candidate_excluded():
    expired = HedgeCandidate(euro("call", 105, 0), q(1.0))
    p = solve(candidates=(expired, stock_candidate()))
    assert any(item.reason == "expired" for item in p.excluded)


def test_a_proposal_failing_verification_raises_rather_than_returning(monkeypatch):
    # The documented rule: it raises rather than returning a bad plan. Verification itself is
    # replaced at its boundary, since a working solver has nothing for it to catch.
    assert solve().verification.passed
    failed = VerificationReport(False, 1.0, 1.0, ("planted",))
    monkeypatch.setattr(optimizer, "verify_proposal", lambda *args: failed)
    with pytest.raises(VerificationError, match="planted"):
        solve()


def test_cvar_objective_improves_worst_tail():
    scenarios = HISTORICAL_STRESSES
    book = (Position(euro("put", 95, 30), -50),)
    candidates = (option_candidate("put", 90, 30), option_candidate("put", 85, 60))
    targets = (GreekTarget("XYZ", GreekName.DELTA, weight=0.0),)
    unhedged = solve(
        book=book,
        candidates=candidates,
        targets=targets,
        scenarios=scenarios,
        config=HedgeConfig(risk_aversion=0.0),
    )
    hedged = solve(
        book=book,
        candidates=candidates,
        targets=targets,
        scenarios=scenarios,
        config=HedgeConfig(risk_aversion=0.0, cvar_weight=1.0, cvar_alpha=0.8),
    )
    assert min(hedged.scenario_pnl_after.values()) > min(unhedged.scenario_pnl_after.values())
    assert hedged.verification.passed


def test_cvar_cannot_farm_scenario_profit_with_unlimited_size():
    # Found by the proxy backtest: 11.6 billion lots and a $10 trillion loss. With crash-heavy
    # scenarios, long puts make every scenario profitable, so CVaR went negative without limit and
    # the solver was paid to buy more. Protection earns nothing once the tail loss reaches zero.
    book = (Position(euro("put", 95, 30), -10),)
    candidates = (option_candidate("put", 90, 30), option_candidate("put", 85, 60))
    targets = (GreekTarget("XYZ", GreekName.DELTA, weight=0.0),)
    p = solve(
        book=book,
        candidates=candidates,
        targets=targets,
        scenarios=HISTORICAL_STRESSES,
        config=HedgeConfig(risk_aversion=0.0, cvar_weight=1.0, cvar_alpha=0.8),
    )
    assert p.gross_lots < 200
    # It still protects: the tail loss is removed, not left in place.
    assert min(p.scenario_pnl_after.values()) > min(p.scenario_pnl_before.values())


def test_objective_cannot_go_negative_from_scenario_gains():
    book = (Position(euro("put", 95, 30), 10),)  # already long protection
    p = solve(
        book=book,
        candidates=(option_candidate("put", 90, 30),),
        targets=(GreekTarget("XYZ", GreekName.DELTA, weight=0.0),),
        scenarios=HISTORICAL_STRESSES,
        config=HedgeConfig(risk_aversion=0.0, cvar_weight=1.0, cvar_alpha=0.8),
    )
    assert p.objective >= 0.0
    assert p.legs == ()


def test_no_option_leg_can_exceed_the_lot_cap():
    # A hard stop independent of the objective, so that no modeling error can size a leg absurdly.
    book = (Position(euro("call", 100, 30), -20000),)
    p = solve(
        book=book,
        candidates=(option_candidate("call", 100, 60, size=100000),),
        targets=(GreekTarget("XYZ", GreekName.VEGA, weight=10.0),),
        config=HedgeConfig(risk_aversion=10.0, max_lots_per_leg=1000),
    )
    assert abs(p.legs[0].quantity) <= 1000


def test_proxy_hedge_reports_basis_risk():
    proxy = LinearInstrument("PROXY", "XYZ", LinearKind.PROXY, delta_ratio=1.0, basis_vol=0.05)
    candidates = (HedgeCandidate(proxy, q(100.0, 0.01, 1_000_000)),)
    targets = (GreekTarget("XYZ", GreekName.DELTA, weight=1.0),)
    p = solve(candidates=candidates, targets=targets)
    assert p.basis_risk_1d > 0.0
    assert any("BASIS_RISK" in issue for issue in p.issues)


def test_multi_underlying_targets_do_not_cross_hedge():
    from greeks_neutralizer.market import DiscountCurve, FlatVol, MarketSnapshot, UnderlyingMarket

    other = UnderlyingMarket("ABC", 50.0, DiscountCurve.flat(0.05), FlatVol(0.3))
    base = flat_snapshot()
    snapshot = MarketSnapshot(AS_OF, {"XYZ": base.market("XYZ"), "ABC": other})
    book = (Position(euro("call", 100, 30), 10),)
    abc_stock = LinearInstrument("ABC", "ABC")
    problem = HedgeProblem(
        book,
        (HedgeCandidate(abc_stock, q(50.0, 0.01, 100000)), stock_candidate()),
        (
            GreekTarget("XYZ", GreekName.DELTA, weight=1.0),
            GreekTarget("ABC", GreekName.DELTA, weight=1.0),
        ),
    )
    p = solve_hedge(problem, snapshot)
    assert {leg.instrument.symbol for leg in p.legs} == {"XYZ"}


def test_manifest_changes_with_inputs_and_replay_is_exact():
    a = solve()
    b = solve()
    assert a.quantities == b.quantities
    assert a.manifest.problem_fingerprint == b.manifest.problem_fingerprint
    c = solve(config=HedgeConfig(risk_aversion=9.0))
    assert a.manifest.problem_fingerprint != c.manifest.problem_fingerprint


def test_stability_condition_number_is_visible_for_near_identical_hedges():
    # The caps force the trade across the twins, whose nearly parallel columns are ill-conditioned.
    twins = (
        option_candidate("call", 100, 60, max_buy=12, max_sell=12),
        option_candidate("call", 100.01, 60, max_buy=12, max_sell=12),
        option_candidate("call", 100.02, 60, max_buy=12, max_sell=12),
    )
    book = (Position(euro("call", 100, 60), -30),)
    p = solve(
        book=book, candidates=twins, config=HedgeConfig(risk_aversion=10.0), run_stability=True
    )
    assert len(p.legs) > 1
    assert p.stability is not None
    assert p.stability.condition_number > 100.0


def test_singular_gamma_vega_from_one_expiry_is_not_called_unstable():
    # Same-expiry options have exactly proportional gamma and vega: a Black-Scholes identity. That
    # makes the Greek matrix singular for a sensible small hedge, which must not read as unstable.
    same_expiry = (
        stock_candidate(),
        option_candidate("call", 105, 60),
        option_candidate("call", 110, 60),
    )
    p = solve(candidates=same_expiry, config=HedgeConfig(risk_aversion=1.0), run_stability=True)
    assert p.stability is not None
    assert p.stability.condition_number > 1e6
    # The trade may swap between the substitutes, which costs little: regret is what decides.
    assert p.stability.regret_ratio < 0.1
    assert p.stability.is_stable


def test_regret_is_infinite_when_the_held_hedge_is_unavailable():
    from greeks_neutralizer.hedging.diagnostics import _regret
    from greeks_neutralizer.hedging.optimizer import _filter_candidates, _Program

    problem = HedgeProblem(BOOK, CANDIDATES, DGV, (), HedgeConfig(risk_aversion=1.0))
    snapshot = flat_snapshot()
    usable, _ = _filter_candidates(problem, snapshot)
    book = value_book(problem.book, snapshot)
    empty = np.zeros((0, len(usable)))
    program = _Program(problem, usable, dict(book.by_underlying), np.zeros(0), empty)
    lots = np.zeros(len(usable), dtype=np.int64)
    lots[1] = 3
    solved = program.solve()
    assert _regret(program, usable, solved.objective, usable, solved.lots) == pytest.approx(
        0.0, abs=1e-6
    )
    assert _regret(program, usable[2:], 1.0, usable, lots) == float("inf")


def test_stock_moving_with_spot_is_not_instability():
    p = solve(
        candidates=(stock_candidate(),),
        targets=(GreekTarget("XYZ", GreekName.DELTA, weight=1.0),),
        run_stability=True,
    )
    assert p.stability is not None
    assert p.stability.lot_change_ratio == 0.0
    assert p.stability.is_stable


def test_stability_report_on_diversified_hedges():
    p = solve(run_stability=True)
    assert p.stability is not None

    def penalized(market):
        # By hand from what the objective says: weight 1 on each of the three Greeks, as DGV sets.
        held = value_book(p.hedged_book(BOOK), market).by_underlying["XYZ"]
        return abs(held.delta_dollars) + abs(held.gamma_1pct) + abs(held.vega_1pt)

    today = penalized(flat_snapshot())
    shocks = {
        "spot +0.25%": Scenario("a", spot_shock=0.0025),
        "spot -0.25%": Scenario("b", spot_shock=-0.0025),
        "vol +1pt": Scenario("c", vol_shock=0.01),
        "vol -1pt": Scenario("d", vol_shock=-0.01),
        "skew steeper": Scenario("e", skew_shock=-0.02),
        "skew flatter": Scenario("f", skew_shock=0.02),
    }
    assert set(p.stability.drift) == set(shocks)
    for name, shock in shocks.items():
        expected = penalized(shock_snapshot(flat_snapshot(), shock)) - today
        assert_allclose(p.stability.drift[name], expected, rtol=1e-9, atol=1e-6)


def test_frontier_contains_baselines_and_cost_rises_with_aversion():
    problem = HedgeProblem(BOOK, CANDIDATES, DGV)
    points = hedge_frontier(problem, flat_snapshot(), risk_aversions=(0.01, 1.0, 10.0))
    labels = [p.label for p in points]
    assert labels[0] == "do nothing"
    assert labels[1] == "delta only"
    optimized = [p for p in points if p.risk_aversion is not None]
    costs = [p.transaction_cost for p in optimized]
    assert costs == sorted(costs)
    # The highest aversion must leave no more residual than doing nothing or delta-only.
    assert optimized[-1].penalized_residual <= points[0].penalized_residual
    assert optimized[-1].penalized_residual <= points[1].penalized_residual


def test_no_trade_band_formula_and_monotonicity():
    band = whalley_wilmott_band(
        gamma_shares=20.0, spot=50.0, proportional_cost=0.001, risk_aversion=1e-4
    )
    # Whalley-Wilmott by hand: (3/2 * 0.001 * 50 * 20^2 / 1e-4)^(1/3) = 300,000^(1/3). Spot and
    # gamma differ, so that swapping their powers, which gives 750,000^(1/3), fails.
    assert_allclose(band.half_width_shares, 66.9432950, rtol=1e-7)
    wider = whalley_wilmott_band(20.0, 50.0, 0.002, 1e-4)
    assert wider.half_width_shares > band.half_width_shares
    assert band.should_rebalance(band.half_width_shares + 1.0)
    assert not band.should_rebalance(band.half_width_shares - 1.0)


def test_decay_forecast_gamma_grows_atm_into_expiry():
    book = (Position(euro("call", 100, 10), 10),)
    steps = hedge_decay_forecast(book, flat_snapshot(), "XYZ", days=(1.0, 5.0, 9.0)).steps
    assert steps[0].exposure.gamma_shares < steps[-1].exposure.gamma_shares


@given(
    qty=st.integers(-200, 200),
    strike=st.sampled_from([90.0, 100.0, 110.0]),
    days=st.sampled_from([20, 40, 80]),
    aversion=st.sampled_from([0.1, 1.0, 10.0]),
)
@settings(max_examples=30, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_properties_of_every_proposal(qty, strike, days, aversion):
    book = (Position(euro("call", strike, days), qty),)
    p = solve(book=book, config=HedgeConfig(risk_aversion=aversion))
    assert p.verification.passed
    assert p.transaction_cost >= 0.0
    # An optimal solution cannot leave the penalized objective above doing nothing.
    before = value_book(book, flat_snapshot()).by_underlying["XYZ"]
    do_nothing = aversion * sum(
        t.effective_weight * abs(getattr(before, t.greek.value)) for t in DGV
    )
    assert p.objective <= do_nothing * (1 + 1e-3) + 1e-6


def test_empty_problem_rejected():
    from greeks_neutralizer.errors import ValidationError

    with pytest.raises(ValidationError, match="needs targets"):
        HedgeProblem(BOOK, CANDIDATES, ())
    with pytest.raises(ValidationError, match="requires scenarios"):
        HedgeProblem(BOOK, CANDIDATES, DGV, (), HedgeConfig(cvar_weight=1.0))


def test_taylor_check_on_small_shock_is_accurate():
    from greeks_neutralizer.risk.scenarios import taylor_vs_full

    small = taylor_vs_full(BOOK, flat_snapshot(), (Scenario("small", spot_shock=0.001),))
    assert abs(small[0].error) < 1.0
    large = taylor_vs_full(BOOK, flat_snapshot(), (Scenario("crash", spot_shock=-0.2),))
    assert abs(large[0].error) > abs(small[0].error)
