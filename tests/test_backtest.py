"""Backtest accounting, no look-ahead, policies, and the proxy market."""

from __future__ import annotations

from dataclasses import replace
from datetime import date, datetime, time, timedelta

import numpy as np
import pytest
from numpy.testing import assert_allclose

from greeks_neutralizer.backtest.engine import BacktestResult, DayMarket, DayResult, run_backtest
from greeks_neutralizer.backtest.policies import DeltaBand, DeltaDaily, OptimizerPolicy, Unhedged
from greeks_neutralizer.backtest.proxy import (
    HistoryRow,
    LongPutOverlayBook,
    ProxyMarket,
    ProxySmile,
    ShortStrangleBook,
    SpreadModel,
    join_history,
    monthly_expiries,
    third_friday,
)
from greeks_neutralizer.backtest.report import compare, max_drawdown, policy_stats, stress_days
from greeks_neutralizer.domain import (
    NEW_YORK,
    LinearInstrument,
    OptionContract,
    OptionRight,
    Position,
    Quote,
)
from greeks_neutralizer.errors import MarketDataError, ValidationError
from greeks_neutralizer.hedging.problem import GreekName, GreekTarget, HedgeCandidate, HedgeConfig
from greeks_neutralizer.market import DiscountCurve, FlatVol, MarketSnapshot, UnderlyingMarket
from greeks_neutralizer.pricing.black import black_greeks
from greeks_neutralizer.time_model import calendar_years

START = date(2024, 1, 2)


def history(spots, vix=0.2, rate=0.0, start=START):
    days = []
    current = start
    while len(days) < len(spots):
        if current.weekday() < 5:
            days.append(current)
        current += timedelta(days=1)
    return tuple(HistoryRow(d, float(s), vix, rate) for d, s in zip(days, spots, strict=True))


def rising(n, step=1.0):
    return history(4000.0 + step * np.arange(n))


class FixedBook:
    """Holds one position, opened once."""

    def __init__(self, position):
        self.position = position

    def trades(self, day, holdings):
        return () if holdings else (self.position,)


class NoCostMarket:
    def __init__(self, spots, vol=0.2, rate=0.0):
        self.spots, self.vol, self.rate = spots, vol, rate

    def __iter__(self):
        for i, spot in enumerate(self.spots):
            as_of = datetime.combine(START + timedelta(days=i), time(16), tzinfo=NEW_YORK)
            market = UnderlyingMarket(
                "XYZ", float(spot), DiscountCurve.flat(self.rate), FlatVol(self.vol)
            )
            snapshot = MarketSnapshot(as_of, {"XYZ": market})

            def quote_for(instrument, as_of=as_of, spot=spot):
                return HedgeCandidate(instrument, Quote(1.0, 1.0, 10**6, 10**6, as_of))

            from greeks_neutralizer.domain import LinearInstrument

            yield DayMarket(
                as_of, snapshot, (quote_for(LinearInstrument("XYZ", "XYZ")),), quote_for
            )


def xyz_call(strike=100.0, days=30):
    from greeks_neutralizer.domain import ExerciseStyle, OptionContract, SettlementKind

    return OptionContract(
        "XYZ",
        OptionRight.CALL,
        strike,
        datetime.combine(START + timedelta(days=days), time(16), tzinfo=NEW_YORK),
        ExerciseStyle.EUROPEAN,
        settlement=SettlementKind.CASH,
    )


def test_unhedged_pnl_equals_change_in_value_by_hand():
    contract = xyz_call()
    spots = [100.0, 103.0]
    result = run_backtest(NoCostMarket(spots), FixedBook(Position(contract, 2)), Unhedged())
    v0 = black_greeks(100.0, 100.0, 30 / 365, 0.0, 0.0, 0.2, 1).price[0]
    v1 = black_greeks(103.0, 100.0, 29 / 365, 0.0, 0.0, 0.2, 1).price[0]
    assert_allclose(result.days[0].book_pnl, 200 * (v1 - v0))
    assert_allclose(result.days[0].pnl, 200 * (v1 - v0))


def test_cost_is_charged_once_at_half_spread():
    contract = xyz_call()

    class Spread(NoCostMarket):
        def __iter__(self):
            for day in super().__iter__():

                def quote_for(instrument, as_of=day.as_of):
                    return HedgeCandidate(instrument, Quote(1.0, 1.2, 100, 100, as_of))

                yield replace(day, quote_for=quote_for)

    result = run_backtest(
        Spread([100.0, 100.0, 100.0]), FixedBook(Position(contract, 3)), Unhedged()
    )
    assert_allclose(result.days[0].book_cost, 3 * 100 * 0.1)
    assert result.days[1].book_cost == 0.0


def test_financing_charges_interest_on_marked_value():
    contract = xyz_call(days=365)
    result = run_backtest(
        NoCostMarket([100.0, 100.0], rate=0.05), FixedBook(Position(contract, 1)), Unhedged()
    )
    value = 100 * black_greeks(100.0, 100.0, 1.0, 0.05, 0.0, 0.2, 1).price[0]
    assert_allclose(result.days[0].financing, value * 0.05 / 365, rtol=1e-6)


class OpenOnce:
    """Opens one position on the first day, and never trades again."""

    def __init__(self, position):
        self.position = position
        self.opened = False

    def trades(self, day, holdings):
        if self.opened:
            return ()
        self.opened = True
        return (self.position,)


def test_expiry_settles_at_intrinsic_then_position_disappears():
    contract = xyz_call(days=1)
    # Spot keeps moving after expiry, so a position that had not gone would keep making P&L.
    spots = [100.0, 110.0, 120.0, 130.0]
    result = run_backtest(NoCostMarket(spots), OpenOnce(Position(contract, 1)), Unhedged())
    v0 = black_greeks(100.0, 100.0, 1 / 365, 0.0, 0.0, 0.2, 1).price[0]
    assert_allclose(result.days[0].book_pnl, 100 * (10.0 - v0))
    assert [day.book_pnl for day in result.days[1:]] == [0.0, 0.0]


def test_delta_hedge_cuts_variance_in_a_black_scholes_world():
    rng = np.random.default_rng(3)
    returns = rng.normal(0.0, 0.2 / np.sqrt(365), 25)
    spots = 100.0 * np.exp(np.cumsum(np.concatenate([[0.0], returns])))
    contract = xyz_call(days=40)
    book = Position(contract, -10)
    naked = run_backtest(NoCostMarket(spots), FixedBook(book), Unhedged())
    hedged = run_backtest(NoCostMarket(spots), FixedBook(book), DeltaDaily("XYZ"))
    assert np.var(hedged.pnl()) < 0.2 * np.var(naked.pnl())


def test_decisions_cannot_see_the_future():
    # Changing every price after day k must not change any decision made through day k.
    decisions = []

    class Spy:
        name = "spy"

        def trades(self, day, book, hedges):
            result = DeltaDaily("SPX").trades(day, book, hedges)
            decisions.append((day.as_of, tuple(t.quantity for t in result)))
            return result

    base = rising(30)
    book = ShortStrangleBook()
    run_backtest(ProxyMarket(base), book, Spy())
    first = list(decisions)
    decisions.clear()
    k = 15
    altered = base[:k] + tuple(replace(r, spot=r.spot * 1.5, vix=0.8) for r in base[k:])
    run_backtest(ProxyMarket(altered), ShortStrangleBook(), Spy())
    assert first[:k] == decisions[:k]
    assert first[k:] != decisions[k:]


def test_policy_failures_are_recorded_not_hidden():
    class Broken:
        name = "broken"

        def trades(self, day, book, hedges):
            raise MarketDataError("no quotes")

    result = run_backtest(rising_market(5), ShortStrangleBook(), Broken())
    assert len(result.failures) == 5
    assert sum(d.failures for d in result.days) == 4
    with pytest.raises(MarketDataError, match="no quotes"):
        run_backtest(rising_market(5), ShortStrangleBook(), Broken(), fail_fast=True)


def rising_market(n):
    return ProxyMarket(rising(n))


def test_days_must_increase():
    market = ProxyMarket(rising(3))
    days = [market.day(1), market.day(0)]
    with pytest.raises(ValidationError, match="increasing"):
        run_backtest(days, ShortStrangleBook(), Unhedged())


def test_optimizer_policy_runs_and_reduces_risk_on_real_shape():
    rng = np.random.default_rng(5)
    returns = rng.normal(0.0, 0.012, 60)
    spots = 4000.0 * np.exp(np.cumsum(returns))
    hist = history(spots, vix=0.2)
    targets = tuple(
        GreekTarget("SPX", greek, weight=weight)
        for greek, weight in (
            (GreekName.DELTA, 0.01),
            (GreekName.GAMMA, 0.045),
            (GreekName.VEGA, 2.0),
        )
    )
    policy = OptimizerPolicy(
        "dgv",
        targets,
        HedgeConfig(risk_aversion=1.0, require_quote_size=True),
    )
    naked = run_backtest(ProxyMarket(hist), ShortStrangleBook(), Unhedged())
    hedged = run_backtest(ProxyMarket(hist), ShortStrangleBook(), policy, fail_fast=True)
    assert hedged.failures == []
    assert np.std(hedged.pnl()) < np.std(naked.pnl())


def test_band_policy_trades_less_than_daily():
    rng = np.random.default_rng(9)
    spots = 4000.0 * np.exp(np.cumsum(rng.normal(0.0, 0.008, 60)))
    hist = history(spots)
    daily = run_backtest(ProxyMarket(hist), ShortStrangleBook(), DeltaDaily("SPX"))
    band = run_backtest(ProxyMarket(hist), ShortStrangleBook(), DeltaBand("SPX", 5e-5, 1e-6))
    assert policy_stats(band).hedge_lots < policy_stats(daily).hedge_lots


def test_long_put_overlay_opens_once_and_rolls():
    class Recording(LongPutOverlayBook):
        def trades(self, day, holdings):
            done = super().trades(day, holdings)
            log.append((day.as_of, tuple(done)))
            return done

    log = []
    result = run_backtest(rising_market(40), Recording(), Unhedged())
    assert len(result.days) == 39
    assert result.days[0].book_cost > 0.0

    active = [(as_of, trades) for as_of, trades in log if trades]
    opening = active[0][1]
    assert len(opening) == 2
    assert sum(isinstance(t.instrument, LinearInstrument) for t in opening) == 1
    assert sum(isinstance(t.instrument, OptionContract) for t in opening) == 1
    # Shares are bought once, whatever else happens.
    linear = [t for _, trades in log for t in trades if isinstance(t.instrument, LinearInstrument)]
    assert len(linear) == 1

    # A roll closes the old put with 10 days or less to go, and buys a later one.
    assert len(active) >= 2
    held = next(t for t in opening if isinstance(t.instrument, OptionContract))
    for as_of, trades in active[1:]:
        closing, buying = trades
        assert closing.instrument == held.instrument
        assert closing.quantity == -held.quantity
        assert calendar_years(as_of, held.instrument.expiry) * 365.0 <= 10.0
        assert buying.instrument.right is OptionRight.PUT
        assert buying.instrument.expiry > held.instrument.expiry
        assert buying.quantity == held.quantity
        held = buying


def test_third_friday_and_expiries():
    assert third_friday(2026, 11) == date(2026, 11, 20)
    assert third_friday(2026, 5) == date(2026, 5, 15)
    assert monthly_expiries(date(2026, 11, 20), 2) == [date(2026, 12, 18), date(2027, 1, 15)]
    assert monthly_expiries(date(2026, 12, 25), 1) == [date(2027, 1, 15)]


def test_history_join_never_uses_a_future_rate():
    spot = {date(2024, 1, 2): 100.0, date(2024, 1, 3): 101.0, date(2024, 1, 4): 102.0}
    vix = dict.fromkeys(spot, 20.0)
    rates = {date(2024, 1, 3): 5.0}
    rows = join_history(spot, vix, rates)
    assert [r.day for r in rows] == [date(2024, 1, 3), date(2024, 1, 4)]
    assert rows[0].rate == 0.05


def test_proxy_smile_shape_and_term():
    smile = ProxySmile(0.3, 0.2)
    tau = np.array([30 / 365] * 3)
    k = np.array([-0.1, 0.0, 0.1])
    vol = smile.vol(k, tau)
    assert vol[0] > vol[1] > vol[2]
    assert_allclose(smile.atm(np.array(30 / 365)), 0.3)
    assert smile.atm(np.array(2.0)) < 0.25


def test_spread_model_scales_with_vix_and_multiple():
    spreads = SpreadModel()
    assert spreads.option_half(100.0, 0.30) > spreads.option_half(100.0, 0.15)
    assert_allclose(
        SpreadModel(multiple=2.0).option_half(100.0, 0.2), 2.0 * spreads.option_half(100.0, 0.2)
    )


def _fake(name, pnl):
    days = [
        DayResult(
            datetime(2024, 1, 1, tzinfo=NEW_YORK) + timedelta(days=i), float(x), 0, 0, 0, 0, 0, 0, 0
        )
        for i, x in enumerate(pnl)
    ]
    return BacktestResult(name, days)


def test_comparison_detects_a_real_reduction_with_ci_excluding_zero():
    rng = np.random.default_rng(11)
    wild = rng.normal(0.0, 10.0, 1000)
    calm = 0.1 * wild + rng.normal(0.0, 1.0, 1000)
    c = compare(_fake("calm", calm), _fake("wild", wild))
    assert 0.9 < c.variance_reduction < 1.0
    assert c.variance_reduction_ci[0] > 0.8
    assert c.es_improvement > 0.0


def test_comparison_does_not_claim_a_difference_in_noise():
    rng = np.random.default_rng(12)
    a, b = rng.normal(0.0, 1.0, 750), rng.normal(0.0, 1.0, 750)
    c = compare(_fake("a", a), _fake("b", b))
    low, high = c.variance_reduction_ci
    assert low < 0.0 < high
    low, high = c.mean_pnl_difference_ci
    assert low < 0.0 < high


def test_comparison_requires_same_days():
    with pytest.raises(ValidationError, match="same days"):
        compare(_fake("a", np.zeros(50)), _fake("b", np.zeros(60)))


def test_stats_and_drawdown():
    assert_allclose(max_drawdown(np.array([1.0, -3.0, 1.0, 5.0, -2.0])), 3.0)
    stats = policy_stats(_fake("x", np.arange(50.0) - 25.0))
    assert stats.days == 50
    assert stats.worst_day == -25.0
    with pytest.raises(ValidationError, match="no days"):
        policy_stats(_fake("x", []))


def test_stress_days_reads_requested_dates_only():
    result = _fake("x", np.arange(5.0))
    wanted = [date(2024, 1, 3), date(2030, 1, 1)]
    rows = stress_days([result], wanted)
    assert len(rows) == 1
    assert rows[0].pnl_by_policy["x"] == 2.0
