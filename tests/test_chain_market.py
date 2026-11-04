"""Historical-chain market: real quotes for costs and marks, and nothing read from later days."""

from __future__ import annotations

import csv
from dataclasses import replace
from datetime import date, datetime, time, timedelta
from pathlib import Path

import numpy as np
import pytest

from greeks_neutralizer.backtest.chains import (
    ChainMarket,
    ChainShortStrangle,
    implied_spot_and_carry,
    is_monthly,
    load_chain_day,
    projected_dividends,
    rate_on,
)
from greeks_neutralizer.backtest.engine import run_backtest
from greeks_neutralizer.backtest.policies import Unhedged, registered_policy
from greeks_neutralizer.backtest.proxy import ProxyMarket, ShortStrangleBook
from greeks_neutralizer.data.chain import ParityForward
from greeks_neutralizer.domain import NEW_YORK, OptionContract
from greeks_neutralizer.errors import MarketDataError
from greeks_neutralizer.market import CashDividend, DiscountCurve
from greeks_neutralizer.pricing.american import AmericanBatch, american_price
from greeks_neutralizer.time_model import calendar_years, us_expiry

RATE = 0.04
VOL = 0.20
EXPIRIES = (date(2021, 3, 19), date(2021, 4, 16), date(2021, 5, 21))
HEADER = (
    "date",
    "expiration",
    "strike",
    "call_put",
    "bid",
    "ask",
    "vol",
    "delta",
    "gamma",
    "underlying_close",
)


def _close(day: date) -> datetime:
    return datetime.combine(day, time(16, 0), tzinfo=NEW_YORK)


def write_chain(folder: Path, day: date, spot: float, *, half: float = 0.02, expiries=None) -> Path:
    """A chain priced by the repo's American engine at a flat vol, with no dividends. Strikes
    every 2.5 around spot, at fixed listed expiries, as a real chain has."""
    rows = []
    for expiry in expiries or EXPIRIES:
        T = calendar_years(_close(day), us_expiry(expiry))
        strikes = np.arange(np.floor(spot * 0.8 / 2.5) * 2.5, spot * 1.2, 2.5)
        for omega, label in ((1.0, "Call"), (-1.0, "Put")):
            n = strikes.size
            batch = AmericanBatch(
                spot=np.full(n, spot),
                strike=strikes,
                T=np.full(n, T),
                tau=np.full(n, T),
                r=np.full(n, RATE),
                q=np.zeros(n),
                sigma=np.full(n, VOL),
                omega=np.full(n, omega),
                div_times=np.zeros((n, 0)),
                div_amounts=np.zeros((n, 0)),
            )
            for strike, price in zip(strikes, american_price(batch), strict=True):
                bid = max(round(price - half, 2), 0.0)
                ask = round(price + half, 2)
                if ask <= 0.01:
                    continue
                rows.append(
                    (day, expiry, f"{strike:.2f}", label, bid, ask, "", "", "", f"{spot:.4f}")
                )
    path = folder / f"{day}.csv"
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(HEADER)
        writer.writerows(rows)
    return path


def market(files, **kwargs) -> ChainMarket:
    rates = {date(2020, 1, 1): RATE * 100.0}
    return ChainMarket("SPY", tuple(files), rates, (), **kwargs)


@pytest.fixture
def chain_files(tmp_path):
    start = date(2021, 3, 1)  # a Monday
    spots = (100.0, 101.0, 99.5, 100.5, 102.0)
    days = [start + timedelta(days=k) for k in (0, 2, 4, 7, 9)]
    return [write_chain(tmp_path, d, s) for d, s in zip(days, spots, strict=True)]


def test_day_uses_real_quotes_and_marks_at_the_quoted_mid(chain_files):
    """Oracle: the file's own bid and ask. A listed contract's cost is its quote plus the fee, and
    its mark is the quoted mid, not the model's value."""
    m = market(chain_files[:1])
    day = next(iter(m))
    chain = load_chain_day(chain_files[0])
    row = next(r for r in chain.rows if r.right.value == "put" and r.strike == 95.0)
    contract = next(
        c
        for c in day.listed
        if c.strike == 95.0 and c.right is row.right and c.expiry.date() == row.expiry
    )
    candidate = day.quote_for(contract)
    assert (candidate.quote.bid, candidate.quote.ask) == (row.bid, row.ask)
    assert candidate.fee_per_unit == 0.50
    assert day.observed_mid is not None
    assert day.observed_mid(contract) == pytest.approx(0.5 * (row.bid + row.ask))
    assert m.modeled_quotes == []
    # The hedge grid offers only contracts that were quoted.
    keys = {c.key for c in day.listed}
    assert all(c.instrument.key in keys for c in day.candidates[1:])


def test_spread_multiple_widens_around_the_mid(chain_files):
    base = next(iter(market(chain_files[:1])))
    wide = next(iter(market(chain_files[:1], spread_multiple=3.0)))
    contract = base.candidates[3].instrument
    a, b = base.quote_for(contract).quote, wide.quote_for(contract).quote
    assert b.mid == pytest.approx(a.mid)
    assert b.ask - b.bid == pytest.approx(3.0 * (a.ask - a.bid))


def test_unlisted_contract_is_priced_from_the_model_and_recorded(chain_files):
    m = market(chain_files[:1])
    day = next(iter(m))
    listed = day.listed[0]
    unlisted = replace(listed, strike=listed.strike + 0.5)
    candidate = day.quote_for(unlisted)
    assert candidate.quote.ask > candidate.quote.bid
    assert day.observed_mid is not None and day.observed_mid(unlisted) is None
    assert m.modeled_quotes == [f"{day.as_of.date()}: {unlisted.key}"]


def test_a_day_never_depends_on_later_files(chain_files):
    """Each day must be the same whether or not later days exist."""
    alone = next(iter(market(chain_files[:1])))
    with_later = next(iter(market(chain_files)))
    assert alone.snapshot.fingerprint() == with_later.snapshot.fingerprint()
    assert [c.quote for c in alone.candidates] == [c.quote for c in with_later.candidates]


def test_uncalibratable_day_is_skipped_and_recorded(chain_files, tmp_path):
    bad = tmp_path / "bad" / "2021-03-05.csv"
    bad.parent.mkdir()
    with chain_files[2].open(newline="") as handle:
        rows = list(csv.reader(handle))
    with bad.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(rows[0])
        # Every quote crossed, so the loader drops them all and nothing can be calibrated.
        for row in rows[1:]:
            writer.writerow([*row[:4], row[5], row[4], *row[6:]])
    files = [*chain_files[:2], bad, *chain_files[3:]]
    m = market(files)
    days = [d.as_of.date() for d in m]
    assert date(2021, 3, 5) not in days
    assert len(days) == len(files) - 1
    assert [s.day for s in m.skipped] == [date(2021, 3, 5)]
    assert "MarketDataError" in m.skipped[0].reason


def test_strangle_trades_listed_contracts_and_pnl_runs_on_mids(chain_files):
    m = market(chain_files)
    result = run_backtest(m, ChainShortStrangle("SPY", lots=10), Unhedged())
    assert len(result.days) == len(chain_files) - 1
    assert all(d.option_positions == 2 for d in result.days)
    # Every held contract was quoted on every day, so no mark came from the model.
    assert all(d.model_marked == 0 for d in result.days)
    assert m.modeled_quotes == []


def test_registered_optimizer_hedges_on_a_real_chain_day(chain_files):
    m = market(chain_files[:3])
    result = run_backtest(
        m, ChainShortStrangle("SPY", lots=10), registered_policy("opt_dgv", "SPY")
    )
    assert result.failures == []
    assert sum(d.hedge_lots for d in result.days) > 0


def test_engine_marks_at_observed_mids_when_given():
    """Oracle: hand-set mids. The book's P&L must be the change in quantity x multiplier x mid,
    whatever the model says, and a missing mid falls back to the model and is counted."""
    proxy = list(ProxyMarket(_history(6)))
    holder = ShortStrangleBook(lots=1)
    mids = iter([5.0, 5.5, 4.0, 4.25, None, 3.0])
    days = []
    for day in proxy:
        mid = next(mids)
        days.append(replace(day, observed_mid=(lambda c, mid=mid: mid)))
    result = run_backtest(days, holder, Unhedged())
    held = [d for d in result.days if d.option_positions]
    assert held, "the book should hold options"
    first = result.days[0]
    # Two short legs, each marked at the same hand-set mid: P&L is -2 x 100 x (5.5 - 5.0).
    assert first.book_pnl == pytest.approx(-2 * 100 * (5.5 - 5.0))
    assert result.days[3].model_marked == 2  # day 4 has no mid, so its end marks are modeled
    assert result.days[4].model_marked == 2  # and that is the next period's start
    assert result.days[2].model_marked == 0
    # Two legs of one lot, 100 units each, each with one modeled mark in each of those periods.
    assert result.days[3].modeled_units_sq == pytest.approx(2 * 100.0**2)
    assert result.days[4].modeled_units_sq == pytest.approx(2 * 100.0**2)
    assert result.days[2].modeled_units_sq == 0.0


def _history(n):
    from greeks_neutralizer.backtest.proxy import HistoryRow

    start = date(2021, 1, 4)
    return [HistoryRow(start + timedelta(days=k), 4000.0 + k, 0.2, 0.04) for k in range(n)]


def test_projected_dividends_use_only_amounts_paid_by_the_day():
    paid = ((date(2024, 3, 15), 1.59), (date(2024, 6, 21), 1.76), (date(2024, 9, 20), 1.75))
    day = date(2024, 7, 1)
    projected = projected_dividends(paid, day, horizon_days=200)
    # Changing a dividend paid after the day cannot change what was known on it.
    assert projected == projected_dividends((*paid[:2], (date(2024, 9, 20), 9.99)), day, 200)
    assert [d.amount for d in projected] == [1.76, 1.76]
    assert [d.ex_date.date() for d in projected] == [date(2024, 9, 20), date(2024, 12, 20)]
    assert projected_dividends(paid, date(2024, 1, 1)) == ()


def test_rate_is_the_last_published_on_or_before_the_day():
    rates = {date(2024, 1, 2): 5.0, date(2024, 1, 5): 4.0}
    assert rate_on(rates, date(2024, 1, 1)) is None
    assert rate_on(rates, date(2024, 1, 4)) == pytest.approx(0.05)
    assert rate_on(rates, date(2024, 1, 5)) == pytest.approx(0.04)


def test_loader_drops_unusable_rows_and_checks_the_file(tmp_path):
    path = tmp_path / "2024-01-02.csv"
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(HEADER)
        writer.writerow(
            ("2024-01-02", "2024-01-19", "470", "Call", "2.0", "2.1", "", "", "", "472")
        )
        writer.writerow(("2024-01-02", "2024-01-19", "475", "Call", "", "1.1", "", "", "", "472"))
        writer.writerow(
            ("2024-01-02", "2024-01-19", "480", "Call", "0.6", "0.5", "", "", "", "472")
        )
    chain = load_chain_day(path)
    assert len(chain.rows) == 1 and chain.close == 472.0
    misnamed = tmp_path / "2024-01-03.csv"
    misnamed.write_text(path.read_text())
    with pytest.raises(MarketDataError, match="does not match"):
        load_chain_day(misnamed)


def test_contracts_are_american_and_physically_settled(chain_files):
    day = next(iter(market(chain_files[:1])))
    contract = day.listed[0]
    assert isinstance(contract, OptionContract)
    assert contract.style.value == "american"
    assert contract.settlement.value == "physical"
    assert contract.expiry.time() == time(16, 0)


def test_parity_spot_and_carry_recover_the_market_that_made_the_forwards():
    """Oracle: forwards built from a known spot, carry, and dividend by the model's own forward
    formula, (S - PV) exp(-q T) / D, written out here independently."""
    as_of = _close(date(2024, 1, 2))
    curve = DiscountCurve.flat(0.05)
    dividend = CashDividend(_close(date(2024, 3, 15)), 1.5)
    spot, carry = 470.0, 0.004
    forwards = []
    for days_out in (17, 45, 80):
        expiry = _close(date(2024, 1, 2) + timedelta(days=days_out))
        T = calendar_years(as_of, expiry)
        D = float(np.exp(-0.05 * T))
        pv = (
            dividend.amount * float(np.exp(-0.05 * calendar_years(as_of, dividend.ex_date)))
            if dividend.ex_date <= expiry
            else 0.0
        )
        forward = (spot - pv) * np.exp(-carry * T) / D
        forwards.append(ParityForward(expiry, T, float(forward), D, 10, 0.0))
    got_spot, got_carry = implied_spot_and_carry(forwards, (dividend,), curve, as_of)
    assert got_spot == pytest.approx(spot, rel=1e-6)
    assert got_carry == pytest.approx(carry, abs=1e-5)
    # With one expiry there is nothing to separate carry from spot, so carry is zero.
    one_spot, one_carry = implied_spot_and_carry(forwards[:1], (), curve, as_of)
    assert one_carry == 0.0
    assert one_spot == pytest.approx(forwards[0].forward * forwards[0].discount)


def test_close_is_a_check_on_parity_spot_not_its_source(tmp_path):
    """Spot comes from the options. A close 3% away, as an asynchronous snapshot gives, moves
    nothing but the recorded gap. One 10% away means the files describe different markets."""
    clean = write_chain(tmp_path, date(2021, 3, 1), 100.0)
    for close, folder in ((103.0, "near"), (110.0, "far")):
        (tmp_path / folder).mkdir()
        path = tmp_path / folder / clean.name
        path.write_text(clean.read_text().replace("100.0000", f"{close:.4f}"))
        m = market([path])
        days = list(m)
        if close == 103.0:
            reference = next(iter(market([clean])))
            assert days[0].snapshot.market("SPY").spot == pytest.approx(
                reference.snapshot.market("SPY").spot, rel=2e-3
            )
            assert m.close_gaps[0] == pytest.approx(100.0 / 103.0 - 1.0, abs=2e-3)
        else:
            assert days == []
            assert len(m.skipped) == 1


def test_monthly_expiries_include_the_holiday_thursday():
    assert is_monthly(us_expiry(date(2021, 3, 19)))
    assert is_monthly(us_expiry(date(2022, 4, 14)))  # Good Friday was the third Friday
    assert not is_monthly(us_expiry(date(2021, 3, 26)))


def test_strangle_rolls_into_the_listed_monthly_not_the_nearest_weekly(tmp_path):
    """The proxy books trade monthlies. A weekly 22 days out is skipped for the monthly 46 days
    out, which the source keeps quoting for longer."""
    expiries = (date(2021, 3, 12), date(2021, 3, 26), date(2021, 4, 16))
    path = write_chain(tmp_path, date(2021, 3, 1), 100.0, expiries=expiries)
    day = next(iter(market([path])))
    trades = ChainShortStrangle("SPY", lots=10).trades(day, {})
    assert {t.instrument.expiry.date() for t in trades} == {date(2021, 4, 16)}


def test_mark_audit_scores_held_out_contracts_against_their_quotes(chain_files):
    """On a chain priced by one flat-vol model, a held-out contract's mark from the rest of the
    chain must match its quote to within rounding and spread: the generator is the oracle."""
    m = market(chain_files[:1])
    audit = m.mark_audit(load_chain_day(chain_files[0]))
    assert audit.strike_holdout and audit.expiry_holdout
    for error, half in (*audit.strike_holdout, *audit.expiry_holdout):
        assert abs(error) < 0.03
        assert half == pytest.approx(0.02, abs=0.006)
