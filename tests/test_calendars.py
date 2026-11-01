"""Exchange calendar adapter, including how a holiday changes variance time."""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from greeks_neutralizer.calendars import exchange_session_predicate
from greeks_neutralizer.errors import MarketDataError
from greeks_neutralizer.time_model import VarianceClock, us_expiry

pytest.importorskip("exchange_calendars")


def _thanksgiving():
    # The fourth Thursday of November this year, by the US rule, so always inside the library's
    # calendar range, which moves with today's date and starts 20 years back.
    year = date.today().year
    first = date(year, 11, 1)
    return first + timedelta(days=(3 - first.weekday()) % 7 + 21)


def test_nyse_holidays_and_weekends():
    is_session = exchange_session_predicate("XNYS")
    thanksgiving = _thanksgiving()
    assert not is_session(thanksgiving)
    assert is_session(thanksgiving - timedelta(days=1))  # The Wednesday before
    assert not is_session(thanksgiving + timedelta(days=2))  # Saturday


def test_out_of_range_fails_closed():
    is_session = exchange_session_predicate("XNYS")
    with pytest.raises(MarketDataError, match="outside"):
        is_session(date(1900, 1, 2))


def test_holiday_removes_session_variance():
    weights = {"session_weight": 1.0, "overnight_weight": 0.1, "closed_day_weight": 0.0}
    aware = VarianceClock(**weights, is_session_day=exchange_session_predicate("XNYS"))
    naive = VarianceClock(**weights)
    start = us_expiry(_thanksgiving() - timedelta(days=1))
    end = start + timedelta(days=2)
    assert aware.tau(start, end) < naive.tau(start, end)
