"""Exchange calendar adapter, including how a holiday changes variance time."""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from greeks_neutralizer.calendars import exchange_session_predicate
from greeks_neutralizer.errors import MarketDataError
from greeks_neutralizer.time_model import VarianceClock, us_expiry

pytest.importorskip("exchange_calendars")


def test_nyse_holidays_and_weekends():
    is_session = exchange_session_predicate("XNYS")
    assert is_session(date(2026, 11, 2))
    assert not is_session(date(2026, 11, 26))  # Thanksgiving
    assert not is_session(date(2026, 11, 7))  # Saturday


def test_out_of_range_fails_closed():
    is_session = exchange_session_predicate("XNYS")
    with pytest.raises(MarketDataError, match="outside"):
        is_session(date(1900, 1, 2))


def test_holiday_removes_session_variance():
    weights = {"session_weight": 1.0, "overnight_weight": 0.1, "closed_day_weight": 0.0}
    aware = VarianceClock(**weights, is_session_day=exchange_session_predicate("XNYS"))
    naive = VarianceClock(**weights)
    start = us_expiry(date(2026, 11, 25))
    end = start + timedelta(days=2)
    assert aware.tau(start, end) < naive.tau(start, end)
