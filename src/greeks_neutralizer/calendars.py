"""Exchange calendars for `VarianceClock`. Requires the `calendars` extra (exchange-calendars)."""

from __future__ import annotations

from collections.abc import Callable
from datetime import date

from greeks_neutralizer.errors import MarketDataError


def exchange_session_predicate(code: str = "XNYS") -> Callable[[date], bool]:
    """Whether a date is a trading session on the exchange, including holidays.

    Dates outside the library's calendar range fail closed instead of guessing: a wrong answer
    would quietly misstate variance time.
    """
    try:
        import exchange_calendars
    except ImportError as exc:
        raise MarketDataError(
            "exchange calendars need the 'calendars' extra: "
            "pip install 'greeks-neutralizer[calendars]'"
        ) from exc

    calendar = exchange_calendars.get_calendar(code)
    first = calendar.first_session.date()
    last = calendar.last_session.date()

    def is_session(day: date) -> bool:
        if not first <= day <= last:
            raise MarketDataError(f"{day} is outside the {code} calendar range {first} to {last}")
        return bool(calendar.is_session(day.isoformat()))

    return is_session
