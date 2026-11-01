"""Time to expiry: calendar time for discounting, and variance time for volatility.

A float T in years is not adequate for 0DTE options, where the distinction between minutes during
the session and hours overnight dominates the premium. `VarianceClock` weights calendar time by
regime so that a Friday afternoon option does not accrue weekend variance as if the market were
open.

The default clock is ACT/365 with equal weights, which is the market's quoting convention and what
most vendors' implied vols assume. When a non-default clock is used, implied vols must be derived
with the same clock; the engine never mixes them because the clock travels with the market snapshot.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo

from greeks_neutralizer.domain import OptionContract, require_aware
from greeks_neutralizer.errors import ValidationError

NEW_YORK = ZoneInfo("America/New_York")
SECONDS_PER_YEAR = 365.0 * 86400.0


def calendar_years(start: datetime, end: datetime) -> float:
    """ACT/365 year fraction between two aware instants. Non-negative; past expiries give 0."""
    require_aware("start", start)
    require_aware("end", end)
    # Through timestamps: Python subtracts wall-clock time when both share a tzinfo, which is an
    # hour wrong across a DST change and would give one instant two prices.
    return max(end.timestamp() - start.timestamp(), 0.0) / SECONDS_PER_YEAR


def us_expiry(settlement_date: date, am_settled: bool = False) -> datetime:
    """Value-fixing instant for US listed options: 09:30 ET if AM-settled, else 16:00 ET.

    The AM value is fixed from opening prints, so 09:30 is the point when market risk ends even
    though the settlement value is published later. Zone-aware, so DST is handled.
    """
    fixing = time(9, 30) if am_settled else time(16, 0)
    return datetime.combine(settlement_date, fixing, tzinfo=NEW_YORK)


@dataclass(frozen=True, slots=True)
class VarianceClock:
    """Maps a time interval to variance-years: integrated weights scaled so a calendar year is ~1.

    weights: relative variance per hour during regular session, overnight, and non-trading days.
    is_session_day: holiday-aware predicate. Weekday-only by default; supply a real exchange
    calendar with `calendars.exchange_session_predicate` for production.
    events: (instant, extra variance in variance-years) for earnings or macro releases, added when
    the interval spans them.
    """

    session_weight: float = 1.0
    overnight_weight: float = 1.0
    closed_day_weight: float = 1.0
    is_session_day: Callable[[date], bool] = lambda d: d.weekday() < 5
    events: tuple[tuple[datetime, float], ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        for name in ("session_weight", "overnight_weight", "closed_day_weight"):
            if getattr(self, name) < 0.0:
                raise ValidationError(f"{name} must be non-negative")
        for instant, extra in self.events:
            require_aware("event instant", instant)
            if extra < 0.0:
                raise ValidationError("event variance must be non-negative")

    @property
    def is_calendar(self) -> bool:
        return (
            self.session_weight == self.overnight_weight == self.closed_day_weight
            and self.session_weight == 1.0
            and not self.events
        )

    def tau(self, start: datetime, end: datetime) -> float:
        """Variance-years between two instants. Exactly equals calendar_years for the default."""
        require_aware("start", start)
        require_aware("end", end)
        if end <= start:
            return 0.0
        if self.is_calendar:
            return calendar_years(start, end)
        return self._weighted_seconds(start, end) / self._year_normalizer() + sum(
            extra for instant, extra in self.events if start < instant <= end
        )

    def _year_normalizer(self) -> float:
        """Weighted seconds in a standard year, so weights only redistribute variance."""
        session_days = 252.0
        closed_days = 365.0 - session_days
        session_seconds = 6.5 * 3600.0
        return session_days * (
            session_seconds * self.session_weight
            + (86400.0 - session_seconds) * self.overnight_weight
        ) + (closed_days * 86400.0 * self.closed_day_weight)

    def _weighted_seconds(self, start: datetime, end: datetime) -> float:
        start_et = start.astimezone(NEW_YORK)
        end_et = end.astimezone(NEW_YORK)
        total = 0.0
        day = start_et.date()
        while day <= end_et.date():
            midnight = datetime.combine(day, time(0, 0), tzinfo=NEW_YORK)
            next_midnight = datetime.combine(day + timedelta(days=1), time(0, 0), tzinfo=NEW_YORK)
            lo = max(start_et, midnight)
            hi = min(end_et, next_midnight)
            if hi > lo:
                total += self._day_weighted(day, lo, hi)
            day += timedelta(days=1)
        return total

    def _day_weighted(self, day: date, lo: datetime, hi: datetime) -> float:
        seconds = hi.timestamp() - lo.timestamp()
        if not self.is_session_day(day):
            return seconds * self.closed_day_weight
        open_ = datetime.combine(day, time(9, 30), tzinfo=NEW_YORK)
        close = datetime.combine(day, time(16, 0), tzinfo=NEW_YORK)
        in_session = max(min(hi, close).timestamp() - max(lo, open_).timestamp(), 0.0)
        return in_session * self.session_weight + (seconds - in_session) * self.overnight_weight


CALENDAR_CLOCK = VarianceClock()


def is_expired(contract: OptionContract, as_of: datetime) -> bool:
    return contract.expiry <= as_of
