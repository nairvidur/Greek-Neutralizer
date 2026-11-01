"""Exception hierarchy. Every failure the library can raise on purpose derives from GNError."""

from __future__ import annotations


class GNError(Exception):
    """Base class for intentional library errors."""


class ValidationError(GNError, ValueError):
    """An input violates a domain invariant (bad sign, NaN, naive datetime, unknown enum)."""


class SolverRangeError(ValidationError):
    """The inputs are past what the solver can represent faithfully, so nothing it says could be
    trusted. This says nothing about whether a hedge exists."""


class MarketDataError(GNError):
    """Market data is missing or unusable. The engine fails closed rather than guessing."""


class HedgeInfeasibleError(GNError):
    """No hedge satisfies the hard constraints. Carries the diagnosed binding constraints."""

    def __init__(self, message: str, binding: tuple[str, ...] = ()) -> None:
        super().__init__(message)
        self.binding = binding


class SearchLimitError(GNError):
    """The search stopped at a limit or solver failure with no hedge. This proves nothing either
    way about whether a hedge exists, so it must never be reported as infeasibility."""
