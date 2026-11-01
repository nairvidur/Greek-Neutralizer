"""Exception hierarchy. Every failure the library can raise on purpose derives from GNError."""

from __future__ import annotations


class GNError(Exception):
    """Base class for intentional library errors."""


class ValidationError(GNError, ValueError):
    """An input violates a domain invariant (bad sign, NaN, naive datetime, unknown enum)."""


class MarketDataError(GNError):
    """Market data is missing or unusable. The engine fails closed rather than guessing."""


class ConvergenceError(GNError):
    """A numerical routine could not reach its stated tolerance."""


class HedgeInfeasibleError(GNError):
    """No hedge satisfies the hard constraints. Carries the diagnosed binding constraints."""

    def __init__(self, message: str, binding: tuple[str, ...] = ()) -> None:
        super().__init__(message)
        self.binding = binding
