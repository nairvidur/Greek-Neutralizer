"""Quote quality gates. Every quote gets flags; nothing unusable reaches a model silently."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import StrEnum

from greeks_neutralizer.domain import Quote


class QuoteFlag(StrEnum):
    NO_OFFER = "no_offer"
    CROSSED = "crossed"
    LOCKED = "locked"
    ZERO_BID = "zero_bid"
    NO_SIZE = "no_size"
    WIDE_SPREAD = "wide_spread"
    STALE = "stale"
    FUTURE_TIMESTAMP = "future_timestamp"
    BELOW_INTRINSIC = "below_intrinsic"
    IV_OUTLIER = "iv_outlier"


# Flags that make a quote unusable for marking or hedging. The rest are warnings.
BLOCKING: frozenset[QuoteFlag] = frozenset(
    {
        QuoteFlag.NO_OFFER,
        QuoteFlag.CROSSED,
        QuoteFlag.STALE,
        QuoteFlag.FUTURE_TIMESTAMP,
        QuoteFlag.BELOW_INTRINSIC,
        QuoteFlag.NO_SIZE,
    }
)


@dataclass(frozen=True, slots=True)
class QualityPolicy:
    max_age: timedelta = timedelta(seconds=30)
    max_relative_spread: float = 0.5
    min_absolute_spread_exempt: float = 0.10
    zero_bid_blocks_selling: bool = True
    # False only for a source known not to populate sizes, so absence is not read as "no market".
    require_size: bool = True


def assess_quote(
    quote: Quote, now: datetime, intrinsic: float = 0.0, policy: QualityPolicy | None = None
) -> frozenset[QuoteFlag]:
    p = policy or QualityPolicy()
    flags: set[QuoteFlag] = set()
    # An ask of zero is no offer at all, not a free option.
    if quote.ask <= 0.0:
        flags.add(QuoteFlag.NO_OFFER)
    if quote.bid > quote.ask:
        flags.add(QuoteFlag.CROSSED)
    elif quote.bid == quote.ask and quote.ask > 0.0:
        flags.add(QuoteFlag.LOCKED)
    if quote.bid == 0.0:
        flags.add(QuoteFlag.ZERO_BID)
    if p.require_size and quote.bid_size == 0 and quote.ask_size == 0:
        flags.add(QuoteFlag.NO_SIZE)
    if quote.timestamp > now + timedelta(seconds=1):
        flags.add(QuoteFlag.FUTURE_TIMESTAMP)
    elif now - quote.timestamp > p.max_age:
        flags.add(QuoteFlag.STALE)
    spread = quote.ask - quote.bid
    if (
        quote.mid > 0.0
        and spread / quote.mid > p.max_relative_spread
        and spread > p.min_absolute_spread_exempt
    ):
        flags.add(QuoteFlag.WIDE_SPREAD)
    # Allow a tick of tolerance so a fair bid resting at parity is not rejected for rounding.
    if quote.ask < intrinsic - 0.01:
        flags.add(QuoteFlag.BELOW_INTRINSIC)
    return frozenset(flags)


def is_usable(flags: frozenset[QuoteFlag]) -> bool:
    return not flags & BLOCKING


@dataclass(frozen=True, slots=True)
class QualitySummary:
    total: int
    usable: int
    by_flag: dict[str, int]


def summarize(all_flags: Sequence[frozenset[QuoteFlag]]) -> QualitySummary:
    counts: dict[str, int] = {}
    for flags in all_flags:
        for flag in flags:
            counts[flag.value] = counts.get(flag.value, 0) + 1
    return QualitySummary(
        total=len(all_flags),
        usable=sum(1 for flags in all_flags if is_usable(flags)),
        by_flag=dict(sorted(counts.items())),
    )
