"""Adapter for the JSON layout of Cboe's public delayed-quotes endpoint.

Delayed data, for development and reconciliation only: this is not a licensed real-time feed, and
terms of use apply. Fetching is left to the caller so core code does no I/O; this only parses.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, cast

from greeks_neutralizer.data.chain import ChainQuote
from greeks_neutralizer.data.symbology import is_am_settled, is_cash_settled_european, parse_osi
from greeks_neutralizer.domain import ExerciseStyle, OptionContract, Quote, SettlementKind
from greeks_neutralizer.errors import MarketDataError
from greeks_neutralizer.time_model import NEW_YORK, us_expiry

STANDARD_ROOTS: dict[str, frozenset[str]] = {
    "SPX": frozenset({"SPX", "SPXW"}),
    "NDX": frozenset({"NDX", "NDXP"}),
    "RUT": frozenset({"RUT", "RUTW"}),
}


@dataclass(frozen=True, slots=True)
class ParsedChain:
    symbol: str
    spot: float
    as_of: datetime
    quotes: tuple[ChainQuote, ...]
    n_rejected: int
    n_non_standard: int
    sizes_available: bool


def _number(row: Mapping[str, Any], key: str) -> float:
    value = row.get(key)
    if value is None:
        return math.nan
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def _vendor_greek(row: Mapping[str, Any], key: str) -> float | None:
    value = _number(row, key)
    return value if math.isfinite(value) else None


def _quote_time(data: Mapping[str, Any], fallback: datetime) -> datetime:
    """When these quotes were live: the latest trade in the payload (ET), not the fetch time.

    The payload `timestamp` is when it was generated. Outside market hours that can be days after
    the quotes, which on a short expiry would badly misstate time to expiry and implied vol.
    """
    stamps = [data.get("last_trade_time")]
    stamps += [row.get("last_trade_time") for row in data.get("options", [])]
    parsed = []
    for stamp in stamps:
        if isinstance(stamp, str):
            try:
                parsed.append(datetime.fromisoformat(stamp).replace(tzinfo=NEW_YORK))
            except ValueError:
                continue
    latest = max(parsed, default=fallback)
    return min(latest, fallback)


def parse_cboe_delayed(payload: Mapping[str, Any]) -> ParsedChain:
    """Parse one underlying's payload. Malformed and non-standard rows are counted, not hidden.

    Roots outside the underlying's standard family (an adjusted contract such as B1SP inside SPY)
    have a non-standard deliverable, and so cannot be valued as 100 shares of the underlying.
    """
    data = payload.get("data")
    if not isinstance(data, Mapping):
        raise MarketDataError("Cboe payload has no 'data' object")
    data = cast(Mapping[str, Any], data)
    timestamp = payload.get("timestamp")
    if not isinstance(timestamp, str):
        raise MarketDataError("Cboe payload has no 'timestamp'")
    # Payload timestamps are UTC, without a zone designator.
    generated = datetime.fromisoformat(timestamp).replace(tzinfo=UTC)
    as_of = _quote_time(data, generated)
    symbol = str(data.get("symbol", "")).lstrip("^_")
    standard_roots = STANDARD_ROOTS.get(symbol, frozenset({symbol}))

    spot = _number(data, "current_price")
    if not math.isfinite(spot) or spot <= 0.0:
        bid, ask = _number(data, "bid"), _number(data, "ask")
        if not (bid > 0.0 and ask > 0.0):
            raise MarketDataError("Cboe payload has no usable underlying price")
        spot = 0.5 * (bid + ask)

    quotes: list[ChainQuote] = []
    rejected = 0
    non_standard = 0
    for row in data.get("options", []):
        try:
            osi = parse_osi(str(row["option"]))
            if osi.root not in standard_roots:
                non_standard += 1
                continue
            european = is_cash_settled_european(osi.root)
            contract = OptionContract(
                underlying=symbol,
                right=osi.right,
                strike=osi.strike,
                expiry=us_expiry(osi.expiry, am_settled=is_am_settled(osi.root)),
                style=ExerciseStyle.EUROPEAN if european else ExerciseStyle.AMERICAN,
                settlement=SettlementKind.CASH if european else SettlementKind.PHYSICAL,
            )
            quote = Quote(
                bid=float(row["bid"]),
                ask=float(row["ask"]),
                bid_size=int(row.get("bid_size") or 0),
                ask_size=int(row.get("ask_size") or 0),
                timestamp=as_of,
            )
        except Exception:
            rejected += 1
            continue
        quotes.append(
            ChainQuote(
                contract=contract,
                quote=quote,
                open_interest=int(row.get("open_interest") or 0),
                volume=int(row.get("volume") or 0),
                vendor_iv=_vendor_greek(row, "iv"),
                vendor_delta=_vendor_greek(row, "delta"),
                vendor_gamma=_vendor_greek(row, "gamma"),
            )
        )
    if not quotes:
        raise MarketDataError(f"Cboe payload has no parseable options ({rejected} rejected)")
    sizes_available = any(q.quote.bid_size > 0 or q.quote.ask_size > 0 for q in quotes)
    return ParsedChain(symbol, spot, as_of, tuple(quotes), rejected, non_standard, sizes_available)
