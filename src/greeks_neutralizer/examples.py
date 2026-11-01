"""A small, runnable sample. All values are illustrative and are not market data.

Quotes are built from the sample market's own model, so they pass the model-against-quote gate.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

from greeks_neutralizer.pricing.black import black_greeks

DEFAULT_AS_OF = datetime(2026, 1, 5, 15, 0, tzinfo=UTC)
_SPOT, _RATE, _VOL = 100.0, 0.05, 0.25


def _option(right: str, strike: float, days: int, as_of: datetime) -> dict[str, Any]:
    return {
        "type": "option",
        "underlying": "XYZ",
        "right": right,
        "strike": strike,
        "expiry": (as_of + timedelta(days=days)).isoformat(),
        "style": "european",
        "settlement": "cash",
    }


def example_market(as_of: datetime = DEFAULT_AS_OF) -> dict[str, Any]:
    return {
        "as_of": as_of.isoformat(),
        "underlyings": {
            "XYZ": {"spot": _SPOT, "rate": _RATE, "surface": {"type": "flat", "sigma": _VOL}}
        },
    }


def example_book(as_of: datetime = DEFAULT_AS_OF) -> list[dict[str, Any]]:
    return [
        {"instrument": _option("call", 100.0, 30, as_of), "quantity": -10},
        {"instrument": _option("put", 95.0, 30, as_of), "quantity": -10},
    ]


def example_limits() -> list[dict[str, Any]]:
    return [
        {"metric": "delta_dollars", "lower": -5000.0, "upper": 5000.0, "underlying": "XYZ"},
        {"metric": "vega_1pt", "lower": -1000.0, "upper": 1000.0},
    ]


def example_liquidity(as_of: datetime = DEFAULT_AS_OF) -> list[dict[str, Any]]:
    return [
        {**position, "half_spread": 0.05, "adv_contracts": 2000.0}
        for position in example_book(as_of)
    ]


def _candidate(right: str, strike: float, days: int, as_of: datetime) -> dict[str, Any]:
    omega = 1.0 if right == "call" else -1.0
    price = float(black_greeks(_SPOT, strike, days / 365.0, _RATE, 0.0, _VOL, omega).price[0])
    half = max(0.01 * price, 0.02)
    return {
        "instrument": _option(right, strike, days, as_of),
        "quote": {
            "bid": round(max(price - half, 0.01), 2),
            "ask": round(price + half, 2),
            "bid_size": 500,
            "ask_size": 500,
            "timestamp": as_of.isoformat(),
        },
    }


def example_request(as_of: datetime = DEFAULT_AS_OF) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "book": example_book(as_of),
        "market": example_market(as_of),
        "candidates": [
            {
                "instrument": {"type": "linear", "symbol": "XYZ", "underlying": "XYZ"},
                "quote": {
                    "bid": 99.99,
                    "ask": 100.01,
                    "bid_size": 100000,
                    "ask_size": 100000,
                    "timestamp": as_of.isoformat(),
                },
            },
            _candidate("put", 90.0, 60, as_of),
            _candidate("put", 95.0, 60, as_of),
            _candidate("call", 105.0, 60, as_of),
            _candidate("call", 110.0, 60, as_of),
            _candidate("call", 100.0, 90, as_of),
        ],
        "targets": [
            {"underlying": "XYZ", "greek": "delta_dollars"},
            {"underlying": "XYZ", "greek": "gamma_1pct"},
            {"underlying": "XYZ", "greek": "vega_1pt"},
        ],
        "scenarios": "all",
        "config": {"risk_aversion": 1.0, "max_legs": 5},
        "run_stability": True,
    }
