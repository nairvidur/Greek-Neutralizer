"""A small, runnable sample. All values are illustrative and are not market data."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

_AS_OF = datetime(2026, 1, 5, 15, 0, tzinfo=UTC)


def _expiry(days: int) -> str:
    return (_AS_OF + timedelta(days=days)).isoformat()


def _option(right: str, strike: float, days: int) -> dict[str, Any]:
    return {
        "type": "option",
        "underlying": "XYZ",
        "right": right,
        "strike": strike,
        "expiry": _expiry(days),
        "style": "european",
        "settlement": "cash",
    }


def example_market() -> dict[str, Any]:
    return {
        "as_of": _AS_OF.isoformat(),
        "underlyings": {
            "XYZ": {"spot": 100.0, "rate": 0.05, "surface": {"type": "flat", "sigma": 0.25}}
        },
    }


def example_book() -> list[dict[str, Any]]:
    return [
        {"instrument": _option("call", 100.0, 30), "quantity": -10},
        {"instrument": _option("put", 95.0, 30), "quantity": -10},
    ]


def _quote(bid: float, ask: float) -> dict[str, Any]:
    return {
        "bid": bid,
        "ask": ask,
        "bid_size": 500,
        "ask_size": 500,
        "timestamp": _AS_OF.isoformat(),
    }


def example_request() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "book": example_book(),
        "market": example_market(),
        "candidates": [
            {
                "instrument": {"type": "linear", "symbol": "XYZ", "underlying": "XYZ"},
                "quote": dict(_quote(99.99, 100.01), bid_size=100000, ask_size=100000),
            },
            {"instrument": _option("put", 90.0, 60), "quote": _quote(1.08, 1.14)},
            {"instrument": _option("put", 95.0, 60), "quote": _quote(2.20, 2.28)},
            {"instrument": _option("call", 105.0, 60), "quote": _quote(2.24, 2.32)},
            {"instrument": _option("call", 110.0, 60), "quote": _quote(1.10, 1.16)},
            {"instrument": _option("call", 100.0, 90), "quote": _quote(5.48, 5.58)},
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
