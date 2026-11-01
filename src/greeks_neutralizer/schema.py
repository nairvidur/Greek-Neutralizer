"""JSON schemas for books, markets, and hedge requests, shared by the CLI and the service.

These are the only way external data enters. Everything is validated by the domain constructors, and
an unknown field is an error: a misspelled option such as `max_legz` must not silently do nothing.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict
from datetime import datetime, timedelta
from typing import Any, TypeVar

from greeks_neutralizer.domain import (
    ExerciseStyle,
    Instrument,
    LinearInstrument,
    LinearKind,
    OptionContract,
    OptionRight,
    Position,
    Quote,
    SettlementKind,
)
from greeks_neutralizer.errors import ValidationError
from greeks_neutralizer.hedging.problem import (
    GreekName,
    GreekTarget,
    HedgeCandidate,
    HedgeConfig,
    HedgeProblem,
    HedgeProposal,
)
from greeks_neutralizer.market import (
    CashDividend,
    DiscountCurve,
    FlatVol,
    MarketSnapshot,
    UnderlyingMarket,
)
from greeks_neutralizer.pricing.book import BookValuation
from greeks_neutralizer.risk.scenarios import HISTORICAL_STRESSES, HYPOTHETICAL_SHOCKS, Scenario
from greeks_neutralizer.surface import SviSlice, SviSurface

T = TypeVar("T")
JsonDict = Mapping[str, Any]

SCHEMA_VERSION = 1


def _require(data: Any, name: str) -> JsonDict:
    if not isinstance(data, Mapping):
        raise ValidationError(f"{name} must be an object, got {type(data).__name__}")
    return data


def _check_fields(
    data: JsonDict, name: str, required: set[str], optional: set[str] | None = None
) -> None:
    optional = optional or set()
    unknown = set(data) - required - optional
    if unknown:
        raise ValidationError(
            f"{name}: unknown field(s) {sorted(unknown)}; valid: {sorted(required | optional)}"
        )
    missing = required - set(data)
    if missing:
        raise ValidationError(f"{name}: missing field(s) {sorted(missing)}")


def _list_of(data: Any, name: str, parse: Callable[[Any], T]) -> tuple[T, ...]:
    if not isinstance(data, Sequence) or isinstance(data, (str, bytes)):
        raise ValidationError(f"{name} must be a list")
    out = []
    for index, item in enumerate(data):
        try:
            out.append(parse(item))
        except (ValidationError, TypeError, ValueError, KeyError) as exc:
            raise ValidationError(f"{name}[{index}]: {exc}") from exc
    return tuple(out)


def _datetime(value: Any, name: str) -> datetime:
    if not isinstance(value, str):
        raise ValidationError(f"{name} must be an ISO 8601 string")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise ValidationError(f"{name}: {exc}") from exc
    if parsed.tzinfo is None:
        raise ValidationError(f"{name} must include a UTC offset, got {value!r}")
    return parsed


def parse_instrument(data: Any) -> Instrument:
    d = _require(data, "instrument")
    kind = d.get("type")
    if kind == "option":
        _check_fields(
            d,
            "option",
            {"type", "underlying", "right", "strike", "expiry"},
            {"style", "multiplier", "settlement", "currency"},
        )
        return OptionContract(
            underlying=d["underlying"],
            right=OptionRight(d["right"]),
            strike=d["strike"],
            expiry=_datetime(d["expiry"], "expiry"),
            style=ExerciseStyle(d.get("style", "american")),
            multiplier=d.get("multiplier", 100.0),
            settlement=SettlementKind(d.get("settlement", "physical")),
            currency=d.get("currency", "USD"),
        )
    if kind == "linear":
        _check_fields(
            d,
            "linear",
            {"type", "symbol", "underlying"},
            {"kind", "multiplier", "delta_ratio", "basis_vol", "expiry", "currency"},
        )
        return LinearInstrument(
            symbol=d["symbol"],
            underlying=d["underlying"],
            kind=LinearKind(d.get("kind", "spot")),
            multiplier=d.get("multiplier", 1.0),
            delta_ratio=d.get("delta_ratio", 1.0),
            basis_vol=d.get("basis_vol", 0.0),
            expiry=_datetime(d["expiry"], "expiry") if "expiry" in d else None,
            currency=d.get("currency", "USD"),
        )
    raise ValidationError(f"instrument type must be 'option' or 'linear', got {kind!r}")


def parse_position(data: Any) -> Position:
    d = _require(data, "position")
    _check_fields(d, "position", {"instrument", "quantity"})
    return Position(parse_instrument(d["instrument"]), d["quantity"])


def parse_book(data: Any) -> tuple[Position, ...]:
    return _list_of(data, "book", parse_position)


def _parse_surface(data: Any) -> FlatVol | SviSurface:
    d = _require(data, "surface")
    kind = d.get("type")
    if kind == "flat":
        _check_fields(d, "flat surface", {"type", "sigma"})
        return FlatVol(d["sigma"])
    if kind == "svi":
        _check_fields(d, "svi surface", {"type", "slices"}, {"quoted_ranges"})

        def parse_slice(item: Any) -> SviSlice:
            s = _require(item, "slice")
            _check_fields(s, "slice", {"tau", "a", "b", "rho", "m", "sigma"})
            return SviSlice(s["tau"], s["a"], s["b"], s["rho"], s["m"], s["sigma"])

        ranges = tuple((float(r[0]), float(r[1])) for r in d.get("quoted_ranges", ()))
        return SviSurface(_list_of(d["slices"], "slices", parse_slice), ranges)
    raise ValidationError(f"surface type must be 'flat' or 'svi', got {kind!r}")


def _parse_underlying(symbol: str, data: Any) -> UnderlyingMarket:
    d = _require(data, f"market {symbol}")
    _check_fields(
        d,
        f"market {symbol}",
        {"spot", "surface"},
        {"rate", "curve", "carry_yield", "dividends"},
    )
    if "curve" in d:
        curve_data = _require(d["curve"], "curve")
        _check_fields(curve_data, "curve", {"times", "zero_rates"})
        curve = DiscountCurve(tuple(curve_data["times"]), tuple(curve_data["zero_rates"]))
    elif "rate" in d:
        curve = DiscountCurve.flat(d["rate"])
    else:
        raise ValidationError(f"market {symbol}: give 'rate' or 'curve'; there is no default rate")

    def parse_dividend(item: Any) -> CashDividend:
        v = _require(item, "dividend")
        _check_fields(v, "dividend", {"ex_date", "amount"})
        return CashDividend(_datetime(v["ex_date"], "ex_date"), v["amount"])

    return UnderlyingMarket(
        symbol=symbol,
        spot=d["spot"],
        curve=curve,
        surface=_parse_surface(d["surface"]),
        carry_yield=d.get("carry_yield", 0.0),
        dividends=_list_of(d.get("dividends", ()), "dividends", parse_dividend),
    )


def parse_market(data: Any) -> MarketSnapshot:
    d = _require(data, "market")
    _check_fields(d, "market", {"as_of", "underlyings"})
    underlyings = _require(d["underlyings"], "underlyings")
    return MarketSnapshot(
        _datetime(d["as_of"], "as_of"),
        {sym: _parse_underlying(sym, value) for sym, value in underlyings.items()},
    )


def _parse_quote(data: Any) -> Quote:
    d = _require(data, "quote")
    _check_fields(d, "quote", {"bid", "ask", "bid_size", "ask_size", "timestamp"})
    return Quote(
        d["bid"], d["ask"], d["bid_size"], d["ask_size"], _datetime(d["timestamp"], "timestamp")
    )


def parse_candidate(data: Any) -> HedgeCandidate:
    d = _require(data, "candidate")
    _check_fields(
        d,
        "candidate",
        {"instrument", "quote"},
        {"fee_per_unit", "max_buy", "max_sell", "impact_per_unit", "lot_size", "displayed_only"},
    )
    return HedgeCandidate(
        instrument=parse_instrument(d["instrument"]),
        quote=_parse_quote(d["quote"]),
        fee_per_unit=d.get("fee_per_unit", 0.0),
        max_buy=d.get("max_buy"),
        max_sell=d.get("max_sell"),
        impact_per_unit=d.get("impact_per_unit", 0.0),
        lot_size=d.get("lot_size", 1),
        displayed_only=d.get("displayed_only", False),
    )


def parse_target(data: Any) -> GreekTarget:
    d = _require(data, "target")
    _check_fields(d, "target", {"underlying", "greek"}, {"lower", "upper", "weight"})
    try:
        greek = GreekName(d["greek"])
    except ValueError:
        valid = sorted(g.value for g in GreekName)
        raise ValidationError(f"unknown greek {d['greek']!r}; valid: {valid}") from None
    return GreekTarget(d["underlying"], greek, d.get("lower"), d.get("upper"), d.get("weight"))


def parse_scenario(data: Any) -> Scenario:
    d = _require(data, "scenario")
    _check_fields(
        d,
        "scenario",
        {"name"},
        {
            "spot_shock",
            "vol_shock",
            "skew_shock",
            "term_power",
            "days_forward",
            "sticky_strike",
            "note",
        },
    )
    return Scenario(**d)


SCENARIO_SETS: dict[str, tuple[Scenario, ...]] = {
    "historical": HISTORICAL_STRESSES,
    "hypothetical": HYPOTHETICAL_SHOCKS,
    "all": (*HISTORICAL_STRESSES, *HYPOTHETICAL_SHOCKS),
}


def _parse_scenarios(data: Any) -> tuple[Scenario, ...]:
    if isinstance(data, str):
        if data not in SCENARIO_SETS:
            raise ValidationError(f"unknown scenario set {data!r}; valid: {sorted(SCENARIO_SETS)}")
        return SCENARIO_SETS[data]
    return _list_of(data, "scenarios", parse_scenario)


def _parse_config(data: Any) -> HedgeConfig:
    d = dict(_require(data, "config"))
    allowed = {
        "risk_aversion",
        "cvar_weight",
        "cvar_alpha",
        "max_legs",
        "ticket_cost",
        "premium_budget",
        "mip_gap",
        "time_limit_seconds",
        "max_quote_age_seconds",
        "require_quote_size",
        "allow_extrapolated_vol",
        "instability_threshold",
        "regret_threshold",
    }
    _check_fields(d, "config", set(), allowed)
    if "max_quote_age_seconds" in d:
        d["max_quote_age"] = timedelta(seconds=d.pop("max_quote_age_seconds"))
    return HedgeConfig(**d)


def parse_hedge_request(data: Any) -> tuple[HedgeProblem, MarketSnapshot, bool]:
    d = _require(data, "request")
    _check_fields(
        d,
        "request",
        {"book", "candidates", "targets", "market"},
        {"scenarios", "config", "run_stability", "schema_version"},
    )
    version = d.get("schema_version", SCHEMA_VERSION)
    if version != SCHEMA_VERSION:
        raise ValidationError(f"schema_version {version} is not supported; use {SCHEMA_VERSION}")
    problem = HedgeProblem(
        book=parse_book(d["book"]),
        candidates=_list_of(d["candidates"], "candidates", parse_candidate),
        targets=_list_of(d["targets"], "targets", parse_target),
        scenarios=_parse_scenarios(d.get("scenarios", ())),
        config=_parse_config(d.get("config", {})),
    )
    return problem, parse_market(d["market"]), bool(d.get("run_stability", False))


def instrument_to_json(instrument: Instrument) -> dict[str, Any]:
    if isinstance(instrument, OptionContract):
        return {
            "type": "option",
            "underlying": instrument.underlying,
            "right": instrument.right.value,
            "strike": instrument.strike,
            "expiry": instrument.expiry.isoformat(),
            "style": instrument.style.value,
            "multiplier": instrument.multiplier,
            "settlement": instrument.settlement.value,
            "currency": instrument.currency,
        }
    return {
        "type": "linear",
        "symbol": instrument.symbol,
        "underlying": instrument.underlying,
        "kind": instrument.kind.value,
        "multiplier": instrument.multiplier,
        "delta_ratio": instrument.delta_ratio,
        "basis_vol": instrument.basis_vol,
        "expiry": instrument.expiry.isoformat() if instrument.expiry else None,
        "currency": instrument.currency,
    }


def book_to_json(book: BookValuation) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "as_of": book.as_of.isoformat(),
        "market_fingerprint": book.market_fingerprint,
        "totals": {
            "value": book.total_value,
            "delta_dollars": book.total_delta_dollars,
            "gamma_1pct": book.total_gamma_1pct,
            "vega_1pt": book.total_vega_1pt,
            "theta_1d": book.total_theta_1d,
        },
        "by_underlying": {sym: e.as_dict() for sym, e in book.by_underlying.items()},
        "lines": [
            {
                "instrument": line.position.instrument.describe(),
                "quantity": line.position.quantity,
                "unit_price": line.unit_price,
                "vol": line.vol,
                "exposure": line.exposure.as_dict(),
            }
            for line in book.lines
        ],
        "issues": list(book.issues),
    }


def proposal_to_json(proposal: HedgeProposal) -> dict[str, Any]:
    stability = None
    if proposal.stability is not None:
        stability = {
            "condition_number": proposal.stability.condition_number,
            "lot_change_ratio": proposal.stability.lot_change_ratio,
            "regret_ratio": proposal.stability.regret_ratio,
            "is_stable": proposal.stability.is_stable,
            "drift": dict(proposal.stability.drift),
            "notes": list(proposal.stability.notes),
        }
    return {
        "schema_version": SCHEMA_VERSION,
        "status": proposal.status.value,
        "legs": [
            {
                "instrument": instrument_to_json(leg.instrument),
                "description": leg.instrument.describe(),
                "side": leg.side,
                "quantity": leg.quantity,
                "expected_price": leg.expected_price,
                "premium": leg.premium,
                "transaction_cost": leg.transaction_cost,
                "rationale": leg.rationale,
            }
            for leg in proposal.legs
        ],
        "objective": proposal.objective,
        "objective_lower_bound": proposal.objective_lower_bound,
        "transaction_cost": proposal.transaction_cost,
        "net_premium": proposal.net_premium,
        "gross_lots": proposal.gross_lots,
        "residuals": [asdict(line) for line in proposal.residuals],
        "before": {k: e.as_dict() for k, e in proposal.before.items()},
        "after": {k: e.as_dict() for k, e in proposal.after.items()},
        "scenario_pnl_before": dict(proposal.scenario_pnl_before),
        "scenario_pnl_after": dict(proposal.scenario_pnl_after),
        "excluded": [asdict(item) for item in proposal.excluded],
        "verification": asdict(proposal.verification),
        "stability": stability,
        "basis_risk_1d": proposal.basis_risk_1d,
        "solver_gap": proposal.solver_gap,
        "issues": list(proposal.issues),
        "manifest": {
            "code_version": proposal.manifest.code_version,
            "market_fingerprint": proposal.manifest.market_fingerprint,
            "problem_fingerprint": proposal.manifest.problem_fingerprint,
            "as_of": proposal.manifest.as_of.isoformat(),
            "solver": proposal.manifest.solver,
        },
    }
