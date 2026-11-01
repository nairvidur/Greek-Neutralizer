"""JSON schemas for books, markets, and hedge requests, shared by the CLI and the service.

These are the only way external data enters. Everything is validated by the domain constructors, and
an unknown field is an error: a misspelled option such as `max_legz` must not silently do nothing.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict
from datetime import UTC, datetime, timedelta
from functools import wraps
from typing import Any, ParamSpec, TypeVar

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
    require_finite,
)
from greeks_neutralizer.errors import GNError, ValidationError
from greeks_neutralizer.hedging.diagnostics import DecayForecast
from greeks_neutralizer.hedging.problem import (
    GreekName,
    GreekTarget,
    HedgeCandidate,
    HedgeConfig,
    HedgeProblem,
    HedgeProposal,
    gate_fields,
)
from greeks_neutralizer.market import (
    CashDividend,
    DiscountCurve,
    FlatVol,
    MarketSnapshot,
    UnderlyingMarket,
)
from greeks_neutralizer.pricing.book import BookValuation
from greeks_neutralizer.risk.measures import (
    ExpiryLadder,
    LimitCheck,
    LiquidityEstimate,
    RiskLimit,
    liquidity_adjusted_cost,
)
from greeks_neutralizer.risk.scenarios import HISTORICAL_STRESSES, HYPOTHETICAL_SHOCKS, Scenario
from greeks_neutralizer.surface import SviSlice, SviSurface

T = TypeVar("T")
P = ParamSpec("P")
JsonDict = Mapping[str, Any]

SCHEMA_VERSION = 1

# Wide of any listed product, and far enough inside what a datetime can hold that converting to New
# York and adding the longest scenario horizon or assignment window cannot overflow.
_EARLIEST_DATE = datetime(1900, 1, 1, tzinfo=UTC)
_LATEST_DATE = datetime(2300, 1, 1, tzinfo=UTC)


def _typed_errors(parse: Callable[P, T]) -> Callable[P, T]:
    """An entry point's input errors are ValidationError, whatever a builtin raised on bad types."""

    @wraps(parse)
    def checked(*args: P.args, **kwargs: P.kwargs) -> T:
        try:
            return parse(*args, **kwargs)
        except GNError:
            raise
        except (TypeError, ValueError, KeyError, AttributeError, OverflowError) as exc:
            raise ValidationError(f"{parse.__name__}: {exc}") from exc

    return checked


def _require(data: Any, name: str) -> JsonDict:
    if not isinstance(data, Mapping):
        raise ValidationError(f"{name} must be an object, got {type(data).__name__}")
    return data


def check_fields(
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
    if not _EARLIEST_DATE <= parsed <= _LATEST_DATE:
        raise ValidationError(f"{name} must be from 1900 to 2300, got {value!r}")
    return parsed


def parse_instrument(data: Any) -> Instrument:
    d = _require(data, "instrument")
    kind = d.get("type")
    if kind == "option":
        check_fields(
            d,
            "option",
            {"type", "underlying", "right", "strike", "expiry"},
            {"style", "multiplier", "settlement", "currency", "exchange_symbol"},
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
            exchange_symbol=d.get("exchange_symbol", ""),
        )
    if kind == "linear":
        check_fields(
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
    check_fields(d, "position", {"instrument", "quantity"})
    return Position(parse_instrument(d["instrument"]), d["quantity"])


@_typed_errors
def parse_book(data: Any) -> tuple[Position, ...]:
    return _list_of(data, "book", parse_position)


def _parse_surface(data: Any) -> FlatVol | SviSurface:
    d = _require(data, "surface")
    kind = d.get("type")
    if kind == "flat":
        check_fields(d, "flat surface", {"type", "sigma"})
        return FlatVol(d["sigma"])
    if kind == "svi":
        check_fields(d, "svi surface", {"type", "slices"}, {"quoted_ranges"})

        def parse_slice(item: Any) -> SviSlice:
            s = _require(item, "slice")
            check_fields(s, "slice", {"tau", "a", "b", "rho", "m", "sigma"})
            return SviSlice(s["tau"], s["a"], s["b"], s["rho"], s["m"], s["sigma"])

        def parse_range(item: Any) -> tuple[float, float]:
            if not isinstance(item, Sequence) or isinstance(item, str) or len(item) != 2:
                raise ValidationError(f"a quoted range is [low, high], got {item!r}")
            require_finite("quoted range", item[0])
            require_finite("quoted range", item[1])
            return float(item[0]), float(item[1])

        ranges = _list_of(d.get("quoted_ranges", ()), "quoted_ranges", parse_range)
        return SviSurface(_list_of(d["slices"], "slices", parse_slice), ranges)
    raise ValidationError(f"surface type must be 'flat' or 'svi', got {kind!r}")


def _parse_underlying(symbol: str, data: Any) -> UnderlyingMarket:
    d = _require(data, f"market {symbol}")
    check_fields(
        d,
        f"market {symbol}",
        {"spot", "surface"},
        {"rate", "curve", "carry_yield", "dividends"},
    )
    if ("curve" in d) == ("rate" in d):
        raise ValidationError(
            f"market {symbol}: give exactly one of 'rate' or 'curve'; there is no default rate"
        )
    if "curve" in d:
        curve_data = _require(d["curve"], "curve")
        check_fields(curve_data, "curve", {"times", "zero_rates"})
        curve = DiscountCurve(
            _list_of(curve_data["times"], "times", float),
            _list_of(curve_data["zero_rates"], "zero_rates", float),
        )
    else:
        curve = DiscountCurve.flat(d["rate"])

    def parse_dividend(item: Any) -> CashDividend:
        v = _require(item, "dividend")
        check_fields(v, "dividend", {"ex_date", "amount"})
        return CashDividend(_datetime(v["ex_date"], "ex_date"), v["amount"])

    return UnderlyingMarket(
        symbol=symbol,
        spot=d["spot"],
        curve=curve,
        surface=_parse_surface(d["surface"]),
        carry_yield=d.get("carry_yield", 0.0),
        dividends=_list_of(d.get("dividends", ()), "dividends", parse_dividend),
    )


@_typed_errors
def parse_market(data: Any) -> MarketSnapshot:
    d = _require(data, "market")
    check_fields(d, "market", {"as_of", "underlyings"})
    underlyings = _require(d["underlyings"], "underlyings")
    return MarketSnapshot(
        _datetime(d["as_of"], "as_of"),
        {sym: _parse_underlying(sym, value) for sym, value in underlyings.items()},
    )


def _parse_quote(data: Any) -> Quote:
    d = _require(data, "quote")
    check_fields(d, "quote", {"bid", "ask", "bid_size", "ask_size", "timestamp"})
    return Quote(
        d["bid"], d["ask"], d["bid_size"], d["ask_size"], _datetime(d["timestamp"], "timestamp")
    )


def parse_candidate(data: Any) -> HedgeCandidate:
    d = _require(data, "candidate")
    check_fields(
        d,
        "candidate",
        {"instrument", "quote"},
        {"fee_per_unit", "max_buy", "max_sell", "impact_per_unit", "lot_size"},
    )
    return HedgeCandidate(
        instrument=parse_instrument(d["instrument"]),
        quote=_parse_quote(d["quote"]),
        fee_per_unit=d.get("fee_per_unit", 0.0),
        max_buy=d.get("max_buy"),
        max_sell=d.get("max_sell"),
        impact_per_unit=d.get("impact_per_unit", 0.0),
        lot_size=d.get("lot_size", 1),
    )


def parse_target(data: Any) -> GreekTarget:
    d = _require(data, "target")
    check_fields(d, "target", {"underlying", "greek"}, {"lower", "upper", "weight"})
    try:
        greek = GreekName(d["greek"])
    except ValueError:
        valid = sorted(g.value for g in GreekName)
        raise ValidationError(f"unknown greek {d['greek']!r}; valid: {valid}") from None
    return GreekTarget(d["underlying"], greek, d.get("lower"), d.get("upper"), d.get("weight"))


def parse_scenario(data: Any) -> Scenario:
    d = _require(data, "scenario")
    check_fields(
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


def _parse_config(data: Any, gate_defaults: HedgeConfig) -> HedgeConfig:
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
        "mip_node_limit",
        "max_quote_age_seconds",
        "require_quote_size",
        "allow_extrapolated_vol",
        "instability_threshold",
        "regret_threshold",
        "max_lots_per_leg",
        "max_model_quote_vol_gap",
        "max_linear_quote_gap",
    }
    check_fields(d, "config", set(), allowed)
    if "max_quote_age_seconds" in d:
        seconds = d.pop("max_quote_age_seconds")
        require_finite("max_quote_age_seconds", seconds)
        d["max_quote_age"] = timedelta(seconds=seconds)
    return HedgeConfig(**{**gate_fields(gate_defaults), **d})


@_typed_errors
def parse_hedge_request(
    data: Any, gate_defaults: HedgeConfig | None = None
) -> tuple[HedgeProblem, MarketSnapshot, bool]:
    """`gate_defaults` is what a gate left out of the request takes: a service's limits, or the
    library defaults if omitted."""
    d = _require(data, "request")
    check_fields(
        d,
        "request",
        {"book", "candidates", "targets", "market"},
        {"scenarios", "config", "run_stability", "schema_version"},
    )
    version = d.get("schema_version", SCHEMA_VERSION)
    if version != SCHEMA_VERSION:
        raise ValidationError(f"schema_version {version} is not supported; use {SCHEMA_VERSION}")
    run_stability = d.get("run_stability", False)
    if not isinstance(run_stability, bool):
        raise ValidationError(f"run_stability must be true or false, got {run_stability!r}")
    problem = HedgeProblem(
        book=parse_book(d["book"]),
        candidates=_list_of(d["candidates"], "candidates", parse_candidate),
        targets=_list_of(d["targets"], "targets", parse_target),
        scenarios=_parse_scenarios(d.get("scenarios", ())),
        config=_parse_config(d.get("config", {}), gate_defaults or HedgeConfig()),
    )
    return problem, parse_market(d["market"]), run_stability


def _parse_limit(data: Any) -> RiskLimit:
    d = _require(data, "limit")
    check_fields(d, "limit", {"metric"}, {"lower", "upper", "underlying"})
    return RiskLimit(d["metric"], d.get("lower"), d.get("upper"), d.get("underlying"))


@_typed_errors
def parse_limits(data: Any) -> tuple[RiskLimit, ...]:
    return _list_of(data, "limits", _parse_limit)


def _liquidity_line(data: Any) -> LiquidityEstimate:
    d = _require(data, "line")
    check_fields(
        d, "line", {"instrument", "quantity", "half_spread", "adv_contracts"}, {"participation"}
    )
    position = Position(parse_instrument(d["instrument"]), d["quantity"])
    return liquidity_adjusted_cost(
        position, d["half_spread"], d["adv_contracts"], participation=d.get("participation", 0.1)
    )


@_typed_errors
def liquidity_estimates(data: Any) -> tuple[LiquidityEstimate, ...]:
    return _list_of(data, "lines", _liquidity_line)


def _parse_day(data: Any) -> float:
    require_finite("days", data)
    return float(data)


@_typed_errors
def parse_days(data: Any) -> tuple[float, ...]:
    return _list_of(data, "days", _parse_day)


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
            "exchange_symbol": instrument.exchange_symbol,
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


def ladder_to_json(ladder: ExpiryLadder) -> dict[str, Any]:
    return {
        "count": len(ladder.buckets),
        "buckets": [
            {
                "expiry": bucket.expiry.isoformat(),
                "n_lines": bucket.n_lines,
                "exposure": bucket.exposure.as_dict(),
            }
            for bucket in ladder.buckets
        ],
        "issues": list(ladder.issues),
    }


def limit_check_to_json(check: LimitCheck) -> dict[str, Any]:
    return {
        "checked": check.checked,
        "breached": len(check.breaches),
        "breaches": [
            {
                "metric": breach.limit.metric,
                "underlying": breach.limit.underlying,
                "lower": breach.limit.lower,
                "upper": breach.limit.upper,
                "value": breach.value if math.isfinite(breach.value) else None,
                # Infinity is not valid JSON, and null alone would hide that it was unbounded.
                "utilization": breach.utilization if math.isfinite(breach.utilization) else None,
                "utilization_unbounded": math.isinf(breach.utilization),
            }
            for breach in check.breaches
        ],
        "issues": list(check.issues),
    }


def liquidity_to_json(estimates: Sequence[LiquidityEstimate]) -> dict[str, Any]:
    return {
        "count": len(estimates),
        "total_half_spread_cost": sum(e.half_spread_cost for e in estimates),
        "longest_days_to_unwind": max((e.days_to_unwind for e in estimates), default=0.0),
        "lines": [asdict(estimate) for estimate in estimates],
    }


def decay_to_json(forecast: DecayForecast, underlying: str) -> dict[str, Any]:
    return {
        "underlying": underlying,
        "count": len(forecast.steps),
        "steps": [
            {"days_forward": step.days_forward, "exposure": step.exposure.as_dict()}
            for step in forecast.steps
        ],
        "issues": list(forecast.issues),
    }


def proposal_to_json(proposal: HedgeProposal) -> dict[str, Any]:
    stability = None
    if proposal.stability is not None:
        regret = proposal.stability.regret_ratio
        condition = proposal.stability.condition_number
        stability = {
            # Infinity is not valid JSON, and null alone would hide that it was unbounded.
            "condition_number": condition if math.isfinite(condition) else None,
            "lot_change_ratio": proposal.stability.lot_change_ratio,
            "regret_ratio": regret if math.isfinite(regret) else None,
            "regret_unbounded": not math.isfinite(regret),
            "is_stable": proposal.stability.is_stable,
            "drift": {
                k: v if math.isfinite(v) else None for k, v in proposal.stability.drift.items()
            },
            "notes": list(proposal.stability.notes),
            "out_of_range": list(proposal.stability.out_of_range),
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
            "scipy_version": proposal.manifest.scipy_version,
            "numpy_version": proposal.manifest.numpy_version,
            "run_stability": proposal.manifest.run_stability,
            "gate_overrides": dict(proposal.manifest.gate_overrides),
        },
    }
