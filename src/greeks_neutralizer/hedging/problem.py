"""The hedge problem and proposal types. Everything is in dollars, so trade-offs are comparable.

Objective of the solver, in dollars:

    transaction cost
    + risk_aversion * sum over targets of weight * |residual exposure|
    + cvar_weight * CVaR of scenario loss
    + risk_aversion * basis risk of any proxy hedges

Premium paid at mid is not a cost: it exchanges cash for an asset worth that much. The cost of a
hedge is crossing the spread, fees, and impact, which is what the book actually loses.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import StrEnum
from typing import Any

from greeks_neutralizer.data.quality import QualityPolicy
from greeks_neutralizer.domain import (
    Instrument,
    Position,
    Quote,
    require_finite,
    require_int,
    require_symbol,
)
from greeks_neutralizer.errors import ValidationError
from greeks_neutralizer.pricing.book import EXPOSURE_NAMES, Exposure
from greeks_neutralizer.risk.scenarios import Scenario


@dataclass(frozen=True, slots=True)
class HedgeCandidate:
    """A tradable hedge with a live market. Marks and costs come from `quote`, never from a model.

    By default a hedge is limited to the displayed size, because size past the quote is not free.
    Setting `max_buy` or `max_sell` above it allows depth: past the displayed size, each unit pays
    `impact_per_unit` more in dollars, which keeps the problem linear and makes the solver work
    through liquidity in cost order. A proposal that uses depth carries an OVER_DISPLAYED issue.
    """

    instrument: Instrument
    quote: Quote
    fee_per_unit: float = 0.0
    max_buy: int | None = None
    max_sell: int | None = None
    impact_per_unit: float = 0.0
    lot_size: int = 1

    def __post_init__(self) -> None:
        require_finite("fee_per_unit", self.fee_per_unit)
        require_finite("impact_per_unit", self.impact_per_unit)
        if self.fee_per_unit < 0.0 or self.impact_per_unit < 0.0:
            raise ValidationError("fees and impact must be non-negative")
        require_int("lot_size", self.lot_size)
        if self.lot_size < 1:
            raise ValidationError("lot_size must be at least 1")
        for name, value in (("max_buy", self.max_buy), ("max_sell", self.max_sell)):
            if value is None:
                continue
            require_int(name, value)
            if value < 0:
                raise ValidationError(f"{name} must be non-negative")


class GreekName(StrEnum):
    DELTA = "delta_dollars"
    GAMMA = "gamma_1pct"
    VEGA = "vega_1pt"
    THETA = "theta_1d"
    RHO = "rho_1pct"
    VANNA = "vanna_1pt"
    VOLGA = "volga_1pt"
    CHARM = "charm_1d"


# The dollar P&L of one unit of residual under a typical shock, so one residual dollar in any
# Greek means about the same thing to the solver.
DEFAULT_WEIGHTS: Mapping[GreekName, float] = {
    GreekName.DELTA: 0.01,  # 1% spot move
    # P&L on a 3% move is 0.5 * gamma_shares * (0.03 S)^2, which is 0.045 per unit of gamma_1pct.
    GreekName.GAMMA: 0.5 * 0.03**2 / 0.01,
    GreekName.VEGA: 2.0,  # 2 vol points
    GreekName.THETA: 1.0,
    GreekName.RHO: 0.25,
    GreekName.VANNA: 0.0,
    GreekName.VOLGA: 0.0,
    GreekName.CHARM: 0.0,
}


@dataclass(frozen=True, slots=True)
class GreekTarget:
    """How one Greek on one underlying should end up.

    `lower` and `upper` are hard bounds on the post-hedge exposure, and `None` leaves a side open.
    The residual's distance from zero is penalized at `weight` dollars per unit in any case.
    """

    underlying: str
    greek: GreekName
    lower: float | None = None
    upper: float | None = None
    weight: float | None = None

    def __post_init__(self) -> None:
        require_symbol("underlying", self.underlying)
        if not isinstance(self.greek, GreekName):
            raise ValidationError(f"greek must be a GreekName, got {self.greek!r}")
        for name, value in (("lower", self.lower), ("upper", self.upper), ("weight", self.weight)):
            if value is not None:
                require_finite(name, value)
        if self.lower is not None and self.upper is not None and self.lower > self.upper:
            raise ValidationError(f"{self.greek.value}: lower exceeds upper")
        if self.weight is not None and self.weight < 0.0:
            raise ValidationError("weight must be non-negative")

    @property
    def effective_weight(self) -> float:
        return DEFAULT_WEIGHTS[self.greek] if self.weight is None else self.weight


@dataclass(frozen=True, slots=True)
class HedgeConfig:
    risk_aversion: float = 1.0
    cvar_weight: float = 0.0
    cvar_alpha: float = 0.95
    max_legs: int | None = None
    ticket_cost: float = 0.0
    premium_budget: float | None = None
    mip_gap: float = 5e-3
    # The search stops at a node count, which does not depend on machine load. Two byte-identical
    # runs on one machine, one under load, are all that was measured: same library versions only,
    # and other architectures are unverified. The wall-clock limit is only a backstop, and a run
    # that reaches it says so.
    mip_node_limit: int = 300
    time_limit_seconds: float = 30.0
    max_quote_age: timedelta = timedelta(seconds=30)
    require_quote_size: bool = True
    allow_extrapolated_vol: bool = False
    instability_threshold: float = 0.5
    regret_threshold: float = 0.1
    max_lots_per_leg: int = 5000
    max_model_quote_vol_gap: float = 0.05
    max_linear_quote_gap: float = 0.25
    # Position terms, both off by default. Without them the objective prices only today's trade,
    # so a daily re-hedge adds legs and never unwinds offsetting ones: on historical chains that
    # stacked 1,300 hedge contracts against a 200-contract book. `price_exit` also charges every
    # candidate's half-spread and fee on the position held after the trade, the cost of closing it
    # later. `max_gross_option_contracts` caps the absolute option contracts of the whole hedged
    # book, the book's own options included.
    price_exit: bool = False
    max_gross_option_contracts: int | None = None

    def __post_init__(self) -> None:
        for name in (
            "risk_aversion",
            "cvar_weight",
            "cvar_alpha",
            "ticket_cost",
            "mip_gap",
            "time_limit_seconds",
            "instability_threshold",
            "regret_threshold",
            "max_model_quote_vol_gap",
            "max_linear_quote_gap",
        ):
            require_finite(name, getattr(self, name))
        if self.premium_budget is not None:
            require_finite("premium_budget", self.premium_budget)
        if not isinstance(self.max_quote_age, timedelta):
            raise ValidationError(f"max_quote_age must be a timedelta, got {self.max_quote_age!r}")
        if self.max_gross_option_contracts is not None:
            require_int("max_gross_option_contracts", self.max_gross_option_contracts)
            if self.max_gross_option_contracts < 0:
                raise ValidationError("max_gross_option_contracts must be non-negative")
        for name in ("require_quote_size", "allow_extrapolated_vol", "price_exit"):
            if not isinstance(getattr(self, name), bool):
                raise ValidationError(f"{name} must be true or false")
        if isinstance(self.mip_node_limit, bool) or not isinstance(self.mip_node_limit, int):
            raise ValidationError("mip_node_limit must be an integer")
        if self.mip_node_limit < 1:
            raise ValidationError("mip_node_limit must be at least 1")
        if self.time_limit_seconds <= 0.0:
            raise ValidationError("time_limit_seconds must be positive")
        if self.risk_aversion < 0.0 or self.cvar_weight < 0.0:
            raise ValidationError("risk_aversion and cvar_weight must be non-negative")
        if not 0.0 < self.cvar_alpha < 1.0:
            raise ValidationError("cvar_alpha must be in (0, 1)")
        if self.max_legs is not None:
            require_int("max_legs", self.max_legs)
            if self.max_legs < 1:
                raise ValidationError("max_legs must be at least 1")
        if self.ticket_cost < 0.0:
            raise ValidationError("ticket_cost must be non-negative")
        require_int("max_lots_per_leg", self.max_lots_per_leg)
        if self.max_lots_per_leg < 1:
            raise ValidationError("max_lots_per_leg must be at least 1")
        if self.max_linear_quote_gap < 0.0:
            raise ValidationError("max_linear_quote_gap must be non-negative")
        if not 0.0 < self.mip_gap < 1.0:
            raise ValidationError("mip_gap must be in (0, 1)")

    def quality_policy(self) -> QualityPolicy:
        return QualityPolicy(max_age=self.max_quote_age, require_size=self.require_quote_size)


# Each fail-closed gate: its name in a request, its HedgeConfig field, and if higher is looser.
_GATES: tuple[tuple[str, str, bool], ...] = (
    ("max_quote_age_seconds", "max_quote_age", True),
    ("instability_threshold", "instability_threshold", True),
    ("regret_threshold", "regret_threshold", True),
    ("max_model_quote_vol_gap", "max_model_quote_vol_gap", True),
    ("max_linear_quote_gap", "max_linear_quote_gap", True),
    ("max_lots_per_leg", "max_lots_per_leg", True),
    ("allow_extrapolated_vol", "allow_extrapolated_vol", True),
    ("require_quote_size", "require_quote_size", False),
    ("mip_gap", "mip_gap", True),
    ("time_limit_seconds", "time_limit_seconds", True),
    ("mip_node_limit", "mip_node_limit", True),
)


def _gate_value(config: HedgeConfig, attribute: str) -> float | bool:
    value = getattr(config, attribute)
    return value.total_seconds() if isinstance(value, timedelta) else value


def gate_fields(config: HedgeConfig) -> dict[str, Any]:
    """The gates of `config` as HedgeConfig arguments, so another config can start from them."""
    return {attribute: getattr(config, attribute) for _, attribute, _ in _GATES}


def gate_overrides(
    config: HedgeConfig, baseline: HedgeConfig | None = None
) -> dict[str, float | bool]:
    """Every gate not at `baseline`, whichever way it moved, for an approver to see.

    A service passes its limits, which are what a request gets by omitting a gate. With no
    deployment the baseline is the library defaults.
    """
    standing = baseline if baseline is not None else HedgeConfig()
    return {
        name: _gate_value(config, attribute)
        for name, attribute, _ in _GATES
        if _gate_value(config, attribute) != _gate_value(standing, attribute)
    }


def loosened_gates(config: HedgeConfig, limits: HedgeConfig) -> tuple[str, ...]:
    """The gates `config` sets looser than `limits`, by their names in a request."""
    return tuple(
        name
        for name, attribute, higher_is_looser in _GATES
        if (_gate_value(config, attribute) > _gate_value(limits, attribute)) is higher_is_looser
        and _gate_value(config, attribute) != _gate_value(limits, attribute)
    )


def _reject_duplicates(kind: str, names: Sequence[str]) -> None:
    repeated = sorted(name for name, seen in Counter(names).items() if seen > 1)
    if repeated:
        raise ValidationError(f"duplicate {kind}: {repeated}")


@dataclass(frozen=True, slots=True)
class HedgeProblem:
    book: tuple[Position, ...]
    candidates: tuple[HedgeCandidate, ...]
    targets: tuple[GreekTarget, ...]
    scenarios: tuple[Scenario, ...] = ()
    config: HedgeConfig = field(default_factory=HedgeConfig)

    def __post_init__(self) -> None:
        keys = [(t.underlying, t.greek) for t in self.targets]
        if len(keys) != len(set(keys)):
            raise ValidationError("duplicate target for the same underlying and Greek")
        # Legs, limits, and scenario P&L are all looked up by these, so each must name one thing.
        _reject_duplicates("candidate", [c.instrument.key for c in self.candidates])
        _reject_duplicates("scenario", [s.name for s in self.scenarios])
        # Scenarios act on the objective only through CVaR. Without it they would do nothing, and
        # "no trade improves the objective" would read as a clean answer.
        if not self.targets and self.config.cvar_weight <= 0.0:
            raise ValidationError("a hedge problem needs targets, or scenarios with a cvar_weight")
        if self.config.cvar_weight > 0.0 and not self.scenarios:
            raise ValidationError("cvar_weight requires scenarios")


class ProposalStatus(StrEnum):
    OPTIMAL = "optimal"
    WITHIN_GAP = "within_gap"
    NO_TRADE = "no_trade"
    NO_USABLE_CANDIDATES = "no_usable_candidates"
    PARTIAL_COVERAGE = "partial_coverage"
    UNSTABLE = "unstable"


@dataclass(frozen=True, slots=True)
class HedgeLeg:
    instrument: Instrument
    quantity: int
    expected_price: float
    premium: float
    transaction_cost: float
    rationale: str

    @property
    def side(self) -> str:
        return "BUY" if self.quantity > 0 else "SELL"


@dataclass(frozen=True, slots=True)
class ExcludedCandidate:
    instrument: str
    reason: str


@dataclass(frozen=True, slots=True)
class ResidualLine:
    underlying: str
    greek: str
    before: float
    after: float
    lower: float | None
    upper: float | None


@dataclass(frozen=True, slots=True)
class VerificationReport:
    """Independent check: the proposal's book repriced through the valuation path, from scratch."""

    passed: bool
    max_abs_greek_error: float
    max_abs_scenario_error: float
    band_violations: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class StabilityReport:
    """How much the answer depends on inputs that are themselves estimates.

    `lot_change_ratio` is the worst change in option contracts after re-solving under a perturbed
    market, as a share of the base trade's contracts. `regret_ratio` is how much worse today's hedge
    is than the re-solved one in that market, as a share of the benefit of hedging. `drift` is the
    dollar change in a held hedge's penalized residual. `condition_number` is for inspection only.
    `hit_time_limit` says a re-solve stopped at the wall-clock backstop. `out_of_range` lists the
    perturbations the solver could not faithfully re-solve: unknown rather than unstable, so such a
    hedge is not called stable, and does not read as unstable unless `notes` says so too.
    """

    condition_number: float
    lot_change_ratio: float
    regret_ratio: float
    drift: Mapping[str, float]
    is_stable: bool
    notes: tuple[str, ...]
    hit_time_limit: bool = False
    out_of_range: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class RunManifest:
    """What a replay of the same request must match. Library versions are recorded because a new
    HiGHS or SciPy can change the search, and `run_stability` because it changes status and issues.
    """

    code_version: str
    market_fingerprint: str
    problem_fingerprint: str
    as_of: datetime
    solver: str
    scipy_version: str
    numpy_version: str
    run_stability: bool
    gate_overrides: Mapping[str, float | bool]


@dataclass(frozen=True, slots=True)
class HedgeProposal:
    status: ProposalStatus
    legs: tuple[HedgeLeg, ...]
    objective: float
    transaction_cost: float
    net_premium: float
    gross_lots: int
    residuals: tuple[ResidualLine, ...]
    before: Mapping[str, Exposure]
    after: Mapping[str, Exposure]
    scenario_pnl_before: Mapping[str, float]
    scenario_pnl_after: Mapping[str, float]
    excluded: tuple[ExcludedCandidate, ...]
    verification: VerificationReport
    stability: StabilityReport | None
    basis_risk_1d: float
    solver_gap: float
    objective_lower_bound: float
    issues: tuple[str, ...]
    manifest: RunManifest

    @property
    def quantities(self) -> dict[str, int]:
        return {leg.instrument.key: leg.quantity for leg in self.legs}

    def hedged_book(self, book: Sequence[Position]) -> tuple[Position, ...]:
        return (*book, *(Position(leg.instrument, leg.quantity) for leg in self.legs))


__all__ = [
    "DEFAULT_WEIGHTS",
    "EXPOSURE_NAMES",
    "ExcludedCandidate",
    "GreekName",
    "GreekTarget",
    "HedgeCandidate",
    "HedgeConfig",
    "HedgeLeg",
    "HedgeProblem",
    "HedgeProposal",
    "ProposalStatus",
    "ResidualLine",
    "RunManifest",
    "StabilityReport",
    "VerificationReport",
    "gate_fields",
    "gate_overrides",
    "loosened_gates",
]
