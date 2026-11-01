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

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import StrEnum

from greeks_neutralizer.data.quality import QualityPolicy
from greeks_neutralizer.domain import Instrument, Position, Quote
from greeks_neutralizer.errors import ValidationError
from greeks_neutralizer.pricing.book import EXPOSURE_NAMES, Exposure
from greeks_neutralizer.risk.scenarios import Scenario


@dataclass(frozen=True, slots=True)
class HedgeCandidate:
    """A tradable hedge with a live market. Marks and costs come from `quote`, never from a model.

    Size is in two tiers. Up to displayed size trades at the touch. Beyond it, up to `max_buy` or
    `max_sell`, it pays `impact_per_unit` in dollars per contract more, which keeps the problem
    linear and makes the solver work through liquidity in cost order.
    """

    instrument: Instrument
    quote: Quote
    fee_per_unit: float = 0.0
    max_buy: int | None = None
    max_sell: int | None = None
    impact_per_unit: float = 0.0
    lot_size: int = 1
    displayed_only: bool = False

    def __post_init__(self) -> None:
        if self.fee_per_unit < 0.0 or self.impact_per_unit < 0.0:
            raise ValidationError("fees and impact must be non-negative")
        if self.lot_size < 1:
            raise ValidationError("lot_size must be at least 1")
        for name, value in (("max_buy", self.max_buy), ("max_sell", self.max_sell)):
            if value is not None and value < 0:
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
    GreekName.GAMMA: 0.5 * 0.01 * 3.0,  # second-order P&L on a 3% move, in units of gamma per 1%
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
        if not isinstance(self.greek, GreekName):
            raise ValidationError(f"greek must be a GreekName, got {self.greek!r}")
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
    time_limit_seconds: float = 1.0
    max_quote_age: timedelta = timedelta(seconds=30)
    require_quote_size: bool = True
    allow_extrapolated_vol: bool = False
    instability_threshold: float = 0.5

    def __post_init__(self) -> None:
        if self.risk_aversion < 0.0 or self.cvar_weight < 0.0:
            raise ValidationError("risk_aversion and cvar_weight must be non-negative")
        if not 0.0 < self.cvar_alpha < 1.0:
            raise ValidationError("cvar_alpha must be in (0, 1)")
        if self.max_legs is not None and self.max_legs < 1:
            raise ValidationError("max_legs must be at least 1")
        if self.ticket_cost < 0.0:
            raise ValidationError("ticket_cost must be non-negative")
        if not 0.0 < self.mip_gap < 1.0:
            raise ValidationError("mip_gap must be in (0, 1)")

    def quality_policy(self) -> QualityPolicy:
        return QualityPolicy(max_age=self.max_quote_age, require_size=self.require_quote_size)


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
        if not self.targets and not self.scenarios:
            raise ValidationError("a hedge problem needs targets or scenarios to act on")
        if self.config.cvar_weight > 0.0 and not self.scenarios:
            raise ValidationError("cvar_weight requires scenarios")


class ProposalStatus(StrEnum):
    OPTIMAL = "optimal"
    WITHIN_GAP = "within_gap"
    NO_TRADE = "no_trade"
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

    `lot_change_ratio` is the worst L1 change in lots after re-solving under a perturbed market,
    divided by the base solution's lots. `drift` is the worst dollar change in a held hedge's
    penalized residual without re-solving.
    """

    condition_number: float
    lot_change_ratio: float
    drift: Mapping[str, float]
    is_stable: bool
    notes: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class RunManifest:
    """Everything needed to reproduce a proposal."""

    code_version: str
    market_fingerprint: str
    problem_fingerprint: str
    as_of: datetime
    solver: str


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
]
