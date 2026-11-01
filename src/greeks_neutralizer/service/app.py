"""HTTP API and web UI on top of the one engine. There is no duplicate math in the browser.

Security posture, stated plainly: this is bearer-token auth with per-actor identity, suitable for a
single-team VPC deployment behind TLS and a reverse proxy. It is not SSO. OIDC, RBAC, and tenant
isolation are not implemented, and the store is in-process, so restarting loses open proposals while
the audit log survives. See docs/operations/security.md.

Configuration, from the environment only:
  GN_TOKENS      required: comma-separated token=actor pairs, for example "s3cret=alice,t0ken=bob"
  GN_AUDIT_LOG   audit log path, default ./gn_audit.jsonl
  GN_MAX_BODY    maximum request size in bytes, default 5000000
  GN_MAX_MARKET_AGE_SECONDS  oldest market a proposal can be submitted on, default 900
  GN_GATE_MAX_QUOTE_AGE_SECONDS, GN_GATE_INSTABILITY_THRESHOLD, GN_GATE_REGRET_THRESHOLD,
  GN_GATE_MAX_MODEL_QUOTE_VOL_GAP, GN_GATE_MAX_LOTS_PER_LEG, GN_GATE_ALLOW_EXTRAPOLATED_VOL,
  GN_GATE_REQUIRE_QUOTE_SIZE, GN_GATE_MIP_GAP, GN_GATE_TIME_LIMIT_SECONDS, GN_GATE_MIP_NODE_LIMIT
                 each gate's limit, default the library default. A request that omits a gate gets
                 this limit, and one that sets it looser gets a 422. GATE_SETTINGS and
                 gate_overrides then list only what a request set differently from these limits
"""

from __future__ import annotations

import hmac
import json
import logging
import os
import time
import uuid
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime, timedelta
from importlib import resources
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse, Response

from greeks_neutralizer._version import __version__
from greeks_neutralizer.errors import (
    GNError,
    HedgeInfeasibleError,
    MarketDataError,
    SearchLimitError,
    ValidationError,
)
from greeks_neutralizer.examples import example_request
from greeks_neutralizer.hedging.diagnostics import hedge_decay_forecast, hedge_frontier
from greeks_neutralizer.hedging.optimizer import VerificationError, solve_hedge
from greeks_neutralizer.hedging.problem import HedgeConfig, HedgeProblem, loosened_gates
from greeks_neutralizer.market import MarketSnapshot
from greeks_neutralizer.pricing.book import value_book
from greeks_neutralizer.risk.measures import check_limits, expiry_ladder, expiry_risks
from greeks_neutralizer.risk.scenarios import book_scenario_pnl, spot_vol_grid
from greeks_neutralizer.schema import (
    SCENARIO_SETS,
    book_to_json,
    check_fields,
    decay_to_json,
    ladder_to_json,
    limit_check_to_json,
    liquidity_estimates,
    liquidity_to_json,
    parse_book,
    parse_days,
    parse_hedge_request,
    parse_limits,
    parse_market,
    proposal_to_json,
)
from greeks_neutralizer.workflow import (
    AuditLog,
    ProposalRecord,
    ProposalStore,
    WorkflowError,
    WorkflowPolicy,
)

logger = logging.getLogger("greeks_neutralizer.service")


def _parse_tokens(raw: str) -> dict[str, str]:
    tokens: dict[str, str] = {}
    for pair in raw.split(","):
        token, _, actor = pair.strip().partition("=")
        if not token or not actor:
            raise RuntimeError("GN_TOKENS must be comma-separated token=actor pairs")
        tokens[token] = actor
    return tokens


def _env_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    if raw not in ("true", "false"):
        raise RuntimeError(f"{name} must be true or false, got {raw!r}")
    return raw == "true"


def _gate_limits_from_env() -> HedgeConfig:
    """This service's gates: what an omitted gate takes, and the loosest a request may set.

    Unset means the library default, so there is no loosening.
    """
    base = HedgeConfig()
    env = os.environ
    return HedgeConfig(
        max_quote_age=timedelta(
            seconds=float(
                env.get("GN_GATE_MAX_QUOTE_AGE_SECONDS", base.max_quote_age.total_seconds())
            )
        ),
        instability_threshold=float(
            env.get("GN_GATE_INSTABILITY_THRESHOLD", base.instability_threshold)
        ),
        regret_threshold=float(env.get("GN_GATE_REGRET_THRESHOLD", base.regret_threshold)),
        max_model_quote_vol_gap=float(
            env.get("GN_GATE_MAX_MODEL_QUOTE_VOL_GAP", base.max_model_quote_vol_gap)
        ),
        max_lots_per_leg=int(env.get("GN_GATE_MAX_LOTS_PER_LEG", base.max_lots_per_leg)),
        allow_extrapolated_vol=_env_flag(
            "GN_GATE_ALLOW_EXTRAPOLATED_VOL", base.allow_extrapolated_vol
        ),
        require_quote_size=_env_flag("GN_GATE_REQUIRE_QUOTE_SIZE", base.require_quote_size),
        mip_gap=float(env.get("GN_GATE_MIP_GAP", base.mip_gap)),
        time_limit_seconds=float(env.get("GN_GATE_TIME_LIMIT_SECONDS", base.time_limit_seconds)),
        mip_node_limit=int(env.get("GN_GATE_MIP_NODE_LIMIT", base.mip_node_limit)),
    )


def _clean(value: Any) -> Any:
    """Make values JSON-safe. NaN and infinity become null rather than invalid JSON."""
    if isinstance(value, dict):
        return {k: _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    if isinstance(value, (float, np.floating)):
        number = float(value)
        return number if np.isfinite(number) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def _safe(payload: dict[str, Any]) -> dict[str, Any]:
    cleaned: dict[str, Any] = _clean(payload)
    return cleaned


def _record_json(record: ProposalRecord) -> dict[str, Any]:
    return {
        "id": record.id,
        "state": record.state.value,
        "created_by": record.created_by,
        "created_at": record.created_at.isoformat(),
        "decided_by": record.decided_by,
        "decided_at": record.decided_at.isoformat() if record.decided_at else None,
        "reason": record.reason,
        "proposal": proposal_to_json(record.proposal),
    }


def _record_summary(record: ProposalRecord) -> dict[str, Any]:
    return {
        "id": record.id,
        "state": record.state.value,
        "status": record.proposal.status.value,
        "issue_count": len(record.proposal.issues),
        "created_by": record.created_by,
        "created_at": record.created_at.isoformat(),
        "leg_count": len(record.proposal.legs),
        "transaction_cost": record.proposal.transaction_cost,
    }


def create_app(
    tokens: dict[str, str] | None = None,
    audit_path: Path | None = None,
    max_body: int | None = None,
    clock: Callable[[], datetime] | None = None,
    gate_limits: HedgeConfig | None = None,
) -> FastAPI:
    """Build the app. With no tokens configured it refuses to start, rather than running open."""
    token_map = tokens if tokens is not None else _parse_tokens(os.environ.get("GN_TOKENS", ""))
    if not token_map:
        raise RuntimeError("no API tokens configured: set GN_TOKENS")
    audit_file = audit_path or Path(os.environ.get("GN_AUDIT_LOG", "gn_audit.jsonl"))
    body_limit = max_body or int(os.environ.get("GN_MAX_BODY", "5000000"))
    max_age = float(os.environ.get("GN_MAX_MARKET_AGE_SECONDS", "900"))
    now = clock or (lambda: datetime.now(UTC))
    limits = gate_limits or _gate_limits_from_env()
    store = ProposalStore(
        AuditLog(audit_file), WorkflowPolicy(max_market_age=timedelta(seconds=max_age)), now
    )

    app = FastAPI(
        title="Greeks Neutralizer",
        version=__version__,
        description="Hedge decisions and options risk. Decision support: no order routing.",
    )
    app.state.store = store
    app.state.started = time.time()
    app.state.counts = {"requests": 0, "errors": 0}

    def parse_within_limits(payload: dict[str, Any]) -> tuple[HedgeProblem, MarketSnapshot, bool]:
        problem, snapshot, stability = parse_hedge_request(payload, gate_defaults=limits)
        loosened = loosened_gates(problem.config, limits)
        if loosened:
            raise ValidationError(
                f"this service does not allow {', '.join(loosened)} looser than its limit"
            )
        return problem, snapshot, stability

    def actor(authorization: str | None = Header(default=None)) -> str:
        if not authorization or not authorization.startswith("Bearer "):
            raise HTTPException(401, "missing bearer token")
        presented = authorization.removeprefix("Bearer ").strip()
        for token, name in token_map.items():
            if hmac.compare_digest(token.encode(), presented.encode()):
                return name
        raise HTTPException(401, "invalid token")

    @app.middleware("http")
    async def envelope(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        request_id = request.headers.get("x-request-id", str(uuid.uuid4()))
        length = request.headers.get("content-length")
        # The server frames a chunked body by its chunks whatever Content-Length says, so a declared
        # length cannot bound it. Any request with Transfer-Encoding is refused outright.
        if request.headers.get("transfer-encoding"):
            return JSONResponse({"error": "a Content-Length is required"}, status_code=411)
        if length and int(length) > body_limit:
            return JSONResponse(
                {"error": "request too large", "limit": body_limit}, status_code=413
            )
        started = time.perf_counter()
        app.state.counts["requests"] += 1
        try:
            response = await call_next(request)
        except Exception:
            app.state.counts["errors"] += 1
            logger.exception("unhandled error, request_id=%s", request_id)
            raise
        elapsed = time.perf_counter() - started
        if response.status_code >= 500:
            app.state.counts["errors"] += 1
        response.headers["x-request-id"] = request_id
        logger.info(
            json.dumps(
                {
                    "request_id": request_id,
                    "path": request.url.path,
                    "status": response.status_code,
                    "seconds": round(elapsed, 4),
                }
            )
        )
        return response

    @app.exception_handler(HedgeInfeasibleError)
    async def infeasible(_: Request, exc: HedgeInfeasibleError) -> JSONResponse:
        return JSONResponse(
            {"error": str(exc), "type": "infeasible", "binding": list(exc.binding)}, 422
        )

    @app.exception_handler(SearchLimitError)
    async def search_limit(_: Request, exc: SearchLimitError) -> JSONResponse:
        return JSONResponse({"error": str(exc), "type": "search_limit"}, 422)

    @app.exception_handler(ValidationError)
    async def invalid(_: Request, exc: ValidationError) -> JSONResponse:
        return JSONResponse({"error": str(exc), "type": "validation"}, 422)

    @app.exception_handler(MarketDataError)
    async def bad_data(_: Request, exc: MarketDataError) -> JSONResponse:
        return JSONResponse({"error": str(exc), "type": "market_data"}, 422)

    @app.exception_handler(ArithmeticError)
    async def out_of_range(_: Request, exc: ArithmeticError) -> JSONResponse:
        logger.warning("arithmetic error on request inputs: %s", exc)
        message = f"a calculation on these inputs is out of range: {exc}"
        return JSONResponse({"error": message, "type": "validation"}, 422)

    @app.exception_handler(WorkflowError)
    async def workflow_error(_: Request, exc: WorkflowError) -> JSONResponse:
        return JSONResponse({"error": str(exc), "type": "workflow"}, 409)

    @app.exception_handler(VerificationError)
    async def failed_verification(_: Request, exc: VerificationError) -> JSONResponse:
        logger.error("verification failed: %s", exc)
        return JSONResponse({"error": str(exc), "type": "verification_failed"}, 500)

    @app.exception_handler(GNError)
    async def engine_error(_: Request, exc: GNError) -> JSONResponse:
        return JSONResponse({"error": str(exc), "type": type(exc).__name__}, 500)

    @app.get("/healthz")
    def healthz() -> JSONResponse:
        """Liveness, and whether the audit chain is intact. A broken chain returns 503, so that a
        monitor watching status codes sees it: it is an incident, and approvals are blocked."""
        check = store.audit.verify()
        return JSONResponse(
            {
                "status": "ok" if check.ok else "audit_chain_broken",
                "version": __version__,
                "audit_records": check.records,
                "detail": check.reason,
            },
            status_code=200 if check.ok else 503,
        )

    @app.get("/metrics", response_class=PlainTextResponse)
    def metrics() -> str:
        counts = app.state.counts
        return (
            f"gn_requests_total {counts['requests']}\n"
            f"gn_server_errors_total {counts['errors']}\n"
            f"gn_uptime_seconds {time.time() - app.state.started:.0f}\n"
        )

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return resources.files("greeks_neutralizer.service").joinpath("ui.html").read_text()

    @app.get("/v1/whoami")
    def whoami(user: str = Depends(actor)) -> dict[str, str]:
        return {"actor": user}

    @app.get("/v1/example")
    def example(user: str = Depends(actor)) -> dict[str, Any]:
        # Dated now, so that the sample can pass the market-age check on submission.
        return example_request(now())

    @app.post("/v1/value")
    def value(payload: dict[str, Any], user: str = Depends(actor)) -> dict[str, Any]:
        check_fields(payload, "request", {"book", "market"})
        book = parse_book(payload.get("book"))
        snapshot = parse_market(payload.get("market"))
        result = book_to_json(value_book(book, snapshot))
        result["expiry_risks"] = [
            {"kind": f.kind, "contract": f.contract, "quantity": f.quantity, "detail": f.detail}
            for f in expiry_risks(book, snapshot)
        ]
        return _safe(result)

    @app.post("/v1/stress")
    def stress(payload: dict[str, Any], user: str = Depends(actor)) -> dict[str, Any]:
        check_fields(payload, "request", {"book", "market"}, {"set"})
        book = parse_book(payload.get("book"))
        snapshot = parse_market(payload.get("market"))
        name = payload.get("set", "all")
        if not isinstance(name, str) or name not in SCENARIO_SETS:
            raise ValidationError(f"unknown scenario set {name!r}; valid: {sorted(SCENARIO_SETS)}")
        scenarios = SCENARIO_SETS[name]
        pnl = book_scenario_pnl(book, snapshot, scenarios)
        return _safe(
            {
                "pnl": [
                    {"scenario": s.name, "pnl": float(v)}
                    for s, v in zip(scenarios, pnl, strict=True)
                ],
                "issues": list(value_book(book, snapshot).issues),
            }
        )

    @app.post("/v1/grid")
    def grid(payload: dict[str, Any], user: str = Depends(actor)) -> dict[str, Any]:
        """A spot and vol grid by full repricing: rows are spot shocks and columns vol shocks."""
        check_fields(payload, "request", {"book", "market"})
        book = parse_book(payload.get("book"))
        snapshot = parse_market(payload.get("market"))
        spots = (-0.2, -0.1, -0.05, -0.02, 0.0, 0.02, 0.05, 0.1)
        vols = (-0.05, 0.0, 0.05, 0.1, 0.2)
        pnl = book_scenario_pnl(book, snapshot, spot_vol_grid(spots, vols))
        return _safe(
            {
                "spot_shocks": list(spots),
                "vol_shocks": list(vols),
                "pnl": np.asarray(pnl).reshape(len(spots), len(vols)).tolist(),
                "issues": list(value_book(book, snapshot).issues),
            }
        )

    @app.post("/v1/ladder")
    def ladder(payload: dict[str, Any], user: str = Depends(actor)) -> dict[str, Any]:
        """Option and future exposures by New York expiry date, shortest first."""
        check_fields(payload, "request", {"book", "market"})
        book = parse_book(payload.get("book"))
        snapshot = parse_market(payload.get("market"))
        return _safe(ladder_to_json(expiry_ladder(book, snapshot)))

    @app.post("/v1/limits")
    def limits_check(payload: dict[str, Any], user: str = Depends(actor)) -> dict[str, Any]:
        """Which of the limits in the request this book breaches. It only reports."""
        check_fields(payload, "request", {"book", "market", "limits"})
        book = parse_book(payload.get("book"))
        snapshot = parse_market(payload.get("market"))
        risk_limits = parse_limits(payload.get("limits"))
        return _safe(limit_check_to_json(check_limits(book, snapshot, risk_limits)))

    @app.post("/v1/liquidity")
    def liquidity(payload: dict[str, Any], user: str = Depends(actor)) -> dict[str, Any]:
        """Half-spread cost and days to exit, from the spread and volume the request supplies."""
        check_fields(payload, "request", {"lines"})
        return _safe(liquidity_to_json(liquidity_estimates(payload.get("lines"))))

    @app.post("/v1/decay")
    def decay(payload: dict[str, Any], user: str = Depends(actor)) -> dict[str, Any]:
        """How one underlying's exposures drift as time passes, with spot and vol unchanged."""
        check_fields(payload, "request", {"book", "market", "underlying"}, {"days"})
        book = parse_book(payload.get("book"))
        snapshot = parse_market(payload.get("market"))
        underlying = payload["underlying"]
        if "days" in payload:
            forecast = hedge_decay_forecast(book, snapshot, underlying, parse_days(payload["days"]))
        else:
            forecast = hedge_decay_forecast(book, snapshot, underlying)
        return _safe(decay_to_json(forecast, underlying))

    @app.post("/v1/hedge")
    def hedge(payload: dict[str, Any], user: str = Depends(actor)) -> dict[str, Any]:
        problem, snapshot, stability = parse_within_limits(payload)
        proposal = solve_hedge(problem, snapshot, run_stability=stability, gate_baseline=limits)
        return _safe(proposal_to_json(proposal))

    @app.post("/v1/frontier")
    def frontier(payload: dict[str, Any], user: str = Depends(actor)) -> dict[str, Any]:
        problem, snapshot, _ = parse_within_limits(payload)
        points = hedge_frontier(problem, snapshot, gate_baseline=limits)
        return _safe(
            {
                "points": [
                    {
                        "label": p.label,
                        "risk_aversion": p.risk_aversion,
                        "transaction_cost": p.transaction_cost,
                        "n_legs": p.n_legs,
                        "penalized_residual": p.penalized_residual,
                        "worst_scenario_pnl": p.worst_scenario_pnl,
                        "unsolved_reason": p.unsolved_reason,
                        "status": p.status,
                        "issues": list(p.issues),
                    }
                    for p in points
                ]
            }
        )

    @app.post("/v1/proposals", status_code=201)
    def submit(payload: dict[str, Any], user: str = Depends(actor)) -> dict[str, Any]:
        problem, snapshot, stability = parse_within_limits(payload)
        proposal = solve_hedge(problem, snapshot, run_stability=stability, gate_baseline=limits)
        return _safe(_record_json(store.submit(proposal, user)))

    @app.get("/v1/proposals")
    def list_proposals(
        limit: int = Query(50, ge=1, le=500), user: str = Depends(actor)
    ) -> dict[str, Any]:
        """Summaries, newest first. `total` shows when `limit` cut the list. A full proposal comes
        from GET /v1/proposals/{id}."""
        records = store.list()
        shown = records[:limit]
        return _safe(
            {
                "total": len(records),
                "count": len(shown),
                "proposals": [_record_summary(r) for r in shown],
            }
        )

    @app.get("/v1/proposals/{record_id}")
    def get_proposal(record_id: str, user: str = Depends(actor)) -> dict[str, Any]:
        return _safe(_record_json(store.get(record_id)))

    @app.post("/v1/proposals/{record_id}/approve")
    def approve(
        record_id: str, payload: dict[str, Any] | None = None, user: str = Depends(actor)
    ) -> dict[str, Any]:
        reason = (payload or {}).get("reason", "")
        return _safe(_record_json(store.approve(record_id, user, reason)))

    @app.post("/v1/proposals/{record_id}/reject")
    def reject(
        record_id: str, payload: dict[str, Any], user: str = Depends(actor)
    ) -> dict[str, Any]:
        return _safe(_record_json(store.reject(record_id, user, payload.get("reason", ""))))

    @app.post("/v1/proposals/{record_id}/tickets", response_class=PlainTextResponse)
    def tickets(record_id: str, fmt: str = "csv", user: str = Depends(actor)) -> str:
        return store.export_tickets(record_id, user, fmt)

    @app.get("/v1/audit")
    def audit(limit: int = Query(50, ge=1), user: str = Depends(actor)) -> dict[str, Any]:
        check = store.audit.verify()
        records, unreadable = store.audit.read()
        return _safe(
            {
                "chain_ok": check.ok,
                "total": check.records,
                "first_bad_line": check.first_bad_line,
                "reason": check.reason,
                "unreadable_lines": unreadable,
                "records": records[-limit:][::-1],
            }
        )

    return app


def factory() -> FastAPI:
    """For `uvicorn --factory greeks_neutralizer.service.app:factory`."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    return create_app()
