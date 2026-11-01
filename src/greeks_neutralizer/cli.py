"""`gn`: the agent-friendly command line (AXI principles). TOON on stdout, no prompts.

The version fast path imports nothing heavy, and the engine loads only when a command runs.
Exit codes: 0 success, 1 domain or data error, 2 usage error. Errors are structured, on stdout.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, NoReturn

from greeks_neutralizer._version import __version__

DESCRIPTION = "Options hedge decisions and risk: value books, stress them, and propose hedges"


class UsageError(Exception):
    def __init__(self, message: str, help_text: str) -> None:
        super().__init__(message)
        self.help_text = help_text


class _Parser(argparse.ArgumentParser):
    """Unknown flags and bad arguments become structured errors listing the valid flags."""

    def error(self, message: str) -> NoReturn:
        flags = sorted(
            option
            for action in self._actions
            for option in action.option_strings
            if option not in ("-h", "--help")
        )
        raise UsageError(message, f"valid flags for `{self.prog}`: {', '.join(flags)}")


def _parser() -> tuple[_Parser, dict[str, _Parser]]:
    parser = _Parser(prog="gn", description=DESCRIPTION, add_help=True)
    commands = parser.add_subparsers(dest="command", parser_class=_Parser)

    value = commands.add_parser("value", help="Value a book and report its Greeks")
    value.add_argument("--book", type=Path, required=True, help="JSON list of positions")
    value.add_argument("--market", type=Path, required=True, help="JSON market snapshot")
    value.add_argument("--full", action="store_true", help="Include every position line")

    stress = commands.add_parser("stress", help="Reprice a book under scenarios")
    stress.add_argument("--book", type=Path, required=True)
    stress.add_argument("--market", type=Path, required=True)
    stress.add_argument(
        "--set", dest="scenario_set", default="all", help="historical, hypothetical, or all"
    )

    hedge = commands.add_parser("hedge", help="Propose a hedge from a request file")
    hedge.add_argument("--request", type=Path, required=True, help="JSON hedge request")
    hedge.add_argument("--full", action="store_true", help="Include Greeks and every scenario")

    frontier = commands.add_parser("frontier", help="Cost against risk, with baselines")
    frontier.add_argument("--request", type=Path, required=True)

    ladder = commands.add_parser("ladder", help="Option and future exposures by expiry date")
    ladder.add_argument("--book", type=Path, required=True)
    ladder.add_argument("--market", type=Path, required=True)
    ladder.add_argument("--full", action="store_true", help="Include every exposure")

    limits = commands.add_parser("limits", help="Report which risk limits a book breaches")
    limits.add_argument("--book", type=Path, required=True)
    limits.add_argument("--market", type=Path, required=True)
    limits.add_argument("--limits", type=Path, required=True, help="JSON list of limits")

    liquidity = commands.add_parser("liquidity", help="Half-spread cost and days to exit")
    liquidity.add_argument("--lines", type=Path, required=True, help="JSON list of lines")

    decay = commands.add_parser("decay", help="How exposures drift as time passes")
    decay.add_argument("--book", type=Path, required=True)
    decay.add_argument("--market", type=Path, required=True)
    decay.add_argument("--underlying", required=True)
    decay.add_argument("--days", type=float, nargs="+", help="Days forward, default 1 2 5 10 21")
    decay.add_argument("--full", action="store_true", help="Include every exposure")

    example = commands.add_parser("example", help="Print a runnable sample request")
    example.add_argument(
        "--kind", default="hedge", choices=["hedge", "book", "market", "limits", "liquidity"]
    )
    return parser, dict(commands.choices)


def _load(path: Path) -> Any:
    from greeks_neutralizer.errors import ValidationError

    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValidationError(f"cannot read {path}: {exc.strerror}") from exc
    except (ValueError, RecursionError) as exc:
        raise ValidationError(f"{path} is not valid JSON: {exc}") from exc


def _home() -> dict[str, Any]:
    binary = str(Path(sys.argv[0]).resolve()).replace(str(Path.home()), "~", 1)
    return {
        "bin": binary,
        "description": DESCRIPTION,
        "version": __version__,
        "commands": [
            "value",
            "stress",
            "hedge",
            "frontier",
            "ladder",
            "limits",
            "liquidity",
            "decay",
            "example",
        ],
        "help": [
            "Run `gn example --kind hedge > request.json` for a runnable sample",
            "Run `gn hedge --request request.json` to propose a hedge",
            "Run `gn value --book book.json --market market.json` for Greeks",
        ],
    }


def _cmd_value(args: argparse.Namespace) -> dict[str, Any]:
    from greeks_neutralizer.pricing.book import value_book
    from greeks_neutralizer.schema import book_to_json, parse_book, parse_market

    book = parse_book(_load(args.book))
    if not book:
        return {"book": "0 positions: nothing to value", "help": ["Add positions to the book file"]}
    result = book_to_json(value_book(book, parse_market(_load(args.market))))
    summary: dict[str, Any] = {
        "as_of": result["as_of"],
        "positions": len(result["lines"]),
        "totals": result["totals"],
        "by_underlying": result["by_underlying"],
        "issues": result["issues"],
    }
    if args.full:
        summary["lines"] = [
            {
                "instrument": line["instrument"],
                "quantity": line["quantity"],
                "price": line["unit_price"],
                "vol": line["vol"],
                "delta_shares": line["exposure"]["delta_shares"],
                "vega_1pt": line["exposure"]["vega_1pt"],
            }
            for line in result["lines"]
        ]
    else:
        summary["help"] = [f"Run with --full to see all {len(result['lines'])} position lines"]
    return summary


def _cmd_stress(args: argparse.Namespace) -> dict[str, Any]:
    from greeks_neutralizer.errors import ValidationError
    from greeks_neutralizer.pricing.book import value_book
    from greeks_neutralizer.risk.scenarios import book_scenario_pnl
    from greeks_neutralizer.schema import SCENARIO_SETS, parse_book, parse_market

    if args.scenario_set not in SCENARIO_SETS:
        raise ValidationError(
            f"unknown scenario set {args.scenario_set!r}; valid: {sorted(SCENARIO_SETS)}"
        )
    book = parse_book(_load(args.book))
    scenarios = SCENARIO_SETS[args.scenario_set]
    snapshot = parse_market(_load(args.market))
    pnl = book_scenario_pnl(book, snapshot, scenarios)
    rows = sorted(zip(scenarios, pnl, strict=True), key=lambda item: float(item[1]))
    return {
        "scenarios": len(rows),
        "worst": {"name": rows[0][0].name, "pnl": float(rows[0][1])},
        "pnl": [{"scenario": s.name, "pnl": float(v)} for s, v in rows],
        "issues": list(value_book(book, snapshot).issues),
    }


def _no_legs_text(status: str) -> str:
    if status == "no_usable_candidates":
        return "0 legs: every candidate was excluded, so this does not mean no hedge is needed"
    if status == "partial_coverage":
        return "0 legs: a targeted underlying has no usable candidate, so see the issues"
    return "0 legs: no trade improves the objective"


def _hedge_summary(proposal: Any, full: bool) -> dict[str, Any]:
    from greeks_neutralizer.schema import proposal_to_json

    data = proposal_to_json(proposal)
    legs = [
        {
            "side": leg["side"],
            "qty": abs(leg["quantity"]),
            "instrument": leg["description"],
            "price": leg["expected_price"],
            "cost": leg["transaction_cost"],
        }
        for leg in data["legs"]
    ]
    worst_before = min(data["scenario_pnl_before"].values(), default=None)
    worst_after = min(data["scenario_pnl_after"].values(), default=None)
    out: dict[str, Any] = {
        "status": data["status"],
        "legs": legs if legs else _no_legs_text(data["status"]),
        "transaction_cost": data["transaction_cost"],
        "net_premium": data["net_premium"],
        "objective": data["objective"],
        "objective_lower_bound": data["objective_lower_bound"],
        "verified": data["verification"]["passed"],
        "residuals": [
            {
                "underlying": r["underlying"],
                "greek": r["greek"],
                "before": r["before"],
                "after": r["after"],
            }
            for r in data["residuals"]
        ],
        "worst_stress_before": worst_before,
        "worst_stress_after": worst_after,
        "excluded": f"{len(data['excluded'])} candidates excluded",
        "issues": data["issues"],
    }
    if data["stability"] is not None:
        out["stable"] = data["stability"]["is_stable"]
    if full:
        out["excluded_detail"] = data["excluded"]
        out["scenario_pnl_after"] = data["scenario_pnl_after"]
        out["manifest"] = data["manifest"]
        out["rationale"] = [leg["rationale"] for leg in data["legs"]]
    else:
        out["help"] = ["Run with --full for rationale, exclusions, scenarios, and the manifest"]
    return out


def _cmd_hedge(args: argparse.Namespace) -> dict[str, Any]:
    from greeks_neutralizer.hedging.optimizer import solve_hedge
    from greeks_neutralizer.schema import parse_hedge_request

    problem, snapshot, stability = parse_hedge_request(_load(args.request))
    return _hedge_summary(solve_hedge(problem, snapshot, run_stability=stability), args.full)


def _cmd_frontier(args: argparse.Namespace) -> dict[str, Any]:
    from greeks_neutralizer.hedging.diagnostics import hedge_frontier
    from greeks_neutralizer.schema import parse_hedge_request

    problem, snapshot, _ = parse_hedge_request(_load(args.request))
    points = hedge_frontier(problem, snapshot)
    return {
        "frontier": [
            {
                "label": p.label,
                "cost": p.transaction_cost,
                "legs": p.n_legs,
                "residual": p.penalized_residual,
                "worst_stress": p.worst_scenario_pnl,
                "status": p.status or "baseline",
                "reason": p.unsolved_reason or "",
                "issues": len(p.issues),
            }
            for p in points
        ],
        "issues": list(dict.fromkeys(issue for p in points for issue in p.issues)),
    }


_EXPOSURE_SUMMARY = ("delta_dollars", "gamma_1pct", "vega_1pt", "theta_1d")


def _exposure_row(exposure: dict[str, float], full: bool) -> dict[str, float]:
    return exposure if full else {name: exposure[name] for name in _EXPOSURE_SUMMARY}


def _cmd_ladder(args: argparse.Namespace) -> dict[str, Any]:
    from greeks_neutralizer.risk.measures import expiry_ladder
    from greeks_neutralizer.schema import ladder_to_json, parse_book, parse_market

    book = parse_book(_load(args.book))
    data = ladder_to_json(expiry_ladder(book, parse_market(_load(args.market))))
    if not data["count"]:
        return {"ladder": "0 dated lines: nothing to ladder", "issues": data["issues"]}
    out: dict[str, Any] = {
        "expiries": data["count"],
        "ladder": [
            {
                "expiry": bucket["expiry"],
                "lines": bucket["n_lines"],
                **_exposure_row(bucket["exposure"], args.full),
            }
            for bucket in data["buckets"]
        ],
        "issues": data["issues"],
    }
    if not args.full:
        out["help"] = ["Run with --full for every exposure"]
    return out


def _cmd_limits(args: argparse.Namespace) -> dict[str, Any]:
    from greeks_neutralizer.risk.measures import check_limits
    from greeks_neutralizer.schema import (
        limit_check_to_json,
        parse_book,
        parse_limits,
        parse_market,
    )

    limits = parse_limits(_load(args.limits))
    book = parse_book(_load(args.book))
    data = limit_check_to_json(check_limits(book, parse_market(_load(args.market)), limits))
    return {
        "checked": data["checked"],
        "breached": data["breached"],
        "breaches": data["breaches"] or f"0 of {data['checked']} limits breached",
        "issues": data["issues"],
    }


def _cmd_liquidity(args: argparse.Namespace) -> dict[str, Any]:
    from greeks_neutralizer.schema import liquidity_estimates, liquidity_to_json

    data = liquidity_to_json(liquidity_estimates(_load(args.lines)))
    if not data["count"]:
        return {"lines": "0 lines: nothing to estimate"}
    return data


def _cmd_decay(args: argparse.Namespace) -> dict[str, Any]:
    from greeks_neutralizer.hedging.diagnostics import hedge_decay_forecast
    from greeks_neutralizer.schema import decay_to_json, parse_book, parse_market

    book = parse_book(_load(args.book))
    snapshot = parse_market(_load(args.market))
    if args.days is None:
        forecast = hedge_decay_forecast(book, snapshot, args.underlying)
    else:
        forecast = hedge_decay_forecast(book, snapshot, args.underlying, args.days)
    data = decay_to_json(forecast, args.underlying)
    out: dict[str, Any] = {
        "underlying": data["underlying"],
        "decay": [
            {"days": step["days_forward"], **_exposure_row(step["exposure"], args.full)}
            for step in data["steps"]
        ],
        "issues": data["issues"],
    }
    if not args.full:
        out["help"] = ["Run with --full for every exposure"]
    return out


def _cmd_example(args: argparse.Namespace) -> Any:
    from greeks_neutralizer.examples import (
        example_book,
        example_limits,
        example_liquidity,
        example_market,
        example_request,
    )

    samples: dict[str, Callable[[], Any]] = {
        "hedge": example_request,
        "book": example_book,
        "market": example_market,
        "limits": example_limits,
        "liquidity": example_liquidity,
    }
    return samples[args.kind]()


def _capture_native_stdout() -> int:
    """Send native-library stdout to stderr, returning the real stdout to write results to.

    The solver's native code can print diagnostics directly to file descriptor 1, which would
    corrupt the structured output an agent parses.
    """
    sys.stdout.flush()
    real = os.dup(1)
    os.dup2(2, 1)
    return real


def _emit(fd: int, text: str) -> None:
    os.write(fd, (text + "\n").encode())


def main(argv: Sequence[str] | None = None) -> int:
    args_list = list(sys.argv[1:] if argv is None else argv)
    if args_list in (["--version"], ["-v"], ["-V"]):
        print(__version__)
        return 0

    from greeks_neutralizer.toon import encode

    if not args_list:
        print(encode(_home()))
        return 0

    # Help and usage errors come first and go to the real stdout. Only the engine needs guarding.
    parsed = _parse(args_list, encode)
    if isinstance(parsed, int):
        return parsed

    real_stdout = _capture_native_stdout()
    try:
        return _run(parsed, real_stdout, encode)
    finally:
        sys.stdout.flush()
        os.dup2(real_stdout, 1)
        os.close(real_stdout)


def _parse(args_list: list[str], encode: Any) -> argparse.Namespace | int:
    """The parsed arguments, or the exit code when parsing has already said all there is to say."""
    parser, subparsers = _parser()
    try:
        # Unknown flags are rejected here, by the subcommand they were given to, so that the error
        # can list the flags that command accepts.
        args, unknown = parser.parse_known_args(args_list)
        if unknown:
            owner = subparsers.get(args.command or "", parser)
            owner.error(f"unknown argument(s): {' '.join(unknown)}")
    except UsageError as exc:
        print(encode({"error": str(exc), "help": exc.help_text}))
        return 2
    except SystemExit as exc:  # --help
        return int(exc.code or 0)
    if args.command is None:
        print(encode({"error": "no command given", "help": "Run `gn` for an overview"}))
        return 2
    return args


def _run(args: argparse.Namespace, fd: int, encode: Any) -> int:
    handlers = {
        "value": _cmd_value,
        "stress": _cmd_stress,
        "hedge": _cmd_hedge,
        "frontier": _cmd_frontier,
        "ladder": _cmd_ladder,
        "limits": _cmd_limits,
        "liquidity": _cmd_liquidity,
        "decay": _cmd_decay,
        "example": _cmd_example,
    }
    from greeks_neutralizer.errors import GNError, HedgeInfeasibleError

    try:
        result = handlers[args.command](args)
    except HedgeInfeasibleError as exc:
        _emit(
            fd,
            encode(
                {
                    "error": str(exc),
                    "binding": list(exc.binding),
                    "help": (
                        "Widen the named bands, raise max_legs if named, "
                        "or add candidates that can move those Greeks"
                    ),
                }
            ),
        )
        return 1
    except GNError as exc:
        _emit(fd, encode({"error": str(exc), "type": type(exc).__name__}))
        return 1
    except ArithmeticError as exc:
        message = f"a calculation on these inputs is out of range: {exc}"
        _emit(fd, encode({"error": message, "type": type(exc).__name__}))
        return 1
    if args.command == "example":
        _emit(fd, json.dumps(result, indent=2))
    else:
        _emit(fd, encode(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
