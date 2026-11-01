"""`gn`: the agent-friendly command line (AXI principles). TOON on stdout, no prompts.

The version fast path imports nothing heavy, and the engine loads only when a command runs.
Exit codes: 0 success, 1 domain or data error, 2 usage error. Errors are structured, on stdout.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Sequence
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

    example = commands.add_parser("example", help="Print a runnable sample request")
    example.add_argument("--kind", default="hedge", choices=["hedge", "book", "market"])
    return parser, dict(commands.choices)


def _load(path: Path) -> Any:
    from greeks_neutralizer.errors import ValidationError

    try:
        return json.loads(path.read_text())
    except OSError as exc:
        raise ValidationError(f"cannot read {path}: {exc.strerror}") from exc
    except json.JSONDecodeError as exc:
        raise ValidationError(f"{path} is not valid JSON: {exc}") from exc


def _home() -> dict[str, Any]:
    binary = str(Path(sys.argv[0]).resolve()).replace(str(Path.home()), "~", 1)
    return {
        "bin": binary,
        "description": DESCRIPTION,
        "version": __version__,
        "commands": ["value", "stress", "hedge", "frontier", "example"],
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
    from greeks_neutralizer.risk.scenarios import book_scenario_pnl
    from greeks_neutralizer.schema import SCENARIO_SETS, parse_book, parse_market

    if args.scenario_set not in SCENARIO_SETS:
        raise ValidationError(
            f"unknown scenario set {args.scenario_set!r}; valid: {sorted(SCENARIO_SETS)}"
        )
    book = parse_book(_load(args.book))
    scenarios = SCENARIO_SETS[args.scenario_set]
    pnl = book_scenario_pnl(book, parse_market(_load(args.market)), scenarios)
    rows = sorted(zip(scenarios, pnl, strict=True), key=lambda item: float(item[1]))
    return {
        "scenarios": len(rows),
        "worst": {"name": rows[0][0].name, "pnl": float(rows[0][1])},
        "pnl": [{"scenario": s.name, "pnl": float(v)} for s, v in rows],
    }


def _no_legs_text(status: str) -> str:
    if status == "no_usable_candidates":
        return "0 legs: every candidate was excluded, so this does not mean no hedge is needed"
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
    return {
        "frontier": [
            {
                "label": p.label,
                "cost": p.transaction_cost,
                "legs": p.n_legs,
                "residual": p.penalized_residual,
                "worst_stress": p.worst_scenario_pnl,
                "status": p.status or ("infeasible" if p.infeasible_reason else "baseline"),
                "reason": p.infeasible_reason or "",
                "issues": len(p.issues),
            }
            for p in hedge_frontier(problem, snapshot)
        ]
    }


def _cmd_example(args: argparse.Namespace) -> Any:
    from greeks_neutralizer.examples import example_book, example_market, example_request

    return {"hedge": example_request, "book": example_book, "market": example_market}[args.kind]()


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

    real_stdout = _capture_native_stdout()
    try:
        return _run(args_list, real_stdout, encode)
    finally:
        sys.stdout.flush()
        os.dup2(real_stdout, 1)
        os.close(real_stdout)


def _run(args_list: list[str], fd: int, encode: Any) -> int:
    parser, subparsers = _parser()
    try:
        # Unknown flags are rejected here, by the subcommand they were given to, so that the error
        # can list the flags that command accepts.
        args, unknown = parser.parse_known_args(args_list)
        if unknown:
            owner = subparsers.get(args.command or "", parser)
            owner.error(f"unknown argument(s): {' '.join(unknown)}")
    except UsageError as exc:
        _emit(fd, encode({"error": str(exc), "help": exc.help_text}))
        return 2
    except SystemExit as exc:  # --help
        sys.stdout.flush()
        return int(exc.code or 0)

    handlers = {
        "value": _cmd_value,
        "stress": _cmd_stress,
        "hedge": _cmd_hedge,
        "frontier": _cmd_frontier,
        "example": _cmd_example,
    }
    if args.command is None:
        _emit(fd, encode({"error": "no command given", "help": "Run `gn` for an overview"}))
        return 2

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
                    "help": "Widen the named bands, or add candidates that can move those Greeks",
                }
            ),
        )
        return 1
    except GNError as exc:
        _emit(fd, encode({"error": str(exc), "type": type(exc).__name__}))
        return 1
    if args.command == "example":
        _emit(fd, json.dumps(result, indent=2))
    else:
        _emit(fd, encode(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
