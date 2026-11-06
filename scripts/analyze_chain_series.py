#!/usr/bin/env python3
"""Post-hoc diagnostics for the chain backtest: does mark noise explain the option hedges' variance?

Usage: analyze_chain_series.py <series_dir> --out horizons.json

Added after the pre-registered results were read, to explain them, so it does not change the
pre-registered decision. <series_dir> is the --series output of scripts/run_chain_backtest.py.

The idea: a model mark that is wrong by e at the end of one period is the start mark of the next,
so its error enters one period's P&L as +e and the next as -e. Summed over k periods, that noise
stays near the size of one mark error, while real P&L variance grows with k. So if mark noise
drives the option hedges' excess variance at one period, their variance ratio against delta falls
toward its true value as k grows, and their period-to-period P&L differences against delta are
negatively autocorrelated, near -0.5 when noise dominates. If the ratio stays high at every k,
the excess is real hedge P&L, not marks.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

HORIZONS = (1, 5, 10, 20)
BASELINE = "delta_daily"
DRAWS = 2000
SEED = 20261105


def load(path: Path) -> dict[str, np.ndarray]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    out: dict[str, np.ndarray] = {"as_of": np.array([r["as_of"] for r in rows])}
    for key in rows[0]:
        if key != "as_of":
            out[key] = np.array([float(r[key]) for r in rows])
    return out


def window_sums(values: np.ndarray, k: int) -> np.ndarray:
    n = values.size // k * k
    return values[:n].reshape(-1, k).sum(axis=1)


def reduction_with_ci(policy: np.ndarray, baseline: np.ndarray, block: int) -> dict[str, float]:
    """1 - var(policy) / var(baseline), with a circular block bootstrap over paired windows."""
    rng = np.random.default_rng(SEED)
    n = policy.size
    blocks = int(np.ceil(n / block))
    starts = rng.integers(0, n, size=(DRAWS, blocks))
    idx = ((starts[:, :, None] + np.arange(block)) % n).reshape(DRAWS, -1)[:, :n]
    ratio = policy[idx].var(axis=1, ddof=1) / baseline[idx].var(axis=1, ddof=1)
    low, high = np.quantile(1.0 - ratio, [0.025, 0.975])
    return {
        "windows": int(n),
        "variance_reduction": float(1.0 - policy.var(ddof=1) / baseline.var(ddof=1)),
        "ci_low": float(low),
        "ci_high": float(high),
    }


def lag1(values: np.ndarray) -> float:
    centered = values - values.mean()
    return float(np.dot(centered[:-1], centered[1:]) / np.dot(centered, centered))


def analyze(folder: Path, book: str) -> dict[str, object]:
    files = {p.stem.split("__")[1]: p for p in sorted(folder.glob(f"{book}__*__1.csv"))}
    if BASELINE not in files:
        raise SystemExit(f"{book}: no {BASELINE} series in {folder}")
    series = {name: load(path) for name, path in files.items()}
    base = series[BASELINE]
    out: dict[str, object] = {}
    for name, s in sorted(series.items()):
        if not np.array_equal(s["as_of"], base["as_of"]):
            raise SystemExit(f"{book}/{name}: dates differ from {BASELINE}")
        entry: dict[str, object] = {
            "periods": int(s["pnl"].size),
            "total_pnl": float(s["pnl"].sum()),
            "hedge_cost": float(s["hedge_cost"].sum()),
            "hedge_option_cost": float(s["hedge_option_cost"].sum()),
            "option_contracts_per_period": float(s["hedge_option_contracts"].mean()),
            "mean_hedge_pnl_ex_cost": float(s["hedge_pnl"].mean()),
        }
        if name != BASELINE:
            entry["lag1_autocorrelation_of_difference"] = lag1(s["pnl"] - base["pnl"])
            entry["by_horizon"] = {
                str(k): reduction_with_ci(
                    window_sums(s["pnl"], k), window_sums(base["pnl"], k), max(2, 20 // k)
                )
                for k in HORIZONS
            }
        out[name] = entry
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("series", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = {
        "note": (
            "Post-hoc: added after the pre-registered chain results were read, to test whether "
            "model-mark noise explains them. It does not change the pre-registered decision."
        ),
        "baseline": BASELINE,
        "horizons_in_periods": list(HORIZONS),
        "books": {
            book: analyze(args.series, book) for book in ("short_strangle", "long_put_overlay")
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
