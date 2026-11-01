#!/usr/bin/env python3
"""Compare HISTORICAL_STRESSES with FRED closes: VIXCLS (1990 onward) and SP500 (last ten years).

Usage: check_stress_table.py <VIXCLS.csv> <SP500.csv>

Spot moves before the SP500 series begins cannot be checked here and are reported as such.
"""

from __future__ import annotations

import csv
import sys
from datetime import date
from pathlib import Path

from greeks_neutralizer.risk.scenarios import HISTORICAL_STRESSES


def load(path: Path) -> dict[date, float]:
    out: dict[date, float] = {}
    with path.open() as handle:
        for row in csv.reader(handle):
            try:
                out[date.fromisoformat(row[0])] = float(row[1])
            except (ValueError, IndexError):
                continue
    return out


def change(series: dict[date, float], day: date) -> tuple[float, float] | None:
    days = sorted(series)
    if day not in series:
        return None
    previous = days[days.index(day) - 1]
    return series[previous], series[day]


def main() -> None:
    vix = load(Path(sys.argv[1]))
    spx = load(Path(sys.argv[2]))
    print("scenario | table spot | actual spot | table vol pts | actual VIX pts")
    for scenario in HISTORICAL_STRESSES:
        day = date.fromisoformat(scenario.name.split()[0])
        spot = change(spx, day)
        vol = change(vix, day)
        actual_spot = f"{spot[1] / spot[0] - 1:+.2%}" if spot else "not in series"
        actual_vol = f"{vol[1] - vol[0]:+.2f} ({vol[0]} to {vol[1]})" if vol else "not in series"
        print(
            f"{scenario.name} | {scenario.spot_shock:+.2%} | {actual_spot} | "
            f"{scenario.vol_shock * 100:+.2f} | {actual_vol}"
        )


if __name__ == "__main__":
    main()
