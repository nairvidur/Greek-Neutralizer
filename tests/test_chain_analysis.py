"""The post-hoc horizon check separates reversing mark noise from real P&L variance."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def _analysis(monkeypatch: pytest.MonkeyPatch) -> Any:
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    return importlib.import_module("analyze_chain_series")


def test_window_sums_and_lag1_match_hand_values(monkeypatch: pytest.MonkeyPatch) -> None:
    a = _analysis(monkeypatch)
    np.testing.assert_array_equal(a.window_sums(np.arange(7.0), 3), [3.0, 12.0])
    # An alternating series is perfectly negatively autocorrelated at lag one.
    assert a.lag1(np.array([1.0, -1.0] * 50)) == pytest.approx(-1.0, abs=0.02)


def test_reversing_noise_inflates_short_horizons_only(monkeypatch: pytest.MonkeyPatch) -> None:
    """Oracle, by construction: baseline variance 1 a period, the policy's real P&L 0.5, plus a
    mark error e_t of variance 4 entering as e_t - e_{t-1}, as a wrong mark does. Over k periods
    the noise telescopes to two marks, so the true reduction is 1 - (0.5 k + 8) / k: -7.5 at one
    period and +0.1 at twenty. The difference's lag-one autocorrelation is -4 / 9.5."""
    a = _analysis(monkeypatch)
    rng = np.random.default_rng(7)
    n = 4000
    base = rng.normal(0.0, 1.0, n)
    true = np.sqrt(0.5) * rng.normal(0.0, 1.0, n)
    marks = rng.normal(0.0, 2.0, n + 1)
    policy = true + np.diff(marks)
    short = a.reduction_with_ci(a.window_sums(policy, 1), a.window_sums(base, 1), 20)
    long = a.reduction_with_ci(a.window_sums(policy, 20), a.window_sums(base, 20), 2)
    assert short["variance_reduction"] == pytest.approx(-7.5, abs=0.6)
    assert long["variance_reduction"] == pytest.approx(0.1, abs=0.2)
    assert short["ci_high"] < 0.0 < long["ci_high"]
    assert a.lag1(policy - base) == pytest.approx(-4 / 9.5, abs=0.04)
