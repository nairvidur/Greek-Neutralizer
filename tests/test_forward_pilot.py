"""The chain runner's support for the forward pilot: a start date, short runs, and input digests."""

from __future__ import annotations

import importlib
import os
import shutil
import subprocess
from datetime import date
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

from greeks_neutralizer.backtest.engine import run_backtest
from greeks_neutralizer.backtest.policies import DeltaDaily, Unhedged
from greeks_neutralizer.backtest.proxy import ShortStrangleBook
from tests.test_backtest import rising_market


def _runner(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    return importlib.import_module("run_chain_backtest")


def _files(*days: str) -> list[Path]:
    return [Path(f"{day}.csv") for day in days]


@pytest.mark.parametrize(
    ("days", "start", "kept", "dropped"),
    [
        # Days before the start are not part of the run, and the history before it was continuous.
        (
            ("2026-10-28", "2026-10-30", "2026-11-11", "2026-11-13"),
            "2026-11-11",
            ("2026-11-11", "2026-11-13"),
            (),
        ),
        # A gap after the start still restarts the history, as it does without a start.
        (
            ("2026-11-11", "2026-12-09", "2026-12-11"),
            "2026-11-11",
            ("2026-12-09", "2026-12-11"),
            ("2026-11-11",),
        ),
        # A gap before the start must not cost days after it: the 2019 day and the long gap after
        # it are outside the run, so nothing from the start on is dropped.
        (
            ("2019-12-30", "2026-11-11", "2026-11-13"),
            "2026-11-11",
            ("2026-11-11", "2026-11-13"),
            (),
        ),
        (
            ("2019-11-29", "2020-01-06", "2020-01-08"),
            None,
            ("2020-01-06", "2020-01-08"),
            ("2019-11-29",),
        ),
    ],
    ids=["before-start", "gap-after-start", "gap-before-start", "no-start"],
)
def test_history_starts_at_the_start_then_applies_the_gap_rule(
    monkeypatch, days, start, kept, dropped
):
    runner = _runner(monkeypatch)
    files, gone = runner.select_history(_files(*days), date.fromisoformat(start) if start else None)
    assert [p.stem for p in files] == list(kept)
    assert [p.stem for p in gone] == list(dropped)


def test_inputs_digest_changes_with_any_byte_or_day_and_not_with_the_folder(monkeypatch, tmp_path):
    runner = _runner(monkeypatch)

    def write(folder: str, name: str, text: str) -> Path:
        path = tmp_path / folder / name
        path.parent.mkdir(exist_ok=True)
        path.write_text(text)
        return path

    a = [write("a", "2026-11-11.csv", "bid,ask\n1.00,1.10\n"), write("a", "rates.csv", "4.1\n")]
    b = [write("b", "2026-11-11.csv", "bid,ask\n1.00,1.10\n"), write("b", "rates.csv", "4.1\n")]
    rates = "4.1\n"
    revised = [write("c", "2026-11-11.csv", "bid,ask\n1.00,1.11\n"), write("c", "rates.csv", rates)]
    redated = [write("d", "2026-11-12.csv", "bid,ask\n1.00,1.10\n"), write("d", "rates.csv", rates)]
    digest = runner.inputs_digest(a)
    assert runner.inputs_digest(list(reversed(b))) == digest
    assert runner.inputs_digest(revised) != digest
    assert runner.inputs_digest(redated) != digest


def _results(periods: int):
    # One more day than periods: the first day opens the book.
    market = rising_market(periods + 1)
    return {
        "unhedged": run_backtest(market, ShortStrangleBook(), Unhedged()),
        "delta_daily": run_backtest(market, ShortStrangleBook(), DeltaDaily("SPX")),
    }


def test_a_run_too_short_for_the_bootstrap_reports_no_intervals_and_says_why(monkeypatch):
    runner = _runner(monkeypatch)
    comparisons, note = runner.comparisons_for(_results(12), block=20)
    assert comparisons == []
    assert note is not None and "12 periods, fewer than the 40" in note


def test_a_run_long_enough_compares_each_policy_with_each_baseline(monkeypatch):
    runner = _runner(monkeypatch)
    comparisons, note = runner.comparisons_for(_results(40), block=20)
    assert note is None
    assert [(c["policy"], c["baseline"]) for c in comparisons] == [
        ("delta_daily", "unhedged"),
        ("unhedged", "delta_daily"),
    ]


def test_stats_on_a_short_run_leave_out_only_the_tail_measure():
    """A 97.5% expected shortfall needs 40 samples, one in its 2.5% tail. Below that the interim
    readout still reports every other statistic, and the ES is absent rather than a crash."""
    from greeks_neutralizer.backtest.report import policy_stats
    from tests.test_backtest import _fake

    short = policy_stats(_fake("x", np.arange(21.0) - 10.0))
    assert short.es_975 is None
    assert short.days == 21
    assert short.worst_day == -10.0
    # 40 samples: the tail is the single worst, a loss of 20.
    assert policy_stats(_fake("x", np.arange(40.0) - 20.0)).es_975 == 20.0


def _isolated_git(monkeypatch: pytest.MonkeyPatch, ceiling: Path) -> None:
    for name in ("GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE", "GIT_COMMON_DIR"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", os.devnull)
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(ceiling))


def _git(repo: Path, *argv: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *argv], capture_output=True, text=True, check=True
    ).stdout.strip()


@pytest.mark.skipif(shutil.which("git") is None, reason="needs git")
def test_code_identity_names_the_checked_out_commit_and_tracks_code_edits(monkeypatch, tmp_path):
    runner = _runner(monkeypatch)
    _isolated_git(monkeypatch, tmp_path)
    repo = tmp_path / "repo"
    (repo / "scripts").mkdir(parents=True)
    (repo / "scripts" / "run.py").write_text("x = 1\n")
    (repo / "notes.md").write_text("notes\n")
    _git(repo, "init", "-q")
    _git(repo, "add", ".")
    _git(repo, "-c", "user.name=t", "-c", "user.email=t@example.com", "commit", "-qm", "c")

    assert runner.code_identity(repo / "scripts") == (_git(repo, "rev-parse", "HEAD"), False)
    # Neither untracked files nor edits outside src/ and scripts/ change the code that ran.
    (repo / "scripts" / "scratch.py").write_text("y = 2\n")
    (repo / "notes.md").write_text("edited\n")
    assert runner.code_identity(repo / "scripts")[1] is False
    (repo / "scripts" / "run.py").write_text("x = 2\n")
    assert runner.code_identity(repo / "scripts") == (_git(repo, "rev-parse", "HEAD"), True)


def test_code_identity_outside_a_git_checkout_is_unknown(monkeypatch, tmp_path):
    runner = _runner(monkeypatch)
    _isolated_git(monkeypatch, tmp_path)
    folder = tmp_path / "plain"
    folder.mkdir()
    assert runner.code_identity(folder) == (None, None)
