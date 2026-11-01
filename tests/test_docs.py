"""Documentation that can run must run."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_readme_library_example_runs(capsys):
    readme = (ROOT / "README.md").read_text()
    blocks = re.findall(r"```python\n(.*?)```", readme, flags=re.S)
    assert blocks, "the README should have a Python example"
    exec(compile(blocks[0], "README.md", "exec"), {})
    assert "Buy" in capsys.readouterr().out


def test_paths_named_in_the_readme_exist():
    readme = (ROOT / "README.md").read_text()
    for name in re.findall(r"`(docs/[\w./-]+)`", readme):
        assert (ROOT / name).exists(), name
