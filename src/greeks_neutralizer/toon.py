"""Minimal TOON encoder for CLI output: nested mappings, scalar lists, and uniform tables."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from typing import Any

_BARE = re.compile(r"^[A-Za-z_][A-Za-z0-9_./:+-]*$")
_NEEDS_QUOTE = {"true", "false", "null"}


def scalar(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            return "null"
        return f"{value:.10g}"
    text = str(value)
    if _BARE.match(text) and text.lower() not in _NEEDS_QUOTE:
        return text
    escaped = text.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
    return f'"{escaped}"'


def _is_scalar(value: Any) -> bool:
    return value is None or isinstance(value, (bool, int, float, str))


def _is_table(items: Sequence[Any]) -> bool:
    if not items or not all(isinstance(i, Mapping) for i in items):
        return False
    keys = list(items[0])
    return all(list(i) == keys and all(_is_scalar(v) for v in i.values()) for i in items)


def encode(value: Any, indent: int = 0) -> str:
    pad = "  " * indent
    lines: list[str] = []
    if isinstance(value, Mapping):
        for key, item in value.items():
            name = scalar(str(key))
            if _is_scalar(item):
                lines.append(f"{pad}{name}: {scalar(item)}")
            elif isinstance(item, Mapping):
                if item:
                    lines.append(f"{pad}{name}:")
                    lines.append(encode(item, indent + 1))
                else:
                    lines.append(f"{pad}{name}: {{}}")
            elif isinstance(item, Sequence):
                lines.append(_encode_list(name, item, indent))
            else:
                lines.append(f"{pad}{name}: {scalar(str(item))}")
        return "\n".join(lines)
    return f"{pad}{scalar(value)}"


def _encode_list(name: str, items: Sequence[Any], indent: int) -> str:
    pad = "  " * indent
    if not items:
        return f"{pad}{name}[0]:"
    if all(_is_scalar(i) for i in items):
        return f"{pad}{name}[{len(items)}]: " + ",".join(scalar(i) for i in items)
    if _is_table(items):
        keys = list(items[0])
        header = ",".join(scalar(str(k)) for k in keys)
        rows = ["  " * (indent + 1) + ",".join(scalar(item[k]) for k in keys) for item in items]
        return f"{pad}{name}[{len(items)}]{{{header}}}:\n" + "\n".join(rows)
    parts = [f"{pad}{name}[{len(items)}]:"]
    for item in items:
        parts.append("  " * (indent + 1) + "-")
        parts.append(encode(item, indent + 2))
    return "\n".join(parts)
