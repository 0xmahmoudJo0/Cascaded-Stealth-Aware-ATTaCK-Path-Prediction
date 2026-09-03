"""Lightweight console logging helpers for CLI commands."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping, Tuple, Union

PathLike = Union[str, Path]


def log_info(message: str) -> None:
    print(f"[INFO] {message}")


def log_success(message: str) -> None:
    print(f"[OK]   {message}")


def log_warning(message: str) -> None:
    print(f"[WARN] {message}")


def log_error(message: str) -> None:
    print(f"[ERR]  {message}")


def _format_kv(pairs: Iterable[Tuple[str, Union[str, Path]]]) -> str:
    rows = [f"    - {key}: {value}" for key, value in pairs]
    return '\n'.join(rows)


def log_inputs(items: Mapping[str, PathLike]) -> None:
    if not items:
        return
    log_info("Inputs:")
    print(_format_kv((k, Path(v)) for k, v in items.items()))


def log_outputs(items: Mapping[str, PathLike]) -> None:
    if not items:
        return
    log_info("Outputs:")
    print(_format_kv((k, Path(v)) for k, v in items.items()))



