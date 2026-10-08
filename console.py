"""Konsolenausgabe mit Zeitstempel.

Gleiches Format wie die Log-Zeilen von qwen_preannotate.py
("2026-10-08 22:58:01 | ..."), damit Konsole und Logdateien zusammenpassen.
"""

from __future__ import annotations

import datetime as dt
import sys
from typing import Any, Callable, NoReturn, TextIO

TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S"


def timestamp() -> str:
    return dt.datetime.now().strftime(TIMESTAMP_FORMAT)


def tprint(*values: Any, sep: str = " ", file: TextIO | None = None, flush: bool = True) -> None:
    """Wie print(), aber jede Zeile beginnt mit Datum und Uhrzeit. Ein Aufruf
    ohne Argumente gibt wie print() eine Leerzeile aus (ohne Zeitstempel)."""
    stream = file if file is not None else sys.stdout
    text = sep.join(str(value) for value in values)
    if not text:
        print(file=stream, flush=flush)
        return
    stamp = timestamp()
    print("\n".join(f"{stamp} | {line}" for line in text.split("\n")), file=stream, flush=flush)


def run_main(main: Callable[[], int | None]) -> NoReturn:
    """Startet main() und gibt Abbruchmeldungen aus raise SystemExit("...")
    ebenfalls mit Zeitstempel (auf stderr) aus statt roh."""
    try:
        code = main()
    except SystemExit as exc:
        if isinstance(exc.code, str):
            tprint(exc.code, file=sys.stderr)
            raise SystemExit(1) from None
        raise
    raise SystemExit(code or 0)
