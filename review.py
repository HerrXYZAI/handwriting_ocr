"""Prüfstatus je Zeile (Box) einer Annotation.

Jede Zeile trägt ``review``:

- ``accepted``  - Box und Text sind geprüft und gehen ins Training.
- ``rejected``  - nicht akzeptiert (falsch, unleserlich, unsicher); geht nicht
  ins Training, ihr Bereich wird im Trainingsbild abgedeckt.
- ``open``      - noch nicht geprüft; wird wie ``rejected`` behandelt.

Ältere Annotationen kennen ``review`` noch nicht: Dort hat die Oberfläche beim
Speichern jede Zeile automatisch auf ``status`` = confirmed/corrected gesetzt,
d.h. die ganze Datei galt als geprüft. Solche Zeilen gelten weiter als
akzeptiert, damit bestehende Exporte unverändert bleiben.
"""

from __future__ import annotations

from typing import Any

ACCEPTED = "accepted"
REJECTED = "rejected"
OPEN = "open"
STATES = (ACCEPTED, REJECTED, OPEN)
NO_TEXT_STATUS = "no_text"

# Anzeige in der Tabellenspalte "Prüfung"
LABELS = {ACCEPTED: "✓ ok", REJECTED: "✗ nein", OPEN: "offen"}

_ACCEPT_WORDS = {"✓", "ok", "ja", "j", "y", "yes", "accepted", "akzeptiert", "+", "1", "true"}
_REJECT_WORDS = {"✗", "x", "nein", "n", "no", "rejected", "abgelehnt", "nicht", "-", "0", "false"}


def review_state(line: dict[str, Any]) -> str:
    value = line.get("review")
    if value in STATES:
        return value
    if line.get("status") in ("confirmed", "corrected", NO_TEXT_STATUS):
        return ACCEPTED
    return OPEN


def parse_label(value: Any, fallback: str = OPEN) -> str:
    """Liest die Tabellenspalte tolerant (auch von Hand getippt: ok/ja/x/nein)."""
    text = str(value or "").strip().lower()
    if not text:
        return fallback
    if text in STATES:
        return text
    first = text.split()[0]
    if first in _ACCEPT_WORDS or text.startswith("✓"):
        return ACCEPTED
    if first in _REJECT_WORDS or text.startswith("✗"):
        return REJECTED
    if text.startswith("offen") or text.startswith("open"):
        return OPEN
    return fallback


def has_text(line: dict[str, Any]) -> bool:
    return bool(str(line.get("text_corrected", line.get("text", "")) or "").strip())


def is_training_line(line: dict[str, Any]) -> bool:
    return line.get("status") != NO_TEXT_STATUS and has_text(line) and review_state(line) == ACCEPTED


def is_excluded_region(line: dict[str, Any]) -> bool:
    """Zeile mit Box, die nicht ins Training geht und deshalb im Trainingsbild
    abgedeckt wird (nicht akzeptiert oder offen)."""
    return line.get("status") != NO_TEXT_STATUS and review_state(line) != ACCEPTED


def counts(lines: list[dict[str, Any]]) -> dict[str, int]:
    result = {ACCEPTED: 0, REJECTED: 0, OPEN: 0}
    for line in lines:
        if isinstance(line, dict):
            result[review_state(line)] += 1
    return result


def summary(lines: list[dict[str, Any]]) -> str:
    c = counts(lines)
    return f"Geprüft: {c[ACCEPTED]} ✓, {c[REJECTED]} ✗, {c[OPEN]} offen"
