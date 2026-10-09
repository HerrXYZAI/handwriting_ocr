"""Speicherung mehrerer Modellergebnisse je Seite unter "model_runs".

Wird an zwei Stellen verwendet:

- ``<bild>_preannotation.json`` (qwen_preannotate.py und GUI-Vorannotation):
  je Modell eine Vorannotation, damit sich in der GUI auswählen lässt, welche
  angezeigt wird. Die obersten ``lines`` bleiben der zuletzt gelaufene Stand
  (abwärtskompatibel zu allem, was nur ``lines`` liest).
- ``<bild>_annotation.json`` (model_compare.py): Vergleichsläufe gegen die
  geprüften Zeilen.

Ein Lauf hat die Form::

    {"model": "...", "created": "2026-10-09T09:12:00", "duration_s": 95.2,
     "image_size": [w, h], "settings": {...},
     "lines": [{"bbox_pixels": [x1, y1, x2, y2], "text": "...",
                "confidence": "high", "angle": 0.0}, ...]}
"""

from __future__ import annotations

import datetime as dt
import json
import os
from pathlib import Path
from typing import Any

RUNS_KEY = "model_runs"


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json_atomic(path: Path, data: dict[str, Any]) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def get_runs(data: dict[str, Any]) -> dict[str, dict[str, Any]]:
    runs = data.get(RUNS_KEY)
    return runs if isinstance(runs, dict) else {}


def now_iso() -> str:
    return dt.datetime.now().isoformat(timespec="seconds")


def make_run(
    model: str,
    lines: list[dict[str, Any]],
    image_size: tuple[int, int] | list[int],
    settings: dict[str, Any] | None = None,
    duration_s: float | None = None,
) -> dict[str, Any]:
    """Baut einen Lauf aus Zeilen mit bbox_pixels und text (bzw.
    text_predicted aus der GUI)."""
    return {
        "model": model,
        "created": now_iso(),
        "duration_s": round(float(duration_s), 1) if duration_s is not None else None,
        "image_size": [int(image_size[0]), int(image_size[1])],
        "settings": settings or {},
        "lines": [
            {
                "bbox_pixels": [int(v) for v in line["bbox_pixels"]],
                "text": str(line.get("text", line.get("text_predicted", ""))),
                "confidence": line.get("confidence", "low"),
                "angle": float(line.get("angle", 0) or 0),
            }
            for line in lines
            if isinstance(line, dict) and line.get("bbox_pixels")
        ],
    }


def preannotation_runs(data: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Alle Vorannotationen einer _preannotation.json je Modell. Ältere
    Dateien ohne model_runs liefern ihren einzigen Stand unter dem Modellnamen
    aus "processing" (oder "unbekanntes Modell")."""
    runs = dict(get_runs(data))
    if runs:
        return runs
    lines = data.get("lines")
    if isinstance(lines, list) and lines:
        processing = data.get("processing") if isinstance(data.get("processing"), dict) else {}
        label = str(processing.get("model") or data.get("model") or "unbekanntes Modell")
        image = data.get("image") if isinstance(data.get("image"), dict) else {}
        size = [int(image.get("width") or 0), int(image.get("height") or 0)]
        runs[label] = make_run(label, lines, size, settings=processing)
        runs[label]["created"] = None
    return runs


def add_preannotation_run(
    path: Path, label: str, run: dict[str, Any], base_document: dict[str, Any]
) -> dict[str, Any]:
    """Schreibt base_document (oberste Ebene = dieser Lauf) nach path und
    behält dabei die Läufe anderer Modelle aus einer vorhandenen Datei."""
    runs: dict[str, dict[str, Any]] = {}
    if path.is_file():
        try:
            runs = preannotation_runs(load_json(path))
        except (OSError, json.JSONDecodeError):
            runs = {}
    runs[label] = run
    document = dict(base_document)
    document[RUNS_KEY] = runs
    write_json_atomic(path, document)
    return document


def has_preannotation_run(path: Path, label: str) -> bool:
    if not path.is_file():
        return False
    try:
        return label in preannotation_runs(load_json(path))
    except (OSError, json.JSONDecodeError):
        return False
