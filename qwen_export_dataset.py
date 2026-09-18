"""
Fasst alle vom Benutzer geprüften Annotations-JSONs (<name>_annotation.json) in einem
Ordner (rekursiv) zu einer einzigen Qwen-Trainings-JSONL zusammen.

Nur *_annotation.json zählt als geprüft: diese Dateien schreibt ausschließlich
qwen_annotation_gui.py per "Annotations-JSON speichern", nachdem ein Mensch die
Vorannotation kontrolliert hat. *_preannotation.json (ungeprüfte Modellausgabe von
qwen_preannotate.py) wird bewusst ignoriert.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import qwen_annotation_gui as gui


def find_annotation_files(folder: Path) -> list[Path]:
    return sorted(folder.rglob("*_annotation.json"))


def build_dataset(folder: Path, dataset_root: Path) -> tuple[list[dict[str, Any]], list[str]]:
    records: dict[str, dict[str, Any]] = {}
    warnings: list[str] = []
    for path in find_annotation_files(folder):
        try:
            annotation = json.loads(path.read_text(encoding="utf-8"))
            image_path = annotation.get("image", {}).get("file")
            if not image_path:
                raise ValueError("Annotation enthält keinen Bildpfad (image.file).")
            record = gui.training_record(annotation, image_path, str(dataset_root))
        except Exception as error:
            warnings.append(f"{path}: {error}")
            continue
        key = gui.record_image(record) or str(path)
        records[key] = record
    return list(records.values()), warnings


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fasst alle geprüften *_annotation.json-Dateien eines Ordners "
        "(rekursiv) zu einer einzigen Qwen-Trainings-JSONL zusammen."
    )
    parser.add_argument("folder", help="Ordner, der rekursiv nach *_annotation.json durchsucht wird")
    parser.add_argument(
        "--dataset-root",
        help="Wurzelverzeichnis für relative Bildpfade in der JSONL; Standard: <folder>",
    )
    parser.add_argument("--output", help="Ausgabedatei; Standard: <folder>/train.jsonl")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    folder = Path(args.folder).resolve()
    if not folder.is_dir():
        raise SystemExit(f"Kein gültiger Ordner: {folder}")
    dataset_root = Path(args.dataset_root).resolve() if args.dataset_root else folder
    output = Path(args.output).resolve() if args.output else folder / "train.jsonl"

    records, warnings = build_dataset(folder, dataset_root)
    for warning in warnings:
        print(f"Übersprungen: {warning}", file=sys.stderr)
    if not records:
        raise SystemExit(f"Keine verwertbaren *_annotation.json-Dateien in {folder} gefunden.")

    output.write_text(
        "".join(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n" for record in records),
        encoding="utf-8",
        newline="\n",
    )
    print(f"Geschrieben: {output} ({len(records)} Datensätze, {len(warnings)} übersprungen)")


if __name__ == "__main__":
    main()
