"""
Konvertiert die von qwen_export_dataset.py erzeugte train.jsonl (Bild als
eigener Content-Block innerhalb "messages") in das von ms-swift erwartete
Custom-Dataset-Format: Nachrichten als reiner String mit "<image>"-Platzhalter
plus eine separate "images"-Liste mit Dateipfaden.

Läuft mit reinem Python (keine Zusatzpakete nötig), z.B. auf dem Host oder im
Trainings-Container:
    python convert_dataset.py train.jsonl train_swift.jsonl --mount-point /data
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path, PurePosixPath
from typing import Any

# Bildpfade wurden von einem Windows-Python-Prozess geschrieben (Path.as_posix()),
# koennen also "C:/..."-Laufwerksbuchstaben enthalten. PurePosixPath.is_absolute()
# erkennt das NICHT als absolut (POSIX kennt keine Laufwerksbuchstaben), daher
# zusaetzlich explizit darauf pruefen.
_WINDOWS_DRIVE_RE = re.compile(r"^[A-Za-z]:[/\\]")


def is_absolute_path(raw_path: str) -> bool:
    normalized = raw_path.replace("\\", "/")
    return normalized.startswith("/") or bool(_WINDOWS_DRIVE_RE.match(normalized))


def resolve_image_path(raw_path: str, mount_point: str) -> tuple[str, bool]:
    """Gibt (Pfad-fuer-ms-swift, war_absolut) zurueck.

    Relative Pfade (der Normalfall, siehe relative_image_path() in
    qwen_annotation_gui.py) werden unter mount_point verortet, da dort das
    Dataset-Wurzelverzeichnis im Container gemountet wird. Absolute Pfade
    (Ausnahmefall: Bild lag ausserhalb der Dataset-Wurzel, z.B. ein temporaerer
    Gradio-Upload-Pfad) werden unveraendert uebernommen und muessen separat
    erreichbar gemacht werden.
    """
    if is_absolute_path(raw_path):
        return raw_path.replace("\\", "/"), True
    normalized = PurePosixPath(raw_path.replace("\\", "/"))
    return f"{mount_point.rstrip('/')}/{normalized.as_posix()}", False


def convert_record(record: dict[str, Any], mount_point: str) -> tuple[dict[str, Any], list[str]]:
    warnings: list[str] = []
    messages = record["messages"]
    user_content = messages[0]["content"]
    assistant_content = messages[1]["content"]

    images: list[str] = []
    text_parts: list[str] = []
    for block in user_content:
        if block.get("type") == "image":
            resolved, was_absolute = resolve_image_path(str(block["image"]), mount_point)
            if was_absolute:
                warnings.append(f"absoluter Bildpfad unveraendert uebernommen: {resolved}")
            images.append(resolved)
            text_parts.append("<image>")
        elif block.get("type") == "text":
            text_parts.append(str(block["text"]))
    user_text = "\n".join(text_parts)

    assistant_text = "".join(
        str(block["text"]) for block in assistant_content if block.get("type") == "text"
    )

    converted = {
        "messages": [
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": assistant_text},
        ],
        "images": images,
    }
    return converted, warnings


def convert_file(input_path: Path, output_path: Path, mount_point: str) -> tuple[int, int]:
    written = 0
    skipped = 0
    with input_path.open(encoding="utf-8") as source, output_path.open(
        "w", encoding="utf-8", newline="\n"
    ) as target:
        for line_number, raw_line in enumerate(source, 1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                converted, warnings = convert_record(record, mount_point)
            except Exception as error:
                print(f"Uebersprungen (Zeile {line_number}): {error}")
                skipped += 1
                continue
            for warning in warnings:
                print(f"Warnung (Zeile {line_number}): {warning}")
            target.write(json.dumps(converted, ensure_ascii=False) + "\n")
            written += 1
    return written, skipped


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="train.jsonl von qwen_export_dataset.py")
    parser.add_argument("output", type=Path, help="Ausgabedatei im ms-swift-Format")
    parser.add_argument(
        "--mount-point",
        default="/data",
        help="Pfad, unter dem das Dataset-Wurzelverzeichnis im Trainings-Container "
        "gemountet wird; wird relativen Bildpfaden vorangestellt (Standard: /data)",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not args.input.is_file():
        raise SystemExit(f"Eingabedatei nicht gefunden: {args.input}")
    written, skipped = convert_file(args.input, args.output, args.mount_point)
    if written == 0:
        raise SystemExit(f"Keine verwertbaren Datensaetze in {args.input} gefunden.")
    print(f"Geschrieben: {args.output} ({written} Datensaetze, {skipped} uebersprungen)")


if __name__ == "__main__":
    main()
