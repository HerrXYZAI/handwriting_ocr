from __future__ import annotations

import argparse
import json
import sys
from itertools import combinations
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps


DEFAULT_OVERLAP_IOU = 0.10
DEFAULT_MIN_AREA = 4


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError("Die JSON-Wurzel muss ein Objekt sein.")
    return value


def open_scan(path: Path) -> Image.Image:
    with Image.open(path) as source:
        return ImageOps.exif_transpose(source).convert("RGB")


def resolve_image(annotation: dict[str, Any], json_path: Path, override: str | None) -> Path:
    if override:
        return Path(override).expanduser().resolve()

    image_data = annotation.get("image", {})
    stored_file = image_data.get("file") if isinstance(image_data, dict) else None
    if not stored_file:
        raise ValueError("Kein Bildpfad in annotation['image']['file'] vorhanden.")

    stored_path = Path(str(stored_file))
    if stored_path.is_file():
        return stored_path.resolve()

    relative_path = json_path.parent / stored_path
    if relative_path.is_file():
        return relative_path.resolve()

    raise FileNotFoundError(
        "Bild nicht gefunden. Verwende --image, wenn sich das Bild an einem "
        "anderen Ort befindet."
    )


def as_number(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label}: Boolescher Wert ist keine Koordinate.")
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label}: Keine numerische Koordinate: {value!r}") from exc


def parse_bbox(value: Any, label: str) -> list[float]:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise ValueError(f"{label}: Erwartet [x1, y1, x2, y2].")
    return [as_number(item, label) for item in value]


def pixel_box(value: Any, label: str) -> list[int]:
    return [round(item) for item in parse_bbox(value, label)]


def iou(a: list[int], b: list[int]) -> float:
    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])
    intersection = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    union = area_a + area_b - intersection
    return intersection / union if union else 0.0


def intersection_over_smaller(a: list[int], b: list[int]) -> float:
    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])
    intersection = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    smaller = min(area_a, area_b)
    return intersection / smaller if smaller else 0.0


def add_issue(issues: list[str], line_label: str, message: str) -> None:
    issues.append(f"{line_label}: {message}")


def validate_annotation(
    annotation: dict[str, Any],
    json_path: Path,
    image_path: Path,
    overlap_iou: float,
    min_area: int,
    check_derived_bbox: bool,
    tolerance: int,
) -> tuple[list[str], list[str], tuple[int, int]]:
    errors: list[str] = []
    warnings: list[str] = []

    try:
        image = open_scan(image_path)
    except Exception as exc:
        return [f"Bild konnte nicht geöffnet werden: {exc}"], warnings, (0, 0)

    actual_width, actual_height = image.size
    image_data = annotation.get("image", {})
    if not isinstance(image_data, dict):
        add_issue(errors, "image", "muss ein Objekt sein.")
        image_data = {}

    stored_width = image_data.get("width")
    stored_height = image_data.get("height")
    if stored_width is None or stored_height is None:
        add_issue(errors, "image", "width/height fehlen.")
    else:
        try:
            if int(stored_width) != actual_width or int(stored_height) != actual_height:
                add_issue(
                    errors,
                    "image",
                    f"JSON-Größe {stored_width}x{stored_height} != Bildgröße {actual_width}x{actual_height}.",
                )
        except (TypeError, ValueError):
            add_issue(errors, "image", "width/height sind nicht ganzzahlig.")

    lines = annotation.get("lines")
    if not isinstance(lines, list):
        return errors + ['lines: muss eine Liste sein.'], warnings, (actual_width, actual_height)

    boxes: list[tuple[int, list[int], dict[str, Any]]] = []
    for index, line in enumerate(lines, 1):
        label = f"line_{index:04d}"
        if not isinstance(line, dict):
            add_issue(errors, label, "muss ein Objekt sein.")
            continue
        line_id = str(line.get("id", label))
        label = line_id

        raw_pixels = line.get("bbox_pixels")
        if raw_pixels is None:
            add_issue(errors, label, "bbox_pixels fehlt.")
            pixel = None
        else:
            try:
                values = pixel_box(raw_pixels, f"{label}.bbox_pixels")
                x1, y1, x2, y2 = values
                if x1 < 0 or y1 < 0 or x2 > actual_width or y2 > actual_height:
                    add_issue(errors, label, f"außerhalb des Bildes: {values} bei {actual_width}x{actual_height}.")
                if x2 <= x1 or y2 <= y1:
                    add_issue(errors, label, f"leere oder invertierte Box: {values}.")
                area = max(0, x2 - x1) * max(0, y2 - y1)
                if area < min_area:
                    add_issue(errors, label, f"Fläche {area} ist kleiner als {min_area} Pixel.")
                pixel = values
            except ValueError as exc:
                add_issue(errors, label, str(exc))
                pixel = None

        raw_norm = line.get("bbox_1000")
        if raw_norm is None:
            add_issue(errors, label, "bbox_1000 fehlt.")
        else:
            try:
                norm = parse_bbox(raw_norm, f"{label}.bbox_1000")
                if any(value < 0 or value > 1000 for value in norm):
                    add_issue(errors, label, f"bbox_1000 außerhalb [0,1000]: {norm}.")
                if norm[2] <= norm[0] or norm[3] <= norm[1]:
                    add_issue(errors, label, f"bbox_1000 leer oder invertiert: {norm}.")
            except ValueError as exc:
                add_issue(errors, label, str(exc))
                norm = None

        if pixel is not None and check_derived_bbox and raw_norm is not None:
            expected = [
                round(pixel[0] * 1000 / actual_width),
                round(pixel[1] * 1000 / actual_height),
                round(pixel[2] * 1000 / actual_width),
                round(pixel[3] * 1000 / actual_height),
            ]
            try:
                actual_norm = [round(item) for item in parse_bbox(raw_norm, f"{label}.bbox_1000")]
                differences = [abs(a - b) for a, b in zip(actual_norm, expected)]
                if max(differences, default=0) > tolerance:
                    add_issue(errors, label, f"bbox_1000 passt nicht zu bbox_pixels: gespeichert {actual_norm}, erwartet {expected}.")
            except ValueError:
                pass

        if pixel is not None:
            boxes.append((index, pixel, line))

    for (index_a, box_a, line_a), (index_b, box_b, line_b) in combinations(boxes, 2):
        overlap = iou(box_a, box_b)
        if overlap >= overlap_iou:
            id_a = line_a.get("id", f"line_{index_a:04d}")
            id_b = line_b.get("id", f"line_{index_b:04d}")
            warnings.append(
                f"Überlappung {id_a}/{id_b}: IoU={overlap:.3f}, "
                f"kleinere-Fläche={intersection_over_smaller(box_a, box_b):.3f}."
            )

    return errors, warnings, (actual_width, actual_height)


def main() -> int:
    parser = argparse.ArgumentParser(description="Validiert OCR-Annotations-JSON gegen das zugehörige Bild.")
    parser.add_argument("annotation", type=Path, help="Annotations-JSON")
    parser.add_argument("--image", help="Bildpfad, falls der in JSON gespeicherte Pfad nicht stimmt")
    parser.add_argument("--overlap-iou", type=float, default=DEFAULT_OVERLAP_IOU, help="Warnschwelle für IoU-Überlappung; Standard: 0.10")
    parser.add_argument("--min-area", type=int, default=DEFAULT_MIN_AREA, help="Minimale Boxfläche in Pixeln; Standard: 4")
    parser.add_argument("--no-derived-check", action="store_true", help="bbox_1000 nicht gegen bbox_pixels prüfen")
    parser.add_argument("--tolerance", type=int, default=1, help="Erlaubte bbox_1000-Abweichung in Einheiten; Standard: 1")
    parser.add_argument("--strict-overlap", action="store_true", help="Überlappungswarnungen als Fehler behandeln")
    args = parser.parse_args()

    try:
        annotation = load_json(args.annotation)
        image_path = resolve_image(annotation, args.annotation.resolve(), args.image)
        errors, warnings, size = validate_annotation(
            annotation,
            args.annotation.resolve(),
            image_path,
            args.overlap_iou,
            args.min_area,
            not args.no_derived_check,
            args.tolerance,
        )
    except Exception as exc:
        print(f"FEHLER: {exc}", file=sys.stderr)
        return 2

    overlap_as_errors = args.strict_overlap and warnings
    error_count = len(errors) + len(warnings) if overlap_as_errors else len(errors)
    warning_count = 0 if overlap_as_errors else len(warnings)

    print(f"Bild: {image_path}")
    print(f"Bildgröße: {size[0]} x {size[1]} Pixel")
    print(f"Fehler: {error_count} | Warnungen: {warning_count}")
    for item in errors:
        print(f"ERROR  {item}")
    for item in warnings:
        prefix = "ERROR  " if args.strict_overlap else "WARN   "
        print(f"{prefix}{item}")

    return 1 if errors or overlap_as_errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
