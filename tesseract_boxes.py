from __future__ import annotations

import base64
import io
import os
from pathlib import Path
from typing import Any

import requests
from PIL import Image

TESSERACT_API = os.environ.get("TESSERACT_API", "http://127.0.0.1:8884/ocr")
DEFAULT_LANG = "deu"
# PSM 11 ("Sparse text: find as much text as possible, no particular order")
# assumes the least about layout, which suits scattered handwritten lines
# better than the column/paragraph-oriented modes.
DEFAULT_PSM = 11


def clamp(value: int, low: int, high: int) -> int:
    return max(low, min(high, value))


def pixels_to_bbox_1000(pixel_box: list[int], width: int, height: int) -> list[int]:
    x1, y1, x2, y2 = pixel_box
    return [
        clamp(round(x1 * 1000 / width), 0, 1000),
        clamp(round(y1 * 1000 / height), 0, 1000),
        clamp(round(x2 * 1000 / width), 0, 1000),
        clamp(round(y2 * 1000 / height), 0, 1000),
    ]


def _encode_image_b64(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def detect_lines(
    image_path: str | Path,
    lang: str = DEFAULT_LANG,
    psm: int = DEFAULT_PSM,
    timeout: int = 60,
) -> list[dict[str, Any]]:
    """Ruft den Tesseract-Docker-Dienst auf und liefert Zeilen-Boxen in
    Originalpixeln des übergebenen Bildes.

    Der von Tesseract erkannte Text wird bewusst nicht übernommen: Tesseracts
    Handschrifterkennung ist unzuverlässig, seine Layout-/Zeilensegmentierung
    liefert aber oft brauchbare Boxgeometrie, die Qwens Boxschätzung ergänzen
    oder verfeinern kann.
    """
    with Image.open(image_path) as source:
        image = source.convert("RGB")
    payload = {"image": _encode_image_b64(image), "lang": lang, "psm": psm}
    try:
        response = requests.post(TESSERACT_API, json=payload, timeout=timeout)
    except requests.ConnectionError as exc:
        raise RuntimeError(
            f"Tesseract-Dienst unter {TESSERACT_API} nicht erreichbar. Container starten: "
            "docker compose -f docker/tesseract-ocr/docker-compose.yml up -d --build"
        ) from exc
    if not response.ok:
        raise RuntimeError(f"Tesseract-Dienst-Fehler {response.status_code}: {response.text}")
    data = response.json()
    if "error" in data:
        raise RuntimeError(f"Tesseract-Dienst-Fehler: {data['error']}")
    return data.get("lines", [])


def iou(box_a: tuple[int, int, int, int], box_b: tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    intersection = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if intersection <= 0:
        return 0.0
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def adjust_lines_with_tesseract(
    lines: list[dict[str, Any]],
    tesseract_lines: list[dict[str, Any]],
    width: int,
    height: int,
    iou_threshold: float = 0.15,
) -> tuple[list[dict[str, Any]], int, int]:
    """Snap + Fill: jede vorhandene Zeile wird auf die am besten überlappende
    Tesseract-Box "eingerastet" (bbox_pixels/bbox_1000 ersetzt, Text und
    Konfidenz bleiben unverändert); Tesseract-Boxen ohne ausreichende
    Überlappung zu einer vorhandenen Zeile werden als neue, unbestätigte
    Zeilen ergänzt (Text leer, damit sie bei der Prüfung auffallen).

    Gibt (neue Zeilenliste, Anzahl angepasster Zeilen, Anzahl ergänzter
    Zeilen) zurück.
    """
    result = [dict(line) for line in lines]
    used_tesseract: set[int] = set()
    snapped = 0

    for line in result:
        try:
            box = tuple(line["bbox_pixels"])
        except (KeyError, TypeError, ValueError):
            continue
        best_index, best_iou = -1, 0.0
        for index, t_line in enumerate(tesseract_lines):
            if index in used_tesseract:
                continue
            score = iou(box, tuple(t_line["bbox_pixels"]))
            if score > best_iou:
                best_index, best_iou = index, score
        if best_index >= 0 and best_iou >= iou_threshold:
            new_box = list(tesseract_lines[best_index]["bbox_pixels"])
            line["bbox_pixels"] = new_box
            line["bbox_1000"] = pixels_to_bbox_1000(new_box, width, height)
            used_tesseract.add(best_index)
            snapped += 1

    added = 0
    next_number = len(result) + 1
    for index, t_line in enumerate(tesseract_lines):
        if index in used_tesseract:
            continue
        new_box = list(t_line["bbox_pixels"])
        result.append({
            "id": f"line_{next_number:04d}",
            "bbox_pixels": new_box,
            "bbox_1000": pixels_to_bbox_1000(new_box, width, height),
            # Both key conventions are set because this module is called from two
            # schemas: qwen_preannotate.py's CLI lines use plain "text", while the
            # GUI's normalize_annotation()/ensure_pixel_boxes() use text_predicted/
            # text_corrected (and derive one from the other only if "text" is set).
            "text": "",
            "text_predicted": "",
            "text_corrected": "",
            "confidence": "low",
            "status": "unreviewed",
            "source": "tesseract",
        })
        next_number += 1
        added += 1

    return result, snapped, added


def detect_and_adjust(
    image_path: str | Path,
    lines: list[dict[str, Any]],
    width: int,
    height: int,
    lang: str = DEFAULT_LANG,
    psm: int = DEFAULT_PSM,
) -> tuple[list[dict[str, Any]], str]:
    """Ruft Tesseract ab und wendet Snap+Fill auf die übergebenen Zeilen an.
    Einstiegspunkt für GUI-Button und Batch-CLI gleichermaßen.
    """
    tesseract_lines = detect_lines(image_path, lang=lang, psm=psm)
    if not tesseract_lines:
        return lines, "Tesseract hat keine Textregionen erkannt; Zeilen unverändert."
    new_lines, snapped, added = adjust_lines_with_tesseract(lines, tesseract_lines, width, height)
    return new_lines, f"Tesseract: {len(tesseract_lines)} Box(en) erkannt, {snapped} Zeile(n) angepasst, {added} neue Zeile(n) ergänzt."
