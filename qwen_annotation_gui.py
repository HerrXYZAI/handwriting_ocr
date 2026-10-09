from __future__ import annotations

import argparse
import base64
import copy
import datetime
import functools
import hashlib
import inspect
import io
import json
import math
import mimetypes
import os
import re
import shutil
import tempfile
import time
import urllib.parse
import uuid
from pathlib import Path, PurePosixPath
from typing import Any

import gradio as gr
import requests
from PIL import Image, ImageDraw, ImageOps, ImageStat

import model_compare
import model_runs
import qwen_preannotate
import review
import pdf_utils
import tesseract_boxes
import tiling

OLLAMA_HOST = "http://127.0.0.1:11434"
OLLAMA_API = f"{OLLAMA_HOST}/api/chat"
OLLAMA_TAGS_API = f"{OLLAMA_HOST}/api/tags"
# 'llamacpp' spricht statt Ollama den OpenAI-kompatiblen llama.cpp-Server an
# (siehe finetune/README.md, Fallback falls Ollama mit einem selbst
# konvertierten Qwen3-VL-GGUF+mmproj-Paar abstürzt). Per Umgebungsvariable
# vor dem GUI-Start umschalten, z.B. "set OCR_BACKEND=llamacpp".
BACKEND = os.environ.get("OCR_BACKEND", "ollama").strip().lower()
LLAMACPP_API = os.environ.get("LLAMACPP_API", "http://127.0.0.1:8080/v1/chat/completions")
DEFAULT_MODEL = "qwen3-vl:4b"
# Gesamt-Timeout je Vorannotation in Sekunden (ohne Streaming); großzügig für
# große Modelle mit CPU-Auslagerung. Per Umgebungsvariable anpassbar.
OLLAMA_TIMEOUT = int(os.environ.get("QWEN_TIMEOUT", "7200"))
DEFAULT_CONTEXT_SIZE = 4096
DEFAULT_DATASET_ROOT = r"C:\test\handwriting_ocr\pictures_for_OCR"
CONFIDENCE_VALUES = {"high", "medium", "low"}
TABLE_HEADERS = ["ID", "Text", "Konfidenz", "x1_px", "y1_px", "x2_px", "y2_px", "Winkel (°)", "Prüfung"]
# Bewusst EIN Typ für alle Spalten statt einer Liste je Spalte: Gradio 6.20
# berechnet bei einer Liste die Spaltentypen beim Aufbau der Tabelle für
# JEDE Zelle neu (inkl. Durchlaufen des kompletten Tabelleninhalts). Bei 30
# Zeilen blockierte das den Browser nach jeder Seitenwahl und jedem
# Akzeptieren für ca. 9 s, bei 100 Zeilen deutlich länger. Zahlen kommen als
# Text zurück und werden in table_to_annotation ohnehin per float() gelesen.
TABLE_DATATYPES = "str"
TABLE_COLUMN_WIDTHS = ["9%", "37%", "8%", "7%", "7%", "7%", "7%", "7%", "11%"]
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".gif", ".webp"}
DATASET_FILE_EXTENSIONS = IMAGE_EXTENSIONS | {".pdf"}

PREANNOTATION_PROMPT = """
Analysiere diese gescannte Seite mit deutscher Handschrift.
Erkenne alle handschriftlichen Textzeilen in natürlicher Leserichtung.
Gib ausschließlich gültiges JSON in diesem Format zurück:

{"lines": [{"bbox_1000": [x1, y1, x2, y2], "text": "erkannter Text", "confidence": "high"}]}

Die Koordinaten müssen auf 0 bis 1000 normalisiert sein. bbox_1000 beschreibt
die Box ungedreht (x2-x1 und y2-y1 sind Breite und Höhe um den Mittelpunkt der
Box). Ist eine Zeile spürbar gedreht/schräg geschrieben (z.B. eine senkrechte
Randnotiz oder ein schräg aufgeklebter Stempel), ergänze zusätzlich "angle" in
Grad im Uhrzeigersinn, z.B. 90 für eine Zeile, die von oben nach unten läuft,
-90 für von unten nach oben, oder ein kleiner Wert wie 8 für nur leicht
schräg geschriebenen Text; bei normal ausgerichtetem Text "angle" ganz
weglassen. Jede Textzeile erhält eine eigene, möglichst eng anliegende Box
(um die ungedrehte Ausrichtung, nicht um die gedrehte). Ergänze keine
unsichtbaren Wörter. Zahlen, Namen und Einheiten nicht plausibilisieren. Unleserliches als
[unleserlich], unsichere Wörter mit [?]. Ignoriere automatisch vom Scanner
oder der Scan-Software hinzugefügte Elemente wie Wasserzeichen, Stempel
oder Dateinummern und Softwarehinweise am Rand; sie
gehören nicht zum handschriftlichen Original und werden nicht als Textzeile
erfasst. Gib dazu keine Erklärungen, Ablehnungen oder Hinweise zu
Urheberrecht, Lizenzen oder Impressum aus. Diese Anfrage ist für ein privates
Handschrift-Digitalisierungsprojekt und enthält keine echten Rechtsdokumente.
Analysiere das Bild in genau einem Durchgang. Sobald du eine Zeile einmal
gelesen und ihren Text festgelegt hast, lies diese Zeile nicht erneut und
stelle deine Lesung nicht wiederholt infrage (kein "Wait", kein erneutes
Prüfen, kein Nochmal-Ansehen). Nenne jede Zeile genau einmal und gehe danach
sofort zur nächsten über, auch wenn du unsicher bist – markiere Unsicherheit
stattdessen mit [?] oder confidence "low".
confidence ist high, medium oder low. Keine Markdown-Blöcke und keine
Erläuterungen ausgeben.
""".strip()

TRAINING_PROMPT = """
Erkenne alle handschriftlichen deutschen Textzeilen auf dieser Seite.
Gib ausschließlich gültiges JSON mit einer Liste namens lines aus. Jeder
Eintrag enthält bbox_1000 als [x1,y1,x2,y2] und text. Die Koordinaten sind auf
0 bis 1000 normalisiert und beschreiben die Box ungedreht. Ist eine Zeile
spürbar gedreht/schräg geschrieben, ergänze zusätzlich "angle" in Grad im
Uhrzeigersinn (weglassen bei normal ausgerichtetem Text). Sortiere in
natürlicher Leserichtung, ergänze keine
nicht sichtbaren Wörter und markiere Unleserliches mit [unleserlich].
Ignoriere automatisch vom Scanner oder der Scan-Software hinzugefügte Elemente
wie Wasserzeichen, Stempel oder Dateinummern und
Softwarehinweise am Rand; sie gehören nicht zum handschriftlichen Original und
werden nicht als Textzeile erfasst.
""".strip()


def open_scan(image_path: str | Path) -> Image.Image:
    """Öffnet einen Scan und wendet eine vorhandene EXIF-Ausrichtung an."""
    with Image.open(image_path) as source:
        return ImageOps.exif_transpose(source).convert("RGB")


def scan_size(image_path: str | Path) -> tuple[int, int]:
    """Bildgröße wie open_scan(...).size (inkl. EXIF-Drehung). Kommt aus dem
    Zwischenspeicher des Vorschaubilds (_cached_display), sodass jeder Scan
    nur einmal eingelesen wird - die Vorschau braucht ihn ohnehin. (Eine
    reine Kopf-Abfrage reicht bei PNG nicht: Pillow liest für die EXIF-Daten
    dort die ganze Datei.)"""
    return _cached_display(*_display_cache_key(image_path, 0))[0].size


def display_image_file(image_path: str | None) -> str | None:
    """Verkleinertes Vorschaubild als Datei für das Feld "Originalscan" -
    statt des kompletten Scans (oft > 10 MB), den der Browser sonst bei jeder
    Seitenauswahl laden und dekodieren müsste. Verarbeitet wird immer der
    Original-Pfad aus image_state, nie dieses Anzeigebild."""
    if not image_path:
        return image_path
    try:
        file_path = _cached_display(*_display_cache_key(image_path, 0))[3]
    except Exception:
        return image_path
    return file_path or image_path


def open_scan_for_display(image_path: str | Path, annotation: dict[str, Any]) -> Image.Image:
    """Wie open_scan(), dreht das Bild aber zusätzlich um eine noch nicht auf
    die Datei angewendete Drehung (siehe rotate_page/annotation["image"]
    ["pending_rotation"]), damit Vorschau und Zeilen-Crop zu den bereits
    gedrehten Box-Koordinaten passen. Die Datei selbst wird erst beim
    Speichern der Annotation tatsächlich gedreht.
    """
    rotation = int(annotation.get("image", {}).get("pending_rotation", 0)) % 360
    return _cached_display(*_display_cache_key(image_path, rotation))[0]


# Vorschaubilder werden einmal als Datei abgelegt und per URL eingebunden statt
# als ~0,5 MB großer data-URI in jedes HTML-Update eingebettet. So muss der
# Browser bei einem Klick auf eine Box/Tabellenzeile nur noch wenige KB neues
# HTML verarbeiten; das Bild kommt aus seinem Zwischenspeicher.
PREVIEW_DIR = Path(tempfile.gettempdir()) / "handschrift_ocr_preview"
PREVIEW_URL_PREFIX = "/gradio_api/file="
try:
    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
    gr.set_static_paths(paths=[str(PREVIEW_DIR)])
    _PREVIEW_FILES_OK = True
except Exception:  # z.B. ältere Gradio-Version - dann wie bisher data-URI
    _PREVIEW_FILES_OK = False


def _prune_preview_dir(max_age_days: float = 7) -> None:
    cutoff = time.time() - max_age_days * 86400
    try:
        for old in PREVIEW_DIR.glob("*.jpg"):
            if old.stat().st_mtime < cutoff:
                old.unlink(missing_ok=True)
    except OSError:
        pass


def display_data_uri(image_path: str | Path, annotation: dict[str, Any]) -> str:
    """Quelle (src) des verkleinerten Vorschaubilds: URL der zwischengespeicherten
    Datei, ersatzweise data-URI (siehe _cached_display)."""
    rotation = int(annotation.get("image", {}).get("pending_rotation", 0)) % 360
    _, data_uri, url, _ = _cached_display(*_display_cache_key(image_path, rotation))
    return url or data_uri


def _display_cache_key(image_path: str | Path, rotation: int) -> tuple[str, int, int, int]:
    path = Path(image_path).resolve()
    stat = path.stat()
    return str(path), stat.st_mtime_ns, stat.st_size, rotation


@functools.lru_cache(maxsize=6)
def _cached_display(path: str, mtime_ns: int, size: int, rotation: int) -> tuple[Image.Image, str, str | None, str | None]:
    """Liest einen Scan nur einmal ein und hält ihn samt fertig kodiertem
    Vorschaubild vor. Vorher wurde bei jedem Klick auf eine Box die komplette
    Scan-Datei zweimal neu dekodiert, verkleinert und als JPEG kodiert, was die
    Oberfläche spürbar einfrieren ließ. Änderungs-Zeitstempel und Größe im
    Schlüssel sorgen dafür, dass ein physisch gedrehtes/neu gespeichertes Bild
    neu eingelesen wird. Aufrufer dürfen das Bild nicht verändern (crop/rotate
    liefern ohnehin neue Objekte)."""
    image = open_scan(path)
    if rotation:
        image = image.rotate(-rotation, expand=True)
    data_uri = encode_display_image(image)
    url = None
    file_path = None
    if _PREVIEW_FILES_OK:
        key = hashlib.sha1(f"{path}|{mtime_ns}|{size}|{rotation}".encode("utf-8")).hexdigest()[:20]
        target = PREVIEW_DIR / f"{key}.jpg"
        try:
            if not target.is_file():
                target.write_bytes(base64.b64decode(data_uri.split(",", 1)[1]))
            url = PREVIEW_URL_PREFIX + urllib.parse.quote(target.resolve().as_posix(), safe="/:")
            file_path = str(target)
        except OSError:
            url = None
    return image, data_uri, url, file_path


def encode_image(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def extract_json(raw: str) -> dict[str, Any]:
    text = raw.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.I)
    text = re.sub(r"\s*```$", "", text)
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("Qwen hat kein JSON-Objekt zurückgegeben.")
    value = json.loads(text[start : end + 1])
    if not isinstance(value, dict):
        raise ValueError("Die JSON-Wurzel ist kein Objekt.")
    return value


def clamp(value: int, low: int, high: int) -> int:
    return max(low, min(high, value))


def validate_bbox(value: Any) -> list[int]:
    # Boxen als Text (z.B. ":[94,112,617,151],") wie in qwen_preannotate reparieren.
    value = qwen_preannotate.coerce_bbox(value)
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise ValueError(f"Ungültige Box: {value!r}")
    coords = [clamp(round(float(v)), 0, 1000) for v in value]
    x1, y1, x2, y2 = coords
    if x2 <= x1:
        x2 = min(1000, x1 + 1)
    if y2 <= y1:
        y2 = min(1000, y1 + 1)
    return [x1, y1, x2, y2]


def validate_pixel_bbox(value: Any, width: int, height: int) -> list[int]:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise ValueError(f"Ungültige Pixel-Box: {value!r}")
    x1 = clamp(round(float(value[0])), 0, width)
    y1 = clamp(round(float(value[1])), 0, height)
    x2 = clamp(round(float(value[2])), 0, width)
    y2 = clamp(round(float(value[3])), 0, height)
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"Leere Pixel-Box: {value!r}")
    return [x1, y1, x2, y2]


def bbox_1000_to_pixels(bbox: list[int], width: int, height: int) -> list[int]:
    x1, y1, x2, y2 = validate_bbox(bbox)
    return [
        clamp(round(x1 * width / 1000), 0, width),
        clamp(round(y1 * height / 1000), 0, height),
        clamp(round(x2 * width / 1000), 0, width),
        clamp(round(y2 * height / 1000), 0, height),
    ]


def pixels_to_bbox_1000(pixel_box: list[int], width: int, height: int) -> list[int]:
    x1, y1, x2, y2 = validate_pixel_bbox(pixel_box, width, height)
    return [
        clamp(round(x1 * 1000 / width), 0, 1000),
        clamp(round(y1 * 1000 / height), 0, 1000),
        clamp(round(x2 * 1000 / width), 0, 1000),
        clamp(round(y2 * 1000 / height), 0, 1000),
    ]


def normalize_annotation(data: dict[str, Any]) -> dict[str, Any]:
    raw_lines = data.get("lines")
    if not isinstance(raw_lines, list):
        raise ValueError('Qwen-Ausgabe enthält keine Liste "lines".')
    result = []
    for item in raw_lines:
        if not isinstance(item, dict):
            continue
        try:
            bbox = validate_bbox(item.get("bbox_1000", item.get("bbox")))
        except (TypeError, ValueError):
            continue
        text = item.get("text_corrected")
        if text is None:
            text = item.get("text_predicted")
        if text is None:
            text = item.get("text", "")
        text = str(text).strip()
        confidence = str(item.get("confidence", "low")).lower().strip()
        if confidence not in CONFIDENCE_VALUES:
            confidence = "low"
        result.append({
            "id": "",
            "bbox_1000": bbox,
            "text_predicted": text,
            "text_corrected": text,
            "confidence": confidence,
            "status": "unreviewed",
            "review": review.OPEN,
            "angle": normalize_angle(item.get("angle", 0)),
        })
    result.sort(key=lambda x: (x["bbox_1000"][1], x["bbox_1000"][0]))
    for number, item in enumerate(result, 1):
        item["id"] = f"line_{number:04d}"
    return {
        "schema_version": "1.3",
        "coordinate_system": "original_pixels",
        "lines": result,
    }


def ensure_pixel_boxes(annotation: dict[str, Any], image_path: str) -> dict[str, Any]:
    """Macht bbox_pixels zur führenden, verlustfreien Koordinatenquelle."""
    result = copy.deepcopy(annotation)
    width, height = scan_size(image_path)

    stored_image = result.get("image")
    if not isinstance(stored_image, dict):
        stored_image = {}

    result["image"] = {
        **stored_image,
        "file": str(Path(image_path).resolve()),
        "file_name": Path(image_path).name,
        "width": width,
        "height": height,
    }

    for index, line in enumerate(result.get("lines", []), 1):
        if not isinstance(line, dict):
            continue
        line.setdefault("id", f"line_{index:04d}")
        if "text" in line:
            line.setdefault("text_predicted", line["text"])
            line.setdefault("text_corrected", line["text"])
        line.setdefault("text_predicted", line.get("text_corrected", ""))
        line.setdefault("text_corrected", line.get("text_predicted", ""))
        confidence = str(line.get("confidence", "low")).lower().strip()
        line["confidence"] = confidence if confidence in CONFIDENCE_VALUES else "low"
        line.setdefault("status", "unreviewed")
        # Prüfstatus je Box explizit festhalten (Altbestand ohne "review":
        # automatisch bestätigte Zeilen gelten als akzeptiert, siehe review.py).
        line["review"] = review.review_state(line)
        # "rotation" war der Feldname der alten 90°-Schritt-Variante dieses
        # Features; bereits gespeicherte Werte werden beim Laden übernommen.
        if "angle" not in line and "rotation" in line:
            line["angle"] = line.pop("rotation")
        line["angle"] = normalize_angle(line.get("angle", 0))

        try:
            pixel_box = validate_pixel_bbox(line.get("bbox_pixels"), width, height)
        except (TypeError, ValueError):
            pixel_box = bbox_1000_to_pixels(line["bbox_1000"], width, height)

        line["bbox_pixels"] = pixel_box
        line["bbox_1000"] = pixels_to_bbox_1000(pixel_box, width, height)

    result["schema_version"] = "1.3"
    result["coordinate_system"] = "original_pixels"
    return result


def list_ollama_models() -> list[str]:
    """Fragt die Ollama-Modellliste ab (siehe finetune/README.md Abschnitt 5
    für per to_ollama.ps1 importierte, eigene Finetunes). Bei nicht
    erreichbarem Ollama wird still eine leere Liste geliefert - das
    Dropdown fällt dann auf DEFAULT_MODEL als freien Text zurück, statt
    den GUI-Start mit einem Fehler zu blockieren.
    """
    if BACKEND == "llamacpp":
        # Der llama.cpp-Server hat immer genau ein Modell geladen (per -m
        # beim Containerstart); eine Modellliste wie bei Ollama gibt es
        # nicht, das "model"-Feld im Request wird ohnehin ignoriert.
        return []
    try:
        response = requests.get(OLLAMA_TAGS_API, timeout=5)
        response.raise_for_status()
        data = response.json()
    except Exception:
        return []
    # Doppelte Namen und interne "llamacpp:<Prüfsumme>"-Einträge ausblenden.
    names = sorted({
        m.get("name") for m in data.get("models", [])
        if m.get("name") and not re.match(r"^[^:/]+:[0-9a-f]{40,}$", m.get("name"))
    })
    return names


def model_dropdown_choices() -> list[str]:
    models = list_ollama_models()
    if DEFAULT_MODEL not in models:
        models = [DEFAULT_MODEL, *models]
    return models


def refresh_ollama_models():
    return gr.update(choices=model_dropdown_choices())


def run_qwen(image_path: str, model: str, context_size: int) -> dict[str, Any]:
    path = Path(image_path)
    if not path.is_file():
        raise FileNotFoundError(path)
    if BACKEND == "llamacpp":
        raw = _run_qwen_llamacpp(path)
    else:
        raw = _run_qwen_ollama(path, model, context_size)
    return normalize_annotation(extract_json(raw))


def _run_qwen_ollama(path: Path, model: str, context_size: int) -> str:
    payload = {
        "model": model.strip(),
        "messages": [{
            "role": "user",
            "content": PREANNOTATION_PROMPT,
            "images": [encode_image(path)],
        }],
        "stream": False,
        # Festes Schema statt nur "json": verhindert Boxen als Text (siehe
        # qwen_preannotate.LINES_SCHEMA).
        "format": qwen_preannotate.LINES_SCHEMA,
        # Denkmodus aus: spart bei großen (teilweise auf die CPU ausgelagerten)
        # Modellen viel Zeit; für reine Instruct-Modelle ohne Wirkung.
        "think": False,
        "options": {"temperature": 0, "num_ctx": int(context_size)},
    }
    # Großzügiges Timeout: Ohne Streaming muss die gesamte Antwort innerhalb
    # dieser Zeit fertig sein - bei großen Modellen mit CPU-Auslagerung
    # (z.B. qwen3-vl:30b-a3b-instruct auf 12 GB VRAM) kann das dauern.
    response = requests.post(OLLAMA_API, json=payload, timeout=OLLAMA_TIMEOUT)
    if not response.ok:
        # Ältere Ollama-Versionen ohne Schema-Unterstützung.
        payload["format"] = "json"
        response = requests.post(OLLAMA_API, json=payload, timeout=OLLAMA_TIMEOUT)
    if not response.ok and "think" in response.text.lower():
        # Ältere Ollama-Version oder Modell ohne Thinking-Unterstützung.
        payload.pop("think")
        response = requests.post(OLLAMA_API, json=payload, timeout=OLLAMA_TIMEOUT)
    if not response.ok:
        raise RuntimeError(f"Ollama-Fehler {response.status_code}: {response.text}")
    raw = response.json().get("message", {}).get("content", "")
    if not raw:
        raise RuntimeError("Ollama hat keine Textantwort geliefert.")
    return raw


def _run_qwen_llamacpp(path: Path) -> str:
    """Spricht die OpenAI-kompatible /v1/chat/completions-API von llama.cpp's
    eigenem Server an (Fallback, falls Ollama mit diesem selbst konvertierten
    Qwen3-VL-GGUF+mmproj-Paar abstürzt - siehe finetune/README.md)."""
    payload = {
        "model": "llamacpp",
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": PREANNOTATION_PROMPT},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{mimetypes.guess_type(path.name)[0] or 'image/jpeg'};base64,{encode_image(path)}"
                    },
                },
            ],
        }],
        "stream": False,
        "temperature": 0,
        # llama.cpp's eigener Default ist 1.0 (keine Bestrafung), anders als
        # Ollamas Modelfile-Default; ohne das kann greedy Decoding (temperature
        # 0) sich in kurzen Wiederholungsschleifen festfahren.
        "repeat_penalty": 1.1,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    response = requests.post(LLAMACPP_API, json=payload, timeout=OLLAMA_TIMEOUT)
    if not response.ok:
        raise RuntimeError(f"llama.cpp-Fehler {response.status_code}: {response.text}")
    choices = response.json().get("choices") or []
    raw = choices[0].get("message", {}).get("content", "") if choices else ""
    if not raw:
        raise RuntimeError("llama.cpp hat keine Textantwort geliefert.")
    return raw


def add_metadata(annotation: dict[str, Any], image_path: str, model: str) -> dict[str, Any]:
    result = ensure_pixel_boxes(annotation, image_path)
    result.update({
        "model": model.strip(),
        "task": "handwritten_line_transcription",
    })
    return result


def line_to_pixels(
    line: dict[str, Any],
    annotation: dict[str, Any],
    actual_width: int,
    actual_height: int,
) -> tuple[int, int, int, int]:
    """Liest Originalpixel direkt; keine Reskalierung von gespeicherten Boxen."""
    try:
        return tuple(validate_pixel_bbox(line.get("bbox_pixels"), actual_width, actual_height))
    except (TypeError, ValueError):
        return tuple(bbox_1000_to_pixels(line["bbox_1000"], actual_width, actual_height))


CONFIDENCE_COLORS = {"high": "#24A148", "medium": "#F1C21B", "low": "#DA1E28"}
SELECTED_COLOR = "#0067C0"
REVIEW_COLORS = {review.ACCEPTED: "#24A148", review.REJECTED: "#DA1E28", review.OPEN: "#8D8D8D"}
CONFIDENCE_NAMES = {"high": "hoch", "medium": "mittel", "low": "niedrig"}
NO_TEXT_STATUS = "no_text"
NO_TEXT_COLOR = "#8D8D8D"
EMPTY_PREVIEW_HTML = "<div class='bbox-empty'>Kein Bild geladen.</div>"

BBOX_STYLE = """
<style>
.bbox-canvas { position: relative; display: inline-block; max-width: 100%; line-height: 0; user-select: none; }
.bbox-image { display: block; width: 100%; height: auto; max-width: 100%; pointer-events: none; }
.bbox-box { position: absolute; border-style: solid; border-width: 2px; box-sizing: border-box; cursor: move; transform-origin: 50% 50%; }
.bbox-label { position: absolute; top: -22px; left: -2px; color: white; font-size: 12px; font-weight: bold; padding: 1px 5px; border-radius: 3px; white-space: nowrap; }
.bbox-handle { position: absolute; width: 10px; height: 10px; background: white; border: 2px solid #0067C0; border-radius: 50%; }
.bbox-handle-nw { top: -6px; left: -6px; cursor: nwse-resize; }
.bbox-handle-ne { top: -6px; right: -6px; cursor: nesw-resize; }
.bbox-handle-sw { bottom: -6px; left: -6px; cursor: nesw-resize; }
.bbox-handle-se { bottom: -6px; right: -6px; cursor: nwse-resize; }
.bbox-rotate-handle { position: absolute; top: -34px; left: calc(50% - 6px); width: 12px; height: 12px; background: #0067C0; border: 2px solid white; border-radius: 50%; cursor: grab; box-shadow: 0 1px 3px rgba(0,0,0,.4); }
.bbox-rotate-handle::after { content: ''; position: absolute; top: 12px; left: 5px; width: 2px; height: 10px; background: #0067C0; }
.bbox-nudge { position: absolute; top: calc(50% - 9px); width: 18px; height: 18px; line-height: 16px; text-align: center; color: white; background: #0067C0; border: 2px solid white; border-radius: 50%; cursor: pointer; font-size: 13px; font-weight: bold; box-shadow: 0 1px 3px rgba(0,0,0,.4); user-select: none; }
.bbox-nudge-minus { left: -26px; }
.bbox-nudge-plus { right: -26px; }
.bbox-apply-angle { position: absolute; bottom: -20px; left: calc(50% - 15px); width: 30px; height: 16px; line-height: 12px; text-align: center; color: white; background: #24A148; border: 2px solid white; border-radius: 8px; cursor: pointer; font-size: 11px; font-weight: bold; box-shadow: 0 1px 3px rgba(0,0,0,.4); user-select: none; }
.bbox-text { position: absolute; z-index: 20; min-width: 220px; max-width: 420px; }
.bbox-tip { display: none; position: absolute; left: 0; top: calc(100% + 22px); z-index: 40; background: rgba(25,25,25,.94); color: #fff; font: 13px/1.45 sans-serif; padding: 5px 8px; border-radius: 5px; width: max-content; max-width: 460px; white-space: normal; box-shadow: 0 2px 8px rgba(0,0,0,.35); pointer-events: none; }
.bbox-box:hover { z-index: 30; }
.bbox-review-accepted { background: rgba(36,161,72,.13); }
.bbox-review-rejected { background: rgba(218,30,40,.13); border-style: dashed !important; }
body.qb-hide-accepted .bbox-review-accepted:not(.bbox-selected) { display: none; }
body.qb-hide-controls .bbox-review, body.qb-hide-controls .bbox-handle, body.qb-hide-controls .bbox-rotate-handle,
body.qb-hide-controls .bbox-nudge, body.qb-hide-controls .bbox-apply-angle, body.qb-hide-controls .bbox-text,
body.qb-hide-controls .bbox-label { display: none; }
body.qb-hide-controls .bbox-box { cursor: pointer; }
.bbox-review { position: absolute; top: calc(50% - 10px); left: calc(100% + 30px); display: flex; gap: 3px; }
.bbox-review-left { left: auto; right: calc(100% + 30px); }
.bbox-review-below { top: calc(100% + 4px); left: auto; right: 0; }
.bbox-review-btn { width: 22px; height: 20px; line-height: 18px; text-align: center; font-size: 13px; font-weight: bold; border-radius: 4px; cursor: pointer; user-select: none; background: white; box-shadow: 0 1px 3px rgba(0,0,0,.35); }
.bbox-review-ok { color: #24A148; border: 1px solid #24A148; }
.bbox-review-no { color: #DA1E28; border: 1px solid #DA1E28; }
.bbox-review-ok.on { background: #24A148; color: white; }
.bbox-review-no.on { background: #DA1E28; color: white; }
.bbox-box:hover .bbox-tip { display: block; }
.bbox-textarea { width: 100%; min-height: 60px; font-size: 14px; padding: 6px; border: 2px solid #0067C0; border-radius: 4px; box-shadow: 0 2px 8px rgba(0,0,0,.25); resize: vertical; font-family: inherit; box-sizing: border-box; }
.bbox-empty { padding: 40px; text-align: center; color: #888; }
#bbox-sync-box, #file-sync-box-all, #file-sync-box-flagged { position: absolute !important; width: 1px !important; height: 1px !important; overflow: hidden !important; opacity: 0 !important; pointer-events: none !important; margin: 0 !important; padding: 0 !important; border: 0 !important; }
.file-list { max-height: 260px; overflow-y: auto; border: 1px solid #ddd; border-radius: 6px; }
.file-list-empty { padding: 16px; text-align: center; color: #888; }
.file-row { display: flex; align-items: center; gap: 8px; padding: 6px 10px; cursor: pointer; border-bottom: 1px solid #eee; font-size: 13px; }
.file-row:last-child { border-bottom: none; }
.file-row:hover { background: rgba(0, 103, 192, 0.1); }
.file-row.annotated { background: rgba(36, 161, 72, 0.18); }
.file-row.annotated:hover { background: rgba(36, 161, 72, 0.3); }
.file-row.active { outline: 2px solid #0067C0; outline-offset: -2px; }
.file-icon { flex-shrink: 0; }
.file-label { flex: 1 1 auto; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.file-badge { flex-shrink: 0; font-weight: bold; color: #24A148; }
.file-badge-pending { color: #F1C21B; }
.file-row.partial { background: rgba(241, 194, 27, 0.16); }
.file-row.partial:hover { background: rgba(241, 194, 27, 0.3); }
.file-badge-partial { color: #8a6d00; font-size: 12px; }
</style>
"""

BBOX_JS = """
() => {
  function qbClamp(v, lo, hi) { return Math.max(lo, Math.min(hi, v)); }

  function qbSyncTo(elemId, payload) {
    const box = document.querySelector('#' + elemId + ' textarea, #' + elemId + ' input');
    if (!box) return;
    box.value = JSON.stringify(payload);
    box.dispatchEvent(new Event('input', { bubbles: true }));
    box.dispatchEvent(new Event('change', { bubbles: true }));
  }

  function qbSync(payload) {
    qbSyncTo('bbox-sync-box', payload);
  }

  // Fokussiert die Textarea automatisch, sobald sie neu ins DOM eingefuegt
  // wird (z.B. nach Klick auf eine Box) - das HTML-Attribut "autofocus"
  // allein greift bei einem via innerHTML ausgetauschten gr.HTML-Element
  // nicht zuverlaessig, weil das kein echtes Neuladen der Seite ist.
  function qbAutofocusTextarea() {
    const wrap = document.getElementById('bbox-preview-wrap');
    if (!wrap) {
      setTimeout(qbAutofocusTextarea, 200);
      return;
    }
    const observer = new MutationObserver(() => {
      const textarea = wrap.querySelector('.bbox-textarea');
      if (textarea && document.activeElement !== textarea) {
        textarea.focus();
        const end = textarea.value.length;
        textarea.setSelectionRange(end, end);
      }
    });
    observer.observe(wrap, { childList: true, subtree: true });
  }
  qbAutofocusTextarea();

  window.qbSelectFile = function(el) {
    const path = el.getAttribute('data-path');
    const syncId = el.getAttribute('data-sync') || 'file-sync-box-all';
    if (!path) return;
    document.querySelectorAll('.file-row.active').forEach((row) => row.classList.remove('active'));
    document.querySelectorAll('.file-row').forEach((row) => {
      if (row.getAttribute('data-path') === path) row.classList.add('active');
    });
    qbSyncTo(syncId, { path: path });
  };

  function qbSyncBox(id) {
    const box = document.getElementById('box-' + id);
    const canvas = document.getElementById('bbox-canvas');
    if (!box || !canvas) return;
    const width = parseFloat(canvas.dataset.width);
    const height = parseFloat(canvas.dataset.height);
    const x1 = Math.round(box.offsetLeft / canvas.clientWidth * width);
    const y1 = Math.round(box.offsetTop / canvas.clientHeight * height);
    const x2 = Math.round((box.offsetLeft + box.offsetWidth) / canvas.clientWidth * width);
    const y2 = Math.round((box.offsetTop + box.offsetHeight) / canvas.clientHeight * height);
    qbSync({ type: 'move', id: id, bbox_pixels: [x1, y1, x2, y2] });
  }

  function qbCommitActiveText() {
    // mousedown handlers below call preventDefault() to allow dragging
    // without also selecting page text - but that also suppresses the
    // browser's default blur of a currently-focused textarea, so an
    // in-progress text edit would otherwise be silently discarded when
    // clicking straight from one box to another. Blur it explicitly first
    // so window.qbCommitText still fires and the edit is saved.
    const active = document.activeElement;
    if (active && active.classList && active.classList.contains('bbox-textarea')) {
      active.blur();
    }
  }

  window.qbStartDrag = function(evt, id) {
    if (evt.target.closest('.bbox-handle')) return;
    qbCommitActiveText();
    evt.preventDefault();
    evt.stopPropagation();
    const box = document.getElementById('box-' + id);
    const canvas = document.getElementById('bbox-canvas');
    if (!box || !canvas) return;
    const startX = evt.clientX, startY = evt.clientY;
    const startLeft = box.offsetLeft, startTop = box.offsetTop;
    let moved = false;
    // Übersichtsmodus ("Bedienelemente ausblenden"): Klick wählt nur aus,
    // Boxen lassen sich nicht versehentlich verschieben.
    const viewOnly = document.body.classList.contains('qb-hide-controls');
    function onMove(e) {
      if (viewOnly) return;
      const dx = e.clientX - startX, dy = e.clientY - startY;
      if (Math.abs(dx) > 3 || Math.abs(dy) > 3) moved = true;
      const newLeft = qbClamp(startLeft + dx, 0, canvas.clientWidth - box.offsetWidth);
      const newTop = qbClamp(startTop + dy, 0, canvas.clientHeight - box.offsetHeight);
      box.style.left = (newLeft / canvas.clientWidth * 100) + '%';
      box.style.top = (newTop / canvas.clientHeight * 100) + '%';
      const textPanel = document.getElementById('text-' + id);
      if (textPanel) {
        textPanel.style.left = box.style.left;
        textPanel.style.top = ((newTop + box.offsetHeight) / canvas.clientHeight * 100) + '%';
      }
    }
    function onUp() {
      document.removeEventListener('mousemove', onMove);
      document.removeEventListener('mouseup', onUp);
      if (moved) {
        qbSyncBox(id);
      } else {
        qbSync({ type: 'toggle', id: id });
      }
    }
    document.addEventListener('mousemove', onMove);
    document.addEventListener('mouseup', onUp);
  };

  window.qbStartResize = function(evt, id, corner) {
    qbCommitActiveText();
    evt.preventDefault();
    evt.stopPropagation();
    const box = document.getElementById('box-' + id);
    const canvas = document.getElementById('bbox-canvas');
    if (!box || !canvas) return;
    const startX = evt.clientX, startY = evt.clientY;
    const startLeft = box.offsetLeft, startTop = box.offsetTop;
    const startWidth = box.offsetWidth, startHeight = box.offsetHeight;
    const minSize = 12;
    function onMove(e) {
      const dx = e.clientX - startX, dy = e.clientY - startY;
      let left = startLeft, top = startTop, width = startWidth, height = startHeight;
      if (corner.includes('e')) width = qbClamp(startWidth + dx, minSize, canvas.clientWidth - startLeft);
      if (corner.includes('s')) height = qbClamp(startHeight + dy, minSize, canvas.clientHeight - startTop);
      if (corner.includes('w')) {
        width = qbClamp(startWidth - dx, minSize, startLeft + startWidth);
        left = startLeft + startWidth - width;
      }
      if (corner.includes('n')) {
        height = qbClamp(startHeight - dy, minSize, startTop + startHeight);
        top = startTop + startHeight - height;
      }
      box.style.left = (left / canvas.clientWidth * 100) + '%';
      box.style.top = (top / canvas.clientHeight * 100) + '%';
      box.style.width = (width / canvas.clientWidth * 100) + '%';
      box.style.height = (height / canvas.clientHeight * 100) + '%';
      const textPanel = document.getElementById('text-' + id);
      if (textPanel) {
        textPanel.style.left = box.style.left;
        textPanel.style.top = ((top + height) / canvas.clientHeight * 100) + '%';
      }
    }
    function onUp() {
      document.removeEventListener('mousemove', onMove);
      document.removeEventListener('mouseup', onUp);
      qbSyncBox(id);
    }
    document.addEventListener('mousemove', onMove);
    document.addEventListener('mouseup', onUp);
  };

  window.qbStartRotate = function(evt, id) {
    qbCommitActiveText();
    evt.preventDefault();
    evt.stopPropagation();
    const box = document.getElementById('box-' + id);
    if (!box) return;
    // Der Mittelpunkt der (ggf. schon gedrehten) Box bleibt beim Drehen um
    // sich selbst konstant, deshalb liefert getBoundingClientRect() hier in
    // jedem Aufruf zuverlaessig denselben Drehpunkt.
    function angleFromEvent(e) {
      const rect = box.getBoundingClientRect();
      const cx = rect.left + rect.width / 2;
      const cy = rect.top + rect.height / 2;
      const dx = e.clientX - cx, dy = e.clientY - cy;
      // +90, weil der Griff in Ruhestellung (Winkel 0) oberhalb der Box
      // sitzt (atan2 dafuer -90 liefert) und positive Winkel im
      // Uhrzeigersinn gezaehlt werden, wie window.qbSync({type:'rotate'}).
      return Math.atan2(dy, dx) * 180 / Math.PI + 90;
    }
    function onMove(e) {
      box.style.transform = 'rotate(' + angleFromEvent(e) + 'deg)';
    }
    function onUp(e) {
      document.removeEventListener('mousemove', onMove);
      document.removeEventListener('mouseup', onUp);
      qbSync({ type: 'rotate', id: id, angle: angleFromEvent(e) });
    }
    document.addEventListener('mousemove', onMove);
    document.addEventListener('mouseup', onUp);
  };

  window.qbReview = function(evt, id, value) {
    evt.preventDefault();
    evt.stopPropagation();
    qbCommitActiveText();
    qbSync({ type: 'review', id: id, value: value });
  };

  window.qbCommitText = function(id) {
    const el = document.getElementById('textarea-' + id);
    if (!el) return;
    // Nur bei echter Änderung senden: sonst löst schon das Weiterklicken zur
    // nächsten Box eine zusätzliche, unnötige Server-Runde (inkl. Neuaufbau
    // der Tabelle) aus, bevor die neue Box geöffnet wird.
    if (el.value === el.defaultValue) return;
    qbSync({ type: 'text', id: id, text: el.value });
  };

  window.qbNudgeAngle = function(evt, id, delta) {
    evt.preventDefault();
    evt.stopPropagation();
    qbSync({ type: 'nudge_angle', id: id, delta: delta });
  };

  window.qbApplyLastAngle = function(evt, id) {
    evt.preventDefault();
    evt.stopPropagation();
    qbSync({ type: 'apply_last_angle', id: id });
  };
}
"""


def escape_html(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def escape_attr(text: str) -> str:
    return escape_html(text).replace('"', "&quot;")


def find_dataset_files(dataset_root: str) -> list[dict[str, Any]]:
    """Findet rekursiv alle Bilder und PDFs unter dataset_root (inkl. Unterordner
    wie *_pages und *_tiles), zusammen mit ihrem Annotationsstatus.
    """
    if not dataset_root or not dataset_root.strip():
        return []
    root = Path(dataset_root)
    if not root.is_dir():
        return []
    entries = []
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in DATASET_FILE_EXTENSIONS:
            continue
        if MASKED_DIR in path.relative_to(root).parts:
            continue  # abgedeckte Trainingskopien (Export), keine eigenen Scans
        annotation_path = path.with_name(path.stem + "_annotation.json")
        review_counts = None
        if annotation_path.is_file():
            try:
                saved_lines = json.loads(annotation_path.read_text(encoding="utf-8")).get("lines", [])
                review_counts = review.counts(saved_lines if isinstance(saved_lines, list) else [])
            except (OSError, json.JSONDecodeError):
                review_counts = None
        entries.append({
            "path": str(path.resolve()),
            "relative": path.relative_to(root).as_posix(),
            "is_pdf": path.suffix.lower() == ".pdf",
            "annotated": annotation_path.is_file(),
            "preannotated": path.with_name(path.stem + "_preannotation.json").is_file(),
            "review": review_counts,
        })
    entries.sort(key=lambda item: item["relative"].lower())
    return entries


FILE_SYNC_ALL = "file-sync-box-all"
FILE_SYNC_FLAGGED = "file-sync-box-flagged"


def render_file_list(
    dataset_root: str,
    selected_path: str | None = None,
    only_flagged: bool = False,
    sync_target: str = FILE_SYNC_ALL,
) -> str:
    """Rendert die Dateiliste als klickbare HTML-Zeilen; bereits annotierte /
    freigegebene Dateien (mit vorhandener *_annotation.json) werden grün markiert.
    Mit only_flagged=True werden nur vor- oder fertig annotierte Dateien gezeigt.
    """
    entries = find_dataset_files(dataset_root)
    if only_flagged:
        entries = [entry for entry in entries if entry["annotated"] or entry["preannotated"]]
    if not entries:
        message = (
            "Keine vorannotierten oder freigegebenen Dateien gefunden."
            if only_flagged
            else "Keine Bilder oder PDFs unter diesem Pfad gefunden."
        )
        return f"<div class='file-list-empty'>{message}</div>"

    selected_resolved = str(Path(selected_path).resolve()) if selected_path else None
    rows = []
    for entry in entries:
        classes = ["file-row"]
        badge = ""
        counts = entry.get("review")
        open_count = counts[review.OPEN] if counts else 0
        if entry["annotated"] and open_count:
            # Gespeichert, aber noch nicht alle Boxen geprüft.
            classes.append("partial")
            done = counts[review.ACCEPTED] + counts[review.REJECTED]
            total = done + open_count
            badge = (
                f"<span class='file-badge file-badge-partial' title='Teilweise geprüft: "
                f"{counts[review.ACCEPTED]} akzeptiert, {counts[review.REJECTED]} nicht akzeptiert, "
                f"{open_count} offen'>{done}/{total}</span>"
            )
        elif entry["annotated"]:
            classes.append("annotated")
            title = "Alle Boxen geprüft"
            if counts:
                title += f": {counts[review.ACCEPTED]} akzeptiert, {counts[review.REJECTED]} nicht akzeptiert"
            badge = f"<span class='file-badge' title='{title}'>&#10003;</span>"
        elif entry["preannotated"]:
            classes.append("preannotated")
            badge = "<span class='file-badge file-badge-pending' title='Vorannotiert, noch nicht geprüft'>&#8226;</span>"
        if selected_resolved and entry["path"] == selected_resolved:
            classes.append("active")
        icon = "\U0001F4C4" if entry["is_pdf"] else "\U0001F5BC"
        rows.append(
            "<div class='{cls}' data-path=\"{path}\" data-sync=\"{sync}\" onclick=\"window.qbSelectFile(this)\">"
            "<span class='file-icon'>{icon}</span>"
            "<span class='file-label'>{label}</span>{badge}"
            "</div>".format(
                cls=" ".join(classes),
                path=escape_attr(entry["path"]),
                sync=escape_attr(sync_target),
                icon=icon,
                label=escape_html(entry["relative"]),
                badge=badge,
            )
        )
    return f"<div class='file-list'>{''.join(rows)}</div>"


def render_file_lists(dataset_root: str, selected_path: str | None = None) -> tuple[str, str]:
    return (
        render_file_list(dataset_root, selected_path, only_flagged=False, sync_target=FILE_SYNC_ALL),
        render_file_list(dataset_root, selected_path, only_flagged=True, sync_target=FILE_SYNC_FLAGGED),
    )


def encode_display_image(image: Image.Image, max_dim: int = 1400) -> str:
    display = image.copy()
    display.thumbnail((max_dim, max_dim), Image.LANCZOS)
    buffer = io.BytesIO()
    display.save(buffer, format="JPEG", quality=85)
    data = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{data}"


def render_interactive_preview(
    image_path: str | None,
    annotation: dict[str, Any],
    selected: int,
    active_text_id: str | None,
) -> str:
    """Rendert das Bild mit verschieb- und skalierbaren Boxen (Drag/Resize per
    Maus). Ein Klick auf eine Box blendet ein editierbares Textfeld darunter ein.
    """
    if not image_path:
        return EMPTY_PREVIEW_HTML
    image = open_scan_for_display(image_path, annotation)
    width, height = image.size
    data_uri = display_data_uri(image_path, annotation)

    parts = []
    for index, line in enumerate(annotation.get("lines", [])):
        try:
            x1, y1, x2, y2 = line_to_pixels(line, annotation, width, height)
        except (KeyError, ValueError):
            continue
        left = x1 / width * 100
        top = y1 / height * 100
        box_width = (x2 - x1) / width * 100
        box_height = (y2 - y1) / height * 100
        line_id = line.get("id") or f"line_{index + 1:04d}"
        # Farbe nach Prüfstatus (grau = offen/erkannt, grün = akzeptiert,
        # rot = nicht akzeptiert); die ausgewählte Box bleibt blau.
        if index == selected:
            color = SELECTED_COLOR
        elif line.get("status") == NO_TEXT_STATUS:
            color = NO_TEXT_COLOR
        else:
            color = REVIEW_COLORS[review.review_state(line)]
        text = str(line.get("text_corrected", ""))
        angle = normalize_angle(line.get("angle", 0))
        state = review.review_state(line)
        mark = {review.ACCEPTED: "✓ ", review.REJECTED: "✗ "}.get(state, "")
        label = f"{mark}{index + 1}" if not angle else f"{mark}{index + 1} ↻{angle:g}°"
        transform_style = f" transform: rotate({angle:.2f}deg);" if angle else ""
        review_class = f" bbox-review-{state}" + (" bbox-selected" if index == selected else "")
        # Prüf-Knöpfe neben die Box (nicht darüber, sonst verdecken sie Text):
        # rechts, bei Boxen am rechten Rand links, bei Boxen über die ganze
        # Breite rechts unterhalb (Zeilenzwischenraum).
        if x2 / width <= 0.86:
            review_side = ""
        elif x1 / width >= 0.14:
            review_side = " bbox-review-left"
        else:
            review_side = " bbox-review-below"
        review_buttons = (
            f"<div class='bbox-review{review_side}'>"
            f"<div class='bbox-review-btn bbox-review-ok{' on' if state == review.ACCEPTED else ''}' "
            f"title='Akzeptieren (nochmal klicken = offen)' "
            f"onmousedown=\"window.qbReview(event,'{line_id}','{review.ACCEPTED}')\">✓</div>"
            f"<div class='bbox-review-btn bbox-review-no{' on' if state == review.REJECTED else ''}' "
            f"title='Nicht akzeptieren - geht nicht ins Training (nochmal klicken = offen)' "
            f"onmousedown=\"window.qbReview(event,'{line_id}','{review.REJECTED}')\">✗</div>"
            f"</div>"
        )

        handles = "".join(
            f"<div class='bbox-handle bbox-handle-{corner}' "
            f"onmousedown=\"window.qbStartResize(event,'{line_id}','{corner}')\"></div>"
            for corner in ("nw", "ne", "sw", "se")
        )
        rotate_handle = (
            f"<div class='bbox-rotate-handle' "
            f"onmousedown=\"window.qbStartRotate(event,'{line_id}')\"></div>"
        )
        nudge_buttons = (
            f"<div class='bbox-nudge bbox-nudge-minus' title='-1°' "
            f"onmousedown=\"window.qbNudgeAngle(event,'{line_id}',-1)\">&minus;</div>"
            f"<div class='bbox-nudge bbox-nudge-plus' title='+1°' "
            f"onmousedown=\"window.qbNudgeAngle(event,'{line_id}',1)\">+</div>"
        )
        apply_last_angle_button = (
            f"<div class='bbox-apply-angle' title='Zuletzt verwendeten Winkel übernehmen' "
            f"onmousedown=\"window.qbApplyLastAngle(event,'{line_id}')\">∠=</div>"
        )

        text_panel = ""
        # Mouse-over: erkannter/korrigierter Text der Box. Bei der gerade
        # geöffneten Box nicht nötig, dort steht der Text im Eingabefeld.
        # Textfarbe inline, da Gradio sonst dunklen Text erzwingt.
        hover_tip = ""
        if active_text_id != line_id:
            shown = text if text else ("(kein sichtbarer Text)" if line.get("status") == NO_TEXT_STATUS else "(leer)")
            predicted = str(line.get("text_predicted", ""))
            extra = ""
            if predicted and predicted != text:
                extra = (
                    f"<br><span style='color:#bbb'>Modell: {escape_html(predicted)}</span>"
                )
            state_text = {
                review.ACCEPTED: "<span style='color:#7ee2a0'>✓ akzeptiert</span>",
                review.REJECTED: "<span style='color:#ff8a8a'>✗ nicht akzeptiert</span>",
                review.OPEN: "<span style='color:#bbb'>offen</span>",
            }[state]
            hover_tip = (
                f"<span class='bbox-tip'><b style='color:{color}'>{index + 1}</b> "
                f"<span style='color:#fff'>{escape_html(shown)}</span>{extra}<br>{state_text}"
                f"<span style='color:#bbb'> · Modell-Konfidenz {CONFIDENCE_NAMES.get(line.get('confidence'), '?')}</span></span>"
            )
        if active_text_id == line_id:
            # Unten an der tatsaechlichen (ggf. gedrehten) Box ausrichten,
            # nicht an ihrer ungedrehten bbox_pixels-Lage - sonst haengt das
            # Textfeld bei gedrehten Boxen sichtbar daneben statt darunter.
            rot_left, _, _, rot_bottom = rotated_box_bounds(
                (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1, angle
            )
            text_left = rot_left / width * 100
            text_top = rot_bottom / height * 100
            text_panel = (
                f"<div class='bbox-text' id='text-{line_id}' "
                f"style='left:{text_left:.3f}%; top:{text_top:.3f}%;'>"
                f"<textarea id='textarea-{line_id}' class='bbox-textarea' autofocus "
                f"onblur=\"window.qbCommitText('{line_id}')\">{escape_html(text)}</textarea>"
                f"</div>"
            )

        parts.append(
            f"<div class='bbox-box{review_class}' id='box-{line_id}' "
            f"style='left:{left:.3f}%; top:{top:.3f}%; width:{box_width:.3f}%; height:{box_height:.3f}%; border-color:{color};{transform_style}' "
            f"onmousedown=\"window.qbStartDrag(event,'{line_id}')\">"
            f"<span class='bbox-label' style='background:{color};'>{label}</span>"
            f"{hover_tip}"
            f"{handles}"
            f"{rotate_handle}"
            f"{nudge_buttons}"
            f"{apply_last_angle_button}"
            f"{review_buttons}"
            f"</div>"
            f"{text_panel}"
        )

    return (
        f"<div class='bbox-canvas' id='bbox-canvas' data-width='{width}' data-height='{height}'>"
        f"<img class='bbox-image' src='{data_uri}' draggable='false' />"
        f"{''.join(parts)}"
        f"</div>"
    )


def normalize_angle(value: Any) -> float:
    """Normalisiert einen Winkel (Grad, im Uhrzeigersinn) auf (-180, 180],
    z.B. für Werte, die direkt in der Tabelle von Hand eingetragen wurden.
    Eine Nachkommastelle reicht für die Neigung einer Textzeile locker."""
    try:
        degrees = float(value)
    except (TypeError, ValueError):
        degrees = 0.0
    degrees = degrees % 360
    if degrees > 180:
        degrees -= 360
    return round(degrees, 1)


def rotated_box_bounds(cx: float, cy: float, w: float, h: float, angle_degrees: float) -> tuple[float, float, float, float]:
    """Achsenparallele Bounding-Box eines um seinen Mittelpunkt (cx,cy) mit
    Breite/Höhe (w,h) im Uhrzeigersinn um angle_degrees gedrehten Rechtecks."""
    angle = math.radians(angle_degrees)
    cos_a, sin_a = abs(math.cos(angle)), abs(math.sin(angle))
    half_w = (w * cos_a + h * sin_a) / 2
    half_h = (w * sin_a + h * cos_a) / 2
    return cx - half_w, cy - half_h, cx + half_w, cy + half_h


def crop_line(image_path: str, annotation: dict[str, Any], selected: int) -> Image.Image | None:
    lines = annotation.get("lines", [])
    if selected < 0 or selected >= len(lines):
        return None
    image = open_scan_for_display(image_path, annotation)
    line = lines[selected]
    x1, y1, x2, y2 = line_to_pixels(line, annotation, image.width, image.height)
    w, h = x2 - x1, y2 - y1
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    mx, my = max(10, image.width // 70), max(8, image.height // 150)
    angle = normalize_angle(line.get("angle", 0))

    if angle:
        # Die tatsächliche (gedrehte) Box ragt über ihre ungedrehte
        # bbox_pixels-Bounding-Box hinaus - erst die AABB der gedrehten Box
        # aus der Seite ausschneiden, dann geraderücken, damit keine Ecken
        # des schräg geschriebenen Texts abgeschnitten werden.
        bx1, by1, bx2, by2 = rotated_box_bounds(cx, cy, w, h, angle)
    else:
        bx1, by1, bx2, by2 = x1, y1, x2, y2

    crop = image.crop((
        max(0, int(bx1) - mx), max(0, int(by1) - my),
        min(image.width, int(bx2) + mx), min(image.height, int(by2) + my),
    ))
    if angle:
        # PIL rotiert bei positivem Winkel gegen den Uhrzeigersinn - das
        # macht die im Uhrzeigersinn positive Neigung der Zeile genau
        # rückgängig, sodass der Text aufrecht/lesbar wird.
        crop = crop.rotate(angle, expand=True)
        target_w, target_h = int(w) + 2 * mx, int(h) + 2 * my
        left = max(0, (crop.width - target_w) // 2)
        top = max(0, (crop.height - target_h) // 2)
        crop = crop.crop((left, top, min(crop.width, left + target_w), min(crop.height, top + target_h)))
    return crop


def annotation_to_table(annotation: dict[str, Any]) -> list[list[Any]]:
    return [
        [
            line["id"], line["text_corrected"], line["confidence"], *line["bbox_pixels"],
            normalize_angle(line.get("angle", 0)), review.LABELS[review.review_state(line)],
        ]
        for line in annotation.get("lines", [])
    ]


def table_to_annotation(table: Any, annotation: dict[str, Any], image_path: str | None = None) -> dict[str, Any]:
    """Übernimmt Tabellenzeilen in die Annotation. annotation["image"] fehlt
    noch komplett, solange ein frisch geladenes Bild weder vorannotiert noch
    aus einer vorhandenen JSON geladen wurde (annotation_state == {}) - in
    dem Fall wird die Bildgröße bei Bedarf direkt aus image_path nachgeladen,
    damit z.B. "Box hinzufügen" auch ganz ohne vorherige Qwen-Vorannotation
    funktioniert.
    """
    result = copy.deepcopy(annotation)

    image = result.get("image")
    if not isinstance(image, dict):
        image = {}
    width = int(image.get("width", 0))
    height = int(image.get("height", 0))
    if (width <= 0 or height <= 0) and image_path:
        width, height = scan_size(image_path)
        result["image"] = {
            **image,
            "file": str(Path(image_path).resolve()),
            "file_name": Path(image_path).name,
            "width": width,
            "height": height,
        }

    if table is None:
        return result
    if hasattr(table, "values"):
        table = table.values.tolist()

    width = int(result.get("image", {}).get("width", 0))
    height = int(result.get("image", {}).get("height", 0))
    if width <= 0 or height <= 0:
        raise ValueError("Die Bildgröße fehlt. Annotation zuerst mit einem Bild laden.")

    old_lines = result.get("lines", [])
    lines = []
    for index, row in enumerate(table):
        if row is None or len(row) < 7:
            continue
        previous = old_lines[index] if index < len(old_lines) else {}
        corrected = str(row[1] if row[1] is not None else "").strip()
        predicted = previous.get("text_predicted", corrected)
        confidence = str(row[2] or "low").lower().strip()
        if confidence not in CONFIDENCE_VALUES:
            confidence = "low"

        pixel_box = validate_pixel_bbox(list(row[3:7]), width, height)
        angle = normalize_angle(row[7]) if len(row) > 7 else normalize_angle(previous.get("angle", 0))
        previous_review = review.review_state(previous) if previous else review.OPEN
        review_value = review.parse_label(row[8], previous_review) if len(row) > 8 else previous_review
        updated = copy.deepcopy(previous)
        updated.update({
            "id": str(row[0] or f"line_{index + 1:04d}"),
            "bbox_pixels": pixel_box,
            "bbox_1000": pixels_to_bbox_1000(pixel_box, width, height),
            "text_predicted": predicted,
            "text_corrected": corrected,
            "confidence": confidence,
            "status": "corrected" if corrected != predicted else "confirmed",
            "review": review_value,
            "angle": angle,
        })
        lines.append(updated)

    result["lines"] = lines
    result["coordinate_system"] = "original_pixels"
    result["schema_version"] = "1.3"
    return result


def store_gui_preannotation(
    image_path: str, model: str, context: int, annotation: dict[str, Any], duration: float
) -> str:
    """Legt eine in der Oberfläche erzeugte Vorannotation zusätzlich in
    <bild>_preannotation.json unter model_runs[<modell>] ab (wie
    qwen_preannotate.py), damit sie später im Auswahlfeld "Angezeigt" wieder
    wählbar ist - auch wenn danach ein anderes Modell vorannotiert. Bilder, die
    nur in Gradios Temp-Ordner liegen (manueller Upload), werden übersprungen."""
    path = Path(image_path)
    if "gradio" in {part.lower() for part in path.parts} or not path.is_file():
        return ""
    target = path.with_name(path.stem + "_preannotation.json")
    lines = [
        {
            "id": line.get("id", f"line_{index:04d}"),
            "bbox_pixels": line["bbox_pixels"],
            "bbox_1000": line.get("bbox_1000"),
            "text": line.get("text_predicted", ""),
            "confidence": line.get("confidence", "low"),
            "angle": line.get("angle", 0.0),
        }
        for index, line in enumerate(annotation.get("lines", []), 1)
    ]
    image = annotation.get("image", {})
    processing = {"model": model.strip(), "context_size": context, "backend": BACKEND, "source": "gui"}
    document = {
        "schema_version": "1.3",
        "task": "handwritten_line_preannotation",
        "coordinate_system": "original_pixels",
        "image": image,
        "processing": processing,
        "lines": lines,
        "errors": [],
    }
    try:
        run = model_runs.make_run(model.strip(), lines, (image.get("width", 0), image.get("height", 0)), processing, duration)
        model_runs.add_preannotation_run(target, model.strip(), run, document)
    except OSError as exc:
        return f" (Vorannotation konnte nicht gespeichert werden: {exc})"
    return f" Gespeichert als Vorannotation '{model.strip()}'."


def start_preannotation(image_path: str | None, model: str, context: int):
    if not image_path:
        raise gr.Error("Bitte zuerst einen Scan auswählen.")
    try:
        started = time.monotonic()
        annotation = add_metadata(run_qwen(image_path, model, context), image_path, model)
        duration = time.monotonic() - started
        stored_note = store_gui_preannotation(image_path, model, int(context), annotation, duration)
        selected = 0 if annotation["lines"] else -1
        return (
            annotation,
            image_path,
            selected,
            render_interactive_preview(image_path, annotation, selected, None),
            crop_line(image_path, annotation, selected),
            annotation_to_table(annotation),
            None,
            f"{len(annotation['lines'])} Zeilen erkannt.{stored_note}",
        )
    except requests.ConnectionError as exc:
        raise gr.Error("Ollama ist unter 127.0.0.1:11434 nicht erreichbar.") from exc
    except Exception as exc:
        raise gr.Error(str(exc)) from exc


def apply_tesseract_boxes(
    table: Any,
    annotation: dict[str, Any],
    image_path: str | None,
    lang: str,
    psm: float | int,
):
    """Ruft den Tesseract-Docker-Dienst ab und rastet die vorhandenen Zeilen
    auf die am besten überlappende Tesseract-Box ein (siehe
    tesseract_boxes.detect_and_adjust). Es werden bewusst keine neuen Zeilen
    aus unzugeordneten Tesseract-Boxen ergänzt - nur bereits vorhandene
    Zeilen werden neu positioniert."""
    if not image_path:
        raise gr.Error("Bitte zuerst einen Scan laden.")
    try:
        annotation = table_to_annotation(table, annotation, image_path)
    except Exception as exc:
        raise gr.Error(str(exc)) from exc

    image = annotation.get("image", {})
    width = int(image.get("width", 0))
    height = int(image.get("height", 0))
    if width <= 0 or height <= 0:
        raise gr.Error("Die Bildgröße fehlt. Annotation zuerst mit einem Bild laden.")
    if int(image.get("pending_rotation", 0)) % 360:
        raise gr.Error(
            "Diese Seite wurde gedreht, aber noch nicht gespeichert - Tesseract würde "
            "sonst auf der ungedrehten Datei suchen. Bitte zuerst 'Annotations-JSON "
            "speichern' klicken."
        )

    try:
        new_lines, status = tesseract_boxes.detect_and_adjust(
            image_path, annotation.get("lines", []), width, height, lang=lang, psm=int(psm), add_unmatched=False
        )
    except RuntimeError as exc:
        raise gr.Error(str(exc)) from exc

    annotation = copy.deepcopy(annotation)
    annotation["lines"] = new_lines
    selected = 0 if new_lines else -1
    return (
        annotation,
        render_interactive_preview(image_path, annotation, selected, None),
        crop_line(image_path, annotation, selected),
        annotation_to_table(annotation),
        selected,
        None,
        status,
    )


def load_pdf_page(pdf_path: str | None, page_number: float | int | None) -> tuple[str, str]:
    if not pdf_path:
        raise gr.Error("Bitte zuerst ein PDF auswählen.")
    try:
        pages = pdf_utils.extract_pdf_pages(pdf_path, dpi=pdf_utils.DEFAULT_PDF_DPI)
    except Exception as exc:
        raise gr.Error(f"PDF konnte nicht gelesen werden: {exc}") from exc
    index = clamp(int(page_number or 1), 1, len(pages)) - 1
    return str(pages[index]), f"Seite {index + 1} von {len(pages)} aus PDF geladen."


def split_into_tiles(image_path: str | None) -> tuple[list[str], str]:
    """Teilt ein großformatiges Bild in Kacheln und speichert jede einzeln.

    Jede Kachel wird anschließend wie ein eigenständiger Scan geladen,
    vorannotiert, korrigiert und gespeichert.
    """
    if not image_path:
        raise gr.Error("Bitte zuerst einen Scan laden.")
    try:
        source = Path(image_path)
        image = open_scan(source)
        tiles = tiling.create_tiles(image)
        if len(tiles) == 1:
            return [], "Bild ist klein genug, keine Aufteilung nötig."
        tile_dir = source.with_name(f"{source.stem}_tiles")
        paths = tiling.save_tiles(tiles, tile_dir, source.stem)
    except Exception as exc:
        raise gr.Error(f"Aufteilen fehlgeschlagen: {exc}") from exc
    return [str(p) for p in paths], f"In {len(paths)} Kacheln aufgeteilt und in {tile_dir} gespeichert."


def load_tile(tile_paths: list[str], tile_number: float | int | None) -> tuple[str, str]:
    if not tile_paths:
        raise gr.Error("Bitte zuerst in Kacheln aufteilen.")
    index = clamp(int(tile_number or 1), 1, len(tile_paths)) - 1
    return tile_paths[index], f"Kachel {index + 1} von {len(tile_paths)} geladen."


def auto_load_existing_annotation(image_path: str):
    """Sucht neben image_path nach einer vorhandenen <bild>_annotation.json
    oder <bild>_preannotation.json und lädt sie automatisch.

    Muss mit dem echten Dateipfad im Dataset aufgerufen werden, nicht mit dem
    von der Gradio-Bildkomponente zurückgelieferten Pfad: Gradio kopiert jedes
    an sie übergebene Bild in ein eigenes Temp-Verzeichnis (siehe
    ensure_image_under_dataset_root), sodass ein Geschwisterdatei-Abgleich über
    image.change() dort nie den echten Ordner (z.B. *_pages/*_tiles) findet.
    """
    stem = Path(image_path)
    annotation_path = stem.with_name(stem.stem + "_annotation.json")
    preannotation_path = stem.with_name(stem.stem + "_preannotation.json")
    source_path = annotation_path if annotation_path.is_file() else preannotation_path if preannotation_path.is_file() else None
    if source_path is None:
        return {}, image_path, -1, EMPTY_PREVIEW_HTML, None, [], None, None, "Keine vorhandene Annotation für diese Datei gefunden."
    try:
        data, loaded_image_path, selected, preview, crop_img, table_rows, active_text, status = load_annotation(
            image_path, str(source_path)
        )
        return data, loaded_image_path, selected, preview, crop_img, table_rows, active_text, str(source_path), status
    except gr.Error as exc:
        return (
            {}, image_path, -1, EMPTY_PREVIEW_HTML, None, [], None, None,
            f"{source_path.name} konnte nicht automatisch geladen werden: {exc}",
        )


def load_pdf_page_and_autoload(pdf_path: str | None, page_number: float | int | None):
    image_path, page_status = load_pdf_page(pdf_path, page_number)
    annotation, image_state_path, selected, preview, crop_img, table_rows, active_text, existing_path, load_status = (
        auto_load_existing_annotation(image_path)
    )
    return (
        display_image_file(image_path), image_state_path, annotation, selected, preview, crop_img, table_rows,
        active_text, existing_path, f"{page_status} {load_status}",
    )


def load_tile_and_autoload(tile_paths: list[str], tile_number: float | int | None):
    image_path, tile_status = load_tile(tile_paths, tile_number)
    annotation, image_state_path, selected, preview, crop_img, table_rows, active_text, existing_path, load_status = (
        auto_load_existing_annotation(image_path)
    )
    return (
        display_image_file(image_path), image_state_path, annotation, selected, preview, crop_img, table_rows,
        active_text, existing_path, f"{tile_status} {load_status}",
    )


def select_dataset_file(payload_json: str, dataset_root: str):
    """Wird ausgelöst, sobald in der Dateiliste auf einen Eintrag geklickt wird.
    Lädt Bilder direkt; bei einem PDF wird automatisch dessen erste Seite
    geladen. Eine bereits vorhandene Annotation wird sofort mit dem echten
    Dateipfad mitgeladen (siehe auto_load_existing_annotation).
    """
    if not payload_json:
        raise gr.Error("Keine Datei ausgewählt.")
    try:
        payload = json.loads(payload_json)
    except json.JSONDecodeError as exc:
        raise gr.Error("Ungültige Auswahl aus der Dateiliste.") from exc

    path = payload.get("path")
    if not path or not Path(path).is_file():
        raise gr.Error("Die ausgewählte Datei wurde nicht gefunden.")

    is_pdf = Path(path).suffix.lower() == ".pdf"
    if is_pdf:
        image_path, page_status = load_pdf_page(path, 1)
    else:
        image_path, page_status = path, "Bild geladen."

    annotation, image_state_path, selected, preview, crop_img, table_rows, active_text, existing_path, load_status = (
        auto_load_existing_annotation(image_path)
    )
    all_html, flagged_html = render_file_lists(dataset_root, path)
    return (
        display_image_file(image_path),
        image_state_path,
        annotation,
        selected,
        preview,
        crop_img,
        table_rows,
        active_text,
        existing_path,
        path if is_pdf else gr.skip(),
        1 if is_pdf else gr.skip(),
        all_html,
        flagged_html,
        f"{page_status} {load_status}",
    )


def reset_image_state(image_path: str | None):
    """Wird nur bei einem echten manuellen Upload/Einfügen in die
    Bildkomponente ausgelöst (image.upload, nicht image.change) und verwirft
    die Annotationsanzeige des vorherigen Bildes, damit sie nicht versehentlich
    unter dem neuen Bildpfad gespeichert wird.
    """
    if not image_path:
        return "", {}, -1, EMPTY_PREVIEW_HTML, None, [], None, None, gr.skip()
    return image_path, {}, -1, EMPTY_PREVIEW_HTML, None, [], None, None, "Neues Bild geladen."


def load_annotation(image_path: str | None, json_path: str | None):
    if not image_path or not json_path:
        raise gr.Error("Bitte Scan und Annotations-JSON auswählen.")
    try:
        data = json.loads(Path(json_path).read_text(encoding="utf-8"))
    except Exception as exc:
        raise gr.Error(f"Laden fehlgeschlagen: {exc}") from exc
    return load_annotation_data(image_path, data)


def load_annotation_data(image_path: str, data: dict[str, Any], status_note: str = ""):
    """Gemeinsamer Ladeweg für eine gespeicherte Annotation, eine
    Vorannotation eines bestimmten Modells oder einen Vergleichslauf."""
    try:
        if not data.get("schema_version"):
            data = normalize_annotation(data)

        stored_image = data.get("image")
        if isinstance(stored_image, dict) and stored_image.get("width") and stored_image.get("height"):
            actual_width, actual_height = scan_size(image_path)
            if (int(stored_image["width"]), int(stored_image["height"])) != (actual_width, actual_height):
                raise ValueError(
                    "Diese Annotation wurde für ein Bild mit "
                    f"{stored_image['width']}x{stored_image['height']} Pixeln erstellt, das "
                    f"aktuell geladene Bild hat aber {actual_width}x{actual_height} Pixel. "
                    "Vermutlich passt das falsche Bild (z.B. die falsche Kachel oder PDF-Seite) "
                    "zu dieser Annotation - bitte das passende Bild laden."
                )

        data = ensure_pixel_boxes(data, image_path)
        selected = 0 if data.get("lines") else -1
        return (
            data,
            image_path,
            selected,
            render_interactive_preview(image_path, data, selected, None),
            crop_line(image_path, data, selected),
            annotation_to_table(data),
            None,
            f"{len(data.get('lines', []))} Zeilen geladen.{status_note}",
        )
    except Exception as exc:
        raise gr.Error(f"Laden fehlgeschlagen: {exc}") from exc


# ---------------------------------------------------------------------------
# Auswahl, welche (Vor-)Annotation im Reiter "Annotation" angezeigt wird
# ---------------------------------------------------------------------------

SOURCE_ANNOTATION = "annotation"


def _source_files(image_path: str) -> tuple[Path, Path]:
    path = Path(image_path)
    return path.with_name(path.stem + "_annotation.json"), path.with_name(path.stem + "_preannotation.json")


def _safe_load(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        return model_runs.load_json(path)
    except (OSError, json.JSONDecodeError):
        return None


def _when(run: dict[str, Any]) -> str:
    created = run.get("created")
    if not created:
        return ""
    try:
        stamp = datetime.datetime.fromisoformat(str(created))
    except ValueError:
        return ""
    return f"  ({stamp:%d.%m. %H:%M})"


def annotation_sources(image_path: str | None) -> tuple[list[tuple[str, str]], str | None]:
    """Auswahlmöglichkeiten für das Feld "Angezeigt": geprüfte Annotation,
    Vorannotationen je Modell (_preannotation.json) und Vergleichsläufe
    (model_runs der geprüften Datei). Standard = was auch automatisch geladen
    wird (geprüfte Annotation, sonst die zuletzt erzeugte Vorannotation)."""
    if not image_path:
        return [], None
    annotation_path, preannotation_path = _source_files(image_path)
    choices: list[tuple[str, str]] = []
    default = None
    annotation = _safe_load(annotation_path)
    if annotation is not None:
        choices.append(("Geprüfte Annotation (gespeichert)", SOURCE_ANNOTATION))
        default = SOURCE_ANNOTATION
    preannotation = _safe_load(preannotation_path)
    if preannotation is not None:
        runs = model_runs.preannotation_runs(preannotation)
        ordered = sorted(runs.items(), key=lambda item: str(item[1].get("created") or ""), reverse=True)
        for label, run in ordered:
            choices.append((f"Vorannotation: {label}{_when(run)}", f"pre:{label}"))
        if default is None and ordered:
            latest = (preannotation.get("processing") or {}).get("model")
            default = f"pre:{latest}" if latest in runs else f"pre:{ordered[0][0]}"
    if annotation is not None:
        for label, run in sorted(model_runs.get_runs(annotation).items()):
            choices.append((f"Vergleichslauf: {label}{_when(run)}", f"cmp:{label}"))
    return choices, default


def refresh_annotation_sources(image_path: str | None, selected: str | None = None):
    choices, default = annotation_sources(image_path)
    values = [value for _, value in choices]
    value = selected if selected in values else default
    return gr.update(choices=choices, value=value)


def _run_as_annotation(run: dict[str, Any], image: dict[str, Any] | None) -> dict[str, Any]:
    lines = sorted(
        (line for line in run.get("lines", []) if isinstance(line, dict) and line.get("bbox_pixels")),
        key=lambda line: (line["bbox_pixels"][1], line["bbox_pixels"][0]),
    )
    size = run.get("image_size") or [0, 0]
    stored_image = dict(image) if isinstance(image, dict) else {}
    if size and size[0] and size[1]:
        stored_image.update({"width": int(size[0]), "height": int(size[1])})
    return {
        "schema_version": "1.3",
        "coordinate_system": "original_pixels",
        "image": stored_image,
        "lines": [
            {
                "id": f"line_{index:04d}",
                "bbox_pixels": line["bbox_pixels"],
                "text": line.get("text", ""),
                "confidence": line.get("confidence", "low"),
                "angle": line.get("angle", 0.0),
            }
            for index, line in enumerate(lines, 1)
        ],
    }


def load_annotation_source(image_path: str | None, source: str | None):
    """Lädt die im Feld "Angezeigt" gewählte Quelle in Tabelle und Vorschau."""
    if not image_path or not source:
        raise gr.Error("Bitte zuerst einen Scan laden.")
    annotation_path, preannotation_path = _source_files(image_path)
    if source == SOURCE_ANNOTATION:
        return load_annotation(image_path, str(annotation_path))
    kind, _, label = source.partition(":")
    if kind == "pre":
        data = _safe_load(preannotation_path) or {}
        runs = model_runs.preannotation_runs(data)
        note = f" Vorannotation von '{label}'."
        if annotation_path.is_file():
            note += " Achtung: Speichern ersetzt die bereits geprüfte Annotation dieser Seite."
    elif kind == "cmp":
        data = _safe_load(annotation_path) or {}
        runs = model_runs.get_runs(data)
        note = (
            f" Vergleichslauf '{label}' (ungeprüft). Achtung: Speichern ersetzt die geprüfte "
            "Annotation dieser Seite durch diese Zeilen."
        )
    else:
        raise gr.Error(f"Unbekannte Auswahl: {source}")
    if label not in runs:
        raise gr.Error(f"'{label}' ist für diese Seite nicht mehr vorhanden.")
    return load_annotation_data(image_path, _run_as_annotation(runs[label], data.get("image")), note)


def select_row(table: Any, annotation: dict[str, Any], image_path: str, evt: gr.SelectData):
    try:
        annotation = table_to_annotation(table, annotation, image_path)
    except Exception as exc:
        raise gr.Error(str(exc)) from exc
    event_index = evt.index
    if isinstance(event_index, (list, tuple)):
        if not event_index:
            raise gr.Error("Gradio hat keinen Tabellenindex geliefert.")
        index = int(event_index[0])
    else:
        index = int(event_index)
    if not annotation.get("lines"):
        raise gr.Error("Die Annotation enthält keine Zeilen.")
    index = clamp(index, 0, len(annotation["lines"]) - 1)
    return (
        annotation,
        render_interactive_preview(image_path, annotation, index, None),
        crop_line(image_path, annotation, index),
        index,
        None,
        f"Ausgewählt: Zeile {index + 1}",
    )


def refresh(table: Any, annotation: dict[str, Any], image_path: str, selected: int):
    if not image_path:
        raise gr.Error("Keine Annotation geladen.")
    try:
        annotation = table_to_annotation(table, annotation, image_path)
    except Exception as exc:
        raise gr.Error(str(exc)) from exc
    if annotation.get("lines"):
        selected = clamp(int(selected), 0, len(annotation["lines"]) - 1)
    else:
        selected = -1
    return (
        annotation,
        render_interactive_preview(image_path, annotation, selected, None),
        crop_line(image_path, annotation, selected),
        selected,
        None,
        "Änderungen übernommen.",
    )


NEW_BOX_WIDTH_FRACTION = 0.2
NEW_BOX_HEIGHT_FRACTION = 0.03
NEW_BOX_MIN_SIZE = 15


def _default_new_box(width: int, height: int) -> list[int]:
    """Platziert eine neue Box mittig im Bild, grob in Textzeilen-Proportionen;
    der Nutzer verschiebt/skaliert sie danach per Maus in der Vorschau."""
    box_width = clamp(round(width * NEW_BOX_WIDTH_FRACTION), NEW_BOX_MIN_SIZE, width)
    box_height = clamp(round(height * NEW_BOX_HEIGHT_FRACTION), NEW_BOX_MIN_SIZE, height)
    x1 = (width - box_width) // 2
    y1 = (height - box_height) // 2
    return [x1, y1, x1 + box_width, y1 + box_height]


def _unique_line_id(lines: list[dict[str, Any]]) -> str:
    existing_ids = {line.get("id") for line in lines}
    number = len(lines) + 1
    new_id = f"line_{number:04d}"
    while new_id in existing_ids:
        number += 1
        new_id = f"line_{number:04d}"
    return new_id


def add_box(table: Any, annotation: dict[str, Any], image_path: str):
    if not image_path:
        raise gr.Error("Kein Bild geladen.")
    try:
        annotation = table_to_annotation(table, annotation, image_path)
    except Exception as exc:
        raise gr.Error(str(exc)) from exc

    image = annotation.get("image", {})
    width = int(image.get("width", 0))
    height = int(image.get("height", 0))
    if width <= 0 or height <= 0:
        raise gr.Error("Die Bildgröße fehlt. Annotation zuerst mit einem Bild laden.")

    lines = annotation.get("lines", [])
    box = _default_new_box(width, height)
    new_line = {
        "id": _unique_line_id(lines),
        "bbox_pixels": box,
        "bbox_1000": pixels_to_bbox_1000(box, width, height),
        "text_predicted": "",
        "text_corrected": "",
        "confidence": "low",
        "status": "unreviewed",
        "review": review.OPEN,
        "angle": 0.0,
    }
    lines = lines + [new_line]
    annotation["lines"] = lines
    selected = len(lines) - 1
    return (
        annotation,
        render_interactive_preview(image_path, annotation, selected, None),
        crop_line(image_path, annotation, selected),
        annotation_to_table(annotation),
        selected,
        None,
        f"{new_line['id']} hinzugefügt - Position/Größe in der Vorschau anpassen.",
    )


def delete_box(table: Any, annotation: dict[str, Any], image_path: str, selected: int):
    if not image_path:
        raise gr.Error("Kein Bild geladen.")
    try:
        annotation = table_to_annotation(table, annotation, image_path)
    except Exception as exc:
        raise gr.Error(str(exc)) from exc

    lines = annotation.get("lines", [])
    index = int(selected)
    if index < 0 or index >= len(lines):
        raise gr.Error("Bitte zuerst eine Zeile auswählen (Tabellenzeile oder Box anklicken).")

    removed_id = lines[index].get("id") or f"Zeile {index + 1}"
    lines = lines[:index] + lines[index + 1 :]
    annotation["lines"] = lines
    new_selected = clamp(index, 0, len(lines) - 1) if lines else -1
    return (
        annotation,
        render_interactive_preview(image_path, annotation, new_selected, None),
        crop_line(image_path, annotation, new_selected),
        annotation_to_table(annotation),
        new_selected,
        None,
        f"{removed_id} gelöscht.",
    )


def selected_line_text(annotation: dict[str, Any], selected: int) -> str:
    """Text der ausgewählten Zeile für das Korrekturfeld unter dem Zeilenausschnitt."""
    lines = (annotation or {}).get("lines", [])
    try:
        index = int(selected)
    except (TypeError, ValueError):
        return ""
    if 0 <= index < len(lines):
        return str(lines[index].get("text_corrected", ""))
    return ""


def refresh_line_text(annotation: dict[str, Any], selected: int, current: str | None):
    """Füllt das Korrekturfeld nach Auswahl-/Datenänderungen. Steht dort schon
    der richtige Text, wird nichts gesendet - so wird nach "akzeptieren und
    weiter" nichts überschrieben, was man in der nächsten Zeile bereits tippt."""
    text = selected_line_text(annotation, selected)
    return gr.skip() if text == (current or "") else text


def _apply_text(line: dict[str, Any], text: str | None) -> bool:
    """Übernimmt einen korrigierten Text in die Zeile; True, falls geändert."""
    if text is None:
        return False
    corrected = str(text).strip()
    if corrected == str(line.get("text_corrected", "")):
        return False
    line["text_corrected"] = corrected
    predicted = line.get("text_predicted", corrected)
    line["status"] = "corrected" if corrected != predicted else "confirmed"
    return True


def set_review(
    value: str | None, text: str | None, table: Any, annotation: dict[str, Any], image_path: str, selected: int,
    hide_accepted: bool = False,
):
    """Übernimmt den Text aus dem Korrekturfeld, setzt den Prüfstatus der
    ausgewählten Zeile und lädt sofort die nächste Zeile (Zeilenausschnitt,
    Korrekturfeld und Markierung in der Vorschau)."""
    if not image_path:
        raise gr.Error("Kein Bild geladen.")
    try:
        annotation = table_to_annotation(table, annotation, image_path)
    except Exception as exc:
        raise gr.Error(str(exc)) from exc
    lines = annotation.get("lines", [])
    index = int(selected)
    if index < 0 or index >= len(lines):
        raise gr.Error("Bitte zuerst eine Zeile auswählen (Tabellenzeile oder Box anklicken).")
    _apply_text(lines[index], text)
    if value is not None:  # None = überspringen, Prüfstatus bleibt
        lines[index]["review"] = value
    annotation["lines"] = lines
    # Nächste Zeile in Reihenfolge (bei ausgeblendeten akzeptierten Boxen die
    # nächste nicht akzeptierte); am Ende zur ersten noch offenen Zeile.
    later = range(index + 1, len(lines))
    if hide_accepted:
        new_selected = next((i for i in later if review.review_state(lines[i]) != review.ACCEPTED), None)
    else:
        new_selected = index + 1 if index + 1 < len(lines) else None
    if new_selected is None:
        new_selected = next((i for i, line in enumerate(lines) if review.review_state(line) == review.OPEN and i != index), index)
    word = {review.ACCEPTED: "akzeptiert", review.REJECTED: "nicht akzeptiert", review.OPEN: "offen", None: "übersprungen"}[value]
    return (
        annotation,
        render_interactive_preview(image_path, annotation, new_selected, None),
        crop_line(image_path, annotation, new_selected),
        annotation_to_table(annotation),
        new_selected,
        None,
        f"Zeile {index + 1} {word}, weiter mit Zeile {new_selected + 1}. {review.summary(lines)}",
        selected_line_text(annotation, new_selected),
    )


def accept_selected(text: str, table: Any, annotation: dict[str, Any], image_path: str, selected: int, hide_accepted: bool = False):
    return set_review(review.ACCEPTED, text, table, annotation, image_path, selected, hide_accepted)


def reject_selected(text: str, table: Any, annotation: dict[str, Any], image_path: str, selected: int, hide_accepted: bool = False):
    return set_review(review.REJECTED, text, table, annotation, image_path, selected, hide_accepted)


def skip_selected(text: str, table: Any, annotation: dict[str, Any], image_path: str, selected: int, hide_accepted: bool = False):
    """Überspringen: Prüfstatus unverändert lassen (bereits getippte Korrektur
    wird trotzdem übernommen) und zur nächsten Zeile gehen."""
    return set_review(None, text, table, annotation, image_path, selected, hide_accepted)


def accept_all_open(text: str, table: Any, annotation: dict[str, Any], image_path: str, selected: int, hide_accepted: bool = False):
    if not image_path:
        raise gr.Error("Kein Bild geladen.")
    try:
        annotation = table_to_annotation(table, annotation, image_path)
    except Exception as exc:
        raise gr.Error(str(exc)) from exc
    lines = annotation.get("lines", [])
    selected = int(selected) if 0 <= int(selected) < len(lines) else (0 if lines else -1)
    if selected >= 0:
        _apply_text(lines[selected], text)
    changed = 0
    for line in lines:
        if review.review_state(line) == review.OPEN:
            line["review"] = review.ACCEPTED
            changed += 1
    annotation["lines"] = lines
    return (
        annotation,
        render_interactive_preview(image_path, annotation, selected, None),
        crop_line(image_path, annotation, selected),
        annotation_to_table(annotation),
        selected,
        None,
        f"{changed} offene Zeile(n) akzeptiert. {review.summary(lines)}",
        selected_line_text(annotation, selected),
    )


ROTATE_BOX_NUDGE_DEGREES = 5.0


def rotate_box(clockwise: bool, table: Any, annotation: dict[str, Any], image_path: str, selected: int):
    """Dreht die ausgewählte Box um ROTATE_BOX_NUDGE_DEGREES° (Feinjustierung
    per Klick - für frei wählbare Winkel siehe der Dreh-Griff über der Box in
    der Vorschau, oder direkte Eingabe in der Tabellenspalte "Winkel (°)").
    Die Box-Rechteckgeometrie (bbox_pixels) bleibt unverändert - nur
    annotation["lines"][i]["angle"] ändert sich; crop_line() und die
    Vorschau drehen die Darstellung dementsprechend, damit z.B. eine
    senkrecht geschriebene Randnotiz lesbar bleibt.
    """
    if not image_path:
        raise gr.Error("Kein Bild geladen.")
    try:
        annotation = table_to_annotation(table, annotation, image_path)
    except Exception as exc:
        raise gr.Error(str(exc)) from exc

    lines = annotation.get("lines", [])
    index = int(selected)
    if index < 0 or index >= len(lines):
        raise gr.Error("Bitte zuerst eine Zeile auswählen (Tabellenzeile oder Box anklicken).")

    line = lines[index]
    delta = ROTATE_BOX_NUDGE_DEGREES if clockwise else -ROTATE_BOX_NUDGE_DEGREES
    angle = normalize_angle(float(line.get("angle", 0)) + delta)
    line["angle"] = angle
    annotation["lines"] = lines
    line_id = line.get("id") or f"Zeile {index + 1}"
    return (
        annotation,
        render_interactive_preview(image_path, annotation, index, None),
        crop_line(image_path, annotation, index),
        annotation_to_table(annotation),
        index,
        None,
        f"{line_id}: Winkel auf {angle:g}° gestellt.",
        angle,
    )


def rotate_box_left(table: Any, annotation: dict[str, Any], image_path: str, selected: int):
    return rotate_box(False, table, annotation, image_path, selected)


def rotate_box_right(table: Any, annotation: dict[str, Any], image_path: str, selected: int):
    return rotate_box(True, table, annotation, image_path, selected)


def mark_no_text(table: Any, annotation: dict[str, Any], image_path: str):
    """Markiert eine Seite ganz ohne erkennbaren Text (z.B. leere Seite, reiner
    Stempel-/Wasserzeichen-Scan ohne jede Handschrift): legt eine einzige,
    das gesamte Bild umfassende Zeile mit Status no_text an, damit die Seite
    als geprüft (bewusst leer) gespeichert werden kann statt einfach
    ungeprüft zu wirken. Da training_answer() nur Zeilen mit nicht-leerem
    text_corrected exportiert, wird die Seite dadurch zugleich vom Training
    ausgeschlossen.

    Nur nutzbar, solange die Seite noch keine einzige Box hat - für einzelne
    leere/falsche Boxen bei sonst beschriebenen Seiten stattdessen
    "Ausgewählte Box löschen" verwenden.
    """
    if not image_path:
        raise gr.Error("Kein Bild geladen.")
    try:
        annotation = table_to_annotation(table, annotation, image_path)
    except Exception as exc:
        raise gr.Error(str(exc)) from exc

    if annotation.get("lines"):
        raise gr.Error(
            "Diese Seite hat noch Boxen. 'Kein sichtbarer Text' ist nur für Seiten "
            "ganz ohne Boxen gedacht (z.B. eine leere Seite) - einzelne leere oder "
            "falsche Boxen stattdessen mit 'Ausgewählte Box löschen' entfernen."
        )

    image = annotation.get("image", {})
    width = int(image.get("width", 0))
    height = int(image.get("height", 0))
    if width <= 0 or height <= 0:
        raise gr.Error("Die Bildgröße fehlt. Annotation zuerst mit einem Bild laden.")

    line = {
        "id": "line_0001",
        "bbox_pixels": [0, 0, width, height],
        "bbox_1000": [0, 0, 1000, 1000],
        "text_predicted": "",
        "text_corrected": "",
        "confidence": "low",
        "status": NO_TEXT_STATUS,
        # Bewusst als leer geprüft: zählt als erledigt (geht mangels Text
        # ohnehin nicht ins Training).
        "review": review.ACCEPTED,
        "angle": 0.0,
        "source": "manual",
    }
    annotation["lines"] = [line]
    return (
        annotation,
        render_interactive_preview(image_path, annotation, 0, None),
        crop_line(image_path, annotation, 0),
        annotation_to_table(annotation),
        0,
        None,
        "Seite als 'kein sichtbarer Text' markiert (grau, vom Training ausgeschlossen).",
    )


def rotate_bbox_cw(bbox: list[int], old_height: int) -> list[int]:
    x1, y1, x2, y2 = bbox
    return [old_height - y2, x1, old_height - y1, x2]


def rotate_bbox_ccw(bbox: list[int], old_width: int) -> list[int]:
    x1, y1, x2, y2 = bbox
    return [y1, old_width - x2, y2, old_width - x1]


def rotate_page(clockwise: bool, table: Any, annotation: dict[str, Any], image_path: str, selected: int):
    """Dreht die Seite in der Vorschau um 90°: Boxen und Bildgröße werden
    sofort mitgedreht (siehe rotate_bbox_cw/ccw), die Bilddatei auf der
    Festplatte aber erst beim Klick auf "Annotations-JSON speichern"
    tatsächlich gedreht (siehe save_annotation) - annotation["image"]
    ["pending_rotation"] merkt sich bis dahin, wie viele Grad noch
    ausstehen. Bis dahin zeigen Vorschau und Zeilen-Crop (open_scan_for_display)
    bereits die gedrehte Ansicht, ohne die Originaldatei anzufassen.
    """
    if not image_path:
        raise gr.Error("Kein Bild geladen.")
    try:
        annotation = table_to_annotation(table, annotation, image_path)
    except Exception as exc:
        raise gr.Error(str(exc)) from exc

    image = annotation.get("image", {})
    width = int(image.get("width", 0))
    height = int(image.get("height", 0))
    if width <= 0 or height <= 0:
        raise gr.Error("Die Bildgröße fehlt. Annotation zuerst mit einem Bild laden.")

    new_lines = []
    for line in annotation.get("lines", []):
        line = dict(line)
        try:
            box = list(line["bbox_pixels"])
        except (KeyError, TypeError):
            new_lines.append(line)
            continue
        new_box = rotate_bbox_cw(box, height) if clockwise else rotate_bbox_ccw(box, width)
        line["bbox_pixels"] = new_box
        line["bbox_1000"] = pixels_to_bbox_1000(new_box, height, width)
        new_lines.append(line)
    annotation["lines"] = new_lines

    pending = int(image.get("pending_rotation", 0)) + (90 if clockwise else -90)
    annotation["image"] = {**image, "width": height, "height": width, "pending_rotation": pending % 360}

    selected = clamp(int(selected), 0, len(new_lines) - 1) if new_lines else -1
    direction = "im Uhrzeigersinn" if clockwise else "gegen den Uhrzeigersinn"
    return (
        annotation,
        render_interactive_preview(image_path, annotation, selected, None),
        crop_line(image_path, annotation, selected),
        annotation_to_table(annotation),
        selected,
        None,
        f"Seite um 90° {direction} gedreht (Vorschau) - wird beim Speichern der Annotation auf die Bilddatei angewendet.",
    )


def rotate_page_left(table: Any, annotation: dict[str, Any], image_path: str, selected: int):
    return rotate_page(False, table, annotation, image_path, selected)


def rotate_page_right(table: Any, annotation: dict[str, Any], image_path: str, selected: int):
    return rotate_page(True, table, annotation, image_path, selected)


def sync_bbox_edit(
    payload_json: str,
    annotation: dict[str, Any],
    image_path: str,
    selected: int,
    active_text: str | None,
    last_angle: float,
):
    """Wird vom versteckten Sync-Textfeld ausgelöst, sobald im Vorschau-Overlay
    eine Box verschoben/skaliert, ihr Text bearbeitet/aufgeklappt oder ihr
    Winkel geändert wurde. last_angle merkt sich seitenweit den zuletzt per
    Dreh-Griff/Nudge-Knopf/±5°-Button gesetzten Winkel, damit ihn der
    "Winkel übernehmen"-Knopf einer anderen Box mit einem Klick übernehmen
    kann (z.B. für mehrere Zeilen derselben schräg geschriebenen Randnotiz).
    """
    if not payload_json or not image_path:
        raise gr.Error("Keine Änderung zum Übernehmen vorhanden.")
    try:
        payload = json.loads(payload_json)
    except json.JSONDecodeError as exc:
        raise gr.Error("Ungültige Synchronisationsdaten aus der Vorschau.") from exc

    lines = annotation.get("lines", [])
    line_id = payload.get("id")
    index = next((i for i, line in enumerate(lines) if line.get("id") == line_id), -1)
    if index < 0:
        raise gr.Error("Zeile wurde in der Annotation nicht gefunden.")

    image = annotation.get("image", {})
    width = int(image.get("width", 0))
    height = int(image.get("height", 0))
    if width <= 0 or height <= 0:
        raise gr.Error("Die Bildgröße fehlt. Annotation zuerst mit einem Bild laden.")

    action = payload.get("type")
    if action == "move":
        try:
            pixel_box = validate_pixel_bbox(payload.get("bbox_pixels"), width, height)
        except (TypeError, ValueError) as exc:
            raise gr.Error(str(exc)) from exc
        lines[index]["bbox_pixels"] = pixel_box
        lines[index]["bbox_1000"] = pixels_to_bbox_1000(pixel_box, width, height)
        selected = index
        status = f"Box {index + 1} verschoben/skaliert."
    elif action == "text":
        corrected = str(payload.get("text", "")).strip()
        lines[index]["text_corrected"] = corrected
        predicted = lines[index].get("text_predicted", corrected)
        lines[index]["status"] = "corrected" if corrected != predicted else "confirmed"
        selected = index
        status = f"Text für Zeile {index + 1} aktualisiert."
    elif action == "toggle":
        active_text = None if active_text == line_id else line_id
        selected = index
        status = f"Zeile {index + 1} ausgewählt." if active_text else "Textfeld geschlossen."
        # Reine Auswahl ändert keine Daten: Tabelle nicht neu übertragen und
        # im Browser neu aufbauen (bei vielen Zeilen spürbar langsam).
        return (
            annotation,
            render_interactive_preview(image_path, annotation, selected, active_text),
            crop_line(image_path, annotation, selected),
            gr.skip(),
            selected,
            active_text,
            status,
            last_angle,
        )
    elif action == "review":
        value = payload.get("value")
        if value not in review.STATES:
            raise gr.Error("Ungültiger Prüfstatus.")
        current = review.review_state(lines[index])
        lines[index]["review"] = review.OPEN if current == value else value
        selected = index
        state_name = {review.ACCEPTED: "akzeptiert", review.REJECTED: "nicht akzeptiert", review.OPEN: "offen"}
        status = f"Zeile {index + 1}: {state_name[lines[index]['review']]}. {review.summary(lines)}"
    elif action == "rotate":
        angle = normalize_angle(payload.get("angle", 0))
        lines[index]["angle"] = angle
        last_angle = angle
        selected = index
        status = f"Box {index + 1} auf {angle:g}° gedreht."
    elif action == "nudge_angle":
        try:
            delta = float(payload.get("delta", 0))
        except (TypeError, ValueError):
            delta = 0.0
        angle = normalize_angle(float(lines[index].get("angle", 0)) + delta)
        lines[index]["angle"] = angle
        last_angle = angle
        selected = index
        status = f"Box {index + 1} auf {angle:g}° gedreht."
    elif action == "apply_last_angle":
        angle = normalize_angle(last_angle)
        lines[index]["angle"] = angle
        selected = index
        status = f"Box {index + 1}: zuletzt verwendeten Winkel ({angle:g}°) übernommen."
    else:
        raise gr.Error("Unbekannte Aktion aus der Vorschau.")

    annotation["lines"] = lines
    return (
        annotation,
        render_interactive_preview(image_path, annotation, selected, active_text),
        crop_line(image_path, annotation, selected),
        annotation_to_table(annotation),
        selected,
        active_text,
        status,
        last_angle,
    )


def ensure_image_under_dataset_root(image_path: str, dataset_root: str) -> str:
    """Kopiert das Bild ins Dataset-Wurzelverzeichnis, falls es dort noch
    nicht liegt (z.B. weil es als Gradio-Upload ins OS-Temp-Verzeichnis
    kopiert wurde - "sources=[upload]" tut das immer, auch wenn die
    Originaldatei schon lokal existiert). Temp-Verzeichnisse ueberstehen
    keinen Neustart zuverlaessig, Annotation und train.jsonl duerfen also
    nicht dauerhaft dorthin zeigen.
    """
    if not dataset_root.strip():
        return image_path
    image = Path(image_path).resolve()
    root = Path(dataset_root).resolve()
    try:
        image.relative_to(root)
        return str(image)
    except ValueError:
        pass
    target_dir = root / "uploads"
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / image.name
    if target.exists() and not target.samefile(image):
        target = target_dir / f"{image.stem}_{uuid.uuid4().hex[:8]}{image.suffix}"
    if not target.exists():
        shutil.copy2(image, target)
    return str(target)


def save_annotation(table: Any, annotation: dict[str, Any], image_path: str, model: str, dataset_root: str):
    if not image_path:
        raise gr.Error("Keine Annotation vorhanden.")
    try:
        image_path = ensure_image_under_dataset_root(image_path, dataset_root)
        annotation = table_to_annotation(table, annotation, image_path)

        # Eine per rotate_page() nur in der Vorschau gedrehte Seite wird erst
        # hier tatsächlich auf die Bilddatei angewendet - VOR add_metadata(),
        # damit dessen ensure_pixel_boxes() beim Neuöffnen der Datei bereits
        # die neue (gedrehte) Bildgröße sieht und nicht versucht, die längst
        # gedrehten Box-Koordinaten gegen die alte, ungedrehte Größe zu
        # validieren.
        pending_rotation = int(annotation.get("image", {}).get("pending_rotation", 0)) % 360
        if pending_rotation:
            rotated = open_scan(image_path).rotate(-pending_rotation, expand=True)
            save_kwargs = {"quality": 95} if Path(image_path).suffix.lower() in {".jpg", ".jpeg"} else {}
            rotated.save(image_path, **save_kwargs)
            annotation["image"]["pending_rotation"] = 0

        annotation = add_metadata(annotation, image_path, model)
        output = Path(image_path).with_name(Path(image_path).stem + "_annotation.json")
        # Modellvergleichsläufe (Reiter "Modellvergleich") werden nur direkt in
        # der Datei gepflegt; die Version auf der Platte ist maßgeblich, damit
        # ein zwischenzeitlich (z.B. per model_compare.py) ergänzter Lauf nicht
        # mit dem älteren Stand aus der geöffneten Annotation überschrieben wird.
        # Nach einer Seitendrehung passen ihre Boxen nicht mehr -> verwerfen.
        disk_runs = model_compare.runs_on_disk(output)
        annotation.pop(model_compare.RUNS_KEY, None)
        runs_note = ""
        if pending_rotation and disk_runs:
            runs_note = " Modellvergleichsläufe dieser Seite verworfen (Boxen passen nach der Drehung nicht mehr)."
        elif disk_runs:
            annotation[model_compare.RUNS_KEY] = disk_runs
        output.write_text(json.dumps(annotation, ensure_ascii=False, indent=2), encoding="utf-8")
        all_html, flagged_html = render_file_lists(dataset_root, image_path)
        rotation_note = (" (Bilddatei physisch gedreht)" if pending_rotation else "") + runs_note
        return (
            annotation,
            str(output),
            f"Annotation gespeichert: {output}{rotation_note} | {review.summary(annotation.get('lines', []))}",
            image_path,
            all_html,
            flagged_html,
        )
    except Exception as exc:
        raise gr.Error(f"Speichern fehlgeschlagen: {exc}") from exc


def relative_image_path(image_path: str, dataset_root: str) -> str:
    image = Path(image_path).resolve()
    if dataset_root.strip():
        try:
            return image.relative_to(Path(dataset_root).resolve()).as_posix()
        except ValueError:
            pass
    return image.as_posix()


def training_answer(annotation: dict[str, Any]) -> dict[str, Any]:
    """Zielantwort fürs Training: nur akzeptierte Zeilen mit Text (siehe
    review.py). Nicht akzeptierte und offene Zeilen werden stattdessen im
    Trainingsbild abgedeckt (training_image)."""
    lines = []
    for line in annotation.get("lines", []):
        if not review.is_training_line(line):
            continue
        text = str(line.get("text_corrected", line.get("text", ""))).strip()
        entry = {"bbox_1000": validate_bbox(line["bbox_1000"]), "text": text}
        # "angle" nur bei spürbar gedrehten Zeilen mit ausgeben (siehe
        # rotate_box/PREANNOTATION_PROMPT/TRAINING_PROMPT) - hält das
        # Zielschema für den weit überwiegenden achsenparallelen Regelfall
        # unverändert zum bisherigen Format.
        angle = normalize_angle(line.get("angle", 0))
        if angle:
            entry["angle"] = angle
        lines.append(entry)
    lines.sort(key=lambda x: (x["bbox_1000"][1], x["bbox_1000"][0]))
    return {"lines": lines}


MASKED_DIR = "_training_masked"
MASK_MARGIN_PX = 4
PARTIAL_MODES = ("mask", "skip")


def _rotated_corners(box: list[float], angle: float, margin: float = 0.0) -> list[tuple[float, float]]:
    """Eckpunkte einer um ihren Mittelpunkt im Uhrzeigersinn gedrehten Box
    (Bildkoordinaten, y nach unten - wie CSS rotate() in der Vorschau)."""
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    hw, hh = (x2 - x1) / 2 + margin, (y2 - y1) / 2 + margin
    rad = math.radians(angle)
    cos_a, sin_a = math.cos(rad), math.sin(rad)
    return [
        (cx + dx * cos_a - dy * sin_a, cy + dx * sin_a + dy * cos_a)
        for dx, dy in ((-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh))
    ]


def training_image(annotation: dict[str, Any], image_path: str, dataset_root: str) -> tuple[str, int]:
    """Bild für den Trainingsdatensatz. Gibt es Zeilen, die nicht ins
    Training gehen (nicht akzeptiert oder offen), wird eine Kopie erzeugt, in
    der diese Bereiche mit der Hintergrundfarbe abgedeckt sind - sonst würde
    das Modell lernen, sichtbare Zeilen wegzulassen. Akzeptierte Zeilen, die
    von einer abgedeckten Box überlappt werden, werden danach wiederhergestellt.
    Das Original bleibt unverändert; die Kopie liegt unter
    <Dataset-Wurzel>/_training_masked/<gleicher Unterordner>/<name>_masked.png.
    Liefert (Bildpfad, Anzahl abgedeckter Zeilen)."""
    lines = [line for line in annotation.get("lines", []) if isinstance(line, dict) and line.get("bbox_pixels")]
    excluded = [line for line in lines if review.is_excluded_region(line)]
    if not excluded:
        return image_path, 0
    image = open_scan(image_path)
    fill = tuple(int(v) for v in ImageStat.Stat(image).median)
    masked = image.copy()
    draw = ImageDraw.Draw(masked)
    for line in excluded:
        draw.polygon(_rotated_corners(line["bbox_pixels"], normalize_angle(line.get("angle", 0)), MASK_MARGIN_PX), fill=fill)
    for line in lines:
        if not review.is_training_line(line):
            continue
        x1, y1, x2, y2 = line["bbox_pixels"]
        bx1, by1, bx2, by2 = rotated_box_bounds((x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1, normalize_angle(line.get("angle", 0)))
        region = (
            max(0, int(bx1)), max(0, int(by1)),
            min(image.width, int(math.ceil(bx2))), min(image.height, int(math.ceil(by2))),
        )
        if region[2] > region[0] and region[3] > region[1]:
            masked.paste(image.crop(region), region[:2])

    root = Path(dataset_root).resolve() if dataset_root and dataset_root.strip() else Path(image_path).resolve().parent
    relative = Path(relative_image_path(image_path, dataset_root))
    sub_dir = relative.parent if not relative.is_absolute() else Path()
    target = root / MASKED_DIR / sub_dir / f"{Path(image_path).stem}_masked.png"
    target.parent.mkdir(parents=True, exist_ok=True)
    masked.save(target)
    return str(target), len(excluded)


def record_key(record: dict[str, Any]) -> str:
    """Vergleichsschlüssel einer JSONL-Zeile: Bildpfad ohne Endung, bei
    abgedeckten Kopien auf das Original zurückgeführt - damit ein erneuter
    Export derselben Seite den alten Datensatz ersetzt, egal ob das Bild
    zwischendurch abgedeckt war oder nicht."""
    image = (record_image(record) or "").replace("\\", "/")
    parts = PurePosixPath(image).parts
    if MASKED_DIR in parts:
        parts = parts[parts.index(MASKED_DIR) + 1 :]
    path = PurePosixPath(*parts) if parts else PurePosixPath(image)
    stem = path.stem[: -len("_masked")] if path.stem.endswith("_masked") and MASKED_DIR in image else path.stem
    return str(path.with_name(stem))


def build_training_record(
    annotation: dict[str, Any], image_path: str, dataset_root: str, partial: str = "mask"
) -> tuple[dict[str, Any], dict[str, int]]:
    """Trainingsdatensatz einer Seite. partial bestimmt den Umgang mit nicht
    vollständig geprüften Seiten: "mask" (Standard) deckt nicht akzeptierte
    und offene Zeilen im Bild ab, "skip" lässt solche Seiten ganz weg."""
    answer = training_answer(annotation)
    counts = review.counts(annotation.get("lines", []))
    if not answer["lines"]:
        raise ValueError(f"Keine akzeptierten Zeilen mit Text ({review.summary(annotation.get('lines', []))}).")
    excluded = sum(1 for line in annotation.get("lines", []) if isinstance(line, dict) and review.is_excluded_region(line))
    if partial == "skip" and excluded:
        raise ValueError(f"Nicht vollständig akzeptiert ({review.summary(annotation.get('lines', []))}).")
    image_for_training, masked = training_image(annotation, image_path, dataset_root) if excluded else (image_path, 0)
    record = {
        "messages": [
            {"role": "user", "content": [
                {"type": "image", "image": relative_image_path(image_for_training, dataset_root)},
                {"type": "text", "text": TRAINING_PROMPT},
            ]},
            {"role": "assistant", "content": [
                {"type": "text", "text": json.dumps(answer, ensure_ascii=False, separators=(",", ":"))}
            ]},
        ]
    }
    return record, {"lines": len(answer["lines"]), "masked": masked, **counts}


def training_record(annotation: dict[str, Any], image_path: str, dataset_root: str, partial: str = "mask") -> dict[str, Any]:
    return build_training_record(annotation, image_path, dataset_root, partial)[0]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if line.strip():
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Ungültiges JSONL in Zeile {number}: {exc}") from exc
    return records


def record_image(record: dict[str, Any]) -> str | None:
    try:
        for item in record["messages"][0]["content"]:
            if item.get("type") == "image":
                return item.get("image")
    except (KeyError, IndexError, TypeError):
        return None
    return None


def export_jsonl(table: Any, annotation: dict[str, Any], image_path: str, dataset_root: str, mode: str):
    if not image_path:
        raise gr.Error("Keine Annotation vorhanden.")
    try:
        image_path = ensure_image_under_dataset_root(image_path, dataset_root)
        annotation = table_to_annotation(table, annotation, image_path)
        if int(annotation.get("image", {}).get("pending_rotation", 0)) % 360:
            raise gr.Error(
                "Diese Seite wurde gedreht, aber noch nicht gespeichert - die "
                "Bilddatei stimmt sonst nicht mit den Box-Koordinaten überein. "
                "Bitte zuerst 'Annotations-JSON speichern' klicken."
            )
        record, info = build_training_record(annotation, image_path, dataset_root, "mask")
        root = Path(dataset_root).resolve() if dataset_root.strip() else Path(image_path).resolve().parent
        root.mkdir(parents=True, exist_ok=True)
        output = root / "train.jsonl"
        records = [] if mode == "Datei ersetzen" else read_jsonl(output)
        image_key = record_key(record)
        records = [item for item in records if record_key(item) != image_key]
        records.append(record)
        output.write_text(
            "".join(json.dumps(x, ensure_ascii=False, separators=(",", ":")) + "\n" for x in records),
            encoding="utf-8",
            newline="\n",
        )
        masked_note = f", {info['masked']} nicht akzeptierte/offene Zeile(n) im Trainingsbild abgedeckt" if info["masked"] else ""
        return (
            annotation,
            str(output),
            f"JSONL exportiert: {output} | {len(records)} Datensätze, aktuelle Seite {info['lines']} akzeptierte Zeilen{masked_note}.",
            image_path,
        )
    except Exception as exc:
        raise gr.Error(f"JSONL-Export fehlgeschlagen: {exc}") from exc


# ---------------------------------------------------------------------------
# Reiter "Modellvergleich" (Logik in model_compare.py)
# ---------------------------------------------------------------------------

def _compare_api_url() -> str:
    return LLAMACPP_API if BACKEND == "llamacpp" else OLLAMA_API


def compare_page_choices(root: str) -> list[tuple[str, str]]:
    choices = []
    base = Path(root) if root and root.strip() else None
    for path in model_compare.find_annotation_files(root):
        try:
            count = len(model_compare.get_runs(model_compare.load_json(path)))
        except (OSError, json.JSONDecodeError):
            continue
        try:
            name = path.relative_to(base).as_posix() if base else path.name
        except ValueError:
            name = path.name
        name = name[: -len(model_compare.ANNOTATION_SUFFIX)]
        choices.append((f"{name}  ({count} {'Lauf' if count == 1 else 'Läufe'})", str(path)))
    return choices


def compare_run_choices(annotation_path: str | None) -> list[str]:
    if not annotation_path or not Path(annotation_path).is_file():
        return []
    return sorted(model_compare.get_runs(model_compare.load_json(annotation_path)))


def compare_refresh_pages(root: str, current: str | None):
    choices = compare_page_choices(root)
    values = [value for _, value in choices]
    value = current if current in values else (values[0] if values else None)
    runs = compare_run_choices(value)
    return (
        gr.update(choices=choices, value=value),
        gr.update(choices=runs, value=runs[0] if runs else None),
        gr.update(choices=runs, value=runs[1] if len(runs) > 1 else None),
        f"{len(choices)} geprüfte Seiten gefunden." if choices else
        "Keine geprüften Seiten (*_annotation.json) unter dem Dataset-Wurzelverzeichnis gefunden.",
    )


def compare_page_changed(annotation_path: str | None, label_a: str | None, label_b: str | None):
    runs = compare_run_choices(annotation_path)
    a = label_a if label_a in runs else (runs[0] if runs else None)
    b = label_b if label_b in runs and label_b != a else next((r for r in runs if r != a), None)
    return gr.update(choices=runs, value=a), gr.update(choices=runs, value=b)


def compare_page(annotation_path: str | None, label_a: str | None, label_b: str | None, threshold: float):
    if not annotation_path or not Path(annotation_path).is_file():
        empty = "<div class='mc-empty'>Bitte eine geprüfte Seite auswählen.</div>"
        return empty, empty, empty, empty
    path = Path(annotation_path)
    data = model_compare.load_json(path)
    runs = model_compare.get_runs(data)
    image_path = model_compare.image_for_annotation(path, data)
    threshold = float(threshold or model_compare.DEFAULT_IOU)

    selected = [label for label in (label_a, label_b) if label and label in runs]
    selected = list(dict.fromkeys(selected))
    metrics = [(label, model_compare.page_metrics(data, runs[label], threshold)) for label in selected]
    overlays = []
    for slot, color in enumerate(model_compare.RUN_COLORS):
        if slot < len(selected):
            label = selected[slot]
            overlays.append(model_compare.overlay_html(image_path, data, runs[label], color, label, threshold))
        elif slot == 0:
            overlays.append(model_compare.overlay_html(image_path, data, None, color, "Nur Referenz", threshold))
        else:
            overlays.append("")
    lines_html = model_compare.line_table_html(data, [(label, runs[label]) for label in selected], threshold)
    return model_compare.page_metrics_html(metrics), overlays[0], overlays[1], lines_html


def compare_summary(root: str, threshold: float, only_common: bool):
    rows, total, compared = model_compare.summarize(root, float(threshold or model_compare.DEFAULT_IOU), bool(only_common))
    if not rows:
        note = f"{total} geprüfte Seiten, aber noch keine Modellläufe. Unten ein Modell ausführen."
    else:
        note = (
            f"**{compared} von {total} geprüften Seiten verglichen**"
            + (" (nur Seiten, auf denen alle Läufe vorhanden sind)." if only_common else ".")
            + " Sortiert nach CER (Seite); niedriger ist besser."
        )
    return model_compare.summary_table(rows), note


def _run_label(model: str, label: str | None) -> str:
    return (label or "").strip() or model.strip()


def compare_run_on_page(
    annotation_path: str | None, model: str, label: str | None, max_side: float, ctx: float, think: bool,
    no_mmap: bool, label_a: str | None, label_b: str | None, threshold: float,
):
    if not annotation_path:
        raise gr.Error("Bitte zuerst eine geprüfte Seite auswählen.")
    if not model or not model.strip():
        raise gr.Error("Bitte ein Modell wählen.")
    opts = model_compare.run_options(
        model, int(max_side), int(ctx), bool(think), BACKEND, _compare_api_url(), OLLAMA_TIMEOUT, no_mmap=bool(no_mmap)
    )
    run_label = _run_label(model, label)
    try:
        run = model_compare.run_model_on_annotation(Path(annotation_path), opts, run_label)
    except requests.ConnectionError as exc:
        raise gr.Error(f"Modellserver unter {_compare_api_url()} nicht erreichbar.") from exc
    except Exception as exc:
        raise gr.Error(f"Modelllauf fehlgeschlagen: {exc}") from exc

    # Neuen Lauf direkt anzeigen: als B, falls A schon belegt ist, sonst als A.
    if not label_a or label_a == run_label:
        label_a, label_b = run_label, (label_b if label_b != run_label else None)
    else:
        label_b = run_label
    runs = compare_run_choices(annotation_path)
    return (
        gr.update(choices=runs, value=label_a),
        gr.update(choices=runs, value=label_b),
        *compare_page(annotation_path, label_a, label_b, threshold),
        f"Lauf '{run_label}' gespeichert: {len(run['lines'])} Zeilen in "
        f"{model_compare.fmt_duration(run['duration_s'])}.",
    )


def compare_run_on_all(
    root: str, model: str, label: str | None, max_side: float, ctx: float, think: bool, no_mmap: bool, force: bool
):
    """Generator: führt das Modell auf allen geprüften Seiten ohne diesen Lauf
    aus und meldet nach jeder Seite den Fortschritt (Abbrechen jederzeit
    möglich, fertige Seiten bleiben gespeichert)."""
    if not model or not model.strip():
        raise gr.Error("Bitte ein Modell wählen.")
    opts = model_compare.run_options(
        model, int(max_side), int(ctx), bool(think), BACKEND, _compare_api_url(), OLLAMA_TIMEOUT, no_mmap=bool(no_mmap)
    )
    run_label = _run_label(model, label)
    files = model_compare.find_annotation_files(root)
    todo = []
    for path in files:
        try:
            if force or run_label not in model_compare.get_runs(model_compare.load_json(path)):
                todo.append(path)
        except (OSError, json.JSONDecodeError):
            continue
    if not todo:
        yield f"Alle {len(files)} geprüften Seiten haben bereits einen Lauf '{run_label}'."
        return

    durations: list[float] = []
    errors: list[str] = []
    for index, path in enumerate(todo, 1):
        eta = ""
        if durations:
            eta = f" - Rest ca. {model_compare.fmt_duration(sum(durations) / len(durations) * (len(todo) - index + 1))}"
        yield f"[{index}/{len(todo)}] '{run_label}' läuft auf {path.name}{eta} …"
        try:
            run = model_compare.run_model_on_annotation(path, opts, run_label)
            durations.append(run["duration_s"])
        except Exception as exc:  # eine Seite darf den Stapel nicht abbrechen
            errors.append(f"{path.name}: {exc}")
    message = f"Fertig: {len(durations)} von {len(todo)} Seiten mit '{run_label}' verarbeitet"
    if durations:
        message += f", Ø {model_compare.fmt_duration(sum(durations) / len(durations))} pro Seite"
    message += "."
    if errors:
        message += f"\n{len(errors)} Fehler:\n" + "\n".join(errors[:10])
    yield message


def compare_delete_run(annotation_path: str | None, label: str | None, label_b: str | None, threshold: float):
    if not annotation_path or not label:
        raise gr.Error("Bitte Seite und Lauf A wählen.")
    model_compare.delete_run(Path(annotation_path), label)
    runs = compare_run_choices(annotation_path)
    label_a = label_b if label_b in runs else (runs[0] if runs else None)
    label_b = next((r for r in runs if r != label_a), None)
    return (
        gr.update(choices=runs, value=label_a),
        gr.update(choices=runs, value=label_b),
        *compare_page(annotation_path, label_a, label_b, threshold),
        f"Lauf '{label}' von dieser Seite entfernt.",
    )


def build_compare_tab(dataset_root: gr.Textbox):
    gr.Markdown(
        "Vergleicht Modelle auf bereits **geprüften** Seiten (`*_annotation.json` unter dem "
        "Dataset-Wurzelverzeichnis aus dem Reiter *Annotation*). Jeder Modelllauf wird in der "
        "geprüften Datei unter `model_runs` gespeichert; die geprüften Zeilen bleiben unverändert. "
        "Referenz sind nur **akzeptierte** Zeilen (✓); nicht akzeptierte und offene Zeilen werden "
        "ausgeklammert (grau) und nicht bewertet.\n\n"
        "- **CER/WER Seite**: Zeichen-/Wortfehlerrate über den ganzen Seitentext in Leserichtung - "
        "unabhängig davon, wie das Modell Zeilen in Boxen aufteilt. Unsicherheitsmarker `[?]` werden ignoriert.\n"
        "- **Zeilen-Recall/-Precision/F1, Ø IoU**: Boxen werden 1:1 über ihre Überlappung (IoU ≥ Schwelle) "
        "zugeordnet. Recall = gefundene Referenzzeilen, Precision = Modellzeilen mit passender Referenz.\n"
        "- **CER Zeilen**: Fehlerrate nur über die zugeordneten Zeilenpaare."
    )
    with gr.Row():
        iou_threshold = gr.Slider(0.1, 0.9, value=model_compare.DEFAULT_IOU, step=0.05, label="IoU-Schwelle für Box-Zuordnung")
        only_common = gr.Checkbox(value=True, label="Nur Seiten, auf denen alle Läufe vorhanden sind")

    gr.Markdown("### Übersicht über alle geprüften Seiten")
    summary_button = gr.Button("Übersicht berechnen", variant="primary")
    summary_note = gr.Markdown()
    summary_table = gr.Dataframe(headers=model_compare.SUMMARY_HEADERS, interactive=False, wrap=True)

    gr.Markdown("### Einzelne Seite vergleichen")
    with gr.Row():
        page = gr.Dropdown(label="Geprüfte Seite", choices=[], scale=4)
        pages_refresh = gr.Button("🔄 Seitenliste", scale=1, min_width=80)
    with gr.Row():
        run_a = gr.Dropdown(label="Lauf A (orange)", choices=[])
        run_b = gr.Dropdown(label="Lauf B (violett)", choices=[])
    with gr.Row():
        compare_button = gr.Button("Vergleichen", variant="primary")
        delete_button = gr.Button("Lauf A von dieser Seite löschen")
    page_metrics = gr.HTML()
    with gr.Row():
        overlay_a = gr.HTML()
        overlay_b = gr.HTML()
    gr.Markdown(
        "Zeilenvergleich: <del style='background:rgba(218,30,40,.22)'>rot</del> = in der Referenz, aber "
        "vom Modell nicht/falsch gelesen; <ins style='background:rgba(36,161,72,.25);text-decoration:none'>grün</ins> "
        "= stattdessen vom Modell geschrieben."
    )
    lines_html = gr.HTML()

    with gr.Accordion("Modell ausführen (Ergebnis wird in der geprüften Datei gespeichert)", open=True):
        gr.Markdown(
            "Für einen fairen Vergleich **beide** Modelle hier (oder per `model_compare.py run`) mit "
            "demselben Ablauf laufen lassen - der ursprüngliche Vorannotations-Text in der Datei "
            "enthält keine Original-Boxen mehr. Für denselben Modellnamen mit anderen Einstellungen "
            "ein eigenes Label vergeben (z.B. `qwen3-vl:4b@1536`)."
        )
        with gr.Row():
            run_model = gr.Dropdown(label="Modell", choices=model_dropdown_choices(), value=DEFAULT_MODEL, allow_custom_value=True, scale=3)
            run_label = gr.Textbox(label="Label (optional, Standard: Modellname)", scale=2)
        with gr.Row():
            run_max_side = gr.Number(label="Max. Bildseite (px)", value=1024, precision=0)
            run_ctx = gr.Number(label="Kontextgröße", value=8192, precision=0)
            run_think = gr.Checkbox(label="Denkmodus (thinking)", value=False)
            run_no_mmap = gr.Checkbox(
                label="Modell komplett in RAM laden (kein mmap; empfohlen für große Modelle)", value=True
            )
            run_force = gr.Checkbox(label="Vorhandene Läufe mit gleichem Label überschreiben (nur 'alle Seiten')", value=False)
        with gr.Row():
            run_page_button = gr.Button("Auf dieser Seite ausführen")
            run_all_button = gr.Button("Auf allen geprüften Seiten ausführen (fehlende)", variant="primary")
            run_stop_button = gr.Button("Abbrechen")
        compare_status = gr.Textbox(label="Status", interactive=False, lines=2)

    page_outputs = [page_metrics, overlay_a, overlay_b, lines_html]
    summary_button.click(compare_summary, [dataset_root, iou_threshold, only_common], [summary_table, summary_note])
    pages_refresh.click(compare_refresh_pages, [dataset_root, page], [page, run_a, run_b, compare_status])
    dataset_root.change(compare_refresh_pages, [dataset_root, page], [page, run_a, run_b, compare_status])
    page.change(compare_page_changed, [page, run_a, run_b], [run_a, run_b]).then(
        compare_page, [page, run_a, run_b, iou_threshold], page_outputs
    )
    compare_button.click(compare_page, [page, run_a, run_b, iou_threshold], page_outputs)
    iou_threshold.release(compare_page, [page, run_a, run_b, iou_threshold], page_outputs)
    delete_button.click(
        compare_delete_run, [page, run_a, run_b, iou_threshold], [run_a, run_b, *page_outputs, compare_status]
    )
    run_page_button.click(
        compare_run_on_page,
        [page, run_model, run_label, run_max_side, run_ctx, run_think, run_no_mmap, run_a, run_b, iou_threshold],
        [run_a, run_b, *page_outputs, compare_status],
    )
    run_all_event = run_all_button.click(
        compare_run_on_all,
        [dataset_root, run_model, run_label, run_max_side, run_ctx, run_think, run_no_mmap, run_force],
        compare_status,
    )
    run_stop_button.click(None, None, None, cancels=[run_all_event])
    return page, run_a, run_b, compare_status


def build_interface(initial_tab: str = "annotation") -> gr.Blocks:
    with gr.Blocks(title="Qwen Handschrift-Annotation") as app:
        annotation_state = gr.State({})
        image_state = gr.State("")
        selected_state = gr.State(-1)
        active_text_state = gr.State(None)
        tile_paths_state = gr.State([])
        # Seitenweit zuletzt gesetzter Box-Winkel (Dreh-Griff/Nudge-Knopf/
        # ±5°-Button), siehe sync_bbox_edit/rotate_box - lässt sich per
        # "Winkel übernehmen"-Knopf unter jeder Box mit einem Klick auf eine
        # andere Box anwenden.
        last_angle_state = gr.State(0.0)
        with gr.Tabs(selected=initial_tab):
            with gr.Tab("Annotation", id="annotation"):
                gr.Markdown("# Qwen-Vorannotation für Handschrift\nScan laden, vorannotieren, Texte korrigieren und als Annotation oder Trainings-JSONL speichern. Pixelkoordinaten sind führend und bleiben beim Import/Export unverändert.\nBoxen im Vorschaubild lassen sich per Maus verschieben (ziehen) und an den Eckpunkten skalieren; der kleine Griff über einer Box dreht sie frei (z.B. für eine schräg geschriebene Zeile); ein Klick auf eine Box blendet ihren Text darunter zum Bearbeiten ein.")
                with gr.Accordion("Dateien im Dataset (Bilder & PDFs, inkl. Unterordner) - grün = alle Boxen geprüft, gelb mit Zähler = teilweise geprüft", open=True):
                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("**Alle Dateien**")
                            file_list_all_html = gr.HTML(
                                render_file_list(DEFAULT_DATASET_ROOT, sync_target=FILE_SYNC_ALL),
                                elem_id="file-list-all-wrap",
                            )
                        with gr.Column():
                            gr.Markdown("**Nur vorannotiert / annotiert**")
                            file_list_flagged_html = gr.HTML(
                                render_file_list(DEFAULT_DATASET_ROOT, only_flagged=True, sync_target=FILE_SYNC_FLAGGED),
                                elem_id="file-list-flagged-wrap",
                            )
                    file_sync_all = gr.Textbox(elem_id=FILE_SYNC_ALL, visible=True, container=False)
                    file_sync_flagged = gr.Textbox(elem_id=FILE_SYNC_FLAGGED, visible=True, container=False)
                    refresh_files_button = gr.Button("Dateilisten aktualisieren")
                with gr.Row():
                    with gr.Column(scale=1):
                        # image_mode=None ist noetig, damit Gradio beim Zurueck-Einlesen des
                        # Pfads (preprocess) den Original-Pfad unveraendert durchreicht: mit dem
                        # Default "RGB" schreibt Gradio jedes Bild, das nicht exakt im PIL-Modus
                        # "RGB" vorliegt (z.B. Graustufen- oder Palette-PNGs aus Scans), still in
                        # ein neues Cache-Temp-File um - dadurch landete die gespeicherte
                        # Annotation nicht mehr neben der Originaldatei/preannotation.json.
                        image = gr.Image(label="Originalscan (Vorschau - Bild hierher ziehen zum Hochladen)", type="filepath", sources=["upload"], image_mode=None)
                        with gr.Row():
                            pdf_upload = gr.File(label="PDF-Scan (mehrseitig)", file_types=[".pdf"], type="filepath")
                            pdf_page = gr.Number(label="Seite", value=1, precision=0, minimum=1)
                        pdf_load = gr.Button("PDF-Seite laden")
                        with gr.Row():
                            split_button = gr.Button("Originalscan in Kacheln aufteilen")
                            tile_number = gr.Number(label="Kachel", value=1, precision=0, minimum=1)
                        load_tile_button = gr.Button("Kachel laden")
                        with gr.Row():
                            rotate_left_button = gr.Button("↺ 90° drehen")
                            rotate_right_button = gr.Button("↻ 90° drehen")
                        with gr.Row():
                            model = gr.Dropdown(
                                label="Ollama-Modell",
                                choices=model_dropdown_choices(),
                                value=DEFAULT_MODEL,
                                allow_custom_value=True,
                                scale=4,
                            )
                            refresh_models_button = gr.Button("🔄", scale=1, min_width=40)
                        context = gr.Number(label="Kontextgröße", value=DEFAULT_CONTEXT_SIZE, precision=0)
                        preannotate = gr.Button("Qwen-Vorannotation starten", variant="primary")
                        with gr.Row():
                            tesseract_lang = gr.Textbox(label="Tesseract-Sprache", value=tesseract_boxes.DEFAULT_LANG)
                            tesseract_psm = gr.Number(label="Tesseract PSM", value=tesseract_boxes.DEFAULT_PSM, precision=0)
                        tesseract_button = gr.Button("Tesseract-Boxen anwenden (Snap)")
                        undo_tesseract_button = gr.Button("Tesseract-Boxen rückgängig (zurück zur geladenen Annotation)")
                        existing = gr.File(label="Vorhandene Annotation", file_types=[".json"], type="filepath")
                        load = gr.Button("Scan und JSON laden")
                    with gr.Column(scale=2):
                        source_select = gr.Dropdown(
                            label="Angezeigt: geprüfte Annotation oder Vorannotation welches Modells",
                            info="Wechsel lädt die gewählte Fassung in Tabelle und Vorschau "
                            "(ungespeicherte Änderungen gehen verloren).",
                            choices=[],
                            interactive=True,
                        )
                        with gr.Row():
                            hide_accepted = gr.Checkbox(
                                label="Akzeptierte Boxen ausblenden",
                                info="Farben: grau = offen, grün = akzeptiert, rot = nicht akzeptiert, blau = ausgewählt.",
                                value=False,
                            )
                            hide_controls = gr.Checkbox(
                                label="Bedienelemente ausblenden (Übersicht)",
                                info="Nur die Rechtecke zeigen - ohne ✓/✗, Griffe, Nummern und Textfeld. "
                                "Klick wählt weiter aus, Text beim Überfahren mit der Maus.",
                                value=False,
                            )
                        preview = gr.HTML(EMPTY_PREVIEW_HTML, label="Zeilenboxen", elem_id="bbox-preview-wrap")
                        # visible=False would unmount this element in Gradio 6, breaking the
                        # JS->Python bridge from render_interactive_preview; hide via CSS instead
                        # so it stays queryable while a box is dragged/resized/edited.
                        bbox_sync = gr.Textbox(elem_id="bbox-sync-box", visible=True, container=False)
                        crop = gr.Image(label="Ausgewählte Zeile", type="pil", interactive=False, height=180)
                        line_text = gr.Textbox(
                            label="Text der ausgewählten Zeile",
                            info="Korrigieren, dann Enter = übernehmen, akzeptieren und nächste Zeile. "
                            "Übernommen wird der Text mit Enter oder den Knöpfen ✓/✗ direkt darunter.",
                            lines=1,
                            max_lines=4,
                            elem_id="line-text-box",
                        )
                        with gr.Row():
                            accept_button = gr.Button("✓ Zeile akzeptieren", variant="primary")
                            skip_button = gr.Button("⏭ Überspringen")
                            reject_button = gr.Button("✗ Zeile nicht akzeptieren", variant="stop")
                            accept_all_button = gr.Button("Alle offenen akzeptieren")
                        with gr.Row():
                            add_box_button = gr.Button("Box hinzufügen")
                            delete_box_button = gr.Button("Ausgewählte Box löschen")
                            no_text_button = gr.Button("Seite ohne sichtbaren Text (leere Seite)")
                        with gr.Row():
                            rotate_box_left_button = gr.Button("↺ Box -5°")
                            rotate_box_right_button = gr.Button("↻ Box +5°")
                        table = gr.Dataframe(headers=TABLE_HEADERS, datatype=TABLE_DATATYPES, column_count=(len(TABLE_HEADERS), "fixed"), column_widths=TABLE_COLUMN_WIDTHS, label="Text und Pixelboxen korrigieren (Spalte \"Prüfung\": ✓ ok / ✗ nein / offen - auch ok/x tippbar)", interactive=True, wrap=True)
                with gr.Row():
                    refresh_button = gr.Button("Änderungen übernehmen")
                    save_button = gr.Button("Annotations-JSON speichern")
                with gr.Row():
                    dataset_root = gr.Textbox(label="Dataset-Wurzelverzeichnis", value=DEFAULT_DATASET_ROOT, placeholder=r"C:\Handschrift-Dataset")
                    export_mode = gr.Radio(["An Datei anhängen / Seite aktualisieren", "Datei ersetzen"], value="An Datei anhängen / Seite aktualisieren", label="Exportmodus")
                    export_button = gr.Button("Als Qwen-Trainings-JSONL exportieren", variant="primary")
                with gr.Row():
                    annotation_file = gr.File(label="Annotations-JSON")
                    training_file = gr.File(label="Qwen-Trainings-JSONL")
                status = gr.Textbox(label="Status", interactive=False)
            with gr.Tab("Modellvergleich", id="vergleich"):
                compare_page_dd, compare_run_a, compare_run_b, compare_status = build_compare_tab(dataset_root)

        # .upload() (not .change()) is required here: only a genuine manual
        # upload should discard the previous annotation. select_dataset_file /
        # load_pdf_page_and_autoload / load_tile_and_autoload also assign to
        # `image` programmatically, which would otherwise retrigger this and
        # immediately wipe out the annotation they just loaded.
        image.upload(
            reset_image_state,
            [image],
            [image_state, annotation_state, selected_state, preview, crop, table, active_text_state, existing, status],
        )
        pdf_load.click(
            load_pdf_page_and_autoload,
            [pdf_upload, pdf_page],
            [image, image_state, annotation_state, selected_state, preview, crop, table, active_text_state, existing, status],
        )
        split_button.click(split_into_tiles, [image_state], [tile_paths_state, status])
        load_tile_button.click(
            load_tile_and_autoload,
            [tile_paths_state, tile_number],
            [image, image_state, annotation_state, selected_state, preview, crop, table, active_text_state, existing, status],
        )
        preannotate.click(start_preannotation, [image_state, model, context], [annotation_state, image_state, selected_state, preview, crop, table, active_text_state, status]).then(
            lambda path, name: refresh_annotation_sources(path, f"pre:{(name or '').strip()}"),
            [image_state, model],
            source_select,
        )
        # Auswahlfeld bei jedem Bildwechsel neu füllen; .input (nicht .change),
        # damit nur eine echte Auswahl durch den Benutzer neu lädt, nicht das
        # programmgesteuerte Setzen des Werts.
        image_state.change(refresh_annotation_sources, [image_state], source_select)
        source_select.input(
            load_annotation_source,
            [image_state, source_select],
            [annotation_state, image_state, selected_state, preview, crop, table, active_text_state, status],
        )
        tesseract_button.click(
            apply_tesseract_boxes,
            [table, annotation_state, image_state, tesseract_lang, tesseract_psm],
            [annotation_state, preview, crop, table, selected_state, active_text_state, status],
        )
        undo_tesseract_button.click(
            load_annotation,
            [image_state, existing],
            [annotation_state, image_state, selected_state, preview, crop, table, active_text_state, status],
        )
        load.click(load_annotation, [image_state, existing], [annotation_state, image_state, selected_state, preview, crop, table, active_text_state, status])
        table.select(
            select_row,
            [table, annotation_state, image_state],
            [annotation_state, preview, crop, selected_state, active_text_state, status],
            show_progress="hidden",
        )
        refresh_button.click(refresh, [table, annotation_state, image_state, selected_state], [annotation_state, preview, crop, selected_state, active_text_state, status])
        rotate_left_button.click(
            rotate_page_left,
            [table, annotation_state, image_state, selected_state],
            [annotation_state, preview, crop, table, selected_state, active_text_state, status],
        )
        rotate_right_button.click(
            rotate_page_right,
            [table, annotation_state, image_state, selected_state],
            [annotation_state, preview, crop, table, selected_state, active_text_state, status],
        )
        add_box_button.click(
            add_box,
            [table, annotation_state, image_state],
            [annotation_state, preview, crop, table, selected_state, active_text_state, status],
        )
        delete_box_button.click(
            delete_box,
            [table, annotation_state, image_state, selected_state],
            [annotation_state, preview, crop, table, selected_state, active_text_state, status],
        )
        no_text_button.click(
            mark_no_text,
            [table, annotation_state, image_state],
            [annotation_state, preview, crop, table, selected_state, active_text_state, status],
        )
        # Korrekturfeld unter dem Zeilenausschnitt: Der Text wird ausdrücklich
        # mit Enter oder ✓/✗ übernommen (nicht beim Verlassen des Felds - das
        # könnte sich mit einem gleichzeitigen Klick auf eine andere Box oder
        # Tabellenzeile überschneiden und die Korrektur überschreiben).
        focus_line_text = "() => { setTimeout(() => { const t = document.querySelector('#line-text-box textarea, #line-text-box input'); if (t) { t.focus(); const n = t.value.length; t.setSelectionRange(n, n); } }, 150); }"
        review_outputs = [annotation_state, preview, crop, table, selected_state, active_text_state, status, line_text]
        review_inputs = [line_text, table, annotation_state, image_state, selected_state, hide_accepted]
        # Ausblenden rein im Browser (CSS-Klasse am <body>), ohne Neuaufbau der Vorschau.
        hide_accepted.change(
            None, [hide_accepted], None,
            js="(v) => { document.body.classList.toggle('qb-hide-accepted', !!v); return []; }",
        )
        # Reine Anzeige-Umschaltung im Browser, ohne Server-Runde.
        hide_controls.change(
            None, [hide_controls], None,
            js="(v) => { document.body.classList.toggle('qb-hide-controls', !!v); return []; }",
        )
        for review_button, review_fn in (
            (accept_button, accept_selected),
            (skip_button, skip_selected),
            (reject_button, reject_selected),
            (accept_all_button, accept_all_open),
        ):
            review_button.click(
                review_fn, review_inputs, review_outputs,
                show_progress="hidden", concurrency_id="line_edit",
            ).then(None, None, None, js=focus_line_text)
        line_text.submit(
            accept_selected, review_inputs, review_outputs, show_progress="hidden", concurrency_id="line_edit",
        ).then(None, None, None, js=focus_line_text)
        # Korrekturfeld bei jeder neuen Auswahl/Änderung mit dem aktuellen Text füllen.
        selected_state.change(refresh_line_text, [annotation_state, selected_state, line_text], line_text, show_progress="hidden")
        annotation_state.change(refresh_line_text, [annotation_state, selected_state, line_text], line_text, show_progress="hidden")
        rotate_box_left_button.click(
            rotate_box_left,
            [table, annotation_state, image_state, selected_state],
            [annotation_state, preview, crop, table, selected_state, active_text_state, status, last_angle_state],
        )
        rotate_box_right_button.click(
            rotate_box_right,
            [table, annotation_state, image_state, selected_state],
            [annotation_state, preview, crop, table, selected_state, active_text_state, status, last_angle_state],
        )
        # .change() (not .input()) is required here: Gradio only fires .input()
        # for events it recognizes as genuine keystrokes, so the synthetic
        # DOM events our JS dispatches after a drag/resize/text-edit only
        # reach the backend through .change().
        bbox_sync.change(
            sync_bbox_edit,
            [bbox_sync, annotation_state, image_state, selected_state, active_text_state, last_angle_state],
            [annotation_state, preview, crop, table, selected_state, active_text_state, status, last_angle_state],
            # Kein Ausgrauen/Ladebalken über Vorschau und Tabelle bei jedem
            # Klick auf eine Box - das wirkte wie ein Einfrieren der Seite.
            show_progress="hidden",
        )
        save_button.click(
            save_annotation,
            [table, annotation_state, image_state, model, dataset_root],
            [annotation_state, annotation_file, status, image_state, file_list_all_html, file_list_flagged_html],
        ).then(
            lambda path: refresh_annotation_sources(path, SOURCE_ANNOTATION), [image_state], source_select
        )
        export_button.click(export_jsonl, [table, annotation_state, image_state, dataset_root, export_mode], [annotation_state, training_file, status, image_state])
        refresh_files_button.click(render_file_lists, [dataset_root, image_state], [file_list_all_html, file_list_flagged_html])
        refresh_models_button.click(refresh_ollama_models, None, model)
        dataset_root.change(render_file_lists, [dataset_root, image_state], [file_list_all_html, file_list_flagged_html])
        file_sync_all.change(
            select_dataset_file,
            [file_sync_all, dataset_root],
            [
                image, image_state, annotation_state, selected_state, preview, crop, table, active_text_state, existing,
                pdf_upload, pdf_page, file_list_all_html, file_list_flagged_html, status,
            ],
        )
        file_sync_flagged.change(
            select_dataset_file,
            [file_sync_flagged, dataset_root],
            [
                image, image_state, annotation_state, selected_state, preview, crop, table, active_text_state, existing,
                pdf_upload, pdf_page, file_list_all_html, file_list_flagged_html, status,
            ],
        )
        app.load(render_file_lists, [dataset_root, image_state], [file_list_all_html, file_list_flagged_html])
        app.load(None, None, None, js=BBOX_JS)
        app.load(
            compare_refresh_pages,
            [dataset_root, compare_page_dd],
            [compare_page_dd, compare_run_a, compare_run_b, compare_status],
        )
    return app


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=7860, type=int)
    parser.add_argument("--share", action="store_true")
    parser.add_argument(
        "--tab",
        choices=("annotation", "vergleich"),
        default="annotation",
        help="Reiter, der beim Start geöffnet ist ('vergleich' = Modellvergleich)",
    )
    args = parser.parse_args()
    _prune_preview_dir()
    build_interface(args.tab).launch(
        allowed_paths=[str(PREVIEW_DIR)],
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        inbrowser=True,
        head=BBOX_STYLE + model_compare.COMPARE_STYLE,
        **launch_speed_options(),
    )


def launch_speed_options() -> dict[str, Any]:
    """Gradio 6 schreibt sonst bei JEDEM Ereignis (Seitenwahl, Boxklick,
    Akzeptieren ...) den kompletten Verlauf aller Ein-/Ausgaben - hier samt
    Vorschau-HTML und Tabelle, bis zu 100 Läufe, schnell mehrere MB - neu in
    den localStorage des Browsers. Das läuft synchron im Browser-Hauptthread;
    ist der Speicher voll, kürzt Gradio den Verlauf Eintrag für Eintrag und
    serialisiert jedes Mal neu. Das Fenster friert dann sekundenlang ein, auch
    Tippen und Markieren im Textfeld hängen. run_history=False schaltet das ab
    und löscht den bisher gespeicherten Verlauf im Browser."""
    os.environ.setdefault("GRADIO_RUN_HISTORY", "False")
    try:
        if "run_history" in inspect.signature(gr.Blocks.launch).parameters:
            return {"run_history": False}
    except (TypeError, ValueError):
        pass
    return {}


if __name__ == "__main__":
    main()
