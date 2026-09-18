from __future__ import annotations

import argparse
import base64
import copy
import io
import json
import re
import shutil
import uuid
from pathlib import Path
from typing import Any

import gradio as gr
import requests
from PIL import Image, ImageOps

import pdf_utils
import tiling

OLLAMA_API = "http://127.0.0.1:11434/api/chat"
DEFAULT_MODEL = "qwen3-vl:4b"
DEFAULT_CONTEXT_SIZE = 4096
DEFAULT_DATASET_ROOT = r"C:\test\handwriting_ocr\pictures_for_OCR"
CONFIDENCE_VALUES = {"high", "medium", "low"}
TABLE_HEADERS = ["ID", "Text", "Konfidenz", "x1_px", "y1_px", "x2_px", "y2_px"]

PREANNOTATION_PROMPT = """
Analysiere diese gescannte Seite mit deutscher Handschrift.
Erkenne alle handschriftlichen Textzeilen in natürlicher Leserichtung.
Gib ausschließlich gültiges JSON in diesem Format zurück:

{"lines": [{"bbox_1000": [x1, y1, x2, y2], "text": "erkannter Text", "confidence": "high"}]}

Die Koordinaten müssen auf 0 bis 1000 normalisiert sein. Jede Textzeile erhält
eine eigene, möglichst eng anliegende Box. Ergänze keine unsichtbaren Wörter.
Zahlen, Namen und Einheiten nicht plausibilisieren. Unleserliches als
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
0 bis 1000 normalisiert. Sortiere in natürlicher Leserichtung, ergänze keine
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
    image = open_scan(image_path)
    width, height = image.size

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

        try:
            pixel_box = validate_pixel_bbox(line.get("bbox_pixels"), width, height)
        except (TypeError, ValueError):
            pixel_box = bbox_1000_to_pixels(line["bbox_1000"], width, height)

        line["bbox_pixels"] = pixel_box
        line["bbox_1000"] = pixels_to_bbox_1000(pixel_box, width, height)

    result["schema_version"] = "1.3"
    result["coordinate_system"] = "original_pixels"
    return result


def run_qwen(image_path: str, model: str, context_size: int) -> dict[str, Any]:
    path = Path(image_path)
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = {
        "model": model.strip(),
        "messages": [{
            "role": "user",
            "content": PREANNOTATION_PROMPT,
            "images": [encode_image(path)],
        }],
        "stream": False,
        "format": "json",
        "options": {"temperature": 0, "num_ctx": int(context_size)},
    }
    response = requests.post(OLLAMA_API, json=payload, timeout=1800)
    if not response.ok:
        raise RuntimeError(f"Ollama-Fehler {response.status_code}: {response.text}")
    raw = response.json().get("message", {}).get("content", "")
    if not raw:
        raise RuntimeError("Ollama hat keine Textantwort geliefert.")
    return normalize_annotation(extract_json(raw))


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
EMPTY_PREVIEW_HTML = "<div class='bbox-empty'>Kein Bild geladen.</div>"

BBOX_STYLE = """
<style>
.bbox-canvas { position: relative; display: inline-block; max-width: 100%; line-height: 0; user-select: none; }
.bbox-image { display: block; width: 100%; height: auto; max-width: 100%; pointer-events: none; }
.bbox-box { position: absolute; border-style: solid; border-width: 2px; box-sizing: border-box; cursor: move; }
.bbox-label { position: absolute; top: -22px; left: -2px; color: white; font-size: 12px; font-weight: bold; padding: 1px 5px; border-radius: 3px; white-space: nowrap; }
.bbox-handle { position: absolute; width: 10px; height: 10px; background: white; border: 2px solid #0067C0; border-radius: 50%; }
.bbox-handle-nw { top: -6px; left: -6px; cursor: nwse-resize; }
.bbox-handle-ne { top: -6px; right: -6px; cursor: nesw-resize; }
.bbox-handle-sw { bottom: -6px; left: -6px; cursor: nesw-resize; }
.bbox-handle-se { bottom: -6px; right: -6px; cursor: nwse-resize; }
.bbox-text { position: absolute; z-index: 20; min-width: 220px; max-width: 420px; }
.bbox-textarea { width: 100%; min-height: 60px; font-size: 14px; padding: 6px; border: 2px solid #0067C0; border-radius: 4px; box-shadow: 0 2px 8px rgba(0,0,0,.25); resize: vertical; font-family: inherit; box-sizing: border-box; }
.bbox-empty { padding: 40px; text-align: center; color: #888; }
#bbox-sync-box { position: absolute !important; width: 1px !important; height: 1px !important; overflow: hidden !important; opacity: 0 !important; pointer-events: none !important; margin: 0 !important; padding: 0 !important; border: 0 !important; }
</style>
"""

BBOX_JS = """
() => {
  function qbClamp(v, lo, hi) { return Math.max(lo, Math.min(hi, v)); }

  function qbSync(payload) {
    const box = document.querySelector('#bbox-sync-box textarea, #bbox-sync-box input');
    if (!box) return;
    box.value = JSON.stringify(payload);
    box.dispatchEvent(new Event('input', { bubbles: true }));
    box.dispatchEvent(new Event('change', { bubbles: true }));
  }

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
    function onMove(e) {
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

  window.qbCommitText = function(id) {
    const el = document.getElementById('textarea-' + id);
    if (!el) return;
    qbSync({ type: 'text', id: id, text: el.value });
  };
}
"""


def escape_html(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


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
    image = open_scan(image_path)
    width, height = image.size
    data_uri = encode_display_image(image)

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
        color = SELECTED_COLOR if index == selected else CONFIDENCE_COLORS.get(line.get("confidence"), "#DA1E28")
        text = str(line.get("text_corrected", ""))

        handles = "".join(
            f"<div class='bbox-handle bbox-handle-{corner}' "
            f"onmousedown=\"window.qbStartResize(event,'{line_id}','{corner}')\"></div>"
            for corner in ("nw", "ne", "sw", "se")
        )

        text_panel = ""
        if active_text_id == line_id:
            text_top = top + box_height
            text_panel = (
                f"<div class='bbox-text' id='text-{line_id}' "
                f"style='left:{left:.3f}%; top:{text_top:.3f}%;'>"
                f"<textarea id='textarea-{line_id}' class='bbox-textarea' autofocus "
                f"onblur=\"window.qbCommitText('{line_id}')\">{escape_html(text)}</textarea>"
                f"</div>"
            )

        parts.append(
            f"<div class='bbox-box' id='box-{line_id}' "
            f"style='left:{left:.3f}%; top:{top:.3f}%; width:{box_width:.3f}%; height:{box_height:.3f}%; border-color:{color};' "
            f"onmousedown=\"window.qbStartDrag(event,'{line_id}')\">"
            f"<span class='bbox-label' style='background:{color};'>{index + 1}</span>"
            f"{handles}"
            f"</div>"
            f"{text_panel}"
        )

    return (
        f"<div class='bbox-canvas' id='bbox-canvas' data-width='{width}' data-height='{height}'>"
        f"<img class='bbox-image' src='{data_uri}' draggable='false' />"
        f"{''.join(parts)}"
        f"</div>"
    )


def crop_line(image_path: str, annotation: dict[str, Any], selected: int) -> Image.Image | None:
    lines = annotation.get("lines", [])
    if selected < 0 or selected >= len(lines):
        return None
    image = open_scan(image_path)
    x1, y1, x2, y2 = line_to_pixels(lines[selected], annotation, image.width, image.height)
    mx, my = max(10, image.width // 70), max(8, image.height // 150)
    return image.crop((max(0, x1 - mx), max(0, y1 - my), min(image.width, x2 + mx), min(image.height, y2 + my)))


def annotation_to_table(annotation: dict[str, Any]) -> list[list[Any]]:
    return [
        [line["id"], line["text_corrected"], line["confidence"], *line["bbox_pixels"]]
        for line in annotation.get("lines", [])
    ]


def table_to_annotation(table: Any, annotation: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(annotation)
    if table is None:
        return result
    if hasattr(table, "values"):
        table = table.values.tolist()

    image = result.get("image", {})
    width = int(image.get("width", 0))
    height = int(image.get("height", 0))
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
        updated = copy.deepcopy(previous)
        updated.update({
            "id": str(row[0] or f"line_{index + 1:04d}"),
            "bbox_pixels": pixel_box,
            "bbox_1000": pixels_to_bbox_1000(pixel_box, width, height),
            "text_predicted": predicted,
            "text_corrected": corrected,
            "confidence": confidence,
            "status": "corrected" if corrected != predicted else "confirmed",
        })
        lines.append(updated)

    result["lines"] = lines
    result["coordinate_system"] = "original_pixels"
    result["schema_version"] = "1.3"
    return result


def start_preannotation(image_path: str | None, model: str, context: int):
    if not image_path:
        raise gr.Error("Bitte zuerst einen Scan auswählen.")
    try:
        annotation = add_metadata(run_qwen(image_path, model, context), image_path, model)
        selected = 0 if annotation["lines"] else -1
        return (
            annotation,
            image_path,
            selected,
            render_interactive_preview(image_path, annotation, selected, None),
            crop_line(image_path, annotation, selected),
            annotation_to_table(annotation),
            None,
            f"{len(annotation['lines'])} Zeilen erkannt.",
        )
    except requests.ConnectionError as exc:
        raise gr.Error("Ollama ist unter 127.0.0.1:11434 nicht erreichbar.") from exc
    except Exception as exc:
        raise gr.Error(str(exc)) from exc


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


def sync_image_state(image_path: str | None):
    """Hält image_state synchron, sobald sich der geladene Scan ändert (Upload,
    PDF-Seite oder Kachel), und verwirft die Annotationsanzeige des vorherigen
    Bildes, damit sie nicht versehentlich unter dem neuen Bildpfad gespeichert wird.
    """
    return image_path or "", {}, -1, EMPTY_PREVIEW_HTML, None, [], None


def load_annotation(image_path: str | None, json_path: str | None):
    if not image_path or not json_path:
        raise gr.Error("Bitte Scan und Annotations-JSON auswählen.")
    try:
        data = json.loads(Path(json_path).read_text(encoding="utf-8"))
        if not data.get("schema_version"):
            data = normalize_annotation(data)

        stored_image = data.get("image")
        if isinstance(stored_image, dict) and stored_image.get("width") and stored_image.get("height"):
            actual_width, actual_height = open_scan(image_path).size
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
            f"{len(data.get('lines', []))} Zeilen geladen.",
        )
    except Exception as exc:
        raise gr.Error(f"Laden fehlgeschlagen: {exc}") from exc


def select_row(table: Any, annotation: dict[str, Any], image_path: str, evt: gr.SelectData):
    try:
        annotation = table_to_annotation(table, annotation)
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
        annotation = table_to_annotation(table, annotation)
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


def sync_bbox_edit(
    payload_json: str,
    annotation: dict[str, Any],
    image_path: str,
    selected: int,
    active_text: str | None,
):
    """Wird vom versteckten Sync-Textfeld ausgelöst, sobald im Vorschau-Overlay
    eine Box verschoben/skaliert oder ihr Text bearbeitet/aufgeklappt wurde.
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
        annotation = add_metadata(table_to_annotation(table, annotation), image_path, model)
        output = Path(image_path).with_name(Path(image_path).stem + "_annotation.json")
        output.write_text(json.dumps(annotation, ensure_ascii=False, indent=2), encoding="utf-8")
        return annotation, str(output), f"Annotation gespeichert: {output}", image_path
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
    lines = []
    for line in annotation.get("lines", []):
        text = str(line.get("text_corrected", "")).strip()
        if text:
            lines.append({"bbox_1000": validate_bbox(line["bbox_1000"]), "text": text})
    lines.sort(key=lambda x: (x["bbox_1000"][1], x["bbox_1000"][0]))
    return {"lines": lines}


def training_record(annotation: dict[str, Any], image_path: str, dataset_root: str) -> dict[str, Any]:
    answer = training_answer(annotation)
    if not answer["lines"]:
        raise ValueError("Keine Zeilen mit korrigiertem Text vorhanden.")
    return {
        "messages": [
            {"role": "user", "content": [
                {"type": "image", "image": relative_image_path(image_path, dataset_root)},
                {"type": "text", "text": TRAINING_PROMPT},
            ]},
            {"role": "assistant", "content": [
                {"type": "text", "text": json.dumps(answer, ensure_ascii=False, separators=(",", ":"))}
            ]},
        ]
    }


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
        annotation = table_to_annotation(table, annotation)
        record = training_record(annotation, image_path, dataset_root)
        root = Path(dataset_root).resolve() if dataset_root.strip() else Path(image_path).resolve().parent
        root.mkdir(parents=True, exist_ok=True)
        output = root / "train.jsonl"
        records = [] if mode == "Datei ersetzen" else read_jsonl(output)
        image_key = record_image(record)
        records = [item for item in records if record_image(item) != image_key]
        records.append(record)
        output.write_text(
            "".join(json.dumps(x, ensure_ascii=False, separators=(",", ":")) + "\n" for x in records),
            encoding="utf-8",
            newline="\n",
        )
        return annotation, str(output), f"JSONL exportiert: {output} | {len(records)} Datensätze, aktuelle Seite {len(training_answer(annotation)['lines'])} Zeilen.", image_path
    except Exception as exc:
        raise gr.Error(f"JSONL-Export fehlgeschlagen: {exc}") from exc


def build_interface() -> gr.Blocks:
    with gr.Blocks(title="Qwen Handschrift-Annotation") as app:
        annotation_state = gr.State({})
        image_state = gr.State("")
        selected_state = gr.State(-1)
        active_text_state = gr.State(None)
        tile_paths_state = gr.State([])
        gr.Markdown("# Qwen-Vorannotation für Handschrift\nScan laden, vorannotieren, Texte korrigieren und als Annotation oder Trainings-JSONL speichern. Pixelkoordinaten sind führend und bleiben beim Import/Export unverändert.\nBoxen im Vorschaubild lassen sich per Maus verschieben (ziehen) und an den Eckpunkten skalieren; ein Klick auf eine Box blendet ihren Text darunter zum Bearbeiten ein.")
        with gr.Row():
            with gr.Column(scale=1):
                image = gr.Image(label="Originalscan", type="filepath", sources=["upload"])
                with gr.Row():
                    pdf_upload = gr.File(label="PDF-Scan (mehrseitig)", file_types=[".pdf"], type="filepath")
                    pdf_page = gr.Number(label="Seite", value=1, precision=0, minimum=1)
                pdf_load = gr.Button("PDF-Seite laden")
                with gr.Row():
                    split_button = gr.Button("Originalscan in Kacheln aufteilen")
                    tile_number = gr.Number(label="Kachel", value=1, precision=0, minimum=1)
                load_tile_button = gr.Button("Kachel laden")
                model = gr.Textbox(label="Ollama-Modell", value=DEFAULT_MODEL)
                context = gr.Number(label="Kontextgröße", value=DEFAULT_CONTEXT_SIZE, precision=0)
                preannotate = gr.Button("Qwen-Vorannotation starten", variant="primary")
                existing = gr.File(label="Vorhandene Annotation", file_types=[".json"], type="filepath")
                load = gr.Button("Scan und JSON laden")
            with gr.Column(scale=2):
                preview = gr.HTML(EMPTY_PREVIEW_HTML, label="Zeilenboxen", elem_id="bbox-preview-wrap")
                # visible=False would unmount this element in Gradio 6, breaking the
                # JS->Python bridge from render_interactive_preview; hide via CSS instead
                # so it stays queryable while a box is dragged/resized/edited.
                bbox_sync = gr.Textbox(elem_id="bbox-sync-box", visible=True, container=False)
                crop = gr.Image(label="Ausgewählte Zeile", type="pil", interactive=False, height=180)
                table = gr.Dataframe(headers=TABLE_HEADERS, datatype=["str", "str", "str", "number", "number", "number", "number"], column_count=(7, "fixed"), label="Text und Pixelboxen korrigieren", interactive=True, wrap=True)
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

        image.change(sync_image_state, [image], [image_state, annotation_state, selected_state, preview, crop, table, active_text_state])
        pdf_load.click(load_pdf_page, [pdf_upload, pdf_page], [image, status])
        split_button.click(split_into_tiles, [image], [tile_paths_state, status])
        load_tile_button.click(load_tile, [tile_paths_state, tile_number], [image, status])
        preannotate.click(start_preannotation, [image, model, context], [annotation_state, image_state, selected_state, preview, crop, table, active_text_state, status])
        load.click(load_annotation, [image, existing], [annotation_state, image_state, selected_state, preview, crop, table, active_text_state, status])
        table.select(select_row, [table, annotation_state, image_state], [annotation_state, preview, crop, selected_state, active_text_state, status])
        refresh_button.click(refresh, [table, annotation_state, image_state, selected_state], [annotation_state, preview, crop, selected_state, active_text_state, status])
        # .change() (not .input()) is required here: Gradio only fires .input()
        # for events it recognizes as genuine keystrokes, so the synthetic
        # DOM events our JS dispatches after a drag/resize/text-edit only
        # reach the backend through .change().
        bbox_sync.change(
            sync_bbox_edit,
            [bbox_sync, annotation_state, image_state, selected_state, active_text_state],
            [annotation_state, preview, crop, table, selected_state, active_text_state, status],
        )
        save_button.click(save_annotation, [table, annotation_state, image_state, model, dataset_root], [annotation_state, annotation_file, status, image_state])
        export_button.click(export_jsonl, [table, annotation_state, image_state, dataset_root, export_mode], [annotation_state, training_file, status, image_state])
        app.load(None, None, None, js=BBOX_JS)
    return app


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=7860, type=int)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()
    build_interface().launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        inbrowser=True,
        head=BBOX_STYLE,
    )


if __name__ == "__main__":
    main()
