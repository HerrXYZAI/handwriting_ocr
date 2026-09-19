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
import tesseract_boxes
import tiling

OLLAMA_API = "http://127.0.0.1:11434/api/chat"
DEFAULT_MODEL = "qwen3-vl:4b"
DEFAULT_CONTEXT_SIZE = 4096
DEFAULT_DATASET_ROOT = r"C:\test\handwriting_ocr\pictures_for_OCR"
CONFIDENCE_VALUES = {"high", "medium", "low"}
TABLE_HEADERS = ["ID", "Text", "Konfidenz", "x1_px", "y1_px", "x2_px", "y2_px"]
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".gif", ".webp"}
DATASET_FILE_EXTENSIONS = IMAGE_EXTENSIONS | {".pdf"}

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
NO_TEXT_STATUS = "no_text"
NO_TEXT_COLOR = "#8D8D8D"
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
        entries.append({
            "path": str(path.resolve()),
            "relative": path.relative_to(root).as_posix(),
            "is_pdf": path.suffix.lower() == ".pdf",
            "annotated": path.with_name(path.stem + "_annotation.json").is_file(),
            "preannotated": path.with_name(path.stem + "_preannotation.json").is_file(),
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
        if entry["annotated"]:
            classes.append("annotated")
            badge = "<span class='file-badge' title='Annotiert / freigegeben'>&#10003;</span>"
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
        if index == selected:
            color = SELECTED_COLOR
        elif line.get("status") == NO_TEXT_STATUS:
            color = NO_TEXT_COLOR
        else:
            color = CONFIDENCE_COLORS.get(line.get("confidence"), "#DA1E28")
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


def apply_tesseract_boxes(
    table: Any,
    annotation: dict[str, Any],
    image_path: str | None,
    lang: str,
    psm: float | int,
):
    """Ruft den Tesseract-Docker-Dienst ab und wendet Snap+Fill auf die
    aktuellen Zeilen an (siehe tesseract_boxes.detect_and_adjust)."""
    if not image_path:
        raise gr.Error("Bitte zuerst einen Scan laden.")
    try:
        annotation = table_to_annotation(table, annotation)
    except Exception as exc:
        raise gr.Error(str(exc)) from exc

    image = annotation.get("image", {})
    width = int(image.get("width", 0))
    height = int(image.get("height", 0))
    if width <= 0 or height <= 0:
        raise gr.Error("Die Bildgröße fehlt. Annotation zuerst mit einem Bild laden.")

    try:
        new_lines, status = tesseract_boxes.detect_and_adjust(
            image_path, annotation.get("lines", []), width, height, lang=lang, psm=int(psm)
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
        image_path, image_state_path, annotation, selected, preview, crop_img, table_rows, active_text, existing_path,
        f"{page_status} {load_status}",
    )


def load_tile_and_autoload(tile_paths: list[str], tile_number: float | int | None):
    image_path, tile_status = load_tile(tile_paths, tile_number)
    annotation, image_state_path, selected, preview, crop_img, table_rows, active_text, existing_path, load_status = (
        auto_load_existing_annotation(image_path)
    )
    return (
        image_path, image_state_path, annotation, selected, preview, crop_img, table_rows, active_text, existing_path,
        f"{tile_status} {load_status}",
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
        image_path,
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
        annotation = table_to_annotation(table, annotation)
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
        annotation = table_to_annotation(table, annotation)
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


def mark_no_text(table: Any, annotation: dict[str, Any], image_path: str, selected: int):
    """Markiert die ausgewählte Zeile als 'kein sichtbarer Text' (z.B. Stempel,
    Wasserzeichen, leerer Rand): Text wird geleert, Status auf no_text gesetzt,
    Box in der Vorschau grau statt konfidenzfarben dargestellt. Wird der Text
    später wieder bearbeitet, setzen die normalen Textbearbeitungspfade den
    Status automatisch auf corrected/confirmed zurück - kein separater
    "Entmarkieren"-Button nötig. Da training_answer() nur Zeilen mit
    nicht-leerem text_corrected exportiert, wird die Zeile dadurch zugleich
    vom Training ausgeschlossen.
    """
    if not image_path:
        raise gr.Error("Kein Bild geladen.")
    try:
        annotation = table_to_annotation(table, annotation)
    except Exception as exc:
        raise gr.Error(str(exc)) from exc

    lines = annotation.get("lines", [])
    index = int(selected)
    if index < 0 or index >= len(lines):
        raise gr.Error("Bitte zuerst eine Zeile auswählen (Tabellenzeile oder Box anklicken).")

    line = lines[index]
    line["text_corrected"] = ""
    line["status"] = NO_TEXT_STATUS
    annotation["lines"] = lines
    line_id = line.get("id") or f"Zeile {index + 1}"
    return (
        annotation,
        render_interactive_preview(image_path, annotation, index, None),
        crop_line(image_path, annotation, index),
        annotation_to_table(annotation),
        index,
        None,
        f"{line_id} als 'kein sichtbarer Text' markiert (grau, vom Training ausgeschlossen).",
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
        all_html, flagged_html = render_file_lists(dataset_root, image_path)
        return annotation, str(output), f"Annotation gespeichert: {output}", image_path, all_html, flagged_html
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
        with gr.Accordion("Dateien im Dataset (Bilder & PDFs, inkl. Unterordner) - annotierte/freigegebene Dateien grün", open=True):
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
                image = gr.Image(label="Originalscan", type="filepath", sources=["upload"], image_mode=None)
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
                with gr.Row():
                    tesseract_lang = gr.Textbox(label="Tesseract-Sprache", value=tesseract_boxes.DEFAULT_LANG)
                    tesseract_psm = gr.Number(label="Tesseract PSM", value=tesseract_boxes.DEFAULT_PSM, precision=0)
                tesseract_button = gr.Button("Tesseract-Boxen anwenden (Snap + Fill)")
                existing = gr.File(label="Vorhandene Annotation", file_types=[".json"], type="filepath")
                load = gr.Button("Scan und JSON laden")
            with gr.Column(scale=2):
                preview = gr.HTML(EMPTY_PREVIEW_HTML, label="Zeilenboxen", elem_id="bbox-preview-wrap")
                # visible=False would unmount this element in Gradio 6, breaking the
                # JS->Python bridge from render_interactive_preview; hide via CSS instead
                # so it stays queryable while a box is dragged/resized/edited.
                bbox_sync = gr.Textbox(elem_id="bbox-sync-box", visible=True, container=False)
                crop = gr.Image(label="Ausgewählte Zeile", type="pil", interactive=False, height=180)
                with gr.Row():
                    add_box_button = gr.Button("Box hinzufügen")
                    delete_box_button = gr.Button("Ausgewählte Box löschen")
                    no_text_button = gr.Button("Kein sichtbarer Text")
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
        split_button.click(split_into_tiles, [image], [tile_paths_state, status])
        load_tile_button.click(
            load_tile_and_autoload,
            [tile_paths_state, tile_number],
            [image, image_state, annotation_state, selected_state, preview, crop, table, active_text_state, existing, status],
        )
        preannotate.click(start_preannotation, [image, model, context], [annotation_state, image_state, selected_state, preview, crop, table, active_text_state, status])
        tesseract_button.click(
            apply_tesseract_boxes,
            [table, annotation_state, image_state, tesseract_lang, tesseract_psm],
            [annotation_state, preview, crop, table, selected_state, active_text_state, status],
        )
        load.click(load_annotation, [image, existing], [annotation_state, image_state, selected_state, preview, crop, table, active_text_state, status])
        table.select(select_row, [table, annotation_state, image_state], [annotation_state, preview, crop, selected_state, active_text_state, status])
        refresh_button.click(refresh, [table, annotation_state, image_state, selected_state], [annotation_state, preview, crop, selected_state, active_text_state, status])
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
            [table, annotation_state, image_state, selected_state],
            [annotation_state, preview, crop, table, selected_state, active_text_state, status],
        )
        # .change() (not .input()) is required here: Gradio only fires .input()
        # for events it recognizes as genuine keystrokes, so the synthetic
        # DOM events our JS dispatches after a drag/resize/text-edit only
        # reach the backend through .change().
        bbox_sync.change(
            sync_bbox_edit,
            [bbox_sync, annotation_state, image_state, selected_state, active_text_state],
            [annotation_state, preview, crop, table, selected_state, active_text_state, status],
        )
        save_button.click(
            save_annotation,
            [table, annotation_state, image_state, model, dataset_root],
            [annotation_state, annotation_file, status, image_state, file_list_all_html, file_list_flagged_html],
        )
        export_button.click(export_jsonl, [table, annotation_state, image_state, dataset_root, export_mode], [annotation_state, training_file, status, image_state])
        refresh_files_button.click(render_file_lists, [dataset_root, image_state], [file_list_all_html, file_list_flagged_html])
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
