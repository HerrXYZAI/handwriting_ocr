from __future__ import annotations

import argparse
import base64
import copy
import json
import re
from pathlib import Path
from typing import Any

import gradio as gr
import requests
from PIL import Image, ImageDraw, ImageFont, ImageOps

OLLAMA_API = "http://127.0.0.1:11434/api/chat"
DEFAULT_MODEL = "qwen3-vl:4b"
DEFAULT_CONTEXT_SIZE = 4096
CONFIDENCE_VALUES = {"high", "medium", "low"}
<<<<<<< HEAD
TABLE_HEADERS = ["ID", "Text", "Konfidenz", "x1", "y1", "x2", "y2"]
=======
TABLE_HEADERS = ["ID", "Text", "Konfidenz", "x1_px", "y1_px", "x2_px", "y2_px"]
>>>>>>> 095a933 (bug fixes)

PREANNOTATION_PROMPT = """
Analysiere diese gescannte Seite mit deutscher Handschrift.
Erkenne alle handschriftlichen Textzeilen in natürlicher Leserichtung.
Gib ausschließlich gültiges JSON in diesem Format zurück:
<<<<<<< HEAD
{
  "lines": [
    {"bbox_1000": [x1, y1, x2, y2], "text": "erkannter Text", "confidence": "high"}
  ]
}
=======

{"lines": [{"bbox_1000": [x1, y1, x2, y2], "text": "erkannter Text", "confidence": "high"}]}

>>>>>>> 095a933 (bug fixes)
Die Koordinaten müssen auf 0 bis 1000 normalisiert sein. Jede Textzeile erhält
eine eigene, möglichst eng anliegende Box. Ergänze keine unsichtbaren Wörter.
Zahlen, Namen und Einheiten nicht plausibilisieren. Unleserliches als
[unleserlich], unsichere Wörter mit [?]. confidence ist high, medium oder low.
Keine Markdown-Blöcke und keine Erläuterungen ausgeben.
""".strip()

TRAINING_PROMPT = """
Erkenne alle handschriftlichen deutschen Textzeilen auf dieser Seite.
Gib ausschließlich gültiges JSON mit einer Liste namens lines aus. Jeder
Eintrag enthält bbox_1000 als [x1,y1,x2,y2] und text. Die Koordinaten sind auf
0 bis 1000 normalisiert. Sortiere in natürlicher Leserichtung, ergänze keine
nicht sichtbaren Wörter und markiere Unleserliches mit [unleserlich].
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
<<<<<<< HEAD
    value = json.loads(text[start:end + 1])
=======
    value = json.loads(text[start : end + 1])
>>>>>>> 095a933 (bug fixes)
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


<<<<<<< HEAD
=======
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


>>>>>>> 095a933 (bug fixes)
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
<<<<<<< HEAD
        text = str(item.get("text_corrected",item.get("text_predicted",item.get("text","")))).strip()
=======
        text = str(item.get("text_corrected", item.get("text_predicted", item.get("text", "")))).strip()
>>>>>>> 095a933 (bug fixes)
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
<<<<<<< HEAD
    return {"schema_version": "1.0", "coordinate_system": "normalized_0_1000", "lines": result}
=======
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
>>>>>>> 095a933 (bug fixes)


def run_qwen(image_path: str, model: str, context_size: int) -> dict[str, Any]:
    path = Path(image_path)
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = {
        "model": model.strip(),
<<<<<<< HEAD
        "messages": [{"role": "user", "content": PREANNOTATION_PROMPT, "images": [encode_image(path)]}],
=======
        "messages": [{
            "role": "user",
            "content": PREANNOTATION_PROMPT,
            "images": [encode_image(path)],
        }],
>>>>>>> 095a933 (bug fixes)
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
<<<<<<< HEAD
    result = copy.deepcopy(annotation)
    image = open_scan(image_path)
    width, height = image.size
    result.update({
        "image": {"file": str(Path(image_path).resolve()), "file_name": Path(image_path).name,
                  "width": width, "height": height},
=======
    result = ensure_pixel_boxes(annotation, image_path)
    result.update({
>>>>>>> 095a933 (bug fixes)
        "model": model.strip(),
        "task": "handwritten_line_transcription",
    })
    return result


<<<<<<< HEAD
def to_pixels(box: list[int], width: int, height: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    return round(x1*width/1000), round(y1*height/1000), round(x2*width/1000), round(y2*height/1000)


=======
>>>>>>> 095a933 (bug fixes)
def line_to_pixels(
    line: dict[str, Any],
    annotation: dict[str, Any],
    actual_width: int,
    actual_height: int,
) -> tuple[int, int, int, int]:
<<<<<<< HEAD
    """Verwendet bevorzugt globale Originalpixel, sonst bbox_1000."""
    pixel_box = line.get("bbox_pixels")
    if isinstance(pixel_box, (list, tuple)) and len(pixel_box) == 4:
        stored_image = annotation.get("image", {})
        try:
            stored_width = int(stored_image.get("width", actual_width))
            stored_height = int(stored_image.get("height", actual_height))
            values = [float(value) for value in pixel_box]
        except (TypeError, ValueError):
            stored_width = stored_height = 0
            values = []
        if stored_width > 0 and stored_height > 0 and len(values) == 4:
            scale_x = actual_width / stored_width
            scale_y = actual_height / stored_height
            x1, y1, x2, y2 = (
                round(values[0] * scale_x),
                round(values[1] * scale_y),
                round(values[2] * scale_x),
                round(values[3] * scale_y),
            )
            return (
                clamp(x1, 0, actual_width),
                clamp(y1, 0, actual_height),
                clamp(x2, 0, actual_width),
                clamp(y2, 0, actual_height),
            )
    return to_pixels(
        validate_bbox(line["bbox_1000"]), actual_width, actual_height
    )
=======
    """Liest Originalpixel direkt; keine Reskalierung von gespeicherten Boxen."""
    try:
        return tuple(validate_pixel_bbox(line.get("bbox_pixels"), actual_width, actual_height))
    except (TypeError, ValueError):
        return tuple(bbox_1000_to_pixels(line["bbox_1000"], actual_width, actual_height))
>>>>>>> 095a933 (bug fixes)


def get_font(size: int) -> ImageFont.ImageFont:
    for name in ("C:/Windows/Fonts/segoeuib.ttf", "C:/Windows/Fonts/arialbd.ttf"):
        if Path(name).exists():
            return ImageFont.truetype(name, size=size)
    return ImageFont.load_default()


def draw_annotations(image_path: str, annotation: dict[str, Any], selected: int = -1) -> Image.Image:
    image = open_scan(image_path)
    draw = ImageDraw.Draw(image)
    font = get_font(max(16, image.width // 90))
    colors = {"high": "#24A148", "medium": "#F1C21B", "low": "#DA1E28"}
    width = max(2, image.width // 700)
    for index, line in enumerate(annotation.get("lines", [])):
        box = line_to_pixels(line, annotation, image.width, image.height)
        color = "#0067C0" if index == selected else colors.get(line.get("confidence"), "#DA1E28")
<<<<<<< HEAD
        draw.rectangle(box, outline=color, width=width*3 if index == selected else width)
=======
        draw.rectangle(box, outline=color, width=width * 3 if index == selected else width)
>>>>>>> 095a933 (bug fixes)
        label = str(index + 1)
        anchor = (box[0], max(0, box[1] - 30))
        text_box = draw.textbbox(anchor, label, font=font)
        draw.rectangle(text_box, fill=color)
        draw.text(anchor, label, fill="white", font=font)
    return image


def crop_line(image_path: str, annotation: dict[str, Any], selected: int) -> Image.Image | None:
    lines = annotation.get("lines", [])
    if selected < 0 or selected >= len(lines):
        return None
    image = open_scan(image_path)
    x1, y1, x2, y2 = line_to_pixels(lines[selected], annotation, image.width, image.height)
<<<<<<< HEAD
    mx, my = max(10, image.width//70), max(8, image.height//150)
    return image.crop((max(0,x1-mx), max(0,y1-my), min(image.width,x2+mx), min(image.height,y2+my)))


def annotation_to_table(annotation: dict[str, Any]) -> list[list[Any]]:
    return [[x["id"], x["text_corrected"], x["confidence"], *x["bbox_1000"]]
            for x in annotation.get("lines", [])]
=======
    mx, my = max(10, image.width // 70), max(8, image.height // 150)
    return image.crop((max(0, x1 - mx), max(0, y1 - my), min(image.width, x2 + mx), min(image.height, y2 + my)))


def annotation_to_table(annotation: dict[str, Any]) -> list[list[Any]]:
    return [
        [line["id"], line["text_corrected"], line["confidence"], *line["bbox_pixels"]]
        for line in annotation.get("lines", [])
    ]
>>>>>>> 095a933 (bug fixes)


def table_to_annotation(table: Any, annotation: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(annotation)
    if table is None:
        return result
    if hasattr(table, "values"):
        table = table.values.tolist()
<<<<<<< HEAD
    old = result.get("lines", [])
=======

    image = result.get("image", {})
    width = int(image.get("width", 0))
    height = int(image.get("height", 0))
    if width <= 0 or height <= 0:
        raise ValueError("Die Bildgröße fehlt. Annotation zuerst mit einem Bild laden.")

    old_lines = result.get("lines", [])
>>>>>>> 095a933 (bug fixes)
    lines = []
    for index, row in enumerate(table):
        if row is None or len(row) < 7:
            continue
<<<<<<< HEAD
        previous = old[index] if index < len(old) else {}
=======
        previous = old_lines[index] if index < len(old_lines) else {}
>>>>>>> 095a933 (bug fixes)
        corrected = str(row[1] if row[1] is not None else "").strip()
        predicted = previous.get("text_predicted", corrected)
        confidence = str(row[2] or "low").lower().strip()
        if confidence not in CONFIDENCE_VALUES:
            confidence = "low"
<<<<<<< HEAD
        updated = copy.deepcopy(previous)
        updated.update({
            "id": str(row[0] or f"line_{index+1:04d}"),
            "bbox_1000": validate_bbox(list(row[3:7])),
=======

        pixel_box = validate_pixel_bbox(list(row[3:7]), width, height)
        updated = copy.deepcopy(previous)
        updated.update({
            "id": str(row[0] or f"line_{index + 1:04d}"),
            "bbox_pixels": pixel_box,
            "bbox_1000": pixels_to_bbox_1000(pixel_box, width, height),
>>>>>>> 095a933 (bug fixes)
            "text_predicted": predicted,
            "text_corrected": corrected,
            "confidence": confidence,
            "status": "corrected" if corrected != predicted else "confirmed",
        })
<<<<<<< HEAD
        # Nach manueller Änderung der normalisierten Box sind alte Pixelwerte
        # nicht mehr verlässlich und werden daher entfernt.
        if previous.get("bbox_1000") != updated["bbox_1000"]:
            updated.pop("bbox_pixels", None)
        lines.append(updated)
    result["lines"] = lines
=======
        lines.append(updated)

    result["lines"] = lines
    result["coordinate_system"] = "original_pixels"
    result["schema_version"] = "1.3"
>>>>>>> 095a933 (bug fixes)
    return result


def start_preannotation(image_path: str | None, model: str, context: int):
    if not image_path:
        raise gr.Error("Bitte zuerst einen Scan auswählen.")
    try:
        annotation = add_metadata(run_qwen(image_path, model, context), image_path, model)
        selected = 0 if annotation["lines"] else -1
<<<<<<< HEAD
        return (annotation, image_path, selected,
                draw_annotations(image_path, annotation, selected),
                crop_line(image_path, annotation, selected),
                annotation_to_table(annotation),
                f"{len(annotation['lines'])} Zeilen erkannt.")
=======
        return (
            annotation,
            image_path,
            selected,
            draw_annotations(image_path, annotation, selected),
            crop_line(image_path, annotation, selected),
            annotation_to_table(annotation),
            f"{len(annotation['lines'])} Zeilen erkannt.",
        )
>>>>>>> 095a933 (bug fixes)
    except requests.ConnectionError as exc:
        raise gr.Error("Ollama ist unter 127.0.0.1:11434 nicht erreichbar.") from exc
    except Exception as exc:
        raise gr.Error(str(exc)) from exc


def load_annotation(image_path: str | None, json_path: str | None):
    if not image_path or not json_path:
        raise gr.Error("Bitte Scan und Annotations-JSON auswählen.")
    try:
        data = json.loads(Path(json_path).read_text(encoding="utf-8"))
        if not data.get("schema_version"):
            data = normalize_annotation(data)
<<<<<<< HEAD
        else:
            for index, line in enumerate(data.get("lines", []), 1):
                if "text" in line:
                    line.setdefault(
                    "text_predicted",
                    line["text"]
                    )
                    line.setdefault(
                    "text_corrected",
                    line["text"]
                    )

                line.setdefault("id", f"line_{index:04d}")
                line["bbox_1000"] = validate_bbox(line["bbox_1000"])
                line.setdefault("text_predicted", line.get("text_corrected", ""))
                line.setdefault("text_corrected", line.get("text_predicted", ""))
                line.setdefault("confidence", "low")
                line.setdefault("status", "unreviewed")
        selected = 0 if data.get("lines") else -1
        return data, image_path, selected, draw_annotations(image_path, data, selected), crop_line(image_path, data, selected), annotation_to_table(data), f"{len(data.get('lines', []))} Zeilen geladen."
=======
        data = ensure_pixel_boxes(data, image_path)
        selected = 0 if data.get("lines") else -1
        return (
            data,
            image_path,
            selected,
            draw_annotations(image_path, data, selected),
            crop_line(image_path, data, selected),
            annotation_to_table(data),
            f"{len(data.get('lines', []))} Zeilen geladen.",
        )
>>>>>>> 095a933 (bug fixes)
    except Exception as exc:
        raise gr.Error(f"Laden fehlgeschlagen: {exc}") from exc


def select_row(table: Any, annotation: dict[str, Any], image_path: str, evt: gr.SelectData):
<<<<<<< HEAD
    """Verarbeitet Dataframe-Auswahlereignisse aus Gradio 6.x robust."""
=======
>>>>>>> 095a933 (bug fixes)
    annotation = table_to_annotation(table, annotation)
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
        draw_annotations(image_path, annotation, index),
        crop_line(image_path, annotation, index),
        index,
<<<<<<< HEAD
        f"Ausgewählt: Zeile {index+1}",
=======
        f"Ausgewählt: Zeile {index + 1}",
>>>>>>> 095a933 (bug fixes)
    )


def refresh(table: Any, annotation: dict[str, Any], image_path: str, selected: int):
    if not image_path:
        raise gr.Error("Keine Annotation geladen.")
    annotation = table_to_annotation(table, annotation)
    if annotation.get("lines"):
<<<<<<< HEAD
        selected = clamp(int(selected), 0, len(annotation["lines"])-1)
=======
        selected = clamp(int(selected), 0, len(annotation["lines"]) - 1)
>>>>>>> 095a933 (bug fixes)
    else:
        selected = -1
    return annotation, draw_annotations(image_path, annotation, selected), crop_line(image_path, annotation, selected), selected, "Änderungen übernommen."


def save_annotation(table: Any, annotation: dict[str, Any], image_path: str, model: str):
    if not image_path:
        raise gr.Error("Keine Annotation vorhanden.")
    annotation = add_metadata(table_to_annotation(table, annotation), image_path, model)
    output = Path(image_path).with_name(Path(image_path).stem + "_annotation.json")
    output.write_text(json.dumps(annotation, ensure_ascii=False, indent=2), encoding="utf-8")
    return annotation, str(output), f"Annotation gespeichert: {output}"


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
<<<<<<< HEAD
    return {"messages": [
        {"role": "user", "content": [
            {"type": "image", "image": relative_image_path(image_path, dataset_root)},
            {"type": "text", "text": TRAINING_PROMPT},
        ]},
        {"role": "assistant", "content": [
            {"type": "text", "text": json.dumps(answer, ensure_ascii=False, separators=(",", ":"))}
        ]},
    ]}
=======
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
>>>>>>> 095a933 (bug fixes)


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
        annotation = table_to_annotation(table, annotation)
        record = training_record(annotation, image_path, dataset_root)
        root = Path(dataset_root).resolve() if dataset_root.strip() else Path(image_path).resolve().parent
        root.mkdir(parents=True, exist_ok=True)
        output = root / "train.jsonl"
        records = [] if mode == "Datei ersetzen" else read_jsonl(output)
        image_key = record_image(record)
<<<<<<< HEAD
        # Deduplizieren: vorhandenen Datensatz derselben Bilddatei ersetzen.
        records = [item for item in records if record_image(item) != image_key]
        records.append(record)
        output.write_text("".join(json.dumps(x, ensure_ascii=False, separators=(",", ":")) + "\n" for x in records), encoding="utf-8", newline="\n")
=======
        records = [item for item in records if record_image(item) != image_key]
        records.append(record)
        output.write_text(
            "".join(json.dumps(x, ensure_ascii=False, separators=(",", ":")) + "\n" for x in records),
            encoding="utf-8",
            newline="\n",
        )
>>>>>>> 095a933 (bug fixes)
        return annotation, str(output), f"JSONL exportiert: {output} | {len(records)} Datensätze, aktuelle Seite {len(training_answer(annotation)['lines'])} Zeilen."
    except Exception as exc:
        raise gr.Error(f"JSONL-Export fehlgeschlagen: {exc}") from exc


def build_interface() -> gr.Blocks:
    with gr.Blocks(title="Qwen Handschrift-Annotation") as app:
        annotation_state = gr.State({})
        image_state = gr.State("")
        selected_state = gr.State(-1)
<<<<<<< HEAD
        gr.Markdown("# Qwen-Vorannotation für Handschrift\nScan laden, vorannotieren, Texte korrigieren und als Annotation oder Trainings-JSONL speichern.")
=======
        gr.Markdown("# Qwen-Vorannotation für Handschrift\nScan laden, vorannotieren, Texte korrigieren und als Annotation oder Trainings-JSONL speichern. Pixelkoordinaten sind führend und bleiben beim Import/Export unverändert.")
>>>>>>> 095a933 (bug fixes)
        with gr.Row():
            with gr.Column(scale=1):
                image = gr.Image(label="Originalscan", type="filepath", sources=["upload"])
                model = gr.Textbox(label="Ollama-Modell", value=DEFAULT_MODEL)
                context = gr.Number(label="Kontextgröße", value=DEFAULT_CONTEXT_SIZE, precision=0)
                preannotate = gr.Button("Qwen-Vorannotation starten", variant="primary")
                existing = gr.File(label="Vorhandene Annotation", file_types=[".json"], type="filepath")
                load = gr.Button("Scan und JSON laden")
            with gr.Column(scale=2):
<<<<<<< HEAD
                preview = gr.Image(
                    label="Zeilenboxen",
                    type="pil",
                    interactive=False,
                )
                crop = gr.Image(
                    label="Ausgewählte Zeile",
                    type="pil",
                    interactive=False,
                    height=180,
                )
        table = gr.Dataframe(headers=TABLE_HEADERS, datatype=["str","str","str","number","number","number","number"], column_count=(7,"fixed"), label="Text und Boxen korrigieren", interactive=True, wrap=True)
=======
                preview = gr.Image(label="Zeilenboxen", type="pil", interactive=False)
                crop = gr.Image(label="Ausgewählte Zeile", type="pil", interactive=False, height=180)
                table = gr.Dataframe(headers=TABLE_HEADERS, datatype=["str", "str", "str", "number", "number", "number", "number"], column_count=(7, "fixed"), label="Text und Pixelboxen korrigieren", interactive=True, wrap=True)
>>>>>>> 095a933 (bug fixes)
        with gr.Row():
            refresh_button = gr.Button("Änderungen übernehmen")
            save_button = gr.Button("Annotations-JSON speichern")
        with gr.Row():
            dataset_root = gr.Textbox(label="Dataset-Wurzelverzeichnis", placeholder=r"C:\Handschrift-Dataset")
            export_mode = gr.Radio(["An Datei anhängen / Seite aktualisieren", "Datei ersetzen"], value="An Datei anhängen / Seite aktualisieren", label="Exportmodus")
<<<<<<< HEAD
        export_button = gr.Button("Als Qwen-Trainings-JSONL exportieren", variant="primary")
=======
            export_button = gr.Button("Als Qwen-Trainings-JSONL exportieren", variant="primary")
>>>>>>> 095a933 (bug fixes)
        with gr.Row():
            annotation_file = gr.File(label="Annotations-JSON")
            training_file = gr.File(label="Qwen-Trainings-JSONL")
        status = gr.Textbox(label="Status", interactive=False)

        preannotate.click(start_preannotation, [image, model, context], [annotation_state, image_state, selected_state, preview, crop, table, status])
        load.click(load_annotation, [image, existing], [annotation_state, image_state, selected_state, preview, crop, table, status])
        table.select(select_row, [table, annotation_state, image_state], [annotation_state, preview, crop, selected_state, status])
        refresh_button.click(refresh, [table, annotation_state, image_state, selected_state], [annotation_state, preview, crop, selected_state, status])
        save_button.click(save_annotation, [table, annotation_state, image_state, model], [annotation_state, annotation_file, status])
        export_button.click(export_jsonl, [table, annotation_state, image_state, dataset_root, export_mode], [annotation_state, training_file, status])
    return app


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=7860, type=int)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()
    build_interface().launch(server_name=args.host, server_port=args.port, share=args.share, inbrowser=True)


if __name__ == "__main__":
    main()
