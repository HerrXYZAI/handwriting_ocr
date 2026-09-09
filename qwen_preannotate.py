from __future__ import annotations

import argparse
import base64
import io
import json
import logging
import math
import re
import sys
import time
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import requests
from PIL import Image, ImageOps

OLLAMA_API = "http://127.0.0.1:11434/api/chat"
DEFAULT_MODEL = "qwen3-vl:8b"
DEFAULT_CONTEXT = 4096
DEFAULT_MAX_SIDE = 1024

PROMPT = """
Analysiere diesen Bildausschnitt einer gescannten Seite mit deutscher Handschrift.
Erkenne alle vollständig oder teilweise sichtbaren handschriftlichen Textzeilen.
Gib ausschließlich gültiges JSON in diesem Format zurück:
{
  "lines": [
    {
      "bbox_1000": [x1, y1, x2, y2],
      "text": "erkannter Text",
      "confidence": "high"
    }
  ]
}
Die Koordinaten beziehen sich ausschließlich auf den übergebenen Bildausschnitt
und sind auf 0 bis 1000 normalisiert. Die Box soll die gesamte sichtbare Zeile
möglichst eng umschließen. Sortiere von oben nach unten, dann von links nach
rechts. Ergänze keine nicht sichtbaren Wörter. Zahlen, Namen und Einheiten nicht
plausibilisieren. Unleserliches als [unleserlich], Unsicheres mit [?] markieren.
confidence darf nur high, medium oder low sein. Keine Markdown-Codeblöcke und
keine Erläuterungen ausgeben.
""".strip()

LOG = logging.getLogger("qwen_preannotate")


@dataclass(frozen=True)
class Tile:
    index: int
    box: tuple[int, int, int, int]
    image: Image.Image


def configure_logging(log_file: Path, verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    LOG.setLevel(level)
    LOG.handlers.clear()
    LOG.propagate = False

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console = logging.StreamHandler(sys.stdout)
    console.setLevel(level)
    console.setFormatter(formatter)

    file_handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)

    LOG.addHandler(console)
    LOG.addHandler(file_handler)


class OllamaLineLogger:
    """Sammelt Streaming-Fragmente und protokolliert fertige Textzeilen."""

    def __init__(self) -> None:
        self.buffer = ""
        self.line_number = 0

    def feed(self, fragment: str) -> None:
        self.buffer += fragment.replace("\r\n", "\n").replace("\r", "\n")
        while "\n" in self.buffer:
            line, self.buffer = self.buffer.split("\n", 1)
            self._write(line)

    def flush(self) -> None:
        if self.buffer:
            self._write(self.buffer)
            self.buffer = ""

    def _write(self, line: str) -> None:
        self.line_number += 1
        LOG.info("OLLAMA %04d | %s", self.line_number, line)


def extract_json(raw: str) -> dict[str, Any]:
    text = raw.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("Die Modellantwort enthält kein JSON-Objekt.")
    result = json.loads(text[start:end + 1])
    if not isinstance(result, dict):
        raise ValueError("Die JSON-Wurzel muss ein Objekt sein.")
    return result


def clamp(value: int, low: int, high: int) -> int:
    return max(low, min(high, value))


def validate_local_bbox(value: Any) -> list[int]:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise ValueError(f"Ungültige Bounding-Box: {value!r}")
    values = [clamp(round(float(v)), 0, 1000) for v in value]
    x1, y1, x2, y2 = values
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"Leere Bounding-Box: {value!r}")
    return values


def open_scan(path: Path) -> Image.Image:
    with Image.open(path) as source:
        return ImageOps.exif_transpose(source).convert("RGB")


def calculate_starts(length: int, tile_size: int, overlap: float) -> list[int]:
    if length <= tile_size:
        return [0]
    stride = max(1, round(tile_size * (1.0 - overlap)))
    count = max(2, math.ceil((length - tile_size) / stride) + 1)
    starts: list[int] = []
    for index in range(count):
        start = min(index * stride, length - tile_size)
        if not starts or start != starts[-1]:
            starts.append(start)
    return starts


def create_tiles(
    image: Image.Image,
    tile_trigger: int,
    tile_size: int,
    overlap: float,
) -> list[Tile]:
    width, height = image.size
    if max(width, height) <= tile_trigger:
        return [Tile(1, (0, 0, width, height), image.copy())]

    x_starts = calculate_starts(width, tile_size, overlap) if width > tile_trigger else [0]
    y_starts = calculate_starts(height, tile_size, overlap) if height > tile_trigger else [0]
    crop_width = min(width, tile_size) if width > tile_trigger else width
    crop_height = min(height, tile_size) if height > tile_trigger else height

    tiles: list[Tile] = []
    index = 1
    for y in y_starts:
        for x in x_starts:
            box = (x, y, min(width, x + crop_width), min(height, y + crop_height))
            tiles.append(Tile(index, box, image.crop(box)))
            index += 1
    return tiles


def scale_for_model(image: Image.Image, max_side: int, upscale: bool) -> Image.Image:
    width, height = image.size
    factor = min(max_side / width, max_side / height)
    if not upscale:
        factor = min(1.0, factor)
    new_size = (max(1, round(width * factor)), max(1, round(height * factor)))
    if new_size == image.size:
        return image.copy()
    LOG.info("Skalierung: %d x %d -> %d x %d Pixel", width, height, *new_size)
    return image.resize(new_size, Image.Resampling.LANCZOS)


def encode_jpeg(image: Image.Image, quality: int = 92) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality, optimize=True)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def format_duration_ns(value: Any) -> str:
    try:
        return f"{float(value) / 1_000_000_000:.2f} s"
    except (TypeError, ValueError):
        return str(value)


def call_qwen(
    image: Image.Image,
    model: str,
    context: int,
    timeout: int,
    tile_index: int,
) -> dict[str, Any]:
    payload = {
        "model": model,
        "messages": [{
            "role": "user",
            "content": PROMPT,
            "images": [encode_jpeg(image)],
        }],
        "stream": True,
        "format": "json",
        "options": {"temperature": 0, "num_ctx": context},
    }

    LOG.info(
        "Ollama-Anfrage für Abschnitt %d: Modell=%s, Kontext=%d, Bild=%dx%d",
        tile_index, model, context, image.width, image.height,
    )
    started = time.monotonic()
    fragments: list[str] = []
    line_logger = OllamaLineLogger()
    final_message: dict[str, Any] = {}
    first_fragment_seen = False

    try:
        with requests.post(
            OLLAMA_API,
            json=payload,
            stream=True,
            timeout=(30, timeout),
        ) as response:
            if not response.ok:
                raise RuntimeError(
                    f"Ollama-Fehler {response.status_code}: {response.text}"
                )

            for raw_line in response.iter_lines(decode_unicode=True):
                if not raw_line:
                    continue
                try:
                    event = json.loads(raw_line)
                except json.JSONDecodeError as error:
                    LOG.warning("Ungültige Ollama-Streamingzeile: %r", raw_line)
                    raise RuntimeError("Ollama lieferte ungültiges Streaming-JSON.") from error

                if "error" in event:
                    raise RuntimeError(f"Ollama-Fehler: {event['error']}")

                fragment = str(event.get("message", {}).get("content", ""))
                if fragment:
                    if not first_fragment_seen:
                        LOG.info("Erstes Antwortfragment nach %.2f s empfangen", time.monotonic() - started)
                        first_fragment_seen = True
                    fragments.append(fragment)
                    line_logger.feed(fragment)

                if event.get("done"):
                    final_message = event
                    break
    finally:
        line_logger.flush()

    raw_content = "".join(fragments)
    if not raw_content:
        raise RuntimeError("Ollama hat keine Textantwort geliefert.")

    elapsed = time.monotonic() - started
    LOG.info("Ollama-Antwort für Abschnitt %d abgeschlossen: %.2f s", tile_index, elapsed)
    for key in ("prompt_eval_count", "eval_count"):
        if key in final_message:
            LOG.info("Ollama-Metrik %s: %s", key, final_message[key])
    for key in ("total_duration", "load_duration", "prompt_eval_duration", "eval_duration"):
        if key in final_message:
            LOG.info("Ollama-Metrik %s: %s", key, format_duration_ns(final_message[key]))

    return extract_json(raw_content)


def local_to_global_bbox(
    local_bbox: list[int],
    tile_box: tuple[int, int, int, int],
    page_width: int,
    page_height: int,
) -> tuple[list[int], list[int]]:
    tx1, ty1, tx2, ty2 = tile_box
    tile_width, tile_height = tx2 - tx1, ty2 - ty1
    lx1, ly1, lx2, ly2 = local_bbox
    pixel_box = [
        round(tx1 + lx1 / 1000 * tile_width),
        round(ty1 + ly1 / 1000 * tile_height),
        round(tx1 + lx2 / 1000 * tile_width),
        round(ty1 + ly2 / 1000 * tile_height),
    ]
    px1, py1, px2, py2 = pixel_box
    global_1000 = [
        clamp(round(px1 / page_width * 1000), 0, 1000),
        clamp(round(py1 / page_height * 1000), 0, 1000),
        clamp(round(px2 / page_width * 1000), 0, 1000),
        clamp(round(py2 / page_height * 1000), 0, 1000),
    ]
    return pixel_box, global_1000


def intersection_over_union(a: list[int], b: list[int]) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    union = area_a + area_b - intersection
    return intersection / union if union else 0.0


def vertical_overlap(a: list[int], b: list[int]) -> float:
    overlap = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    smaller = min(max(1, a[3] - a[1]), max(1, b[3] - b[1]))
    return overlap / smaller


def text_similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a.casefold().strip(), b.casefold().strip()).ratio()


def is_duplicate(candidate: dict[str, Any], existing: dict[str, Any]) -> bool:
    a, b = candidate["bbox_pixels"], existing["bbox_pixels"]
    if intersection_over_union(a, b) >= 0.35:
        return True
    return (
        vertical_overlap(a, b) >= 0.70
        and text_similarity(candidate["text"], existing["text"]) >= 0.72
    )


def confidence_rank(value: str) -> int:
    return {"low": 0, "medium": 1, "high": 2}.get(value, 0)


def merge_lines(lines: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    for candidate in lines:
        duplicate_index = next(
            (i for i, old in enumerate(merged) if is_duplicate(candidate, old)),
            None,
        )
        if duplicate_index is None:
            merged.append(candidate)
            continue
        old = merged[duplicate_index]
        old_score = (confidence_rank(old["confidence"]), len(old["text"]))
        new_score = (confidence_rank(candidate["confidence"]), len(candidate["text"]))
        if new_score > old_score:
            merged[duplicate_index] = candidate

    merged.sort(key=lambda line: (line["bbox_pixels"][1], line["bbox_pixels"][0]))
    for index, line in enumerate(merged, 1):
        line["id"] = f"line_{index:04d}"
    return merged


def process_scan(args: argparse.Namespace) -> Path:
    source = Path(args.image).resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Bild nicht gefunden: {source}")

    output = (
        Path(args.output).resolve()
        if args.output
        else source.with_name(source.stem + "_preannotation.json")
    )
    log_file = (
        Path(args.log_file).resolve()
        if args.log_file
        else source.with_name(source.stem + "_preannotation.log")
    )
    configure_logging(log_file, args.verbose)

    LOG.info("Start: %s", source)
    LOG.info("Logdatei: %s", log_file)
    image = open_scan(source)
    page_width, page_height = image.size
    tiles = create_tiles(image, args.tile_trigger, args.tile_size, args.overlap)

    tile_directory: Path | None
    if args.save_tiles:
        tile_directory = source.parent / f"{source.stem}_tiles"
        tile_directory.mkdir(exist_ok=True)
    else:
        tile_directory = None

    collected: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    LOG.info("Bild: %d x %d Pixel", page_width, page_height)
    LOG.info("Verarbeitung in %d Abschnitt(en)", len(tiles))

    for tile in tiles:
        prepared = scale_for_model(tile.image, args.max_side, args.upscale)
        LOG.info(
            "[%d/%d] Bereich %s, Modellbild %d x %d",
            tile.index, len(tiles), tile.box, prepared.width, prepared.height,
        )
        if tile_directory:
            tile_path = tile_directory / f"tile_{tile.index:03d}.jpg"
            prepared.save(tile_path, quality=92)
            LOG.info("Modellbild gespeichert: %s", tile_path)

        try:
            result = call_qwen(
                prepared, args.model, args.ctx, args.timeout, tile.index
            )
            raw_lines = result.get("lines", [])
            if not isinstance(raw_lines, list):
                raise ValueError('Antwort enthält keine Liste "lines".')

            accepted = 0
            for raw_line in raw_lines:
                if not isinstance(raw_line, dict):
                    continue
                try:
                    local_bbox = validate_local_bbox(
                        raw_line.get("bbox_1000", raw_line.get("bbox"))
                    )
                except (TypeError, ValueError) as error:
                    LOG.warning("Zeile wegen ungültiger Box übersprungen: %s", error)
                    continue

                bbox_pixels, bbox_1000 = local_to_global_bbox(
                    local_bbox, tile.box, page_width, page_height
                )
                confidence = str(raw_line.get("confidence", "low")).lower().strip()
                if confidence not in {"high", "medium", "low"}:
                    confidence = "low"
                text = str(raw_line.get("text", "")).strip()
                if not text:
                    continue

                collected.append({
                    "id": "",
                    "bbox_pixels": bbox_pixels,
                    "bbox_1000": bbox_1000,
                    "text": text,
                    "confidence": confidence,
                    "source_tile": tile.index,
                })
                accepted += 1
            LOG.info("Abschnitt %d: %d gültige Zeilen übernommen", tile.index, accepted)

        except Exception as error:
            errors.append({"tile": tile.index, "box": list(tile.box), "error": str(error)})
            LOG.exception("Fehler in Abschnitt %d: %s", tile.index, error)
            if not args.continue_on_error:
                raise

    final_lines = merge_lines(collected)
    document = {
        "schema_version": "1.2",
        "task": "handwritten_line_preannotation",
        "coordinate_system": "normalized_0_1000_and_pixels",
        "image": {"file": str(source), "width": page_width, "height": page_height},
        "processing": {
            "model": args.model,
            "context_size": args.ctx,
            "max_model_image_side": args.max_side,
            "tile_trigger": args.tile_trigger,
            "tile_size": args.tile_size,
            "tile_overlap": args.overlap,
            "tile_count": len(tiles),
            "ollama_api": OLLAMA_API,
            "streaming": True,
            "log_file": str(log_file),
        },
        "lines": final_lines,
        "errors": errors,
    }
    output.write_text(json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8")
    LOG.info("Gespeichert: %s", output)
    LOG.info("Erkannte Zeilen nach Zusammenführung: %d", len(final_lines))
    LOG.info("Fehlerhafte Abschnitte: %d", len(errors))
    return output


def percentage(value: str) -> float:
    number = float(value)
    if not 0 <= number < 0.5:
        raise argparse.ArgumentTypeError(
            "Überlappung muss zwischen 0 und kleiner 0,5 liegen."
        )
    return number


def positive_int(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("Der Wert muss größer als 0 sein.")
    return number


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Qwen-VL-Vorannotation mit Skalierung, Kachelung, Streaming-Logging "
            "und globalen Zeilenboxen."
        )
    )
    parser.add_argument("image", help="PNG-, JPEG- oder anderes von Pillow unterstütztes Bild")
    parser.add_argument("--output", help="Ausgabe-JSON; Standard: <bild>_preannotation.json")
    parser.add_argument("--log-file", help="Logdatei; Standard: <bild>_preannotation.log")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Ollama-Modell; Standard: {DEFAULT_MODEL}")
    parser.add_argument("--ctx", type=positive_int, default=DEFAULT_CONTEXT, help=f"Ollama-Kontextgröße; Standard: {DEFAULT_CONTEXT}")
    parser.add_argument("--max-side", type=positive_int, default=DEFAULT_MAX_SIDE, help=f"Maximale Seitenlänge je Modellbild; Standard: {DEFAULT_MAX_SIDE}")
    parser.add_argument("--tile-trigger", type=positive_int, default=2800, help="Ab dieser Seitenlänge wird unterteilt; Standard: 2800")
    parser.add_argument("--tile-size", type=positive_int, default=2200, help="Kachelgröße in Originalpixeln; Standard: 2200")
    parser.add_argument("--overlap", type=percentage, default=0.15, help="Kachelüberlappung; Standard: 0.15")
    parser.add_argument("--upscale", action="store_true", help="Kleine Abschnitte bis max-side hochskalieren")
    parser.add_argument("--save-tiles", action="store_true", help="An Ollama gesendete Abschnitte als JPEG speichern")
    parser.add_argument("--continue-on-error", action="store_true", help="Nach Fehler eines Abschnitts fortfahren")
    parser.add_argument("--timeout", type=positive_int, default=1800, help="Read-Timeout je Abschnitt in Sekunden; Standard: 1800")
    parser.add_argument("--verbose", action="store_true", help="Ausführlicheres Debug-Logging aktivieren")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.max_side < 512 or args.tile_size < 512 or args.tile_trigger < 512:
        raise SystemExit("max-side, tile-size und tile-trigger müssen mindestens 512 sein.")
    try:
        process_scan(args)
    except KeyboardInterrupt:
        LOG.error("Verarbeitung durch Benutzer abgebrochen.")
        raise SystemExit(130)
    except Exception as error:
        if not LOG.handlers:
            print(f"FEHLER: {error}", file=sys.stderr)
        else:
            LOG.error("Verarbeitung abgebrochen: %s", error)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
