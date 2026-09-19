from __future__ import annotations

import argparse
import base64
import io
import json
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

import requests
from PIL import Image, ImageOps

import pdf_utils
import tesseract_boxes
from tiling import Tile, create_tiles, save_tiles

OLLAMA_API = os.environ.get("OLLAMA_API", "http://127.0.0.1:11434/api/chat")
DEFAULT_MODEL = "qwen3-vl:4b"
DEFAULT_CONTEXT = 8192
DEFAULT_MAX_SIDE = 1024
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".gif", ".webp"}

PROMPT = """
Analysiere diesen Bildausschnitt einer gescannten Seite mit deutscher Handschrift.
Erkenne alle vollständig oder teilweise sichtbaren handschriftlichen Textzeilen.
Gib ausschließlich gültiges JSON in diesem Format zurück:
{"lines":[{"bbox_1000":[x1,y1,x2,y2],"text":"erkannter Text","confidence":"high"}]}

Die Koordinaten beziehen sich ausschließlich auf den übergebenen Bildausschnitt
und sind auf 0 bis 1000 normalisiert. Die Box soll die gesamte sichtbare Zeile
möglichst eng umschließen. Sortiere von oben nach unten, dann von links nach
rechts. Ergänze keine nicht sichtbaren Wörter. Zahlen, Namen und Einheiten nicht
plausibilisieren. Unleserliches als [unleserlich], Unsicheres mit [?] markieren.
Ignoriere automatisch vom Scanner oder der Scan-Software hinzugefügte Elemente
wie Wasserzeichen, Stempel oder Dateinummern und
Softwarehinweise am Rand; sie gehören nicht zum handschriftlichen Original und
werden nicht als Textzeile erfasst. Gib dazu keine Erklärungen, Ablehnungen
oder Hinweise zu Urheberrecht, Lizenzen oder Impressum aus. Diese Anfrage ist
für ein privates Handschrift-Digitalisierungsprojekt und enthält keine echten
Rechtsdokumente.
Analysiere das Bild in genau einem Durchgang. Sobald du eine Zeile einmal
gelesen und ihren Text festgelegt hast, lies diese Zeile nicht erneut und
stelle deine Lesung nicht wiederholt infrage (kein "Wait", kein erneutes
Prüfen, kein Nochmal-Ansehen). Nenne jede Zeile genau einmal und gehe danach
sofort zur nächsten über, auch wenn du unsicher bist – markiere Unsicherheit
stattdessen mit [?] oder confidence "low".
confidence darf nur high, medium oder low sein. Keine Markdown-Codeblöcke und
keine Erläuterungen ausgeben.
""".strip()

LOG = logging.getLogger("qwen_preannotate")


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


def is_repeating(lines: list[str], min_cycles: int = 15, max_period: int = 4) -> bool:
    """Erkennt, ob die letzten Zeilen sich als kurzer Zyklus endlos wiederholen."""
    for period in range(1, max_period + 1):
        window = period * min_cycles
        if len(lines) < window:
            continue
        tail = lines[-window:]
        cycle = tail[:period]
        if all(tail[i] == cycle[i % period] for i in range(window)):
            return True
    return False


class RepetitionLoopError(RuntimeError):
    pass


class OllamaLineLogger:
    """Sammelt Streaming-Fragmente, protokolliert fertige Textzeilen und bricht bei Wiederholungsschleifen ab."""

    STREAK_LIMIT = 30

    def __init__(self, tag: str = "OLLAMA") -> None:
        self.tag = tag
        self.buffer = ""
        self.line_number = 0
        self.recent_lines: list[str] = []
        self.seen_lines: set[str] = set()
        self.repeat_streak = 0

    def feed(self, fragment: str) -> None:
        self.buffer += fragment.replace("\r\n", "\n").replace("\r", "\n")
        while "\n" in self.buffer:
            line, self.buffer = self.buffer.split("\n", 1)
            self._write(line)
            self._check_repetition(line)

    def flush(self) -> None:
        if self.buffer:
            self._write(self.buffer)
        self.buffer = ""

    def _write(self, line: str) -> None:
        self.line_number += 1
        LOG.info("%s %04d | %s", self.tag, self.line_number, line)

    def _check_repetition(self, line: str) -> None:
        stripped = line.strip()
        if not stripped:
            return

        # Kurze, eng getaktete Wiederholungszyklen (z.B. zwei alternierende Zeilen).
        self.recent_lines.append(stripped)
        del self.recent_lines[:-64]
        if is_repeating(self.recent_lines):
            raise RepetitionLoopError(
                f"Modell wiederholt sich endlos ({self.tag}); Abschnitt abgebrochen."
            )

        # Lange, unregelmäßige Wiederholungsschleifen (z.B. das Modell zweifelt seine
        # eigene Analyse wiederholt an und leitet dieselben Zeilen mehrfach neu her).
        # Hier reicht keine feste Zyklenlänge, da der Abstand zwischen Wiederholungen
        # sehr groß und die Formulierung leicht variabel sein kann.
        if stripped in self.seen_lines:
            self.repeat_streak += 1
            if self.repeat_streak >= self.STREAK_LIMIT:
                raise RepetitionLoopError(
                    f"Modell wiederholt bereits gesehene Zeilen ({self.tag}); "
                    f"{self.repeat_streak} Wiederholungen in Folge; Abschnitt abgebrochen."
                )
        else:
            self.seen_lines.add(stripped)
            self.repeat_streak = 0


def extract_json(raw: str) -> dict[str, Any]:
    text = raw.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("Die Modellantwort enthält kein JSON-Objekt.")
    result = json.loads(text[start : end + 1])
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
    """Verwendet dieselbe EXIF-korrigierte Originalpixelbasis wie die GUI."""
    with Image.open(path) as source:
        return ImageOps.exif_transpose(source).convert("RGB")


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


def stop_ollama_model(model: str) -> None:
    """Erzwingt über die Ollama-API das sofortige Entladen des Modells (keep_alive=0),
    damit eine hängende oder in einer Schleife feststeckende Generierung wirklich beendet
    wird, statt sich nur auf das Schließen der Python-Verbindung zu verlassen."""
    generate_api = OLLAMA_API.rsplit("/", 1)[0] + "/generate"
    try:
        requests.post(generate_api, json={"model": model, "keep_alive": 0}, timeout=10)
        LOG.warning("Ollama-Modell '%s' über die API zum sofortigen Entladen angefordert (%s).", model, generate_api)
    except Exception as error:
        LOG.warning("Ollama-Modell '%s' konnte nicht über die API gestoppt werden: %s", model, error)


def call_qwen(image: Image.Image, model: str, context: int, timeout: int, tile_index: int) -> dict[str, Any]:
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
        "Ollama-Anfrage für Abschnitt %d: URL=%s, Modell=%s, Kontext=%d, Bild=%dx%d",
        tile_index,
        OLLAMA_API,
        model,
        context,
        image.width,
        image.height,
    )

    started = time.monotonic()
    fragments: list[str] = []
    thinking_fragments: list[str] = []
    line_logger = OllamaLineLogger("OLLAMA")
    thinking_logger = OllamaLineLogger("DENKEN")
    final_message: dict[str, Any] = {}
    first_fragment_seen = False
    first_thinking_seen = False

    try:
        with requests.post(OLLAMA_API, json=payload, stream=True, timeout=(30, timeout)) as response:
            if not response.ok:
                raise RuntimeError(f"Ollama-Fehler {response.status_code}: {response.text}")
            for raw_line in response.iter_lines(decode_unicode=True):
                if not raw_line:
                    continue
                try:
                    event = json.loads(raw_line)
                except json.JSONDecodeError as error:
                    raise RuntimeError("Ollama lieferte ungültiges Streaming-JSON.") from error
                if "error" in event:
                    raise RuntimeError(f"Ollama-Fehler: {event['error']}")
                message = event.get("message", {})
                fragment = str(message.get("content", ""))
                if fragment:
                    if not first_fragment_seen:
                        LOG.info("Erstes Antwortfragment nach %.2f s empfangen", time.monotonic() - started)
                        first_fragment_seen = True
                    fragments.append(fragment)
                    line_logger.feed(fragment)
                thinking = str(message.get("thinking", ""))
                if thinking:
                    if not first_thinking_seen:
                        LOG.info("Erstes Denkfragment (thinking) nach %.2f s empfangen", time.monotonic() - started)
                        first_thinking_seen = True
                    thinking_fragments.append(thinking)
                    thinking_logger.feed(thinking)
                if event.get("done"):
                    final_message = event
                    break
    except RepetitionLoopError:
        stop_ollama_model(model)
        raise
    except requests.exceptions.ConnectionError as error:
        raise RuntimeError(
            f"Ollama ist unter {OLLAMA_API} nicht erreichbar. "
            "Prüfe Docker-Portfreigabe (-p 11434:11434), Ollama-Status und Firewall."
        ) from error
    except requests.exceptions.Timeout as error:
        raise RuntimeError(f"Timeout beim Zugriff auf Ollama ({OLLAMA_API}).") from error
    finally:
        line_logger.flush()
        thinking_logger.flush()

    raw_content = "".join(fragments)
    if not raw_content:
        LOG.error("Kein content-Fragment empfangen. Letztes Ereignis: %s", json.dumps(final_message, ensure_ascii=False))
        if thinking_fragments:
            LOG.error(
                "Es wurden nur %d Denkfragment(e) (thinking) empfangen, aber kein content. "
                "Das Modell hat vermutlich das Kontextlimit während des Denkens erreicht oder "
                "unterstützt format=json nicht zuverlässig.",
                len(thinking_fragments),
            )
        raise RuntimeError(
            "Ollama hat keine Textantwort geliefert (siehe Log für das letzte Ereignis "
            "und ggf. empfangene Denkfragmente)."
        )
    LOG.info("Ollama-Antwort für Abschnitt %d abgeschlossen: %.2f s", tile_index, time.monotonic() - started)
    for key in ("prompt_eval_count", "eval_count"):
        if key in final_message:
            LOG.info("Ollama-Metrik %s: %s", key, final_message[key])
    for key in ("total_duration", "load_duration", "prompt_eval_duration", "eval_duration"):
        if key in final_message:
            LOG.info("Ollama-Metrik %s: %s", key, format_duration_ns(final_message[key]))
    return extract_json(raw_content)


def local_bbox_to_pixels(local_bbox: list[int], width: int, height: int) -> list[int]:
    """Rechnet eine 0-1000-normalisierte Box in Originalpixel der Kachel um."""
    x1, y1, x2, y2 = local_bbox
    return [
        round(x1 / 1000 * width),
        round(y1 / 1000 * height),
        round(x2 / 1000 * width),
        round(y2 / 1000 * height),
    ]


def run_tile(tile: Tile, args: argparse.Namespace) -> list[dict[str, Any]]:
    """Fragt Qwen für eine Kachel ab und liefert deren Zeilen in kachellokalen Pixelkoordinaten."""
    prepared = scale_for_model(tile.image, args.max_side, args.upscale)
    LOG.info("Abschnitt %d: Bereich %s, Modellbild %d x %d", tile.index, tile.box, prepared.width, prepared.height)
    result = call_qwen(prepared, args.model, args.ctx, args.timeout, tile.index)
    raw_lines = result.get("lines", [])
    if not isinstance(raw_lines, list):
        raise ValueError('Antwort enthält keine Liste "lines".')

    tile_width, tile_height = tile.image.size
    lines: list[dict[str, Any]] = []
    for raw_line in raw_lines:
        if not isinstance(raw_line, dict):
            continue
        try:
            local_bbox = validate_local_bbox(raw_line.get("bbox_1000", raw_line.get("bbox")))
        except (TypeError, ValueError) as error:
            LOG.warning("Zeile wegen ungültiger Box übersprungen: %s", error)
            continue
        text = str(raw_line.get("text", "")).strip()
        if not text:
            continue
        confidence = str(raw_line.get("confidence", "low")).lower().strip()
        if confidence not in {"high", "medium", "low"}:
            confidence = "low"
        lines.append({
            "id": "",
            "bbox_pixels": local_bbox_to_pixels(local_bbox, tile_width, tile_height),
            "bbox_1000": local_bbox,
            "text": text,
            "confidence": confidence,
        })
    LOG.info("Abschnitt %d: %d gültige Zeilen übernommen", tile.index, len(lines))
    return lines


def finalize_lines(lines: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ordered = sorted(lines, key=lambda line: (line["bbox_pixels"][1], line["bbox_pixels"][0]))
    for index, line in enumerate(ordered, 1):
        line["id"] = f"line_{index:04d}"
    return ordered


def maybe_adjust_with_tesseract(
    args: argparse.Namespace, image_path: Path, lines: list[dict[str, Any]], width: int, height: int
) -> list[dict[str, Any]]:
    """Wendet bei --tesseract-adjust zusätzlich Snap+Fill an (siehe
    tesseract_boxes.py). Ein nicht erreichbarer Dienst oder sonstiger Fehler
    wird nur protokolliert - er darf den sonst erfolgreichen Qwen-Lauf nicht
    abbrechen."""
    if not args.tesseract_adjust:
        return lines
    try:
        new_lines, status = tesseract_boxes.detect_and_adjust(
            image_path, lines, width, height, lang=args.tesseract_lang, psm=args.tesseract_psm
        )
    except Exception as error:
        LOG.warning("Tesseract-Anpassung übersprungen: %s", error)
        return lines
    LOG.info("Tesseract-Anpassung: %s", status)
    return finalize_lines(new_lines)


def build_processing_block(args: argparse.Namespace, log_file: Path, tile_count: int) -> dict[str, Any]:
    return {
        "model": args.model,
        "context_size": args.ctx,
        "max_model_image_side": args.max_side,
        "tile_trigger": args.tile_trigger,
        "tile_size": args.tile_size,
        "tile_overlap": args.overlap,
        "tile_count": tile_count,
        "ollama_api": OLLAMA_API,
        "streaming": True,
        "log_file": str(log_file),
    }


def resolve_page_paths(args: argparse.Namespace, source: Path) -> tuple[Path, Path]:
    output = Path(args.output).resolve() if args.output else source.with_name(source.stem + "_preannotation.json")
    log_file = Path(args.log_file).resolve() if args.log_file else source.with_name(source.stem + "_preannotation.log")
    return output, log_file


def is_supported_input(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_EXTENSIONS or pdf_utils.is_pdf(path)


def iter_folder_inputs(folder: Path) -> list[Path]:
    return sorted(
        (item for item in folder.iterdir() if item.is_file() and is_supported_input(item)),
        key=lambda item: item.name.lower(),
    )


def process_input(args: argparse.Namespace) -> list[Path]:
    source = Path(args.image).resolve()
    if source.is_dir():
        return process_folder(args, source)
    return process_scan(args)


def process_folder(args: argparse.Namespace, folder: Path) -> list[Path]:
    if args.output or args.log_file:
        raise SystemExit(
            "--output und --log-file sind bei einem Ordner als Eingabe nicht zulässig; "
            "Dateinamen werden automatisch je Datei vergeben."
        )
    items = iter_folder_inputs(folder)
    if not items:
        raise ValueError(f"Keine Bild- oder PDF-Dateien in {folder} gefunden.")

    # Kein Vorab-Filter auf Dateiebene mehr: process_page()/process_tiled_page()
    # überspringen jede einzelne Seite bzw. Kachel selbst, wenn deren eigene
    # _preannotation.json schon existiert (siehe dort). Ein Dateiebenen-Filter
    # ("irgendeine Seite/Kachel hat schon eine Datei -> ganze Datei überspringen")
    # würde bei einem nur teilweise verarbeiteten mehrseitigen PDF oder groß-
    # formatigen Bild dazu führen, dass die fehlenden Seiten/Kacheln dauerhaft nie
    # nachgeholt werden - das war der Bug, der noch fehlende PDF-Seiten für immer
    # unverarbeitet ließ, sobald mindestens eine Seite bereits vorannotiert war.
    outputs: list[Path] = []
    for index, item in enumerate(items, 1):
        print(f"[{index}/{len(items)}] Verarbeite: {item.name}")
        item_args = argparse.Namespace(**vars(args))
        item_args.image = str(item)
        try:
            outputs.extend(process_scan(item_args))
        except Exception as error:
            print(f"FEHLER bei {item.name}: {error}", file=sys.stderr)

    print(f"Ordner fertig: {len(items)} Datei(en) verarbeitet (bereits vorhandene Seiten/Kacheln je Datei einzeln übersprungen, siehe Ausgabe/Log oben).")
    return outputs


def process_scan(args: argparse.Namespace) -> list[Path]:
    source = Path(args.image).resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Datei nicht gefunden: {source}")

    if pdf_utils.is_pdf(source):
        pages = pdf_utils.extract_pdf_pages(source, dpi=args.pdf_dpi)
        if not pages:
            raise ValueError(f"PDF enthält keine Seiten: {source}")
        if len(pages) > 1 and (args.output or args.log_file):
            raise SystemExit(
                "--output und --log-file sind bei mehrseitigen PDFs nicht zulässig; "
                "Dateinamen werden automatisch je Seite vergeben."
            )
        outputs: list[Path] = []
        for page in pages:
            outputs.extend(process_page(args, page))
        return outputs

    return process_page(args, source)


def process_page(args: argparse.Namespace, source: Path) -> list[Path]:
    """Verarbeitet eine einzelne Bild-/PDF-Seitendatei.

    Wird die Seite in mehrere Abschnitte aufgeteilt, wird jeder Abschnitt als
    eigenständiges Bild samt eigener Annotation gespeichert, statt sie zu
    einer Seitenannotation zusammenzuführen.
    """
    image = open_scan(source)
    page_width, page_height = image.size
    tiles = create_tiles(image, args.tile_trigger, args.tile_size, args.overlap)

    if len(tiles) > 1 and (args.output or args.log_file):
        raise SystemExit(
            "--output und --log-file sind nicht zulässig, wenn ein Bild in mehrere "
            "Abschnitte aufgeteilt wird; Dateinamen werden automatisch je Abschnitt vergeben."
        )

    output, log_file = resolve_page_paths(args, source)

    if len(tiles) == 1 and output.is_file() and not args.force:
        print(f"Bereits vorhanden, übersprungen: {output}")
        return [output]

    configure_logging(log_file, args.verbose)
    LOG.info("Start: %s", source)
    LOG.info("Ollama API: %s", OLLAMA_API)
    LOG.info("Bild: %d x %d Pixel", page_width, page_height)
    LOG.info("Verarbeitung in %d Abschnitt(en)", len(tiles))

    if len(tiles) == 1:
        return [process_untiled_page(args, source, output, log_file, tiles[0], page_width, page_height)]
    return process_tiled_page(args, source, log_file, tiles, page_width, page_height)


def process_untiled_page(
    args: argparse.Namespace,
    source: Path,
    output: Path,
    log_file: Path,
    tile: Tile,
    page_width: int,
    page_height: int,
) -> Path:
    errors: list[dict[str, Any]] = []
    try:
        lines = finalize_lines(run_tile(tile, args))
    except Exception as error:
        errors.append({"tile": tile.index, "box": list(tile.box), "error": str(error)})
        LOG.exception("Fehler in Abschnitt %d: %s", tile.index, error)
        if not args.continue_on_error:
            raise
        lines = []

    lines = maybe_adjust_with_tesseract(args, source, lines, page_width, page_height)

    document = {
        "schema_version": "1.3",
        "task": "handwritten_line_preannotation",
        "coordinate_system": "original_pixels",
        "image": {
            "file": str(source),
            "file_name": source.name,
            "width": page_width,
            "height": page_height,
        },
        "processing": build_processing_block(args, log_file, tile_count=1),
        "lines": lines,
        "errors": errors,
    }
    output.write_text(json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8")
    LOG.info("Gespeichert: %s", output)
    LOG.info("Erkannte Zeilen: %d", len(lines))
    return output


def process_tiled_page(
    args: argparse.Namespace,
    source: Path,
    log_file: Path,
    tiles: list[Tile],
    page_width: int,
    page_height: int,
) -> list[Path]:
    tile_dir = source.with_name(f"{source.stem}_tiles")
    tile_paths = save_tiles(tiles, tile_dir, source.stem)
    outputs: list[Path] = []

    for tile, tile_image_path in zip(tiles, tile_paths):
        tile_output_path = tile_image_path.with_name(tile_image_path.stem + "_preannotation.json")
        if tile_output_path.is_file() and not args.force:
            LOG.info("Abschnitt %d bereits vorhanden, übersprungen: %s", tile.index, tile_output_path)
            outputs.append(tile_output_path)
            continue

        errors: list[dict[str, Any]] = []
        try:
            lines = finalize_lines(run_tile(tile, args))
        except Exception as error:
            errors.append({"tile": tile.index, "box": list(tile.box), "error": str(error)})
            LOG.exception("Fehler in Abschnitt %d: %s", tile.index, error)
            if not args.continue_on_error:
                raise
            lines = []

        tile_width, tile_height = tile.image.size
        lines = maybe_adjust_with_tesseract(args, tile_image_path, lines, tile_width, tile_height)
        document = {
            "schema_version": "1.3",
            "task": "handwritten_line_preannotation",
            "coordinate_system": "original_pixels",
            "image": {
                "file": str(tile_image_path),
                "file_name": tile_image_path.name,
                "width": tile_width,
                "height": tile_height,
            },
            "source": {
                "page_file": str(source),
                "page_width": page_width,
                "page_height": page_height,
                "tile_index": tile.index,
                "tile_count": len(tiles),
                "tile_box_in_page": list(tile.box),
            },
            "processing": build_processing_block(args, log_file, tile_count=len(tiles)),
            "lines": lines,
            "errors": errors,
        }
        tile_output_path.write_text(json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8")
        LOG.info("Abschnitt %d gespeichert: %s (%d Zeilen)", tile.index, tile_output_path, len(lines))
        outputs.append(tile_output_path)

    LOG.info("Alle Abschnitte gespeichert: %d Datei(en) in %s", len(outputs), tile_dir)
    return outputs


def percentage(value: str) -> float:
    number = float(value)
    if not 0 <= number < 0.5:
        raise argparse.ArgumentTypeError("Überlappung muss zwischen 0 und kleiner 0,5 liegen.")
    return number


def positive_int(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("Der Wert muss größer als 0 sein.")
    return number


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Qwen-VL-Vorannotation mit Ollama in Docker.")
    parser.add_argument(
        "image",
        help="PNG-, JPEG-, PDF- oder anderes von Pillow unterstütztes Bild, oder ein "
        "Ordner mit solchen Dateien (bei PDF wird jede Seite einzeln verarbeitet; bei "
        "einem Ordner wird jede darin enthaltene Bild- oder PDF-Datei einzeln verarbeitet "
        "und bereits vorannotierte Dateien werden übersprungen)",
    )
    parser.add_argument("--output", help="Ausgabe-JSON; Standard: <bild>_preannotation.json (nicht bei mehrseitigem PDF oder Ordner)")
    parser.add_argument("--log-file", help="Logdatei; Standard: <bild>_preannotation.log (nicht bei mehrseitigem PDF oder Ordner)")
    parser.add_argument(
        "--pdf-dpi",
        type=positive_int,
        default=pdf_utils.DEFAULT_PDF_DPI,
        help=f"Rasterauflösung für PDF-Seiten in DPI; Standard: {pdf_utils.DEFAULT_PDF_DPI}",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Ollama-Modell; Standard: {DEFAULT_MODEL}")
    parser.add_argument("--ctx", type=positive_int, default=DEFAULT_CONTEXT, help=f"Ollama-Kontextgröße; Standard: {DEFAULT_CONTEXT}")
    parser.add_argument("--max-side", type=positive_int, default=DEFAULT_MAX_SIDE, help=f"Maximale Seitenlänge je Modellbild; Standard: {DEFAULT_MAX_SIDE}")
    parser.add_argument(
        "--tile-trigger",
        type=positive_int,
        default=2800,
        help="Ab dieser Seitenlänge wird unterteilt und jeder Abschnitt einzeln als "
        "eigene Datei mit eigener Annotation gespeichert; Standard: 2800",
    )
    parser.add_argument("--tile-size", type=positive_int, default=2200, help="Kachelgröße in Originalpixeln; Standard: 2200")
    parser.add_argument("--overlap", type=percentage, default=0.15, help="Kachelüberlappung; Standard: 0.15")
    parser.add_argument("--upscale", action="store_true", help="Kleine Abschnitte bis max-side hochskalieren")
    parser.add_argument("--continue-on-error", action="store_true", help="Nach Fehler eines Abschnitts fortfahren")
    parser.add_argument("--timeout", type=positive_int, default=1800, help="Read-Timeout je Abschnitt in Sekunden; Standard: 1800")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Bereits vorhandene Vorannotationen (je Seite/Kachel) erneut erzeugen statt sie "
        "zu überspringen. Ohne diese Option wird jede Seite/Kachel einzeln übersprungen, "
        "deren _preannotation.json schon existiert - so lässt sich ein abgebrochener oder "
        "erweiterter Lauf (Einzeldatei, PDF, Kacheln oder Ordner) fortsetzen, ohne bereits "
        "fertige Seiten/Kacheln erneut an Qwen zu schicken.",
    )
    parser.add_argument(
        "--tesseract-adjust",
        action="store_true",
        help="Nach jeder Qwen-Erkennung zusätzlich Tesseract-Boxen abrufen und die "
        "Zeilenboxen per Snap+Fill anpassen (siehe tesseract_boxes.py); erfordert den "
        "laufenden Tesseract-Docker-Dienst (docker/tesseract-ocr/). Ein nicht erreichbarer "
        "Dienst wird nur protokolliert, nicht als Fehler behandelt.",
    )
    parser.add_argument(
        "--tesseract-lang",
        default=tesseract_boxes.DEFAULT_LANG,
        help=f"Tesseract-Sprachcode; Standard: {tesseract_boxes.DEFAULT_LANG}",
    )
    parser.add_argument(
        "--tesseract-psm",
        type=int,
        default=tesseract_boxes.DEFAULT_PSM,
        help=f"Tesseract Page-Segmentation-Mode; Standard: {tesseract_boxes.DEFAULT_PSM}",
    )
    parser.add_argument("--verbose", action="store_true", help="Ausführlicheres Debug-Logging aktivieren")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.max_side < 512 or args.tile_size < 512 or args.tile_trigger < 512:
        raise SystemExit("max-side, tile-size und tile-trigger müssen mindestens 512 sein.")
    try:
        process_input(args)
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
