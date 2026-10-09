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
from console import run_main, tprint

OLLAMA_API = os.environ.get("OLLAMA_API", "http://127.0.0.1:11434/api/chat")
LLAMACPP_API = os.environ.get("LLAMACPP_API", "http://127.0.0.1:8080/v1/chat/completions")
DEFAULT_BACKEND = "ollama"
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
und sind auf 0 bis 1000 normalisiert und beschreiben die Box ungedreht. Ist
eine Zeile spürbar gedreht/schräg geschrieben (z.B. eine senkrechte
Randnotiz), ergänze zusätzlich "angle" in Grad im Uhrzeigersinn (weglassen
bei normal ausgerichtetem Text). Die Box soll die gesamte sichtbare Zeile
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


# Erkennt eine kurze Zeichenfolge (2-50 Zeichen), die sich mindestens 15-mal
# unmittelbar hintereinander wiederholt - unabhängig von Zeilenumbrüchen.
# Ergänzt is_repeating()/_check_repetition() (die nur fertige Zeilen
# vergleichen) für den Fall, dass sich das Modell innerhalb eines einzelnen,
# nie durch "\n" unterbrochenen JSON-Strings festfährt (z.B. ein Zahlenwert,
# der endlos wiederholt wird).
_RAW_REPEAT_RE = re.compile(r"(.{2,50}?)\1{14,}", re.DOTALL)
_RAW_TAIL_MAX = 1200


class RepetitionLoopError(RuntimeError):
    pass


class OllamaLineLogger:
    """Sammelt Streaming-Fragmente, protokolliert fertige Textzeilen und bricht bei Wiederholungsschleifen ab."""

    STREAK_LIMIT = 30

    def __init__(self, tag: str = "OLLAMA") -> None:
        self.tag = tag
        self.buffer = ""
        self.raw_tail = ""
        self.line_number = 0
        self.recent_lines: list[str] = []
        self.seen_lines: set[str] = set()
        self.repeat_streak = 0

    def feed(self, fragment: str) -> None:
        self.raw_tail = (self.raw_tail + fragment)[-_RAW_TAIL_MAX:]
        if _RAW_REPEAT_RE.search(self.raw_tail):
            raise RepetitionLoopError(
                f"Modell wiederholt eine kurze Zeichenfolge endlos ({self.tag}); Abschnitt abgebrochen."
            )
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


_BBOX_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")


def coerce_bbox(value: Any) -> Any:
    """Repariert Boxen, die das Modell als Text statt als Zahlenliste
    liefert, z.B. ":[94,112,617,151]," (kommt bei großen Modellen mit
    format=json vor). Genau vier Zahlen im Text -> Liste; sonst unverändert."""
    if isinstance(value, str):
        numbers = _BBOX_NUMBER_RE.findall(value)
        if len(numbers) == 4:
            return [float(n) for n in numbers]
    if isinstance(value, (list, tuple)) and len(value) == 4 and all(isinstance(v, str) for v in value):
        try:
            return [float(v) for v in value]
        except ValueError:
            return value
    return value


def validate_local_bbox(value: Any) -> list[int]:
    value = coerce_bbox(value)
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


_PLACEMENT_LOGGED: set[str] = set()


def log_ollama_placement(model: str, api_url: str) -> None:
    """Protokolliert einmal je Modell, welcher Anteil im VRAM liegt (wie `ollama ps`).

    Passt ein großes Modell (z.B. qwen3-vl:30b-a3b-instruct) nicht vollständig in
    den Grafikspeicher, lagert Ollama die übrigen Schichten automatisch in den
    Arbeitsspeicher aus. Diese Aufteilung bestimmt maßgeblich die Laufzeit."""
    if model in _PLACEMENT_LOGGED:
        return
    _PLACEMENT_LOGGED.add(model)
    ps_api = api_url.rsplit("/", 1)[0] + "/ps"
    try:
        response = requests.get(ps_api, timeout=10)
        response.raise_for_status()
        entries = response.json().get("models", [])
    except Exception as error:
        LOG.debug("Modellaufteilung konnte nicht abgefragt werden (%s): %s", ps_api, error)
        return
    for entry in entries:
        if entry.get("name") != model and entry.get("model") != model:
            continue
        size = int(entry.get("size") or 0)
        size_vram = int(entry.get("size_vram") or 0)
        if size <= 0:
            return
        gpu_share = size_vram / size * 100
        LOG.info(
            "Modellaufteilung '%s': %.1f GB gesamt, %.1f GB im VRAM (%.0f%% GPU / %.0f%% CPU)",
            model,
            size / 1e9,
            size_vram / 1e9,
            gpu_share,
            100 - gpu_share,
        )
        if gpu_share < 99.5:
            LOG.info(
                "Modell läuft teilweise auf der CPU (Auslagerung in den Arbeitsspeicher) - "
                "langsamer, aber funktionsfähig."
            )
        return


def call_qwen(
    image: Image.Image,
    model: str,
    context: int,
    timeout: int,
    tile_index: int,
    backend: str,
    api_url: str,
    think: bool = False,
    no_mmap: bool = False,
) -> dict[str, Any]:
    if backend == "llamacpp":
        return _call_llamacpp(image, model, timeout, tile_index, api_url, think)
    return _call_ollama(image, model, context, timeout, tile_index, api_url, think, no_mmap)


def _post_ollama_stream(api_url: str, payload: dict[str, Any], timeout: int) -> requests.Response:
    """Sendet die Anfrage; lehnt eine ältere Ollama-Version oder ein Modell den
    Parameter "think" ab, wird einmal ohne ihn wiederholt."""
    response = requests.post(api_url, json=payload, stream=True, timeout=(30, timeout))
    if not response.ok and "think" in payload and "think" in response.text.lower():
        LOG.warning(
            "Ollama akzeptiert den Parameter 'think' für dieses Modell nicht (%s); "
            "Anfrage wird ohne ihn wiederholt.",
            response.text.strip()[:200],
        )
        response.close()
        retry_payload = {key: value for key, value in payload.items() if key != "think"}
        response = requests.post(api_url, json=retry_payload, stream=True, timeout=(30, timeout))
    return response


def _call_ollama(
    image: Image.Image,
    model: str,
    context: int,
    timeout: int,
    tile_index: int,
    api_url: str,
    think: bool = False,
    no_mmap: bool = False,
) -> dict[str, Any]:
    options: dict[str, Any] = {"temperature": 0, "num_ctx": context}
    if no_mmap:
        # Modell beim Laden komplett in den Arbeitsspeicher lesen statt per
        # mmap stückweise nachzuladen. Bei teilweise auf die CPU ausgelagerten
        # Modellen (z.B. MoE-Experten im RAM) lädt das deutlich schneller und
        # verhindert, dass Ollama das Laden wegen Zeitüberschreitung abbricht
        # ("timed out waiting for llama-server to start").
        options["use_mmap"] = False
    payload = {
        "model": model,
        "messages": [{
            "role": "user",
            "content": PROMPT,
            "images": [encode_jpeg(image)],
        }],
        "stream": True,
        "format": "json",
        # Thinking-Modelle (z.B. qwen3-vl:*-thinking) erzeugen sonst vor der
        # eigentlichen Antwort lange Denktexte. Das kostet bei großen, teilweise
        # auf die CPU ausgelagerten Modellen viel Zeit und füllt den Kontext, bevor
        # das JSON überhaupt beginnt. Für reine Instruct-Modelle ohne Wirkung.
        "think": bool(think),
        "options": options,
    }
    LOG.info(
        "Ollama-Anfrage für Abschnitt %d: URL=%s, Modell=%s, Kontext=%d, Bild=%dx%d, Denken=%s",
        tile_index,
        api_url,
        model,
        context,
        image.width,
        image.height,
        "an" if think else "aus",
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
        with _post_ollama_stream(api_url, payload, timeout) as response:
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
                    if event.get("done_reason") == "length":
                        LOG.warning(
                            "Antwort wurde wegen Kontext-/Längenlimit abgeschnitten (done_reason=length). "
                            "--ctx erhöhen oder --max-side verringern."
                        )
                    break
    except RepetitionLoopError:
        stop_ollama_model(model)
        raise
    except requests.exceptions.ConnectionError as error:
        raise RuntimeError(
            f"Ollama ist unter {api_url} nicht erreichbar. "
            "Prüfe Docker-Portfreigabe (-p 11434:11434), Ollama-Status und Firewall."
        ) from error
    except requests.exceptions.Timeout as error:
        raise RuntimeError(f"Timeout beim Zugriff auf Ollama ({api_url}).") from error
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
    log_ollama_placement(model, api_url)
    prompt_tokens = final_message.get("prompt_eval_count")
    output_tokens = final_message.get("eval_count")
    if isinstance(prompt_tokens, int) and isinstance(output_tokens, int) and prompt_tokens + output_tokens > 0.9 * context:
        LOG.warning(
            "Kontext fast ausgeschöpft: %d Eingabe- + %d Ausgabetokens bei --ctx %d. "
            "Ausgabe könnte abgeschnitten sein; --ctx erhöhen.",
            prompt_tokens,
            output_tokens,
            context,
        )
    eval_duration = final_message.get("eval_duration")
    if isinstance(output_tokens, int) and isinstance(eval_duration, (int, float)) and eval_duration > 0:
        LOG.info("Generierungsgeschwindigkeit: %.2f Tokens/s", output_tokens / (eval_duration / 1e9))
    for key in ("prompt_eval_count", "eval_count"):
        if key in final_message:
            LOG.info("Ollama-Metrik %s: %s", key, final_message[key])
    for key in ("total_duration", "load_duration", "prompt_eval_duration", "eval_duration"):
        if key in final_message:
            LOG.info("Ollama-Metrik %s: %s", key, format_duration_ns(final_message[key]))
    return extract_json(raw_content)


def _call_llamacpp(
    image: Image.Image, model: str, timeout: int, tile_index: int, api_url: str, think: bool = False
) -> dict[str, Any]:
    """Spricht die OpenAI-kompatible /v1/chat/completions-API von llama.cpp's
    eigenem Server an (Fallback, falls Ollama mit diesem selbst konvertierten
    Qwen3-VL-GGUF+mmproj-Paar abstürzt - siehe finetune/README.md)."""
    payload = {
        "model": model,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": PROMPT},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encode_jpeg(image)}"}},
            ],
        }],
        "stream": True,
        "temperature": 0,
        # llama.cpp's eigener Default ist 1.0 (keine Bestrafung), anders als
        # Ollamas Modelfile-Default; ohne das kann greedy Decoding (temperature
        # 0) sich in kurzen Wiederholungsschleifen festfahren (siehe
        # RepetitionLoopError/_RAW_REPEAT_RE oben).
        "repeat_penalty": 1.1,
        # Entspricht "think" bei Ollama; von Templates ohne Thinking ignoriert.
        "chat_template_kwargs": {"enable_thinking": bool(think)},
    }
    LOG.info(
        "llama.cpp-Anfrage für Abschnitt %d: URL=%s, Modell=%s, Bild=%dx%d",
        tile_index,
        api_url,
        model,
        image.width,
        image.height,
    )

    started = time.monotonic()
    fragments: list[str] = []
    line_logger = OllamaLineLogger("LLAMACPP")
    first_fragment_seen = False

    try:
        with requests.post(api_url, json=payload, stream=True, timeout=(30, timeout)) as response:
            if not response.ok:
                raise RuntimeError(f"llama.cpp-Fehler {response.status_code}: {response.text}")
            # llama.cpp deklariert im Content-Type keinen Charset; ohne diese
            # Zeile nimmt requests nach RFC 2616 ISO-8859-1 an und zerlegt
            # UTF-8-Mehrbyte-Zeichen (z.B. "€") in Mojibake ("â¬").
            response.encoding = "utf-8"
            for raw_line in response.iter_lines(decode_unicode=True):
                if not raw_line or not raw_line.startswith("data:"):
                    continue
                data = raw_line[len("data:"):].strip()
                if data == "[DONE]":
                    break
                try:
                    event = json.loads(data)
                except json.JSONDecodeError as error:
                    raise RuntimeError("llama.cpp lieferte ungültiges Streaming-JSON.") from error
                if "error" in event:
                    raise RuntimeError(f"llama.cpp-Fehler: {event['error']}")
                choices = event.get("choices") or []
                delta = choices[0].get("delta", {}) if choices else {}
                fragment = str(delta.get("content") or "")
                if fragment:
                    if not first_fragment_seen:
                        LOG.info("Erstes Antwortfragment nach %.2f s empfangen", time.monotonic() - started)
                        first_fragment_seen = True
                    fragments.append(fragment)
                    line_logger.feed(fragment)
    except RepetitionLoopError:
        LOG.warning("llama.cpp-Backend unterstützt kein erzwungenes Entladen; Verbindung wird geschlossen.")
        raise
    except requests.exceptions.ConnectionError as error:
        raise RuntimeError(
            f"llama.cpp-Server ist unter {api_url} nicht erreichbar. "
            "Prüfe Docker-Portfreigabe (-p 8080:8080) und Container-Status."
        ) from error
    except requests.exceptions.Timeout as error:
        raise RuntimeError(f"Timeout beim Zugriff auf den llama.cpp-Server ({api_url}).") from error
    finally:
        line_logger.flush()

    raw_content = "".join(fragments)
    if not raw_content:
        raise RuntimeError("llama.cpp hat keine Textantwort geliefert.")
    LOG.info("llama.cpp-Antwort für Abschnitt %d abgeschlossen: %.2f s", tile_index, time.monotonic() - started)
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
    result = call_qwen(
        prepared, args.model, args.ctx, args.timeout, tile.index, args.backend, args.api_url, args.think,
        getattr(args, "no_mmap", False),
    )
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
        try:
            angle = float(raw_line.get("angle", 0))
        except (TypeError, ValueError):
            angle = 0.0
        lines.append({
            "id": "",
            "bbox_pixels": local_bbox_to_pixels(local_bbox, tile_width, tile_height),
            "bbox_1000": local_bbox,
            "text": text,
            "confidence": confidence,
            "angle": angle,
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
    """Wendet bei --tesseract-adjust zusätzlich Snap an (siehe
    tesseract_boxes.py): vorhandene Zeilen werden auf die am besten
    überlappende Tesseract-Box eingerastet, es werden aber keine neuen
    Zeilen aus unzugeordneten Tesseract-Boxen ergänzt. Ein nicht
    erreichbarer Dienst oder sonstiger Fehler wird nur protokolliert - er
    darf den sonst erfolgreichen Qwen-Lauf nicht abbrechen."""
    if not args.tesseract_adjust:
        return lines
    try:
        new_lines, status = tesseract_boxes.detect_and_adjust(
            image_path, lines, width, height, lang=args.tesseract_lang, psm=args.tesseract_psm, add_unmatched=False
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
        "think": args.think,
        "no_mmap": args.no_mmap,
        "max_model_image_side": args.max_side,
        "tile_trigger": args.tile_trigger,
        "tile_size": args.tile_size,
        "tile_overlap": args.overlap,
        "tile_count": tile_count,
        "backend": args.backend,
        "api_url": args.api_url,
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
        tprint(f"[{index}/{len(items)}] Verarbeite: {item.name}")
        item_args = argparse.Namespace(**vars(args))
        item_args.image = str(item)
        try:
            outputs.extend(process_scan(item_args))
        except Exception as error:
            tprint(f"FEHLER bei {item.name}: {error}", file=sys.stderr)

    tprint(f"Ordner fertig: {len(items)} Datei(en) verarbeitet (bereits vorhandene Seiten/Kacheln je Datei einzeln übersprungen, siehe Ausgabe/Log oben).")
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
        tprint(f"Bereits vorhanden, übersprungen: {output}")
        return [output]

    configure_logging(log_file, args.verbose)
    LOG.info("Start: %s", source)
    LOG.info("Backend: %s, API: %s", args.backend, args.api_url)
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
    parser.add_argument(
        "--ctx",
        type=positive_int,
        default=DEFAULT_CONTEXT,
        help=f"Ollama-Kontextgröße; Standard: {DEFAULT_CONTEXT}. Bei --max-side 1536 und "
        "vollen Seiten ggf. 12288.",
    )
    parser.add_argument(
        "--no-mmap",
        action="store_true",
        help="Modell beim Laden komplett in den Arbeitsspeicher lesen (Ollama use_mmap=false). "
        "Empfohlen für große, teilweise auf die CPU ausgelagerte Modelle: lädt schneller und "
        "verhindert Abbrüche mit 'timed out waiting for llama-server to start'.",
    )
    parser.add_argument(
        "--think",
        action="store_true",
        help="Denkmodus (thinking) des Modells zulassen. Standard: aus - spart bei großen, "
        "teilweise auf die CPU ausgelagerten Modellen viel Zeit und verhindert, dass "
        "Denktext den Kontext füllt, bevor das JSON beginnt.",
    )
    parser.add_argument(
        "--backend",
        choices=("ollama", "llamacpp"),
        default=DEFAULT_BACKEND,
        help="Inferenz-Backend; Standard: ollama. 'llamacpp' spricht stattdessen den "
        "OpenAI-kompatiblen llama.cpp-Server an (siehe finetune/README.md, Fallback "
        "falls Ollama mit einem selbst konvertierten Qwen3-VL-GGUF+mmproj-Paar abstürzt).",
    )
    parser.add_argument(
        "--api-url",
        default=None,
        help=f"API-URL; Standard je Backend: ollama={OLLAMA_API}, llamacpp={LLAMACPP_API}",
    )
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
        "vorhandenen Zeilenboxen per Snap einrasten (siehe tesseract_boxes.py); es werden "
        "keine neuen Zeilen aus unzugeordneten Tesseract-Boxen ergänzt. Erfordert den "
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
    if args.api_url is None:
        args.api_url = LLAMACPP_API if args.backend == "llamacpp" else OLLAMA_API
    if args.max_side < 512 or args.tile_size < 512 or args.tile_trigger < 512:
        raise SystemExit("max-side, tile-size und tile-trigger müssen mindestens 512 sein.")
    try:
        process_input(args)
    except KeyboardInterrupt:
        LOG.error("Verarbeitung durch Benutzer abgebrochen.")
        raise SystemExit(130)
    except Exception as error:
        if not LOG.handlers:
            tprint(f"FEHLER: {error}", file=sys.stderr)
        else:
            LOG.error("Verarbeitung abgebrochen: %s", error)
        raise SystemExit(1)


if __name__ == "__main__":
    run_main(main)
