"""Ollama-Modelle herunterladen oder löschen (aufgerufen von run.bat).

    python ollama_models.py pull     # Modellnamen einfügen, herunterladen
    python ollama_models.py delete   # installierte Modelle auswählen, löschen

Spricht direkt mit der Ollama-API (Standard http://127.0.0.1:11434, änderbar
über OLLAMA_HOST_URL) und funktioniert damit gleich, egal ob Ollama lokal
installiert ist oder im Docker-Container läuft.

Beim Herunterladen darf statt des reinen Namens auch der komplette Befehl oder
die Adresse von ollama.com eingefügt werden, z.B.
    qwen3-vl:8b
    ollama pull qwen3-vl:8b
    ollama run qwen3.5:9b
    https://ollama.com/library/qwen3-vl:8b
    hf.co/unsloth/Qwen3-VL-8B-Instruct-GGUF:Q4_K_M

Nur Standardbibliothek (plus console.py), damit es mit jedem Python läuft.
Exit-Code 0 = erledigt, 1 = Fehler oder abgebrochen.
"""

from __future__ import annotations

import json
import os
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path

from console import run_main, tprint

OLLAMA_HOST = os.environ.get("OLLAMA_HOST_URL", "http://127.0.0.1:11434").rstrip("/")
LAST_MODEL_FILE = Path(__file__).resolve().with_name(".last_preannotate_model")
_INTERNAL_NAME = re.compile(r"^[^:/]+:[0-9a-f]{40,}$")
_VALID_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._\-/:]*$")


def _request(path: str, payload: dict | None = None, method: str | None = None, timeout: float = 15):
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    request = urllib.request.Request(
        OLLAMA_HOST + path,
        data=data,
        method=method,
        headers={"Content-Type": "application/json"} if data else {},
    )
    return urllib.request.urlopen(request, timeout=timeout)


def _error_text(error: urllib.error.HTTPError) -> str:
    try:
        body = error.read().decode("utf-8", "replace")
        return json.loads(body).get("error", body)
    except Exception:
        return str(error)


def ollama_reachable() -> bool:
    try:
        with _request("/api/tags", timeout=5):
            return True
    except Exception:
        tprint(f"Ollama unter {OLLAMA_HOST} nicht erreichbar. Läuft der Docker-Container 'ollama' bzw. Ollama?")
        return False


def normalize_model_name(raw: str) -> str:
    """Macht aus einem eingefügten Befehl/Link den reinen Modellnamen."""
    name = raw.strip().strip('"').strip("'").strip()
    name = re.sub(r"^(docker\s+exec\s+(-\w+\s+)*\S+\s+)?ollama\s+(pull|run)\s+", "", name, flags=re.I)
    name = re.sub(r"^https?://(www\.)?ollama\.com/", "", name, flags=re.I)
    name = re.sub(r"^library/", "", name)
    name = re.sub(r"^https?://(www\.)?huggingface\.co/", "hf.co/", name, flags=re.I)
    name = re.sub(r"^https?://", "", name, flags=re.I)
    return name.split()[0].rstrip("/") if name else ""


def installed_models() -> list[dict]:
    with _request("/api/tags") as response:
        models = json.load(response).get("models", [])
    seen: set[str] = set()
    result = []
    for model in models:
        name = model.get("name")
        if not name or name in seen or _INTERNAL_NAME.match(name):
            continue
        seen.add(name)
        details = model.get("details") or {}
        result.append({
            "name": name,
            "size_gb": (model.get("size") or 0) / 1e9,
            "params": details.get("parameter_size", ""),
            "quant": details.get("quantization_level", ""),
        })
    return sorted(result, key=lambda m: m["name"])


def capabilities(name: str) -> list[str] | None:
    try:
        with _request("/api/show", {"model": name}) as response:
            return json.load(response).get("capabilities")
    except Exception:
        return None


def pull(name: str) -> bool:
    """Lädt das Modell und zeigt den Fortschritt je Datei-Schicht an."""
    tprint(f"Lade '{name}' herunter ...")
    last_line = ""
    try:
        # Kein Gesamt-Timeout: große Modelle brauchen leicht eine Stunde.
        with _request("/api/pull", {"model": name, "stream": True}, timeout=None) as response:
            for raw in response:
                raw = raw.strip()
                if not raw:
                    continue
                event = json.loads(raw)
                if "error" in event:
                    if last_line:
                        print()
                    tprint(f"FEHLER: {event['error']}")
                    return False
                status = event.get("status", "")
                total, completed = event.get("total"), event.get("completed")
                if total and completed is not None:
                    percent = completed * 100 / total
                    line = f"{status[:40]:<40} {percent:5.1f}%  ({completed / 1e9:.2f} / {total / 1e9:.2f} GB)"
                    print("\r" + line.ljust(len(last_line)), end="", flush=True)
                    last_line = line
                elif status:
                    if last_line:
                        print()
                        last_line = ""
                    tprint(status)
    except urllib.error.HTTPError as error:
        if last_line:
            print()
        tprint(f"FEHLER: {_error_text(error)}")
        return False
    except Exception as error:
        if last_line:
            print()
        tprint(f"FEHLER beim Herunterladen: {error}")
        return False
    if last_line:
        print()
    return True


def cmd_pull() -> int:
    if not ollama_reachable():
        return 1
    print()
    print("Modellnamen einfügen (z.B. qwen3-vl:8b), auch als kompletter Befehl")
    print("\"ollama pull ...\" oder als Link von ollama.com. Modelle mit Bildverarbeitung")
    print("(\"vision\") sind für die Vorannotation nötig. Leer = abbrechen.")
    raw = input("Modell: ")
    name = normalize_model_name(raw)
    if not name:
        return 1
    if not _VALID_NAME.match(name):
        tprint(f"Ungültiger Modellname: {name!r}")
        return 1
    if name != raw.strip():
        tprint(f"Modellname: {name}")
    if not pull(name):
        return 1

    tprint(f"'{name}' ist installiert.")
    caps = capabilities(name)
    if caps is not None and "vision" not in caps:
        tprint(
            f"Hinweis: '{name}' kann keine Bilder verarbeiten (Fähigkeiten: {', '.join(caps) or '-'}) "
            "und ist für die Vorannotation nicht geeignet."
        )
        return 0
    answer = input("Als Standard für die Vorannotation (run.bat, Punkt 3) vormerken? (J/n): ").strip().lower()
    if answer in ("", "j", "ja", "y", "yes"):
        LAST_MODEL_FILE.write_text(name + "\n", encoding="utf-8")
        tprint("Vorgemerkt.")
    return 0


def parse_selection(answer: str, count: int) -> list[int] | None:
    """'2', '1,3', '2-4' oder Kombinationen -> 0-basierte Indizes."""
    indices: list[int] = []
    for part in answer.replace(";", ",").replace(" ", ",").split(","):
        if not part:
            continue
        match = re.fullmatch(r"(\d+)(?:-(\d+))?", part)
        if not match:
            return None
        start = int(match.group(1))
        end = int(match.group(2) or start)
        if start < 1 or end > count or end < start:
            return None
        indices.extend(i - 1 for i in range(start, end + 1) if i - 1 not in indices)
    return indices or None


def cmd_delete() -> int:
    if not ollama_reachable():
        return 1
    models = installed_models()
    if not models:
        tprint("Keine Modelle in Ollama installiert.")
        return 1
    last = LAST_MODEL_FILE.read_text(encoding="utf-8").strip() if LAST_MODEL_FILE.is_file() else ""
    print()
    print("Installierte Ollama-Modelle:")
    for index, model in enumerate(models, 1):
        info = ", ".join(v for v in (model["params"], model["quant"]) if v)
        marker = "  <- Standard Vorannotation" if model["name"] == last else ""
        print(f" {index:>2}. {model['name']:<40} {model['size_gb']:6.1f} GB  {info}{marker}")
    print()
    print("Nummer(n) der zu löschenden Modelle, z.B. 3 oder 1,4 oder 2-5. Leer = abbrechen.")
    while True:
        answer = input("Löschen: ").strip()
        if not answer:
            return 1
        selection = parse_selection(answer, len(models))
        if selection is not None:
            break
        print("Ungültige Eingabe.")

    chosen = [models[i] for i in selection]
    print()
    for model in chosen:
        print(f"   {model['name']}  ({model['size_gb']:.1f} GB)")
    total = sum(m["size_gb"] for m in chosen)
    confirm = input(f"Diese {len(chosen)} Modell(e) endgültig löschen ({total:.1f} GB)? (j/N): ").strip().lower()
    if confirm not in ("j", "ja", "y", "yes"):
        tprint("Abgebrochen, nichts gelöscht.")
        return 1

    failed = 0
    for model in chosen:
        try:
            with _request("/api/delete", {"model": model["name"]}, method="DELETE"):
                pass
            tprint(f"Gelöscht: {model['name']}")
            if model["name"] == last:
                LAST_MODEL_FILE.unlink(missing_ok=True)
        except urllib.error.HTTPError as error:
            failed += 1
            tprint(f"FEHLER bei {model['name']}: {_error_text(error)}")
        except Exception as error:
            failed += 1
            tprint(f"FEHLER bei {model['name']}: {error}")
    tprint(
        "Hinweis: Vorannotationen dieser Modelle in den JSON-Dateien bleiben erhalten "
        "(Modellvergleich und Annotations-GUI zeigen sie weiter an)."
    )
    return 1 if failed else 0


def main() -> int:
    if len(sys.argv) != 2 or sys.argv[1] not in ("pull", "delete"):
        print("Aufruf: ollama_models.py pull|delete", file=sys.stderr)
        return 1
    return cmd_pull() if sys.argv[1] == "pull" else cmd_delete()


if __name__ == "__main__":
    run_main(main)
