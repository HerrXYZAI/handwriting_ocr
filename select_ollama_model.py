"""Auswahl des Ollama-Modells für run.bat (Vorannotation).

Listet die in Ollama installierten Modelle mit Bildverarbeitung (Fähigkeit
"vision" laut /api/show; Text- und Embedding-Modelle sind für die
Vorannotation unbrauchbar und werden ausgeblendet). Das zuletzt gewählte
Modell ist vorausgewählt und wird in .last_preannotate_model gemerkt.

Aufruf: python select_ollama_model.py <ausgabedatei>
Schreibt den gewählten Modellnamen in die Ausgabedatei.
Exit-Code 0 = gewählt, 1 = abgebrochen.

Nur Standardbibliothek, damit es mit jedem Python auf dem Host läuft.
"""

from __future__ import annotations

import json
import os
import re
import sys
import urllib.request
from pathlib import Path

OLLAMA_HOST = os.environ.get("OLLAMA_HOST_URL", "http://127.0.0.1:11434").rstrip("/")
LAST_MODEL_FILE = Path(__file__).resolve().with_name(".last_preannotate_model")
FALLBACK_MODEL = "qwen3-vl:4b"


def _request(path: str, payload: dict | None = None, timeout: float = 5) -> dict:
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    request = urllib.request.Request(
        OLLAMA_HOST + path, data=data, headers={"Content-Type": "application/json"} if data else {}
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


_INTERNAL_NAME = re.compile(r"^[^:/]+:[0-9a-f]{40,}$")


def is_internal_name(name: str) -> bool:
    """z.B. "llamacpp:4695d1593d...": Prüfsummen-Alias ohne lesbaren Namen."""
    return bool(_INTERNAL_NAME.match(name))


def list_models() -> list[dict] | None:
    """Installierte Modelle mit Größe und (falls von Ollama gemeldet) Fähigkeiten."""
    try:
        models = _request("/api/tags").get("models", [])
    except Exception:
        return None
    result = []
    seen: set[str] = set()
    for model in models:
        name = model.get("name")
        # Ollama führt manche Modelle mehrfach (gleicher Name) und zusätzlich
        # als interne "llamacpp:<Prüfsumme>"-Einträge ohne lesbaren Namen -
        # jeden Namen nur einmal zeigen, Prüfsummen-Einträge ausblenden.
        if not name or name in seen or is_internal_name(name):
            continue
        seen.add(name)
        capabilities = None
        try:
            capabilities = _request("/api/show", {"model": name}).get("capabilities")
        except Exception:
            pass
        details = model.get("details") or {}
        result.append({
            "name": name,
            "size_gb": (model.get("size") or 0) / 1e9,
            "params": details.get("parameter_size", ""),
            "capabilities": capabilities,
        })
    return sorted(result, key=lambda m: m["name"])


def read_last_model() -> str:
    try:
        return LAST_MODEL_FILE.read_text(encoding="utf-8").strip() or FALLBACK_MODEL
    except OSError:
        return FALLBACK_MODEL


def main() -> int:
    if len(sys.argv) != 2:
        print("Aufruf: select_ollama_model.py <ausgabedatei>", file=sys.stderr)
        return 1
    target = Path(sys.argv[1])
    last = read_last_model()
    last_label = "zuletzt genutzt" if LAST_MODEL_FILE.is_file() else "Standard"

    models = list_models()
    print()
    if models is None:
        print(f"Ollama unter {OLLAMA_HOST} nicht erreichbar - Modellname bitte von Hand eingeben.")
        answer = input(f"Modell (Enter = {last}, q = Abbrechen): ").strip()
        if answer.lower() == "q":
            return 1
        chosen = answer or last
    else:
        vision = [m for m in models if m["capabilities"] is None or "vision" in m["capabilities"]]
        hidden = [m["name"] for m in models if m not in vision]
        if not vision:
            print("Keine Modelle mit Bildverarbeitung in Ollama gefunden (z.B. ollama pull qwen3-vl:4b).")
        print("Modell für die Vorannotation wählen:")
        default_index = None
        for index, model in enumerate(vision, 1):
            marker = f"  <- {last_label}" if model["name"] == last else ""
            if model["name"] == last:
                default_index = index
            params = f"{model['params']}, " if model["params"] else ""
            print(f" {index:>2}. {model['name']:<36} ({params}{model['size_gb']:.1f} GB){marker}")
        if hidden:
            print(f"     (ohne Bildverarbeitung ausgeblendet: {', '.join(hidden)})")
        print("  m. Modellname von Hand eingeben")
        print("  q. Abbrechen")
        default_label = str(default_index) if default_index else last
        while True:
            answer = input(f"Auswahl (Enter = {default_label}): ").strip()
            if answer.lower() == "q":
                return 1
            if answer.lower() == "m":
                typed = input("Modellname: ").strip()
                if typed:
                    chosen = typed
                    break
                continue
            if not answer:
                chosen = vision[default_index - 1]["name"] if default_index else last
                break
            if answer.isdigit() and 1 <= int(answer) <= len(vision):
                chosen = vision[int(answer) - 1]["name"]
                break
            print("Ungültige Eingabe.")

    target.write_text(chosen + "\n", encoding="utf-8")
    try:
        LAST_MODEL_FILE.write_text(chosen + "\n", encoding="utf-8")
    except OSError:
        pass
    print(f"Modell: {chosen}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
