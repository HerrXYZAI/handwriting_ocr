"""Auswahl des Basismodells für das Finetuning (aufgerufen von run.bat).

Fragt die in Ollama installierten Modelle ab und ordnet jedem das passende
Hugging-Face-Modell zu. Ollama-Modelle selbst (GGUF, quantisiert) lassen sich
nicht feinabstimmen - ms-swift lädt die Originalgewichte von Hugging Face.
Die Wahl wird als KEY=VALUE-Zeilen in die übergebene Datei geschrieben, die
run.bat einliest:

    FT_MODEL=Qwen/Qwen3-VL-8B-Instruct
    FT_OUTPUT_DIR_CONTAINER=/output/qwen3-vl-8b-handschrift
    FT_MODEL_SLUG=qwen3-vl-8b

Nur Standardbibliothek, damit es mit jedem Python auf dem Host läuft.
Exit-Code 0 = gewählt, 1 = abgebrochen.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path

OLLAMA_TAGS = os.environ.get("OLLAMA_TAGS_API", "http://127.0.0.1:11434/api/tags")
DEFAULT_HF_MODEL = "Qwen/Qwen3-VL-4B-Instruct"

# Ollama-Familie -> (Größe im Tag -> Hugging-Face-Name ohne Variante, Parameter in Mrd.)
QWEN3_VL = {
    "2b": ("Qwen/Qwen3-VL-2B", 2),
    "4b": ("Qwen/Qwen3-VL-4B", 4),
    "8b": ("Qwen/Qwen3-VL-8B", 8),
    "30b": ("Qwen/Qwen3-VL-30B-A3B", 30),
    "32b": ("Qwen/Qwen3-VL-32B", 32),
    "235b": ("Qwen/Qwen3-VL-235B-A22B", 235),
}
# Qwen3.5/3.6 sind nativ multimodal (kein eigenes "-VL"-Modell, kein
# "-Instruct"-Suffix auf Hugging Face). Hybride Architektur (Gated DeltaNet +
# Attention) - braucht ein aktuelles ms-swift/transformers im Trainings-Image.
QWEN35 = {
    "0.8b": ("Qwen/Qwen3.5-0.8B", 0.8),
    "2b": ("Qwen/Qwen3.5-2B", 2),
    "4b": ("Qwen/Qwen3.5-4B", 4),
    "9b": ("Qwen/Qwen3.5-9B", 9),
    "27b": ("Qwen/Qwen3.5-27B", 27),
    "35b": ("Qwen/Qwen3.5-35B-A3B", 35),
    "122b": ("Qwen/Qwen3.5-122B-A10B", 122),
    "397b": ("Qwen/Qwen3.5-397B-A17B", 397),
}
QWEN36 = {
    "27b": ("Qwen/Qwen3.6-27B", 27),
    "35b": ("Qwen/Qwen3.6-35B-A3B", 35),
}
NEW_ARCH_NOTE = "neue Architektur, ggf. Image neu bauen"

QWEN25_VL = {
    "3b": ("Qwen/Qwen2.5-VL-3B-Instruct", 3),
    "7b": ("Qwen/Qwen2.5-VL-7B-Instruct", 7),
    "32b": ("Qwen/Qwen2.5-VL-32B-Instruct", 32),
    "72b": ("Qwen/Qwen2.5-VL-72B-Instruct", 72),
}


@dataclass
class Candidate:
    ollama_name: str
    hf_model: str | None
    params_b: float | None
    note: str = ""


def size_token(tag: str, sizes: dict) -> str | None:
    for token in sorted(sizes, key=len, reverse=True):
        if re.search(rf"(^|[^0-9]){re.escape(token)}($|[^0-9a-z]|-)", tag):
            return token
    return None


def map_ollama_model(name: str) -> Candidate:
    """Ordnet einen Ollama-Namen (z.B. 'qwen3-vl:30b-a3b-instruct') dem
    Hugging-Face-Basismodell zu."""
    family, _, tag = name.partition(":")
    tag = (tag or "latest").lower()
    family = family.lower().rsplit("/", 1)[-1]
    if family == "qwen3-vl":
        token = size_token(tag, QWEN3_VL) or ("8b" if tag == "latest" else None)
        if token:
            base, params = QWEN3_VL[token]
            variant = "Thinking" if "thinking" in tag else "Instruct"
            return Candidate(name, f"{base}-{variant}", params)
    for prefix, sizes, default in (("qwen3.5", QWEN35, "9b"), ("qwen3.6", QWEN36, "35b")):
        if family == prefix:
            token = size_token(tag, sizes) or (default if tag == "latest" else None)
            if token:
                hf, params = sizes[token]
                return Candidate(name, hf, params, NEW_ARCH_NOTE)
    if family in ("qwen2.5vl", "qwen2.5-vl"):
        token = size_token(tag, QWEN25_VL) or ("7b" if tag == "latest" else None)
        if token:
            hf, params = QWEN25_VL[token]
            return Candidate(name, hf, params)
    return Candidate(name, None, None, "kein Hugging-Face-Basismodell bekannt (z.B. eigenes Finetune)")


def list_ollama_models() -> list[str] | None:
    try:
        with urllib.request.urlopen(OLLAMA_TAGS, timeout=5) as response:
            data = json.load(response)
    except Exception:
        return None
    return sorted({m.get("name") for m in data.get("models", []) if m.get("name")})


def gpu_memory_gb() -> float | None:
    try:
        output = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10, check=True,
        ).stdout
        return max(float(v) for v in output.split()) / 1024
    except Exception:
        return None


def qlora_vram_gb(params_b: float) -> float:
    """Grobe Schätzung für QLoRA (4-bit, Batch 1, Gradient Checkpointing,
    wie in train.sh): ~0,6 GB je Mrd. Parameter + ~3 GB für Aktivierungen,
    Bild-Encoder und CUDA."""
    return params_b * 0.6 + 3.0


def slug_for(hf_model: str) -> str:
    name = hf_model.rsplit("/", 1)[-1].lower()
    name = name.replace("-instruct", "")
    return re.sub(r"[^a-z0-9.\-]+", "-", name).strip("-")


def main() -> int:
    if len(sys.argv) != 2:
        print("Aufruf: select_model.py <ausgabedatei>", file=sys.stderr)
        return 1
    target = Path(sys.argv[1])
    current = os.environ.get("FT_MODEL") or DEFAULT_HF_MODEL

    names = list_ollama_models()
    if names is None:
        print(f"Ollama unter {OLLAMA_TAGS} nicht erreichbar - zeige bekannte Basismodelle.")
        candidates = [Candidate(f"({hf.rsplit('/', 1)[-1]})", hf, p) for hf, p in (
            ("Qwen/Qwen3-VL-2B-Instruct", 2), ("Qwen/Qwen3-VL-4B-Instruct", 4), ("Qwen/Qwen3-VL-8B-Instruct", 8),
        )]
    else:
        candidates = [map_ollama_model(name) for name in names]

    trainable = [c for c in candidates if c.hf_model]
    # Gleiches HF-Modell nur einmal anbieten (z.B. qwen3-vl:30b und :30b-a3b-instruct).
    seen: dict[str, Candidate] = {}
    for c in trainable:
        seen.setdefault(c.hf_model, c)
    options = sorted(seen.values(), key=lambda c: (c.params_b or 0, c.hf_model))
    if not any(c.hf_model == current for c in options):
        options.insert(0, Candidate("(aktuelle Einstellung)", current, None))

    vram = gpu_memory_gb()
    print()
    print("Basismodell für das Finetuning wählen")
    print("(Ollama-Modelle sind quantisiert und nicht trainierbar - trainiert wird das")
    print(" zugehörige Original von Hugging Face, Download beim ersten Mal.)")
    if vram:
        print(f"Grafikkarte: {vram:.0f} GB")
    print()
    default_index = 1
    for index, c in enumerate(options, 1):
        if c.hf_model == current:
            default_index = index
        need = qlora_vram_gb(c.params_b) if c.params_b else None
        warn = ""
        if need and vram and need > vram:
            warn = f"  !! braucht ca. {need:.0f} GB, passt nicht in {vram:.0f} GB"
        elif need:
            warn = f"  (ca. {need:.0f} GB)"
        note = f"  [{c.note}]" if c.note else ""
        print(f" {index:>2}. {c.hf_model:<34} <- {c.ollama_name}{warn}{note}")
    skipped = [c for c in candidates if not c.hf_model]
    if skipped:
        print()
        print("Nicht trainierbar (kein Basismodell zugeordnet):")
        for c in skipped:
            print(f"     {c.ollama_name}")
    print()
    print(" m. Hugging-Face-Modell von Hand eingeben")
    print(" q. Abbrechen")

    while True:
        answer = input(f"Auswahl (Enter = {default_index}): ").strip().lower()
        if answer == "q":
            return 1
        if answer == "m":
            hf_model = input("Hugging-Face-Modell (z.B. Qwen/Qwen3-VL-8B-Instruct): ").strip()
            if hf_model:
                chosen = Candidate("(manuell)", hf_model, None)
                break
            continue
        if not answer:
            chosen = options[default_index - 1]
            break
        if answer.isdigit() and 1 <= int(answer) <= len(options):
            chosen = options[int(answer) - 1]
            break
        print("Ungültige Eingabe.")

    need = qlora_vram_gb(chosen.params_b) if chosen.params_b else None
    if need and vram and need > vram:
        confirm = input(
            f"{chosen.hf_model} braucht voraussichtlich ca. {need:.0f} GB Grafikspeicher, "
            f"vorhanden sind {vram:.0f} GB. Trotzdem starten? (j/N): "
        ).strip().lower()
        if confirm not in ("j", "ja", "y", "yes"):
            return 1

    slug = slug_for(chosen.hf_model)
    target.write_text(
        f"FT_MODEL={chosen.hf_model}\n"
        f"FT_OUTPUT_DIR_CONTAINER=/output/{slug}-handschrift\n"
        f"FT_MODEL_SLUG={slug}\n"
        f"FT_NEW_ARCH={1 if chosen.note == NEW_ARCH_NOTE else 0}\n",
        encoding="ascii",
    )
    print()
    print(f"Basismodell: {chosen.hf_model}")
    print(f"Ausgabe:     /output/{slug}-handschrift (im OUTPUT_DIR aus finetune\\.env)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
