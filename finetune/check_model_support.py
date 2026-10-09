"""Prüft im Trainings-Image, ob das installierte ms-swift ein Modell kennt.

Aufruf (von run.bat, im vorhandenen Image, ohne es neu zu bauen):
    docker run --rm -v "<finetune>:/chk:ro" --entrypoint python3 \
        handschrift-ocr-finetune:latest /chk/check_model_support.py Qwen/Qwen3.5-9B

ms-swift registriert jedes unterstützte Modell mit seiner Hugging-Face-ID im
Quelltext. Taucht die ID dort auf, ist das Paket aktuell genug und muss nicht
neu installiert werden.

Exit-Code 0 = unterstützt, 1 = nicht gefunden (Pakete aktualisieren),
2 = Prüfung nicht möglich.
"""

from __future__ import annotations

import os
import sys


def main() -> int:
    if len(sys.argv) != 2:
        return 2
    model_id = sys.argv[1]
    try:
        import swift  # noqa: F401
    except Exception:
        return 2
    root = os.path.dirname(swift.__file__)
    needles = (model_id, model_id.lower())
    for directory, _, files in os.walk(root):
        for name in files:
            if not name.endswith(".py"):
                continue
            try:
                with open(os.path.join(directory, name), encoding="utf-8", errors="ignore") as handle:
                    content = handle.read()
            except OSError:
                continue
            if any(needle in content for needle in needles):
                version = getattr(sys.modules["swift"], "__version__", "?")
                print(f"ms-swift {version} kennt {model_id} bereits.")
                return 0
    print(f"{model_id} ist im installierten ms-swift noch nicht bekannt.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
