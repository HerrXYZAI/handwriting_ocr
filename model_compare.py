"""Modellvergleich auf bereits geprüften Seiten.

Ablauf
------
1. Ein Modell wird auf jede Seite angewendet, für die eine von Hand geprüfte
   ``<bild>_annotation.json`` existiert. Das Ergebnis wird *in dieser Datei*
   unter ``model_runs[<label>]`` abgelegt (Label = Modellname, optional mit
   eigenem Zusatz wie ``qwen3-vl:4b@1536``). Die geprüften Zeilen (``lines``)
   bleiben unverändert - Export, Training und GUI sehen sie wie bisher.
2. Verglichen wird jeder Lauf mit den korrigierten Zeilen (``text_corrected``,
   ``bbox_pixels``) als Referenz:

   - **Text (Seite)**: CER/WER über den ganzen Seitentext in Leserichtung -
     unabhängig davon, wie das Modell Zeilen in Boxen aufteilt.
   - **Boxen**: Referenz- und Modellzeilen werden 1:1 über die Überlappung
     (IoU) zugeordnet; daraus Zeilen-Recall/-Precision/F1 und mittlere IoU.
   - **Text (Zeile)**: CER nur über die zugeordneten Zeilenpaare.

Ausführung über die GUI (Reiter "Modellvergleich") oder per Kommandozeile::

    python model_compare.py run    C:\\Dataset --model qwen3-vl:4b
    python model_compare.py run    C:\\Dataset --model qwen3-vl:30b-a3b-instruct --max-side 1536 --ctx 12288
    python model_compare.py report C:\\Dataset
"""

from __future__ import annotations

import argparse
import base64
import datetime as dt
import difflib
import hashlib
import html
import io
import json
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Iterable

from PIL import Image

import qwen_preannotate as qp
from tiling import Tile

try:  # Schnell (C-Implementierung), falls installiert - sonst reines Python.
    from rapidfuzz.distance import Levenshtein as _RFLevenshtein
except ImportError:  # pragma: no cover - abhängig von der Umgebung
    _RFLevenshtein = None

RUNS_KEY = "model_runs"
ANNOTATION_SUFFIX = "_annotation.json"
NO_TEXT_STATUS = "no_text"
DEFAULT_IOU = 0.3
IMAGE_EXTENSIONS = qp.IMAGE_EXTENSIONS

REF_COLOR = "#24A148"
RUN_COLORS = ("#E8590C", "#7048E8")


# ---------------------------------------------------------------------------
# Dateien
# ---------------------------------------------------------------------------

def find_annotation_files(root: str | Path) -> list[Path]:
    """Alle geprüften Annotationen unter root (rekursiv, inkl. *_pages/*_tiles)."""
    if not root or not str(root).strip():
        return []
    base = Path(root)
    if not base.is_dir():
        return []
    return sorted(base.rglob("*" + ANNOTATION_SUFFIX), key=lambda p: str(p).lower())


def image_for_annotation(annotation_path: Path, data: dict[str, Any] | None = None) -> Path | None:
    """Findet das Bild zu <stem>_annotation.json (gleicher Ordner, gleicher Stamm)."""
    stem = annotation_path.name[: -len(ANNOTATION_SUFFIX)]
    for candidate in sorted(annotation_path.parent.iterdir()):
        if candidate.is_file() and candidate.stem == stem and candidate.suffix.lower() in IMAGE_EXTENSIONS:
            return candidate
    stored = (data or {}).get("image", {}).get("file") if isinstance((data or {}).get("image"), dict) else None
    if stored and Path(stored).is_file():
        return Path(stored)
    return None


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_json_atomic(path: Path, data: dict[str, Any]) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def get_runs(data: dict[str, Any]) -> dict[str, dict[str, Any]]:
    runs = data.get(RUNS_KEY)
    return runs if isinstance(runs, dict) else {}


def store_run(annotation_path: Path, label: str, run: dict[str, Any]) -> None:
    """Liest die Datei frisch ein und ergänzt nur model_runs[label] - die
    geprüften Zeilen werden nicht angefasst."""
    data = load_json(annotation_path)
    runs = get_runs(data)
    runs[label] = run
    data[RUNS_KEY] = runs
    _write_json_atomic(annotation_path, data)


def delete_run(annotation_path: Path, label: str) -> bool:
    data = load_json(annotation_path)
    runs = get_runs(data)
    if label not in runs:
        return False
    del runs[label]
    data[RUNS_KEY] = runs
    _write_json_atomic(annotation_path, data)
    return True


def runs_on_disk(annotation_path: Path) -> dict[str, Any] | None:
    """model_runs der bereits gespeicherten Datei (für die GUI beim Speichern:
    Läufe werden nur auf der Platte gepflegt, die Platte ist maßgeblich)."""
    if not annotation_path.is_file():
        return None
    try:
        runs = load_json(annotation_path).get(RUNS_KEY)
    except (OSError, json.JSONDecodeError):
        return None
    return runs if isinstance(runs, dict) and runs else None


# ---------------------------------------------------------------------------
# Modell ausführen
# ---------------------------------------------------------------------------

def run_options(
    model: str,
    max_side: int = qp.DEFAULT_MAX_SIDE,
    ctx: int = qp.DEFAULT_CONTEXT,
    think: bool = False,
    backend: str = qp.DEFAULT_BACKEND,
    api_url: str | None = None,
    timeout: int = 1800,
    upscale: bool = False,
    no_mmap: bool = False,
) -> argparse.Namespace:
    """Dieselben Parameter, die qwen_preannotate.run_tile erwartet."""
    if api_url is None:
        api_url = qp.LLAMACPP_API if backend == "llamacpp" else qp.OLLAMA_API
    return argparse.Namespace(
        model=model.strip(),
        max_side=int(max_side),
        ctx=int(ctx),
        think=bool(think),
        backend=backend,
        api_url=api_url,
        timeout=int(timeout),
        upscale=bool(upscale),
        no_mmap=bool(no_mmap),
    )


def default_label(opts: argparse.Namespace) -> str:
    return opts.model


def run_model_on_annotation(
    annotation_path: Path,
    opts: argparse.Namespace,
    label: str | None = None,
    log_to_file: bool = True,
    quiet_console: bool = False,
) -> dict[str, Any]:
    """Wendet das Modell auf das Bild einer geprüften Annotation an und
    speichert das Ergebnis unter model_runs[label]. Es wird bewusst nicht
    gekachelt: Der Lauf soll exakt dasselbe Bild sehen, auf das sich die
    Referenzboxen beziehen."""
    label = (label or "").strip() or default_label(opts)
    data = load_json(annotation_path)
    image_path = image_for_annotation(annotation_path, data)
    if image_path is None:
        raise FileNotFoundError(f"Kein Bild zu {annotation_path.name} gefunden.")
    if int(data.get("image", {}).get("pending_rotation", 0) or 0) % 360:
        raise ValueError(f"{annotation_path.name}: Seite hat eine ungespeicherte Drehung.")

    image = qp.open_scan(image_path)
    width, height = image.size
    stored = data.get("image", {})
    if stored.get("width") and stored.get("height") and (int(stored["width"]), int(stored["height"])) != (width, height):
        raise ValueError(
            f"{annotation_path.name}: Bildgröße {width}x{height} passt nicht zur Annotation "
            f"({stored['width']}x{stored['height']})."
        )

    if log_to_file:
        log_file = annotation_path.with_name(annotation_path.name[: -len(ANNOTATION_SUFFIX)] + "_modelrun.log")
        qp.configure_logging(log_file)
        if quiet_console:
            # Gestreamte Modellausgabe nur in die Logdatei; auf der Konsole
            # bleibt die Fortschrittsanzeige lesbar (Warnungen/Fehler weiterhin).
            for handler in qp.LOG.handlers:
                if not isinstance(handler, logging.FileHandler):
                    handler.setLevel(logging.WARNING)
    qp.LOG.info("Modellvergleich: %s mit '%s' (Label '%s')", image_path.name, opts.model, label)

    started = time.monotonic()
    lines = qp.finalize_lines(qp.run_tile(Tile(1, (0, 0, width, height), image), opts))
    duration = time.monotonic() - started

    run = {
        "model": opts.model,
        "created": dt.datetime.now().isoformat(timespec="seconds"),
        "duration_s": round(duration, 1),
        "image_size": [width, height],
        "settings": {
            "backend": opts.backend,
            "max_side": opts.max_side,
            "context_size": opts.ctx,
            "think": opts.think,
            "upscale": opts.upscale,
            "no_mmap": getattr(opts, "no_mmap", False),
            "prompt_sha1": hashlib.sha1(qp.PROMPT.encode("utf-8")).hexdigest()[:10],
        },
        "lines": [
            {
                "bbox_pixels": line["bbox_pixels"],
                "text": line["text"],
                "confidence": line["confidence"],
                "angle": line.get("angle", 0.0),
            }
            for line in lines
        ],
    }
    store_run(annotation_path, label, run)
    qp.LOG.info("Lauf gespeichert: %s -> %s[%s] (%d Zeilen, %.1f s)", annotation_path.name, RUNS_KEY, label, len(lines), duration)
    return run


# ---------------------------------------------------------------------------
# Metriken
# ---------------------------------------------------------------------------

_UNCERTAIN_RE = re.compile(r"\[\?\]")
_SPACE_RE = re.compile(r"\s+")


def normalize_text(text: Any) -> str:
    """Entfernt Unsicherheitsmarker [?] (die Referenz enthält sie nach der
    Korrektur meist nicht mehr) und vereinheitlicht Leerraum."""
    value = _UNCERTAIN_RE.sub("", str(text or ""))
    return _SPACE_RE.sub(" ", value).strip()


def edit_distance(a: str | list[str], b: str | list[str]) -> int:
    if _RFLevenshtein is not None:
        return int(_RFLevenshtein.distance(a, b))
    if len(a) < len(b):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for i, item_a in enumerate(a, 1):
        current = [i]
        for j, item_b in enumerate(b, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (item_a != item_b)))
        previous = current
    return previous[-1]


def iou(a: list[float], b: list[float]) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter)


def _reading_order(lines: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(lines, key=lambda line: (line["box"][1], line["box"][0]))


def reference_lines(data: dict[str, Any]) -> list[dict[str, Any]]:
    """Geprüfte Zeilen mit Text; als 'kein sichtbarer Text' markierte entfallen."""
    result = []
    for line in data.get("lines", []):
        if not isinstance(line, dict) or line.get("status") == NO_TEXT_STATUS:
            continue
        text = normalize_text(line.get("text_corrected", line.get("text", "")))
        box = line.get("bbox_pixels")
        if not text or not isinstance(box, list) or len(box) != 4:
            continue
        result.append({"box": [float(v) for v in box], "text": text, "angle": float(line.get("angle", 0) or 0)})
    return _reading_order(result)


def run_lines(run: dict[str, Any]) -> list[dict[str, Any]]:
    result = []
    for line in run.get("lines", []):
        text = normalize_text(line.get("text", ""))
        box = line.get("bbox_pixels")
        if not text or not isinstance(box, list) or len(box) != 4:
            continue
        result.append({
            "box": [float(v) for v in box],
            "text": text,
            "angle": float(line.get("angle", 0) or 0),
            "confidence": line.get("confidence", "low"),
        })
    return _reading_order(result)


def match_lines(
    ref: list[dict[str, Any]], pred: list[dict[str, Any]], threshold: float = DEFAULT_IOU
) -> tuple[dict[int, tuple[int, float]], set[int]]:
    """Greedy 1:1-Zuordnung nach absteigender IoU. Liefert {ref_index:
    (pred_index, iou)} und die Menge nicht zugeordneter Modellzeilen."""
    candidates = []
    for ri, r in enumerate(ref):
        for pi, p in enumerate(pred):
            value = iou(r["box"], p["box"])
            if value >= threshold:
                candidates.append((value, ri, pi))
    candidates.sort(reverse=True)
    matched: dict[int, tuple[int, float]] = {}
    used: set[int] = set()
    for value, ri, pi in candidates:
        if ri in matched or pi in used:
            continue
        matched[ri] = (pi, value)
        used.add(pi)
    return matched, set(range(len(pred))) - used


def run_matches_image(data: dict[str, Any], run: dict[str, Any]) -> bool:
    size = run.get("image_size")
    image = data.get("image", {})
    if not size or not image.get("width"):
        return True
    return [int(image["width"]), int(image["height"])] == [int(v) for v in size]


def page_metrics(data: dict[str, Any], run: dict[str, Any], threshold: float = DEFAULT_IOU) -> dict[str, Any]:
    ref = reference_lines(data)
    pred = run_lines(run)
    ref_text = " ".join(line["text"] for line in ref)
    pred_text = " ".join(line["text"] for line in pred)
    ref_words, pred_words = ref_text.split(), pred_text.split()
    matched, _ = match_lines(ref, pred, threshold)

    matched_errors = sum(edit_distance(ref[ri]["text"], pred[pi]["text"]) for ri, (pi, _) in matched.items())
    matched_chars = sum(len(ref[ri]["text"]) for ri in matched)
    return {
        "ref_chars": len(ref_text),
        "char_errors": edit_distance(ref_text, pred_text),
        "ref_words": len(ref_words),
        "word_errors": edit_distance(ref_words, pred_words),
        "n_ref": len(ref),
        "n_pred": len(pred),
        "n_matched": len(matched),
        "iou_sum": sum(value for _, value in matched.values()),
        "matched_chars": matched_chars,
        "matched_errors": matched_errors,
        "duration_s": float(run.get("duration_s") or 0),
        "pages": 1,
        "stale": not run_matches_image(data, run),
    }


def _ratio(num: float, den: float) -> float | None:
    return num / den if den else None


def finish_metrics(total: dict[str, Any]) -> dict[str, Any]:
    """Aus aufsummierten Zählern die Kennzahlen (mikro-gemittelt) berechnen."""
    precision = _ratio(total["n_matched"], total["n_pred"])
    recall = _ratio(total["n_matched"], total["n_ref"])
    f1 = 2 * precision * recall / (precision + recall) if precision and recall else (0.0 if precision is not None and recall is not None else None)
    cer = _ratio(total["char_errors"], total["ref_chars"])
    if cer is None and total["char_errors"]:
        cer = 1.0
    return {
        **total,
        "cer": cer,
        "wer": _ratio(total["word_errors"], total["ref_words"]),
        "line_cer": _ratio(total["matched_errors"], total["matched_chars"]),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mean_iou": _ratio(total["iou_sum"], total["n_matched"]),
        "avg_duration_s": _ratio(total["duration_s"], total["pages"]),
    }


_SUM_KEYS = (
    "ref_chars", "char_errors", "ref_words", "word_errors", "n_ref", "n_pred", "n_matched",
    "iou_sum", "matched_chars", "matched_errors", "duration_s", "pages",
)


def summarize(
    root: str | Path,
    threshold: float = DEFAULT_IOU,
    only_common: bool = True,
    labels: Iterable[str] | None = None,
) -> tuple[list[dict[str, Any]], int, int]:
    """Kennzahlen je Lauf-Label über alle geprüften Seiten. Mit only_common
    zählen nur Seiten, auf denen *alle* (gewählten) Labels gelaufen sind -
    sonst wären die Zahlen nicht direkt vergleichbar.

    Liefert (Zeilen je Label, Anzahl geprüfter Seiten, Anzahl verglichener Seiten)."""
    pages = []
    for path in find_annotation_files(root):
        try:
            data = load_json(path)
        except (OSError, json.JSONDecodeError):
            continue
        runs = {k: v for k, v in get_runs(data).items() if run_matches_image(data, v)}
        pages.append((data, runs))

    wanted = set(labels) if labels else set().union(*(set(r) for _, r in pages)) if pages else set()
    if only_common:
        considered = [(d, r) for d, r in pages if wanted and wanted <= set(r)]
    else:
        considered = pages

    totals: dict[str, dict[str, Any]] = {}
    for data, runs in considered:
        for label in wanted:
            if label not in runs:
                continue
            metrics = page_metrics(data, runs[label], threshold)
            total = totals.setdefault(label, {key: 0 for key in _SUM_KEYS} | {"model": runs[label].get("model", label)})
            for key in _SUM_KEYS:
                total[key] += metrics[key]
    rows = [finish_metrics(total) | {"label": label} for label, total in totals.items()]
    rows.sort(key=lambda row: (row["cer"] is None, row["cer"] if row["cer"] is not None else 0))
    return rows, len(pages), len(considered)


def fmt_pct(value: float | None) -> str:
    return "–" if value is None else f"{value * 100:.1f} %"


def fmt_num(value: float | None, digits: int = 2) -> str:
    return "–" if value is None else f"{value:.{digits}f}"


def fmt_duration(seconds: float | None) -> str:
    if seconds is None:
        return "–"
    if seconds < 90:
        return f"{seconds:.0f} s"
    return f"{seconds / 60:.1f} min"


SUMMARY_HEADERS = [
    "Lauf (Label)", "Seiten", "CER Seite", "WER Seite", "CER Zeilen",
    "Zeilen-Recall", "Zeilen-Precision", "F1", "Ø IoU", "Ø Zeit/Seite",
]


def summary_table(rows: list[dict[str, Any]]) -> list[list[Any]]:
    return [
        [
            row["label"], row["pages"], fmt_pct(row["cer"]), fmt_pct(row["wer"]), fmt_pct(row["line_cer"]),
            fmt_pct(row["recall"]), fmt_pct(row["precision"]), fmt_pct(row["f1"]),
            fmt_num(row["mean_iou"]), fmt_duration(row["avg_duration_s"]),
        ]
        for row in rows
    ]


# ---------------------------------------------------------------------------
# HTML-Darstellung (für die GUI)
# ---------------------------------------------------------------------------

def _esc(text: Any) -> str:
    return html.escape(str(text), quote=True)


_TOKEN_RE = re.compile(r"\w+|\s+|[^\w\s]", re.UNICODE)


def _char_diff(reference: str, hypothesis: str) -> list[str]:
    matcher = difflib.SequenceMatcher(None, reference, hypothesis, autojunk=False)
    parts = []
    for op, a1, a2, b1, b2 in matcher.get_opcodes():
        if op == "equal":
            parts.append(_esc(reference[a1:a2]))
            continue
        if a2 > a1:
            parts.append(f"<del class='mc-del'>{_esc(reference[a1:a2])}</del>")
        if b2 > b1:
            parts.append(f"<ins class='mc-ins'>{_esc(hypothesis[b1:b2])}</ins>")
    return parts


def diff_html(reference: str, hypothesis: str) -> str:
    """Unterschied Referenz -> Modelltext: rot durchgestrichen = fehlt/falsch
    im Modelltext, grün = vom Modell stattdessen geschrieben. Zuerst wird
    wortweise verglichen (ganze fehlende Wörter erscheinen als ein Block),
    innerhalb ersetzter Wörter dann zeichenweise (z.B. 'angekom~~m~~en')."""
    ref_tokens = _TOKEN_RE.findall(reference)
    hyp_tokens = _TOKEN_RE.findall(hypothesis)
    matcher = difflib.SequenceMatcher(None, ref_tokens, hyp_tokens, autojunk=False)
    parts: list[str] = []
    for op, a1, a2, b1, b2 in matcher.get_opcodes():
        ref_part, hyp_part = "".join(ref_tokens[a1:a2]), "".join(hyp_tokens[b1:b2])
        if op == "equal":
            parts.append(_esc(ref_part))
        elif op == "replace" and difflib.SequenceMatcher(None, ref_part, hyp_part, autojunk=False).ratio() >= 0.5:
            parts.extend(_char_diff(ref_part, hyp_part))
        else:
            if ref_part:
                parts.append(f"<del class='mc-del'>{_esc(ref_part)}</del>")
            if hyp_part:
                parts.append(f"<ins class='mc-ins'>{_esc(hyp_part)}</ins>")
    return "".join(parts)


def _encode_display_image(image: Image.Image, max_dim: int = 1200) -> str:
    display = image.copy()
    display.thumbnail((max_dim, max_dim), Image.LANCZOS)
    buffer = io.BytesIO()
    display.save(buffer, format="JPEG", quality=82)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def _box_div(box: list[float], width: int, height: int, color: str, label: str, dashed: bool, angle: float) -> str:
    x1, y1, x2, y2 = box
    style = (
        f"left:{x1 / width * 100:.3f}%;top:{y1 / height * 100:.3f}%;"
        f"width:{(x2 - x1) / width * 100:.3f}%;height:{(y2 - y1) / height * 100:.3f}%;"
        f"border:2px {'dashed' if dashed else 'solid'} {color};"
    )
    if angle:
        style += f"transform:rotate({angle:.2f}deg);"
    tag_pos = "bottom:-18px;right:-2px;" if dashed else "top:-18px;left:-2px;"
    return (
        f"<div class='mc-box' style='{style}'>"
        f"<span class='mc-tag' style='background:{color};{tag_pos}'>{_esc(label)}</span></div>"
    )


def overlay_html(
    image_path: Path | None, data: dict[str, Any], run: dict[str, Any] | None, color: str, title: str, threshold: float
) -> str:
    """Bild mit Referenzboxen (grün, durchgezogen, Nummer oben links) und
    Modellboxen (gestrichelt, Nummer der zugeordneten Referenzzeile unten
    rechts bzw. '+' für zusätzliche Zeilen)."""
    if image_path is None:
        return "<div class='mc-empty'>Kein Bild gefunden.</div>"
    image = qp.open_scan(image_path)
    width, height = image.size
    ref = reference_lines(data)
    parts = [_box_div(line["box"], width, height, REF_COLOR, str(i + 1), False, line["angle"]) for i, line in enumerate(ref)]
    if run is not None:
        pred = run_lines(run)
        matched, extra = match_lines(ref, pred, threshold)
        pred_to_ref = {pi: ri for ri, (pi, _) in matched.items()}
        for pi, line in enumerate(pred):
            tag = str(pred_to_ref[pi] + 1) if pi in pred_to_ref else "+"
            parts.append(_box_div(line["box"], width, height, color if pi not in extra else "#DA1E28", tag, True, line["angle"]))
    legend = (
        f"<div class='mc-legend'><b>{_esc(title)}</b> &nbsp; "
        f"<span style='color:{REF_COLOR}'>━ Referenz</span> &nbsp; "
        f"<span style='color:{color}'>┅ Modell (zugeordnet)</span> &nbsp; "
        f"<span style='color:#DA1E28'>┅ Modell (zusätzlich, ohne Referenz)</span></div>"
    )
    return (
        f"{legend}<div class='mc-canvas'><img class='mc-image' src='{_encode_display_image(image)}' />"
        f"{''.join(parts)}</div>"
    )


def page_metrics_html(metrics_by_label: list[tuple[str, dict[str, Any]]]) -> str:
    if not metrics_by_label:
        return "<div class='mc-empty'>Für diese Seite gibt es noch keinen Modelllauf.</div>"
    head = "".join(f"<th>{_esc(h)}</th>" for h in SUMMARY_HEADERS)
    rows = []
    for label, metrics in metrics_by_label:
        cells = summary_table([finish_metrics(metrics) | {"label": label}])[0]
        warning = " ⚠ Bildgröße geändert - neu ausführen" if metrics.get("stale") else ""
        cells[0] = f"{cells[0]}{warning}"
        rows.append("<tr>" + "".join(f"<td>{_esc(c)}</td>" for c in cells) + "</tr>")
    return f"<table class='mc-table'><thead><tr>{head}</tr></thead><tbody>{''.join(rows)}</tbody></table>"


def line_table_html(data: dict[str, Any], runs: list[tuple[str, dict[str, Any]]], threshold: float) -> str:
    """Zeilenweiser Textvergleich: je Referenzzeile die zugeordnete Modellzeile
    jedes Laufs als Diff, darunter zusätzliche Modellzeilen ohne Referenz."""
    ref = reference_lines(data)
    per_run = []
    for label, run in runs:
        pred = run_lines(run)
        matched, extra = match_lines(ref, pred, threshold)
        per_run.append((label, pred, matched, extra))

    head = "<th>#</th><th>Referenz (geprüft)</th>" + "".join(f"<th>{_esc(label)}</th>" for label, *_ in per_run)
    rows = []
    for ri, line in enumerate(ref):
        cells = [f"<td class='mc-num'>{ri + 1}</td>", f"<td>{_esc(line['text'])}</td>"]
        for _, pred, matched, _ in per_run:
            if ri not in matched:
                cells.append("<td class='mc-missing'>— nicht gefunden (keine passende Box)</td>")
                continue
            pi, overlap = matched[ri]
            hyp = pred[pi]["text"]
            errors = edit_distance(line["text"], hyp)
            cer = errors / len(line["text"]) if line["text"] else 0.0
            cells.append(
                f"<td>{diff_html(line['text'], hyp)}"
                f"<div class='mc-meta'>CER {cer * 100:.0f} % · IoU {overlap:.2f}</div></td>"
            )
        rows.append("<tr>" + "".join(cells) + "</tr>")

    for label, pred, _, extra in per_run:
        for pi in sorted(extra):
            cells = ["<td class='mc-num'>+</td>", "<td class='mc-missing'>— keine Referenzzeile</td>"]
            for other_label, *_ in per_run:
                cells.append(
                    f"<td><ins class='mc-ins'>{_esc(pred[pi]['text'])}</ins></td>" if other_label == label else "<td></td>"
                )
            rows.append("<tr class='mc-extra'>" + "".join(cells) + "</tr>")

    if not rows:
        return "<div class='mc-empty'>Keine Zeilen.</div>"
    return f"<table class='mc-table mc-lines'><thead><tr>{head}</tr></thead><tbody>{''.join(rows)}</tbody></table>"


COMPARE_STYLE = """
<style>
.mc-canvas { position: relative; display: inline-block; max-width: 100%; line-height: 0; }
.mc-image { display: block; width: 100%; height: auto; }
.mc-box { position: absolute; box-sizing: border-box; pointer-events: none; transform-origin: 50% 50%; }
.mc-tag { position: absolute; color: #fff; font: bold 11px/1.4 sans-serif; padding: 0 4px; border-radius: 3px; white-space: nowrap; }
.mc-legend { font-size: 13px; margin: 4px 0 8px; line-height: 1.5; }
.mc-table { border-collapse: collapse; width: 100%; font-size: 14px; }
.mc-table th, .mc-table td { border: 1px solid rgba(128,128,128,.35); padding: 4px 8px; vertical-align: top; text-align: left; }
.mc-table th { background: rgba(128,128,128,.12); }
.mc-lines td { font-family: ui-monospace, Consolas, monospace; white-space: pre-wrap; }
.mc-num { text-align: right !important; color: #888; width: 2.5em; }
.mc-meta { font-family: sans-serif; font-size: 11px; color: #888; margin-top: 2px; }
.mc-missing { color: #888; font-style: italic; }
.mc-extra td { background: rgba(218,30,40,.06); }
.mc-del { background: rgba(218,30,40,.22); color: inherit; text-decoration: line-through; }
.mc-ins { background: rgba(36,161,72,.25); color: inherit; text-decoration: none; }
.mc-empty { padding: 12px; color: #888; }
</style>
"""


# ---------------------------------------------------------------------------
# Kommandozeile
# ---------------------------------------------------------------------------

def _print_summary(root: str, threshold: float, only_common: bool) -> None:
    rows, total_pages, compared = summarize(root, threshold, only_common)
    print(f"Geprüfte Seiten: {total_pages}, verglichen: {compared} (IoU-Schwelle {threshold})")
    if not rows:
        print("Noch keine Modellläufe vorhanden.")
        return
    table = [SUMMARY_HEADERS, *summary_table(rows)]
    widths = [max(len(str(row[i])) for row in table) for i in range(len(SUMMARY_HEADERS))]
    for index, row in enumerate(table):
        print("  ".join(str(cell).ljust(widths[i]) for i, cell in enumerate(row)))
        if index == 0:
            print("  ".join("-" * w for w in widths))


def main() -> None:
    parser = argparse.ArgumentParser(description="Modelle auf geprüften Seiten vergleichen (Ergebnisse in *_annotation.json unter model_runs).")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="Modell auf allen geprüften Seiten ausführen und Ergebnis speichern")
    run.add_argument("root", help="Dataset-Ordner (rekursiv nach *_annotation.json durchsucht)")
    run.add_argument("--model", required=True)
    run.add_argument("--label", help="Name des Laufs; Standard: Modellname. Z.B. für denselben Lauf mit anderer --max-side.")
    run.add_argument("--max-side", type=qp.positive_int, default=qp.DEFAULT_MAX_SIDE)
    run.add_argument("--ctx", type=qp.positive_int, default=qp.DEFAULT_CONTEXT)
    run.add_argument("--think", action="store_true")
    run.add_argument(
        "--no-mmap",
        action="store_true",
        help="Modell komplett in den Arbeitsspeicher laden (Ollama use_mmap=false); empfohlen für große Modelle",
    )
    run.add_argument("--upscale", action="store_true")
    run.add_argument("--backend", choices=("ollama", "llamacpp"), default=qp.DEFAULT_BACKEND)
    run.add_argument("--api-url", default=None)
    run.add_argument("--timeout", type=qp.positive_int, default=1800)
    run.add_argument("--limit", type=qp.positive_int, help="Höchstens so viele Seiten verarbeiten")
    run.add_argument("--force", action="store_true", help="Vorhandene Läufe mit gleichem Label neu erzeugen")

    report = sub.add_parser("report", help="Vergleichstabelle ausgeben")
    report.add_argument("root")
    report.add_argument("--iou", type=float, default=DEFAULT_IOU, help=f"IoU-Schwelle für die Box-Zuordnung; Standard {DEFAULT_IOU}")
    report.add_argument("--all-pages", action="store_true", help="Auch Seiten zählen, auf denen nicht alle Läufe vorhanden sind")

    args = parser.parse_args()
    if args.command == "report":
        _print_summary(args.root, args.iou, not args.all_pages)
        return

    opts = run_options(
        args.model, args.max_side, args.ctx, args.think, args.backend, args.api_url, args.timeout, args.upscale,
        args.no_mmap,
    )
    label = args.label or default_label(opts)
    files = find_annotation_files(args.root)
    todo = []
    for path in files:
        try:
            if args.force or label not in get_runs(load_json(path)):
                todo.append(path)
        except (OSError, json.JSONDecodeError) as error:
            print(f"Übersprungen (nicht lesbar): {path}: {error}", file=sys.stderr)
    if args.limit:
        todo = todo[: args.limit]
    print(f"{len(files)} geprüfte Seiten, {len(todo)} davon ohne Lauf '{label}' -> werden verarbeitet.")

    durations = []
    for index, path in enumerate(todo, 1):
        eta = ""
        if durations:
            remaining = sum(durations) / len(durations) * (len(todo) - index + 1)
            eta = f", Rest ca. {fmt_duration(remaining)}"
        print(f"[{index}/{len(todo)}] {path}{eta}", flush=True)
        try:
            result = run_model_on_annotation(path, opts, label, quiet_console=True)
            durations.append(result["duration_s"])
            print(
                f"    {len(result['lines'])} Zeilen in {fmt_duration(result['duration_s'])} "
                f"(Details: {path.name[: -len(ANNOTATION_SUFFIX)]}_modelrun.log)"
            )
        except KeyboardInterrupt:
            print("Abgebrochen. Bereits fertige Seiten sind gespeichert.")
            break
        except Exception as error:  # eine Seite darf den Lauf nicht abbrechen
            print(f"    FEHLER: {error}", file=sys.stderr)
    print()
    _print_summary(args.root, DEFAULT_IOU, True)


if __name__ == "__main__":
    logging.getLogger("qwen_preannotate").setLevel(logging.INFO)
    main()
