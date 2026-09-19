from __future__ import annotations

import base64
import io

import pytesseract
from flask import Flask, jsonify, request
from PIL import Image

app = Flask(__name__)


def group_words_into_lines(data: dict) -> list[dict]:
    """Fasst Tesseracts wortweise TSV-Ausgabe (image_to_data) zu Zeilen-Boxen
    zusammen, gruppiert nach (block, paragraph, line). Der erkannte Text wird
    mitgeliefert, ist bei Handschrift aber meist unzuverlässig - Aufrufer
    sollten sich auf die Boxgeometrie verlassen, nicht auf den Text.
    """
    lines: dict[tuple[int, int, int], dict] = {}
    count = len(data["level"])
    for i in range(count):
        if data["level"][i] != 5:  # 5 = Wort-Ebene
            continue
        try:
            conf = float(data["conf"][i])
        except (TypeError, ValueError):
            continue
        if conf < 0:
            continue
        text = str(data["text"][i]).strip()
        key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
        x, y, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
        line = lines.setdefault(key, {"x1": x, "y1": y, "x2": x + w, "y2": y + h, "words": [], "confs": []})
        line["x1"] = min(line["x1"], x)
        line["y1"] = min(line["y1"], y)
        line["x2"] = max(line["x2"], x + w)
        line["y2"] = max(line["y2"], y + h)
        if text:
            line["words"].append(text)
            line["confs"].append(conf)

    result = []
    for line in lines.values():
        if not line["words"]:
            continue
        result.append({
            "bbox_pixels": [line["x1"], line["y1"], line["x2"], line["y2"]],
            "text": " ".join(line["words"]),
            "confidence": round(sum(line["confs"]) / len(line["confs"]), 1),
        })
    return result


@app.route("/ocr", methods=["POST"])
def ocr():
    payload = request.get_json(force=True, silent=True) or {}
    image_b64 = payload.get("image")
    if not image_b64:
        return jsonify({"error": "Kein Bild übergeben."}), 400
    lang = str(payload.get("lang", "deu"))
    try:
        psm = int(payload.get("psm", 11))
    except (TypeError, ValueError):
        psm = 11

    try:
        image = Image.open(io.BytesIO(base64.b64decode(image_b64))).convert("RGB")
    except Exception as exc:
        return jsonify({"error": f"Bild konnte nicht dekodiert werden: {exc}"}), 400

    try:
        data = pytesseract.image_to_data(
            image, lang=lang, config=f"--psm {psm}", output_type=pytesseract.Output.DICT
        )
    except Exception as exc:
        return jsonify({"error": f"Tesseract-Fehler: {exc}"}), 500

    lines = group_words_into_lines(data)
    return jsonify({"lines": lines, "width": image.width, "height": image.height})


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8884)
