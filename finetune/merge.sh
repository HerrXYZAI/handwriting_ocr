#!/usr/bin/env bash
# Führt einen trainierten LoRA-Adapter-Checkpoint mit dem Basismodell zusammen.
# Aufruf: ./merge.sh /output/qwen3-vl-4b-handschrift/checkpoint-XXX [Ausgabeordner]
set -euo pipefail

ADAPTERS="${1:?Aufruf: merge.sh <adapter_checkpoint_verzeichnis> [ausgabeordner]}"
OUTPUT_DIR="${2:-${ADAPTERS%/}-merged}"

# train.sh trainiert per QLoRA (--quant_method bnb --quant_bits 4), das legt
# in checkpoint-XXX/args.json quant_method=bnb ab. "swift export
# --merge_lora true" liest diese args.json und uebernimmt quant_method, wenn
# es nicht per CLI explizit gesetzt ist (swift/arguments/base_args/base_args.py,
# load_args_from_ckpt). Der Merge selbst passiert zwar korrekt in voller
# Praezision (merge_lora.py setzt quant_method waehrend des Merges intern auf
# None, wegen https://github.com/huggingface/peft/issues/2321), aber danach
# fuehrt export.py trotzdem "if args.quant_method: quantize_model(args)" aus
# und quantisiert das gerade erst dequantisierte Modell direkt wieder auf
# 4-bit zurueck -- inklusive kaputtem model.safetensors.index.json vom
# ueberschriebenen Voll-Praezisions-Save (Shard-Namen stimmen dann nicht mehr
# mit der tatsaechlich gespeicherten, unsharded 4-bit-Datei ueberein, was
# spaeter bei "hf-to-gguf" mit "FileNotFoundError" abbricht).
#
# Umgehung: eine Kopie des Adapter-Checkpoints mit gepatchter args.json
# (quant_method/quant_bits entfernt) an swift export uebergeben, damit nichts
# zum Nach-Quantisieren gefunden wird.
PATCH_DIR="$(mktemp -d)"
PATCHED_ADAPTERS="$PATCH_DIR/adapter"
cp -r "$ADAPTERS" "$PATCHED_ADAPTERS"
python3 -c "
import json
path = '$PATCHED_ADAPTERS/args.json'
with open(path) as f:
    data = json.load(f)
data['quant_method'] = None
data['quant_bits'] = None
with open(path, 'w') as f:
    json.dump(data, f)
"
cleanup() { rm -rf "$PATCH_DIR"; }
trap cleanup EXIT

# Vorheriger (moeglicherweise kaputter/quantisierter) Merge-Output blockiert
# sonst stillschweigend einen erneuten Lauf: swift export ueberspringt das
# Merging komplett, wenn OUTPUT_DIR bereits existiert.
if [ -d "$OUTPUT_DIR" ]; then
    echo "Entferne vorhandenen Ausgabeordner: $OUTPUT_DIR"
    rm -rf "$OUTPUT_DIR"
fi

swift export \
    --adapters "$PATCHED_ADAPTERS" \
    --merge_lora true \
    --output_dir "$OUTPUT_DIR"

echo ""
echo "Zusammengeführtes Modell: $OUTPUT_DIR"
echo "Naechster Schritt: mit llama.cpp's convert-Skript nach GGUF konvertieren,"
echo "dann per Modelfile mit 'ollama create' registrieren."
