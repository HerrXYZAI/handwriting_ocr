#!/usr/bin/env bash
# Führt einen trainierten LoRA-Adapter-Checkpoint mit dem Basismodell zusammen.
# Aufruf: ./merge.sh /output/qwen3-vl-4b-handschrift/checkpoint-XXX [Ausgabeordner]
set -euo pipefail

ADAPTERS="${1:?Aufruf: merge.sh <adapter_checkpoint_verzeichnis> [ausgabeordner]}"
OUTPUT_DIR="${2:-${ADAPTERS%/}-merged}"

swift export \
    --adapters "$ADAPTERS" \
    --merge_lora true \
    --output_dir "$OUTPUT_DIR"

echo ""
echo "Zusammengeführtes Modell: $OUTPUT_DIR"
echo "Naechster Schritt: mit llama.cpp's convert-Skript nach GGUF konvertieren,"
echo "dann per Modelfile mit 'ollama create' registrieren."
