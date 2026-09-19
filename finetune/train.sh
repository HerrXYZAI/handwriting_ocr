#!/usr/bin/env bash
# QLoRA-Finetuning von Qwen3-VL auf dem Handschrift-Datensatz mit ms-swift.
# Alle Werte per Umgebungsvariable überschreibbar; zusätzliche Argumente
# werden unverändert an "swift sft" durchgereicht, z.B.:
#   ./train.sh --num_train_epochs 5 --freeze_vit false
set -euo pipefail

MODEL="${FT_MODEL:-Qwen/Qwen3-VL-4B-Instruct}"
DATASET="${FT_DATASET:-/data/train_swift.jsonl}"
OUTPUT_DIR="${FT_OUTPUT_DIR:-/output/qwen3-vl-4b-handschrift}"
EPOCHS="${FT_EPOCHS:-3}"
# Deckelt die Bildaufloesung vor dem Encoding (Qwen macht daraus ~1 Vision-
# Token je 28x28px-Patch). Ohne Deckel erzeugen grosse Scans (z.B. 2480x2200)
# ueber 6000 Bild-Tokens allein und sprengen selbst grosszuegige max_length-
# Werte, wodurch das Sample beim Laden verworfen wird.
MAX_PIXELS="${FT_MAX_PIXELS:-1003520}"
MAX_LENGTH="${FT_MAX_LENGTH:-8192}"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

swift sft \
    --model "$MODEL" \
    --tuner_type lora \
    --quant_method bnb \
    --quant_bits 4 \
    --bnb_4bit_quant_type nf4 \
    --torch_dtype bfloat16 \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --freeze_vit true \
    --freeze_aligner true \
    --gradient_checkpointing true \
    --dataset "$DATASET" \
    --split_dataset_ratio 0.05 \
    --num_train_epochs "$EPOCHS" \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --learning_rate 1e-4 \
    --max_pixels "$MAX_PIXELS" \
    --max_length "$MAX_LENGTH" \
    --eval_steps 50 \
    --save_steps 50 \
    --save_total_limit 2 \
    --logging_steps 5 \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 2 \
    --output_dir "$OUTPUT_DIR" \
    "$@"
