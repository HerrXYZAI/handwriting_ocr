# Qwen3-VL-Finetuning (separater Docker-Container)

QLoRA-Finetuning von `Qwen/Qwen3-VL-4B-Instruct` auf dem eigenen
Handschrift-Datensatz, mit [ms-swift](https://github.com/modelscope/ms-swift)
in einem eigenständigen Docker-Container (getrennt vom Ollama-Container).
Ollama selbst kann nicht trainieren – es ist nur ein Inferenz-Server für
GGUF-Modelle. Trainiert wird hier; das Ergebnis wird erst am Ende wieder in
Ollama importiert (siehe unten).

Ausgelegt für eine einzelne GPU mit ca. 12 GB VRAM (z.B. RTX 4070). QLoRA
lädt das Basismodell 4-bit-quantisiert (~3-4 GB) und trainiert nur kleine
LoRA-Adapter obendrauf – das passt auf so einer Karte deutlich sicherer als
ein volles Finetuning oder LoRA in bf16.

## Voraussetzungen

- Docker mit NVIDIA-Runtime (`docker run --gpus all ...` muss funktionieren)
- Eine reale, per Oberfläche geprüfte `train.jsonl` (siehe Hauptordner-README,
  Abschnitt "Trainings-JSONL aus einem ganzen Ordner zusammenfassen"). Für
  spürbare Verbesserungen sind deutlich mehr als ein paar Beispiele nötig.
- Falls Ollama gerade ein Modell geladen hat, belegt das VRAM, das dann beim
  Training fehlt: `ollama stop qwen3-vl:4b` vor dem Training ausführen (oder
  einfach ein paar Minuten warten, bis Ollamas Idle-Timeout das Modell von
  selbst entlädt).

## 1. Dataset konvertieren

`qwen_export_dataset.py` schreibt Bilder als eigenen Content-Block innerhalb
`messages` – ms-swift erwartet stattdessen einen `<image>`-Platzhalter im
Text plus eine separate `images`-Liste. `convert_dataset.py` übernimmt diese
Umwandlung (reines Python, keine Zusatzpakete nötig):

```powershell
python finetune\convert_dataset.py C:\Handschrift-Dataset\train.jsonl C:\Handschrift-Dataset\train_swift.jsonl --mount-point /data
```

`--mount-point` muss zu dem Pfad passen, unter dem die Dataset-Wurzel unten
in den Container gemountet wird (Standard `/data`). Bilder, deren Pfad in der
Original-JSONL bereits absolut war (Ausnahmefall, z.B. ein Bild außerhalb der
Dataset-Wurzel), bleiben unverändert – das Skript gibt dafür eine Warnung aus;
solche Pfade müssen dann separat in den Container gemountet werden.

## 2. Image bauen

```powershell
cd finetune
docker build -t handschrift-ocr-finetune .
```

Lädt ein CUDA-12.4-Devel-Image, installiert PyTorch (cu124-Build), ms-swift,
bitsandbytes, accelerate, qwen-vl-utils. Dauert beim ersten Mal einige Minuten.

## 3. Training starten

```powershell
docker run --rm -it --gpus all `
  -v C:\Handschrift-Dataset:/data:ro `
  -v C:\Handschrift-Dataset\finetune-output:/output `
  -v huggingface-cache:/root/.cache/huggingface `
  handschrift-ocr-finetune ./train.sh
```

- `/data` (read-only): die Dataset-Wurzel mit `train_swift.jsonl` und den
  Bildern, passend zu `--mount-point` aus Schritt 1.
- `/output`: Checkpoints landen hier, also auf einem echten Host-Ordner statt
  nur im Container, damit sie einen Container-Neustart überleben.
- Das benannte Volume `huggingface-cache` sorgt dafür, dass das Basismodell
  (~9 GB) nur beim ersten Lauf heruntergeladen wird, nicht bei jedem Start neu.

Wichtige Umgebungsvariablen (vor `./train.sh` mit `-e NAME=wert` setzen, oder
als zusätzliche CLI-Argumente anhängen – alles nach `./train.sh` wird 1:1 an
`swift sft` durchgereicht):

| Variable         | Standard                              | Bedeutung                          |
|------------------|----------------------------------------|-------------------------------------|
| `FT_MODEL`       | `Qwen/Qwen3-VL-4B-Instruct`            | Basismodell (Hugging Face)          |
| `FT_DATASET`     | `/data/train_swift.jsonl`              | konvertierte Trainingsdatei         |
| `FT_OUTPUT_DIR`  | `/output/qwen3-vl-4b-handschrift`      | Checkpoint-Ausgabeordner            |
| `FT_EPOCHS`      | `3`                                     | Anzahl Trainings-Epochen            |

Beispiel mit mehr Epochen und aufgetautem Vision-Tower:

```powershell
docker run --rm -it --gpus all `
  -v C:\Handschrift-Dataset:/data:ro `
  -v C:\Handschrift-Dataset\finetune-output:/output `
  -v huggingface-cache:/root/.cache/huggingface `
  -e FT_EPOCHS=5 `
  handschrift-ocr-finetune ./train.sh --freeze_vit false
```

Voreingestellt ist der Vision-Tower eingefroren (`--freeze_vit true
--freeze_aligner true`) – LoRA passt nur die Sprachmodell-Schichten an. Das
ist bei kleinen Datensätzen sicherer (weniger Overfitting-Risiko, weniger
VRAM). Erst bei deutlich mehr Trainingsdaten lohnt es sich, das mit
`--freeze_vit false` zu deaktivieren.

## 4. Adapter mit Basismodell zusammenführen

Nach dem Training liegt in `/output/.../checkpoint-XXX` nur der kleine
LoRA-Adapter, kein vollständiges Modell:

```powershell
docker run --rm -it --gpus all `
  -v C:\Handschrift-Dataset\finetune-output:/output `
  -v huggingface-cache:/root/.cache/huggingface `
  handschrift-ocr-finetune ./merge.sh /output/qwen3-vl-4b-handschrift/checkpoint-XXX
```

Schreibt das zusammengeführte, vollständige Modell nach
`/output/.../checkpoint-XXX-merged`.

## 5. Zurück nach Ollama

Ollama kann nur GGUF laden. Das zusammengeführte Modell muss also noch
konvertiert werden – dieser Schritt ist (Stand jetzt) nicht Teil dieses
Ordners, da die GGUF/Vision-Unterstützung für die Qwen3-VL-Modellfamilie in
llama.cpp sich noch schnell weiterentwickelt. Grober Ablauf, sobald das
zusammengeführte Modell vorliegt:

1. Mit llama.cpp's `convert_hf_to_gguf.py` (Text- und ggf. Vision-/mmproj-Teil
   getrennt) nach GGUF konvertieren. Vorher prüfen, ob die aktuelle
   llama.cpp-Version die genutzte Qwen3-VL-Variante bereits unterstützt.
2. `Modelfile` schreiben:
   ```
   FROM ./qwen3-vl-4b-handschrift.gguf
   ```
   (plus mmproj-Zeile, falls Vision-Unterstützung als separate Datei vorliegt)
3. `ollama create qwen3-vl-4b-handschrift -f Modelfile`
4. In `qwen_annotation_gui.py` / `qwen_preannotate.py` das Modellfeld auf
   `qwen3-vl-4b-handschrift` statt `qwen3-vl:4b` umstellen.

## Fehlerbehebung

- **`OSError: Cannot find empty port` o.ä. beim Training selbst kommt nicht
  vor** – das betrifft nur Gradio/Ollama, nicht diesen Container.
- **Out of Memory**: `FT_EPOCHS` senken hilft nicht gegen OOM (das betrifft
  nur die Trainingsdauer). Stattdessen `--max_length` senken (z.B. 2048),
  sicherstellen, dass Ollama kein Modell geladen hat, oder
  `--gradient_accumulation_steps` erhöhen bei gleichzeitig kleinerer
  `--per_device_train_batch_size` (die ist schon auf 1 voreingestellt).
- **Modell-Download hängt/bricht ab**: `HF_HOME` liegt im
  `huggingface-cache`-Volume; bei Abbruch reicht ein erneuter `docker run`
  mit demselben Volume, der Download wird fortgesetzt statt neu gestartet.
