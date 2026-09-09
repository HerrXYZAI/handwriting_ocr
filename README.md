# Qwen Handschrift-Annotation

Lokale Windows-Anwendung für:

- Vorannotation gescannter Handschrift mit einem Qwen-VL-Modell über Ollama
- Zeilenboxen mit normalisierten Koordinaten von 0 bis 1000
- visuelle Prüfung und Textkorrektur in Gradio
- Speicherung einer ausführlichen Annotations-JSON
- Export als multimodale Qwen-Trainings-JSONL
- automatische Aktualisierung statt doppeltem Export derselben Bilddatei

## Voraussetzungen

- Windows 11
- Python 3.11 oder 3.12 empfohlen
- laufendes Ollama
- ein installiertes Vision-Modell, dessen exakter Name in `ollama list` erscheint

## Installation

PowerShell im Projektordner öffnen:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Falls PowerShell die Aktivierung sperrt, kann alternativ `run.bat` verwendet werden.

## Modell bereitstellen

Beispiel, sofern dieser Modellname in deiner Ollama-Version verfügbar ist:

```powershell
ollama pull qwen3-vl:4b
ollama list
```

Den exakten Namen aus `ollama list` in der Oberfläche eintragen.

## Start

```powershell
python qwen_annotation_gui.py
```

Oder `run.bat` doppelklicken. Anwendung: `http://127.0.0.1:7860`

## Empfohlene Dataset-Struktur

```text
C:\Handschrift-Dataset\
├── images\
│   ├── seite_0001.png
│   └── seite_0002.png
├── annotations\
└── train.jsonl
```

Als Dataset-Wurzel `C:\Handschrift-Dataset` angeben. Bilder innerhalb dieser Wurzel werden mit relativem Pfad exportiert.

## Workflow

1. Originalscan laden.
2. Modellnamen und Kontextgröße kontrollieren.
3. Qwen-Vorannotation starten.
4. Tabellenzeile auswählen und Text korrigieren.
5. Bei Bedarf Boxwerte im Bereich 0 bis 1000 verändern.
6. Änderungen übernehmen.
7. Annotations-JSON speichern.
8. Als Trainings-JSONL exportieren.

Beim erneuten Export derselben Bilddatei wird deren vorhandener JSONL-Datensatz ersetzt, nicht verdoppelt.

## Datenschutz

Standardmäßig bindet Gradio nur an `127.0.0.1`. Für vertrauliche Scans `--share` nicht benutzen. Ollama wird lokal über `127.0.0.1:11434` angesprochen.

## Ausgabeformat

Die Annotations-JSON bewahrt Vorhersage, Korrektur, Konfidenz, Status und Position. Die Trainings-JSONL enthält pro Seite eine User-Nachricht mit Bild und Prompt sowie eine Assistant-Nachricht mit korrigierten Zeilenboxen und Text.

## Hinweise

- Qwen-Ausgaben und Boxen müssen fachlich geprüft werden.
- Namen, Zahlen, Messwerte und Einheiten besonders sorgfältig kontrollieren.
- Ein Modell-Finetuning selbst ist nicht Bestandteil dieser Anwendung.
