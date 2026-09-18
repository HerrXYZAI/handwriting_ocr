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

## PDF-Scans

Sowohl die Oberfläche als auch die Kommandozeilen-Vorannotation akzeptieren PDF-Dateien
zusätzlich zu Bilddateien. Jede PDF-Seite wird mit 300 DPI (Oberfläche fest, CLI über
`--pdf-dpi` einstellbar) in ein PNG neben der PDF gerastert (`<pdf-name>_pages\`) und wie
ein normaler Scan weiterverarbeitet. Bereits gerasterte Seiten werden bei erneutem Laden
wiederverwendet.

- Oberfläche: PDF über "PDF-Scan (mehrseitig)" hochladen, Seitenzahl eintragen, "PDF-Seite
  laden" klicken. Die gerasterte Seite erscheint im Originalscan-Feld und wird wie ein
  Bild-Upload behandelt.
- Kommandozeile: `python qwen_preannotate.py scan.pdf` verarbeitet automatisch jede Seite
  und schreibt `<pdf-name>_p001_preannotation.json`, `<pdf-name>_p002_preannotation.json`
  usw. Bei mehrseitigen PDFs sind `--output`/`--log-file` nicht zulässig, da die Namen je
  Seite automatisch vergeben werden.

## Großformatige Scans (Kacheln)

Übersteigt ein Bild `--tile-trigger` (Standard 2800 px Kantenlänge), wird es in
überlappende Kacheln aufgeteilt. Jede Kachel wird als eigenständige Bilddatei in
`<scan-name>_tiles\` gespeichert und einzeln vorannotiert bzw. annotiert – es gibt
keine automatische Zusammenführung zu einer Seitenannotation mehr. Duplikate im
Überlappungsbereich zweier Kacheln bleiben also als getrennte Einträge in den
jeweiligen Kachel-Annotationen bestehen.

- Oberfläche: Scan laden, "Originalscan in Kacheln aufteilen" klicken. Ist das Bild
  klein genug, passiert nichts. Sonst werden die Kacheln gespeichert; Kachelnummer
  eintragen und "Kachel laden" klicken, um sie wie einen eigenen Scan zu vorannotieren,
  zu korrigieren und zu speichern.
- Kommandozeile: `python qwen_preannotate.py scan.png` erkennt automatisch, ob eine
  Aufteilung nötig ist, und schreibt bei Aufteilung `<scan-name>_tiles\<scan-name>_tile001_preannotation.json`
  usw. statt einer einzelnen `<scan-name>_preannotation.json`. `--output`/`--log-file`
  sind dann nicht zulässig, da die Namen je Kachel automatisch vergeben werden.

## Ordner-Stapelverarbeitung (nur Kommandozeile)

`python qwen_preannotate.py C:\Ordner` verarbeitet statt einer einzelnen Datei jede
Bild- oder PDF-Datei direkt in diesem Ordner (nicht rekursiv in Unterordnern). Dateien,
für die bereits eine Vorannotation existiert (`<name>_preannotation.json`, oder bei
Kacheln/PDF-Seiten mindestens eine `*_preannotation.json` im zugehörigen
`<name>_tiles\`- bzw. `<name>_pages\`-Ordner), werden übersprungen. Damit lässt sich ein
abgebrochener oder erweiterter Ordnerlauf einfach fortsetzen, ohne bereits fertige
Dateien erneut zu verarbeiten. `--output`/`--log-file` sind bei einem Ordner nicht
zulässig, da die Namen je Datei automatisch vergeben werden. Ein Fehler bei einer Datei
bricht den Ordnerlauf nicht ab; die Datei wird mit Fehlermeldung übersprungen und mit
den restlichen Dateien fortgefahren.

## Workflow

1. Originalscan laden (Bild oder PDF-Seite, siehe oben).
2. Modellnamen und Kontextgröße kontrollieren.
3. Qwen-Vorannotation starten.
4. Tabellenzeile auswählen und Text korrigieren.
5. Bei Bedarf Boxwerte im Bereich 0 bis 1000 verändern.
6. Änderungen übernehmen.
7. Annotations-JSON speichern.
8. Als Trainings-JSONL exportieren.

Beim erneuten Export derselben Bilddatei wird deren vorhandener JSONL-Datensatz ersetzt, nicht verdoppelt.

## Trainings-JSONL aus einem ganzen Ordner zusammenfassen

`python qwen_export_dataset.py C:\Handschrift-Dataset` durchsucht den angegebenen Ordner
rekursiv nach allen `*_annotation.json`-Dateien – also nur von einem Menschen in der
Oberfläche geprüften und gespeicherten Annotationen, nicht den ungeprüften
`*_preannotation.json`-Dateien – und schreibt sie gesammelt als eine einzige
`train.jsonl` (Standard: `<Ordner>\train.jsonl`). Optionen: `--dataset-root` für relative
Bildpfade (Standard: der durchsuchte Ordner) und `--output` für einen anderen Dateinamen.
Fehlerhafte oder leere Annotationsdateien werden mit Warnung übersprungen, nicht die
gesamte Zusammenfassung abgebrochen.

## Datenschutz

Standardmäßig bindet Gradio nur an `127.0.0.1`. Für vertrauliche Scans `--share` nicht benutzen. Ollama wird lokal über `127.0.0.1:11434` angesprochen.

## Ausgabeformat

Die Annotations-JSON bewahrt Vorhersage, Korrektur, Konfidenz, Status und Position. Die Trainings-JSONL enthält pro Seite eine User-Nachricht mit Bild und Prompt sowie eine Assistant-Nachricht mit korrigierten Zeilenboxen und Text.

## Hinweise

- Qwen-Ausgaben und Boxen müssen fachlich geprüft werden.
- Namen, Zahlen, Messwerte und Einheiten besonders sorgfältig kontrollieren.
- Ein Modell-Finetuning selbst ist nicht Bestandteil dieser Anwendung.
