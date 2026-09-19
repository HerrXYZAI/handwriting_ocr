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
Bild- oder PDF-Datei direkt in diesem Ordner (nicht rekursiv in Unterordnern). Dabei wird
jede Seite (bei PDFs) bzw. jede Kachel (bei großformatigen Bildern) einzeln übersprungen,
wenn ihre eigene `*_preannotation.json` schon existiert - nicht die ganze Datei anhand
irgendeiner vorhandenen Teil-Datei. Ein mehrseitiges PDF, bei dem z.B. nur Seite 1 bereits
vorannotiert ist, verarbeitet beim erneuten Aufruf also nur die fehlenden Seiten weiter,
statt entweder komplett übersprungen oder komplett neu erzeugt zu werden. Das gilt auch
für einen direkten Aufruf auf eine einzelne mehrseitige PDF- oder Kachel-Datei außerhalb
eines Ordnerlaufs. `--force` erzwingt eine erneute Erkennung aller Seiten/Kacheln, auch
wenn ihre `_preannotation.json` schon existiert. `--output`/`--log-file` sind bei einem
Ordner nicht zulässig, da die Namen je Datei automatisch vergeben werden. Ein Fehler bei
einer Datei bricht den Ordnerlauf nicht ab; die Datei wird mit Fehlermeldung übersprungen
und mit den restlichen Dateien fortgefahren.

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

Über der Tabelle stehen drei weitere Werkzeuge für die ausgewählte Zeile (Tabellenzeile
oder Box in der Vorschau anklicken):

- **Box hinzufügen**: legt eine neue, mittig platzierte Box an, die sich per Maus an die
  richtige Stelle ziehen und skalieren lässt.
- **Ausgewählte Box löschen**: entfernt die ausgewählte Zeile vollständig.
- **Kein sichtbarer Text**: markiert die ausgewählte Box als Stempel, Wasserzeichen,
  leeren Rand o.Ä. ohne echten Handschrifttext. Die Box erscheint danach grau statt in
  der Konfidenzfarbe, ihr Text wird geleert, und sie wird beim Export als Trainings-JSONL
  automatisch übersprungen (siehe unten). Wird später doch wieder Text eingetragen, hebt
  das die Markierung automatisch auf.

Alle drei Werkzeuge wirken sofort auf Tabelle und Vorschau; gespeichert wird erst mit
"Annotations-JSON speichern".

## Tesseract-Boxen ergänzen (optional, per Docker)

Qwen schätzt Zeilenboxen als Teil seiner Texterkennung; das ist manchmal ungenau
oder übersieht Zeilen. Optional lässt sich zusätzlich Tesseract abrufen, das
zwar Handschrift selbst kaum lesen kann, dessen Layout-/Zeilenerkennung aber oft
brauchbare Boxgeometrie liefert. Tesseracts erkannter Text wird dabei verworfen -
nur die Boxen werden verwendet ("Snap + Fill"):

- Zeilen, die Qwen bereits gefunden hat, werden auf die am besten überlappende
  Tesseract-Box eingerastet (Text und Konfidenz bleiben unverändert).
- Tesseract-Boxen ohne ausreichende Überlappung zu einer vorhandenen Zeile
  werden als neue, unbestätigte Zeile ergänzt (Text leer) - zum Auffinden von
  Zeilen, die Qwen übersehen hat.

Tesseract läuft dafür als kleiner lokaler Dienst in Docker (kein GPU nötig),
den die Oberfläche und die Kommandozeile über HTTP ansprechen, ähnlich wie
Ollama:

```powershell
docker compose -f docker\tesseract-ocr\docker-compose.yml up -d --build
```

Danach ist der Dienst unter `http://127.0.0.1:8884` erreichbar. Verwendung:

- Oberfläche: nach der Qwen-Vorannotation auf "Tesseract-Boxen anwenden
  (Snap + Fill)" klicken.
- Kommandozeile: `python qwen_preannotate.py scan.png --tesseract-adjust`
  wendet Snap+Fill automatisch nach jeder Qwen-Erkennung an (auch bei Ordner-,
  PDF- und Kachel-Läufen). `--tesseract-lang`/`--tesseract-psm` passen Sprache
  und Page-Segmentation-Mode an; Standard ist `deu` bzw. `11` (sparse text).

Ist der Dienst nicht erreichbar, meldet die Oberfläche das direkt; die
Kommandozeile protokolliert eine Warnung und fährt ohne Tesseract-Anpassung
fort statt den ganzen Lauf abzubrechen.

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

Standardmäßig bindet Gradio nur an `127.0.0.1`. Für vertrauliche Scans `--share` nicht benutzen. Ollama wird lokal über `127.0.0.1:11434` angesprochen, der optionale Tesseract-Dienst über `127.0.0.1:8884`.

## Ausgabeformat

Die Annotations-JSON bewahrt Vorhersage, Korrektur, Konfidenz, Status und Position. Die Trainings-JSONL enthält pro Seite eine User-Nachricht mit Bild und Prompt sowie eine Assistant-Nachricht mit korrigierten Zeilenboxen und Text. Zeilen mit leerem korrigiertem Text - insbesondere als "Kein sichtbarer Text" markierte - werden beim Export nicht aufgenommen.

## Hinweise

- Qwen-Ausgaben und Boxen müssen fachlich geprüft werden.
- Namen, Zahlen, Messwerte und Einheiten besonders sorgfältig kontrollieren.
- Ein Modell-Finetuning selbst ist nicht Bestandteil dieser Anwendung.

## Finetuning der Trainings-JSONL

Für ein QLoRA-Finetuning von Qwen3-VL auf der exportierten `train.jsonl`, in
einem separaten Docker-Container mit GPU-Zugriff: siehe
[`finetune/README.md`](finetune/README.md).
