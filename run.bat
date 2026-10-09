@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"

set "PYEXE="

rem Prefer the "qwenocr" conda env explicitly, found by its actual install
rem path rather than by "conda activate" (which needs a shell hook that may
rem not be set up for cmd.exe). This way run.bat always uses the right
rem interpreter even if the calling shell never activated it itself.
where conda >nul 2>nul
if %errorlevel%==0 (
  for /f "delims=" %%B in ('conda info --base 2^>nul') do set "CONDA_BASE=%%B"
  if defined CONDA_BASE if exist "!CONDA_BASE!\envs\qwenocr\python.exe" (
    set "PYEXE=!CONDA_BASE!\envs\qwenocr\python.exe"
  )
)

rem Otherwise fall back to whatever Python is already active on PATH (e.g.
rem a manually activated conda env), since "py" (the Windows Python
rem launcher) is not installed on every machine.
rem "where python" alone is not enough: on a plain (non-activated) shell,
rem Windows often resolves "python" to the Microsoft Store stub, which is
rem on PATH but not a real interpreter. Actually invoke it to check.
if not defined PYEXE (
  python --version >nul 2>nul
  if !errorlevel!==0 set "PYEXE=python"
)

if defined PYEXE goto :menu

if not exist .venv\Scripts\python.exe (
  where py >nul 2>nul
  if not %errorlevel%==0 (
    echo Kein Python gefunden. Bitte Python installieren oder eine
    echo passende Umgebung ^(z.B. "conda activate qwenocr"^) aktivieren,
    echo bevor run.bat gestartet wird.
    pause
    exit /b 1
  )
  py -m venv .venv
  .venv\Scripts\python.exe -m pip install --upgrade pip
  .venv\Scripts\python.exe -m pip install -r requirements.txt
)
set "PYEXE=.venv\Scripts\python.exe"

:menu
cls
echo ===================================================
echo  Qwen Handschrift-OCR
echo ===================================================
echo  1. Annotations-GUI starten
echo  2. Modellvergleich-GUI starten (GUI, Reiter Modellvergleich)
echo  3. Vorannotation (qwen_preannotate.py)
echo  4. Annotation validieren (validate_annotations.py)
echo  5. Modellvergleich per Kommandozeile (model_compare.py)
echo  6. Tesseract-Dienst starten (Docker, fuer Box-Anpassung)
echo  7. Trainings-Datensatz exportieren (qwen_export_dataset.py)
echo  8. Finetuning starten (Docker, in finetune\)
echo  9. Adapter mit Basismodell zusammenfuehren (Docker, finetune\merge.sh)
echo 10. Zurueck nach Ollama (GGUF konvertieren, quantisieren, importieren)
echo 11. Beenden
echo ===================================================
set "CHOICE="
set /p CHOICE="Auswahl (1-11, Enter = 1): "
if "%CHOICE%"=="" set "CHOICE=1"

if "%CHOICE%"=="1" goto :gui
if "%CHOICE%"=="2" goto :compare_gui
if "%CHOICE%"=="3" goto :preannotate
if "%CHOICE%"=="4" goto :validate
if "%CHOICE%"=="5" goto :compare
if "%CHOICE%"=="6" goto :tesseract
if "%CHOICE%"=="7" goto :export
if "%CHOICE%"=="8" goto :finetune
if "%CHOICE%"=="9" goto :merge_adapter
if "%CHOICE%"=="10" goto :to_ollama
if "%CHOICE%"=="11" goto :eof
goto :menu

:gui
"%PYEXE%" qwen_annotation_gui.py
goto :done

:compare_gui
echo.
echo Startet die Oberflaeche direkt im Reiter "Modellvergleich". Laeuft die
echo Annotations-GUI (Punkt 1) bereits, dort einfach den Reiter wechseln.
"%PYEXE%" qwen_annotation_gui.py --tab vergleich
goto :done

:preannotate
echo.
echo Pfad ohne Anfuehrungszeichen eingeben, auch bei Leerzeichen im Pfad.
set "INPUT_PATH="
set /p INPUT_PATH="Bild-/PDF-Datei oder Ordner: "
if "%INPUT_PATH%"=="" (
  echo Kein Pfad angegeben.
  goto :done
)
echo.
echo Optionale zusaetzliche Argumente, z.B. --model qwen3-vl:4b --verbose
echo Groesseres Modell (12 GB VRAM + 32 GB RAM, langsamer, genauer^):
echo   --model qwen3-vl:30b-a3b-instruct --max-side 1536 --ctx 12288 --no-mmap
echo (Liste aller Optionen: qwen_preannotate.py --help^). Leer lassen fuer Standard.
set "EXTRA_ARGS="
set /p EXTRA_ARGS="Zusaetzliche Optionen: "
"%PYEXE%" qwen_preannotate.py "%INPUT_PATH%" %EXTRA_ARGS%
goto :done

:export
echo.
echo Ordner wird rekursiv nach *_annotation.json durchsucht.
set "DATASET_FOLDER="
set /p DATASET_FOLDER="Ordner mit geprueften Annotationen: "
if "%DATASET_FOLDER%"=="" (
  echo Kein Ordner angegeben.
  goto :done
)
set "DATASET_ROOT="
set /p DATASET_ROOT="Dataset-Wurzelverzeichnis fuer relative Bildpfade (Enter = Ordner selbst): "
set "DATASET_OUTPUT="
set /p DATASET_OUTPUT="Ausgabedatei (Enter = <Ordner>\train.jsonl): "
set "DATASET_ARGS="
if not "%DATASET_ROOT%"=="" set "DATASET_ARGS=%DATASET_ARGS% --dataset-root "%DATASET_ROOT%""
if not "%DATASET_OUTPUT%"=="" set "DATASET_ARGS=%DATASET_ARGS% --output "%DATASET_OUTPUT%""
"%PYEXE%" qwen_export_dataset.py "%DATASET_FOLDER%" !DATASET_ARGS!
goto :done

:validate
echo.
set "ANNOTATION_FILE="
set /p ANNOTATION_FILE="Annotations-JSON-Datei: "
if "%ANNOTATION_FILE%"=="" (
  echo Keine Datei angegeben.
  goto :done
)
echo.
echo Optionale zusaetzliche Argumente, z.B. --strict-overlap --min-area 8
echo (Liste aller Optionen: validate_annotations.py --help^). Leer lassen fuer Standard.
set "VALIDATE_ARGS="
set /p VALIDATE_ARGS="Zusaetzliche Optionen: "
"%PYEXE%" validate_annotations.py "%ANNOTATION_FILE%" %VALIDATE_ARGS%
goto :done

:compare
echo.
echo Fuehrt ein Modell auf allen geprueften Seiten (*_annotation.json) aus und
echo speichert das Ergebnis dort unter model_runs; danach Vergleichstabelle.
echo Ansicht im Detail: GUI, Reiter "Modellvergleich". Seiten mit vorhandenem
echo Lauf werden uebersprungen (Abbruch mit Strg+C jederzeit moeglich).
set "COMPARE_ROOT="
set /p COMPARE_ROOT="Dataset-Ordner: "
if "%COMPARE_ROOT%"=="" (
  echo Kein Ordner angegeben.
  goto :done
)
set "COMPARE_MODEL="
set /p COMPARE_MODEL="Modell (Enter = nur Vergleichstabelle anzeigen): "
if "%COMPARE_MODEL%"=="" (
  "%PYEXE%" model_compare.py report "%COMPARE_ROOT%"
  goto :done
)
echo.
echo Optionale Argumente, z.B. --max-side 1536 --ctx 12288 --no-mmap --limit 5
set "COMPARE_ARGS="
set /p COMPARE_ARGS="Zusaetzliche Optionen: "
"%PYEXE%" model_compare.py run "%COMPARE_ROOT%" --model %COMPARE_MODEL% %COMPARE_ARGS%
goto :done

:tesseract
where docker >nul 2>nul
if not %errorlevel%==0 (
  echo Docker wurde nicht gefunden. Bitte Docker Desktop installieren und starten.
  goto :done
)
echo.
echo Baut bei Bedarf das Image und startet den Tesseract-Dienst im Hintergrund
echo unter http://127.0.0.1:8884 ^(siehe docker\tesseract-ocr\^).
rem Wie beim Finetuning direkt bauen statt "compose up --build" (buildx bake).
docker build -t handschrift-ocr-tesseract:latest docker\tesseract-ocr
if errorlevel 1 (
  echo Bau des Tesseract-Images fehlgeschlagen ^(siehe Meldungen oben^).
  goto :done
)
docker compose -f docker\tesseract-ocr\docker-compose.yml up -d --no-build
goto :done

:finetune
where docker >nul 2>nul
if not %errorlevel%==0 (
  echo Docker wurde nicht gefunden. Bitte Docker Desktop installieren und starten.
  goto :done
)
rem Basismodell waehlen: finetune\select_model.py fragt die Ollama-Modelle ab,
rem ordnet ihnen das Hugging-Face-Original zu und schreibt die Wahl als
rem KEY=VALUE-Zeilen (FT_MODEL, FT_OUTPUT_DIR_CONTAINER) in eine Temp-Datei.
rem docker compose uebernimmt diese Umgebungsvariablen vor den Werten aus .env.
set "FT_SELECT_FILE=%TEMP%\handschrift_ft_model.txt"
if exist "%FT_SELECT_FILE%" del "%FT_SELECT_FILE%"
"%PYEXE%" finetune\select_model.py "%FT_SELECT_FILE%"
if not %errorlevel%==0 (
  echo Abgebrochen.
  goto :done
)
for /f "usebackq tokens=1,* delims==" %%A in ("%FT_SELECT_FILE%") do set "%%A=%%B"
del "%FT_SELECT_FILE%" >nul 2>nul
echo.
echo Von Ollama geladene Modelle belegen Grafikspeicher, der dem Training fehlt.
set "FT_UNLOAD="
set /p FT_UNLOAD="Geladene Modelle im Docker-Container 'ollama' jetzt entladen? (J/n): "
if /i not "%FT_UNLOAD%"=="n" (
  for /f "skip=1 tokens=1" %%M in ('docker exec ollama ollama ps 2^>nul') do (
    echo Entlade %%M
    docker exec ollama ollama stop %%M >nul 2>nul
  )
)
echo.
echo Baut bei Bedarf das Image und startet das Finetuning von !FT_MODEL! im
echo Vordergrund ^(siehe finetune\^). Mit Strg+C abbrechen.
echo Checkpoints: OUTPUT_DIR aus finetune\.env, Unterordner !FT_MODEL_SLUG!-handschrift
pushd finetune
rem Image direkt mit "docker build" bauen statt per "docker compose up --build":
rem Compose nutzt dafuer "buildx bake", das bei manchen Docker-Desktop-Versionen
rem ohne Angabe von Gruenden mit "failed to execute bake" abbricht.
rem Fuer Qwen3.5/3.6 braucht das Image ein aktuelles ms-swift/transformers;
rem PACKAGES_REFRESH mit neuem Wert baut nur die Paket-Schicht neu.
set "FT_BUILD_ARGS="
set "FT_REFRESH=n"
rem Erst im vorhandenen Image pruefen, ob ms-swift das Modell schon kennt
rem (z.B. weil die Pakete in einem frueheren Lauf aktualisiert wurden) -
rem nur sonst nach einer Aktualisierung fragen.
set "FT_SUPPORTED=0"
if "!FT_NEW_ARCH!"=="1" (
  echo.
  echo Pruefe, ob das Trainings-Image !FT_MODEL! bereits unterstuetzt ...
  docker run --rm -v "%CD%:/chk:ro" --entrypoint python3 handschrift-ocr-finetune:latest /chk/check_model_support.py "!FT_MODEL!"
  if !errorlevel!==0 set "FT_SUPPORTED=1"
)
if "!FT_NEW_ARCH!"=="1" if "!FT_SUPPORTED!"=="0" (
  echo !FT_MODEL! braucht ein aktuelles ms-swift im Trainings-Image.
  set /p FT_REFRESH="Python-Pakete im Image jetzt aktualisieren? Dauert einige Minuten. (J/n): "
  if "!FT_REFRESH!"=="" set "FT_REFRESH=j"
)
rem Der Paketstand wird in finetune\.packages_refresh gemerkt und bei jedem Bau
rem wiederverwendet - sonst wuerde ein wechselnder Wert die Paket-Schicht jedes
rem Mal neu bauen. Nur eine gewollte Aktualisierung erzeugt einen neuen Wert.
set "FT_PKG_STAMP=0"
if exist ".packages_refresh" set /p FT_PKG_STAMP=<".packages_refresh"
if /i "!FT_REFRESH!"=="j" (
  set "FT_PKG_STAMP=%DATE:~-4%%RANDOM%%RANDOM%"
  >".packages_refresh" echo !FT_PKG_STAMP!
)
set "FT_BUILD_ARGS=--build-arg PACKAGES_REFRESH=!FT_PKG_STAMP!"
docker build !FT_BUILD_ARGS! -t handschrift-ocr-finetune:latest .
if errorlevel 1 (
  echo Bau des Trainings-Images fehlgeschlagen ^(siehe Meldungen oben^).
  popd
  goto :done
)
docker compose up --no-build
popd
rem Auswahl nicht an spaetere Menuepunkte weiterreichen.
set "FT_MODEL="
set "FT_OUTPUT_DIR_CONTAINER="
set "FT_MODEL_SLUG="
set "FT_NEW_ARCH="
goto :done

:merge_adapter
where docker >nul 2>nul
if not %errorlevel%==0 (
  echo Docker wurde nicht gefunden. Bitte Docker Desktop installieren und starten.
  goto :done
)
call :get_output_dir
echo.
echo Fuehrt einen trainierten LoRA-Adapter-Checkpoint mit dem Basismodell
echo zusammen (siehe finetune\README.md, Abschnitt 4). Pfad relativ zum
echo OUTPUT_DIR aus finetune\.env eingeben (aktuell: !OUTPUT_DIR_ABS!),
echo z.B. qwen3-vl-4b-handschrift\checkpoint-3
set "ADAPTER_REL="
set /p ADAPTER_REL="Adapter-Checkpoint-Ordner (relativ zu OUTPUT_DIR): "
if "%ADAPTER_REL%"=="" (
  echo Kein Pfad angegeben.
  goto :done
)
set "ADAPTER_REL=!ADAPTER_REL:\=/!"
echo.
set "MERGE_OUT_REL="
set /p MERGE_OUT_REL="Ausgabeordner, relativ zu OUTPUT_DIR (Enter = <Checkpoint>-merged): "
set "MERGE_ARGS="
if not "%MERGE_OUT_REL%"=="" (
  set "MERGE_OUT_REL=!MERGE_OUT_REL:\=/!"
  set "MERGE_ARGS="/output/!MERGE_OUT_REL!""
)
pushd finetune
docker compose run --rm finetune ./merge.sh "/output/!ADAPTER_REL!" !MERGE_ARGS!
popd
if "%MERGE_OUT_REL%"=="" (set "MERGE_RESULT_REL=!ADAPTER_REL!-merged") else (set "MERGE_RESULT_REL=!MERGE_OUT_REL!")
echo.
echo Zusammengefuehrtes Modell (relativ zu OUTPUT_DIR, fuer den naechsten
echo Schritt "Zurueck nach Ollama"):
echo   !MERGE_RESULT_REL!
goto :done

:to_ollama
where docker >nul 2>nul
if not %errorlevel%==0 (
  echo Docker wurde nicht gefunden. Bitte Docker Desktop installieren und starten.
  goto :done
)
where powershell >nul 2>nul
if not %errorlevel%==0 (
  echo PowerShell wurde nicht gefunden.
  goto :done
)
call :get_output_dir
echo.
echo Setzt einen bereits per merge.sh zusammengefuehrten Checkpoint voraus
echo (siehe finetune\README.md, Abschnitt 4 und 5). Pfad relativ zum
echo OUTPUT_DIR aus finetune\.env eingeben (aktuell: !OUTPUT_DIR_ABS!),
echo z.B. qwen3-vl-4b-handschrift\checkpoint-3-merged
set "MERGED_REL="
set /p MERGED_REL="Zusammengefuehrtes Modell (relativ zu OUTPUT_DIR): "
if "%MERGED_REL%"=="" (
  echo Kein Pfad angegeben.
  goto :done
)
set "MERGED_DIR=!OUTPUT_DIR_ABS!\%MERGED_REL%"
rem Namensvorschlag aus dem ersten Ordner des Pfads, z.B.
rem qwen3-vl-8b-handschrift\v4-...\checkpoint-3-merged -> qwen3-vl-8b-handschrift
set "DEFAULT_OLLAMA_NAME=qwen3-vl-4b-handschrift"
for /f "tokens=1 delims=\/" %%S in ("%MERGED_REL%") do set "DEFAULT_OLLAMA_NAME=%%S"
set "OLLAMA_MODEL_NAME="
set /p OLLAMA_MODEL_NAME="Modellname in Ollama (Enter = !DEFAULT_OLLAMA_NAME!): "
if "%OLLAMA_MODEL_NAME%"=="" set "OLLAMA_MODEL_NAME=!DEFAULT_OLLAMA_NAME!"
set "OLLAMA_QUANT="
set /p OLLAMA_QUANT="Quantisierung (Enter = Q4_K_M): "
set "TO_OLLAMA_ARGS=-MergedDir "%MERGED_DIR%""
if not "%OLLAMA_MODEL_NAME%"=="" set "TO_OLLAMA_ARGS=%TO_OLLAMA_ARGS% -ModelName "%OLLAMA_MODEL_NAME%""
if not "%OLLAMA_QUANT%"=="" set "TO_OLLAMA_ARGS=%TO_OLLAMA_ARGS% -Quant "%OLLAMA_QUANT%""
powershell -ExecutionPolicy Bypass -File finetune\to_ollama.ps1 !TO_OLLAMA_ARGS!
goto :done

:done
echo.
pause
goto :menu

rem Liest OUTPUT_DIR aus finetune\.env (Standard "./output", falls nicht
rem gesetzt oder die Datei fehlt) und setzt OUTPUT_DIR_ABS auf den
rem aufgeloesten absoluten Pfad, relativ zu finetune\ ausgewertet.
:get_output_dir
set "OUTPUT_DIR_RAW=./output"
if exist finetune\.env (
  for /f "usebackq tokens=1,* delims==" %%A in ("finetune\.env") do (
    if "%%A"=="OUTPUT_DIR" set "OUTPUT_DIR_RAW=%%B"
  )
)
pushd finetune >nul 2>nul
for %%I in ("%OUTPUT_DIR_RAW%") do set "OUTPUT_DIR_ABS=%%~fI"
popd >nul 2>nul
goto :eof
