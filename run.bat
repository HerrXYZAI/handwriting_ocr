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
echo  2. Vorannotation (qwen_preannotate.py)
echo  3. Trainings-Datensatz exportieren (qwen_export_dataset.py)
echo  4. Annotation validieren (validate_annotations.py)
echo  5. Beenden
echo ===================================================
set "CHOICE="
set /p CHOICE="Auswahl (1-5, Enter = 1): "
if "%CHOICE%"=="" set "CHOICE=1"

if "%CHOICE%"=="1" goto :gui
if "%CHOICE%"=="2" goto :preannotate
if "%CHOICE%"=="3" goto :export
if "%CHOICE%"=="4" goto :validate
if "%CHOICE%"=="5" goto :eof
goto :menu

:gui
"%PYEXE%" qwen_annotation_gui.py
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

:done
echo.
pause
goto :menu
