<#
.SYNOPSIS
    Konvertiert ein mit merge.sh zusammengefuehrtes Qwen3-VL-Modell nach GGUF
    und importiert es als Vision-Modell in Ollama.

.DESCRIPTION
    Fuehrt drei Schritte aus, alle ueber das offizielle llama.cpp-"full"-Image
    (Konvertierung/Quantisierung brauchen keine GPU):
      1. Textmodell nach GGUF konvertieren (f16)
      2. Vision-Projektor (mmproj) separat exportieren (f16)
      3. Textmodell quantisieren (Standard: Q4_K_M)
    Danach ein Modelfile mit beiden GGUF-Dateien schreiben und per
    "ollama create" importieren.

    Voraussetzung: merge.sh wurde bereits erfolgreich auf einem
    Trainings-Checkpoint ausgefuehrt (siehe README.md, Abschnitt 4) -
    -MergedDir muss auf dessen "...-merged"-Ausgabeordner zeigen.

.PARAMETER MergedDir
    Pfad zum zusammengefuehrten Modell (z.B.
    C:\Handschrift-Dataset\finetune-output\qwen3-vl-4b-handschrift\v4-...\checkpoint-3-merged)

.PARAMETER ModelName
    Name, unter dem das Modell in Ollama registriert wird.

.PARAMETER Quant
    Quantisierungstyp fuer llama-quantize, z.B. Q4_K_M (klein/schnell) oder
    Q8_0 (groesser, naeher am Original).

.EXAMPLE
    .\to_ollama.ps1 -MergedDir C:\Handschrift-Dataset\finetune-output\qwen3-vl-4b-handschrift\v4-20260919-094304\checkpoint-3-merged
#>
param(
    [Parameter(Mandatory = $true)]
    [string]$MergedDir,
    [string]$ModelName = "qwen3-vl-4b-handschrift",
    [string]$Quant = "Q4_K_M",
    [string]$LlamaCppImage = "ghcr.io/ggml-org/llama.cpp:full"
)

$ErrorActionPreference = "Stop"

where.exe docker *>$null
if ($LASTEXITCODE -ne 0) {
    Write-Error "Docker wurde nicht gefunden. Bitte Docker Desktop installieren und starten."
    exit 1
}
where.exe ollama *>$null
if ($LASTEXITCODE -ne 0) {
    Write-Error "Ollama wurde nicht gefunden. Bitte Ollama installieren und sicherstellen, dass es im PATH liegt."
    exit 1
}

if (-not (Test-Path $MergedDir)) {
    Write-Error "MergedDir nicht gefunden: $MergedDir"
    exit 1
}
$MergedDir = (Resolve-Path $MergedDir).Path

# GGUF-Dateien landen in einem "gguf"-Ordner neben dem zusammengefuehrten Modell,
# nicht darin - das Konvertierungsskript wuerde sonst versuchen, die neuen
# .gguf-Dateien als weitere Modell-Shards mit einzulesen.
$GgufDir = Join-Path (Split-Path $MergedDir -Parent) "gguf"
New-Item -ItemType Directory -Force -Path $GgufDir | Out-Null
$GgufDir = (Resolve-Path $GgufDir).Path

$TextGguf = "model-f16.gguf"
$MmprojGguf = "mmproj-f16.gguf"
$QuantGguf = "model-$Quant.gguf"

Write-Host "1/4 Konvertiere Textmodell nach GGUF (f16)..."
docker run --rm -v "${MergedDir}:/model:ro" -v "${GgufDir}:/gguf" $LlamaCppImage `
    --convert /model --outfile "/gguf/$TextGguf" --outtype f16
if ($LASTEXITCODE -ne 0) { Write-Error "Konvertierung des Textmodells fehlgeschlagen."; exit 1 }

Write-Host "2/4 Exportiere Vision-Projektor (mmproj, f16)..."
docker run --rm -v "${MergedDir}:/model:ro" -v "${GgufDir}:/gguf" $LlamaCppImage `
    --convert /model --outfile "/gguf/$MmprojGguf" --outtype f16 --mmproj
if ($LASTEXITCODE -ne 0) { Write-Error "Export des Vision-Projektors fehlgeschlagen."; exit 1 }

Write-Host "3/4 Quantisiere Textmodell nach $Quant..."
docker run --rm -v "${GgufDir}:/gguf" $LlamaCppImage `
    --quantize "/gguf/$TextGguf" "/gguf/$QuantGguf" $Quant
if ($LASTEXITCODE -ne 0) { Write-Error "Quantisierung fehlgeschlagen."; exit 1 }

$ModelfilePath = Join-Path $GgufDir "Modelfile"
$ModelfileContent = "FROM ./$QuantGguf`nFROM ./$MmprojGguf`n"
Set-Content -Path $ModelfilePath -Value $ModelfileContent -Encoding utf8 -NoNewline

Write-Host "4/4 Importiere als '$ModelName' in Ollama..."
Push-Location $GgufDir
try {
    ollama create $ModelName -f Modelfile
    if ($LASTEXITCODE -ne 0) { Write-Error "ollama create fehlgeschlagen."; exit 1 }
} finally {
    Pop-Location
}

Write-Host ""
Write-Host "Fertig. GGUF-Dateien liegen in: $GgufDir"
Write-Host "Testen mit: ollama run $ModelName"
Write-Host ""
Write-Host "Bekannte Einschraenkung: der Import von selbst konvertierten Qwen3-VL-"
Write-Host "GGUF+mmproj-Paaren in Ollama ist noch nicht durchgehend stabil (Stand"
Write-Host "jetzt gibt es offene Ollama-Bugs, bei denen 'ollama show' das Modell"
Write-Host "korrekt als vision-faehig erkennt, ein Bild aber trotzdem zum Absturz"
Write-Host "des Model-Runners fuehrt). Falls 'ollama run' bei einem Bild abstuerzt"
Write-Host "oder mit 'model runner has unexpectedly stopped' fehlschlaegt, das"
Write-Host "Modell ersatzweise direkt mit llama.cpp statt Ollama testen:"
Write-Host "  docker run --rm -p 8080:8080 -v `"${GgufDir}:/gguf`" $LlamaCppImage --server -m /gguf/$QuantGguf --mmproj /gguf/$MmprojGguf --host 0.0.0.0"
Write-Host "und qwen_annotation_gui.py/qwen_preannotate.py voruebergehend auf"
Write-Host "dessen OpenAI-kompatible API (http://127.0.0.1:8080) statt Ollama zeigen."
