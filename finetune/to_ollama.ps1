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
    [string]$LlamaCppImage = "ghcr.io/ggml-org/llama.cpp:full",
    [string]$OllamaContainer = "ollama"
)

$ErrorActionPreference = "Stop"

# Konsolenausgabe mit Zeitstempel (gleiches Format wie die Python-Skripte).
function Write-Log([string]$Message) {
    if ($Message) { Write-Host ("{0} | {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $Message) } else { Write-Host "" }
}
function Write-LogError([string]$Message) {
    Write-Error ("{0} | {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $Message)
}

if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
    Write-LogError "Docker wurde nicht gefunden. Bitte Docker Desktop installieren und starten."
    exit 1
}

# Ollama kann entweder lokal installiert sein (CLI im PATH, spricht ueber
# OLLAMA_HOST mit dem Server) oder komplett als Docker-Container laufen (z.B.
# als Teil eines RAG-Setups). Im zweiten Fall gibt es keine lokale "ollama"-
# CLI, daher faellt dieses Skript auf "docker exec" in den laufenden
# Container zurueck.
$OllamaMode = $null
if (Get-Command ollama -ErrorAction SilentlyContinue) {
    $OllamaMode = "local"
} else {
    $running = docker ps --filter "name=^/${OllamaContainer}$" --format "{{.Names}}" 2>$null
    if ($running -eq $OllamaContainer) {
        $OllamaMode = "docker"
        Write-Log "Ollama-CLI nicht lokal gefunden, verwende laufenden Docker-Container '$OllamaContainer'."
    } else {
        Write-LogError "Ollama wurde nicht gefunden: weder lokal im PATH noch als laufender Docker-Container '$OllamaContainer'. Bitte Ollama installieren, den Container starten, oder -OllamaContainer mit dem richtigen Namen angeben."
        exit 1
    }
}

if (-not (Test-Path $MergedDir)) {
    Write-LogError "MergedDir nicht gefunden: $MergedDir"
    exit 1
}
$MergedDir = (Resolve-Path $MergedDir).Path

# ms-swift speichert tokenizer_config.json mit einem "extra_special_tokens"-
# Feld als flache Liste (Feature neuerer Transformers-Versionen; das
# urspruengliche Qwen3-VL-4B-Instruct-Modell hat dieses Feld gar nicht). Die
# in llama.cpp:full gebuendelte (aeltere) Transformers-Version erwartet dort
# ein dict und stuerzt sonst mit "AttributeError: 'list' object has no
# attribute 'keys'" ab. Das Feld ist reine Zusatz-Metadaten (alle Tokens
# stehen bereits in tokenizer.json) und kann gefahrlos entfernt werden.
$TokenizerConfigPath = Join-Path $MergedDir "tokenizer_config.json"
if (Test-Path $TokenizerConfigPath) {
    $tokenizerConfig = Get-Content $TokenizerConfigPath -Raw | ConvertFrom-Json
    if ($tokenizerConfig.PSObject.Properties.Name -contains "extra_special_tokens") {
        Write-Log "Entferne inkompatibles 'extra_special_tokens'-Feld aus tokenizer_config.json..."
        $tokenizerConfig.PSObject.Properties.Remove("extra_special_tokens")
        $utf8NoBom = New-Object System.Text.UTF8Encoding $false
        [System.IO.File]::WriteAllText($TokenizerConfigPath, ($tokenizerConfig | ConvertTo-Json -Depth 10), $utf8NoBom)
    }
}

# GGUF-Dateien landen in einem "gguf"-Ordner neben dem zusammengefuehrten Modell,
# nicht darin - das Konvertierungsskript wuerde sonst versuchen, die neuen
# .gguf-Dateien als weitere Modell-Shards mit einzulesen.
$GgufDir = Join-Path (Split-Path $MergedDir -Parent) "gguf"
New-Item -ItemType Directory -Force -Path $GgufDir | Out-Null
$GgufDir = (Resolve-Path $GgufDir).Path

$TextGguf = "model-f16.gguf"
$MmprojGguf = "mmproj-f16.gguf"
$QuantGguf = "model-$Quant.gguf"

Write-Log "1/4 Konvertiere Textmodell nach GGUF (f16)..."
docker run --rm -v "${MergedDir}:/model:ro" -v "${GgufDir}:/gguf" $LlamaCppImage `
    --convert /model --outfile "/gguf/$TextGguf" --outtype f16
if ($LASTEXITCODE -ne 0) { Write-LogError "Konvertierung des Textmodells fehlgeschlagen."; exit 1 }

Write-Log "2/4 Exportiere Vision-Projektor (mmproj, f16)..."
docker run --rm -v "${MergedDir}:/model:ro" -v "${GgufDir}:/gguf" $LlamaCppImage `
    --convert /model --outfile "/gguf/$MmprojGguf" --outtype f16 --mmproj
if ($LASTEXITCODE -ne 0) { Write-LogError "Export des Vision-Projektors fehlgeschlagen."; exit 1 }

Write-Log "3/4 Quantisiere Textmodell nach $Quant..."
docker run --rm -v "${GgufDir}:/gguf" $LlamaCppImage `
    --quantize "/gguf/$TextGguf" "/gguf/$QuantGguf" $Quant
if ($LASTEXITCODE -ne 0) { Write-LogError "Quantisierung fehlgeschlagen."; exit 1 }

$ModelfilePath = Join-Path $GgufDir "Modelfile"
$ModelfileContent = "FROM ./$QuantGguf`nFROM ./$MmprojGguf`n"
Set-Content -Path $ModelfilePath -Value $ModelfileContent -Encoding utf8 -NoNewline

Write-Log "4/4 Importiere als '$ModelName' in Ollama..."
if ($OllamaMode -eq "local") {
    Push-Location $GgufDir
    try {
        ollama create $ModelName -f Modelfile
        if ($LASTEXITCODE -ne 0) { Write-LogError "ollama create fehlgeschlagen."; exit 1 }
    } finally {
        Pop-Location
    }
    $RunHint = "ollama run $ModelName"
} else {
    $ContainerDir = "/tmp/ollama-import-$ModelName"
    docker exec $OllamaContainer sh -c "rm -rf '$ContainerDir' && mkdir -p '$ContainerDir'"
    if ($LASTEXITCODE -ne 0) { Write-LogError "Konnte Import-Verzeichnis im Container '$OllamaContainer' nicht anlegen."; exit 1 }
    docker cp $ModelfilePath "${OllamaContainer}:${ContainerDir}/Modelfile"
    docker cp (Join-Path $GgufDir $QuantGguf) "${OllamaContainer}:${ContainerDir}/$QuantGguf"
    docker cp (Join-Path $GgufDir $MmprojGguf) "${OllamaContainer}:${ContainerDir}/$MmprojGguf"
    docker exec -w $ContainerDir $OllamaContainer ollama create $ModelName -f Modelfile
    if ($LASTEXITCODE -ne 0) { Write-LogError "ollama create fehlgeschlagen (im Container '$OllamaContainer')."; exit 1 }
    docker exec $OllamaContainer sh -c "rm -rf '$ContainerDir'"
    $RunHint = "docker exec -it $OllamaContainer ollama run $ModelName"
}

Write-Log ""
Write-Log "Fertig. GGUF-Dateien liegen in: $GgufDir"
Write-Log "Testen mit: $RunHint"
Write-Log ""
Write-Log "Bekannte Einschraenkung: der Import von selbst konvertierten Qwen3-VL-"
Write-Log "GGUF+mmproj-Paaren in Ollama ist noch nicht durchgehend stabil (Stand"
Write-Log "jetzt gibt es offene Ollama-Bugs, bei denen 'ollama show' das Modell"
Write-Log "korrekt als vision-faehig erkennt, ein Bild aber trotzdem zum Absturz"
Write-Log "des Model-Runners fuehrt). Falls 'ollama run' bei einem Bild abstuerzt"
Write-Log "oder mit 'model runner has unexpectedly stopped' fehlschlaegt, das"
Write-Log "Modell ersatzweise direkt mit llama.cpp statt Ollama testen:"
Write-Log "  docker run --rm -p 8080:8080 -v `"${GgufDir}:/gguf`" $LlamaCppImage --server -m /gguf/$QuantGguf --mmproj /gguf/$MmprojGguf --host 0.0.0.0"
Write-Log "und qwen_annotation_gui.py/qwen_preannotate.py voruebergehend auf"
Write-Log "dessen OpenAI-kompatible API (http://127.0.0.1:8080) statt Ollama zeigen."
