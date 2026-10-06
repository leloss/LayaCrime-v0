$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$VenvDir = if ($env:LAYA_VENV_DIR) {
    $env:LAYA_VENV_DIR
} else {
    Join-Path (Split-Path -Parent $ProjectRoot) ".venv-laya"
}
$Python = Join-Path $VenvDir "Scripts\python.exe"
$LayaSource = Join-Path $ProjectRoot "third_party\laya"
$LayaRepo = if ($env:LAYA_HF_REPO_ID) { $env:LAYA_HF_REPO_ID } else { "convaiinnovations/laya" }
$LayaRevision = if ($env:LAYA_HF_REVISION) { $env:LAYA_HF_REVISION } else { "main" }
$LayaCrimeRepo = if ($env:LAYACRIME_HF_REPO_ID) { $env:LAYACRIME_HF_REPO_ID } else { "leloss/layacrime" }
$LayaCrimeRevision = if ($env:LAYACRIME_HF_REVISION) { $env:LAYACRIME_HF_REVISION } else { "main" }
$SnapshotDownloader = Join-Path $PSScriptRoot "download_huggingface_snapshot.py"

if (-not (Test-Path $Python)) {
    py -3.12 -m venv $VenvDir
    if ($LASTEXITCODE -ne 0) { throw "virtual environment creation failed" }
}

if (-not (Test-Path (Join-Path $LayaSource ".git"))) {
    git clone --depth 1 --branch v0.3.10 `
        https://github.com/NandhaKishorM/laya.git $LayaSource
}

& $Python -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { throw "pip upgrade failed" }
& $Python -m pip install --editable $LayaSource
if ($LASTEXITCODE -ne 0) { throw "Laya installation failed" }
& $Python -m pip install --editable "${ProjectRoot}[dev]"
if ($LASTEXITCODE -ne 0) { throw "project installation failed" }
& $Python $SnapshotDownloader `
    --repo-id $LayaRepo `
    --revision $LayaRevision `
    --cache-dir (Join-Path $ProjectRoot "models") `
    --allow-pattern "encoder/**" `
    --allow-pattern "model.safetensors" `
    --allow-pattern "rl_agent_config.json" `
    --allow-pattern "tokenizer/**" `
    --allow-pattern "multilingual/**" `
    --allow-pattern "typed-decisions/**"
if ($LASTEXITCODE -ne 0) { throw "Laya checkpoint download failed" }
& $Python $SnapshotDownloader `
    --repo-id $LayaCrimeRepo `
    --revision $LayaCrimeRevision `
    --local-dir (Join-Path $ProjectRoot "models\fine-tuned\layacrime-public") `
    --allow-pattern "encoder/**" `
    --allow-pattern "model.safetensors" `
    --allow-pattern "rl_agent_config.json" `
    --allow-pattern "tokenizer/**" `
    --allow-pattern "training_report.json" `
    --allow-pattern "data_manifest.json" `
    --allow-pattern "README.md"
if ($LASTEXITCODE -ne 0) { throw "LayaCrime checkpoint download failed" }

Write-Host "Laya source installed from $LayaSource"
Write-Host "Laya downloaded from https://huggingface.co/$LayaRepo"
Write-Host "LayaCrime downloaded from https://huggingface.co/$LayaCrimeRepo"
Write-Host "Python environment ready at $VenvDir"