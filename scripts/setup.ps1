$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$LayaSource = Join-Path $ProjectRoot "third_party\laya"

if (-not (Test-Path $Python)) {
    py -3.12 -m venv (Join-Path $ProjectRoot ".venv")
}

if (-not (Test-Path (Join-Path $LayaSource ".git"))) {
    git clone --depth 1 --branch v0.3.10 `
        https://github.com/NandhaKishorM/laya.git $LayaSource
}

& $Python -m pip install --upgrade pip
& $Python -m pip install --editable $LayaSource
& $Python -m pip install --editable "${ProjectRoot}[dev]"
& $Python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='convaiinnovations/laya', cache_dir=r'$ProjectRoot\models')"

Write-Host "Laya source installed from $LayaSource"
Write-Host "Public Laya checkpoints downloaded into $ProjectRoot\models"