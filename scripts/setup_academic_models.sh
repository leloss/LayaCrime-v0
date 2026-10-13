#!/usr/bin/env bash
set -Eeuo pipefail

trap 'echo "Academic model setup failed at line ${LINENO}." >&2' ERR

PROJECT_ROOT="${PROJECT_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
VENV_DIR="${VENV_DIR:-${XDG_CACHE_HOME:-$HOME/.cache}/laya-adverse-media/venv}"
TARTU_REPOSITORY="${LAYA_TARTU_REPOSITORY:-https://github.com/kristjanr/ut-ml-adverse-media}"
TARTU_COMMIT="12fa6ada0a6f46ce098a654e39aea0ebbf1d55f5"
TARTU_SOURCE="${LAYA_TARTU_SOURCE:-$PROJECT_ROOT/third_party/ut-ml-adverse-media}"
# NewsMTSC needs its own Python 3.8-3.11 environment; set to false to skip it.
LAYA_SETUP_NEWSMTSC="${LAYA_SETUP_NEWSMTSC:-true}"

PYTHON="$VENV_DIR/bin/python"
if [[ ! -x "$PYTHON" ]]; then
    echo "Laya environment not found at $VENV_DIR." >&2
    echo "Run $PROJECT_ROOT/scripts/setup_environment.sh first." >&2
    exit 1
fi

echo "Installing Khandpur and Tartu dependencies (scikit-learn, spaCy)"
"$PYTHON" -m pip install --editable "${PROJECT_ROOT}[academic]"
if ! "$PYTHON" -c 'import en_core_web_sm' >/dev/null 2>&1; then
    "$PYTHON" -m spacy download en_core_web_sm
fi

echo "Fetching the Tartu training archives at $TARTU_COMMIT"
if [[ ! -d "$TARTU_SOURCE/.git" ]]; then
    mkdir -p "$TARTU_SOURCE"
    git -C "$TARTU_SOURCE" init --quiet
    git -C "$TARTU_SOURCE" remote add origin "$TARTU_REPOSITORY"
fi
if [[ "$(git -C "$TARTU_SOURCE" rev-parse HEAD 2>/dev/null || true)" != "$TARTU_COMMIT" ]]; then
    git -C "$TARTU_SOURCE" fetch --depth 1 origin "$TARTU_COMMIT"
    git -C "$TARTU_SOURCE" checkout --quiet --detach "$TARTU_COMMIT"
fi

newsmtsc_ready=true
if [[ "$LAYA_SETUP_NEWSMTSC" == "true" ]]; then
    echo "Setting up the isolated NewsMTSC environment"
    if ! LAYA_MAIN_VENV_DIR="$VENV_DIR" bash "$PROJECT_ROOT/scripts/setup_newsmtsc_environment.sh"; then
        newsmtsc_ready=false
        echo "WARNING: NewsMTSC could not be set up; the other academic models remain available." >&2
        echo "Retry later with: $PROJECT_ROOT/scripts/setup_newsmtsc_environment.sh" >&2
    fi
fi

PROJECT_ROOT="$PROJECT_ROOT" "$PYTHON" - <<'PY'
import os
from pathlib import Path

from laya_adverse_media.academic_models import AcademicModelRuntime

for option in AcademicModelRuntime(Path(os.environ["PROJECT_ROOT"])).options():
    state = "ready" if option["runtime_available"] else option["availability_error"]
    print(f"{option['label']}: {state}")
PY

if [[ "$newsmtsc_ready" != "true" ]]; then
    exit 1
fi
