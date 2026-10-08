#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
echo "run_finetuning_backend.sh is an alias for run_ui.sh."
exec "$PROJECT_ROOT/scripts/run_ui.sh" "$@"
