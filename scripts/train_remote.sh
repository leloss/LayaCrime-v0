#!/usr/bin/env bash
set -Eeuo pipefail

trap 'echo "Remote CLI training failed at line ${LINENO}." >&2' ERR

PROJECT_ROOT="${PROJECT_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$PROJECT_ROOT"

if [[ -f "$PROJECT_ROOT/.env.local" ]]; then
    set -a
    source "$PROJECT_ROOT/.env.local"
    set +a
fi

VENV_DIR="${VENV_DIR:-${XDG_CACHE_HOME:-$HOME/.cache}/laya-adverse-media/venv}"
DATASET_ID="${LAYA_ACTIVE_DATASET:-}"
ANNOTATION_SET_ID="${LAYA_BENCHMARK_LABEL_SET:-}"
if [[ -z "$DATASET_ID" || -z "$ANNOTATION_SET_ID" ]]; then
    echo "Set LAYA_ACTIVE_DATASET and LAYA_BENCHMARK_LABEL_SET in .env.local." >&2
    exit 1
fi
DATASET_DIR="${LAYA_DATASETS_DIR:-$PROJECT_ROOT/datasets}/$DATASET_ID"
CORPUS="${LAYA_BENCHMARK_CORPUS:-$DATASET_DIR/corpus.jsonl}"
LABELS="${LAYA_BENCHMARK_LABELS:-$DATASET_DIR/annotations/$ANNOTATION_SET_ID.jsonl}"
PREPARED_DIR="${PREPARED_DIR:-$PROJECT_ROOT/artifacts/prepared-data/$DATASET_ID/$ANNOTATION_SET_ID}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/models/fine-tuned/laya-adverse-media-$DATASET_ID-$ANNOTATION_SET_ID}"
PYTHON="$VENV_DIR/bin/python"
TORCHRUN="$VENV_DIR/bin/torchrun"

export TOKENIZERS_PARALLELISM=false

if [[ ! -x "$PYTHON" || ! -x "$TORCHRUN" ]]; then
    echo "Fine-tuning environment not found at $VENV_DIR." >&2
    echo "Run $PROJECT_ROOT/scripts/setup_environment.sh first." >&2
    exit 1
fi

required_prepared_files=(train.pt calibration.pt manifest.json split-assignments.jsonl)
prepared=true
for filename in "${required_prepared_files[@]}"; do
    if [[ ! -f "$PREPARED_DIR/$filename" ]]; then
        prepared=false
        break
    fi
done

if [[ "$prepared" == false ]]; then
    if [[ ! -f "$CORPUS" || ! -f "$LABELS" ]]; then
        echo "Prepared tensors are incomplete and external training inputs are unavailable." >&2
        echo "Expected dataset bundle inputs at $CORPUS and $LABELS." >&2
        exit 1
    fi
    echo "Preparing training tensors from external labels"
    "$PYTHON" scripts/fine_tune_laya.py prepare \
        --label-source external_labels \
        --corpus "$CORPUS" \
        --labels "$LABELS" \
        --output-dir "$PREPARED_DIR"
else
    echo "Using prepared tensors: $PREPARED_DIR"
fi

CUDA_SUMMARY="$("$PYTHON" - <<'PY'
import json
import torch

print(json.dumps({
    "torch": torch.__version__,
    "cuda_runtime": torch.version.cuda,
    "cuda_available": torch.cuda.is_available(),
    "devices": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
}))
PY
)"
echo "CUDA environment: $CUDA_SUMMARY"

GPU_COUNT="$("$PYTHON" -c 'import torch; print(torch.cuda.device_count())')"
if (( GPU_COUNT < 1 )); then
    echo "No CUDA GPU is visible to PyTorch. Check the NVIDIA driver and CUDA wheel." >&2
    exit 1
fi

NPROC_PER_NODE="${NPROC_PER_NODE:-$GPU_COUNT}"
if ! [[ "$NPROC_PER_NODE" =~ ^[1-9][0-9]*$ ]] || (( NPROC_PER_NODE > GPU_COUNT )); then
    echo "NPROC_PER_NODE must be between 1 and the visible GPU count ($GPU_COUNT)." >&2
    exit 1
fi

if [[ -f "$OUTPUT_DIR/model.safetensors" && "${ALLOW_OVERWRITE:-0}" != "1" ]]; then
    echo "A completed model already exists at $OUTPUT_DIR." >&2
    echo "Set OUTPUT_DIR to a new path or ALLOW_OVERWRITE=1." >&2
    exit 1
fi

train_args=(
    scripts/fine_tune_laya.py train
    --prepared-dir "$PREPARED_DIR"
    --output-dir "$OUTPUT_DIR"
)
train_args+=("$@")

echo "Starting fine-tuning with $NPROC_PER_NODE GPU process(es)"
if (( NPROC_PER_NODE == 1 )); then
    "$PYTHON" "${train_args[@]}"
else
    "$TORCHRUN" --standalone --nproc_per_node="$NPROC_PER_NODE" "${train_args[@]}"
fi

echo "Training complete. Start the benchmark and fine-tuning UI with:"
printf '  %q\n' "$PROJECT_ROOT/scripts/run_ui.sh"