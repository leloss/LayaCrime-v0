#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$PROJECT_ROOT"

if [[ -f "$PROJECT_ROOT/.env.local" ]]; then
    set -a
    source <(sed 's/\r$//' "$PROJECT_ROOT/.env.local")
    set +a
fi

VENV_DIR="${VENV_DIR:-${XDG_CACHE_HOME:-$HOME/.cache}/laya-adverse-media/venv}"
PYTHON="$VENV_DIR/bin/python"
MATRIX_ROOT="${MATRIX_ROOT:-$PROJECT_ROOT/artifacts/public-experiment-matrix}"
RESULTS_TABLE="${RESULTS_TABLE:-$MATRIX_ROOT/results.tsv}"
AGGREGATE_TABLE="${AGGREGATE_TABLE:-$MATRIX_ROOT/aggregate.tsv}"
TUNING_DIR="${TUNING_DIR:-$PROJECT_ROOT/datasets/adverse-media-public-tuning-2000}"
HOLDOUT_DIR="${HOLDOUT_DIR:-$PROJECT_ROOT/datasets/adverse-media-public-holdout-1000}"
TRAIN_CORPUS="${TRAIN_CORPUS:-$TUNING_DIR/corpus.jsonl}"
TRAIN_LABELS="${TRAIN_LABELS:-$TUNING_DIR/annotations/human.jsonl}"
HOLDOUT_CORPUS="${HOLDOUT_CORPUS:-$HOLDOUT_DIR/corpus.jsonl}"
HOLDOUT_LABELS="${HOLDOUT_LABELS:-$HOLDOUT_DIR/annotations/human.jsonl}"
PROMPT_FILE="${PROMPT_FILE:-$TUNING_DIR/dataset.json}"
MATRIX_STRATEGIES="${MATRIX_STRATEGIES:-public_natural_staged balanced maximum_task_adaptation preserve_entity_matching relational_low_drift relational_adaptation relational_extended relational_rlcd}"
MATRIX_PARTITIONS="${MATRIX_PARTITIONS:-group_k_fold:5:0 group_k_fold:5:1 group_k_fold:5:2 group_k_fold:5:3 group_k_fold:5:4}"
MATRIX_SEEDS="${MATRIX_SEEDS:-20260923 20260924 20260925}"
KEEP_GOING="${KEEP_GOING:-1}"
REPORT_EVERY="${REPORT_EVERY:-25}"
MATRIX_DRY_RUN="${MATRIX_DRY_RUN:-0}"
MATRIX_MAX_RUNS="${MATRIX_MAX_RUNS:-0}"
MATRIX_RUN_BENCHMARK="${MATRIX_RUN_BENCHMARK:-0}"
MATRIX_PLANNED_EPOCHS="${MATRIX_PLANNED_EPOCHS:-10}"
MATRIX_MAX_EPOCHS="${MATRIX_MAX_EPOCHS:-14}"
MATRIX_MINIMUM_EPOCHS="${MATRIX_MINIMUM_EPOCHS:-4}"
MATRIX_EARLY_STOPPING_PATIENCE="${MATRIX_EARLY_STOPPING_PATIENCE:-2}"
MATRIX_EARLY_STOPPING_MIN_DELTA="${MATRIX_EARLY_STOPPING_MIN_DELTA:-0.001}"
MATRIX_EXTENSION_EPOCHS="${MATRIX_EXTENSION_EPOCHS:-2}"

export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export HF_HUB_OFFLINE=1

if [[ ! -x "$PYTHON" ]]; then
    echo "Python environment not found at $VENV_DIR. Run scripts/setup_environment.sh first." >&2
    exit 1
fi

required_inputs=("$TRAIN_CORPUS" "$TRAIN_LABELS" "$PROMPT_FILE")
if [[ "$MATRIX_RUN_BENCHMARK" == "1" ]]; then
    required_inputs+=("$HOLDOUT_CORPUS" "$HOLDOUT_LABELS")
fi
for path in "${required_inputs[@]}"; do
    if [[ ! -f "$path" ]]; then
        echo "Required matrix input is missing: $path" >&2
        exit 1
    fi
done

MODEL_BUNDLE="${LAYA_MODEL_BUNDLE:-$PROJECT_ROOT/models/models--convaiinnovations--laya}"
if [[ -n "${LAYA_MODEL_PATH:-}" ]]; then
    BASE_MODEL_DIR="$(cd -- "$LAYA_MODEL_PATH" && pwd)"
else
    REVISION_FILE="$MODEL_BUNDLE/refs/main"
    if [[ ! -f "$REVISION_FILE" ]]; then
        echo "Base Laya revision file is missing: $REVISION_FILE" >&2
        exit 1
    fi
    BASE_MODEL_DIR="$MODEL_BUNDLE/snapshots/$(tr -d '\r\n' < "$REVISION_FILE")"
fi
if [[ ! -f "$BASE_MODEL_DIR/model.safetensors" ]]; then
    echo "Base Laya checkpoint is incomplete: $BASE_MODEL_DIR" >&2
    exit 1
fi

if [[ "$MATRIX_DRY_RUN" != "1" && "${MATRIX_ALLOW_SHARED_GPU:-0}" != "1" ]] && pgrep -f '/laya-adverse-media|/llama-server|fine_tune_laya.py train' >/dev/null 2>&1; then
    echo "An inference or training process is already using the GPU." >&2
    echo "Stop it first, or set MATRIX_ALLOW_SHARED_GPU=1 only when GPU isolation is intentional." >&2
    exit 1
fi

mkdir -p "$MATRIX_ROOT"/{prepared,models,runs,logs,benchmarks}
LOCK_DIR="$MATRIX_ROOT/.lock"
if ! mkdir "$LOCK_DIR" 2>/dev/null; then
    echo "Another experiment matrix appears to be running: $LOCK_DIR" >&2
    exit 1
fi
trap 'rm -rf "$LOCK_DIR"' EXIT

log() {
    printf '[%s] %s\n' "$(date -u +'%Y-%m-%dT%H:%M:%SZ')" "$*"
}

strategy_args() {
    case "$1" in
        preserve_entity_matching)
            printf '%s\n' --group-size 4 --encoder-lr 0 --head-lr 5e-5 --rl-weight 0 --weight-decay 0.01 --sigma-start 0.4 --sigma-end 0.1 --encoder-warmup-epochs 0 --lr-warmup-ratio 0 --label-smoothing 0
            ;;
        balanced)
            printf '%s\n' --group-size 4 --encoder-lr 3e-6 --head-lr 5e-5 --rl-weight 0 --weight-decay 0.01 --sigma-start 0.4 --sigma-end 0.1 --encoder-warmup-epochs 0 --lr-warmup-ratio 0 --label-smoothing 0
            ;;
        maximum_task_adaptation)
            printf '%s\n' --group-size 4 --encoder-lr 1e-5 --head-lr 5e-5 --rl-weight 0 --weight-decay 0.01 --sigma-start 0.4 --sigma-end 0.1 --encoder-warmup-epochs 0 --lr-warmup-ratio 0 --label-smoothing 0
            ;;
        relational_low_drift)
            printf '%s\n' --group-size 4 --encoder-lr 1e-6 --head-lr 3e-5 --rl-weight 0 --weight-decay 0.02 --sigma-start 0.4 --sigma-end 0.1 --encoder-warmup-epochs 0 --lr-warmup-ratio 0 --label-smoothing 0
            ;;
        relational_adaptation)
            printf '%s\n' --group-size 4 --encoder-lr 5e-6 --head-lr 5e-5 --rl-weight 0 --weight-decay 0.02 --sigma-start 0.4 --sigma-end 0.1 --encoder-warmup-epochs 0 --lr-warmup-ratio 0 --label-smoothing 0
            ;;
        relational_extended)
            printf '%s\n' --group-size 4 --encoder-lr 3e-6 --head-lr 5e-5 --rl-weight 0 --weight-decay 0.02 --sigma-start 0.4 --sigma-end 0.1 --encoder-warmup-epochs 0 --lr-warmup-ratio 0 --label-smoothing 0
            ;;
        relational_rlcd)
            printf '%s\n' --group-size 4 --encoder-lr 3e-6 --head-lr 5e-5 --rl-weight 0.1 --weight-decay 0.02 --sigma-start 0.3 --sigma-end 0.05 --encoder-warmup-epochs 0 --lr-warmup-ratio 0 --label-smoothing 0
            ;;
        public_natural_staged)
            printf '%s\n' --group-size 4 --encoder-lr 1e-6 --head-lr 2e-5 --rl-weight 0 --weight-decay 0.05 --sigma-start 0.3 --sigma-end 0.05 --encoder-warmup-epochs 1 --lr-warmup-ratio 0.1 --label-smoothing 0.05
            ;;
        *)
            echo "Unknown strategy: $1" >&2
            return 1
            ;;
    esac
}

partition_args() {
    local spec="$1"
    local strategy fold_count fold_index
    IFS=: read -r strategy fold_count fold_index <<< "$spec"
    printf '%s\n' --partition-strategy "$strategy"
    if [[ "$strategy" == "group_k_fold" ]]; then
        printf '%s\n' --fold-count "${fold_count:-5}" --fold-index "${fold_index:-0}"
    elif [[ "$strategy" == "leave_one_group_out" ]]; then
        printf '%s\n' --fold-index "${fold_index:-0}"
    fi
}

write_run_state() {
    local status="$1"
    local error="${2:-}"
    RUN_STATUS="$status" RUN_ERROR="$error" RUN_ID="$RUN_ID" STRATEGY="$STRATEGY" \
        PARTITION="$PARTITION" SEED="$SEED" PREPARED_DIR="$PREPARED_DIR" \
        MODEL_DIR="$MODEL_DIR" BENCHMARK_REPORT="$BENCHMARK_REPORT" LOG_FILE="$LOG_FILE" \
        "$PYTHON" - "$RUN_DIR/run.json" <<'PY'
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

path = Path(sys.argv[1])
previous = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
value = {
    **previous,
    "run_id": os.environ["RUN_ID"],
    "strategy": os.environ["STRATEGY"],
    "partition": os.environ["PARTITION"],
    "seed": int(os.environ["SEED"]),
    "prepared_dir": os.environ["PREPARED_DIR"],
    "model_dir": os.environ["MODEL_DIR"],
    "benchmark_report": os.environ["BENCHMARK_REPORT"],
    "log": os.environ["LOG_FILE"],
    "status": os.environ["RUN_STATUS"],
    "error": os.environ.get("RUN_ERROR") or None,
    "updated_at": datetime.now(timezone.utc).isoformat(),
}
path.parent.mkdir(parents=True, exist_ok=True)
temporary = path.with_suffix(".tmp")
temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
temporary.replace(path)
PY
}

refresh_table() {
    "$PYTHON" scripts/summarize_public_experiment_matrix.py \
        --matrix-root "$MATRIX_ROOT" \
    --output "$RESULTS_TABLE" \
    --aggregate-output "$AGGREGATE_TABLE"
}

prepared_is_compatible() {
    PARTITION_SPEC="$PARTITION" MATRIX_SEED="$SEED" "$PYTHON" - "$PREPARED_DIR/manifest.json" <<'PY'
import json
import os
import sys
from pathlib import Path

path = Path(sys.argv[1])
if not path.is_file():
    raise SystemExit(1)
manifest = json.loads(path.read_text(encoding="utf-8"))
requested = os.environ["PARTITION_SPEC"].split(":")
partition = manifest.get("partition") or {}
compatible = (
    manifest.get("seed") == int(os.environ["MATRIX_SEED"])
    and partition.get("strategy") == requested[0]
)
if requested[0] == "group_k_fold":
    compatible = compatible and partition.get("fold_count") == int(requested[1] or 5)
    compatible = compatible and partition.get("calibration_fold") == int(requested[2] or 0)
elif requested[0] == "leave_one_group_out":
    compatible = compatible and partition.get("calibration_group") == int(requested[2] or 0)
raise SystemExit(0 if compatible else 1)
PY
}

run_experiment() {
    local partition_slug strategy_parameters partition_parameters config_id
    mapfile -t strategy_parameters < <(strategy_args "$STRATEGY")
    config_id="$("$PYTHON" - \
        "$STRATEGY" \
        "$MATRIX_PLANNED_EPOCHS" \
        "$MATRIX_MAX_EPOCHS" \
        "$MATRIX_MINIMUM_EPOCHS" \
        "$MATRIX_EARLY_STOPPING_PATIENCE" \
        "$MATRIX_EARLY_STOPPING_MIN_DELTA" \
        "$MATRIX_EXTENSION_EPOCHS" \
        "${strategy_parameters[@]}" <<'PY'
import hashlib
import json
import sys

payload = json.dumps(["adaptive-v1", *sys.argv[1:]], separators=(",", ":"))
print(hashlib.sha256(payload.encode("utf-8")).hexdigest()[:10])
PY
)"
    partition_slug="${PARTITION//:/-}"
    RUN_ID="${partition_slug}__${STRATEGY}__config-${config_id}__seed-${SEED}"
    PREPARED_DIR="$MATRIX_ROOT/prepared/$partition_slug/seed-$SEED"
    MODEL_DIR="$MATRIX_ROOT/models/$RUN_ID"
    RUN_DIR="$MATRIX_ROOT/runs/$RUN_ID"
    LOG_FILE="$MATRIX_ROOT/logs/$RUN_ID.log"
    BENCHMARK_LEDGER="$MATRIX_ROOT/benchmarks/$RUN_ID.jsonl"
    BENCHMARK_REPORT="$MATRIX_ROOT/benchmarks/$RUN_ID.json"
    mkdir -p "$RUN_DIR" "$(dirname "$LOG_FILE")" "$(dirname "$BENCHMARK_LEDGER")"
    touch "$LOG_FILE"

    log "run=$RUN_ID event=start" | tee -a "$LOG_FILE"
    write_run_state running

    mapfile -t partition_parameters < <(partition_args "$PARTITION")
    if ! prepared_is_compatible; then
        log "run=$RUN_ID event=prepare partition=$PARTITION seed=$SEED" | tee -a "$LOG_FILE"
        rm -rf "$PREPARED_DIR"
        if ! "$PYTHON" scripts/fine_tune_laya.py prepare \
            --label-source external_labels \
            --corpus "$TRAIN_CORPUS" \
            --labels "$TRAIN_LABELS" \
            --model-dir "$BASE_MODEL_DIR" \
            --output-dir "$PREPARED_DIR" \
            --seed "$SEED" \
            --prompt-file "$PROMPT_FILE" \
            "${partition_parameters[@]}" 2>&1 | tee -a "$LOG_FILE"; then
            write_run_state failed "preparation failed"
            refresh_table
            return 1
        fi
    else
        log "run=$RUN_ID event=prepare_skip reason=compatible" | tee -a "$LOG_FILE"
    fi

    if [[ ! -f "$MODEL_DIR/training_report.json" || ! -f "$MODEL_DIR/model.safetensors" ]]; then
        if [[ -d "$MODEL_DIR" ]]; then
            mv "$MODEL_DIR" "$MODEL_DIR.failed-$(date -u +'%Y%m%dT%H%M%SZ')"
        fi
        log "run=$RUN_ID event=train strategy=$STRATEGY" | tee -a "$LOG_FILE"
        if ! "$PYTHON" scripts/fine_tune_laya.py train \
            --prepared-dir "$PREPARED_DIR" \
            --model-dir "$BASE_MODEL_DIR" \
            --output-dir "$MODEL_DIR" \
            --strategy "$STRATEGY" \
            --batch-size "${BATCH_SIZE:-4}" \
            --eval-batch-size "${EVAL_BATCH_SIZE:-16}" \
            --gradient-accumulation "${GRADIENT_ACCUMULATION:-8}" \
            --epochs "$MATRIX_PLANNED_EPOCHS" \
            --max-epochs "$MATRIX_MAX_EPOCHS" \
            --minimum-epochs "$MATRIX_MINIMUM_EPOCHS" \
            --early-stopping-patience "$MATRIX_EARLY_STOPPING_PATIENCE" \
            --early-stopping-min-delta "$MATRIX_EARLY_STOPPING_MIN_DELTA" \
            --extension-epochs "$MATRIX_EXTENSION_EPOCHS" \
            --seed "$SEED" \
            --log-every "${LOG_EVERY:-50}" \
            "${strategy_parameters[@]}" 2>&1 | tee -a "$LOG_FILE"; then
            write_run_state failed "training failed"
            refresh_table
            return 1
        fi
    else
        log "run=$RUN_ID event=train_skip reason=complete" | tee -a "$LOG_FILE"
    fi

    if [[ "$MATRIX_RUN_BENCHMARK" == "1" ]]; then
        write_run_state benchmarking
        log "run=$RUN_ID event=benchmark" | tee -a "$LOG_FILE"
        if ! "$PYTHON" scripts/benchmark_laya_checkpoint.py \
            --run-id "$RUN_ID" \
            --model-dir "$MODEL_DIR" \
            --corpus "$HOLDOUT_CORPUS" \
            --labels "$HOLDOUT_LABELS" \
            --output "$BENCHMARK_LEDGER" \
            --report "$BENCHMARK_REPORT" \
            --report-every "$REPORT_EVERY" \
            --resume 2>&1 | tee -a "$LOG_FILE"; then
            write_run_state failed "benchmark failed"
            refresh_table
            return 1
        fi
    else
        log "run=$RUN_ID event=benchmark_skip reason=exploration_protocol" | tee -a "$LOG_FILE"
    fi

    write_run_state complete
    refresh_table
    log "run=$RUN_ID event=complete table=$RESULTS_TABLE" | tee -a "$LOG_FILE"
}

log "matrix_start strategies=[$MATRIX_STRATEGIES] partitions=[$MATRIX_PARTITIONS] seeds=[$MATRIX_SEEDS]"
failures=0
planned=0
completed_runs=0
for SEED in $MATRIX_SEEDS; do
    for PARTITION in $MATRIX_PARTITIONS; do
        for STRATEGY in $MATRIX_STRATEGIES; do
            planned=$((planned + 1))
            if [[ "$MATRIX_DRY_RUN" == "1" ]]; then
                log "plan index=$planned run=${PARTITION//:/-}__${STRATEGY}__seed-${SEED}"
                continue
            fi
            if (( MATRIX_MAX_RUNS > 0 && completed_runs >= MATRIX_MAX_RUNS )); then
                log "matrix_stop reason=max_runs limit=$MATRIX_MAX_RUNS"
                refresh_table
                exit 0
            fi
            if ! run_experiment; then
                failures=$((failures + 1))
                log "run=${RUN_ID:-unknown} event=failed failures=$failures"
                if [[ "$KEEP_GOING" != "1" ]]; then
                    exit 1
                fi
            fi
            completed_runs=$((completed_runs + 1))
        done
    done
done
if [[ "$MATRIX_DRY_RUN" == "1" ]]; then
    log "matrix_plan_complete runs=$planned"
    exit 0
fi
refresh_table
log "matrix_complete failures=$failures table=$RESULTS_TABLE"
if (( failures > 0 )); then
    exit 1
fi
