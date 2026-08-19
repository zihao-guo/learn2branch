#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/activate.sh"
cd "$LEARN2BRANCH_ROOT"

usage() {
    echo "Usage: $0 [gcnn|competitors|all]" >&2
}

family="${1:-all}"
case "$family" in
    gcnn|competitors|all|_task) ;;
    *) usage; exit 2 ;;
esac

GCNN_JOBS="${LEARN2BRANCH_GCNN_JOBS:-8}"
GCNN_CORES_PER_JOB="${LEARN2BRANCH_GCNN_CORES_PER_JOB:-8}"
GCNN_CPU_OFFSET="${LEARN2BRANCH_GCNN_CPU_OFFSET:-0}"
COMPETITOR_JOBS="${LEARN2BRANCH_COMPETITOR_JOBS:-12}"
COMPETITOR_CPU_OFFSET="${LEARN2BRANCH_COMPETITOR_CPU_OFFSET:-64}"
COMPETITOR_CPU_SPAN="${LEARN2BRANCH_COMPETITOR_CPU_SPAN:-32}"

for value in "$GCNN_JOBS" "$GCNN_CORES_PER_JOB" "$COMPETITOR_JOBS" "$COMPETITOR_CPU_SPAN"; do
    if [[ ! "$value" =~ ^[1-9][0-9]*$ ]]; then
        echo "Concurrency and CPU span values must be positive integers" >&2
        exit 2
    fi
done

ARTIFACT_ROOT="$LEARN2BRANCH_ROOT/reproduction_artifacts/full_training"
LOG_DIR="$ARTIFACT_ROOT/logs"
MARKER_DIR="$ARTIFACT_ROOT/markers"
mkdir -p "$LOG_DIR" "$MARKER_DIR"

task_output_dir() {
    local task="$1"
    local kind problem model seed model_dir
    IFS=: read -r kind problem model seed <<<"$task"
    if [[ "$kind" == "gcnn" ]]; then
        echo "$LEARN2BRANCH_ROOT/trained_models/$problem/$model/$seed"
        return
    fi
    case "$model" in
        extratrees) model_dir="extratrees_gcnn_agg" ;;
        svmrank) model_dir="svmrank_khalil" ;;
        lambdamart) model_dir="lambdamart_khalil" ;;
        *) echo "Unknown competitor: $model" >&2; return 2 ;;
    esac
    echo "$LEARN2BRANCH_ROOT/trained_models/$problem/$model_dir/$seed"
}

task_is_complete() {
    local task="$1"
    local kind problem model seed output_dir
    IFS=: read -r kind problem model seed <<<"$task"
    output_dir="$(task_output_dir "$task")"
    if [[ "$kind" == "gcnn" ]]; then
        [[ -s "$output_dir/best_params.pkl" ]] &&
            [[ -s "$output_dir/log.txt" ]] &&
            grep -q "BEST VALID LOSS" "$output_dir/log.txt"
        return
    fi
    [[ -s "$output_dir/feat_specs.pkl" ]] || return 1
    [[ -s "$output_dir/normalization.pkl" ]] || return 1
    [[ -s "$output_dir/log.txt" ]] || return 1
    case "$model" in
        extratrees)
            [[ -s "$output_dir/model.pkl" ]] && grep -q "Validation RMSE" "$output_dir/log.txt"
            ;;
        lambdamart)
            [[ -s "$output_dir/model.pkl" ]] && grep -q "Validation log-NDCG" "$output_dir/log.txt"
            ;;
        svmrank)
            [[ -s "$output_dir/model.txt" ]] && grep -q "Best model with C=" "$output_dir/log.txt"
            ;;
    esac
}

run_task() {
    local task="$1"
    local kind problem model seed output_dir marker logfile slot cpu_start cpu_end cpu
    IFS=: read -r kind problem model seed <<<"$task"
    output_dir="$(task_output_dir "$task")"
    marker="$MARKER_DIR/${task//:/_}.complete"
    logfile="$LOG_DIR/${task//:/_}.log"

    if [[ -f "$marker" ]]; then
        if ! task_is_complete "$task"; then
            echo "[$task] marker exists but outputs are incomplete" >&2
            return 1
        fi
        echo "[$task] already complete"
        return
    fi
    if task_is_complete "$task"; then
        touch "$marker"
        echo "[$task] recovered completion marker"
        return
    fi
    if [[ -e "$output_dir" ]]; then
        echo "[$task] refusing to overwrite incomplete output: $output_dir" >&2
        return 1
    fi

    slot="${LEARN2BRANCH_SLOT:-0}"
    echo "[$task] starting; external log: $logfile"
    if [[ "$kind" == "gcnn" ]]; then
        cpu_start=$((GCNN_CPU_OFFSET + slot * GCNN_CORES_PER_JOB))
        cpu_end=$((cpu_start + GCNN_CORES_PER_JOB - 1))
        if ! taskset -c "$cpu_start-$cpu_end" \
            env OMP_NUM_THREADS="$GCNN_CORES_PER_JOB" \
                MKL_NUM_THREADS="$GCNN_CORES_PER_JOB" \
                OPENBLAS_NUM_THREADS="$GCNN_CORES_PER_JOB" \
            python 03_train_gcnn.py "$problem" -m "$model" -s "$seed" -g -1 \
            >"$logfile" 2>&1; then
            tail -n 50 "$logfile" >&2 || true
            return 1
        fi
    else
        cpu=$((COMPETITOR_CPU_OFFSET + slot % COMPETITOR_CPU_SPAN))
        if ! taskset -c "$cpu" python 03_train_competitor.py \
            "$problem" -m "$model" -s "$seed" >"$logfile" 2>&1; then
            tail -n 50 "$logfile" >&2 || true
            return 1
        fi
    fi

    if ! task_is_complete "$task"; then
        echo "[$task] command returned successfully but expected outputs are missing" >&2
        tail -n 50 "$logfile" >&2 || true
        return 1
    fi
    touch "$marker"
    echo "[$task] complete"
}

run_gcnn_queue() {
    local tasks=()
    local problem seed model
    for problem in setcover cauctions facilities indset; do
        for seed in 0 1 2 3 4; do
            tasks+=("gcnn:$problem:baseline:$seed")
            if [[ "$problem" == "setcover" ]]; then
                for model in mean_convolution no_prenorm; do
                    tasks+=("gcnn:$problem:$model:$seed")
                done
            fi
        done
    done
    printf "%s\n" "${tasks[@]}" | xargs -r -n 1 -P "$GCNN_JOBS" \
        --process-slot-var=LEARN2BRANCH_SLOT bash "$SCRIPT_DIR/run_full_training_queue.sh" _task
}

run_competitor_queue() {
    local tasks=()
    local problem seed model
    for problem in setcover cauctions facilities indset; do
        for seed in 0 1 2 3 4; do
            for model in extratrees svmrank lambdamart; do
                tasks+=("competitor:$problem:$model:$seed")
            done
        done
    done
    printf "%s\n" "${tasks[@]}" | xargs -r -n 1 -P "$COMPETITOR_JOBS" \
        --process-slot-var=LEARN2BRANCH_SLOT bash "$SCRIPT_DIR/run_full_training_queue.sh" _task
}

case "$family" in
    _task)
        [[ $# -eq 2 ]] || { usage; exit 2; }
        run_task "$2"
        ;;
    gcnn) run_gcnn_queue ;;
    competitors) run_competitor_queue ;;
    all)
        run_gcnn_queue
        run_competitor_queue
        ;;
esac
