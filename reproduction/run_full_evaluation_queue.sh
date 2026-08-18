#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/activate.sh"
cd "$LEARN2BRANCH_ROOT"

usage() {
    echo "Usage: $0 [all|_task TASK]" >&2
}

mode="${1:-all}"
case "$mode" in
    all|_task) ;;
    *) usage; exit 2 ;;
esac

EVALUATION_JOBS="${LEARN2BRANCH_EVALUATION_JOBS:-80}"
EVALUATION_CPU_OFFSET="${LEARN2BRANCH_EVALUATION_CPU_OFFSET:-0}"
EVALUATION_CPU_SPAN="${LEARN2BRANCH_EVALUATION_CPU_SPAN:-96}"
EVALUATION_TIME_LIMIT="${LEARN2BRANCH_EVALUATION_TIME_LIMIT:-3600}"

for value in "$EVALUATION_JOBS" "$EVALUATION_CPU_SPAN" "$EVALUATION_TIME_LIMIT"; do
    if [[ ! "$value" =~ ^[1-9][0-9]*$ ]]; then
        echo "Jobs, CPU span, and time limit must be positive integers" >&2
        exit 2
    fi
done

ARTIFACT_ROOT="$LEARN2BRANCH_ROOT/reproduction_artifacts/full_evaluation/${EVALUATION_TIME_LIMIT}s"
RESULT_DIR="$ARTIFACT_ROOT/results"
LOG_DIR="$ARTIFACT_ROOT/logs"
MARKER_DIR="$ARTIFACT_ROOT/markers"
mkdir -p "$RESULT_DIR" "$LOG_DIR" "$MARKER_DIR"

task_result_file() {
    local task="$1"
    echo "$RESULT_DIR/${task//:/_}.csv"
}

task_is_complete() {
    local task="$1"
    local result_file
    result_file="$(task_result_file "$task")"
    [[ -s "$result_file" ]] || return 1
    [[ "$(wc -l < "$result_file")" -eq 2 ]] || return 1
    head -n 1 "$result_file" | grep -q '^policy,seed,type,instance,nnodes,nlps,stime,gap,status,'
}

run_task() {
    local task="$1"
    local problem instance_index policy_type policy_name seed result_file marker logfile slot cpu
    IFS=: read -r problem instance_index policy_type policy_name seed <<<"$task"
    result_file="$(task_result_file "$task")"
    marker="$MARKER_DIR/${task//:/_}.complete"
    logfile="$LOG_DIR/${task//:/_}.log"

    if [[ -f "$marker" ]]; then
        if ! task_is_complete "$task"; then
            echo "[$task] marker exists but result is incomplete" >&2
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
    if [[ -e "$result_file" ]]; then
        echo "[$task] refusing to overwrite incomplete result: $result_file" >&2
        return 1
    fi

    slot="${LEARN2BRANCH_SLOT:-0}"
    cpu=$((EVALUATION_CPU_OFFSET + slot % EVALUATION_CPU_SPAN))
    echo "[$task] starting on CPU $cpu"
    if ! taskset -c "$cpu" \
        env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
        python 05_evaluate.py "$problem" -g -1 \
            --instance-index "$instance_index" \
            --policy-type "$policy_type" \
            --policy-name "$policy_name" \
            --seed "$seed" \
            --time-limit "$EVALUATION_TIME_LIMIT" \
            --result-file "$result_file" >"$logfile" 2>&1; then
        tail -n 50 "$logfile" >&2 || true
        return 1
    fi
    if ! task_is_complete "$task"; then
        echo "[$task] command returned successfully but the expected row is missing" >&2
        tail -n 50 "$logfile" >&2 || true
        return 1
    fi
    touch "$marker"
    echo "[$task] complete"
}

append_problem_tasks() {
    local problem="$1"
    shift
    local policy instance_index seed
    for policy in "$@"; do
        for seed in 0 1 2 3 4; do
            for instance_index in {0..59}; do
                TASKS+=("$problem:$instance_index:$policy:$seed")
            done
        done
    done
}

run_queue() {
    TASKS=()
    append_problem_tasks setcover \
        internal:relpscost \
        ml-competitor:extratrees_gcnn_agg \
        ml-competitor:lambdamart_khalil \
        ml-competitor:svmrank_khalil \
        gcnn:baseline \
        gcnn:mean_convolution \
        gcnn:no_prenorm
    for problem in cauctions facilities indset; do
        append_problem_tasks "$problem" \
            internal:relpscost \
            ml-competitor:extratrees_gcnn_agg \
            ml-competitor:lambdamart_khalil \
            ml-competitor:svmrank_khalil \
            gcnn:baseline
    done
    if [[ "${#TASKS[@]}" -ne 6600 ]]; then
        echo "Internal error: generated ${#TASKS[@]} tasks instead of 6600" >&2
        return 1
    fi
    printf "%s\n" "${TASKS[@]}" | xargs -r -n 1 -P "$EVALUATION_JOBS" \
        --process-slot-var=LEARN2BRANCH_SLOT bash "$SCRIPT_DIR/run_full_evaluation_queue.sh" _task
}

case "$mode" in
    _task)
        [[ $# -eq 2 ]] || { usage; exit 2; }
        run_task "$2"
        ;;
    all) run_queue ;;
esac
