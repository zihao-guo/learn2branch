#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/activate.sh"
cd "$LEARN2BRANCH_ROOT"

usage() {
    echo "Usage: $0 [all|_task PROBLEM]" >&2
}

mode="${1:-all}"
case "$mode" in
    all|_task) ;;
    *) usage; exit 2 ;;
esac

TEST_JOBS="${LEARN2BRANCH_TEST_JOBS:-4}"
TEST_CORES_PER_JOB="${LEARN2BRANCH_TEST_CORES_PER_JOB:-16}"
TEST_CPU_OFFSET="${LEARN2BRANCH_TEST_CPU_OFFSET:-0}"
for value in "$TEST_JOBS" "$TEST_CORES_PER_JOB"; do
    if [[ ! "$value" =~ ^[1-9][0-9]*$ ]]; then
        echo "Concurrency and cores per job must be positive integers" >&2
        exit 2
    fi
done

ARTIFACT_ROOT="$LEARN2BRANCH_ROOT/reproduction_artifacts/full_test"
RESULT_DIR="$ARTIFACT_ROOT/results"
LOG_DIR="$ARTIFACT_ROOT/logs"
MARKER_DIR="$ARTIFACT_ROOT/markers"
mkdir -p "$RESULT_DIR" "$LOG_DIR" "$MARKER_DIR"

task_is_complete() {
    local problem="$1"
    local result_file="$RESULT_DIR/$problem.csv"
    local expected_lines=21
    if [[ "$problem" == "setcover" ]]; then
        expected_lines=31
    fi
    [[ -s "$result_file" ]] || return 1
    [[ "$(wc -l < "$result_file")" -eq "$expected_lines" ]] || return 1
    head -n 1 "$result_file" | grep -q '^policy,seed,acc@1,acc@3,acc@5,acc@10'
}

run_task() {
    local problem="$1"
    local result_file="$RESULT_DIR/$problem.csv"
    local logfile="$LOG_DIR/$problem.log"
    local marker="$MARKER_DIR/$problem.complete"
    local slot cpu_start cpu_end

    case "$problem" in
        setcover|cauctions|facilities|indset) ;;
        *) echo "Unknown problem: $problem" >&2; return 2 ;;
    esac
    if [[ -f "$marker" ]]; then
        if ! task_is_complete "$problem"; then
            echo "[$problem] marker exists but result is incomplete" >&2
            return 1
        fi
        echo "[$problem] already complete"
        return
    fi
    if task_is_complete "$problem"; then
        touch "$marker"
        echo "[$problem] recovered completion marker"
        return
    fi
    if [[ -e "$result_file" ]]; then
        echo "[$problem] refusing to overwrite incomplete result: $result_file" >&2
        return 1
    fi

    slot="${LEARN2BRANCH_SLOT:-0}"
    cpu_start=$((TEST_CPU_OFFSET + slot * TEST_CORES_PER_JOB))
    cpu_end=$((cpu_start + TEST_CORES_PER_JOB - 1))
    echo "[$problem] starting on CPUs $cpu_start-$cpu_end"
    if ! taskset -c "$cpu_start-$cpu_end" \
        env OMP_NUM_THREADS="$TEST_CORES_PER_JOB" \
            MKL_NUM_THREADS="$TEST_CORES_PER_JOB" \
            OPENBLAS_NUM_THREADS="$TEST_CORES_PER_JOB" \
        python 04_test.py "$problem" -g -1 --result-file "$result_file" \
        >"$logfile" 2>&1; then
        tail -n 50 "$logfile" >&2 || true
        return 1
    fi
    if ! task_is_complete "$problem"; then
        echo "[$problem] command returned successfully but expected rows are missing" >&2
        tail -n 50 "$logfile" >&2 || true
        return 1
    fi
    touch "$marker"
    echo "[$problem] complete"
}

run_queue() {
    printf "%s\n" setcover cauctions facilities indset | \
        xargs -r -n 1 -P "$TEST_JOBS" --process-slot-var=LEARN2BRANCH_SLOT \
        bash "$SCRIPT_DIR/run_full_test_queue.sh" _task
}

case "$mode" in
    _task)
        [[ $# -eq 2 ]] || { usage; exit 2; }
        run_task "$2"
        ;;
    all) run_queue ;;
esac
