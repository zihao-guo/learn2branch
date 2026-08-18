#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/activate.sh"
cd "$LEARN2BRANCH_ROOT"

usage() {
    echo "Usage: $0 [--problem all|setcover|cauctions|facilities|indset]" >&2
    echo "          [--phase all|instances|samples|train|test|evaluate] [--jobs N]" >&2
}

problem="all"
phase="all"
jobs="4"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --problem)
            problem="$2"
            shift 2
            ;;
        --phase)
            phase="$2"
            shift 2
            ;;
        --jobs)
            jobs="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            usage
            exit 2
            ;;
    esac
done

case "$problem" in
    all) problems=(setcover cauctions facilities indset) ;;
    setcover|cauctions|facilities|indset) problems=("$problem") ;;
    *) usage; exit 2 ;;
esac

case "$phase" in
    all) phases=(instances samples train test evaluate) ;;
    instances|samples|train|test|evaluate) phases=("$phase") ;;
    *) usage; exit 2 ;;
esac

run_training() {
    local current_problem="$1"
    local seed
    local gcnn_models=(baseline)
    local model_name
    local competitor

    if [[ "$current_problem" == "setcover" ]]; then
        gcnn_models+=(mean_convolution no_prenorm)
    fi

    for seed in 0 1 2 3 4; do
        for model_name in "${gcnn_models[@]}"; do
            python 03_train_gcnn.py "$current_problem" -m "$model_name" -s "$seed" -g -1
        done
        for competitor in extratrees svmrank lambdamart; do
            python 03_train_competitor.py "$current_problem" -m "$competitor" -s "$seed"
        done
    done
}

python reproduction/verify_environment.py

for current_problem in "${problems[@]}"; do
    for current_phase in "${phases[@]}"; do
        echo "[$current_problem] phase: $current_phase"
        case "$current_phase" in
            instances)
                python 01_generate_instances.py "$current_problem" -s 0
                ;;
            samples)
                python 02_generate_dataset.py "$current_problem" -s 0 -j "$jobs"
                ;;
            train)
                run_training "$current_problem"
                ;;
            test)
                python 04_test.py "$current_problem" -g -1
                ;;
            evaluate)
                python 05_evaluate.py "$current_problem" -g -1
                ;;
        esac
    done
done
