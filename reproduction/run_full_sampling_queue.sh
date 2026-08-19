#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/activate.sh"
cd "$LEARN2BRANCH_ROOT"

jobs="${LEARN2BRANCH_SAMPLE_JOBS:-80}"
problems=("$@")
if [[ ${#problems[@]} -eq 0 ]]; then
    # Slowest first so a long tail does not remain after faster datasets finish.
    problems=(indset facilities setcover cauctions)
fi

log_dir="reproduction_artifacts/full/logs"
marker_dir="reproduction_artifacts/full/markers"
mkdir -p "$log_dir" "$marker_dir"

for problem in "${problems[@]}"; do
    case "$problem" in
        setcover|cauctions|facilities|indset) ;;
        *)
            echo "Unknown problem: $problem" >&2
            exit 2
            ;;
    esac

    marker="$marker_dir/${problem}_samples.complete"
    if [[ -f "$marker" ]]; then
        echo "[$problem] completion marker exists; skipping."
        continue
    fi
    if [[ -d "data/samples/$problem" ]]; then
        echo "[$problem] partial sample directory exists without a completion marker." >&2
        echo "Move it aside before restarting to avoid mixing runs." >&2
        exit 1
    fi

    log="$log_dir/${problem}_samples_${jobs}workers.log"
    echo "[$problem] collecting 100000/20000/20000 samples with $jobs workers"
    bash reproduction/run_full.sh --problem "$problem" --phase samples --jobs "$jobs" \
        > "$log" 2>&1
    touch "$marker"
    echo "[$problem] complete; log: $log"
done
