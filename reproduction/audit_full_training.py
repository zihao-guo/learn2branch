#!/usr/bin/env python
from __future__ import print_function

import argparse
import hashlib
import json
import pathlib
import re


SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
SEEDS = range(5)

GCNN_MODELS = {
    "setcover": ("baseline", "mean_convolution", "no_prenorm"),
    "cauctions": ("baseline",),
    "facilities": ("baseline",),
    "indset": ("baseline",),
}
COMPETITOR_MODELS = {
    "extratrees_gcnn_agg": {
        "model_file": "model.pkl",
        "completion": r"Validation RMSE:\s*([-+0-9.eE]+)",
        "metric": "validation_rmse",
    },
    "lambdamart_khalil": {
        "model_file": "model.pkl",
        "completion": r"Validation log-NDCG:\s*([-+0-9.eE]+)",
        "metric": "validation_log_ndcg",
    },
    "svmrank_khalil": {
        "model_file": "model.txt",
        "completion": r"Best model with C=([-+0-9.eE]+), validation loss:\s*([-+0-9.eE]+)",
        "metric": "validation_loss",
    },
}


def update_content_hash(digest, path):
    relative = path.relative_to(ROOT).as_posix().encode("utf-8")
    digest.update(relative)
    digest.update(b"\0")
    with path.open("rb") as stream:
        while True:
            chunk = stream.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    digest.update(b"\0")


def require_files(directory, names):
    missing = [name for name in names if not (directory / name).is_file()]
    if missing:
        raise RuntimeError("{} is missing required files: {}".format(directory, missing))
    empty = [name for name in names if (directory / name).stat().st_size == 0]
    if empty:
        raise RuntimeError("{} contains empty required files: {}".format(directory, empty))


def read_log(directory):
    with (directory / "log.txt").open(errors="replace") as stream:
        return stream.read()


def summarize_files(directory):
    files = sorted(path for path in directory.iterdir() if path.is_file())
    digest = hashlib.sha256()
    for path in files:
        update_content_hash(digest, path)
    return {
        "files": len(files),
        "bytes": sum(path.stat().st_size for path in files),
        "path_and_content_sha256": digest.hexdigest(),
        "artifacts": [path.name for path in files],
    }


def audit_gcnn(problem, model, seed):
    directory = ROOT / "trained_models" / problem / model / str(seed)
    require_files(directory, ("best_params.pkl", "log.txt"))
    log = read_log(directory)
    matches = re.findall(r"BEST VALID LOSS:\s*([-+0-9.eE]+)([^\n]*)", log)
    if not matches:
        raise RuntimeError("GCNN completion line missing from {}".format(directory / "log.txt"))
    loss, accuracy_text = matches[-1]
    accuracies = {
        "acc@{}".format(k): float(value)
        for k, value in re.findall(r"acc@(\d+):\s*([-+0-9.eE]+)", accuracy_text)
    }
    result = {
        "family": "gcnn",
        "problem": problem,
        "model": model,
        "seed": seed,
        "directory": directory.relative_to(ROOT).as_posix(),
        "best_validation_loss": float(loss),
        "validation_accuracy": accuracies,
        "epochs_started": len(re.findall(r"EPOCH\s+\d+\.\.\.", log)),
    }
    result.update(summarize_files(directory))
    return result


def audit_competitor(problem, model, seed):
    spec = COMPETITOR_MODELS[model]
    directory = ROOT / "trained_models" / problem / model / str(seed)
    require_files(directory, ("feat_specs.pkl", "normalization.pkl", spec["model_file"], "log.txt"))
    log = read_log(directory)
    matches = re.findall(spec["completion"], log)
    if not matches:
        raise RuntimeError("competitor completion line missing from {}".format(directory / "log.txt"))
    final = matches[-1]
    result = {
        "family": "competitors",
        "problem": problem,
        "model": model,
        "seed": seed,
        "directory": directory.relative_to(ROOT).as_posix(),
    }
    if model == "svmrank_khalil":
        result["best_c"] = float(final[0])
        result[spec["metric"]] = float(final[1])
    else:
        result[spec["metric"]] = float(final)
    result.update(summarize_files(directory))
    return result


def audit_family(family):
    runs = []
    if family == "gcnn":
        for problem in ("setcover", "cauctions", "facilities", "indset"):
            for model in GCNN_MODELS[problem]:
                for seed in SEEDS:
                    runs.append(audit_gcnn(problem, model, seed))
    else:
        for problem in ("setcover", "cauctions", "facilities", "indset"):
            for model in sorted(COMPETITOR_MODELS):
                for seed in SEEDS:
                    runs.append(audit_competitor(problem, model, seed))
    return {
        "runs": runs,
        "total_runs": len(runs),
        "total_files": sum(run["files"] for run in runs),
        "total_bytes": sum(run["bytes"] for run in runs),
    }


def load_manifest(path):
    if path.is_file():
        with path.open() as stream:
            return json.load(stream)
    return {
        "paper": "arXiv:1906.01629",
        "upstream_commit": "57e82603fba39c34ff81baf9bd0b5768f5a1a860",
        "training_scripts": ["03_train_gcnn.py", "03_train_competitor.py"],
        "seeds": list(SEEDS),
        "families": {},
    }


def main():
    parser = argparse.ArgumentParser(description="Audit full official Learn2Branch training artifacts")
    parser.add_argument("--family", choices=("gcnn", "competitors", "all"), default="all")
    parser.add_argument(
        "--output",
        default=str(ROOT / "reproduction" / "results" / "full_training_manifest.json"),
    )
    args = parser.parse_args()

    output = pathlib.Path(args.output).resolve()
    manifest = load_manifest(output)
    selected = ("gcnn", "competitors") if args.family == "all" else (args.family,)
    audited = {}
    for family in selected:
        print("Auditing {}...".format(family), flush=True)
        audited[family] = audit_family(family)
    manifest["families"].update(audited)
    manifest["completed_families"] = sorted(manifest["families"])
    manifest["total_runs"] = sum(item["total_runs"] for item in manifest["families"].values())
    manifest["total_files"] = sum(item["total_files"] for item in manifest["families"].values())
    manifest["total_bytes"] = sum(item["total_bytes"] for item in manifest["families"].values())

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as stream:
        json.dump(manifest, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    print("Manifest written to {}".format(output))


if __name__ == "__main__":
    main()
