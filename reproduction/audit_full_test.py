#!/usr/bin/env python
from __future__ import print_function

import argparse
import csv
import hashlib
import json
import math
import pathlib


SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
METRICS = ("acc@1", "acc@3", "acc@5", "acc@10")
POLICIES = {
    "setcover": (
        "gcnn:baseline",
        "gcnn:mean_convolution",
        "gcnn:no_prenorm",
        "ml-competitor:extratrees_gcnn_agg",
        "ml-competitor:lambdamart_khalil",
        "ml-competitor:svmrank_khalil",
    ),
    "cauctions": (
        "gcnn:baseline",
        "ml-competitor:extratrees_gcnn_agg",
        "ml-competitor:lambdamart_khalil",
        "ml-competitor:svmrank_khalil",
    ),
    "facilities": (
        "gcnn:baseline",
        "ml-competitor:extratrees_gcnn_agg",
        "ml-competitor:lambdamart_khalil",
        "ml-competitor:svmrank_khalil",
    ),
    "indset": (
        "gcnn:baseline",
        "ml-competitor:extratrees_gcnn_agg",
        "ml-competitor:lambdamart_khalil",
        "ml-competitor:svmrank_khalil",
    ),
}


def mean_std(values):
    mean = sum(values) / len(values)
    std = math.sqrt(sum((value - mean) ** 2 for value in values) / len(values))
    return mean, std


def hash_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            chunk = stream.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def audit_problem(problem, input_root, references):
    path = input_root / "{}.csv".format(problem)
    if not path.is_file():
        raise RuntimeError("missing official test result: {}".format(path))
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))

    expected_rows = len(POLICIES[problem]) * 5
    if len(rows) != expected_rows:
        raise RuntimeError("{} contains {} rows, expected {}".format(path, len(rows), expected_rows))
    by_key = {}
    for row in rows:
        key = (row["policy"], int(row["seed"]))
        if key in by_key:
            raise RuntimeError("duplicate policy/seed in {}: {}".format(path, key))
        by_key[key] = row

    expected_keys = set(
        (policy, seed)
        for policy in POLICIES[problem]
        for seed in range(5)
    )
    if set(by_key) != expected_keys:
        raise RuntimeError("unexpected policy/seed coverage in {}".format(path))

    policies = {}
    for policy in POLICIES[problem]:
        metrics = {}
        for metric in METRICS:
            values = [100.0 * float(by_key[(policy, seed)][metric]) for seed in range(5)]
            mean, std = mean_std(values)
            metrics[metric] = {
                "values": values,
                "mean": mean,
                "population_std": std,
            }
            reference = references.get(policy, {}).get(metric)
            if reference is not None:
                metrics[metric]["paper_mean"] = reference[0]
                metrics[metric]["paper_std"] = reference[1]
                metrics[metric]["mean_delta"] = mean - reference[0]
        policies[policy] = metrics

    return {
        "result_file": path.relative_to(ROOT).as_posix(),
        "rows": len(rows),
        "bytes": path.stat().st_size,
        "sha256": hash_file(path),
        "policies": policies,
    }


def main():
    parser = argparse.ArgumentParser(description="Audit official offline test accuracy and compare with the paper")
    parser.add_argument(
        "--input-root",
        default=str(ROOT / "reproduction_artifacts" / "full_test" / "results"),
    )
    parser.add_argument(
        "--reference",
        default=str(ROOT / "reproduction" / "results" / "paper_test_reference.json"),
    )
    parser.add_argument(
        "--output",
        default=str(ROOT / "reproduction" / "results" / "full_test_manifest.json"),
    )
    args = parser.parse_args()

    input_root = pathlib.Path(args.input_root).resolve()
    reference_path = pathlib.Path(args.reference).resolve()
    with reference_path.open() as stream:
        reference = json.load(stream)

    problems = {}
    for problem in ("setcover", "cauctions", "facilities", "indset"):
        print("Auditing {}...".format(problem), flush=True)
        problems[problem] = audit_problem(
            problem,
            input_root,
            reference["problems"][problem],
        )

    manifest = {
        "paper": reference["paper"],
        "reference": reference_path.relative_to(ROOT).as_posix(),
        "results": problems,
        "total_rows": sum(item["rows"] for item in problems.values()),
        "total_bytes": sum(item["bytes"] for item in problems.values()),
    }
    output = pathlib.Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as stream:
        json.dump(manifest, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    print("Manifest written to {}".format(output))


if __name__ == "__main__":
    main()
