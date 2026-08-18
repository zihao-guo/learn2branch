#!/usr/bin/env python
from __future__ import print_function

import argparse
import csv
import hashlib
import json
import math
import pathlib
from collections import Counter, defaultdict


SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
DIFFICULTIES = ("small", "medium", "big")
SEEDS = tuple(range(5))
MAIN_POLICIES = (
    "internal:vanillafullstrong",
    "internal:relpscost",
    "ml-competitor:extratrees_gcnn_agg",
    "ml-competitor:svmrank_khalil",
    "ml-competitor:lambdamart_khalil",
    "gcnn:baseline",
)
MAIN_NODE_BASELINES = tuple(
    policy for policy in MAIN_POLICIES if policy != "internal:vanillafullstrong"
)
ABLATION_POLICIES = (
    "gcnn:mean_convolution",
    "gcnn:no_prenorm",
    "gcnn:baseline",
)
POLICIES = {
    "setcover": MAIN_POLICIES[:-1] + ABLATION_POLICIES,
    "cauctions": MAIN_POLICIES,
    "facilities": MAIN_POLICIES,
    "indset": MAIN_POLICIES,
}
INSTANCE_DIRS = {
    "setcover": (
        "setcover/transfer_500r_1000c_0.05d",
        "setcover/transfer_1000r_1000c_0.05d",
        "setcover/transfer_2000r_1000c_0.05d",
    ),
    "cauctions": (
        "cauctions/transfer_100_500",
        "cauctions/transfer_200_1000",
        "cauctions/transfer_300_1500",
    ),
    "facilities": (
        "facilities/transfer_100_100_5",
        "facilities/transfer_200_100_5",
        "facilities/transfer_400_100_5",
    ),
    "indset": (
        "indset/transfer_500_4",
        "indset/transfer_1000_4",
        "indset/transfer_1500_4",
    ),
}
REQUIRED_COLUMNS = (
    "policy", "seed", "type", "instance", "nnodes", "nlps", "stime",
    "gap", "status", "ndomchgs", "ncutoffs", "walltime", "proctime",
)
SOLVED_STATUSES = frozenset(("optimal", "infeasible"))


def expected_tasks():
    tasks = []
    for problem in ("setcover", "cauctions", "facilities", "indset"):
        for policy in POLICIES[problem]:
            policy_type, policy_name = policy.split(":", 1)
            for seed in SEEDS:
                for instance_index in range(60):
                    task = "{}_{}_{}_{}_{}".format(
                        problem, instance_index, policy_type, policy_name, seed
                    )
                    tasks.append((task, problem, instance_index, policy, seed))
    if len(tasks) != 7800:
        raise AssertionError("expected 7800 tasks, got {}".format(len(tasks)))
    return tasks


def expected_instance(problem, instance_index):
    difficulty_index = instance_index // 20
    difficulty = DIFFICULTIES[difficulty_index]
    instance_number = instance_index % 20 + 1
    path = "data/instances/{}/instance_{}.lp".format(
        INSTANCE_DIRS[problem][difficulty_index], instance_number
    )
    return difficulty, path


def parse_nonnegative(row, field, integer=False):
    try:
        value = int(row[field]) if integer else float(row[field])
    except (KeyError, ValueError):
        raise RuntimeError("invalid {} value: {!r}".format(field, row.get(field)))
    if value < 0 or not math.isfinite(float(value)):
        raise RuntimeError("invalid {} value: {!r}".format(field, row.get(field)))
    return value


def read_result(path, problem, instance_index, policy, seed):
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != REQUIRED_COLUMNS:
            raise RuntimeError("unexpected CSV header in {}".format(path))
        rows = list(reader)
    if len(rows) != 1:
        raise RuntimeError("{} contains {} result rows, expected 1".format(path, len(rows)))

    row = rows[0]
    difficulty, instance = expected_instance(problem, instance_index)
    expected = {
        "policy": policy,
        "seed": str(seed),
        "type": difficulty,
        "instance": instance,
    }
    for field, value in expected.items():
        if row[field] != value:
            raise RuntimeError(
                "{} has {}={!r}, expected {!r}".format(path, field, row[field], value)
            )

    parsed = {
        "problem": problem,
        "instance_index": instance_index,
        "policy": policy,
        "seed": seed,
        "difficulty": difficulty,
        "instance": instance,
        "nnodes": parse_nonnegative(row, "nnodes", integer=True),
        "nlps": parse_nonnegative(row, "nlps", integer=True),
        "stime": parse_nonnegative(row, "stime"),
        "gap": parse_nonnegative(row, "gap"),
        "status": row["status"],
        "ndomchgs": parse_nonnegative(row, "ndomchgs", integer=True),
        "ncutoffs": parse_nonnegative(row, "ncutoffs", integer=True),
        "walltime": parse_nonnegative(row, "walltime"),
        "proctime": parse_nonnegative(row, "proctime"),
    }
    parsed["fair_nodes"] = (
        parsed["nnodes"] + 2 * (parsed["ndomchgs"] + parsed["ncutoffs"])
    )
    parsed["solved"] = parsed["status"] in SOLVED_STATUSES
    return parsed


def population_std(values):
    mean = sum(values) / len(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / len(values))


def shifted_geometric_mean(values, shift=1.0):
    return math.exp(sum(math.log(value + shift) for value in values) / len(values)) - shift


def per_instance_cv_percent(rows, field):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["instance"]].append(float(row[field]))
    cvs = []
    for values in grouped.values():
        mean = sum(values) / len(values)
        cvs.append(0.0 if mean == 0 else 100.0 * population_std(values) / mean)
    return sum(cvs) / len(cvs) if cvs else None


def mean_or_none(values):
    return sum(values) / len(values) if values else None


def metric_delta(actual, reference):
    if actual is None or reference is None:
        return None
    return actual - reference


def aggregate_table(rows, policies, node_baselines, references):
    by_attempt = defaultdict(dict)
    by_policy = defaultdict(list)
    for row in rows:
        key = (row["instance"], row["seed"])
        by_attempt[key][row["policy"]] = row
        by_policy[row["policy"]].append(row)

    expected_attempts = 20 * len(SEEDS)
    for policy in policies:
        if len(by_policy[policy]) != expected_attempts:
            raise RuntimeError(
                "{} has {} rows, expected {}".format(
                    policy, len(by_policy[policy]), expected_attempts
                )
            )

    common_solved = set()
    for key, attempt in by_attempt.items():
        if all(attempt[policy]["solved"] for policy in node_baselines):
            common_solved.add(key)

    wins = Counter()
    for attempt in by_attempt.values():
        solved = [attempt[policy] for policy in policies if attempt[policy]["solved"]]
        if not solved:
            continue
        best_time = min(row["stime"] for row in solved)
        for row in solved:
            if row["stime"] == best_time:
                wins[row["policy"]] += 1

    metrics = {}
    for policy in policies:
        policy_rows = by_policy[policy]
        solved_rows = [row for row in policy_rows if row["solved"]]
        node_rows = [
            row for row in policy_rows
            if (row["instance"], row["seed"]) in common_solved and row["solved"]
        ]
        own_nodes = [row["fair_nodes"] for row in solved_rows]
        nodes = [row["fair_nodes"] for row in node_rows]
        result = {
            "time_shifted_geomean": shifted_geometric_mean(
                [row["stime"] for row in policy_rows]
            ),
            "time_seed_cv_percent": per_instance_cv_percent(policy_rows, "stime"),
            "wins": wins[policy],
            "solved": len(solved_rows),
            "nodes_common_solved_attempts": len(node_rows),
            "nodes_arithmetic_mean": mean_or_none(nodes),
            "nodes_seed_cv_percent": per_instance_cv_percent(node_rows, "fair_nodes"),
            "own_solved_nodes_arithmetic_mean": mean_or_none(own_nodes),
        }
        reference = references.get(policy, {})
        if reference:
            result["paper"] = reference
            result["deltas"] = {
                "time_shifted_geomean": metric_delta(
                    result["time_shifted_geomean"], reference.get("time")
                ),
                "time_seed_cv_percent": metric_delta(
                    result["time_seed_cv_percent"], reference.get("time_cv_percent")
                ),
                "wins": metric_delta(result["wins"], reference.get("wins")),
                "solved": metric_delta(result["solved"], reference.get("solved")),
                "nodes_arithmetic_mean": metric_delta(
                    result["nodes_arithmetic_mean"], reference.get("nodes")
                ),
                "nodes_seed_cv_percent": metric_delta(
                    result["nodes_seed_cv_percent"], reference.get("nodes_cv_percent")
                ),
            }
        metrics[policy] = result
    return {
        "common_solved_attempts": len(common_solved),
        "policies": metrics,
    }


def load_records(input_root, allow_incomplete):
    digest = hashlib.sha256()
    records = []
    missing = []
    expected_names = set()
    total_bytes = 0
    for task, problem, instance_index, policy, seed in expected_tasks():
        result_path = input_root / "results" / (task + ".csv")
        marker_path = input_root / "markers" / (task + ".complete")
        expected_names.add(result_path.name)
        if not result_path.is_file() or not marker_path.is_file():
            missing.append(task)
            continue
        record = read_result(result_path, problem, instance_index, policy, seed)
        records.append(record)
        relative = result_path.relative_to(input_root).as_posix().encode("utf-8")
        content = result_path.read_bytes()
        digest.update(relative + b"\0" + content + b"\0")
        total_bytes += len(content)

    extras = sorted(
        path.name for path in (input_root / "results").glob("*.csv")
        if path.name not in expected_names
    )
    if extras:
        raise RuntimeError("unexpected result CSV files: {}".format(", ".join(extras[:10])))
    if missing and not allow_incomplete:
        raise RuntimeError(
            "missing {} of 7800 completed tasks; first missing: {}".format(
                len(missing), ", ".join(missing[:10])
            )
        )
    return records, missing, total_bytes, digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(
        description="Audit and aggregate the paper-scale online SCIP evaluation"
    )
    parser.add_argument(
        "--input-root",
        default=str(ROOT / "reproduction_artifacts" / "full_evaluation" / "3600s"),
    )
    parser.add_argument(
        "--reference",
        default=str(ROOT / "reproduction" / "results" / "paper_evaluation_reference.json"),
    )
    parser.add_argument(
        "--output",
        default=str(ROOT / "reproduction" / "results" / "full_evaluation_manifest.json"),
    )
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="validate completed shards and report coverage without requiring all 7800",
    )
    args = parser.parse_args()

    input_root = pathlib.Path(args.input_root).resolve()
    with pathlib.Path(args.reference).resolve().open() as stream:
        reference = json.load(stream)
    records, missing, total_bytes, sha256 = load_records(
        input_root, args.allow_incomplete
    )

    status_counts = Counter(row["status"] for row in records)
    manifest = {
        "paper": reference["paper"],
        "coverage": {
            "expected_tasks": 7800,
            "completed_tasks": len(records),
            "missing_tasks": len(missing),
            "first_missing_tasks": missing[:20],
        },
        "input_root": input_root.relative_to(ROOT).as_posix()
            if ROOT in input_root.parents else str(input_root),
        "total_bytes": total_bytes,
        "aggregate_sha256": sha256,
        "status_counts": dict(sorted(status_counts.items())),
        "aggregation": {
            "time": "1-shifted geometric mean over all 100 attempts; timeouts are not additionally penalized",
            "variability": "mean across 20 instances of the population seed standard deviation divided by that instance mean",
            "wins": "fastest solved policy per instance/seed attempt, within the policies in that table",
            "nodes": "arithmetic mean of fair nodes on attempts solved by all non-expert policies in that table",
            "fair_nodes": "nnodes + 2 * (ndomchgs + ncutoffs)",
        },
        "tables": {},
    }

    if not missing:
        for problem in ("setcover", "cauctions", "facilities", "indset"):
            manifest["tables"][problem] = {}
            for difficulty in DIFFICULTIES:
                subset = [
                    row for row in records
                    if row["problem"] == problem and row["difficulty"] == difficulty
                    and row["policy"] in MAIN_POLICIES
                ]
                refs = reference["tables"]["main"][problem][difficulty]
                manifest["tables"][problem][difficulty] = aggregate_table(
                    subset, MAIN_POLICIES, MAIN_NODE_BASELINES, refs
                )

        manifest["tables"]["setcover_ablation"] = {}
        for difficulty in DIFFICULTIES:
            subset = [
                row for row in records
                if row["problem"] == "setcover" and row["difficulty"] == difficulty
                and row["policy"] in ABLATION_POLICIES
            ]
            refs = reference["tables"]["setcover_ablation"][difficulty]
            manifest["tables"]["setcover_ablation"][difficulty] = aggregate_table(
                subset, ABLATION_POLICIES, ABLATION_POLICIES, refs
            )

    output = pathlib.Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as stream:
        json.dump(manifest, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    print("Manifest written to {}".format(output))


if __name__ == "__main__":
    main()
