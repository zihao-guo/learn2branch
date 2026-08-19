#!/usr/bin/env python
from __future__ import print_function

import argparse
import hashlib
import importlib.util
import json
import math
import os
import pathlib
import subprocess
import sys
import time

import numpy as np
import pyscipopt as scip
import tensorflow as tf
import tensorflow.contrib.eager as tfe


SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import utilities


def load_source(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def get_files(directory, pattern):
    return sorted(pathlib.Path(directory).glob(pattern))


def require_empty_or_complete(directory, pattern, expected):
    files = get_files(directory, pattern)
    if len(files) == expected:
        return files
    if files:
        raise RuntimeError(
            "{} contains {} files, expected {}. Move the partial smoke artifact "
            "directory aside before retrying.".format(directory, len(files), expected)
        )
    return []


def generate_instances(generator, artifact_dir, seed, counts):
    rng = np.random.RandomState(seed)
    split_order = ("train", "valid", "test", "transfer")
    existing = {}
    for split in split_order:
        split_dir = artifact_dir / "instances" / split
        split_dir.mkdir(parents=True, exist_ok=True)
        existing[split] = get_files(split_dir, "instance_*.lp")

    if any(existing.values()):
        if all(len(existing[split]) == counts[split] for split in split_order):
            return {split: [str(path) for path in existing[split]] for split in split_order}
        raise RuntimeError(
            "The instance artifact is partial. Move it aside before retrying so "
            "the single official RNG stream is not shifted between data splits."
        )

    result = {}
    for split in split_order:
        split_dir = artifact_dir / "instances" / split
        for index in range(counts[split]):
            filename = split_dir / "instance_{}.lp".format(index + 1)
            generator.generate_setcover(
                nrows=500,
                ncols=1000,
                density=0.05,
                filename=str(filename),
                rng=rng,
                max_coef=100,
            )
        files = get_files(split_dir, "instance_*.lp")
        result[split] = [str(path) for path in files]
    return result


def generate_samples(dataset, artifact_dir, instances, counts, jobs, query_prob, time_limit):
    result = {}
    split_seeds = {"train": 0, "valid": 1, "test": 2}
    for split in ("train", "valid", "test"):
        split_dir = artifact_dir / "samples" / split
        files = require_empty_or_complete(split_dir, "sample_*.pkl", counts[split])
        if not files:
            dataset.collect_samples(
                instances=instances[split],
                out_dir=str(split_dir),
                rng=np.random.RandomState(split_seeds[split]),
                n_samples=counts[split],
                n_jobs=jobs,
                exploration_policy="pscost",
                query_expert_prob=query_prob,
                time_limit=time_limit,
            )
            files = get_files(split_dir, "sample_*.pkl")
        if len(files) != counts[split]:
            raise RuntimeError("sample collection did not reach the requested count")
        result[split] = [str(path) for path in files]
    return result


def make_gcnn_dataset(training, files, batch_size):
    data = tf.data.Dataset.from_tensor_slices(files)
    data = data.batch(batch_size)
    data = data.map(training.load_batch_tf)
    return data.prefetch(1)


def train_and_test(training, model_module, artifact_dir, samples, seed):
    rng = np.random.RandomState(seed)
    tf.set_random_seed(rng.randint(np.iinfo(int).max))

    model = model_module.GCNPolicy()
    pretrain_files = samples["train"][::10] or samples["train"][:1]
    pretrain_data = make_gcnn_dataset(training, pretrain_files, min(4, len(pretrain_files)))
    pretrain_layers = training.pretrain(model=model, dataloader=pretrain_data)
    model.call = tfe.defun(model.call, input_signature=model.input_signature)

    train_data = make_gcnn_dataset(training, samples["train"], min(4, len(samples["train"])))
    valid_data = make_gcnn_dataset(training, samples["valid"], min(4, len(samples["valid"])))
    test_data = make_gcnn_dataset(training, samples["test"], min(4, len(samples["test"])))

    optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
    train_loss, train_kacc = training.process(model, train_data, [1], optimizer)
    valid_loss, valid_kacc = training.process(model, valid_data, [1], None)
    test_loss, test_kacc = training.process(model, test_data, [1], None)

    model_dir = artifact_dir / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    parameters = model_dir / "best_params.pkl"
    model.save_state(str(parameters))

    return {
        "pretrained_layers": int(pretrain_layers),
        "train_loss": float(train_loss),
        "train_acc_at_1": float(train_kacc[0]),
        "valid_loss": float(valid_loss),
        "valid_acc_at_1": float(valid_kacc[0]),
        "test_loss": float(test_loss),
        "test_acc_at_1": float(test_kacc[0]),
    }, str(parameters)


def finite_float(value):
    value = float(value)
    return value if math.isfinite(value) else None


def solve_instance(evaluation, policy, instance, seed, time_limit):
    tf.set_random_seed(seed)
    model = scip.Model()
    model.setIntParam("display/verblevel", 0)
    model.readProblem(instance)
    utilities.init_scip_params(model, seed=seed)
    model.setIntParam("timing/clocktype", 1)
    model.setRealParam("limits/time", time_limit)

    brancher = evaluation.PolicyBranching(policy)
    model.includeBranchrule(
        branchrule=brancher,
        name="{}:{}".format(policy["type"], policy["name"]),
        desc="Learn2Branch smoke policy",
        priority=666666,
        maxdepth=-1,
        maxbounddist=1,
    )

    wall_start = time.perf_counter()
    model.optimize()
    walltime = time.perf_counter() - wall_start
    result = {
        "policy": "{}:{}".format(policy["type"], policy["name"]),
        "status": str(model.getStatus()),
        "nodes": int(model.getNNodes()),
        "fair_nodes": int(model.getNNodes() + 2 * (brancher.ndomchgs + brancher.ncutoffs)),
        "lps": int(model.getNLPs()),
        "solving_time": float(model.getSolvingTime()),
        "walltime": float(walltime),
        "gap": finite_float(model.getGap()),
        "domain_changes": int(brancher.ndomchgs),
        "cutoffs": int(brancher.ncutoffs),
    }
    model.freeProb()
    return result


def online_evaluation(evaluation, model_module, parameters, instance, seed, time_limit):
    policies = [
        {"type": "internal", "name": "relpscost"},
        {
            "type": "gcnn",
            "name": "baseline",
            "model": model_module.GCNPolicy(),
            "parameters": parameters,
        },
    ]
    return [solve_instance(evaluation, policy, instance, seed, time_limit) for policy in policies]


def main():
    parser = argparse.ArgumentParser(description="End-to-end Learn2Branch smoke reproduction")
    parser.add_argument("--artifact-dir", default=str(ROOT / "reproduction_artifacts" / "smoke_setcover"))
    parser.add_argument("--summary", default=str(ROOT / "reproduction" / "results" / "smoke_setcover.json"))
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--query-expert-prob", type=float, default=0.05)
    parser.add_argument("--sampling-time-limit", type=float, default=300.0)
    parser.add_argument("--solve-time-limit", type=float, default=120.0)
    parser.add_argument("--skip-online", action="store_true")
    args = parser.parse_args()

    if os.environ.get("CONDA_DEFAULT_ENV") != "learn2branch":
        raise RuntimeError("activate the learn2branch Conda environment first")
    if not os.environ.get("SCIPOPTDIR"):
        raise RuntimeError("source reproduction/activate.sh before running this script")

    os.chdir(str(ROOT))
    artifact_dir = pathlib.Path(args.artifact_dir).resolve()
    artifact_dir.mkdir(parents=True, exist_ok=True)
    reused_instances = bool(list((artifact_dir / "instances").glob("*/*.lp")))
    reused_samples = bool(list((artifact_dir / "samples").glob("*/*.pkl")))
    stage_times = {}
    pipeline_start = time.perf_counter()

    generator = load_source("official_generate_instances", ROOT / "01_generate_instances.py")
    dataset = load_source("official_generate_dataset", ROOT / "02_generate_dataset.py")
    training = load_source("official_train_gcnn", ROOT / "03_train_gcnn.py")
    evaluation = load_source("official_evaluate", ROOT / "05_evaluate.py")
    model_module = load_source("official_baseline_model", ROOT / "models" / "baseline" / "model.py")

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    tf.enable_eager_execution(config)

    instance_counts = {"train": 4, "valid": 2, "test": 2, "transfer": 1}
    sample_counts = {"train": 8, "valid": 4, "test": 4}

    stage_start = time.perf_counter()
    instances = generate_instances(generator, artifact_dir, seed=0, counts=instance_counts)
    stage_times["instance_generation"] = time.perf_counter() - stage_start

    stage_start = time.perf_counter()
    samples = generate_samples(
        dataset,
        artifact_dir,
        instances,
        sample_counts,
        args.jobs,
        args.query_expert_prob,
        args.sampling_time_limit,
    )
    stage_times["sample_generation"] = time.perf_counter() - stage_start

    stage_start = time.perf_counter()
    offline_metrics, parameters = train_and_test(training, model_module, artifact_dir, samples, seed=0)
    stage_times["train_and_offline_test"] = time.perf_counter() - stage_start

    online_results = []
    if not args.skip_online:
        stage_start = time.perf_counter()
        online_results = online_evaluation(
            evaluation,
            model_module,
            parameters,
            instances["transfer"][0],
            seed=0,
            time_limit=args.solve_time_limit,
        )
        stage_times["online_solve"] = time.perf_counter() - stage_start

    with pathlib.Path(__file__).open("rb") as stream:
        smoke_script_sha256 = hashlib.sha256(stream.read()).hexdigest()
    repository_head = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
    working_tree_dirty = bool(subprocess.check_output(["git", "status", "--porcelain"]).strip())

    summary = {
        "paper": "arXiv:1906.01629",
        "upstream_commit": "57e82603fba39c34ff81baf9bd0b5768f5a1a860",
        "repository_head_at_run": repository_head,
        "working_tree_dirty_at_run": working_tree_dirty,
        "smoke_script_sha256": smoke_script_sha256,
        "problem": "setcover",
        "instance_parameters": {"nrows": 500, "ncols": 1000, "density": 0.05, "max_coef": 100},
        "instance_counts": instance_counts,
        "sample_counts": sample_counts,
        "reused_artifacts": {
            "instances": reused_instances,
            "samples": reused_samples,
        },
        "sampling": {
            "exploration_policy": "pscost",
            "expert": "vanillafullstrong",
            "query_expert_probability": args.query_expert_prob,
            "time_limit": args.sampling_time_limit,
            "jobs": args.jobs,
        },
        "training": {
            "model": "baseline",
            "seed": 0,
            "epochs": 1,
            "learning_rate": 0.001,
            "device": "CPU",
        },
        "offline_metrics": offline_metrics,
        "online_results": online_results,
        "stage_seconds": {name: float(value) for name, value in stage_times.items()},
        "total_seconds": float(time.perf_counter() - pipeline_start),
    }

    summary_path = pathlib.Path(args.summary).resolve()
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w") as stream:
        json.dump(summary, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    print("Summary written to {}".format(summary_path))


if __name__ == "__main__":
    main()
