#!/usr/bin/env python
from __future__ import print_function

import argparse
import hashlib
import json
import pathlib


SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent

EXPECTED = {
    "setcover": {
        "train_500r_1000c_0.05d": 10000,
        "valid_500r_1000c_0.05d": 2000,
        "transfer_500r_1000c_0.05d": 100,
        "transfer_1000r_1000c_0.05d": 100,
        "transfer_2000r_1000c_0.05d": 100,
        "test_500r_1000c_0.05d": 2000,
    },
    "cauctions": {
        "train_100_500": 10000,
        "valid_100_500": 2000,
        "transfer_100_500": 100,
        "transfer_200_1000": 100,
        "transfer_300_1500": 100,
        "test_100_500": 2000,
    },
    "facilities": {
        "train_100_100_5": 10000,
        "valid_100_100_5": 2000,
        "transfer_100_100_5": 100,
        "transfer_200_100_5": 100,
        "transfer_400_100_5": 100,
        "test_100_100_5": 2000,
    },
    "indset": {
        "train_500_4": 10000,
        "valid_500_4": 2000,
        "transfer_500_4": 100,
        "transfer_1000_4": 100,
        "transfer_1500_4": 100,
        "test_500_4": 2000,
    },
}


def expected_files(directory, count):
    files = [directory / "instance_{}.lp".format(index) for index in range(1, count + 1)]
    missing = [str(path) for path in files if not path.is_file()]
    if missing:
        raise RuntimeError("missing expected files, first entries: {}".format(missing[:5]))
    actual = sorted(directory.glob("instance_*.lp"))
    if len(actual) != count:
        raise RuntimeError("{} contains {} LP files, expected {}".format(directory, len(actual), count))
    return files


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


def audit_problem(problem, include_hash):
    base = ROOT / "data" / "instances" / problem
    digest = hashlib.sha256()
    directories = {}
    total_files = 0
    total_bytes = 0

    for name, count in EXPECTED[problem].items():
        directory = base / name
        files = expected_files(directory, count)
        size_bytes = sum(path.stat().st_size for path in files)
        directories[name] = {"files": count, "bytes": size_bytes}
        total_files += count
        total_bytes += size_bytes
        if include_hash:
            for path in files:
                update_content_hash(digest, path)

    result = {
        "directories": directories,
        "files": total_files,
        "bytes": total_bytes,
    }
    if include_hash:
        result["path_and_content_sha256"] = digest.hexdigest()
    return result


def main():
    parser = argparse.ArgumentParser(description="Audit the full official Learn2Branch instance dataset")
    parser.add_argument(
        "--output",
        default=str(ROOT / "reproduction" / "results" / "full_instances_manifest.json"),
    )
    parser.add_argument("--skip-content-hash", action="store_true")
    args = parser.parse_args()

    problems = {}
    for problem in ("setcover", "cauctions", "facilities", "indset"):
        print("Auditing {}...".format(problem), flush=True)
        problems[problem] = audit_problem(problem, include_hash=not args.skip_content_hash)

    manifest = {
        "paper": "arXiv:1906.01629",
        "upstream_commit": "57e82603fba39c34ff81baf9bd0b5768f5a1a860",
        "generator": "01_generate_instances.py",
        "seed": 0,
        "problems": problems,
        "total_files": sum(item["files"] for item in problems.values()),
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
