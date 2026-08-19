#!/usr/bin/env python
from __future__ import print_function

import argparse
import hashlib
import json
import pathlib


SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent

SAMPLE_DIRECTORIES = {
    "setcover": "500r_1000c_0.05d",
    "cauctions": "100_500",
    "facilities": "100_100_5",
    "indset": "500_4",
}
EXPECTED_SPLITS = {
    "train": 100000,
    "valid": 20000,
    "test": 20000,
}


def expected_files(directory, count):
    files = [directory / "sample_{}.pkl".format(index) for index in range(1, count + 1)]
    missing = [str(path) for path in files if not path.is_file()]
    if missing:
        raise RuntimeError("missing expected files, first entries: {}".format(missing[:5]))
    actual = sorted(directory.glob("sample_*.pkl"))
    if len(actual) != count:
        raise RuntimeError("{} contains {} sample files, expected {}".format(directory, len(actual), count))
    if (directory / "tmp").exists():
        raise RuntimeError("temporary sampling directory still exists: {}".format(directory / "tmp"))
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
    base = ROOT / "data" / "samples" / problem / SAMPLE_DIRECTORIES[problem]
    digest = hashlib.sha256()
    splits = {}
    total_files = 0
    total_bytes = 0

    for split, count in EXPECTED_SPLITS.items():
        files = expected_files(base / split, count)
        size_bytes = sum(path.stat().st_size for path in files)
        splits[split] = {"files": count, "bytes": size_bytes}
        total_files += count
        total_bytes += size_bytes
        if include_hash:
            for path in files:
                update_content_hash(digest, path)

    result = {
        "directory": base.relative_to(ROOT).as_posix(),
        "splits": splits,
        "files": total_files,
        "bytes": total_bytes,
    }
    if include_hash:
        result["path_and_content_sha256"] = digest.hexdigest()
    return result


def load_manifest(path):
    if not path.is_file():
        return {
            "paper": "arXiv:1906.01629",
            "upstream_commit": "57e82603fba39c34ff81baf9bd0b5768f5a1a860",
            "generator": "02_generate_dataset.py",
            "seed": 0,
            "sampling": {
                "exploration_policy": "pscost",
                "expert": "vanillafullstrong",
                "expert_query_probability": 0.05,
            },
            "problems": {},
        }
    with path.open() as stream:
        return json.load(stream)


def main():
    parser = argparse.ArgumentParser(description="Audit completed official Learn2Branch sample datasets")
    parser.add_argument(
        "--problem",
        action="append",
        choices=sorted(SAMPLE_DIRECTORIES),
        help="completed problem to audit; repeat for multiple problems (default: all)",
    )
    parser.add_argument(
        "--output",
        default=str(ROOT / "reproduction" / "results" / "full_samples_manifest.json"),
    )
    parser.add_argument("--skip-content-hash", action="store_true")
    args = parser.parse_args()

    output = pathlib.Path(args.output).resolve()
    manifest = load_manifest(output)
    selected = args.problem or sorted(SAMPLE_DIRECTORIES)
    for problem in selected:
        print("Auditing {}...".format(problem), flush=True)
        manifest["problems"][problem] = audit_problem(
            problem,
            include_hash=not args.skip_content_hash,
        )

    manifest["completed_problems"] = sorted(manifest["problems"])
    manifest["total_files"] = sum(item["files"] for item in manifest["problems"].values())
    manifest["total_bytes"] = sum(item["bytes"] for item in manifest["problems"].values())

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as stream:
        json.dump(manifest, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    print("Manifest written to {}".format(output))


if __name__ == "__main__":
    main()
