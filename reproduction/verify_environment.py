#!/usr/bin/env python
from __future__ import print_function

import json
import os
import subprocess
import sys

import numpy
import pyltr
import pyscipopt
import scipy
import sklearn
import svmrank
import tensorflow as tf
from pyscipopt import Model


EXPECTED = {
    "python": "3.6.13",
    "numpy": "1.16.6",
    "scipy": "1.2.1",
    "scikit-learn": "0.20.2",
    "tensorflow": "1.12.0",
    "pyscipopt": "2.1.5",
}


def version(module):
    return getattr(module, "__version__", "unknown")


actual = {
    "python": ".".join(map(str, sys.version_info[:3])),
    "numpy": version(numpy),
    "scipy": version(scipy),
    "scikit-learn": version(sklearn),
    "tensorflow": version(tf),
    "pyscipopt": version(pyscipopt),
    "pyltr": version(pyltr),
    "svmrank": version(svmrank),
}
for package, expected in EXPECTED.items():
    assert actual[package] == expected, (package, actual[package], expected)

tf.enable_eager_execution()
tf_result = (tf.constant([1.0, 2.0]) + 1.0).numpy().tolist()
assert tf_result == [2.0, 3.0]

model = Model("learn2branch-environment-check")
required_methods = [
    "getState",
    "getKhalilState",
    "getVanillafullstrongData",
    "executeBranchRule",
]
missing = [method for method in required_methods if not hasattr(model, method)]
assert not missing, "modified PySCIPOpt methods missing: {}".format(missing)
x = model.addVar("x", vtype="B")
model.setObjective(x, "maximize")
model.hideOutput()
model.optimize()
assert model.getStatus() == "optimal"
assert round(model.getObjVal()) == 1

scip_bin = os.path.join(os.environ["SCIPOPTDIR"], "bin", "scip")
scip_version = subprocess.check_output([scip_bin, "--version"]).decode("utf-8").splitlines()[0]
assert "SCIP version 6.0.1" in scip_version
assert "SoPlex 4.0.1" in scip_version

report = {
    "packages": actual,
    "scip": scip_version,
    "tensorflow_eager_result": tf_result,
    "custom_pyscipopt_methods": required_methods,
    "tiny_scip_solve": {"status": str(model.getStatus()), "objective": model.getObjVal()},
}
print(json.dumps(report, indent=2, sort_keys=True))
