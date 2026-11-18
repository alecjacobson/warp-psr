"""Run the five field-query comparisons with CPU/GPU isolation monitoring."""
import importlib.util
import json
import os
from pathlib import Path
import sys


root = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location(
    "isolation", root / "docs/results/depth9-audit-scripts/isolation.py")
isolation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(isolation)
environment = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1",
               "OMP_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "1"}
scratch = root / ".build/device-query"
scratch.mkdir(parents=True, exist_ok=True)

for dataset, depth, precision in [("bunny", 8, "float"), ("horse", 8, "float"),
                                  ("eagle", 8, "float"), ("eagle", 9, "float"),
                                  ("bunny", 8, "double")]:
    name = f"{dataset}-d{depth}-{precision}"
    print("START", name, flush=True)
    command = [sys.executable, "-m", "tools.benchmark_device_field",
               "--fixture", f".build/gpu-baselines/inputs/{dataset}-full-d8-spsr",
               "--depth", str(depth), "--precision", precision,
               "--output", f"docs/results/device-field/{name}.json"]
    observations = isolation.measured_process(
        command, scratch / f"{name}.log", root=root, env=environment)
    (root / f"docs/results/device-field/{name}.isolation.json").write_text(
        json.dumps(observations, indent=2) + "\n")
    if observations["foreign_work_observed"]:
        raise RuntimeError(f"Foreign work observed during {name}; reject the measurements")
    report = json.loads((root / f"docs/results/device-field/{name}.json").read_text())
    print(name, report["reconstruction_median_ms"], flush=True)
