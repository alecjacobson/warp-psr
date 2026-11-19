"""Schedule an isolated ABBA comparison of horse profiling."""
import importlib.util
import json
import os
from pathlib import Path
import sys

root = Path.cwd()
scratch = root / ".build/func-composition"
spec = importlib.util.spec_from_file_location("isolation", root/"docs/results/depth9-audit-scripts/isolation.py")
isolation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(isolation)
environment = {**os.environ, "PYTHONPATH": "/home/horde/projects/warp-field-composition:"+str(root)+":"+str(root/"src"),
               "PYTHONDONTWRITEBYTECODE":"1", "OPENBLAS_NUM_THREADS":"1", "OMP_NUM_THREADS":"4"}
for index, variant in enumerate(("original", "fixed", "fixed", "original")):
    name = f"profile-{index}-{variant}"
    command = [sys.executable,str(scratch/"profile_horse.py"),"--variant",variant,"--output",str(scratch/f"{name}.json")]
    observations = isolation.measured_process(command,scratch/f"{name}.log",root=root,env=environment)
    (scratch/f"{name}.isolation.json").write_text(json.dumps(observations,indent=2)+"\n")
    if observations["foreign_work_observed"]:
        raise RuntimeError(f"Foreign work observed in {name}; measurements rejected")
    print("DONE",name,flush=True)
