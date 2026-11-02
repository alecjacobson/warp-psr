"""Run reproducible reference, FEM, sample and field gates; NOT a reconstruction claim."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import warp as wp
from tools.make_fixture import ROOT


def run(*args):
    subprocess.run([sys.executable,*args],cwd=ROOT,check=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--skip-build",action="store_true")
    args=p.parse_args()
    wp.init()
    if not wp.is_cuda_available():
        raise RuntimeError("This validation command requires CUDA; pytest alone permits CPU-only runs")
    if not args.skip_build:
        run("tools/build_reference.py")
    fixtures=[("sphere",[]),("sphere-exact",["--exact","--grid","16"]),
              ("sphere-psr-double",["--weight","0","--precision","double","--grid","16"]),
              ("torus",["--shape","torus","--count","8192","--depth","6","--full-depth","3","--grid","64"])]
    results=[]
    for name,extra in fixtures:
        run("tools/make_fixture.py","--name",name,*extra)
        directory=ROOT/".build/fixtures"/name
        run("-m","tools.check_field",str(directory))
        if name in ("sphere","sphere-exact"):
            run("-m","tools.check_cli",str(directory))
        results.append({"fixture":name,
                        "manifest":json.loads((directory/"manifest.json").read_text()),
                        "evaluation":json.loads((directory/"field_report.json").read_text())})
        if (directory/"cli_report.json").exists():
            results[-1]["official_cli"]=json.loads((directory/"cli_report.json").read_text())
    run("-m","pytest","-q")
    run("-m","tools.render_comparison",str(ROOT/".build/fixtures/torus"))
    source_paths=sorted([*ROOT.glob("src/**/*.py"),*ROOT.glob("reference/*.cpp"),*ROOT.glob("tools/*.py")])
    report={"scope":"Compatibility building blocks, not an independent GPU reconstruction",
            "end_to_end_reconstruction_passed":False,
            "source_sha256":{str(path.relative_to(ROOT)):hashlib.sha256(path.read_bytes()).hexdigest() for path in source_paths},
            "fixtures":results,
            "open_gates":["effective interpolation records and per-level operator oracle", "adaptive basis/transfer equivalence",
                          "GPU density/normal splats/support completion", "GPU reference solve schedule", "float64 FEM assembly (#8)",
                          "repeatability and complete query families", "performance optimization after compatibility"]}
    (ROOT/"docs/results/validation.json").write_text(json.dumps(report,indent=2)+"\n")

if __name__=="__main__":
    main()
