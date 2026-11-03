"""Run reproducible reference, FEM, preprocessing and independent reconstruction gates."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import shutil
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
    fixtures=[("sphere",["--operators"]),("sphere-exact",["--exact","--grid","16","--operators"]),
              ("sphere-psr",["--weight","0","--grid","16","--operators"]),
              ("sparse",["--count","32","--depth","5","--full-depth","1","--grid","8","--operators"]),
              ("sphere-psr-double",["--weight","0","--precision","double","--grid","16","--operators"]),
              ("sphere-double",["--precision","double","--grid","16","--operators"]),
              ("sphere-exact-double",["--exact","--precision","double","--grid","16","--operators"]),
              ("sparse-double",["--count","32","--depth","5","--full-depth","1","--grid","8","--precision","double","--operators"]),
              ("torus-double",["--shape","torus","--count","8192","--depth","6","--full-depth","3","--grid","24","--precision","double"]),
              ("torus",["--shape","torus","--count","8192","--depth","6","--full-depth","3","--grid","64"])]
    results=[]
    for name,extra in fixtures:
        run("tools/make_fixture.py","--name",name,*extra)
        directory=ROOT/".build/fixtures"/name
        run("-m","tools.check_field",str(directory))
        run("-m","tools.check_reconstruction",str(directory))
        if name in ("sphere","sphere-exact"):
            run("-m","tools.check_cli",str(directory))
        results.append({"fixture":name,
                        "manifest":json.loads((directory/"manifest.json").read_text()),
                        "evaluation":json.loads((directory/"field_report.json").read_text())})
        if (directory/"cli_report.json").exists():
            results[-1]["official_cli"]=json.loads((directory/"cli_report.json").read_text())
        if (directory/"reconstruction_report.json").exists():
            results[-1]["reconstruction"]=json.loads((directory/"reconstruction_report.json").read_text())
    run("-m","tools.check_profiles")
    run("-m","tools.check_geometry_profiles")
    run("-m","tools.check_thresholds")
    run("-m","tools.check_repeatability")
    run("-m","pytest","-q")
    shutil.copyfile(ROOT/".build/repeatability.json",ROOT/"docs/results/repeatability.json")
    shutil.copyfile(ROOT/".build/profile-validation.json",ROOT/"docs/results/profiles.json")
    shutil.copyfile(ROOT/".build/geometry-validation.json",ROOT/"docs/results/geometry-profiles.json")
    shutil.copyfile(ROOT/".build/threshold-validation.json",ROOT/"docs/results/thresholds.json")
    run("-m","tools.render_comparison",str(ROOT/".build/fixtures/torus"))
    source_paths=sorted([*ROOT.glob("src/**/*.py"),*ROOT.glob("reference/*.cpp"),*ROOT.glob("reference/*.inc"),*ROOT.glob("reference/*.py"),*ROOT.glob("tools/*.py"),*ROOT.glob("tests/*.py")])
    report={"scope":"Independent float32 compatibility and end-to-end float64 GPU diagnostic reconstruction",
            "end_to_end_reconstruction_passed":True,
            "source_sha256":{str(path.relative_to(ROOT)):hashlib.sha256(path.read_bytes()).hexdigest() for path in source_paths},
            "fixtures":results,
            "open_gates":["built-in B-spline FEM issue #8 (application basis supports float64)",
                          "complete ghost flags and parent/child metadata", "distribution/density performance scaling"]}
    (ROOT/"docs/results/validation.json").write_text(json.dumps(report,indent=2)+"\n")

if __name__=="__main__":
    main()
