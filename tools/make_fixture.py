"""Generate deterministic offline fixtures and prove instrumentation neutrality."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import numpy as np
import warp as wp

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))
from reference.queries import coverage_queries
from tools.datasets import DATASETS, download, load_input


def read_output(path):
    rows = [line.split() for line in Path(path).read_text().splitlines()]
    return {tag: np.array([[float(v) for v in row[1:]] for row in rows if row[0] == tag])
            for tag in {row[0] for row in rows}}


def points_for(shape, count, seed):
    rng = np.random.default_rng(seed)
    if shape in ("plane", "sheets"):
        p = rng.uniform(-1,1,(count,3))
        p[:,2] = 0
        n = np.zeros_like(p)
        n[:,2] = 1
        if shape == "sheets":
            sign = np.where(np.arange(count)%2, 1., -1.)
            p[:,2] = 0.035*sign
            n[:,2] = sign
        return p,n
    if shape in ("sphere", "cluster", "disconnected"):
        n = rng.normal(size=(count, 3))
        if shape == "cluster":
            n[:9*count//10] = rng.normal(scale=0.12,size=(9*count//10,3)) + [0,0,1]
        n /= np.linalg.norm(n, axis=1)[:,None]
        if shape == "disconnected":
            p = 0.25*n
            p[:,0] += np.where(np.arange(count)%2, 0.65, -0.65)
            return p,n
        return n, n.copy()
    if shape != "torus":
        raise ValueError(f"Unknown fixture shape: {shape}")
    u, v = rng.uniform(0, 2*np.pi, size=(2,count))
    # A mildly nonuniform torus intentionally exercises adaptive density.
    p = np.array([(0.7+0.26*np.cos(v))*np.cos(u),
                  (0.7+0.26*np.cos(v))*np.sin(u), 0.26*np.sin(v)]).T
    n = np.array([np.cos(v)*np.cos(u), np.cos(v)*np.sin(u), np.sin(v)]).T
    return p,n


def oracle_parameters(p):
    """All effective settings, shared by the fixture and official CLI checks."""
    return [str(p[key]) for key in ('depth','full_depth','weight')]+[str(int(p['exact'])),str(p['iters']),p['precision']]+[
        str(p[key]) for key in ('base_depth','solve_depth','kernel_depth','scale','samples_per_node','cg_accuracy','base_v_cycles')]


def generate(args):
    if args.base_depth is None:args.base_depth=args.full_depth
    if args.solve_depth is None:args.solve_depth=args.depth
    if args.kernel_depth is None:args.kernel_depth=max(args.depth-2,0)
    out = ROOT / ".build/fixtures" / args.name
    out.mkdir(parents=True, exist_ok=True)
    source_metadata=None
    if args.dataset or args.input:
        source=download(args.dataset) if args.dataset else Path(args.input)
        p,n,source_metadata=load_input(source,max_points=args.max_points,seed=args.seed,dataset=args.dataset)
        args.count=len(p)
        args.shape=args.dataset or 'external'
    else:
        p,n = points_for(args.shape, args.count, args.seed)
    p = p*args.input_scale + np.asarray(args.translation)
    dtype = np.float32 if args.precision == "float" else np.float64
    p,n = p.astype(dtype), n.astype(dtype)
    np.savetxt(out/"input.pts", np.column_stack([p,n]), fmt="%.17g")
    width = np.max(np.max(p,axis=0)-np.min(p,axis=0))*dtype(args.scale)
    center = (np.max(p,axis=0)+np.min(p,axis=0))*dtype(.5)
    rng = np.random.default_rng(args.seed+1)
    # Regular grid for optional mesh extraction; interior/random/sample queries for fields.
    x = np.linspace(.001,.999,args.grid, dtype=dtype)
    grid = np.stack(np.meshgrid(x,x,x,indexing="ij"),axis=-1).reshape(-1,3)
    q = np.concatenate([grid, rng.uniform(.0001,.9999,size=(1024,3))])
    q = q*width+center-width/2
    q = np.concatenate([q,p]).astype(dtype)
    np.savetxt(out/"queries.txt", q, fmt="%.17g")
    command_args = [str(out/"input.pts"), str(out/"queries.txt"), "OUTPUT", *oracle_parameters(vars(args))]
    timings = {}
    environment={**os.environ, "WARP_PSR_ORACLE_DIR": str(out), "WARP_PSR_REFERENCE_PARALLELISM":"none"}
    # The pilot exports support cells used only to choose offline diagnostic
    # queries. Both final drivers independently solve again on the full query set.
    command_args[2] = str(out/'pilot.txt')
    start = time.perf_counter()
    subprocess.run([str(ROOT/'.build/oracle_instrumented'),*command_args],env=environment,check=True)
    timings['query_coverage_pilot'] = time.perf_counter()-start
    pilot = read_output(out/'pilot.txt')
    cube_to_model = pilot['transform'].reshape(4,4).T
    model_to_cube = np.linalg.inv(cube_to_model)
    cube_samples = p @ model_to_cube[:3,:3].T + model_to_cube[:3,3]
    families,coverage = coverage_queries(pilot['node'],read_output(out/'stages.txt')['normal'],cube_samples,args.depth,args.precision,args.seed+2)
    ranges = {'visualization_grid':[0,args.grid**3], 'random_volume':[args.grid**3,args.grid**3+1024],
              'input_samples':[args.grid**3+1024,len(q)]}
    for name,cube in families.items():
        extra = (cube @ cube_to_model[:3,:3].T + cube_to_model[:3,3]).astype(dtype)
        ranges[name] = [len(q),len(q)+len(extra)]
        q = np.concatenate([q,extra])
    np.savetxt(out/'queries.txt',q,fmt='%.17g')
    for exe in ["oracle", "oracle_instrumented"]:
        command_args[2] = str(out/(exe+".txt"))
        start = time.perf_counter()
        if args.operators:
            environment["WARP_PSR_OPERATORS"]=str(out)
        subprocess.run([str(ROOT/".build"/exe), *command_args],env=environment, check=True)
        timings[exe] = time.perf_counter()-start
    baseline = read_output(out/"oracle.txt")
    instrumented = read_output(out/"oracle_instrumented.txt")
    for key in ["q", "iso", "transform"]:
        np.testing.assert_array_equal(baseline[key], instrumented[key])
    wp.init()
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    def checksum(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = {
        "fixture_format": 2, "name": args.name, "pins": json.loads((ROOT/"reference/pins.json").read_text()),
        "application_revision": revision, "application_dirty": bool(subprocess.check_output(["git","status","--porcelain"],cwd=ROOT)),
        "build": json.loads((ROOT/".build/build.json").read_text()),
        "input_sha256": checksum(out/"input.pts"), "queries_sha256": checksum(out/"queries.txt"),
        "input_order": "file order; seeded numpy Generator", "seed": args.seed,
        "input_source":source_metadata,
        "precision": args.precision, "parameters": {**vars(args), "align_dir": 0, "degree": 1, "boundary": "Neumann", "confidence": False}, "reference_parallelism": "ThreadPool::NONE (serial)",
        "reference_thread_settings": {key:int(baseline[key][0,0]) for key in ("parallel_async","threads","schedule","chunk_size")},
        "instrumentation_max_abs_error": 0.0, "timings_seconds": timings,
        "grid": {"resolution": args.grid, "cube_min": .001, "cube_max": .999, "order": "ij, z fastest"},
        "query_families":ranges, "query_coverage":coverage,
        "environment": {"python": platform.python_version(), "numpy": np.__version__, "warp": wp.__version__,
                        "devices": [str(d) for d in wp.get_devices()],
                        "gpu": subprocess.check_output(["nvidia-smi","--query-gpu=name,driver_version,memory.total","--format=csv,noheader"],text=True).strip()},
        "status": "Reference fixture only; not an end-to-end Warp reconstruction comparison",
    }
    (out/"manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    print(out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default="sphere")
    inputs=parser.add_mutually_exclusive_group()
    inputs.add_argument("--dataset",choices=DATASETS)
    inputs.add_argument("--input",help="Oriented PLY or six-column ASCII file")
    parser.add_argument("--max-points",type=int,help="Deterministic subset size for file inputs; default uses every point")
    parser.add_argument("--shape", choices=["sphere","torus","plane","sheets","cluster","disconnected"], default="sphere")
    parser.add_argument("--input-scale",type=float,default=1.0)
    parser.add_argument("--translation",type=float,nargs=3,default=[0.,0.,0.])
    parser.add_argument("--count", type=int, default=512)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--full-depth", type=int, default=2)
    parser.add_argument("--base-depth", type=int)
    parser.add_argument("--solve-depth", type=int)
    parser.add_argument("--kernel-depth", type=int)
    parser.add_argument("--scale", type=float, default=1.1)
    parser.add_argument("--samples-per-node", type=float, default=1.5)
    parser.add_argument("--cg-accuracy", type=float, default=1e-3)
    parser.add_argument("--base-v-cycles", type=int, default=1)
    parser.add_argument("--weight", type=float, default=2.)
    parser.add_argument("--exact", action="store_true")
    parser.add_argument("--iters", type=int, default=8)
    parser.add_argument("--precision", choices=["float","double"], default="float")
    parser.add_argument("--grid", type=int, default=24)
    parser.add_argument("--operators", action="store_true", help="Export small explicit matrices and interpolation records")
    generate(parser.parse_args())
