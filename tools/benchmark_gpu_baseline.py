"""Compare different GPU PSR formulations; these are not reference-equivalent solvers.

Run each method in its own process/environment. Upstream sources stay external;
the adapters execute their original solver functions, excluding unused imports.
"""
import argparse
import ast
import gc
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

import numpy as np

from tools.cpu_environment import cpu_environment


SAP_COMMIT = "a094733f127b9be888aa8728d41cca3dd9b36394"
NG_COMMIT = "5ec184004edbbf7492e49167d5063ac16712f629"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git(path, *args):
    return subprocess.check_output(["git", "-C", str(path), *args], text=True).strip()


def source_info(path, expected):
    revision = git(path, "rev-parse", "HEAD")
    if revision != expected:
        raise ValueError(f"Expected upstream revision {expected}, got {revision}")
    diff = git(path, "diff", "HEAD")
    return dict(revision=revision, tracked_diff=diff,
                files={str(p.relative_to(path)): digest(p)
                       for p in path.rglob("*") if p.suffix in (".py", ".cu", ".cpp", ".h")
                       and ".git" not in p.parts and "build" not in p.parts})


def sap_class(root):
    """Load verbatim DPSR/helper ASTs without SAP's unrelated rendering imports."""
    import torch
    names = {"fftfreqs", "img", "spec_gaussian_filter", "grid_interp",
             "scatter_to_grid", "point_rasterize"}
    path = root / "src/utils.py"
    tree = ast.parse(path.read_text())
    tree.body = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    if {n.name for n in tree.body} != names:
        raise ValueError("Upstream DPSR helpers changed")
    namespace = dict(torch=torch, np=np)
    exec(compile(tree, str(path), "exec"), namespace)
    path = root / "src/dpsr.py"
    tree = ast.parse(path.read_text())
    tree.body = [n for n in tree.body if not
                 (isinstance(n, ast.ImportFrom) and n.module == "src.utils")]
    exec(compile(tree, str(path), "exec"), namespace)
    return namespace["DPSR"]


def ng_field_function(root):
    """Only replace the public wrapper's final mesh extraction with a field return."""
    sys.path.insert(0, str(root.resolve()))
    import torch
    path = root / "torch_spsr/__init__.py"
    tree = ast.parse(path.read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "reconstruct")
    if not (isinstance(fn.body[-2], ast.Assign) and
            isinstance(fn.body[-2].value, ast.Call) and
            fn.body[-2].value.func.attr == "extract_multiscale_mesh"):
        raise ValueError("Upstream reconstruct wrapper changed")
    fn.body[-2:] = [ast.Return(value=ast.Name(id="reconstructor", ctx=ast.Load()))]
    fn.returns = None
    tree = ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[]))
    namespace = {"torch": torch}
    exec(compile(tree, str(path), "exec"), namespace)
    return namespace["reconstruct"]


def gpu_processes():
    text = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
                                    "--format=csv,noheader"], text=True)
    rows = text.strip().splitlines()
    foreign = [r for r in rows if r.split(",")[0].strip() != str(os.getpid())]
    if foreign:
        raise RuntimeError(f"GPU not isolated: {foreign}")
    return rows


def cpu_snapshot():
    """Snapshot visible processes; short-lived processes between samples can be missed."""
    result = {}
    for path in Path("/proc").glob("[0-9]*/stat"):
        try:
            raw = path.read_text()
            fields = raw[raw.rfind(")") + 2:].split()
            result[int(path.parent.name)] = (fields[19], int(fields[11]) + int(fields[12]),
                                             raw[raw.find("(") + 1:raw.rfind(")")])
        except (OSError, ValueError, IndexError):
            continue
    return result


def foreign_cpu(before):
    ticks = os.sysconf("SC_CLK_TCK")
    rows = []
    for pid, (started, used, name) in cpu_snapshot().items():
        if pid == os.getpid() or pid not in before or started != before[pid][0]:
            continue
        seconds = (used - before[pid][1]) / ticks
        if seconds > .01:
            rows.append(dict(pid=pid, name=name, cpu_seconds=seconds))
    return rows


def main(args):
    manifest = json.loads((args.fixture / "manifest.json").read_text())
    if manifest["precision"] != "float":
        raise ValueError("These external baseline comparisons require a float32 fixture")
    path = args.fixture / "input.pts"
    if digest(path) != manifest["input_sha256"]:
        raise ValueError("Fixture input hash mismatch")
    data = np.loadtxt(path, dtype=np.float32)
    positions, normals = data[:, :3].copy(), data[:, 3:].copy()
    parameters = dict(manifest["parameters"])
    parameters.update(depth=args.depth, solve_depth=args.depth, kernel_depth=args.depth - 2)
    lo, hi = positions.min(0), positions.max(0)
    # Explicit scalar precision avoids NumPy 1.x/2.x promotion differences.
    side = np.float32(np.max(hi - lo) * np.float32(parameters["scale"]))
    origin = (lo + hi) * np.float32(0.5) - side * np.float32(0.5)
    normalized = (positions - origin) / side
    env_before = cpu_environment()
    setup_start = time.perf_counter()
    report = dict(method=args.method, fixture=manifest["name"], count=len(positions), numpy=np.__version__,
                  depth=args.depth, input_sha256=manifest["input_sha256"],
                  normalization=dict(origin=origin.tolist(), side=float(side), scale=parameters["scale"]),
                  execution_environment=env_before,
                  source_revision=git(Path.cwd(), "rev-parse", "HEAD"),
                  harness_sha256=digest(Path(__file__)),
                  gpu=subprocess.check_output(["nvidia-smi", "--query-gpu=name,driver_version",
                                               "--format=csv,noheader"], text=True).strip(),
                  timing_scope="Synchronized wall time, uploaded inputs to solved field/isovalue; includes Python, excludes setup, meshing, evaluation and input upload",
                  first_run_note="First invocation after setup; existing disk caches may be reused, not a clean-cache compile time")
    if args.method == "warp":
        import warp as wp
        from warp_psr import reconstruct
        from warp_psr.meshing import extract_mesh
        from tools.benchmark_input import reconstruction_parameters
        from tools.fixture_provenance import production_hashes
        wp.init()
        wp.set_device("cuda:0")
        p, n = (wp.array(a, dtype=wp.vec3, device="cuda:0") for a in (positions, normals))
        options = reconstruction_parameters(parameters)
        report.update(parameters=options, warp=wp.__version__, production_source_sha256=production_hashes())
        sync = wp.synchronize
        reconstruct_field = lambda: reconstruct(p, n, **options)

        def mesh_field(field):
            # A resolution-d cube has 2**d cells. Avoid evaluating exactly at the domain boundary.
            bounds = (origin + side * .001, origin + side * .999)
            return extract_mesh(field, resolution=2 ** args.depth + 1,
                                bounds=bounds, backend="sparse")

        def download_mesh(mesh):
            return (mesh.vertices.numpy() - origin) / side, mesh.indices.numpy().reshape(-1, 3)

        def evaluate(field, q):
            v, _, valid = field.evaluate(wp.array(origin + q * side, dtype=wp.vec3, device="cuda:0"))
            if not valid.numpy().all():
                raise ValueError("Invalid slice query")
            return v.numpy() - field.iso_value

        memory = lambda: {"warp_pool_high_bytes": wp.get_mempool_used_mem_high("cuda:0")}
    else:
        import torch
        torch.set_num_threads(args.threads)
        torch.set_num_interop_threads(1)
        torch.set_grad_enabled(False)
        torch.manual_seed(2026)
        p, n = (torch.tensor(a, device="cuda:0") for a in (normalized, normals))
        sync = torch.cuda.synchronize
        report.update(torch=torch.__version__, torch_cuda=torch.version.cuda, cpu_threads=args.threads,
                      pytorch_cuda_alloc_conf=os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "default"))
        if args.method == "sap":
            report["upstream"] = source_info(args.upstream, SAP_COMMIT)
            dpsr = sap_class(args.upstream)((2 ** args.depth,) * 3, sig=args.sigma).cuda()
            report["parameters"] = dict(resolution=2 ** args.depth, sigma=args.sigma,
                                        scale=True, shift=True, boundary="periodic", screening=False)
            reconstruct_field = lambda: dpsr(p[None], n[None])

            def mesh_field(field):
                from skimage.measure import marching_cubes
                # Matches the nonperiodic domain covered by the upstream output samples.
                v, f, _, _ = marching_cubes(field[0].cpu().numpy(), level=0.,
                                            spacing=(1. / (2 ** args.depth),) * 3)
                return v, f

            download_mesh = lambda mesh: mesh

            def evaluate(field, q):
                from scipy.ndimage import map_coordinates
                return map_coordinates(field[0].cpu().numpy(), (q * 2 ** args.depth).T,
                                       order=1, mode="grid-wrap", prefilter=False)
        else:
            report["upstream"] = source_info(args.upstream, NG_COMMIT)
            fn = ng_field_function(args.upstream)
            from torch_spsr.ext import sparse_solve
            original_pcg = sparse_solve.solve_pcg_diag
            cg_calls = []

            def counted_pcg(*arguments):
                result, iterations = original_pcg(*arguments)
                cg_calls.append(dict(rows=arguments[3].numel(), iterations=iterations))
                return result, iterations

            sparse_solve.solve_pcg_diag = counted_pcg
            report["parameters"] = dict(voxel_size=1. / 2 ** args.depth, hierarchy_levels=args.ng_levels,
                                        min_density=args.min_density, screen_alpha=args.screen_alpha,
                                        solver="pcg", tolerance=1.e-5, maxiter=-1,
                                        basis="quadratic Bezier tensor", n_upsample=2)

            def reconstruct_field():
                result = fn(p, n, voxel_size=1. / 2 ** args.depth, depth=args.ng_levels,
                            min_density=args.min_density, screen_alpha=args.screen_alpha)
                # Include the data-derived surface isovalue, as in the other two field timings.
                result.set_fixed_level_set(result.get_mean_chi())
                return result

            mesh_field = lambda field: field.extract_multiscale_mesh(n_upsample=2, build_o3d_mesh=False)
            download_mesh = lambda mesh: (mesh[0].cpu().numpy(), mesh[1].cpu().numpy())
            evaluate = lambda field, q: field.evaluate_chi(torch.tensor(q, device="cuda:0")).cpu().numpy()
        memory = lambda: {"torch_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                          "torch_peak_reserved_bytes": torch.cuda.max_memory_reserved()}
    sync()
    report["setup_seconds"] = time.perf_counter() - setup_start
    records = []
    for iteration in range(args.repeats + 1):
        if args.method == "ng":
            cg_calls.clear()
        gc.collect()
        processes = gpu_processes()
        cpu_before = cpu_snapshot()
        sync()
        start = time.perf_counter()
        field = reconstruct_field()
        sync()
        elapsed = time.perf_counter() - start
        record = dict(iteration=iteration, reconstruction_seconds=elapsed, gpu_processes_before=processes)
        if args.method == "ng":
            record["pcg_calls"] = list(cg_calls)
        if args.mesh:
            start = time.perf_counter()
            mesh = mesh_field(field)
            sync()
            record.update(meshing_seconds=time.perf_counter() - start)
            record["points_to_mesh_seconds"] = elapsed + record["meshing_seconds"]
        record.update(memory())
        record["visible_foreign_cpu_activity"] = foreign_cpu(cpu_before)
        record["gpu_processes_after"] = gpu_processes()
        records.append(record)
        print(json.dumps(record), flush=True)
        if iteration != args.repeats:
            if args.mesh:
                del mesh
            del field
    report["runs"] = records
    report["warm_median_seconds"] = statistics.median(r["reconstruction_seconds"] for r in records[1:])
    report["execution_environment_after"] = cpu_environment()
    if args.mesh:
        v, f = download_mesh(mesh)
        if not np.isfinite(v).all() or len(f) == 0:
            raise ValueError("Non-finite or empty mesh")
        report["mesh"] = dict(vertices=len(v), triangles=len(f), bounds=[v.min(0).tolist(), v.max(0).tolist()],
                              extraction="GPU sparse MC" if args.method == "warp" else
                              "CPU skimage MC plus field download" if args.method == "sap" else "GPU dual multiscale MC")
        np.savez_compressed(args.output.with_suffix(".mesh.npz"), vertices=v, faces=f)
        report["warm_points_to_mesh_median_seconds"] = statistics.median(r["points_to_mesh_seconds"] for r in records[1:])
    # Three central slices, normalized coordinates. Field scales differ; compare zero contours.
    grid = np.linspace(.01, .99, 192, dtype=np.float32)
    a, b = np.meshgrid(grid, grid, indexing="ij")
    slices = []
    for axis in range(3):
        q = np.full((a.size, 3), .5, dtype=np.float32)
        q[:, [i for i in range(3) if i != axis]] = np.column_stack([a.ravel(), b.ravel()])
        values = evaluate(field, q)
        if not np.isfinite(values).all():
            raise ValueError("Non-finite slice values")
        slices.append(values.reshape(a.shape))
    np.savez_compressed(args.output.with_suffix(".slices.npz"), coordinates=grid, values=np.array(slices))
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=["warp", "sap", "ng"], required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--upstream", type=Path)
    parser.add_argument("--depth", type=int, choices=[8, 9], default=8)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--sigma", type=float, default=10.)
    parser.add_argument("--ng-levels", type=int, default=4)
    parser.add_argument("--min-density", type=float, default=32.)
    parser.add_argument("--screen-alpha", type=float, default=4.)
    parser.add_argument("--mesh", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.repeats < 1 or args.threads < 1 or (args.method != "warp" and not args.upstream):
        parser.error("Require repeats/threads >= 1 and --upstream for external methods")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    main(args)
