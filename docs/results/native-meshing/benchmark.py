import argparse
import gc
import json
from pathlib import Path
import subprocess
import time
from types import SimpleNamespace

import numpy as np
import warp as wp
import warp.geometry as geometry
from warp._src.geometry.poisson import meshing
from warp._src.geometry.poisson.field_adjoint import cube_field
from tools.benchmark_input import benchmark_input, reconstruction_parameters
from tools.cpu_environment import cpu_environment

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='eagle')
parser.add_argument('--depth', type=int, default=8)
parser.add_argument('--variant', choices=['native', 'slabs', 'dense'], default='native')
parser.add_argument('--repeats', type=int, default=3)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--profile', action='store_true')
args = parser.parse_args()
activity = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,process_name', '--format=csv,noheader'], text=True).strip()
if activity:
    print(activity, flush=True)
    raise SystemExit(75)
report = dict(dataset=args.dataset, depth=args.depth, variant=args.variant, environment_before=cpu_environment())
wp.init()
wp.config.quiet = True
positions, normals, parameters, provenance = benchmark_input(SimpleNamespace(fixture=Path(f'.build/gpu-baselines/inputs/{args.dataset}-full-d8-spsr')))
options = reconstruction_parameters(parameters)
options.pop('exact_interpolation')
options['iterations'] = options.pop('iters')
options.update(depth=args.depth, solve_depth=args.depth, kernel_depth=args.depth-2)
points = wp.array(positions, dtype=wp.vec3, device='cuda:0')
normals = wp.array(normals, dtype=wp.vec3, device='cuda:0')
result = geometry.poisson_reconstruct(points, normals, **options)
raw = result._result
cube = cube_field(raw.field.keys, raw.field.coefficients, raw.field.cumulative, raw.field.depth)
transforms = wp.array([raw.metadata['cube_to_model']], dtype=wp.mat44, device='cuda:0')
resolution = (1 << args.depth) + 1

def sample(points):
    values, _, _ = cube.evaluate(points)
    out = wp.empty_like(values)
    wp.launch(meshing._center_kernel, len(values), inputs=[values, result.iso_value_array, out], device='cuda:0', record_tape=False)
    return out

statistics = {}
def extract():
    if args.variant == 'native':
        return meshing.extract_native_mesh(raw, resolution=None, backend='sparse', iso=result.iso_value_array, differentiable=False, statistics=statistics)
    if args.variant == 'dense':
        return geometry.poisson_extract_mesh(result, method='dense')
    vertices, indices = meshing.extract_isosurface(sample, resolution=resolution, bounds=((0,0,0),(1,1,1)), device='cuda:0')
    out = wp.empty_like(vertices)
    wp.launch(meshing._cube_vertices_kernel, len(vertices), inputs=[vertices, transforms, out], device='cuda:0')
    return out, indices

mesh = extract()
wp.synchronize_device()
report.update(points=len(points), input=provenance, vertices=len(mesh[0]), triangles=len(mesh[1])//3)
report['seconds'] = []
for iteration in range(args.repeats):
    del mesh
    gc.collect()
    wp.synchronize_device()
    start = time.perf_counter()
    mesh = extract()
    wp.synchronize_device()
    report['seconds'].append(time.perf_counter()-start)
report['statistics'] = statistics
report['extraction_phase_mempool_peak_bytes'] = wp.get_mempool_used_mem_high('cuda:0')
report['points_to_mesh_seconds'] = []
for iteration in range(args.repeats):
    del mesh, result, raw, cube, transforms
    gc.collect()
    wp.synchronize_device()
    start = time.perf_counter()
    result = geometry.poisson_reconstruct(points, normals, **options)
    raw = result._result
    cube = cube_field(raw.field.keys, raw.field.coefficients, raw.field.cumulative, raw.field.depth)
    transforms = wp.array([raw.metadata['cube_to_model']], dtype=wp.mat44, device='cuda:0')
    mesh = extract()
    wp.synchronize_device()
    report['points_to_mesh_seconds'].append(time.perf_counter()-start)

if args.variant != 'dense':
    from unittest import mock
    saved = {}
    implementation = meshing.sparse_marching_cubes_from_cells
    def capture(cells, values, **options):
        saved['cells'], saved['values'] = cells, values
        return implementation(cells, values, **options)
    with mock.patch.object(meshing, 'sparse_marching_cubes_from_cells', capture):
        validation_mesh = extract()
    cells, values = saved['cells'].numpy(), saved['values'].numpy()
    order = np.lexsort(cells.T)
    import hashlib
    report['crossing_cells_digest'] = hashlib.sha256(cells[order].tobytes()).hexdigest()
    report['crossing_values_digest'] = hashlib.sha256(values[order].tobytes()).hexdigest()
    del validation_mesh, saved

# Independent vertex ordering; quantization only affects this reporting hash.
import hashlib
vertices = np.round(mesh[0].numpy().astype(np.float64), 6)
order = np.lexsort(vertices.T)
rank = np.argsort(order)
faces = rank[mesh[1].numpy()].reshape(-1, 3)
faces = np.sort(faces, axis=1)
faces = faces[np.lexsort(faces.T)]
report['mesh_digest'] = hashlib.sha256(vertices[order].tobytes()+faces.tobytes()).hexdigest()
if args.profile:
    from unittest import mock
    events = []
    launch = wp.launch
    def profiled_launch(kernel, *a, **kw):
        begin, end = wp.Event('cuda:0', enable_timing=True), wp.Event('cuda:0', enable_timing=True)
        wp.record_event(begin)
        output = launch(kernel, *a, **kw)
        wp.record_event(end)
        name = kernel.key
        if name == '_build_bounds_kernel': name += '_depth_' + str(kw['inputs'][5])
        events.append((name, begin, end))
        return output
    from warp._src.geometry.poisson import meshing_bounds
    stage_events = []
    def stage_wrapper(name, implementation):
        def wrapped(*a, **kw):
            begin, end = wp.Event('cuda:0', enable_timing=True), wp.Event('cuda:0', enable_timing=True)
            wp.record_event(begin)
            output = implementation(*a, **kw)
            wp.record_event(end)
            stage_events.append((name, begin, end))
            return output
        return wrapped
    with mock.patch.object(meshing_bounds, 'candidate_cells', stage_wrapper('candidate_construction', meshing_bounds.candidate_cells)), mock.patch.object(meshing, 'sparse_marching_cubes_from_cells', stage_wrapper('sparse_mc', meshing.sparse_marching_cubes_from_cells)), mock.patch.object(wp, 'launch', profiled_launch):
        diagnostic_mesh = extract()
    wp.synchronize_device()
    groups = {}
    for name, begin, end in events:
        groups[name] = groups.get(name, 0) + wp.get_event_elapsed_time(begin, end, synchronize=False)
    report['instrumented_kernel_ms'] = groups
    report['instrumented_stage_ms'] = {name: wp.get_event_elapsed_time(begin, end, synchronize=False) for name, begin, end in stage_events}
    print('PROFILE', json.dumps(groups), flush=True)
    keys = raw.field.keys.numpy()
    d = raw.field.depth - 1
    level_keys = keys[(keys >> 48) == d]
    coords = np.stack([(level_keys >> b) & 65535 for b in (32,16,0)],axis=1)
    coords = coords[np.max(coords,axis=1)<(1<<d)]
    offsets = np.array([[x,y,z] for x in (0,1) for y in (0,1) for z in (0,1)])
    children = coords[:,None,:]*2+offsets[None,:,:]
    codes = ((d+1)<<48)|(children[:,:,0]<<32)|(children[:,:,1]<<16)|children[:,:,2]
    indices = np.searchsorted(keys,codes)
    present = (indices<len(keys)) & (keys[np.minimum(indices,len(keys)-1)]==codes)
    print('CHILDREN',np.bincount(present.sum(axis=1),minlength=9).tolist(),flush=True)

report['mempool_peak_bytes'] = wp.get_mempool_used_mem_high('cuda:0')
report['environment_after'] = cpu_environment()
report['source_sha256'] = {name: hashlib.sha256((Path(meshing.__file__).parent/name).read_bytes()).hexdigest() for name in ('meshing.py','meshing_bounds.py','field.py','reconstruction.py')}
args.output.write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps({key:report[key] for key in ('dataset','depth','variant','vertices','triangles','seconds','statistics','mempool_peak_bytes')}), flush=True)
