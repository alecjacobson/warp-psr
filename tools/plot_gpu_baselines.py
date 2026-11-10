"""Render matched views/zero contours and sampling-based mesh distances for eagle."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree


def sample_mesh(vertices, faces, count=200000, seed=2026):
    triangles = vertices[faces]
    area = np.linalg.norm(np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]), axis=1) / 2
    rng = np.random.default_rng(seed)
    selected = triangles[rng.choice(len(faces), count, p=area / area.sum())]
    uv = rng.random((count, 2))
    root = np.sqrt(uv[:, 0])
    weights = np.column_stack([1 - root, root * (1 - uv[:, 1]), root * uv[:, 1]])
    return (weights[:, :, None] * selected).sum(1), float(area.sum())


def main(args):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import pyvista as pv
    labels = [('warp', 'Warp SPSR', '#6bd8ed'), ('sap', 'SAP FFT / sigma=10', '#ffc078'),
              ('sap-sigma1', 'SAP FFT / sigma=1', '#f582ae'), ('ng', 'NeuralGalerkin SPSR / density=32', '#b6a0ff')]
    plotter = pv.Plotter(shape=(2, 4), off_screen=True, window_size=(2000, 1080), border=False)
    metrics = {}
    fig, axes = plt.subplots(2, 3, figsize=(13, 8), layout='constrained')
    for row, depth in enumerate([8, 9]):
        samples = {}
        for col, (method, label, color) in enumerate(labels):
            stem = args.directory / f'eagle-d{depth}-{method}'
            report = json.loads(stem.with_suffix('.json').read_text())
            data = np.load(stem.with_suffix('.mesh.npz'))
            v, f = data['vertices'], data['faces']
            samples[method], area = sample_mesh(v, f)
            metrics[f'd{depth}-{method}'] = dict(surface_area=area, vertices=len(v), triangles=len(f))
            plotter.subplot(row, col)
            plotter.set_background('#101923')
            mesh = pv.PolyData(v, np.column_stack([np.full(len(f), 3), f]).ravel())
            plotter.add_mesh(mesh, color=color, smooth_shading=True, specular=.22, specular_power=25)
            timing = f'{report["warm_median_seconds"] * 1000:.1f} ms field'
            if method == 'sap':
                cached = json.loads((args.directory / f'eagle-d{depth}-sap-cached.json').read_text())
                timing += f' / {cached["warm_median_seconds"] * 1000:.1f} ms cached'
            plotter.add_text(f'{label}\n{timing}\nNominal depth {depth}',
                             font_size=10, color='white', position='upper_left')
            plotter.camera_position = [(2.5, 0., -1.5), (.5, .5, .5), (0, -1, 0)]
            plotter.enable_parallel_projection()
            plotter.camera.parallel_scale = .58
            slices = np.load(stem.with_suffix('.slices.npz'))
            for axis in range(3):
                ax = axes[row, axis]
                ax.contour(slices['coordinates'], slices['coordinates'], slices['values'][axis].T,
                           levels=[0], colors=[color], linewidths=1.25)
                ax.set_aspect('equal'); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
                ax.set_title(f'Depth {depth} · {"xyz"[axis]} = 0.5')
                ax.set_facecolor('#101923')
        reference = samples['warp']
        for method, _, _ in labels:
            # Also evaluate an independent Warp resampling to expose the sampling floor.
            sample = samples[method]
            if method == 'warp':
                stem = args.directory / f'eagle-d{depth}-warp.mesh.npz'
                data = np.load(stem)
                sample, _ = sample_mesh(data['vertices'], data['faces'], seed=2027)
            a = cKDTree(reference).query(sample, workers=4)[0]
            b = cKDTree(sample).query(reference, workers=4)[0]
            metrics[f'd{depth}-{method}']['distance_to_warp'] = dict(
                symmetric_mean=float((a.mean() + b.mean()) / 2),
                symmetric_rms=float(np.sqrt((np.mean(a*a) + np.mean(b*b)) / 2)),
                max_directed_p95=float(max(np.quantile(a, .95), np.quantile(b, .95))))
    fig.legend([plt.Line2D([0], [0], color=c) for _, _, c in labels],
               [label for _, label, _ in labels], loc='outside lower center', ncol=4)
    fig.savefig(args.output.with_suffix('.slices.png'), dpi=160)
    plotter.screenshot(str(args.output.with_suffix('.meshes.png')))
    plotter.close()
    args.output.with_suffix('.metrics.json').write_text(json.dumps(dict(
        note='Distances between area-weighted random surface samples, in normalized cube units; '
             '200,000 samples/method. Warp self-resampling estimates the sampling floor. '
             'These are disagreement measures, not ground-truth reconstruction accuracy.',
        metrics=metrics), indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    main(parser.parse_args())
