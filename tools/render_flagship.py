"""Render actual official/Warp surfaces with identical extraction, camera and lighting."""
import argparse
import json
import subprocess
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
from skimage.measure import marching_cubes
from tools.datasets import checksum
from tools.make_fixture import ROOT


BACKGROUND = '#f3f5f6'
INK = '#192d3c'


def surface(grid, iso):
    vertices, faces, _, _ = marching_cubes(grid, level=iso, allow_degenerate=False)
    vertices = vertices/(np.asarray(grid.shape)-1)-.5
    connectivity = np.column_stack([np.full(len(faces), 3), faces])
    return pv.PolyData(vertices, connectivity).compute_normals(auto_orient_normals=True, consistent_normals=True)


def photograph(mesh, camera, up, scale, size=(1000, 1100)):
    plotter = pv.Plotter(off_screen=True, window_size=size, lighting='none')
    plotter.set_background(BACKGROUND)
    plotter.add_mesh(mesh, color='#78aaa9', smooth_shading=True, split_sharp_edges=False,
                     ambient=.23, diffuse=.8, specular=.3, specular_power=32)
    center = np.zeros(3)
    # All transforms are shared between panels; no geometry smoothing or trimming.
    plotter.camera_position = [tuple(center+np.asarray(camera)), tuple(center), tuple(up)]
    plotter.enable_parallel_projection()
    plotter.camera.parallel_scale = scale
    view = np.asarray(camera)/np.linalg.norm(camera)
    vertical = np.asarray(up)/np.linalg.norm(up)
    right = np.cross(view, vertical); right /= np.linalg.norm(right)
    for position, intensity, color in [(3*view+3*vertical-2*right, 1.1, '#fff5e8'), (2*view+3*right, .5, '#d9eaff'), (-2*view+2*vertical, .7, 'white')]:
        plotter.add_light(pv.Light(position=position, focal_point=tuple(center), color=color, intensity=intensity))
    plotter.enable_ssao(radius=.045, bias=.001, kernel_size=128)
    plotter.enable_anti_aliasing('ssaa')
    result = plotter.screenshot(return_img=True)
    plotter.close()
    return result


def render(args):
    directory = Path(args.directory)
    metadata = json.loads((directory/f'grid-{args.resolution}.json').read_text())
    report = json.loads((directory/'reconstruction_report.json').read_text())
    manifest = json.loads((directory/'manifest.json').read_text())
    for key in ('official', 'warp'):
        if checksum(directory/metadata[f'{key}_file']) != metadata[f'{key}_sha256']:
            raise ValueError(f'{key} grid checksum differs from its comparison report')
    a = np.fromfile(directory/metadata['official_file'], dtype=metadata['dtype']).reshape((args.resolution,)*3)
    b = np.load(directory/metadata['warp_file'])
    meshes = [surface(a, metadata['official_iso_value']), surface(b, metadata['warp_iso_value'])]
    # A shared reference-centered camera also makes tiny positional differences visible.
    view = np.asarray(args.camera)/np.linalg.norm(args.camera)
    right = np.cross(view, args.up); right /= np.linalg.norm(right)
    vertical = np.cross(right, view)
    frame = np.column_stack([right, vertical, view])
    projected = meshes[0].points@frame
    low, high = projected.min(axis=0), projected.max(axis=0)
    common_center = frame@((low+high)/2)
    scale = args.scale or float(max((high[1]-low[1])/2, (high[0]-low[0])/2/(1000/1100))*1.12)
    for mesh in meshes:
        mesh.points -= common_center
    pictures = [photograph(mesh, args.camera, args.up, scale) for mesh in meshes]
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'text.color': INK, 'axes.labelcolor': INK})
    fig = plt.figure(figsize=(15, 9), facecolor=BACKGROUND)
    fig.text(.045, .94, args.title, fontsize=25, weight='bold')
    method = 'screened Poisson' if manifest['parameters']['weight'] else 'Poisson'
    fig.text(.045, .9, f"{manifest['parameters']['count']:,} oriented points  /  {method}  /  depth {manifest['parameters']['depth']}", fontsize=12, color='#516673')
    for i, (picture, title, subtitle) in enumerate(zip(pictures, ['Official PoissonRecon', 'Pure Warp'],
                                                       ['Pinned C++ reference', 'Independent GPU reconstruction'])):
        ax = fig.add_axes([.015+i*.365, .125, .38, .73])
        ax.imshow(picture); ax.axis('off')
        fig.text(.205+i*.365, .845, title, ha='center', fontsize=17, weight='bold')
        fig.text(.205+i*.365, .813, subtitle, ha='center', fontsize=10, color='#516673')
    error = np.abs(a.astype(np.float64)-b)
    ax = fig.add_axes([.775, .435, .2, .31])
    im = ax.imshow(error[:, args.resolution//2, :].T, origin='lower', cmap='magma', vmin=0, vmax=max(error.max(), 1e-12), extent=[0, 1, 0, 1])
    ax.set_title('Volume field difference', fontsize=11, pad=12, color=INK)
    ax.set_xlabel('Normalized model x', fontsize=9); ax.set_ylabel('Normalized model z', fontsize=9)
    ax.tick_params(labelsize=8)
    cax = fig.add_axes([.795, .365, .16, .012])
    fig.colorbar(im, cax=cax, orientation='horizontal').ax.tick_params(labelsize=8)
    fig.text(.775, .255, f"{metadata['max_abs_value_error']:.2e}", fontsize=23, weight='bold')
    fig.text(.775, .198, f"max |Δf| across {metadata['grid_queries']:,}\nshared grid queries", fontsize=10, linespacing=1.5)
    fig.text(.775, .15, f"{report['coefficients']:,} coefficients\nExact active / support keys", fontsize=10, linespacing=1.6)
    fig.text(.045, .072, f"Same {args.resolution}³ marching-cubes grid, material, camera and lighting. No mesh smoothing or trimming.", fontsize=10, color='#516673')
    fig.text(.045, .041, args.credit, fontsize=9, color='#516673')
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160, facecolor=fig.get_facecolor())
    plt.close(fig)
    rendering = {'grid': metadata, 'camera': args.camera, 'up': args.up, 'parallel_scale': scale, 'model_grid_center': common_center.tolist(),
                 'render_source_revision': subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                 'render_source_sha256': checksum(Path(__file__)),
                 'pyvista': pv.__version__, 'vtk': pv.vtk_version_info,
                 'extractor': 'skimage.measure.marching_cubes; no smoothing, decimation or trimming',
                 'meshes': [{'vertices': mesh.n_points, 'triangles': mesh.n_cells} for mesh in meshes]}
    output.with_suffix('.json').write_text(json.dumps(rendering, indent=2)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory')
    parser.add_argument('--resolution', type=int, default=256)
    parser.add_argument('--camera', type=float, nargs=3, default=[2, -.5, -2])
    parser.add_argument('--up', type=float, nargs=3, default=[0, -1, 0])
    parser.add_argument('--scale', type=float, help='Override automatic shared framing')
    parser.add_argument('--title', default='The eagle, reconstructed independently')
    parser.add_argument('--credit', default='Eagle scan: EPFL Geometric Computing Laboratory, “Scanning 3D Statues from Photos”; distributed by PoissonRecon.')
    parser.add_argument('--output', default='docs/results/eagle-reconstruction.png')
    render(parser.parse_args())
