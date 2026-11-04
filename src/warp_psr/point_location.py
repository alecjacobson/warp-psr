"""Explicit float64 containing-cell coordinates for reconstruction grids.

Warp 1.19's general Nanogrid nearest-cell lookup mixes distance precisions
(issue #31), as does Grid3D particle lookup's distance bound (issue #66).
These constraints need only a containing-cell lookup.
Feed the result through PicQuadrature's public (cell, coordinates) constructor.
"""

import warp as wp
import warp.fem as fem


@wp.kernel
def _locate_grid(positions: wp.array(dtype=wp.vec3d), origin: wp.vec3d,
                 cell_size: wp.vec3d, res: wp.vec3i,
                 cells: wp.array(dtype=int), coordinates: wp.array(dtype=wp.vec3d)):
    i = wp.tid()
    p = wp.cw_div(positions[i] - origin, cell_size)
    cells[i] = -1
    coordinates[i] = wp.vec3d(wp.float64(0.0))
    if wp.min(p) >= wp.float64(0.0) and p[0] <= wp.float64(res[0]) and p[1] <= wp.float64(res[1]) and p[2] <= wp.float64(res[2]):
        c = wp.vec3i(wp.min(int(wp.floor(p[0])), res[0] - 1),
                     wp.min(int(wp.floor(p[1])), res[1] - 1),
                     wp.min(int(wp.floor(p[2])), res[2] - 1))
        cells[i] = fem.Grid3D.cell_index(res, c)
        coordinates[i] = p - wp.vec3d(c)


@wp.kernel
def _locate(
    positions: wp.array(dtype=wp.vec3d),
    grid: wp.uint64,
    levels: wp.array(dtype=wp.uint8),
    cells: wp.array(dtype=int),
    coordinates: wp.array(dtype=wp.vec3d),
):
    i = wp.tid()
    p = wp.volume_world_to_index(grid, positions[i]) + wp.vec3d(wp.float64(0.5))
    fine = wp.vec3i(int(wp.floor(p[0])), int(wp.floor(p[1])), int(wp.floor(p[2])))
    cells[i] = -1
    coordinates[i] = wp.vec3d(wp.float64(0.0))
    for shift in range(8):
        c = wp.vec3i(
            (fine[0] >> shift) << shift,
            (fine[1] >> shift) << shift,
            (fine[2] >> shift) << shift,
        )
        cell = wp.volume_lookup_index(grid, c[0], c[1], c[2])
        if cell >= 0:
            if int(levels[cell]) == shift:
                cells[i] = cell
                coordinates[i] = (p - wp.vec3d(c)) / wp.float64(1 << shift)
                break


def locate_constraints(geometry, positions):
    cells = wp.empty(len(positions), dtype=int, device=positions.device)
    coordinates = wp.empty(len(positions), dtype=wp.vec3d, device=positions.device)
    args = geometry.cell_arg_value(positions.device)
    if isinstance(geometry, fem.Grid3D):
        wp.launch(_locate_grid, len(positions),
                  inputs=[positions, args.origin, args.cell_size, args.res, cells, coordinates],
                  device=positions.device)
        return cells, coordinates
    levels = (args.cell_level if isinstance(geometry, fem.AdaptiveNanogrid)
              else wp.zeros(geometry.cell_count(), dtype=wp.uint8, device=positions.device))
    wp.launch(
        _locate,
        len(positions),
        inputs=[positions, geometry.cell_grid.id, levels, cells, coordinates],
        device=positions.device,
    )
    return cells, coordinates
