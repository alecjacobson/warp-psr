"""Check float64 containing-cell lookup across an adaptive refinement interface."""
import itertools
import numpy as np
import pytest
import warp as wp
import warp.fem as fem
from warp_psr.point_location import locate_constraints


@pytest.mark.cuda
def test_float64_adaptive_point_location():
    wp.init()
    if not wp.is_cuda_available():
        pytest.skip("CUDA unavailable")
    # Refine one octant of the unit cube: seven coarse and eight fine cells.
    anchors = {tuple(2 * x for x in c): 1 for c in itertools.product(range(2), repeat=3) if any(c)}
    anchors.update({c: 0 for c in itertools.product(range(2), repeat=3)})
    with wp.ScopedDevice("cuda"):
        volume = wp.Volume.allocate_by_voxels(
            wp.array(list(anchors), dtype=wp.vec3i),
            voxel_size=0.25, translation=(0.125, 0.125, 0.125),
        )
        origins = volume.get_voxels().numpy()
        levels = np.array([anchors[tuple(c)] for c in origins], dtype=np.uint8)
        geometry = fem.AdaptiveNanogrid(
            volume, cell_level=wp.array(levels, dtype=wp.uint8),
            level_count=2, scalar_type=wp.float64, temporary_store=None,
        )
        epsilon = 2.0 ** -40  # Deliberately below float32 coordinate resolution.
        probes = [[0.5 + delta, y, z] for delta in (-epsilon, 0, epsilon)
                  for y in (0.125, 0.375, 0.625) for z in (0.125, 0.625)]
        points = np.vstack([np.random.default_rng(41).uniform(size=(256, 3)), probes])
        indices, coords = locate_constraints(geometry, wp.array(points, dtype=wp.vec3d))
        indices, coords = indices.numpy(), coords.numpy()
    lower = origins / 4.0
    widths = 2.0 ** levels.astype(int) / 4.0
    inside = np.all((points[:, None] >= lower) & (points[:, None] < lower + widths[:, None]), axis=2)
    np.testing.assert_array_equal(inside.sum(axis=1), 1)
    np.testing.assert_array_equal(indices, inside.argmax(axis=1))
    np.testing.assert_allclose(lower[indices] + widths[indices, None] * coords, points, atol=1e-14, rtol=0)
