import unittest
from unittest import mock
import warp as wp
from warp._src.geometry.poisson import meshing_bounds
from warp.tests.geometry.test_poisson_reconstruction import TestPoissonReconstruction

# Simulate the erroneous coarse-corner certificate that misses finer support.
original = meshing_bounds.candidate_cells
def coarse_corners_only(field, depth, iso, *, statistics=None):
    points = wp.array([[x,y,z] for x in (0,1) for y in (0,1) for z in (0,1)], dtype=wp.vec3, device=field.coefficients.device)
    centered = field.evaluate(points)[0].numpy() - iso.numpy()[0]
    if (centered >= 0).all() or (centered < 0).all():
        return wp.empty(0, dtype=wp.vec3i, device=field.coefficients.device)
    return original(field, depth, iso, statistics=statistics)

with mock.patch.object(meshing_bounds, 'candidate_cells', coarse_corners_only):
    suite = unittest.TestSuite([TestPoissonReconstruction('test_mesh_certificates_cuda_0')])
    outcome = unittest.TextTestRunner(verbosity=1).run(suite)
if len(outcome.failures) != 1 or outcome.errors or '0 not greater than 0' not in outcome.failures[0][1]:
    raise RuntimeError('The adversarial certificate regression did not fail as expected')
print('Expected regression failure confirmed: coarse corners miss the interior fine feature.')
