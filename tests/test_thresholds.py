import numpy as np
import pytest

from tools.check_thresholds import ownership_input


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_boundary_probes_survive_input_quantization(dtype):
    data, probes = ownership_input(dtype)
    # Reproduce the exact scale-2 affine map, including reference axis ordering.
    cube = data[:, [1, 2, 0]] * dtype(.5) + dtype(.5)
    for probe in probes:
        actual = cube[probe['input_index']]
        np.testing.assert_array_equal(actual, np.array(probe['cube'], dtype=dtype))
        value = actual[probe['axis']]
        split = dtype(probe['split'])
        if probe['side'] == 'below':
            assert value == np.nextafter(split, dtype(-np.inf))
        elif probe['side'] == 'above':
            assert value == np.nextafter(split, dtype(np.inf))
        else:
            assert value == split
