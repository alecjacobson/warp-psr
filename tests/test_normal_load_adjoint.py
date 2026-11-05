"""Multilevel assembly VJPs including the cancellation-sensitive ordered root."""
from types import SimpleNamespace as NS
import numpy as np
import pytest
import warp as wp
from warp_psr.keyed import Records
from warp_psr.normal_loads import assemble_reference_normal_loads
from warp_psr.normal_load_adjoint import replay_normal_loads, prepare_normal_restrictions
from warp_psr.solver import prolongation
from test_sample_adjoint import device


def fixture(device, dtype, adaptive):
    rng = np.random.default_rng(328)
    vector = wp.vec4d if dtype == wp.float64 else wp.vec4
    average_type = wp.vec3d if dtype == wp.float64 else wp.vec3
    basis, levels, all_keys, all_values = [], [], [], []
    for d in range(4):
        xyz = np.indices(((1 << d) + 1,) * 3).reshape(3, -1).T.astype(np.int64)
        if adaptive and d > 1:
            xyz = xyz[(xyz[:, 0] < (1 << (d - 1)) + 1) | (xyz[:, 1] < 2)]
        key = (d << 48) | (xyz[:, 0] << 32) | (xyz[:, 1] << 16) | xyz[:, 2]
        basis.append(NS(keys=wp.array(key, dtype=wp.int64, device=device)))
        normal_keys = key[(xyz < 1 << d).all(axis=1)]
        if adaptive:
            normal_keys = normal_keys[rng.random(len(normal_keys)) > .4]
        all_keys.append(normal_keys)
        values = rng.normal(size=(len(normal_keys), 4))
        values[:, 3] = 0
        all_values.append(values)
    records = Records(wp.array(np.concatenate(all_keys), dtype=wp.int64, device=device),
                      wp.array(np.concatenate(all_values), dtype=vector, device=device, requires_grad=True))
    begin = 0
    for keys in all_keys:
        end = begin + len(keys)
        if len(keys):
            levels.append(Records(records.keys[begin:end], records.values[begin:end]))
        else:
            levels.append(Records(wp.empty(0, dtype=wp.int64, device=device), wp.empty(0, dtype=vector, device=device, requires_grad=True)))
        begin = end
    normals = NS(records=records, levels=levels, average=wp.zeros(1, dtype=average_type, device=device))
    transfers = [prolongation(basis[d - 1].keys, basis[d].keys, d, dtype) for d in range(1, 4)]
    return NS(basis=basis), normals, transfers


@pytest.mark.parametrize('dtype', [wp.float32, wp.float64])
@pytest.mark.parametrize('base', [0, 1, 2])
@pytest.mark.parametrize('adaptive', [False, True])
def test_multilevel_forward_and_normal_vjp(device, dtype, base, adaptive):
    tree, normals, transfers = fixture(device, dtype, adaptive)
    expected = assemble_reference_normal_loads(tree, None, normals, 3, base, transfers)
    restrictions = prepare_normal_restrictions(transfers)
    with wp.Tape() as tape:
        actual = replay_normal_loads(tree, normals, 3, base, transfers, restrictions=restrictions)
    for a, e in zip(actual[base:], expected[base:]):
        np.testing.assert_array_equal(a.numpy(), e.numpy())
    rng = np.random.default_rng(129)
    seeds = [rng.normal(size=len(a)) for a in actual[base:]]
    tape.backward(grads={a: wp.array(s, dtype=a.dtype, device=device) for a, s in zip(actual[base:], seeds)})
    source = normals.records.values
    original, gradient = source.numpy().copy(), source.grad.numpy().copy()
    direction = rng.normal(size=original.shape).astype(original.dtype)
    epsilon = 1e-3 if dtype == wp.float32 else 1e-6
    losses = []
    for sign in [1, -1]:
        source.assign(original + sign * epsilon * direction)
        values = assemble_reference_normal_loads(tree, None, normals, 3, base, transfers)
        losses.append(sum(np.sum(a.numpy() * s) for a, s in zip(values[base:], seeds)))
    source.assign(original)
    assert np.isfinite(gradient).all()
    np.testing.assert_allclose(np.sum(gradient.astype(float) * direction), (losses[0] - losses[1]) / (2 * epsilon),
                               rtol=2e-3 if dtype == wp.float32 else 2e-7, atol=1e-4 if dtype == wp.float32 else 2e-8)
