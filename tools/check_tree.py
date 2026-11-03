"""Compare exact canonical active and non-ghost support keys, outside production."""
import hashlib
import numpy as np


def check_tree(tree, nodes):
    reports = []
    for d, level in enumerate(tree.levels):
        selected = nodes[(nodes[:, 0] == d) & (nodes[:, 5] == 0), 1:4].astype(np.int64)
        x, y, z = (selected + 524288).T
        expected_support = np.sort((x << 40) | (y << 20) | z)
        np.testing.assert_array_equal(level.keys.numpy(), expected_support,
                                      err_msg=f'Non-ghost support keys differ at depth {d}')
        selected = nodes[(nodes[:, 0] == d) & ((nodes[:, 4].astype(np.int64) & 2) != 0), 1:4].astype(np.int64)
        x, y, z = selected.T
        expected_active = np.sort((d << 48) | (x << 32) | (y << 16) | z)
        np.testing.assert_array_equal(tree.basis[d].keys.numpy(), expected_active,
                                      err_msg=f'Active basis keys differ at depth {d}')
        reports.append({'depth': d, 'support_count': len(expected_support), 'active_count': len(expected_active),
                        'support_sha256': hashlib.sha256(expected_support.astype('<i8').tobytes()).hexdigest(),
                        'active_sha256': hashlib.sha256(expected_active.astype('<i8').tobytes()).hexdigest()})
    return {'exact_active_and_support_keys': True, 'levels': reports}
