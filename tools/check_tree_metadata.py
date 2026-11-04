"""Check complete GPU topology against unique canonical reference identities."""
import numpy as np


def coordinates(keys):
    return np.column_stack([((keys >> shift) & 1048575) - 524288 for shift in (40, 20, 0)])


def check_tree_metadata(metadata, reference):
    reference = np.asarray(reference, dtype=np.int64)
    assert len(np.unique(reference[:, :4], axis=0)) == len(reference), 'Duplicate global reference identities'
    np.testing.assert_array_equal(reference[:, 0] - reference[:, 4], metadata.depth_offset)
    xyz = [coordinates(level.keys.numpy()) for level in metadata.levels]
    levels = []
    for g, level in enumerate(metadata.levels):
        rows = reference[reference[:, 0] == g]
        rows = rows[np.lexsort(rows[:, 1:4].T[::-1])]
        np.testing.assert_array_equal(xyz[g], rows[:, 1:4])
        np.testing.assert_array_equal(rows[:, 4], level.local_depth)
        np.testing.assert_array_equal(level.flags.numpy(), rows[:, 8])
        np.testing.assert_array_equal(level.ghost.numpy(), rows[:, 9])
        parents, children = level.parents.numpy(), level.children.numpy()
        if g:
            assert np.all(parents >= 0)
            np.testing.assert_array_equal(rows[:, 10], g - 1)
            np.testing.assert_array_equal(xyz[g - 1][parents], rows[:, 11:14])
        else:
            np.testing.assert_array_equal(parents, -1)
            np.testing.assert_array_equal(rows[:, 10:14], -1)
        np.testing.assert_array_equal((children >= 0).sum(axis=1), rows[:, 14])
        for k in range(8):
            present = children[:, k] >= 0
            if present.any():
                np.testing.assert_array_equal(xyz[g + 1][children[present, k]],
                    2 * xyz[g][present] + [k & 1, (k >> 1) & 1, (k >> 2) & 1])
                np.testing.assert_array_equal(metadata.levels[g + 1].parents.numpy()[children[present, k]],
                                              np.flatnonzero(present))
        levels.append({'local_depth': level.local_depth, 'nodes': len(rows),
                       'ghosts': int(np.count_nonzero(rows[:, 9]))})
    assert sum(item['nodes'] for item in levels) == len(reference)
    return {'exact_complete_keys_flags_and_links': True, 'depth_offset': metadata.depth_offset,
            'nodes': len(reference), 'levels': levels}
