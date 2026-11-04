"""Compare input classifications and aggregate membership with read-only traces."""
import numpy as np


def check_samples(samples, reference):
    reference = np.asarray(reference, dtype=np.int64)
    np.testing.assert_array_equal(reference[:, 0], np.arange(len(reference)))
    status = samples.input_status.numpy()
    np.testing.assert_array_equal(status, reference[:, 1])
    accepted = status == 0
    groups = samples.input_group_indices.numpy()
    np.testing.assert_array_equal(groups[~accepted], -1)
    expected_keys = ((reference[accepted, 2] << 48) | (reference[accepted, 3] << 32)
                     | (reference[accepted, 4] << 16) | reference[accepted, 5])
    keys = samples.keys.numpy()
    assert np.all((groups[accepted] >= 0) & (groups[accepted] < len(keys)))
    np.testing.assert_array_equal(keys[groups[accepted]], expected_keys)
    per_input = np.full(len(reference), np.iinfo(np.int64).max, dtype=np.int64)
    per_input[accepted] = expected_keys
    order = np.argsort(per_input, kind='stable')
    np.testing.assert_array_equal(samples.input_order.numpy(), order)
    counts = np.bincount(groups[accepted], minlength=len(keys))
    offsets = np.r_[0, np.cumsum(counts)]
    np.testing.assert_array_equal(samples.group_offsets.numpy(), offsets)
    np.testing.assert_array_equal(samples.weights.numpy(), counts)
    np.testing.assert_array_equal(samples.first_indices.numpy(), order[offsets[:-1]])
    return {'exact_input_status_and_membership': True, 'input_count': len(reference),
            'accepted': int(accepted.sum()), 'aggregates': len(keys),
            'rejected_by_status': {str(code): int(np.count_nonzero(status == code))
                                   for code in np.unique(status[~accepted])}}
