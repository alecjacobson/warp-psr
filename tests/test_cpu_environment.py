from tools.cpu_environment import cpu_environment, available_threads


def test_parent_quota_limits_unlimited_child(tmp_path):
    root = tmp_path/'cgroups'
    (root/'parent'/'child').mkdir(parents=True)
    (root/'parent'/'child'/'cpu.max').write_text('max 100000\n')
    (root/'parent'/'cpu.max').write_text('400000 100000\n')
    (root/'cpu.max').write_text('800000 100000\n')
    membership = tmp_path/'membership'
    membership.write_text('0::/parent/child\n')
    actual = cpu_environment(root, membership)
    assert actual['minimum_visible_cpu_quota'] == 4
    assert len(actual['visible_cgroup_quotas']) == 3
    assert available_threads({**actual, 'affinity_cpus': list(range(128)), 'logical_cpus': 128}) == 4
    assert available_threads({**actual, 'affinity_cpus': [0, 1], 'logical_cpus': 128}) == 2


def test_missing_cgroups_are_unknown(tmp_path):
    actual = cpu_environment(tmp_path/'missing', tmp_path/'missing-membership')
    assert actual['minimum_visible_cpu_quota'] is None
    assert actual['visible_cgroup_quotas'] == []
