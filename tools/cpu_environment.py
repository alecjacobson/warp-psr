"""Visible CPU affinity and cgroup-v2 limits for honest benchmark provenance."""
import datetime
import os
from pathlib import Path


def cpu_environment(root=Path('/sys/fs/cgroup'), membership=Path('/proc/self/cgroup')):
    quotas = []
    counters = []
    try:
        group = next(line[3:] for line in membership.read_text().splitlines() if line.startswith('0::'))
        current = root/group.lstrip('/')
        while current.is_relative_to(root):
            path = current/'cpu.max'
            if path.exists():
                quota, period = path.read_text().split()
                quotas.append({'group': str(current.relative_to(root)), 'cpu_max': f'{quota} {period}',
                               'cpu_equivalents': None if quota == 'max' else int(quota)/int(period)})
            path = current/'cpu.stat'
            if path.exists():
                counters.append({'group': str(current.relative_to(root)),
                                 'counters': {key: int(value) for key, value in (line.split() for line in path.read_text().splitlines())}})
            if current == root:
                break
            current = current.parent
    except (OSError, StopIteration, ValueError):
        pass
    affinity = sorted(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else None
    finite = [q['cpu_equivalents'] for q in quotas if q['cpu_equivalents'] is not None]
    return {'captured_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'logical_cpus': os.cpu_count(), 'affinity_cpus': affinity,
            'visible_cgroup_quotas': quotas, 'minimum_visible_cpu_quota': min(finite) if finite else None,
            'cgroup_cpu_counters': counters,
            'limit_note': 'Visible cgroup-v2 ancestors only; unavailable limits are unknown, not evidence of an unrestricted host.'}


def available_threads(environment):
    limits = [environment['logical_cpus'] or 1]
    if environment['affinity_cpus']:
        limits.append(len(environment['affinity_cpus']))
    if environment['minimum_visible_cpu_quota'] is not None:
        limits.append(max(1, int(environment['minimum_visible_cpu_quota'])))
    return min(limits)
