"""Deterministic offline coverage queries derived from reference support cells."""
import itertools
import numpy as np
from scipy.spatial import cKDTree
from reference.cells import refinement_sets, widths_from_refinement


def sample_leaf_cells(refined, seed, per_depth=64):
    """Bound the query count while covering every occupied integration depth."""
    rng = np.random.default_rng(seed)
    selected = []
    corners = list(itertools.product(range(2), repeat=3))
    if not refined[0]:
        return [(0, (0, 0, 0))]
    for d in range(1, len(refined)):
        parents = sorted(refined[d-1])
        leaf_count = 8 * len(parents) - len(refined[d])
        target = min(per_depth, leaf_count)
        if target <= 0:
            continue
        leaves = set()
        for _ in range(4096):
            p = parents[rng.integers(len(parents))]
            c = corners[rng.integers(8)]
            key = tuple(2*a+b for a,b in zip(p,c))
            if key not in refined[d]:
                leaves.add(key)
            if len(leaves) == target:
                break
        # Rare isolated leaves must still be covered if random probing misses them.
        if len(leaves) < target:
            for p in parents:
                for c in corners:
                    key = tuple(2*a+b for a,b in zip(p,c))
                    if key not in refined[d]:
                        leaves.add(key)
                    if len(leaves) == target:
                        break
                if len(leaves) == target:
                    break
        selected.extend((d,c) for c in sorted(leaves))
    return selected


def coverage_queries(nodes, normals, cube_samples, depth, precision, seed):
    """Return named cube-coordinate families and measured coverage metadata."""
    refined = refinement_sets(nodes, normals, depth)
    leaves = sample_leaf_cells(refined, seed)
    widths = np.array([2.0**-d for d,_ in leaves])
    centers = np.array([(np.asarray(c)+0.5)*h for (_,c),h in zip(leaves,widths)])
    double = precision == 'double'
    epsilon = np.maximum(widths*(2.0**(-30 if double else -10)), 2.0**(-40 if double else -20))
    left, right = [], []
    for axis in range(3):
        for direction in (-1,1):
            face = centers.copy()
            face[:,axis] += direction * widths/2
            a,b = face.copy(),face.copy()
            a[:,axis] -= epsilon
            b[:,axis] += epsilon
            interior = (np.min(a,axis=1)>0) & (np.max(b,axis=1)<1)
            a,b = a[interior],b[interior]
            different = widths_from_refinement(refined,a) != widths_from_refinement(refined,b)
            left.extend(a[different]); right.extend(b[different])
    families = {'cell_interiors':centers}
    if left:
        families['interface_minus'] = np.asarray(left)
        families['interface_plus'] = np.asarray(right)
    rng = np.random.default_rng(seed+1)
    boundary = []
    margin = 1e-10 if double else 1e-5
    for axis in range(3):
        for side in (margin,1-margin):
            q = rng.uniform(0.05,0.95,(32,3))
            q[:,axis] = side
            boundary.extend(q)
    families['cube_boundary'] = np.asarray(boundary)
    candidates = rng.uniform(0.0001,0.9999,(2048,3))
    distance,_ = cKDTree(cube_samples).query(candidates)
    farthest = np.argsort(distance)[-128:]
    families['far_from_samples'] = candidates[farthest]
    return families, {
        'cell_interiors_by_depth': {str(d):sum(k==d for k,_ in leaves) for d in sorted({d for d,_ in leaves})},
        'refinement_interface_pairs':len(left),
        'boundary_distance_in_cube':margin,
        'far_from_samples_min_distance_in_cube':float(distance[farthest].min()),
    }
