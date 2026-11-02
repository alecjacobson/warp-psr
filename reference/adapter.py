"""Offline CPU oracle-tree import. Deliberately outside the production package."""
import numpy as np
import warp as wp
from warp_psr.field import LevelField
from tools.make_fixture import read_output


def load_field(path, dtype=wp.float32, device="cuda:0"):
    data = read_output(path)
    nodes = data["node"]
    # Skip ghost nodes and indices outside the degree-1 Neumann basis domain.
    selected = []
    for d,x,y,z,flags,ghost,idx,c in nodes:
        d,x,y,z = int(d),int(x),int(y),int(z)
        if d >= 0 and not ghost and idx >= 0 and 0<=x<=2**d and 0<=y<=2**d and 0<=z<=2**d:
            selected.append(((d<<48)|(x<<32)|(y<<16)|z, c))
    selected.sort()
    keys = np.array([k for k,c in selected], dtype=np.int64)
    if len(np.unique(keys)) != len(keys):
        raise ValueError("Duplicate canonical basis key")
    coeff = np.array([c for k,c in selected], dtype=np.float64 if dtype == wp.float64 else np.float32)
    transform = data["transform"].reshape(4,4)
    # PoissonRecon's XForm indexing is column first.
    inverse = np.linalg.inv(transform.T)
    linear = tuple(inverse[:3,:3].reshape(-1))
    translation = tuple(inverse[:3,3])
    return LevelField(wp.array(keys, dtype=wp.int64, device=device), wp.array(coeff, dtype=dtype, device=device),
                      linear, translation, int(max(nodes[:,0])), float(data["iso"][0,0]))
