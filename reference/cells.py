"""Offline canonical integration partition derived from exported reference supports."""
import numpy as np


def refinement_sets(nodes,normals,depth):
    """Canonical refined-cell coordinates; no dense maximum-depth allocation."""
    required=[set() for _ in range(depth+1)]
    for row in nodes:
        d=int(row[0])
        if d<0 or not (int(row[4])&2):continue
        x,y,z=map(int,row[1:4]);n=1<<d
        for a in range(2):
            for b in range(2):
                for c in range(2):
                    p=(x-a,y-b,z-c)
                    if min(p)>=0 and max(p)<n:required[d].add(p)
    for row in normals:
        d,x,y,z=map(int,row[:4])
        if d<0:continue
        n=1<<d
        for a in range(-1,2):
            for b in range(-1,2):
                for c in range(-1,2):
                    p=(x+a,y+b,z+c)
                    if min(p)>=0 and max(p)<n:required[d].add(p)
    refined=[set() for _ in range(depth+1)]
    for d in range(depth-1,-1,-1):
        refined[d]={(x//2,y//2,z//2) for x,y,z in required[d+1]}
        required[d].update(refined[d])
    return refined


def widths_from_refinement(refined,cube_queries):
    widths=np.ones(len(cube_queries))
    for d in range(len(refined)-1):
        keys=np.array(sorted((x<<32)|(y<<16)|z for x,y,z in refined[d]),dtype=np.int64)
        if not len(keys):continue
        coord=np.minimum(np.floor(cube_queries*(1<<d)).astype(np.int64),(1<<d)-1)
        q=(coord[:,0]<<32)|(coord[:,1]<<16)|coord[:,2]
        index=np.searchsorted(keys,q)
        valid=index<len(keys)
        valid[valid]&=keys[index[valid]]==q[valid]
        widths[valid]=1.0/(1<<(d+1))
    return widths


def query_cell_widths(nodes,normals,cube_queries,depth):
    return widths_from_refinement(refinement_sets(nodes,normals,depth),cube_queries)
