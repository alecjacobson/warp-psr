"""Untimed full-grid coverage proof and matched-grid CPU comparison."""
from pathlib import Path
import argparse
import json,time,gc
import numpy as np
from scipy.spatial import cKDTree
from skimage.measure import marching_cubes
import warp as wp
from warp_psr import reconstruct
import warp_psr.meshing as m
from tools.benchmark_input import benchmark_input,reconstruction_parameters
from tools.fixture_provenance import production_hashes


@wp.kernel
def count_crossing(grid: wp.array3d(dtype=float), counts: wp.array(dtype=int)):
    i,j,k=wp.tid()
    code=int(0)
    for c in range(8):
        q=wp.vec3i(i,j,k)+m._corner(c)
        if grid[q[0],q[1],q[2]]>=0.0:code=code|(1<<c)
    if code!=0 and code!=255:wp.atomic_add(counts,0,1)


@wp.kernel
def compare_corners(grid: wp.array3d(dtype=float), cells: wp.array(dtype=wp.vec3i),
                    values: wp.array2d(dtype=float), counts: wp.array(dtype=int)):
    i,c=wp.tid()
    q=cells[i]+m._corner(c)
    if grid[q[0],q[1],q[2]]!=values[i,c]:wp.atomic_add(counts,1,1)


def canon(indices):
    f=indices.reshape(-1,3).copy()
    for step in [1,2]:
        g=np.roll(indices.reshape(-1,3),-step,axis=1)
        lower=(g[:,0]<f[:,0])|((g[:,0]==f[:,0])&(g[:,1]<f[:,1]))|((g[:,0]==f[:,0])&(g[:,1]==f[:,1])&(g[:,2]<f[:,2]))
        f[lower]=g[lower]
    return f[np.lexsort(f.T[::-1])]


def topology(vertices, indices, bounds, tolerance):
    faces = indices.reshape(-1, 3)
    edges = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    edges.sort(axis=1)
    edges, counts = np.unique(edges, axis=0, return_counts=True)
    boundary = vertices[edges[counts == 1]]
    on_box = np.zeros(len(boundary), dtype=bool)
    for corner in bounds:
        on_box |= np.any(np.all(np.abs(boundary - corner) <= tolerance, axis=1), axis=1)
    triangles = vertices[faces]
    area = np.linalg.norm(np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]), axis=1)
    return {'boundary_edges': int(np.count_nonzero(counts == 1)),
            'boundary_edges_away_from_box': int(np.count_nonzero(~on_box)),
            'edges_with_more_than_two_faces': int(np.count_nonzero(counts > 2)),
            'zero_area_triangles': int(np.count_nonzero(area == 0))}


def check(args):
    resolution=args.resolution
    p,n,params,provenance=benchmark_input(args)
    wp.init()
    capture={}
    with wp.ScopedDevice('cuda'):
        result=reconstruct(wp.array(p,dtype=wp.vec3),wp.array(n,dtype=wp.vec3),**reconstruction_parameters(params))
        inverse=np.linalg.inv(np.array(result.field.linear).reshape(3,3))
        cube=np.array([(x,y,z) for x in (0.001,0.999) for y in (0.001,0.999) for z in (0.001,0.999)])
        model=(cube-result.field.translation)@inverse.T
        bounds=(model.min(axis=0).tolist(),model.max(axis=0).tolist())
        dense=m.IsoSurfaceMarchingCubes.extract
        descriptor=m.IsoSurfaceMarchingCubes.__dict__['extract']
        sparse=m.sparse_marching_cubes_from_cells
        def capture_dense(cls,field,**kw):
            capture['grid']=field
            return dense(field,**kw)
        def capture_sparse(cells,corner_values,**kw):
            capture['cells']=cells;capture['corners']=corner_values
            return sparse(cells,corner_values,**kw)
        try:
            m.IsoSurfaceMarchingCubes.extract=classmethod(capture_dense)
            m.sparse_marching_cubes_from_cells=capture_sparse
            a=m.extract_mesh(result,resolution=resolution,bounds=bounds,backend='dense')
            b=m.extract_mesh(result,resolution=resolution,bounds=bounds,backend='sparse')
        finally:
            m.IsoSurfaceMarchingCubes.extract=descriptor
            m.sparse_marching_cubes_from_cells=sparse
        counters=wp.zeros(2,dtype=int)
        wp.launch(count_crossing,(resolution-1,)*3,inputs=[capture['grid'],counters])
        wp.launch(compare_corners,(len(capture['cells']),8),inputs=[capture['grid'],capture['cells'],capture['corners'],counters])
        count,mismatched=map(int,counters.numpy())
        assert count==len(capture['cells']) and mismatched==0
        cells=capture['cells'].numpy().astype(np.int64)
        codes=(cells[:,0]*(resolution-1)+cells[:,1])*(resolution-1)+cells[:,2]
        assert np.all(np.diff(codes)>0)
        va,fa=a.vertices.numpy(),a.indices.numpy()
        vb,fb=b.vertices.numpy(),b.indices.numpy()
        assert len(va)==len(vb) and len(fa)==len(fb)
        tree=cKDTree(va)
        distance,match=tree.query(vb)
        assert distance.max()<args.geometry_tolerance
        # Exact-threshold/near-zero edges create coincident float32 vertices. Merge
        # reference positions within the measured mapping roundoff for a geometric
        # oriented-face comparison, retaining the raw vertex/face counts above.
        parent=np.arange(len(va))
        def find(x):
            while parent[x]!=x:
                parent[x]=parent[parent[x]];x=parent[x]
            return x
        pairs=tree.query_pairs(args.geometry_tolerance,output_type='ndarray')
        for x,y in pairs:
            x,y=find(x),find(y);parent[max(x,y)]=min(x,y)
        representatives=np.array([find(i) for i in range(len(va))])
        assert np.array_equal(canon(representatives[fa]),canon(representatives[match[fb]]))
        report={'production_source_sha256':production_hashes(),'input':provenance,'resolution':resolution,
                'dense_crossing_cells':count,'sparse_cells':len(cells),'sparse_cells_unique_and_sorted':True,
                'mismatched_corner_values':mismatched,'all_crossing_cells_and_values_match':True,
                'vertices':len(va),'triangles':len(fa)//3,'max_vertex_distance':float(distance.max()),
                'close_reference_vertex_pairs':len(pairs),'geometry_comparison_tolerance':args.geometry_tolerance,
                'oriented_geometric_triangles_match':True,
                'geometry_note':'Near-coincident float32 reference positions merged only for oriented geometric triangle comparison; raw counts unchanged. Exact cell coverage and all corner values checked independently.'}
        def residual(vertices):
            values,_,valid=result.evaluate(wp.array(vertices,dtype=wp.vec3,device='cuda:0'))
            assert valid.numpy().all()
            r=values.numpy().astype(np.float64)-result.iso_value
            return {'max':float(np.abs(r).max(initial=0)),'rms':float(np.sqrt(np.mean(r*r))) if len(r) else 0.0}
        report['dense_vertex_field_residual']=residual(va)
        report['sparse_vertex_field_residual']=residual(vb)
        report['dense_topology']=topology(va,fa,bounds,args.geometry_tolerance)
        report['sparse_topology']=topology(vb,fb,bounds,args.geometry_tolerance)
        grid=capture['grid'].numpy()

    # CPU extraction has no simultaneous GPU validation/benchmark workload. The
    # scalar field is already on the host; these timings exclude sampling/download.
    runs=[]
    for i in range(4):
        gc.collect();start=time.perf_counter()
        cv,cf,_,_=marching_cubes(grid,level=0.0,spacing=tuple((np.asarray(bounds[1])-bounds[0])/(resolution-1)))
        runs.append(time.perf_counter()-start)
    cv+=np.asarray(bounds[0])
    forward=cKDTree(va).query(cv)[0];reverse=cKDTree(cv).query(va)[0]
    report['cpu_skimage']={'topology':topology(cv,cf,bounds,args.geometry_tolerance),
                          'vertex_field_residual':residual(cv),'runs_seconds':runs,'warm_median_seconds':float(np.median(runs[1:])),
                          'scope':'CPU extraction only, from the same centered scalar grid; sampling and GPU-to-host transfer excluded.',
                          'vertices':len(cv),'triangles':len(cf),
                          'cpu_to_warp_max_vertex_distance':float(forward.max()),
                          'warp_to_cpu_max_vertex_distance':float(reverse.max()),
                          'ambiguity_policy':'scikit-image Lewiner; triangle counts/order need not equal Warp MC'}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ['production_source_sha256','input']},indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fixture',type=Path,required=True)
    parser.add_argument('--resolution',type=int,default=512)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--geometry-tolerance',type=float,default=2e-6,help='absolute model-space distance')
    args=parser.parse_args()
    if args.resolution<2: parser.error('resolution must be at least two')
    if args.geometry_tolerance<=0 or not np.isfinite(args.geometry_tolerance):
        parser.error('geometry tolerance must be finite and positive')
    check(args)
