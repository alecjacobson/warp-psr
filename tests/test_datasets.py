import hashlib
import json
from types import SimpleNamespace
import numpy as np
import pytest
from tools.datasets import load_input
from tools.benchmark_input import benchmark_input, reconstruction_parameters


def test_source_normals_and_reproducible_subsets(tmp_path):
    # Non-unit and zero normals must reach reconstruction unchanged.
    data = np.arange(600, dtype=np.float64).reshape(100, 6)
    data[3, 3:] = 0
    path = tmp_path/'points.npts'
    np.savetxt(path, data, fmt='%.17g')
    p, n, metadata = load_input(path)
    np.testing.assert_array_equal(np.column_stack([p, n]), data)
    assert metadata['original_count'] == metadata['selected_count'] == 100
    p, n, metadata = load_input(path, max_points=15, seed=42)
    indices = np.sort(np.random.default_rng(42).choice(100, 15, replace=False))
    np.testing.assert_array_equal(np.column_stack([p, n]), data[indices])
    assert metadata['selection_indices_sha256'] == hashlib.sha256(indices.astype('<i8').tobytes()).hexdigest()
    with pytest.raises(ValueError, match='positive'):
        load_input(path, max_points=0)


def test_ply_attributes_and_required_normals(tmp_path):
    plyfile = pytest.importorskip('plyfile')
    dtype = [(name, 'f4') for name in ('x', 'y', 'z', 'nx', 'ny', 'nz')] + [('red', 'u1')]
    data = np.zeros(4, dtype=dtype)
    data['x'] = [1, 2, 3, 4]
    data['nz'] = [0, 2, -3, 1]
    path = tmp_path/'cloud.ply'
    plyfile.PlyData([plyfile.PlyElement.describe(data, 'vertex')]).write(path)
    p, n, metadata = load_input(path)
    np.testing.assert_array_equal(p[:, 0], data['x'])
    np.testing.assert_array_equal(n[:, 2], data['nz'])
    assert metadata['ignored_properties'] == ['red']
    data = np.zeros(4, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
    plyfile.PlyData([plyfile.PlyElement.describe(data, 'vertex')]).write(path)
    with pytest.raises(ValueError, match='nx'):
        load_input(path)


def test_benchmark_uses_fixture_parameters_and_rejects_changed_input(tmp_path):
    p, n, parameters, _ = benchmark_input(SimpleNamespace(fixture=None, count=10, depth=5, full_depth=2))
    path = tmp_path/'input.pts'
    np.savetxt(path, np.column_stack([p, n]), fmt='%.17g')
    parameters.update(weight=0., samples_per_node=3., base_depth=1, shape='test')
    manifest = {'name': 'test', 'parameters': parameters, 'precision': 'float',
                'input_sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    (tmp_path/'manifest.json').write_text(json.dumps(manifest))
    # Contradictory synthetic defaults must not override fixture parameters.
    actual_p, actual_n, actual_parameters, _ = benchmark_input(SimpleNamespace(fixture=tmp_path, depth=9, full_depth=4))
    np.testing.assert_array_equal(actual_p, p)
    np.testing.assert_array_equal(actual_n, n)
    assert reconstruction_parameters(actual_parameters)['point_weight'] == 0.
    assert actual_parameters == parameters
    path.write_text(path.read_text()+'0 0 0 0 0 1\n')
    with pytest.raises(ValueError, match='checksum'):
        benchmark_input(SimpleNamespace(fixture=tmp_path))


def test_mesh_normal_policy_preserves_winding_and_zero_vertices(tmp_path,monkeypatch):
    from plyfile import PlyData,PlyElement
    from tools import datasets
    points=np.array([[0,0,0],[2,0,0],[0,1,0],[0,0,1],[7,7,7]],dtype=float)
    faces=np.array([[0,1,2],[0,3,1],[0,0,1]])
    expected=np.array([[0,1,1],[0,1,1],[0,0,1],[0,1,0],[0,0,0]],dtype=float)
    expected[:2]/=np.sqrt(2)
    normals,policy=datasets.mesh_vertex_normals(points,faces)
    np.testing.assert_allclose(normals,expected,atol=1e-15)
    reverse,_=datasets.mesh_vertex_normals(points,faces[:,::-1])
    np.testing.assert_allclose(reverse,-expected,atol=1e-15)
    assert policy['zero_vertex_normals']==1 and policy['degenerate_triangles']==1
    vertices=np.empty(len(points),dtype=[('x','f4'),('y','f4'),('z','f4')])
    for axis,name in enumerate(['x','y','z']):vertices[name]=points[:,axis]
    triangles=np.empty(len(faces),dtype=[('vertex_indices','i4',(3,))]);triangles['vertex_indices']=faces
    path=tmp_path/'mesh.ply';PlyData([PlyElement.describe(vertices,'vertex'),PlyElement.describe(triangles,'face')]).write(path)
    monkeypatch.setitem(datasets.DATASETS,'tiny',dict(file=path.name,count=5,faces=3,normal_source='triangle_winding',
                         sha256=datasets.checksum(path),description='test mesh'))
    p,n,metadata=load_input(path,dataset='tiny',max_points=3,seed=2)
    indices=np.sort(np.random.default_rng(2).choice(5,3,replace=False))
    np.testing.assert_array_equal(p,points[indices]);np.testing.assert_allclose(n,expected[indices],atol=1e-15)
    assert metadata['input_kind']=='reconstructed_mesh_vertices'
    assert metadata['normal_policy']['triangles']==3
    with pytest.raises(ValueError,match='nx'):
        load_input(path)  # Mesh normal derivation is never implicit for arbitrary PLYs.
    with pytest.raises(ValueError,match='invalid indices'):
        datasets.mesh_vertex_normals(points,np.array([[0,1,8]]))
    with pytest.raises(ValueError,match='triangular'):
        datasets.mesh_vertex_normals(points,np.array([[0,1,2,3]]))


@pytest.mark.parametrize('kind',['tar','gzip','zip'])
def test_pinned_archive_extracts_only_the_requested_member(tmp_path,monkeypatch,kind):
    import gzip,io,tarfile,zipfile
    from tools import datasets
    data=b'pinned source bytes\n';member='nested/mesh.ply'
    buffer=io.BytesIO()
    if kind=='gzip':payload=gzip.compress(data)
    elif kind=='tar':
        with tarfile.open(fileobj=buffer,mode='w:gz') as archive:
            for name in [member,'../unrelated']:
                info=tarfile.TarInfo(name);info.size=len(data);archive.addfile(info,io.BytesIO(data))
        payload=buffer.getvalue()
    else:
        with zipfile.ZipFile(buffer,'w') as archive:
            archive.writestr(member,data);archive.writestr('../unrelated',data)
        payload=buffer.getvalue()
    entry=dict(file='mesh.ply',archive='data.'+kind,compression=kind,member=member,url='https://example.invalid/data',
               archive_sha256=hashlib.sha256(payload).hexdigest(),sha256=hashlib.sha256(data).hexdigest())
    monkeypatch.setitem(datasets.DATASETS,'tiny',entry)
    monkeypatch.setattr(datasets,'urlopen',lambda url,timeout:io.BytesIO(payload))
    target=datasets.download('tiny',tmp_path/'raw');assert target.read_bytes()==data
    assert not (tmp_path/'unrelated').exists()
    target.write_bytes(b'changed')
    with pytest.raises(ValueError,match='checksum'):datasets.download('tiny',tmp_path/'raw')
    monkeypatch.setitem(entry,'archive_sha256','0'*64)
    with pytest.raises(ValueError,match='checksum'):datasets.download('tiny',tmp_path/'bad')
    assert not (tmp_path/'bad'/'mesh.ply').exists()


@pytest.mark.parametrize('variant',['position-noise','normal-noise','missing-cap','outliers'])
def test_input_defects_are_reproducible_and_explicit(variant):
    from tools.point_variants import apply_variant
    rng=np.random.default_rng(52)
    points=rng.uniform(-1,1,(1000,3));normals=np.tile([0.,0.,1.],(1000,1));normals[0]=0
    p0,n0=points.copy(),normals.copy()
    p,n,metadata=apply_variant(points,normals,variant,seed=2026)
    pp,nn,again=apply_variant(points,normals,variant,seed=2026)
    np.testing.assert_array_equal(p,pp);np.testing.assert_array_equal(n,nn);assert metadata==again
    np.testing.assert_array_equal(points,p0);np.testing.assert_array_equal(normals,n0)
    assert metadata['output_count']==len(p) and np.isfinite(p).all() and np.isfinite(n).all()
    if variant=='position-noise':
        np.testing.assert_array_equal(n,n0);assert not np.array_equal(p,p0)
        np.testing.assert_allclose(metadata['sigma_source_units'],.002*np.linalg.norm(np.ptp(points,axis=0)))
    elif variant=='normal-noise':
        np.testing.assert_array_equal(p,p0);np.testing.assert_array_equal(n[0],[0,0,0])
        np.testing.assert_allclose(np.linalg.norm(n[1:],axis=1),1.,atol=1e-14)
        assert not np.array_equal(n[1:],n0[1:])
    elif variant=='missing-cap':
        mask=points[:,2]<=np.quantile(points[:,2],.85)
        np.testing.assert_array_equal(p,points[mask]);np.testing.assert_array_equal(n,normals[mask])
    else:
        assert len(p)==1010 and metadata['appended_count']==10
        np.testing.assert_array_equal(p[:1000],p0);np.testing.assert_array_equal(n[:1000],n0)
        np.testing.assert_allclose(np.linalg.norm(n[1000:],axis=1),1.,atol=1e-14)


def test_case_catalog_preserves_original_profiles_and_balances_new_modes():
    from pathlib import Path
    from tools.check_datasets import cases
    root=Path(__file__).resolve().parents[1]
    for suite,filename in [('smoke','public-smoke.json'),('scaling','public-scaling-accuracy.json')]:
        golden=json.loads((root/'docs/results'/filename).read_text())['profiles']
        actual=cases(suite)
        assert [item[0] for item in actual]==[item['fixture'] for item in golden]
        for (name,dataset,options),record in zip(actual,golden):
            parameters=record['manifest']['parameters'];assert parameters['dataset']==dataset
            for option,value in zip(options[::2],options[1::2]):
                original=parameters[option[2:].replace('-','_')]
                assert type(original)(value)==original
    extended,stress=cases('extended'),cases('stress')
    assert len(extended)==24 and len(stress)==16
    for subset in [extended,stress]:
        weights=[dict(zip(options[::2],options[1::2]))['--weight'] for _,_,options in subset]
        assert weights.count('0')==weights.count('2')
    assert len(cases('all'))==len({name for name,_,_ in cases('all')})==71
    with pytest.raises(ValueError,match='Unknown'):cases('typo')
