"""Pinned offline oriented clouds and explicitly identified mesh-derived inputs."""
import argparse
from contextlib import ExitStack
import gzip
import hashlib
from pathlib import Path
import shutil
import tarfile
from urllib.request import urlopen
import zipfile
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
SOURCE='https://github.com/mkazhdan/PoissonRecon/blob/262b0f539d404057d1f36e1adc07fc9388678899/README.md'
BASE='https://www.cs.jhu.edu/~misha/Code/PoissonRecon/'
STANFORD='https://graphics.stanford.edu/data/3Dscanrep/'
STANFORD_FILES='https://graphics.stanford.edu/pub/3Dscanrep/'
DATASETS={
 'horse':{'file':'horse.npts','count':100000,'sha256':'e6fd6bfccccb1aad72c83144acd7543f672f1a59641071ef008ae6055df053ec',
          'description':'Curvature-weighted, nonuniform samples of a virtual horse; supplied by the PoissonRecon project'},
 'bunny':{'file':'bunny.points.ply','count':362271,'sha256':'859f3a8fda84bd2707abc5ede664b11cde20fef9d5f9ffde36cc52088a486e15',
          'description':'Merged Stanford Bunny range scans; normals estimated from scan connectivity by the PoissonRecon project'},
 'eagle':{'file':'eagle.points.ply','count':796825,'sha256':'468e41b3679f775b98fe8cc79132cb99e8d6dfc4dcc1bb684e571a90e326e478',
          'description':'EPFL Scanning 3D Statues from Photos; oriented eagle scan with color'},
 'torso':{'file':'torso.points.ply','count':3488432,'sha256':'ad7955bb71fe4a042e125b407beed7eeabb79a2062157b28ea955cb621e66524',
          'archive':'torso.zip','archive_sha256':'a9ee6b6af62921e2223b5cd47f0e80d3345ec93379a7b9b67131d3c1fcf6c25c',
          'description':'PoissonRecon torso scan; the separate envelope constraint is outside the supported profile'},
 'drill':{'file':'drill.ply','count':1961,'sha256':'ba5a1fc9fd9d5ec3a7458d5cc3d95989da555e8b03e13ef6120961f604243ba0',
          'archive':'drill.tar.gz','archive_sha256':'2da2acaba36903a9893c920c11506e1bd01c531bdf738d037bc505b313df5e1b',
          'url':STANFORD_FILES+'drill.tar.gz','member':'drill/reconstruction/drill_shaft_vrip.ply','compression':'tar',
          'normal_source':'triangle_winding','faces':3855,'source_page':STANFORD,
          'description':'Stanford drill reconstruction: thin, elongated geometry; mesh-derived input'},
 'armadillo':{'file':'Armadillo.ply','count':172974,'sha256':'483563fca686b66168e1e45339c50e71a5dc8a526a04a545a16b9237d5eaa023',
          'archive':'Armadillo.ply.gz','archive_sha256':'8b9b56cc36e66d54429b1e1e75bd89e833645bfe0dc7c1afd1205877a7356a3f',
          'url':STANFORD_FILES+'armadillo/Armadillo.ply.gz','compression':'gzip',
          'normal_source':'triangle_winding','faces':345944,'source_page':STANFORD,
          'description':'Stanford armadillo reconstruction; mesh-derived input'},
 'dragon':{'file':'dragon.ply','count':437645,'sha256':'fea87ff48f2aba22fb53e7b67c3ff3f7b8c2a3b3a0653af62c48bba67c6d5744',
          'archive':'dragon_recon.tar.gz','archive_sha256':'74ac1d90989c9b1732edee82d57e9ce71452144cf4355f108d8c9c616d28d02f',
          'url':STANFORD_FILES+'dragon/dragon_recon.tar.gz','member':'dragon_recon/dragon_vrip.ply','compression':'tar',
          'normal_source':'triangle_winding','faces':871414,'source_page':STANFORD,
          'description':'Stanford dragon reconstruction; mesh-derived input'},
 'buddha':{'file':'buddha.ply','count':543652,'sha256':'2283371216d748a08376a3c88698e283cc8f18d10ced348d6d133051bcf217ab',
          'archive':'happy_recon.tar.gz','archive_sha256':'409cd294efbfd8244e15a382b95a9423f153b7776e736c9b09f19ec9d3c10ed0',
          'url':STANFORD_FILES+'happy/happy_recon.tar.gz','member':'happy_recon/happy_vrip.ply','compression':'tar',
          'normal_source':'triangle_winding','faces':1087716,'source_page':STANFORD,
          'description':'Stanford Happy Buddha reconstruction; mesh-derived input'},
}


def checksum(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def download(name,directory=None):
    entry=DATASETS[name]
    directory=Path(directory) if directory else ROOT/'.build/datasets'
    directory.mkdir(parents=True,exist_ok=True)
    path=directory/entry['file']
    if not path.exists():
        filename=entry.get('archive',entry['file'])
        downloaded=directory/filename
        if not downloaded.exists():
            temporary=downloaded.with_suffix(downloaded.suffix+'.part')
            with urlopen(entry.get('url',BASE+filename),timeout=60) as source,temporary.open('wb') as dest:
                shutil.copyfileobj(source,dest)
            expected=entry.get('archive_sha256',entry['sha256'])
            if checksum(temporary)!=expected:raise ValueError(f'{name}: download checksum differs from the pinned dataset')
            temporary.replace(downloaded)
        if 'archive' in entry:
            if checksum(downloaded)!=entry['archive_sha256']:raise ValueError(f'{name}: archive checksum mismatch')
            temporary=path.with_suffix(path.suffix+'.part')
            with ExitStack() as stack:
                kind=entry.get('compression','zip')
                if kind=='zip':
                    archive=stack.enter_context(zipfile.ZipFile(downloaded))
                    source=stack.enter_context(archive.open(entry.get('member',entry['file'])))
                elif kind=='tar':
                    archive=stack.enter_context(tarfile.open(downloaded,'r:gz'))
                    member=archive.getmember(entry['member'])
                    if not member.isfile():raise ValueError('Pinned archive member must be a regular file')
                    source=stack.enter_context(archive.extractfile(member))
                elif kind=='gzip':
                    source=stack.enter_context(gzip.open(downloaded,'rb'))
                else:raise ValueError(f'Unsupported dataset compression: {kind}')
                dest=stack.enter_context(temporary.open('wb'))
                shutil.copyfileobj(source,dest)
            if checksum(temporary)!=entry['sha256']:raise ValueError(f'{name}: extracted point-cloud checksum mismatch')
            temporary.replace(path)
    if checksum(path)!=entry['sha256']:raise ValueError(f'{name}: point-cloud checksum mismatch')
    return path


def mesh_vertex_normals(points, faces):
    """Area-weighted vertex normals, preserving the source triangle winding.

    Float64 sums traverse corner 0/1/2, then stored face order. Zero accumulated
    normals stay zero; no vertex is dropped, welded, reoriented or resampled.
    """
    points=np.asarray(points,dtype=np.float64)
    faces=np.asarray(faces)
    if points.ndim!=2 or points.shape[1]!=3:raise ValueError('Mesh positions must be Nx3')
    if faces.ndim!=2 or faces.shape[1]!=3 or not np.issubdtype(faces.dtype,np.integer):
        raise ValueError('Normal derivation requires triangular integer faces')
    if np.any(faces<0) or np.any(faces>=len(points)) or not np.isfinite(points).all():
        raise ValueError('Mesh has invalid indices or nonfinite positions')
    area=np.cross(points[faces[:,1]]-points[faces[:,0]],points[faces[:,2]]-points[faces[:,0]])
    if not np.isfinite(area).all():raise ValueError('Triangle normal computation overflowed')
    normals=np.zeros_like(points)
    for corner in range(3):np.add.at(normals,faces[:,corner],area)
    length=np.linalg.norm(normals,axis=1)
    if not np.isfinite(length).all():raise ValueError('Vertex normal computation overflowed')
    np.divide(normals,length[:,None],out=normals,where=length[:,None]>0)
    return normals,{'algorithm':'area-weighted triangle-winding vertex normals v1; float64 corner-major face-order sums',
                    'triangles':len(faces),'degenerate_triangles':int(np.count_nonzero(np.linalg.norm(area,axis=1)==0)),
                    'zero_vertex_normals':int(np.count_nonzero(length==0)),
                    'orientation':'source face winding; no automatic reorientation',
                    'vertex_policy':'retain every source vertex in source order; no welding, filtering or resampling'}


def load_input(path,*,max_points=None,seed=2026,dataset=None):
    path=Path(path)
    ignored=[]
    derived=None
    if path.suffix.lower()=='.ply':
        from plyfile import PlyData
        mesh=PlyData.read(path);vertex=mesh['vertex']
        fields=('x','y','z','nx','ny','nz')
        if dataset and DATASETS[dataset].get('normal_source')=='triangle_winding':
            points=np.column_stack([vertex[name] for name in fields[:3]])
            faces=np.stack(mesh['face']['vertex_indices'])
            if len(faces)!=DATASETS[dataset]['faces']:raise ValueError(f'{dataset}: unexpected triangle count')
            normals,derived=mesh_vertex_normals(points,faces)
            data=np.column_stack([points,normals])
        else:
            if not all(name in vertex.data.dtype.names for name in fields):
                raise ValueError('PLY input must supply x, y, z, nx, ny, nz')
            data=np.column_stack([vertex[name] for name in fields])
        ignored=[name for name in vertex.data.dtype.names if name not in fields]
    else:
        data=np.loadtxt(path,ndmin=2)
        if data.shape[1]!=6:raise ValueError('ASCII oriented points must have exactly six columns')
    total=len(data)
    if not total:raise ValueError('Input point cloud is empty')
    metadata={'source_file':path.name,'source_sha256':checksum(path),'original_count':total,
              'normal_policy':'Use supplied normals; reconstruction applies its own reference normalization',
              'ignored_properties':ignored,'selection':'all rows in original file order'}
    if derived:
        metadata.update(input_kind='reconstructed_mesh_vertices',normal_policy=derived)
    if dataset:
        entry=DATASETS[dataset]
        if metadata['source_sha256']!=entry['sha256']:raise ValueError(f'{dataset}: point-cloud checksum mismatch')
        if total!=entry['count']:raise ValueError(f'{dataset}: unexpected vertex count')
        metadata.update(dataset=dataset,url=entry.get('url',BASE+entry.get('archive',entry['file'])),
                        description=entry['description'],source_readme=entry.get('source_page',SOURCE))
        if 'archive' in entry:metadata.update(archive_sha256=entry['archive_sha256'],archive_member=entry.get('member',entry['file']))
    if max_points is not None:
        if max_points<1:raise ValueError('max_points must be positive')
        if max_points<total:
            indices=np.sort(np.random.default_rng(seed).choice(total,max_points,replace=False))
            data=data[indices]
            metadata.update(selection='seeded uniform subset, retaining original file order',selection_seed=seed,
                            selection_indices_sha256=hashlib.sha256(indices.astype('<i8').tobytes()).hexdigest())
    metadata['selected_count']=len(data)
    return data[:,:3],data[:,3:],metadata


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('datasets',nargs='+',choices=DATASETS)
    a=p.parse_args()
    for name in a.datasets:print(name,download(name),flush=True)
