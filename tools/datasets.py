"""Offline inputs linked by the pinned PoissonRecon README; never production code."""
import argparse
import hashlib
from pathlib import Path
import shutil
from urllib.request import urlopen
import zipfile
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
SOURCE='https://github.com/mkazhdan/PoissonRecon/blob/262b0f539d404057d1f36e1adc07fc9388678899/README.md'
BASE='https://www.cs.jhu.edu/~misha/Code/PoissonRecon/'
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
            with urlopen(BASE+filename,timeout=60) as source,temporary.open('wb') as dest:
                shutil.copyfileobj(source,dest)
            expected=entry.get('archive_sha256',entry['sha256'])
            if checksum(temporary)!=expected:raise ValueError(f'{name}: download checksum differs from the pinned dataset')
            temporary.replace(downloaded)
        if 'archive' in entry:
            if checksum(downloaded)!=entry['archive_sha256']:raise ValueError(f'{name}: archive checksum mismatch')
            with zipfile.ZipFile(downloaded) as archive,archive.open(entry['file']) as source,path.open('wb') as dest:
                shutil.copyfileobj(source,dest)
    if checksum(path)!=entry['sha256']:raise ValueError(f'{name}: point-cloud checksum mismatch')
    return path


def load_input(path,*,max_points=None,seed=2026,dataset=None):
    path=Path(path)
    ignored=[]
    if path.suffix.lower()=='.ply':
        from plyfile import PlyData
        vertex=PlyData.read(path)['vertex']
        fields=('x','y','z','nx','ny','nz')
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
    if dataset:
        entry=DATASETS[dataset]
        if total!=entry['count']:raise ValueError(f'{dataset}: unexpected vertex count')
        metadata.update(dataset=dataset,url=BASE+entry.get('archive',entry['file']),
                        description=entry['description'],source_readme=SOURCE)
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
