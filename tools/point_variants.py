"""Deterministic offline input defects; never used by production reconstruction."""
import hashlib
import numpy as np

VARIANTS=('none','position-noise','normal-noise','missing-cap','outliers')


def apply_variant(positions, normals, variant, *, seed):
    if variant not in VARIANTS:raise ValueError(f'Unknown point-cloud variant: {variant}')
    p,n=np.array(positions,dtype=np.float64,copy=True),np.array(normals,dtype=np.float64,copy=True)
    if p.ndim!=2 or p.shape[1]!=3 or p.shape!=n.shape or not len(p):
        raise ValueError('Variants require matching nonempty Nx3 positions and normals')
    if not np.isfinite(p).all() or not np.isfinite(n).all():
        raise ValueError('Input-defect variants require finite source data')
    rng=np.random.default_rng(seed)
    meta={'name':variant,'algorithm_version':1,'seed':seed,'input_count':len(p),
          'order':'source order retained; synthetic outliers appended',
          'scope':'controlled input defect; not untouched scanner data'}
    if variant=='position-noise':
        sigma=.002*float(np.linalg.norm(np.ptp(p,axis=0)))
        p+=rng.normal(0,sigma,p.shape)
        meta.update(sigma_bbox_diagonal_fraction=.002,sigma_source_units=sigma,normals='unchanged')
    elif variant=='normal-noise':
        length=np.linalg.norm(n,axis=1)
        valid=length>0
        n[valid]/=length[valid,None]
        n[valid]+=rng.normal(0,.15,(np.count_nonzero(valid),3))
        n[valid]/=np.linalg.norm(n[valid],axis=1)[:,None]
        meta.update(component_sigma=.15,normal_policy='perturb unit normals then renormalize; zero normals remain zero')
    elif variant=='missing-cap':
        boundary=float(np.quantile(p[:,2],.85))
        indices=np.flatnonzero(p[:,2]<=boundary)
        p,n=p[indices],n[indices]
        meta.update(axis='source z',keep_quantile=.85,cutoff_source_units=boundary,
                    retained_indices_sha256=hashlib.sha256(indices.astype('<i8').tobytes()).hexdigest())
    elif variant=='outliers':
        count=max(1,int(np.ceil(.01*len(p))))
        lo,hi=p.min(axis=0),p.max(axis=0);padding=.1*(hi-lo)
        extra=rng.uniform(lo-padding,hi+padding,(count,3))
        extra_normals=rng.normal(size=(count,3));extra_normals/=np.linalg.norm(extra_normals,axis=1)[:,None]
        p=np.concatenate([p,extra]);n=np.concatenate([n,extra_normals])
        meta.update(appended_count=count,count_fraction=.01,bbox_padding_fraction=.1,
                    normal_policy='source normals unchanged; independent random unit normals for appended points')
    meta['output_count']=len(p)
    return p,n,meta
