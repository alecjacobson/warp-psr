"""Reference float32 cofactor inverses for normalization and world queries."""
import warp as wp
wp.set_module_options({"fuse_fp":False})

@wp.func
def _det3(m: wp.mat33):
    return m[0,0]*(m[1,1]*m[2,2]-m[1,2]*m[2,1])-m[0,1]*(m[1,0]*m[2,2]-m[1,2]*m[2,0])+m[0,2]*(m[1,0]*m[2,1]-m[1,1]*m[2,0])

@wp.func
def _minor(m: wp.mat44, row: int, col: int):
    out=wp.mat33(0.0)
    rr=int(0)
    for r in range(4):
        if r!=row:
            cc=int(0)
            for c in range(4):
                if c!=col:
                    out[rr,cc]=m[r,c]
                    cc+=1
            rr+=1
    return _det3(out)

@wp.func
def reference_inverse(m: wp.mat44):
    det=float(0.0)
    for c in range(4):
        if c%2==0:det+=m[0,c]*_minor(m,0,c)
        else:det-=m[0,c]*_minor(m,0,c)
    out=wp.mat44(0.0)
    for r in range(4):
        for c in range(4):
            value=_minor(m,r,c)
            if (r+c)%2!=0:value=-value
            out[c,r]=value/det
    return out

@wp.kernel
def evaluation_transform(normalization: wp.array(dtype=float), transforms: wp.array(dtype=wp.mat44)):
    s=normalization[0]
    m=wp.mat44(0.,s,0.,normalization[1],0.,0.,s,normalization[2],s,0.,0.,normalization[3],0.,0.,0.,1.)
    inverse=reference_inverse(m)
    transforms[0]=inverse
    transforms[1]=reference_inverse(inverse)
