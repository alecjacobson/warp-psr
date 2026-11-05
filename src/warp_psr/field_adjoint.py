"""Tracked device transforms, cumulative coefficients and weighted isovalues."""
from dataclasses import dataclass
from functools import lru_cache
from types import SimpleNamespace
import warp as wp
from .field import LevelField, basis_key, find_key, reference_evaluation_kernel
from .keyed import coordinates

wp.set_module_options({'fuse_fp': False})


@lru_cache(None)
def precision_kernels(dtype):
    vec3 = wp.vec3d if dtype == wp.float64 else wp.vec3
    mat33 = wp.mat33d if dtype == wp.float64 else wp.mat33
    mat44 = wp.mat44d if dtype == wp.float64 else wp.mat44

    @wp.kernel
    def cumulative_backward(keys: wp.array(dtype=wp.int64), grad: wp.array(dtype=dtype),
                            grad_coefficients: wp.array(dtype=dtype), depth: int):
        i = wp.tid()
        if int(keys[i] >> wp.int64(48)) == depth:
            g = grad[i]
            wp.atomic_add(grad_coefficients, i, g)
            if depth > 0:
                c = coordinates(keys[i])
                for corner in range(8):
                    p = wp.vec3i(c[0] // 2 + corner // 4, c[1] // 2 + (corner // 2) % 2, c[2] // 2 + corner % 2)
                    h = ((dtype(1) - wp.abs(dtype(c[0]) * dtype(.5) - dtype(p[0])))
                         * (dtype(1) - wp.abs(dtype(c[1]) * dtype(.5) - dtype(p[1])))
                         * (dtype(1) - wp.abs(dtype(c[2]) * dtype(.5) - dtype(p[2]))))
                    if h > dtype(0):
                        j = find_key(keys, basis_key(depth - 1, p[0], p[1], p[2]))
                        if j >= 0:
                            wp.atomic_add(grad, j, g * h)

    @wp.kernel
    def evaluate_backward(keys: wp.array(dtype=wp.int64), coefficients: wp.array(dtype=dtype),
                          cumulative: wp.array(dtype=dtype), queries: wp.array(dtype=vec3), depth: int,
                          grad_value: wp.array(dtype=dtype), grad_spatial: wp.array(dtype=vec3),
                          grad_coefficients: wp.array(dtype=dtype), grad_cumulative: wp.array(dtype=dtype),
                          grad_queries: wp.array(dtype=vec3)):
        i = wp.tid()
        gv, gg = grad_value[i], grad_spatial[i]
        if gv == dtype(0) and gg[0] == dtype(0) and gg[1] == dtype(0) and gg[2] == dtype(0):
            return
        original = queries[i]
        p = vec3(original)
        ok = wp.isfinite(p[0]) and wp.isfinite(p[1]) and wp.isfinite(p[2]) and wp.min(p) >= dtype(0) and wp.max(p) <= dtype(1)
        if ok:
            leaf = int(0)
            for d in range(depth, -1, -1):
                n = 1 << d
                cell = wp.vec3i(wp.clamp(int(wp.ceil(p[0]*dtype(n)))-1,0,n-1),
                                wp.clamp(int(wp.ceil(p[1]*dtype(n)))-1,0,n-1),
                                wp.clamp(int(wp.ceil(p[2]*dtype(n)))-1,0,n-1))
                if find_key(keys,basis_key(d,cell[0],cell[1],cell[2])) >= 0:
                    leaf = d
                    break
            for axis in range(3):
                if p[axis] == dtype(0):
                    p[axis] = dtype(1e-6)
                elif p[axis] == dtype(1):
                    p[axis] = dtype(1.0-1e-6)
            qp = vec3(dtype(0))
            for d in range(wp.max(leaf-1,0),wp.min(leaf+1,depth)+1):
                n = 1 << d
                t = p*dtype(n)
                cell = wp.vec3i(wp.clamp(int(wp.ceil(t[0]))-1,0,n-1),wp.clamp(int(wp.ceil(t[1]))-1,0,n-1),wp.clamp(int(wp.ceil(t[2]))-1,0,n-1))
                base = wp.vec3i(int(wp.floor(t[0])),int(wp.floor(t[1])),int(wp.floor(t[2])))
                f = t-vec3(base)
                for corner in range(8):
                    node = cell+wp.vec3i(corner//4,(corner//2)%2,corner%2)
                    if wp.min(node-base) >= 0:
                        index = find_key(keys,basis_key(d,node[0],node[1],node[2]))
                        if index >= 0:
                            h = vec3(dtype(1))-f
                            sign = vec3(dtype(-1))
                            for a in range(3):
                                if node[a] > base[a]:
                                    h[a] = f[a]
                                    sign[a] = dtype(1)
                            basis_gradient = dtype(n)*vec3(sign[0]*h[1]*h[2],h[0]*sign[1]*h[2],h[0]*h[1]*sign[2])
                            gc = gv*h[0]*h[1]*h[2] + wp.dot(gg,basis_gradient)
                            c = dtype(coefficients[index])
                            if d == leaf-1:
                                c = dtype(cumulative[index])
                                wp.atomic_add(grad_cumulative,index,gc)
                            else:
                                wp.atomic_add(grad_coefficients,index,gc)
                            xy = c*dtype(n*n)*sign[0]*sign[1]*h[2]
                            xz = c*dtype(n*n)*sign[0]*h[1]*sign[2]
                            yz = c*dtype(n*n)*h[0]*sign[1]*sign[2]
                            qp += gv*c*basis_gradient + vec3(gg[1]*xy+gg[2]*xz,gg[0]*xy+gg[2]*yz,gg[0]*xz+gg[1]*yz)
            for a in range(3):
                if original[a] == dtype(0) or original[a] == dtype(1):
                    qp[a] = dtype(0)
            wp.atomic_add(grad_queries,i,qp)

    @wp.kernel
    def to_cube(queries: wp.array(dtype=vec3), transforms: wp.array(dtype=mat44), out: wp.array(dtype=vec3)):
        i = wp.tid()
        t = transforms[1]
        linear = mat33(t[0,0],t[0,1],t[0,2],t[1,0],t[1,1],t[1,2],t[2,0],t[2,1],t[2,2])
        out[i] = linear * queries[i] + vec3(t[0,3],t[1,3],t[2,3])

    @wp.kernel
    def to_model_gradient(gradient: wp.array(dtype=vec3), transforms: wp.array(dtype=mat44), out: wp.array(dtype=vec3)):
        i = wp.tid()
        t = transforms[1]
        linear = mat33(t[0,0],t[0,1],t[0,2],t[1,0],t[1,1],t[1,2],t[2,0],t[2,1],t[2,2])
        out[i] = wp.transpose(linear) * gradient[i]

    @wp.kernel
    def center(values: wp.array(dtype=dtype), iso: wp.array(dtype=dtype), out: wp.array(dtype=dtype)):
        i = wp.tid()
        out[i] = values[i] - iso[0]

    @wp.kernel
    def mass_partials(weights: wp.array(dtype=dtype), partials: wp.array(dtype=wp.float64)):
        block = wp.tid()
        total = wp.float64(0)
        for j in range(block * 256, wp.min((block + 1) * 256, weights.shape[0])):
            total += wp.float64(weights[j])
        partials[block] = total

    @wp.kernel
    def mass_finish(partials: wp.array(dtype=wp.float64), total: wp.array(dtype=wp.float64)):
        value = wp.float64(0)
        for i in range(partials.shape[0]):
            value += partials[i]
        total[0] = value

    @wp.kernel
    def iso_backward(values: wp.array(dtype=dtype), weights: wp.array(dtype=dtype), iso: wp.array(dtype=dtype),
                     mass: wp.array(dtype=wp.float64), grad_iso: wp.array(dtype=dtype),
                     grad_values: wp.array(dtype=dtype), grad_weights: wp.array(dtype=dtype)):
        i = wp.tid()
        scale = dtype(wp.float64(grad_iso[0]) / mass[0])
        wp.atomic_add(grad_values, i, scale * weights[i])
        if grad_weights.shape[0]:
            wp.atomic_add(grad_weights, i, scale * (values[i] - iso[0]))

    return SimpleNamespace(cumulative_backward=cumulative_backward, evaluate_backward=evaluate_backward, to_cube=to_cube,
                           to_model_gradient=to_model_gradient, center=center,
                           mass_partials=mass_partials, mass_finish=mass_finish, iso_backward=iso_backward)


def record_cumulative_vjp(keys, coefficients, cumulative, depth, *, tape):
    def backward():
        work = wp.clone(cumulative.grad, requires_grad=False)
        for d in range(depth, -1, -1):
            wp.launch(precision_kernels(coefficients.dtype).cumulative_backward, len(keys),
                      inputs=[keys, work, coefficients.grad, d], device=keys.device, record_tape=False)
    tape.record_func(backward, [coefficients, cumulative])


def record_isovalue_vjp(values, weights, iso, *, tape):
    kernels, device = precision_kernels(values.dtype), values.device
    partials = wp.empty((len(weights) + 255) // 256, dtype=wp.float64, device=device)
    mass = wp.empty(1, dtype=wp.float64, device=device)
    wp.launch(kernels.mass_partials, len(partials), inputs=[weights, partials], device=device, record_tape=False)
    wp.launch(kernels.mass_finish, 1, inputs=[partials, mass], device=device, record_tape=False)
    empty = wp.empty(0, dtype=weights.dtype, device=device)
    def backward():
        wp.launch(kernels.iso_backward, len(values),
                  inputs=[values, weights, iso, mass, iso.grad, values.grad,
                          weights.grad if weights.requires_grad else empty], device=device, record_tape=False)
    tape.record_func(backward, [values, iso] + ([weights] if weights.requires_grad else []))


def cube_field(keys, coefficients, cumulative, depth):
    return LevelField(keys, coefficients, (1.,0.,0.,0.,1.,0.,0.,0.,1.), (0.,0.,0.), depth, 0., cumulative)


def evaluate_cube(field, queries, *, tape):
    dtype, device = field.coefficients.dtype, queries.device
    vector = wp.vec3d if dtype == wp.float64 else wp.vec3
    matrix = wp.mat33d if dtype == wp.float64 else wp.mat33
    values = wp.empty(len(queries),dtype=dtype,device=device,requires_grad=tape is not None)
    gradients = wp.empty_like(queries,requires_grad=tape is not None)
    valid = wp.empty(len(queries),dtype=int,device=device)
    wp.launch(reference_evaluation_kernel(dtype),len(queries),
              inputs=[field.keys,field.coefficients,field.cumulative,queries,matrix(1,0,0,0,1,0,0,0,1),vector(0),field.depth,
                      values,gradients,valid],device=device,record_tape=False)
    def backward():
        wp.launch(precision_kernels(dtype).evaluate_backward,len(queries),
                  inputs=[field.keys,field.coefficients,field.cumulative,queries,field.depth,values.grad,gradients.grad,
                          field.coefficients.grad,field.cumulative.grad,queries.grad],device=device,record_tape=False)
    if tape is not None:
        tape.record_func(backward,[field.coefficients,field.cumulative,queries,values,gradients])
    return values,gradients,valid


@dataclass
class DifferentiableField:
    cube: LevelField
    transforms: wp.array
    iso_value: wp.array
    tape: wp.Tape
    ready: object = None

    @property
    def coefficients(self):
        return self.cube.coefficients

    @property
    def cumulative(self):
        return self.cube.cumulative

    def evaluate(self, queries, *, centered=False, requires_grad=True):
        """Evaluate with device transforms; opt out of recording for fixed selection.

        ``requires_grad=False`` returns untracked values/spatial gradients and
        records no operations, even inside the reconstruction tape. This is for
        discrete classification, not for differentiable losses.
        """
        dtype, device = self.cube.coefficients.dtype, self.cube.coefficients.device
        vector = wp.vec3d if dtype == wp.float64 else wp.vec3
        if queries.dtype != vector or queries.device != device:
            raise ValueError('queries must match field precision and device')
        if self.ready is not None:
            wp.get_stream(device).wait_event(self.ready)
        kernels = precision_kernels(dtype)
        local = wp.empty_like(queries, requires_grad=requires_grad)
        wp.launch(kernels.to_cube, len(queries), inputs=[queries, self.transforms, local], device=device,
                  record_tape=requires_grad)
        values, gradients, valid = evaluate_cube(self.cube, local, tape=self.tape if requires_grad else None)
        model_gradients = wp.empty_like(gradients, requires_grad=requires_grad)
        wp.launch(kernels.to_model_gradient, len(queries), inputs=[gradients, self.transforms, model_gradients], device=device,
                  record_tape=requires_grad)
        if centered:
            centered_values = wp.empty_like(values, requires_grad=requires_grad)
            wp.launch(kernels.center, len(values), inputs=[values, self.iso_value, centered_values], device=device,
                      record_tape=requires_grad)
            values = centered_values
        return values, model_gradients, valid
