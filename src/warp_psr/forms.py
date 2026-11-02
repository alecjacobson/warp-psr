"""Exact volume and explicitly weighted point forms using Warp FEM."""
import warp as wp
import warp.fem as fem


@fem.integrand
def stiffness(s: fem.Sample, u: fem.Field, v: fem.Field):
    return wp.dot(fem.grad(u, s), fem.grad(v, s))


@fem.integrand
def mass(s: fem.Sample, u: fem.Field, v: fem.Field):
    return u(s) * v(s)


@fem.integrand
def value(s: fem.Sample, u: fem.Field):
    return u(s)


@fem.integrand
def gradient(s: fem.Sample, u: fem.Field):
    return fem.grad(u, s)


def uniform_space(depth, dtype=wp.float32):
    """Verification space only; never a maximum-depth reconstruction fallback."""
    if not isinstance(depth, int) or not 0 <= depth <= 5:
        raise ValueError("Uniform verification grid requires integer depth 0..5")
    geometry = fem.Grid3D(wp.vec3i(1 << depth))
    return fem.make_polynomial_space(geometry, degree=1, dtype=dtype,
                                     element_basis=fem.ElementBasis.BSPLINE)


def assemble_stiffness(space, device=None):
    domain = fem.Cells(space.geometry)
    u, v = fem.make_trial(space, domain=domain), fem.make_test(space, domain=domain)
    quadrature = fem.RegularQuadrature(domain, order=2)
    return fem.integrate(stiffness, quadrature=quadrature, fields={"u": u, "v": v},
                         output_dtype=space.dtype, device=device)


def assemble_screening(space, positions, effective_weights):
    """Sum w_s phi_i(p_s) phi_j(p_s); weights are NOT the public pointWeight."""
    if positions.shape[0] != effective_weights.shape[0]:
        raise ValueError("One effective weight per point is required")
    domain = fem.Cells(space.geometry)
    quadrature = fem.PicQuadrature(domain, positions, measures=effective_weights)
    u, v = fem.make_trial(space, domain=domain), fem.make_test(space, domain=domain)
    return fem.integrate(mass, quadrature=quadrature, fields={"u": u, "v": v},
                         output_dtype=space.dtype, device=positions.device)
