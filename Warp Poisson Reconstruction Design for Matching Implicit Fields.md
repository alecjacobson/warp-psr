**Goal: Same adaptive discretization and numerically matching implicit field.**

Implement Poisson Surface Reconstruction and Screened Poisson Surface Reconstruction using GPU computation in pure Warp, reproducing the field produced by a pinned revision and explicit configuration of [mkazhdan/PoissonRecon](https://github.com/mkazhdan/PoissonRecon). Reuse existing Warp and `warp.fem` geometry, bases, quadrature, assembly, and solvers wherever they preserve this goal.

Start with the existing APIs. If a required operation cannot be expressed correctly through them, first demonstrate the limitation with a minimal failing example. Then implement the smallest reusable adjustment on a branch of a Warp fork and prepare a merge request to NVIDIA/warp. A new function space is an acceptable extension; replacing the FEM framework is not the starting architecture.

The delivered reconstruction must compute its own adaptive hierarchy on the GPU. Importing a CPU reference tree is permitted only in development tests. Mesh extraction, triangle connectivity, and bitwise CPU/GPU identity are outside this scope.

# Scope and compatibility contract

The first supported profile is three-dimensional reconstruction with degree-1 elements, Neumann boundaries, and oriented points. It includes unscreened reconstruction, positive screening weights, and both the reference default approximate point-interpolation path and its exact path. Degree 2 follows after the degree-1 compatibility gates pass.

The repository currently documents degree 1, Neumann boundaries, and a default point weight of twice the degree. These observations guide the initial profile; the implementation must read and record the effective settings of the pinned build instead of depending on moving defaults. [1](https://github.com/mkazhdan/PoissonRecon)

| Area | Initial supported behavior |
| - | - |
| Input | Finite 3D positions and oriented normals; reference handling of invalid or zero normals and coincident samples |
| Normal weighting | Confidence disabled initially; preserve normal processing and sample multiplicity |
| Geometry | Reference normalization, complete coarse region, adaptive refinement and support completion |
| Parameters | Explicit depth, full depth, base depth, solve depth, kernel depth, samples per node, scale, point weight, interpolation mode and solver settings |
| Arithmetic | Float64 diagnostic fixtures and float32 compatibility runs, with matching input quantization and reference arithmetic profiles |
| Result | GPU evaluation of field values and gradients, reference isovalue, coordinate transform and diagnostic metadata |
| Later extensions | Degree 2, confidence weighting, other boundary types and additional reference options |
| Excluded | Envelope constraints, color or attribute reconstruction, distributed reconstruction, surface trimming and matching mesh extraction |

Do not silently ignore unsupported options. Report them before allocating the reconstruction.

“Pure Warp” permits Python launch orchestration, Warp arrays and kernels, and built-in Warp sorting, scans, sparse operations and volume allocation. Production preprocessing, tree construction, assembly, solving and evaluation must execute on the GPU. Host reads of counts or small status records are permitted in the initial implementation and must be measured. NumPy, SciPy, PyTorch, CuPy and the reference C++ implementation may be used by offline tests, not to perform production reconstruction.

Compatibility means the same domain, refinement decisions, active basis space, normal-field construction, point constraints, boundary conditions and level coupling, up to an explicit change of basis or permutation. It also requires agreement with the field of the specified reference run. Changing the discretization while obtaining similar surfaces does not satisfy the contract.

# Reference profiles and mathematical formulation

Maintain two named validation profiles.

**Run compatibility** compares against the pinned reference executable with its recorded iteration schedule and tolerances. This is the acceptance target for a claim that the implementation matches that configuration.

**Converged discretization** uses more stringent solves on both implementations to diagnose discrepancies in their discrete problems. It does not substitute for run compatibility. If the reference uses level-dependent approximations or a sequential correction procedure that is not equivalent to one independently converged global system, record that behavior and compare its level operators and updates directly.

For an independent coefficient representation, the conceptual objective and linear system are:

```math
E(c)=\frac12\int_\Omega\left\|\nabla\!\left(\sum_j c_j\phi_j\right)-V\right\|^2\,dx
+\frac12\sum_s\omega_s\left(\sum_j c_j\phi_j(p_s)-t_s\right)^2,
\qquad
(K+P^\mathsf{T}WP)c=b+P^\mathsf{T}Wt.
```

Here K contains gradient inner products; b contains the inner products of the test gradients with the reconstructed vector field V; P evaluates basis functions at the effective constraint locations; and W contains the effective weights, including screening strength. V includes the reference normal orientation and sign convention. Unscreened reconstruction removes the point term. Screening acts at points rather than as a volume mass term. [2](https://hhoppe.com/proj/screenedpoisson/)

This equation specifies the algebra, not a license to replace the reference’s multilevel representation with independent coefficients at every tree node. Determine which coefficients are independent, which are prolonged values or corrections, and how they are combined during evaluation. A naïve concatenation of nested bases can introduce linear dependencies.

The reference reconstruction code includes density estimation, normal-field construction, approximate or exact interpolation setup, and weighted evaluation of the extraction isovalue. Record those intermediate quantities directly in the oracle fixtures. The public point-weight parameter is not a complete description of the effective point weights. [3](https://git.brlcad.org/BRL-CAD/PoissonRecon/src/branch/master/Src/Reconstructors.h)

# Reference instrumentation and reproducibility

Create a small reference driver around the pinned PoissonRecon sources. Instrument intermediate data without changing mathematical behavior. Verify that instrumentation preserves the field of the corresponding uninstrumented executable.

Every fixture manifest must contain both repository commit hashes, compiler and build options, scalar types, input checksum and ordering, effective reconstruction parameters, reference thread configuration, random seed, CUDA and GPU information for the Warp run, and the fixture-format version. Resolve negative or automatic depth settings to their actual values. Keep diagnostic oracle output separate from production dependencies.

Export the following stages using stable mathematical keys rather than pointer addresses or incidental array order.

| Stage | Required observations |
| - | - |
| Input processing | Accepted and rejected samples, normalized positions and normals, aggregation membership and weights, model-to-cube transform |
| Tree | Depth and integer coordinates, parent and child relationships, relevant flags, support nodes, active basis indices and per-level counts |
| Density and normal field | Density coefficients, effective splat depths, normal coefficients, weights and scale factors |
| Basis | Values and gradients at fixed queries, boundary behavior and any coefficient transformations |
| Constraints | Effective point positions, targets, weights and their level associations |
| Operators | Small explicit matrices and right-hand sides; large-system operator actions; transfers and coarse/fine coupling |
| Solve | Initial state, per-level updates, residuals, stopping rules and final coefficient representation |
| Evaluation | Values, gradients, isovalue and gauge information at a fixed query set |

Source entry points to inspect are `Src/PoissonRecon.cpp`, `Src/Reconstructors.h`, `Src/BSplineData.h`, `Src/FEMTree.inl`, and `Src/FEMTree.System.inl`. Trace their calls in the pinned checkout to locate density, splatting, initialization and evaluation routines. Do not infer current behavior solely from the 2006 or 2013 papers.

Generate baseline fixtures with one reference thread, then repeat a subset with its normal parallel configuration to quantify reference variability. Keep a high-precision diagnostic profile distinct from the production reference profile. A float64 solve after float32 preprocessing is not an end-to-end float64 fixture.

Do not select a hash in advance from a moving web page. The first implementation milestone records the actual checked-out revisions and replaces moving source URLs in the test report with commit permalinks.

# Reuse strategy and decision gates

Existing capabilities include adaptive grid geometry, B-spline bases, particle quadrature and sparse FEM assembly. The latest development API also exposes conforming and constrained bases. API presence and behavior must be tested on the pinned Warp revision; documentation for `latest` is not a release guarantee. [4](https://nvidia.github.io/warp/latest/api_reference/warp_fem.html)

| Requirement | First implementation choice | Evidence required before changing Warp |
| - | - | - |
| Adaptive integration cells | `AdaptiveNanogrid` from explicit cells or hierarchy | Cell coverage or indexing cannot represent the required integration partition |
| Uniform spline space | `ElementBasis.BSPLINE` on supported grids | Basis values, support or boundary treatment differ and existing composition cannot correct them |
| Adaptive degree-1 space | Existing basis plus supported constraints, or per-level grid spaces | A tiny adaptive fixture demonstrates unequal spaces or an inaccessible required operation |
| FEM forms | `integrand`, `integrate`, explicit quadrature | Correct basis and quadrature still cannot express a required coupling |
| Point constraints | `PicQuadrature` with explicit effective weights | Reference locations or evaluation support cannot be represented |
| Linear algebra | `warp.sparse` and `warp.optim.linear` | A missing primitive prevents the required operator or solver |
| Evaluation | Existing fields and basis evaluation | Reference level accumulation or lookup cannot be expressed correctly |

Use three gates in order.

1. **Uniform basis gate.** Compare degree-1 basis evaluations, gradients, boundary functions and local integrals on a complete small grid. Resolve indexing, coordinate shifts, normal signs and weight scaling before adding adaptivity.

2. **Adaptive basis gate.** Refine one region of a tiny tree, then add neighboring regions at different depths. Compare basis spans, function values, coarse/fine couplings and transfers. Matching degree or node count alone is insufficient.

3. **Public extension gate.** If the built-in space is unsuitable, try an application-local subclass using the existing `BasisSpace` and topology interfaces. Request an upstream adjustment only for a demonstrated interface or reusable functionality gap.

A new application-local function space implemented through existing extension points still counts as reuse of `warp.fem`. It does not automatically require a fork.

The conforming-basis helper eliminates hanging nodes of interpolatory Lagrange bases; it does not establish equivalence with PoissonRecon’s adaptive spline space. Its suitability is an experiment to resolve, not an assumption. [5](https://nvidia.github.io/warp/latest/api_reference/_generated/warp.fem.make_conforming_basis_space.html)

# GPU hierarchy construction

Keep a logical reconstruction tree as the authoritative data structure. Derive FEM geometry and coefficient mappings from it. NanoVDB is an indexing backend, not the definition of PoissonRecon’s logical tree.

Use structure-of-arrays storage containing depth, integer coordinates, parent index, child information, reference-relevant flags, level ranges, sample ranges and per-level coefficient mappings. Sort spatial keys for locality. Maintain a canonical key for comparison with the oracle. Represent padded or negative coordinates explicitly; do not let a Morton encoding silently discard sign or exceed its bit capacity.

The hierarchy builder must reproduce the observable result of the reference construction and finalization stages. It may use a different order of independent operations when that leaves decisions unchanged. The reference has explicit finalization and ghost/support handling, so occupied sample cells alone are not the target tree. [6](https://git.brlcad.org/BRL-CAD/PoissonRecon/src/branch/master/Src/FEMTree.inl)

## Construction stages

1. **Normalize and classify samples.** Reproduce the bounding transform, boundary ownership rules, normal processing and sample acceptance. Quantize using the selected arithmetic profile. Compare this output before constructing any tree.

2. **Build occupied paths and sample groups.** Generate finest-level keys; sort and identify groups; generate unique parent keys level by level. Preserve sample multiplicity and reference aggregation semantics. Materialize complete child sets where required.

3. **Compute density and splat decisions.** Use GPU kernels for the reference density estimator, depth selection, weighting and normal contributions. Reproduce contributions distributed across levels if the oracle demonstrates them.

4. **Complete the reconstruction tree.** Add the full coarse region, stencil support and required ancestors or children. Apply the reference pruning and flag rules in the dependencies established by the source audit.

5. **Compact and index.** Construct stable per-level indices, parent/child links, coefficient maps and neighborhoods. Validate keys and flags against the oracle.

6. **Derive integration geometry.** Emit nonoverlapping cells covering the required domain, with level and coordinate maps into `AdaptiveNanogrid`. Preserve the distinction between integration cells and basis-support nodes.

Warp’s radix sort and prefix scan provide the main grouping and compaction primitives. Use a scan over marks to assign deterministic output indices instead of atomically assigning IDs. Floating-point sums need a separate reproducibility strategy. [7](https://nvidia.github.io/warp/latest/api_reference/_generated/warp.utils.radix_sort_pairs.html) [8](https://nvidia.github.io/warp/latest/api_reference/_generated/warp.utils.array_scan.html)

Use deterministic grouped reductions for quantities that decide refinement, aggregation or support activation. If a threshold-sensitive fixture still disagrees, match the reference arithmetic order for that decision or identify the precise source of divergence. Do not repair compatibility by adding arbitrary rounding thresholds.

For N samples and maximum depth D, occupied sample paths contain at most O(ND) entries before deduplication. Account separately for the complete coarse region and support-completion nodes. Process levels incrementally so the implementation need not allocate all sample–ancestor pairs. Report time and memory against actual node counts; no linear-complexity claim is accepted solely because radix sorting is linear.

The FEM hierarchy constructor requires a compatible nonoverlapping hierarchy and can impose additional grading. Do not enable grading or refinement that changes the reference space without an equivalence proof. Extra cells may be used solely for integration if they introduce no extra degrees of freedom and preserve the integrals. [9](https://nvidia.github.io/warp/latest/api_reference/_generated/warp.fem.adaptive_nanogrid_from_hierarchy.html)

## Allocation and synchronization

Initially permit host reads of scalar counts to size buffers, with all topology decisions and bulk data remaining on the GPU. Track synchronization time separately. Introduce capacity reuse after correctness; define overflow detection and recovery before attempting graph capture.

Never allocate a dense grid at maximum depth as an intermediate. Preserve coarse coverage of the reconstruction cube rather than solving only in a narrow band. Keep reference full-depth behavior even when that coarse region is a noticeable fraction of memory.

# Function spaces and assembly

Separate three concepts in the implementation: integration cells, basis functions with their support, and the solver hierarchy. They can share indexing structures, but their membership need not coincide.

## Existing spaces first

Test uniform B-splines and adaptive conforming degree-1 spaces against oracle basis evaluations. Where the coefficient convention differs, establish an explicit permutation or change of basis. Check gradients and integrated forms as well as sampled values. Equal values at a few vertices do not establish equal functions.

Compare a hierarchy of per-level `Nanogrid` spaces with the reference’s level representation before inventing a single global adaptive space. Existing same-domain or mixed-form assembly may provide the needed pieces. Verify that every cross-level integral is evaluated on a common partition resolving both functions.

Boundary functions require explicit tests. Selecting “Neumann” at the PDE level does not prove that basis modification, support outside the cube or reflected coefficients match the reference.

## Local polynomial representation as a verification route

On cells that resolve every relevant spline breakpoint, a reference spline restricts to a polynomial. Construct a sparse map R from an independent reference coefficient representation to local polynomial coefficients and use FEM assembly to verify:

```math
K_{\mathrm{ref}}=R^\mathsf{T}K_{\mathrm{local}}R,
\qquad
b_{\mathrm{ref}}=R^\mathsf{T}b_{\mathrm{local}},
\qquad
P_{\mathrm{ref}}=P_{\mathrm{local}}R.
```

Treat this as an equivalence experiment and small-fixture oracle. Measure the integration-partition size and row widths of R before adopting it for production. Fine-scale support can cross a coarse leaf, requiring extra integration subdivisions. If the reference coefficient storage is redundant, derive the independent mapping or perform the comparison per level rather than solving an artificial singular global problem.

The current constrained-basis API requires each independent coefficient to have an anchor row. An arbitrary spline-to-polynomial map need not satisfy this condition. First use explicit sparse transformations or a custom basis through existing extension points. Only propose a generic unanchored-basis adjustment if a minimal reproducer shows why those routes are insufficient. [10](https://nvidia.github.io/warp/latest/api_reference/_generated/warp.fem.make_constrained_basis_space.html)

## Application-local compatible basis

If existing concrete spaces fail the equivalence gate, implement the reference basis through the existing `BasisSpace` and `SpaceTopology` contracts. The basis supplies element support lists, values and gradients; topology maps element-local entries to coefficients. Reuse collocated scalar fields, test/trial construction and FEM integration. The published interface already separates basis evaluation from topology and value type. [11](https://nvidia.github.io/warp/latest/api_reference/_generated/warp.fem.BasisSpace.html)

Preserve exact support activation and boundary handling from the oracle. Do not label a generic hierarchical or truncated hierarchical B-spline construction “compatible” without verifying that its space and level operations are the same.

Use bounded per-cell support lists where the reference permits them. If many levels contribute to an element, measure the support width and assembly cost. Prefer a verified per-level formulation over allocating an array sized for the worst possible depth at every cell.

## Quadrature and forms

Use ordinary volume quadrature for the gradient inner product and normal-field load. Choose enough tensor-product quadrature points to integrate the actual polynomial products on each integration cell, including the independently determined degree of the normal field. Validate polynomial exactness and compare against the reference integrals.

Use point quadrature only for effective interpolation constraints. Avoid nodal or lumped assembly unless it reproduces the corresponding reference operator. Keep the normal field as an evaluable field with its own coefficients and degree rather than assuming it is piecewise constant.

The first production candidate assembles scalar BSR operators with `fem.integrate`. Once its correctness is established, equivalent matrix-free application or cached stencils can replace expensive assembly. Every optimized operator must pass the same action tests as the assembled version.

# Screening and sample weights

Implement unscreened and screened reconstruction through the same tree, basis, normal-field and solver infrastructure. Screening adds its reference constraints; it must not trigger a separate approximate reconstruction method.

Export and compare effective constraint locations, targets, weights and level associations. Preserve the reference default approximate path as a distinct supported mode. Testing the exact point-interpolation path first can simplify diagnosis, but success there does not establish default compatibility.

Supply explicit weights to `PicQuadrature`. Its default measure divides cell measure among particles and therefore must not be used as an implicit interpretation of screening weights. Confirm a single-point test produces the expected outer product of basis evaluations. [12](https://nvidia.github.io/warp/latest/api_reference/_generated/warp.fem.PicQuadrature.html)

Check aggregation by comparing one sample, repeated identical samples, multiple distinct samples within one leaf, and samples straddling a cell boundary. Do not assume that evaluating a constraint at an averaged position is equivalent to summing its individual outer products.

Reference point clustering at coarse levels may define coarse screening operators that differ from a simple Galerkin product. Compare the actual coarse operator and transfer behavior before substituting a generic multigrid hierarchy.

Keep the user’s point-weight parameter separate from effective numerical weights in code and diagnostics. Track every dimensional and depth scaling factor so that a scale-dependent discrepancy can be located without fitting an empirical correction.

# Solver and field evaluation

Use Warp sparse operations and iterative solvers first. They already provide GPU CG and other Krylov methods, a linear-operator interface, and preconditioning infrastructure. No large sparse direct solver is required. [13](https://nvidia.github.io/warp/latest/api_reference/warp_optim_linear.html)

## Solve the correct representation

Before choosing CG, establish a symmetric positive-definite independent system after nullspace handling. A redundant union of nested bases is not such a system. Where the reference advances through per-level correction equations, preserve those equations and their accumulation.

Implement a small assembled verification solve before optimizing. Compare each relevant operator action and right-hand side, then compare updates or converged fields. For larger inputs, use the verified hierarchy for a geometric multigrid preconditioner or the corresponding reference multilevel procedure.

If using multigrid-preconditioned CG, keep the preconditioner symmetric and positive definite: use adjoint-compatible restriction/prolongation, symmetric smoothing and an appropriate coarse solve. A changed or nonsymmetric preconditioner requires a compatible outer method. Do not feed an arbitrary reference iteration procedure into CG as though these properties were automatic.

The reference exposes per-level iteration settings and multilevel solver routines. Match their observable effect where required by run compatibility. Parallel smoothing, a different stopping tolerance or a more accurate solve may alter the output field. [14](https://github.com/mkazhdan/PoissonRecon/blob/master/Src/PoissonRecon.cpp) [15](https://git.brlcad.org/BRL-CAD/PoissonRecon/src/branch/master/Src/FEMTree.System.inl)

## Gauge and screening

For unscreened Neumann reconstruction, explicitly handle the constant nullspace. Record the reference convention. If coefficients use a different legitimate gauge, apply one declared constant-offset transformation based on the same fixed volume-mean functional before comparing raw values. Report both the offset and the unaligned error. Do not fit a scale, spatial correction or sample-dependent offset.

Compare the isovalue independently and compare the centered field, field minus isovalue. A common constant shift must affect both consistently. For screened reconstruction, compare raw field values and targets directly; do not remove a fitted constant to conceal an error.

If a candidate system has additional null modes, investigate basis redundancy, disconnected active components or missing couplings. Do not regularize them away silently.

## Evaluation contract

Return arbitrary-query field values and gradients directly from the solved representation, without sampling to a dense grid. Accumulate levels using the verified reference rule. Support queries throughout the reconstruction cube and define the behavior of out-of-domain queries.

Store the model-to-cube affine transform and its inverse. Convert gradients back using the transpose Jacobian of model-to-cube. Test this with translated and uniformly scaled inputs.

Degree-1 gradients may be discontinuous on knot or cell interfaces. Compare gradients at interior queries and with separate one-sided interface tests. At exact interfaces, reproduce or explicitly expose the reference side convention; do not average gradients as an undocumented smoothing operation.

# Numerical acceptance criteria

These are proposed initial test budgets, not measured accuracy claims. Freeze them with the first oracle fixtures. Any change requires a documented numerical reason and a regenerated compatibility report; an implementation cannot make a failure disappear by silently loosening tolerances.

Topology, discrete classifications and active function membership must match exactly after canonicalization. Coefficient arrays may differ by a documented permutation or verified basis transformation.

For a sampled quantity z, require each scalar component to satisfy:

```math
|z_{\mathrm{warp}}-z_{\mathrm{ref}}|
\le a+r\,|z_{\mathrm{ref}}|.
```

| Comparison | Float64 diagnostic target | Float32 compatibility target |
| - | - | - |
| Normalization, weights, basis values and scaled basis gradients | a = 1e-10, r = 1e-9 | a = 1e-6, r = 1e-5 |
| Normalized operator actions and loads | a = 1e-10, r = 1e-9 | a = 1e-6, r = 1e-5 |
| Field values, centered values and isovalue | a = 1e-8, r = 1e-6 | a = 1e-5, r = 1e-4 |
| Cell-width-scaled field gradients at interior queries | a = 1e-8, r = 1e-6 | a = 1e-5, r = 1e-4 |

For a nonzero operator action or load, divide both results by one shared scale: the maximum absolute entry of the reference result. For an identically zero reference result, test unnormalized absolute error and verify structural zeros where applicable. Report the unnormalized infinity and RMS errors too. Scale a gradient at a query by the width of its corresponding canonical reference integration cell, in normalized cube coordinates. Apply the same width to both gradients. Also report raw gradient errors and errors grouped by depth.

Use the same tolerance policy for run compatibility and converged-discretization reports, but label them separately. Set algebraic solve error below the field budget where convergence is the diagnostic target. If a default reference run differs because it stopped early, that is not proof of a discretization error, but it remains a failure to match that run.

The query set must contain accepted sample positions, deterministic random points across the full cube, points in low-density and empty regions, cell interiors at every depth, points near both sides of refinement interfaces, and points near the cube boundary. Report maximum, RMS and per-depth errors. Surface-only queries are insufficient.

Initial diagnostic fixtures should be small enough to inspect full operators and basis transformations. Large fixtures compare deterministic random-vector actions and selected rows. Test symmetry, positive energy on the gauge-fixed space, and expected constant reproduction or nullspace behavior where applicable.

Measure reference repeatability and GPU repeatability separately. Nondeterministic accumulation cannot excuse topology differences. Test threshold-sensitive inputs explicitly; retain a deterministic compatibility path for decisions that affect the hierarchy.

# Tests and performance experiments

| Fixture family | Failure mode it is intended to expose |
| - | - |
| Complete small grids and single basis impulses | Coordinate shifts, basis normalization, boundary functions and incorrect integrals |
| One refined region and adjacent refinement levels | Hanging-node assumptions, support completion and level coupling |
| Single sample, duplicates and a dense cluster | Aggregation, normal normalization and screening weights |
| Plane patches, spheres and a torus | Orientation, field sign, analytic geometry behavior and off-surface evaluation |
| Highly nonuniform sampling and large empty regions | Density-driven refinement and incomplete domain coverage |
| Two nearby sheets and disconnected sample components | Lost couplings and errors hidden by surface-only validation |
| Samples near cell boundaries and refinement thresholds | Arithmetic-sensitive classification and deterministic construction |
| Translated and scaled copies | Normalization and gradient-transform errors |
| Multiple point weights including zero | Screening assembly, nullspace treatment and gauge handling |
| Approximate and exact interpolation modes | Incorrect substitution of point constraints |

Run each relevant family through staged comparisons: processed samples, logical tree, density and normal fields, basis, constraints, operators, solver and final evaluation. Stop diagnosis at the earliest mismatch.

For performance, vary sample count, maximum depth, samples per node and sampling distribution independently. Include approximately 10 thousand, 100 thousand and 1 million samples when resources permit. Always report actual active/support node counts, degrees of freedom and per-level sizes; input count alone is not a sufficient problem-size description.

Record normalization, sorting, density/splatting, tree completion, FEM topology, assembly, preconditioner setup, solve and query-evaluation times separately. Record cold compilation, warm end-to-end execution, synchronization and peak device memory. Compare against the same pinned reference configuration and accepted numerical error.

Measure temporary memory as well as final matrices. Generic assembly triplets can dominate the peak. Once the assembled implementation passes, prioritize buffer reuse, precomputed polynomial integrals, cached support lists and equivalent matrix-free operators according to measured cost.

No fixed speedup is an acceptance criterion for the first compatible implementation. The performance report must identify scaling bottlenecks and show that the production path does not depend on dense maximum-depth grids or CPU tree generation.

# Project structure and proposed interface

Keep the reconstruction implementation separate from any optional Warp extension. The following module names and API are proposed application interfaces, not existing Warp APIs.

| Module | Responsibility |
| - | - |
| `reference/` | Instrumented C++ oracle, fixture exporter and provenance |
| `reconstruction/parameters.py` | Supported profiles, effective settings and validation |
| `reconstruction/tree.py` | GPU hierarchy construction and canonical keys |
| `reconstruction/samples.py` | Normalization, aggregation, density and normal splatting |
| `reconstruction/spaces.py` | Existing-space adapters and any local compatible basis |
| `reconstruction/forms.py` | Volume and point forms, quadrature and assembly |
| `reconstruction/solver.py` | Reference-compatible level operations and optional equivalent solvers |
| `reconstruction/field.py` | Field values, gradients, transforms and isovalue |
| `tests/` | Staged compatibility tests and tiny diagnostic fixtures |
| `benchmarks/` | Reproducible timing and memory experiments |

```python
# Proposed application interface
result = reconstruct(
    points, normals,              # GPU Warp arrays
    parameters=parameters,        # Explicit compatibility settings
    device="cuda",
)

result.evaluate(query_points, values=values_out, gradients=gradients_out)
result.iso_value
result.model_to_cube
result.parameters
result.diagnostics
```

The result owns the hierarchy and coefficients needed for evaluation. Define which small fields are host metadata and which remain device arrays. Make precision, gauge, supported interpolation mode, and reference profile discoverable.

Use explicit names for representation conversions. A helper that samples a field to another grid must not be confused with exact field evaluation or a proven change of basis.

Optional diagnostic export is outside timed production execution. Production imports must not load or invoke the reference executable.

# Conditional Warp fork and merge request

A fork is justified only after an existing-API route fails a specific compatibility requirement. Document the smallest fixture, expected reference behavior, observed Warp behavior and attempted compositions. Distinguish a missing concrete space from a limitation of the extension interfaces.

The escalation order is:

1. Configure existing spaces, quadrature, constraints and sparse operators.

2. Add a compatible space or adapter in the reconstruction package using existing extension points.

3. Patch the smallest general-purpose Warp interface or add a reusable function space.

4. If a small adjustment still cannot represent the required behavior, describe the remaining design gap explicitly rather than expanding into an unreviewable FEM rewrite.

Potential adjustments include a supported adaptive spline space, a narrow capability for noninterpolatory constrained bases, or a missing evaluation/topology operation required by an externally defined basis. These are candidates, not predetermined changes.

Use a dedicated branch on the user’s Warp fork, for example `alecjacobson/poisson-fem-compat`, based on the pinned upstream commit. Confirm the actual fork before creating a branch. Keep the reconstruction application and reference-specific density, weighting and CLI policies out of the general FEM change.

A merge request to NVIDIA/warp must include:

- A concrete failing example on unmodified upstream.

- The smallest API or function-space addition that resolves it.

- Tests for basis values and gradients, boundary behavior, refinement interfaces and independent scalar FEM problems.

- A tiny reconstruction compatibility test where useful, without requiring a large external dataset.

- Documentation of supported geometries, degrees, scalar types and constraints.

- Before/after setup, assembly and memory measurements, including any regression on existing spaces.

- The exact upstream base and fork commit used by the reconstruction package.

Prefer one independently reviewable change per demonstrated gap. Pin the fork commit while review is pending; support an unmodified-upstream path if no patch is required. Do not claim upstream support until the change is merged and available in a specified revision.

Producing this design document does not create a repository branch or submit a merge request. Those are implementation deliverables only if the compatibility experiments establish a need.

# Milestones and completion criteria

| Milestone | Deliverable | Exit criterion |
| - | - | - |
| 0 Reference baseline | Pinned builds, explicit profiles, fixture format and instrumented oracle | Instrumented and original reference fields agree |
| 1 Uniform FEM compatibility | Degree-1 basis, forms, loads and point constraints using existing APIs | Small uniform fixtures pass numerical budgets |
| 2 Adaptive representation | Tiny refined-tree comparisons and a chosen basis representation | Space equivalence, boundary behavior and level coupling established |
| 3 GPU hierarchy | GPU sample processing, density, tree completion and mappings | Canonical tree and intermediate data match the oracle |
| 4 Reconstruction field | Unscreened and screened solve plus arbitrary-query evaluation | Required run-compatibility and diagnostic profiles pass |
| 5 Scalability | Timing, peak-memory report and equivalent optimizations | No CPU reconstruction stage or dense maximum-depth allocation; compatibility retained |
| 6 Optional upstream change | Focused fork branch and merge request | Demonstrated gap resolved with regression tests and documented dependency |

Milestones 1 and 2 may load oracle trees to isolate FEM behavior. Milestone 3 removes that dependency; milestone 4 must exercise the complete GPU construction path.

Completion requires:

- Matching normalized sample handling and the logical adaptive discretization for every supported fixture.

- Matching normal-field and effective interpolation data within the declared budgets.

- Verified equivalence of basis representation, operators, boundary treatment and level coupling.

- Matching field values, gradients and isovalue against the pinned run, with explicit unscreened gauge treatment.

- Both zero and positive screening weights, and both required interpolation modes.

- A production implementation expressed in Warp with bounded, measured host synchronization.

- A documented performance/memory profile and a reproducible test command.

- Either no Warp modifications or a minimal, tested fork change with a reviewable merge request.

The first decisive experiment is a degree-1 tree with one refined region. Export the reference basis and level operators, compare the existing Warp representations, and resolve that result before optimizing GPU reconstruction.

# Sources

The implementation must pin commit-specific sources during milestone 0. The links below identify the relevant upstream interfaces and reference implementation.

1. [PoissonRecon repository and documented parameters](https://github.com/mkazhdan/PoissonRecon)

2. [Kazhdan and Hoppe 2013 Screened Poisson Surface Reconstruction](https://hhoppe.com/proj/screenedpoisson/)

3. [Reconstructors.h source mirror](https://git.brlcad.org/BRL-CAD/PoissonRecon/src/branch/master/Src/Reconstructors.h) — useful for browsing; verify against the pinned upstream checkout.

4. [Warp FEM API](https://nvidia.github.io/warp/latest/api_reference/warp_fem.html)

5. [Conforming basis construction](https://nvidia.github.io/warp/latest/api_reference/_generated/warp.fem.make_conforming_basis_space.html)

6. [FEMTree.inl source mirror](https://git.brlcad.org/BRL-CAD/PoissonRecon/src/branch/master/Src/FEMTree.inl)

7. [Warp radix sort](https://nvidia.github.io/warp/latest/api_reference/_generated/warp.utils.radix_sort_pairs.html)

8. [Warp prefix scan](https://nvidia.github.io/warp/latest/api_reference/_generated/warp.utils.array_scan.html)

9. [Adaptive geometry from a hierarchy](https://nvidia.github.io/warp/latest/api_reference/_generated/warp.fem.adaptive_nanogrid_from_hierarchy.html)

10. [Constrained basis construction](https://nvidia.github.io/warp/latest/api_reference/_generated/warp.fem.make_constrained_basis_space.html)

11. [BasisSpace extension interface](https://nvidia.github.io/warp/latest/api_reference/_generated/warp.fem.BasisSpace.html)

12. [Particle quadrature](https://nvidia.github.io/warp/latest/api_reference/_generated/warp.fem.PicQuadrature.html)

13. [Warp iterative linear solvers](https://nvidia.github.io/warp/latest/api_reference/warp_optim_linear.html)

14. [PoissonRecon command-line driver](https://github.com/mkazhdan/PoissonRecon/blob/master/Src/PoissonRecon.cpp)

15. [FEMTree.System.inl source mirror](https://git.brlcad.org/BRL-CAD/PoissonRecon/src/branch/master/Src/FEMTree.System.inl)
