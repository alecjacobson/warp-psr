# Mesh API revision

Written by Codex (AI agent), 2026-11-26.

Updated [NVIDIA/warp#2452](https://github.com/NVIDIA/warp/pull/2452) in response to API review. The draft now returns `(vertices, indices)` directly from `poisson_extract_mesh`; callers can construct the existing `wp.Mesh` when they need spatial queries. The draft-only `PoissonMesh` and `PoissonMeshMethod` types were removed. Host algorithm selection uses `method: Literal["sparse", "dense"] = "sparse"`, following Warp's repository guidance for host-side mode/algorithm choices.

The Poisson-specific Lipschitz option and its private implementation were removed. Generic Warp Lipschitz pruning remains available, but requires a proven field bound that Poisson reconstruction does not supply. The current sparse route classifies a uniform extraction grid in slabs and calls `sparse_marching_cubes_from_cells` on retained crossings. It does not directly reuse adaptive reconstruction leaves. [Issue 105](https://github.com/alecjacobson/warp-psr/issues/105) tracks conservative hierarchy-driven candidates and refinement-interface handling.

The PR description now has an executed end-to-end oriented-ellipsoid example, returning triangle arrays and constructing a `wp.Mesh`, plus an executed `@wp.func`/kernel composition example. It explicitly documents both `poisson_evaluate` overloads: `PoissonField`/`wp.vec3` and `PoissonField64`/`wp.vec3d`, returning raw value, spatial gradient and validity.

Validation after the revision:

- All 22 native API tests pass, including dense/sparse vertex gradients, empty meshes, rejected methods, an assertion that the default path actually calls explicit-cell sparse MC, and regular `wp.Mesh` construction from the returned arrays.
- The new PR example produces 9,932 vertices and 19,860 triangles; the composed query example produces finite outputs. The repository's headless example produces 9,922 vertices and 19,840 triangles.
- Documentation builds with warnings treated as errors. Pre-commit and `git diff --check` pass. The generated API reference no longer exposes the removed types.
- Reconstruction, fast batched evaluation and device evaluation source files are unchanged. All nine retained mesh kernel ASTs are identical to the prior draft; only the unused Lipschitz clamp kernel is removed. No new performance claim or benchmark rerun is needed for these unchanged numerical paths.
- GPU tests and examples ran under the machine's shared GPU lock. The historical exclusive-lock performance report remains tied to its recorded source revisions.

The PR remains draft and unmerged. This API revision does not resolve the separate automated-review findings about numerical extremes, derivative edge cases, or padded sparse matrices; those threads remain available on the draft for focused follow-up. The separate compiler prerequisite remains unchanged.
