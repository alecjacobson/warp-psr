# Field-function composition before the Warp port

The CUDA composition fault is addressed by
[Warp PR #2363](https://github.com/NVIDIA/warp/pull/2363), at
[`d21984b`](https://github.com/alecjacobson/warp/commit/d21984b7470f6e696549a2e11e646b577572fe76).
The standalone reproducer and compiler evidence are in
[Warp #2362](https://github.com/NVIDIA/warp/issues/2362). Adoption by this package
remains tracked in [#101](https://github.com/alecjacobson/warp-psr/issues/101).

This is preparation for an internal port. PSR production code, its pinned Warp
1.19 dependency, and license files are unchanged. On that released dependency,
continue using `@wp.func(inline=True)` on helpers accepting a field handle.

## Diagnosis and fix

An ordinary helper can fail on its first forward launch, independently of
autodiff. Both explicit `inline=True` and explicit `inline=False` avoid the
reproduced fault; the earlier description as an out-of-line-call problem was too
broad. The failure reproduces on the official 1.19 wheel with NVRTC 12.9 and on
current Warp main built with CUDA 12.6.85, on an L40 with driver 570.158.01.

The failing PTX converts a kernel-parameter address into a local-memory address
before loading a struct member. Compute Sanitizer reports an invalid local read.
The proposed Warp workaround forces CUDA functions taking Warp structs to inline
when the caller supplied no hint. It preserves explicit hints and CPU behavior,
applies to forward and backward code, and includes the policy in cache hashes for
concrete functions, generic specializations, and custom callbacks. No arrays are
copied and no synchronization or launch-time dispatch is added. The exact NVRTC
root cause is not established.

## Correctness

[validation.json](validation.json) records commits, source hashes, hardware,
test outcomes, and limitations. The recorded native build predates the rebased
target by one CI-only commit; native sources are identical.

- The new regression fails with CUDA error 700 when the policy is disabled and
  passes with the fix. Fifteen final inlining tests pass, including CPU/CUDA,
  both precisions, explicit noinline, forward/adjoint attributes, cache keys, and
  directional finite differences for coefficient and query derivatives.
- The related Warp core run passed 519 of 520 tests initially. The sole failure
  was a subprocess unable to import the source checkout; that exact test passed
  with `PYTHONPATH` corrected in the subsequent 102-test FEM/hash/inlining run.
- The full PSR suite plus its field tests routed through ordinary helpers passed:
  **742 passed, 6 skipped, 1 existing expected failure**. These include direct and
  reference fields, both precisions, query/coefficient derivatives, and gradients
  through reconstruction inputs.
- Compute Sanitizer reported **zero errors** for the four core precision/hint
  combinations and five PSR composition/gradient/reconstruction-adjoint cases.
- Three checks also passed with the equivalent codegen policy injected into the
  1.19 wheel in memory. This experiment did not modify the installed wheel.

The tests cover first-order derivatives with fixed discrete reconstruction
choices. This change does not extend differentiation across topology changes.

## Performance

Measurements on 2026-11-19 use the same native Warp build for both variants, with
only the inlining policy enabled or disabled. The machine has a **four-core
cgroup quota** despite exposing 128 logical CPUs. Each process starts after ten
idle observations; one-second sampling records foreign GPU processes and CPU
activity over 0.25 cores. All 20 accepted benchmark/profile subprocesses had no
observed foreign work. This sampling cannot exclude shorter interference.

The main sweep uses original/fixed/fixed/original process order. Each process
warms reconstruction, then records seven synchronized wall times per case.
Query timings use seven CUDA-event samples, each comprising ten graph replays
of ten kernel launches, at 1, 4,096, and all input points. Allocations, uploads,
validation, and compilation are outside warmed timings. The composed baseline
uses the existing explicit-inline workaround; the fixed variant uses an ordinary
helper. Raw samples are retained in `bench-*.json` and the associated isolation
files; [summary.json](summary.json) contains pooled medians.

| Reconstruction | Before (ms) | After (ms) | Change |
| --- | ---: | ---: | ---: |
| Bunny, depth 8, float | 100.444 | 100.166 | −0.28% |
| Horse, depth 8, float | 58.511 | 60.195 | +2.88% |
| Eagle, depth 8, float | 78.987 | 78.702 | −0.36% |
| Eagle, depth 9, float | 193.427 | 193.120 | −0.16% |
| Bunny, depth 8, double | 133.334 | 131.793 | −1.16% |

Reconstruction coefficients/isovalues and query outputs agree **bitwise** across
all four processes. The fastest batched field module has the same cache hashes
in both variants. Generated CUDA for composed queries is identical after
normalizing only the exported kernel names; see
[query-codegen.json](query-codegen.json). Full-cloud query timing changes range
from −1.51% to +0.21% for the batch route and −0.27% to +0.33% for composition.

The horse result was investigated rather than discarded. Two further ABBA
rounds gave **57.787 → 58.838 ms (+1.82%)**, with individual ranges
55.87–63.46 and 56.15–62.11 ms. A separate ABBA profile measured GPU activity at
**28.024 → 28.091 ms (+0.24%)** and found zero policy/hash-function calls during
each set of five warmed reconstructions. These observations support timing
variation as the explanation for the residual wall-time difference, rather
than new policy work in the hot path; they are not a blanket performance
guarantee. See [horse-followup.json](horse-followup.json), `horse-*.json`, and
`profile-*.json` for all observations.

The broader policy was also checked outside PSR using Grid3D stiffness assembly
with 35,937 DOFs. Q1 at 32³ cells measured **2.392 → 1.070 ms**; Q2 at 16³ cells
measured **2.758 → 2.767 ms (+0.34%)**. Matrices agree bitwise. These are warmed
assembly wall times, including allocation and dispatch, in ABBA order; see
[fem-summary.json](fem-summary.json) and `fem-*.json`.

First reconstruction calls with cache reads disabled for not-yet-loaded modules
measured **34.748 → 34.547 s** for float and **37.500 → 37.648 s** for double.
The first composed-query calls measured 8.587 → 8.633 s and 8.166 → 8.513 s.
These are single first-call observations including compilation, not statistically
resolved JIT regressions; the composed CUDA differs only in exported names.
Shared caches were never cleared, and later cases reuse already-loaded modules.

Only L40/Linux and the stated compilers were tested. Other GPUs, operating
systems, and CUDA versions remain an upstream CI/review responsibility.

## Reproduction

Use the recorded Warp commit and an existing environment containing its native
build, NumPy, SciPy, pytest, and plyfile. Run from the PSR root. For example, with
this machine's checkout:

```bash
export PYTHONPATH=/home/horde/projects/warp-field-composition:$PWD/src:$PWD
uv run --no-project --python /home/horde/projects/warp-field-composition/.venv/bin/python \
  python docs/results/device-field-composition/validate.py --mode full
```

`validate.py` makes an asserted, temporary substitution into the existing field
tests: public queries pass through ordinary helpers, and the nested two-query
helper loses its explicit inlining hint. It leaves production files unchanged.
Run it with `--mode memcheck` under Compute Sanitizer for the five focused cases.
Use `--mode wheel-policy` with the original PSR environment only to reproduce the
experimental in-memory 1.19 compiler-policy injection.

The original benchmark runners and workers are preserved verbatim. Copy their
Python files to `.build/func-composition/` and run `bench_run.py`, `fem_run.py`,
`horse_run.py`, and `profile_run.py` **sequentially** with the selected Warp
environment. The runners record this machine's Warp checkout path; adjust their
environment line if using another location. They use the existing full bunny,
horse, and eagle depth-8 fixtures under `.build/gpu-baselines/inputs`, with the
eagle depth-9 parameters overridden explicitly. Do not run correctness tests or
other GPU work alongside the benchmarks. Remove regenerated scratch after saving
any new results that need to be retained.
