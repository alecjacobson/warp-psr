# Depth-9 measurement audit scripts

These are exact snapshots of the one-off November 6, 2026 orchestration, retained
to audit the hashes in the result reports. They contain the measurement VM's
absolute paths and are not portable command-line tools. Use the reproduction
commands in [the performance guide](../../performance.md#reproduce-a-depth-9-row)
for ordinary reruns, or adapt paths explicitly when repeating this orchestration.

`recheck_eagle.py` runs two ABBA rounds against the earlier production revision,
then a fresh four-thread CPU baseline. `benchmark_cpu.py` brackets the unchanged
reference solve while reading the immutable input directly, avoiding a redundant
ASCII input copy. `run_depth9.py` checks exact input/parameter identity, generates
fresh serial reference exports, applies the unchanged GPU and OpenMP acceptance
gates, and measures each passing row in isolation.

`isolation.initial.py` was used by the completed eagle recheck. The initial matrix
runner completed eagle SPSR and both eagle PSR accuracy checks before unrelated
CPU/GPU work appeared. It was held before timing, then resumed with accepted
results and fixture hashes revalidated. The first resumed timing attempt detected
contention; both of its timing reports are conservatively rejected because its
in-memory activity observations were not serialized before the gate failed.
The final `isolation.py` waits for ten idle observations before each measured
child. Every subsequent child has an immediate persistent activity audit;
contended attempts are retained separately and retried after waiting for idle.
Initial and intermediate snapshots and scheduling checkpoints are retained. No
reconstruction source or numerical acceptance criterion changed during the sweep.

After five contended full-torso SPSR CPU batches, the next attempt was stopped
and excluded so the remaining datasets could proceed. Its available stdout and
the operator-abort reason are retained in that profile. The final runner retains
accepted GPU/CPU measurements independently and defers a row after three newly
contended attempts, returning to pending rows after independent cases. This
changes scheduling only; medians, sample counts and isolation thresholds remain
unchanged. The torso's accepted GPU measurement was preserved when its CPU batch
was deferred. Pending raw scratch stayed available until the row completed.

Eight torso CPU batches were ultimately rejected for observed activity. The
final `cpu_per_process.py` isolates each invocation separately, using
`benchmark_cpu_single.py` with the same binary, input, parameters and unchanged
`Solver::Solve` timer. It discards the first accepted invocation and takes the
median of the first five subsequent clean invocations. It retains every accepted
and rejected process report and activity audit, then supplies the ordinary
aggregate CPU report to the matrix runner. No samples are selected by speed.

Only disposable, reproducible reference exports used `/opt` scratch. Source
inputs, repositories, Git history, manifests, full acceptance reports, timing
samples and output hashes remained under persistent `/home/horde`. Each scratch
case was deleted only after successful validation and persistent result capture.
