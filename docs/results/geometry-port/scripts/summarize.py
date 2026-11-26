"""Summarize fixed, load-screened comparison groups without selecting by timing."""
import argparse
import json
from pathlib import Path
import statistics

parser = argparse.ArgumentParser()
parser.add_argument("--directory", type=Path, default=Path(__file__).parent)
parser.add_argument("--runs", nargs="+", type=int, default=[8, 9, 10, 11])
parser.add_argument("--output", type=Path)
args = parser.parse_args()
runs = {}
for index in args.runs:
    matches = list(args.directory.glob(f"bench-{index}-*.json"))
    paths = [path for path in matches if not path.name.endswith("-isolation.json")]
    if len(paths) != 1:
        raise ValueError(f"Expected one report for run {index}: {paths}")
    path = paths[0]
    isolation = json.loads(path.with_name(path.stem + "-isolation.json").read_text())
    if isolation["foreign_work_observed"]:
        raise ValueError(f"Foreign work was observed in run {index}")
    runs[index] = json.loads(path.read_text())

def metric(before, after, multiplier=1.0):
    baseline = statistics.median(before) * multiplier
    candidate = statistics.median(after) * multiplier
    return dict(baseline_ms=baseline, geometry_ms=candidate,
                delta_ms=candidate-baseline, change_percent=100*(candidate/baseline-1),
                baseline_samples=len(before), geometry_samples=len(after))

groups = {variant: [run for run in runs.values() if run["variant"] == variant]
          for variant in ("original", "port")}
if len(groups["original"]) != len(groups["port"]):
    raise ValueError("Use equal process counts for both variants")
summary = dict(selected_runs=args.runs, selection="Fixed run indices; require no observed foreign work; do not select by timing.",
               bitwise_outputs_equal=True, cases=[], throttling={})
for index in range(5):
    cases = {variant: [run["cases"][index] for run in values] for variant, values in groups.items()}
    example = cases["original"][0]
    identity = (example["dataset"], example["depth"], example["precision"], example["points"])
    if any((case["dataset"], case["depth"], case["precision"], case["points"]) != identity
           for values in cases.values() for case in values):
        raise ValueError("Case order differs")
    if len({case["reconstruction_sha256"] for values in cases.values() for case in values}) != 1:
        raise ValueError(f"Reconstruction mismatch: {identity}")
    row = dict(dataset=identity[0], depth=identity[1], precision=identity[2], points=identity[3],
               reconstruction=metric(*[[sample for case in cases[variant] for sample in case["reconstruction_seconds"]]
                                       for variant in groups], multiplier=1000), queries=[])
    for query_index, query in enumerate(example["queries"]):
        queries = {variant: [case["queries"][query_index] for case in values] for variant, values in cases.items()}
        if len({query["sha256"] for values in queries.values() for query in values}) != 1:
            raise ValueError(f"Query mismatch: {identity}, {query_index}")
        query_row = dict(count=query["count"])
        for name in ("batch", "composed"):
            query_row[name] = metric(*[[sample for item in queries[variant] for sample in item["kernel_ms"][name]]
                                      for variant in groups])
        query_row["batch_host"] = metric(*[[sample for item in queries[variant] for sample in item["batch_host_seconds"]]
                                           for variant in groups], multiplier=1000)
        row["queries"].append(query_row)
    summary["cases"].append(row)
for index, run in runs.items():
    before = run["environment_before"]["cgroup_cpu_counters"]
    after = run["environment_after"]["cgroup_cpu_counters"]
    summary["throttling"][str(index)] = [
        {key: new["counters"][key]-old["counters"][key]
         for key in ("usage_usec", "nr_periods", "nr_throttled", "throttled_usec")}
        for old, new in zip(before[:2], after[:2])]
output = args.output or args.directory / "performance-summary.json"
output.write_text(json.dumps(summary, indent=2)+"\n")
for row in summary["cases"]:
    print(row["dataset"], row["depth"], row["precision"], row["reconstruction"])
    for query in row["queries"]:
        print(" ", query["count"], {name: round(query[name]["change_percent"], 3) for name in ("batch", "composed", "batch_host")},
              "host delta us", round(query["batch_host"]["delta_ms"]*1000, 3))
print("Throttling:", summary["throttling"])
