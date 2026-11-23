import csv
import io
import os
import statistics
import subprocess
import sys
from collections import defaultdict
from pathlib import Path


def main():
    build, results = map(Path, sys.argv[1:])
    cpu = min(os.sched_getaffinity(0))
    os.sched_setaffinity(0, {cpu})
    samples = defaultdict(lambda: defaultdict(list))
    with (results / "samples.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow([
            "round", "variant", "case", "metric", "dimension", "candidates",
            "subquantizers", "ns_per_call",
        ])
        for iteration in range(7):
            order = ("baseline", "neon") if iteration % 2 == 0 else ("neon", "baseline")
            for variant in order:
                raw = subprocess.check_output([str(build / variant)], text=True)
                for case, metric, d, ny, m, ns in csv.reader(io.StringIO(raw)):
                    samples[(case, metric, int(d), int(ny), int(m))][variant].append(float(ns))
                    writer.writerow([iteration + 1, variant, case, metric, d, ny, m, ns])
            print(f"Completed paired round {iteration + 1}/7 on logical CPU {cpu}", flush=True)

    lines = [
        "| Case | Metric | Dimension | Candidates | Subquantizers | Baseline ns | NEON ns | Speedup |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    with (results / "summary.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow([
            "case", "metric", "dimension", "candidates", "subquantizers",
            "baseline_median_ns", "neon_median_ns", "median_paired_speedup",
            "min_paired_speedup", "max_paired_speedup",
        ])
        for key, sample in sorted(samples.items()):
            baseline, neon = sample["baseline"], sample["neon"]
            assert len(baseline) == len(neon) == 7
            base_ns, neon_ns = map(statistics.median, (baseline, neon))
            ratios = [b / n for b, n in zip(baseline, neon)]
            speedup = statistics.median(ratios)
            writer.writerow([*key, base_ns, neon_ns, speedup, min(ratios), max(ratios)])
            row = " | ".join(map(str, key))
            lines.append(f"| {row} | {base_ns:.2f} | {neon_ns:.2f} | {speedup:.3f}x |")
    summary = "\n".join(lines) + "\n"
    print(summary)
    (results / "summary.md").write_text(summary)


if __name__ == "__main__":
    main()
