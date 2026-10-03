#!/usr/bin/env python3
"""Plot validated CPU/CUDA application latencies and GPU speed-ups."""

import argparse
import csv
import json
import math
import sys
from pathlib import Path


def main():
    """Export editable vector figures, 300-dpi images and a comparison CSV."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument(
        "--style-dir",
        type=Path,
        default=Path.home() / ".codex/skills/graph-plotting",
    )
    args = parser.parse_args()
    sys.path.insert(0, str(args.style_dir / "scripts"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from mpl_style import audit_figure, finish_axis, publication_style, save_figure

    out = args.results.resolve()
    complete = json.loads((out / "complete.json").read_text())
    if complete.get("status") != "passed":
        parser.error("Plotting requires a complete, validated comparison")
    rows = json.loads((out / "summary.json").read_text())
    if len(rows) != complete["cases"]:
        parser.error("Summary case count differs from the validated completion record")
    labels = {
        "square1024": "Square 1024",
        "square4096": "Square 4096",
        "square8192": "Square 8192",
        "transpose": "Transpose",
        "gram": "Gram",
        "mlp": "MLP",
        "attention": "Attention",
        "backward": "Forward + backward",
    }
    dtypes = list(dict.fromkeys(row["dtype"] for row in rows))
    workloads = list(dict.fromkeys(row["workload"] for row in rows))
    methods = [
        ("camblas", "CAMBLAS CPU (64 cores)", "#4C78A8", "o"),
        ("nvpl", "NVPL CPU (64 cores)", "#595959", "D"),
        ("cuda_resident", "PyTorch GH200: resident", "#E45756", "s"),
        ("cuda_transfer", "PyTorch GH200: with transfers", "#54A24B", "^"),
    ]
    threads = rows[0]["threads"]
    methods = [
        (key, label.replace("64 cores", f"{threads} cores"), colour, marker)
        for key, label, colour, marker in methods
    ]
    all_latencies = [
        value
        for row in rows
        for interval in row["ranges_ms"].values()
        for value in interval
    ]
    lower = 10 ** math.floor(math.log10(min(all_latencies)))
    upper = 10 ** math.ceil(math.log10(max(all_latencies)))
    audits = {}
    with publication_style(overrides={"mathtext.cal": "Helvetica Neue"}):
        figure, axes = plt.subplots(
            1,
            len(dtypes),
            figsize=(3.6 * len(dtypes), 4.0),
            sharey=True,
            squeeze=False,
            constrained_layout=True,
        )
        for axis, dtype in zip(axes[0], dtypes):
            subset = {row["workload"]: row for row in rows if row["dtype"] == dtype}
            y = np.arange(len(workloads))
            for index, (key, label, colour, marker) in enumerate(methods):
                centre = np.array(
                    [subset[workload]["medians_ms"][key] for workload in workloads]
                )
                low = np.array(
                    [subset[workload]["ranges_ms"][key][0] for workload in workloads]
                )
                high = np.array(
                    [subset[workload]["ranges_ms"][key][1] for workload in workloads]
                )
                axis.errorbar(
                    centre,
                    y + (index - 1.5) * 0.17,
                    xerr=np.vstack((centre - low, high - centre)),
                    fmt=marker,
                    markersize=3.5,
                    color=colour,
                    label=label,
                    linestyle="none",
                    capsize=1.5,
                    elinewidth=0.7,
                )
            axis.set(
                xscale="log",
                xlim=(lower, upper),
                xlabel="Latency (ms; lower is better)",
            )
            axis.set_yticks(y, [labels[workload] for workload in workloads])
            axis.set_ylim(len(workloads) - 0.5, -0.9)
            axis.text(
                0.97,
                0.98,
                dtype.replace("float", "FP"),
                transform=axis.transAxes,
                ha="right",
                va="top",
                fontsize=8,
            )
            axis.grid(
                axis="x", which="major", color="#cccccc", alpha=0.45, linewidth=0.5
            )
            finish_axis(axis)
        handles, legend_labels = axes[0, 0].get_legend_handles_labels()
        figure.legend(
            handles, legend_labels, loc="outside upper center", ncol=2, frameon=False
        )
        audits["latency"] = audit_figure(figure)
        if audits["latency"]:
            raise RuntimeError("\n".join(audits["latency"]))
        save_figure(figure, out / "cpu_vs_gpu_latency")
        plt.close(figure)

        figure, axes = plt.subplots(
            1,
            len(dtypes),
            figsize=(3.6 * len(dtypes), 4.0),
            sharey=True,
            squeeze=False,
            constrained_layout=True,
        )
        ratios = []
        for axis, dtype in zip(axes[0], dtypes):
            subset = {row["workload"]: row for row in rows if row["dtype"] == dtype}
            y = np.arange(len(workloads))
            for index, (key, label, colour, marker) in enumerate(methods[2:]):
                centre, low, high = [], [], []
                for workload in workloads:
                    row = subset[workload]
                    value = row["medians_ms"]
                    centre.append(value["camblas"] / value[key])
                    paired = [
                        cpu / gpu
                        for cpu, gpu in zip(
                            row["round_medians_ms"]["camblas"],
                            row["round_medians_ms"][key],
                        )
                    ]
                    low.append(min(paired))
                    high.append(max(paired))
                centre, low, high = map(np.array, (centre, low, high))
                ratios.extend([*low, *high])
                axis.errorbar(
                    centre,
                    y + (index - 0.5) * 0.25,
                    xerr=np.maximum(0, np.vstack((centre - low, high - centre))),
                    fmt=marker,
                    markersize=3.5,
                    color=colour,
                    label=label,
                    linestyle="none",
                    capsize=1.5,
                    elinewidth=0.7,
                )
            axis.axvline(1, color="#595959", linestyle="--", linewidth=0.7, zorder=0)
            axis.set(xscale="log", xlabel="GPU speed-up over CAMBLAS (×)")
            axis.set_yticks(y, [labels[workload] for workload in workloads])
            axis.set_ylim(len(workloads) - 0.5, -0.9)
            axis.text(
                0.97,
                0.98,
                dtype.replace("float", "FP"),
                transform=axis.transAxes,
                ha="right",
                va="top",
                fontsize=8,
            )
            axis.grid(
                axis="x", which="major", color="#cccccc", alpha=0.45, linewidth=0.5
            )
            finish_axis(axis)
        for axis in axes[0]:
            ratio_lower, ratio_upper = min(1, *ratios) * 0.8, max(1, *ratios) * 1.15
            axis.set_xlim(ratio_lower, ratio_upper)
            ticks = [
                value
                for exponent in range(-3, 6)
                for value in (10.0**exponent, 2 * 10.0**exponent, 5 * 10.0**exponent)
                if ratio_lower <= value <= ratio_upper
            ]
            axis.set_xticks(ticks, [f"{value:g}" for value in ticks])
            axis.tick_params(axis="x", which="minor", labelbottom=False)
        handles, legend_labels = axes[0, 0].get_legend_handles_labels()
        figure.legend(
            handles, legend_labels, loc="outside upper center", ncol=2, frameon=False
        )
        audits["speedup"] = audit_figure(figure)
        if audits["speedup"]:
            raise RuntimeError("\n".join(audits["speedup"]))
        save_figure(figure, out / "gpu_speedup")
        plt.close(figure)
    (out / "figure_audit.json").write_text(json.dumps(audits, indent=2) + "\n")
    with (out / "comparison.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "workload",
                "precision",
                "cpu_cores",
                "backend",
                "median_ms",
                "min_round_median_ms",
                "max_round_median_ms",
            ]
        )
        for row in rows:
            for key, _, _, _ in methods:
                writer.writerow(
                    [
                        row["workload"],
                        row["dtype"],
                        row["threads"],
                        key,
                        row["medians_ms"][key],
                        *row["ranges_ms"][key],
                    ]
                )
    print(f"Saved figures and CSV to {out}")


if __name__ == "__main__":
    main()
