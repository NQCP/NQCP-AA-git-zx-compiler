"""
Compare parallelism / utilization profiles across multiple TOTAL_LOGICAL_BLOCKS
budgets. Overlays the rolling-mean curves for each capacity so the temporal
profile shifts (and the runtime/parallelism tradeoff) are visible in one plot.

Reuses the loader, scheduler, and _rolling_stats from reaction_depth.py — so
any changes to those (strict-FIFO scheduling, stream loading, etc.) are
inherited automatically.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ppr_parallelization import (
    load_sequences,
    schedule_sequences,
    _rolling_stats,
    T_STATE_OVERHEAD,
)


# Workspace capacities (logical blocks) to overlay. Pick log-ish spacing that
# covers from "barely any parallelism" up to "workspace not the bottleneck".
CAPACITIES = [500, 1000, 2000]

circuit = "fermi_hubbard_2d_step_s4_universal_paulis_commuted"
input_dir = "logical_network_files"
output_dir = "paper_plots"
LOGICAL_BLOCKS_FILE = f"{input_dir}/logical_blocks_{circuit}.jsonl.gz"
OUTPUT_FILE = f"{output_dir}/ppr_parallelization_different_volume.pdf"


def run_for_capacity(sequences, capacity):
    """Schedule once at the given capacity; return parallel_ops and util arrays."""
    schedule = schedule_sequences(sequences, total_capacity=capacity)
    parallel_ops = np.asarray([s[1] for s in schedule])
    used_blocks = np.asarray([s[2] for s in schedule])
    util = used_blocks / capacity
    return schedule, parallel_ops, util


def plot_capacity_overlay(sequences, capacities, output_file=OUTPUT_FILE):
    try:
        plt.style.use("plotstylefile.mplstyle")
    except OSError:
        pass

    cmap = plt.get_cmap("viridis")
    n = len(capacities)
    colors = [cmap(i / max(1, n - 1)) for i in range(n)]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    summary_rows = []
    max_cycles = 0

    for capacity, color in zip(capacities, colors):
        print(f"\n--- Scheduling at capacity = {capacity} ---")
        schedule, po, util = run_for_capacity(sequences, capacity)
        n_cycles = len(schedule)
        max_cycles = max(max_cycles, n_cycles)

        # Window size per curve: aim for ~500 windows
        window = max(1, n_cycles // 500)

        x, mean, p05, p25, p75, p95 = _rolling_stats(po, window)
        ax1.fill_between(x, p25, p75, alpha=0.15, color=color, linewidth=0)
        ax1.plot(
            x, mean, color=color, lw=1.6,
            label=f"capacity={capacity}  ({n_cycles:,} cycles, mean l={po.mean():.2f})",
        )

        x, mean, p05, p25, p75, p95 = _rolling_stats(util, window)
        ax2.fill_between(x, p25, p75, alpha=0.15, color=color, linewidth=0)
        ax2.plot(x, mean, color=color, lw=1.6)

        summary_rows.append((capacity, n_cycles, float(po.mean()), float(po.max()),
                             float(util.mean())))

    ax1.set_ylabel("PPRs per cycle")
    ax1.set_ylim(bottom=0)
    ax1.set_xlim(0, max_cycles)
    ax1.legend(loc="upper right", fontsize=9)

    ax2.axhline(y=1.0, color="r", linestyle="--", alpha=0.6, label="capacity")
    ax2.set_xlabel("Logical cycle")
    ax2.set_ylabel("Workspace utilization")
    ax2.set_xlim(0, max_cycles)
    ax2.set_ylim(bottom=0)  # let top auto-scale (small capacities can exceed 1.0)
    ax2.legend(loc="upper right", fontsize=9)

    plt.tight_layout()
    plt.savefig(output_file, bbox_inches="tight", pad_inches=0.1)
    print(f"\nPlot saved to {output_file}")
    plt.close("all")

    print("\n" + "=" * 70)
    print("SUMMARY: parallelism / utilization vs capacity")
    print("=" * 70)
    print(f"{'capacity':>10}  {'cycles':>12}  {'mean l':>8}  {'max l':>6}  {'mean util':>10}")
    for cap, nc, ml, mxl, mu in summary_rows:
        print(f"{cap:>10}  {nc:>12,}  {ml:>8.2f}  {mxl:>6.0f}  {mu:>10.2f}")
    print("=" * 70)


if __name__ == "__main__":
    print(f"Loading sequences from {LOGICAL_BLOCKS_FILE}...")
    sequences = load_sequences(LOGICAL_BLOCKS_FILE)
    print(f"T_state_overhead (inherited from reaction_depth.py): {T_STATE_OVERHEAD}")
    plot_capacity_overlay(sequences, CAPACITIES)
