"""
Plot total logical block volume (constant 86) vs active volume per logical cycle.
Shows how much idle volume is not being used in each cycle.
Also includes reaction-depth plot (parallel PPRs vs logical cycles).
"""

import gzip
import json
from collections import deque

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Constants
TOTAL_LOGICAL_BLOCKS = 86
T_STATE_OVERHEAD = 35 / 2
LOGICAL_BLOCKS_JSON = "logical_blocks.json"


def calculate_sequence_cost(sequence):
    """Total logical blocks = active_volume + T_state_overhead."""
    return sequence["active_volume"] + T_STATE_OVERHEAD


def schedule_sequences(sequences, total_capacity=TOTAL_LOGICAL_BLOCKS):
    """Schedule sequences in JSON order, respecting block constraints. Returns (cycle, num_parallel_ops, used_blocks)."""
    sequence_costs = [(i, calculate_sequence_cost(seq)) for i, seq in enumerate(sequences)]
    schedule = []
    remaining = deque(sequence_costs)

    current_cycle = 0
    while remaining:
        current_used = 0
        parallel_ops = 0
        temp_queue = deque()
        while remaining:
            seq_idx, cost = remaining.popleft()
            if current_used + cost <= total_capacity:
                current_used += cost
                parallel_ops += 1
            else:
                temp_queue.append((seq_idx, cost))
        schedule.append((current_cycle, parallel_ops, current_used))
        remaining = temp_queue
        current_cycle += 1
    return schedule


def load_sequences(filename=LOGICAL_BLOCKS_JSON):
    """Load sequences from a JSON file, or a gzipped JSON-Lines file (.jsonl.gz)."""
    if filename.endswith('.gz'):
        with gzip.open(filename, 'rt') as f:
            return [json.loads(line) for line in f if line.strip()]
    with open(filename) as f:
        return json.load(f)


def main():
    data = load_sequences(LOGICAL_BLOCKS_JSON)

    try:
        plt.style.use("plotstylefile.mplstyle")
    except OSError:
        pass

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 10))

    # ---- Subplot 1: Total vol vs active volume per logical cycle ----
    cycles_av = list(range(len(data)))
    active_volumes = [(seq["active_volume"] + T_STATE_OVERHEAD) / TOTAL_LOGICAL_BLOCKS for seq in data ]
    idle_volumes = [1 - av for av in active_volumes]

    ax1.axhline(y=1, color="lightgray", linewidth=2, linestyle="--", zorder=1)
    ax1.fill_between(cycles_av, 0, 1, alpha=0.15, color="gray", zorder=0)
    ax1.bar(cycles_av, active_volumes, color="steelblue", alpha=0.8, label="Active volume", zorder=2, width=0.9)
    ax1.bar(cycles_av, idle_volumes, bottom=active_volumes, color="none", edgecolor="red", alpha=0.3, linewidth=0.5, label="Idle volume", zorder=2, width=0.9)
    ax1.set_xlabel("Logical cycles")
    ax1.set_ylabel("Circuit volume")
    ax1.legend(loc="upper right")
    ax1.set_ylim(0, 1.05)
    ax1.set_xlim(-0.5, len(cycles_av) - 0.5)

    # ---- Subplot 2: Reaction depth - parallel PPRs vs logical cycles ----
    schedule = schedule_sequences(data)
    cycles_rd = [s[0] for s in schedule]
    parallel_ops = [s[1] for s in schedule]
    used_blocks = [s[2] for s in schedule]

    ax2.plot(cycles_rd, parallel_ops, "o-")
    ax2.set_xlabel("Logical Cycles")
    ax2.set_ylabel("Number of PPRs")
    ax2.set_xlim(left=0)
    ax2.set_ylim(bottom=0, top=max(parallel_ops) * 1.15 if parallel_ops else 1)

    textstr = f"Total Cycles: {len(cycles_rd)}\nTotal Operations: {sum(parallel_ops)}"
    ax2.text(0.98, 0.98, textstr, transform=ax2.transAxes, verticalalignment="top", horizontalalignment="right")

    # ---- Subplot 3: Workspace utilization vs logical cycles ----
    normalized_blocks = [b / TOTAL_LOGICAL_BLOCKS for b in used_blocks]
    ax3.plot(cycles_rd, normalized_blocks, "s-", alpha=0.7)
    ax3.axhline(y=1.0, color="r", linestyle="--", label="Total Capacity")
    ax3.fill_between(cycles_rd, 0, normalized_blocks, alpha=0.3, color="green")
    ax3.set_xlabel("Logical Cycles")
    ax3.set_ylabel("Usage of workspace")
    ax3.set_xlim(left=0)
    ax3.set_ylim(bottom=0, top=1.1)
    ax3.legend()

    plt.tight_layout()
    plt.savefig("total_vol_vs_av.pdf", dpi=150, bbox_inches="tight")
    print("Saved total_vol_vs_av.pdf")


if __name__ == "__main__":
    main()
