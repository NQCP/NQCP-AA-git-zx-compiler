"""
Reaction depth and bell pairs vs. percentage of quantum computer used.

Uses logic from bell_pairs_count.py with variable capacity (percentage of 86 blocks).
Computes reaction_depths_list and total_bell_pairs for 4 different volume percentages,
then plots stalling-time heat maps (tau_c vs tau_r) for each, like reaction-limited.py.

Minimum percentage = (max sequence cost over all sequences) / 86 * 100,
where sequence cost = active_volume + T_state_overhead.
"""

import gzip
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from collections import deque, Counter
import warnings

warnings.filterwarnings("ignore")

# Constants (match bell_pairs_count.py)
TOTAL_LOGICAL_BLOCKS = 86
T_STATE_OVERHEAD = 35 / 2
LOGICAL_BLOCKS_FILE = "logical_blocks.json"


def load_sequences(filename=LOGICAL_BLOCKS_FILE):
    """Load sequences from a JSON file, or a gzipped JSON-Lines file (.jsonl.gz)."""
    if filename.endswith('.gz'):
        with gzip.open(filename, 'rt') as f:
            return [json.loads(line) for line in f if line.strip()]
    with open(filename) as f:
        return json.load(f)


def calculate_sequence_cost(sequence):
    """Total logical blocks = active_volume + T_state_overhead."""
    return sequence["active_volume"] + T_STATE_OVERHEAD


def count_internal_bell_pairs(sequence):
    """Count bell pairs within a sequence from hexagon ports (b-prefixed)."""
    bell_pairs = set()
    for hexagon in sequence.get("hexagons", []):
        for port_val in hexagon.get("ports", {}).values():
            if isinstance(port_val, str) and port_val.startswith("b"):
                bell_pairs.add(port_val)
    return len(bell_pairs)


def get_qubits_used(sequence):
    """Extract set of qubit identifiers used by a sequence (q-prefixed)."""
    qubits = set()
    for hexagon in sequence.get("hexagons", []):
        for port_val in hexagon.get("ports", {}).values():
            if isinstance(port_val, str) and port_val.startswith("q"):
                qubits.add(port_val)
    return qubits


def schedule_sequences_with_indices(sequences, total_capacity):
    """
    Schedule sequences in JSON order with given capacity.
    Returns list of (cycle, list_of_seq_indices, used_blocks).
    """
    sequence_costs = [(i, calculate_sequence_cost(seq)) for i, seq in enumerate(sequences)]
    schedule = []
    remaining = deque(sequence_costs)
    current_cycle = 0
    while remaining:
        current_used = 0
        scheduled_indices = []
        temp_queue = deque()
        while remaining:
            seq_idx, cost = remaining.popleft()
            if current_used + cost <= total_capacity:
                current_used += cost
                scheduled_indices.append(seq_idx)
            else:
                temp_queue.append((seq_idx, cost))
        schedule.append((current_cycle, scheduled_indices, current_used))
        remaining = temp_queue
        current_cycle += 1
    return schedule


def count_bell_pairs_per_cycle_with_capacity(sequences, total_capacity):
    """
    Count total bell pairs per logical cycle when using total_capacity blocks.
    Returns list of (cycle, total_bell_pairs, internal, cross_sequence, l).
    """
    internal_bell_pairs = [count_internal_bell_pairs(seq) for seq in sequences]
    qubits_per_sequence = [get_qubits_used(seq) for seq in sequences]
    schedule = schedule_sequences_with_indices(sequences, total_capacity)
    result = []
    for cycle, seq_indices, used_blocks in schedule:
        internal = sum(internal_bell_pairs[i] for i in seq_indices)
        qubit_counts = Counter()
        for seq_idx in seq_indices:
            for q in qubits_per_sequence[seq_idx]:
                qubit_counts[q] += 1
        cross_sequence = sum(c - 1 for c in qubit_counts.values() if c > 1)
        total = internal + cross_sequence
        result.append((cycle, total, internal, cross_sequence, len(seq_indices)))
    return result


def get_reaction_depth_per_cycle(bell_pairs_result):
    """Reaction depth per cycle: l-1 if internal==0 else l."""
    result = []
    for cycle, total, internal, cross_sequence, l in bell_pairs_result:
        reaction_depth = l - 1 if internal == 0 else l
        result.append((cycle, reaction_depth))
    return result


def run_for_capacity(sequences, total_capacity):
    """
    Compute bell pairs and reaction depths for a given capacity.
    Returns dict with reaction_depths_list, total_bell_pairs (list), n_cycles, total_bell_pairs_sum.
    """
    bell_pairs = count_bell_pairs_per_cycle_with_capacity(sequences, total_capacity)
    reaction_depths = get_reaction_depth_per_cycle(bell_pairs)
    reaction_depths_list = [r[1] for r in reaction_depths]
    total_bell_pairs = [r[1] for r in bell_pairs]
    return {
        "reaction_depths_list": reaction_depths_list,
        "total_bell_pairs": total_bell_pairs,
        "n_cycles": len(bell_pairs),
        "total_bell_pairs_sum": sum(total_bell_pairs),
    }


def compute_min_percentage(sequences):
    """Minimum feasible percentage = (max active_volume + T_state_overhead) / 86 * 100."""
    max_cost = max(calculate_sequence_cost(seq) for seq in sequences)
    min_pct = max_cost / TOTAL_LOGICAL_BLOCKS * 100
    return min_pct, max_cost


def compute_stalling_grid(reaction_depths, d, tau_r, tau_c):
    """Compute stalling time grid for given reaction depths and tau ranges."""
    stalling_time_grid = np.zeros((len(tau_r), len(tau_c)))
    for i in range(len(tau_r)):
        for j in range(len(tau_c)):
            stalling_time = sum(
                max(0, (reaction_depths[k] * tau_r[i]) - d * tau_c[j])
                for k in range(len(reaction_depths))
            )
            stalling_time_grid[i, j] = stalling_time
    return stalling_time_grid


def get_no_stalling_boundary(reaction_depths, d, tau_c):
    """Get tau_r values for the no-stalling boundary at given tau_c points.

    Stalling = sum max(0, rd_k*tau_r - d*tau_c). For stalling=0 we need rd_k*tau_r <= d*tau_c
    for all k, so tau_r <= d*tau_c/max(rd_k). Boundary: tau_r = d*tau_c/max(rd_k).
    """
    max_rd = max(reaction_depths) if reaction_depths else 0
    if max_rd > 0:
        tau_r_boundary = d * tau_c / max_rd
    else:
        tau_r_boundary = np.full_like(tau_c, np.nan)
    return tau_r_boundary


def main():
    sequences = load_sequences()
    print(f"Loaded {len(sequences)} sequences from {LOGICAL_BLOCKS_FILE}")

    min_pct, max_cost = compute_min_percentage(sequences)
    print(f"Max sequence cost (active_volume + T_state_overhead): {max_cost:.1f} blocks")
    print(f"Minimum feasible percentage: {min_pct:.1f}%")

    # Four percentages: 100%, then evenly down to min_pct
    pct_100 = 100.0
    pct_min = min_pct
    step = (pct_100 - pct_min) / 3 if pct_100 > pct_min else 0
    percentages_display = (
        [pct_100, pct_100 - step, pct_100 - 2 * step, pct_min] if step > 0 else [pct_100] * 4
    )
    capacities = [
        max(int(TOTAL_LOGICAL_BLOCKS * p / 100), int(np.ceil(max_cost))) for p in percentages_display
    ]

    results = []
    for pct, cap in zip(percentages_display, capacities):
        r = run_for_capacity(sequences, float(cap))
        r["pct"] = pct
        r["capacity"] = cap
        results.append(r)
        print(f"\n--- {pct:.1f}% volume (capacity={cap} blocks) ---")
        print(f"  Logical cycles: {r['n_cycles']}")
        print(f"  Total bell pairs (all cycles): {r['total_bell_pairs_sum']}")
        print(f"  Reaction depth: min={min(r['reaction_depths_list'])}, max={max(r['reaction_depths_list'])}, avg={np.mean(r['reaction_depths_list']):.1f}")

    # Combined plot: contours + no-stalling boundaries (log scale)
    d = 10
    tau_r = np.logspace(1.5, 3, 100)
    tau_c = np.logspace(-2, 2, 100)
    tau_c_line = np.logspace(np.log10(tau_c.min()), np.log10(tau_c.max()), 200)

    try:
        plt.style.use("plotstylefile.mplstyle")
    except OSError:
        pass

    colors = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3"]

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_xscale("log")
    ax.set_yscale("log")
    X, Y = np.meshgrid(tau_c, tau_r)

    for idx, r in enumerate(results):
        stalling_grid = compute_stalling_grid(r["reaction_depths_list"], d, tau_r, tau_c)
        tau_r_boundary = get_no_stalling_boundary(r["reaction_depths_list"], d, tau_c_line)
        mask = (tau_r_boundary >= tau_r.min()) & (tau_r_boundary <= tau_r.max())

        # Shade no-stalling region (below boundary) with transparent color
        y_top = np.clip(tau_r_boundary, tau_r.min(), tau_r.max())
        ax.fill_between(
            tau_c_line, tau_r.min(), y_top,
            alpha=0.25, color=colors[idx], zorder=0,
        )

        # Contour at 10^4 (dotted) and 10^5 (dashed)
        ax.contour(
            X, Y, stalling_grid,
            levels=[1e4],
            colors=[colors[idx]],
            linewidths=2,
            linestyles=":",
            alpha=0.9,
            zorder=2,
        )
        ax.contour(
            X, Y, stalling_grid,
            levels=[1e5],
            colors=[colors[idx]],
            linewidths=2,
            linestyles="--",
            alpha=0.9,
            zorder=2,
        )
        # No stalling boundary (volume and cycles)
        ax.plot(
            tau_c_line[mask], tau_r_boundary[mask],
            color=colors[idx], linewidth=2.5, linestyle="-",
            zorder=3,
        )

    ax.set_xlabel(r"$\tau_c$ (µs)")
    ax.set_ylabel(r"$\tau_r$ (µs)")
    # Build legend: linestyle for stalling time, no stalling region, then volume/cycles as color boxes
    stalling_104 = Line2D([0], [0], color="gray", linestyle=":", linewidth=2, label=r"Stalling time = $10^4$ µs")
    stalling_105 = Line2D([0], [0], color="gray", linestyle="--", linewidth=2, label=r"Stalling time = $10^5$ µs")
    no_stalling = Line2D([0], [0], color="gray", linestyle="-", linewidth=2.5, label="No stalling boundary")
    vol_handles = [
        Patch(facecolor=colors[idx], alpha=0.25, edgecolor=colors[idx], label=f"{r['pct']:.1f}% vol, {r['n_cycles']} cycles")
        for idx, r in enumerate(results)
    ]
    vol_labels = [f"{r['pct']:.1f}% vol, {r['n_cycles']} cycles" for r in results]
    # Two legends: column 1 = stalling, column 2 = vol/cycles
    leg1 = ax.legend(
        handles=[stalling_104, stalling_105, no_stalling],
        labels=[r"Stalling time = $10^4$ µs", r"Stalling time = $10^5$ µs", "No stalling boundary"],
        loc="upper center", bbox_to_anchor=(0.25, -0.16), frameon=True, ncol=1,
    )
    ax.add_artist(leg1)
    ax.legend(
        handles=vol_handles,
        labels=vol_labels,
        loc="upper center", bbox_to_anchor=(0.75, -0.16), frameon=True, ncol=1,
    )
    ax.set_xlim(tau_c.min(), tau_c.max())
    ax.set_ylim(tau_r.min(), tau_r.max())
    # ax.grid(True, which="both", alpha=0.3)

    plt.tight_layout(rect=[0, 0.12, 1, 1])
    plt.savefig("reaction_limited_different_volume.pdf", bbox_inches="tight", pad_inches=0.1)
    print("\nSaved reaction_limited_different_volume.pdf")
    # plt.show()


if __name__ == "__main__":
    main()
