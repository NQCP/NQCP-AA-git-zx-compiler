"""
Reaction-limited stalling contours for T-AV transversal circuit (different capacities).

Uses scheduling from bell_pairs_count_t_av.py (transversal-logical-blocks.json).
total_logical_blocks = 100, no T state overhead (already in factory allocation).
tau_c = code cycle time, tau_r = reaction time.

Stalling = sum over cycles k of max(0, tau_r * reaction_depth_k - tau_c)
tau_c range: 10 µs to 100 ms

Contours only (no heatmap), like reaction_limited_different_volume.pdf.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import warnings

# Import scheduling from bell_pairs_count_t_av
from bell_pairs_count_t_av import (
    load_sequences,
    schedule_and_count_bell_pairs,
    available_patches,
)

d = 7  # code distance

warnings.filterwarnings("ignore")

TOTAL_LOGICAL_BLOCKS = 250
SEQUENCES_FILE = "transversal-logical-blocks-heisenberg.json"


def run_for_capacity(sequences, capacity):
    """
    Compute reaction depths for a given capacity.
    Returns dict with reaction_depths_list, n_cycles.
    """
    results = schedule_and_count_bell_pairs(sequences, capacity=capacity)
    reaction_depths_list = [r[5] for r in results]
    return {
        "reaction_depths_list": reaction_depths_list,
        "n_cycles": len(results),
    }


def compute_stalling_grid(reaction_depths, tau_r, tau_c):
    """
    Stalling = sum over k of max(0, tau_r * reaction_depth_k - tau_c)
    """
    grid = np.zeros((len(tau_r), len(tau_c)))
    for i in range(len(tau_r)):
        for j in range(len(tau_c)):
            stalling = sum(
                max(0, tau_r[i] * reaction_depths[k] - tau_c[j])
                for k in range(len(reaction_depths))
            )
            grid[i, j] = stalling
    return grid


def get_no_stalling_boundary(reaction_depths, tau_c):
    """
    No stalling when tau_r * rd_k <= tau_c for all k.
    So tau_r <= tau_c / max(rd_k). Boundary: tau_r = tau_c / max(rd_k).
    """
    max_rd = max(reaction_depths) if reaction_depths else 0
    if max_rd > 0:
        return tau_c / max_rd
    return np.full_like(tau_c, np.nan)


def main():
    sequences = load_sequences(SEQUENCES_FILE)
    print(f"Loaded {len(sequences)} sequences from {SEQUENCES_FILE}")
    print(f"Base capacity (100%): {available_patches} patches")

    # Four capacities: 100%, 75%, 50%, 25% of available_patches
    pct_100 = 250.0
    pct_min = 170.0
    step = (pct_100 - pct_min) / 3
    percentages_display = [pct_100, pct_100 - step, pct_100 - 2 * step, pct_min]
    capacities = [max(1, int(available_patches * p / 100)) for p in percentages_display]

    results = []
    for pct, cap in zip(percentages_display, capacities):
        r = run_for_capacity(sequences, float(cap))
        r["pct"] = pct
        r["capacity"] = cap
        results.append(r)
        rd_list = r["reaction_depths_list"]
        active = [x for x in rd_list if x > 0]
        print(f"\n--- {pct:.1f}% capacity ({cap} patches) ---")
        print(f"  Code cycles: {r['n_cycles']}")
        if active:
            print(f"  Reaction depth: min={min(active)}, max={max(active)}, avg={np.mean(active):.1f}")

    # tau_c: 10 µs to 100 ms
    tau_c = np.logspace(1, 5, 100)  # 10 to 100000 µs
    # tau_r: cover no-stalling boundary range
    tau_r = np.logspace(-1, 4, 100)  # 0.1 to 10000 µs
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
        stalling_grid = compute_stalling_grid(r["reaction_depths_list"], tau_r, tau_c)
        tau_r_boundary = get_no_stalling_boundary(r["reaction_depths_list"], tau_c_line)
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
        # No stalling boundary
        ax.plot(
            tau_c_line[mask], tau_r_boundary[mask],
            color=colors[idx], linewidth=2.5, linestyle="-",
            zorder=3,
        )

    ax.set_xlabel(r"$\tau_c$ (µs)")
    ax.set_ylabel(r"$\tau_r$ (µs)")
    ax.set_xlim(tau_c.min(), tau_c.max())
    ax.set_ylim(tau_r.min(), tau_r.max())

    # Two legends: linestyle for stalling, then capacity/cycles as color
    stalling_104 = Line2D([0], [0], color="gray", linestyle=":", linewidth=2, label=r"Stalling time = $10^4$ µs")
    stalling_105 = Line2D([0], [0], color="gray", linestyle="--", linewidth=2, label=r"Stalling time = $10^5$ µs")
    no_stalling = Line2D([0], [0], color="gray", linestyle="-", linewidth=2.5, label="No stalling boundary")
    vol_handles = [
        Patch(facecolor=colors[idx], alpha=0.25, edgecolor=colors[idx], label=f"{r['pct']:.0f}% cap, {r['n_cycles']/d:.0f} cycles")
        for idx, r in enumerate(results)
    ]
    leg1 = ax.legend(
        handles=[stalling_104, stalling_105, no_stalling],
        labels=[r"Stalling time = $10^4$ µs", r"Stalling time = $10^5$ µs", "No stalling boundary"],
        loc="upper center", bbox_to_anchor=(0.25, -0.16), frameon=True, ncol=1,
    )
    ax.add_artist(leg1)
    ax.legend(
        handles=vol_handles,
        labels=[f"{r['pct']:.0f}% cap, {r['n_cycles']/d:.0f} cycles" for r in results],
        loc="upper center", bbox_to_anchor=(0.75, -0.16), frameon=True, ncol=1,
    )

    plt.tight_layout(rect=[0, 0.12, 1, 1])
    plt.savefig("t_av_reaction_limited_different_volume_heisenberg.pdf", bbox_inches="tight", pad_inches=0.1)
    print("\nSaved t_av_reaction_limited_different_volume_heisenberg.pdf")


if __name__ == "__main__":
    main()
