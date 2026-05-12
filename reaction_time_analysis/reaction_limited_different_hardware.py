"""
Side-by-side plot: reaction-limited-different-volume and t-av-reaction-limited-different-volume.

No logic changes - just combines both plots as two subplots.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import importlib.util
import warnings

warnings.filterwarnings("ignore")

# Load modules by file path (handles hyphens in filenames)
_script_dir = os.path.dirname(os.path.abspath(__file__))

spec_rldv = importlib.util.spec_from_file_location(
    "rldv",
    os.path.join(_script_dir, "reaction-limited-different-volume.py")
)
rldv = importlib.util.module_from_spec(spec_rldv)
spec_rldv.loader.exec_module(rldv)

spec_tav = importlib.util.spec_from_file_location(
    "t_av_rldv",
    os.path.join(_script_dir, "t-av-reaction-limited-different-volume.py")
)
t_av_rldv = importlib.util.module_from_spec(spec_tav)
spec_tav.loader.exec_module(t_av_rldv)


def plot_left(ax):
    """Plot reaction-limited-different-volume (logical_blocks.json)."""
    sequences = rldv.load_sequences()
    min_pct, max_cost = rldv.compute_min_percentage(sequences)
    pct_100 = 100.0
    pct_min = min_pct
    step = (pct_100 - pct_min) / 3 if pct_100 > pct_min else 0
    percentages_display = (
        [pct_100, pct_100 - step, pct_100 - 2 * step, pct_min] if step > 0 else [pct_100] * 4
    )
    capacities = [
        max(int(rldv.TOTAL_LOGICAL_BLOCKS * p / 100), int(np.ceil(max_cost))) for p in percentages_display
    ]
    results = []
    for pct, cap in zip(percentages_display, capacities):
        r = rldv.run_for_capacity(sequences, float(cap))
        r["pct"] = pct
        r["capacity"] = cap
        results.append(r)

    d = 7
    # tau_r = np.logspace(1.5, 3, 100)
    tau_r = np.logspace(-1, 4, 100)
    tau_c = np.logspace(-1, 2, 100)
    tau_c_line = np.logspace(np.log10(tau_c.min()), np.log10(tau_c.max()), 200)
    colors = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3"]
    X, Y = np.meshgrid(tau_c, tau_r)

    for idx, r in enumerate(results):
        stalling_grid = rldv.compute_stalling_grid(r["reaction_depths_list"], d, tau_r, tau_c)
        tau_r_boundary = rldv.get_no_stalling_boundary(r["reaction_depths_list"], d, tau_c_line)
        mask = (tau_r_boundary >= tau_r.min()) & (tau_r_boundary <= tau_r.max())
        y_top = np.clip(tau_r_boundary, tau_r.min(), tau_r.max())
        ax.fill_between(tau_c_line, tau_r.min(), y_top, alpha=0.25, color=colors[idx], zorder=0)
        ax.contour(X, Y, stalling_grid, levels=[1e4], colors=[colors[idx]], linewidths=2, linestyles=":", alpha=0.9, zorder=2)
        ax.contour(X, Y, stalling_grid, levels=[1e5], colors=[colors[idx]], linewidths=2, linestyles="--", alpha=0.9, zorder=2)
        ax.plot(tau_c_line[mask], tau_r_boundary[mask], color=colors[idx], linewidth=2.5, linestyle="-", zorder=3)

    ax.set_xlabel(r"$\tau_c$ (µs)")
    ax.set_ylabel(r"$\tau_r$ (µs)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(tau_c.min(), tau_c.max())
    ax.set_ylim(tau_r.min(), tau_r.max())
    stalling_104 = Line2D([0], [0], color="gray", linestyle=":", linewidth=2, label=r"Stalling = $10^4$ µs")
    stalling_105 = Line2D([0], [0], color="gray", linestyle="--", linewidth=2, label=r"Stalling = $10^5$ µs")
    no_stalling = Line2D([0], [0], color="gray", linestyle="-", linewidth=2.5, label="No stalling boundary")
    vol_handles = [Patch(facecolor=colors[idx], alpha=0.25, edgecolor=colors[idx], label=f"{r['pct']:.1f}% vol, {r['n_cycles']} cycles") for idx, r in enumerate(results)]
    leg1 = ax.legend(handles=[stalling_104, stalling_105, no_stalling], loc="upper center", bbox_to_anchor=(0.25, -0.2), frameon=True, ncol=1)
    ax.add_artist(leg1)
    ax.legend(handles=vol_handles, labels=[f"{r['pct']:.1f}% vol, {r['n_cycles']} cycles" for r in results], loc="upper center", bbox_to_anchor=(0.75, -0.2), frameon=True, ncol=1)
    # ax.set_title("Lattice surgery AV compilation")
    # "No stalling zone" label inside the no-stalling region (lower left)
    ax.text(0.50, 0.15, "no stalling zone", transform=ax.transAxes, fontsize=14, alpha=0.8, va="center", ha="center")


def plot_right(ax):
    """Plot t-av-reaction-limited-different-volume (transversal-logical-blocks.json)."""
    sequences = t_av_rldv.load_sequences(t_av_rldv.SEQUENCES_FILE)
    pct_100 = 100.0
    pct_min = 25.0
    step = (pct_100 - pct_min) / 3
    percentages_display = [pct_100, pct_100 - step, pct_100 - 2 * step, pct_min]
    capacities = [max(1, int(t_av_rldv.available_patches * p / 100)) for p in percentages_display]
    results = []
    for pct, cap in zip(percentages_display, capacities):
        r = t_av_rldv.run_for_capacity(sequences, float(cap))
        r["pct"] = pct
        r["capacity"] = cap
        results.append(r)

    d = 7
    tau_c = np.logspace(0, 5, 100)
    tau_r = np.logspace(-1, 4, 100)
    tau_c_line = np.logspace(np.log10(tau_c.min()), np.log10(tau_c.max()), 200)
    colors = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3"]
    X, Y = np.meshgrid(tau_c, tau_r)

    for idx, r in enumerate(results):
        stalling_grid = t_av_rldv.compute_stalling_grid(r["reaction_depths_list"], tau_r, tau_c)
        tau_r_boundary = t_av_rldv.get_no_stalling_boundary(r["reaction_depths_list"], tau_c_line)
        mask = (tau_r_boundary >= tau_r.min()) & (tau_r_boundary <= tau_r.max())
        y_top = np.clip(tau_r_boundary, tau_r.min(), tau_r.max())
        ax.fill_between(tau_c_line, tau_r.min(), y_top, alpha=0.25, color=colors[idx], zorder=0)
        ax.contour(X, Y, stalling_grid, levels=[1e4], colors=[colors[idx]], linewidths=2, linestyles=":", alpha=0.9, zorder=2)
        ax.contour(X, Y, stalling_grid, levels=[1e5], colors=[colors[idx]], linewidths=2, linestyles="--", alpha=0.9, zorder=2)
        ax.plot(tau_c_line[mask], tau_r_boundary[mask], color=colors[idx], linewidth=2.5, linestyle="-", zorder=3)

    ax.set_xlabel(r"$\tau_c$ (µs)")
    ax.set_ylabel(r"$\tau_r$ (µs)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(tau_c.min(), tau_c.max())
    ax.set_ylim(tau_r.min(), tau_r.max())
    stalling_104 = Line2D([0], [0], color="gray", linestyle=":", linewidth=2, label=r"Stalling = $10^4$ µs")
    stalling_105 = Line2D([0], [0], color="gray", linestyle="--", linewidth=2, label=r"Stalling = $10^5$ µs")
    no_stalling = Line2D([0], [0], color="gray", linestyle="-", linewidth=2.5, label="No stalling boundary")
    vol_handles = [Patch(facecolor=colors[idx], alpha=0.25, edgecolor=colors[idx], label=f"{r['pct']:.0f}% cap, {r['n_cycles']/d:.0f} cycles") for idx, r in enumerate(results)]
    leg1 = ax.legend(handles=[stalling_104, stalling_105, no_stalling], loc="upper center", bbox_to_anchor=(0.25, -0.2), frameon=True, ncol=1)
    ax.add_artist(leg1)
    ax.legend(handles=vol_handles, labels=[f"{r['pct']:.0f}% vol, {r['n_cycles']/d:.0f} cycles" for r in results], loc="upper center", bbox_to_anchor=(0.75, -0.2), frameon=True, ncol=1)
    # ax.set_title("Transversal-AV compilation")
    # "No stalling zone" label inside the no-stalling region (lower left)
    ax.text(0.50, 0.15, "no stalling zone", transform=ax.transAxes, fontsize=14, alpha=0.8, va="center", ha="center")


def main():
    try:
        plt.style.use("plotstylefile.mplstyle")
    except OSError:
        pass
    # ax.grid(True)  # grid disabled
    plt.rcParams["axes.grid"] = False

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    plot_left(ax1)
    plot_right(ax2)
    plt.tight_layout(rect=[0, 0.15, 1, 1])
    plt.savefig("reaction_limited_different_hardware.pdf", bbox_inches="tight", pad_inches=0.1)
    print("Saved reaction_limited_different_hardware.pdf")


if __name__ == "__main__":
    main()
