"""
Bell pairs per cycle — plot helper script.

Loads precomputed sweep data from data/bell_pairs_sweep_<circuit>.npz
(written by bell_pairs_count.py) and renders the overlay plots.

Regenerate the data only when the input file, scheduler, or capacity list changes:
    python bell_pair_analysis_helpers/bell_pairs_count.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from matplotlib.lines import Line2D

SAVE_TO_OVERLEAF = False

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
DATA_DIR = HERE / "data"
TWO_COLUMN_STYLE_FILE = PROJECT_ROOT / "plotstylefile_two_column.mplstyle"

FERMI_CIRCUIT = "fermi_hubbard_2d_step_s4_universal_paulis_commuted"
TMM_CIRCUIT = "tmm_paulis_commuted"

SCALING_CIRCUITS = {
    "TMM": TMM_CIRCUIT,
    "Fermi-Hubbard": FERMI_CIRCUIT,
}
TRANSVERSAL_SCALING_CIRCUITS = {
    "TMM": "transversal-logical-blocks-tmm",
    "Fermi-Hubbard": "transversal-logical-blocks-fermi-hubbard",
}

# Max active volume per PPR for each circuit (used to normalise workspace capacity axis)
AV_NORMALIZATION = {
    "TMM": 25,
    "Fermi-Hubbard": 413,
}


# ── data loaders ──────────────────────────────────────────────────────────────

def load_sweep(path):
    data = np.load(path)
    caps = data["capacities"].tolist()
    return [
        {
            "capacity": int(c),
            "total_bp": data[f"total_bp_{int(c)}"],
            "used_blocks": data[f"used_blocks_{int(c)}"],
        }
        for c in caps
    ]


def load_transversal_sweep(path):
    data = np.load(path)
    t_counts = data["t_counts"].tolist()
    return [
        {
            "t_count": int(t),
            "total_bp": data[f"total_bp_{int(t)}"],
        }
        for t in t_counts
    ]


# ── shared helpers ─────────────────────────────────────────────────────────────

def _rolling_stats(arr, window):
    n_full = (len(arr) // window) * window
    if n_full == 0:
        x = np.array([len(arr) / 2.0])
        m = np.array([arr.mean()])
        return x, m, m.copy(), m.copy(), m.copy(), m.copy()
    chunks = arr[:n_full].reshape(-1, window)
    x = (np.arange(chunks.shape[0]) + 0.5) * window
    return (
        x,
        chunks.mean(axis=1),
        np.percentile(chunks, 5, axis=1),
        np.percentile(chunks, 25, axis=1),
        np.percentile(chunks, 75, axis=1),
        np.percentile(chunks, 95, axis=1),
    )


def _norm_tick_label(x):
    if x == int(x):
        return f"{int(x)}x"
    if x >= 1:
        return f"{x:.1f}x"
    return f"{x:g}x"


def _format_capacity_axis(ax, capacities, max_ticks=5):
    capacities = np.asarray(capacities)
    ax.set_xscale("log")
    if len(capacities) > max_ticks:
        targets = np.geomspace(capacities.min(), capacities.max(), max_ticks)
        tick_idx = np.unique([np.abs(capacities - t).argmin() for t in targets])
        ticks = capacities[tick_idx]
    else:
        ticks = capacities
    ax.set_xticks(ticks)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: _norm_tick_label(x)))
    ax.xaxis.set_minor_formatter(mticker.NullFormatter())
    ax.tick_params(axis="x", top=True, labeltop=False)


def _style_twin_ax(ax, ax2):
    """Apply consistent tick/spine styling to a primary + twin axis pair."""
    for a in (ax, ax2):
        a.tick_params(axis="both", direction="in", width=1.5, length=5.5)
        for spine in a.spines.values():
            spine.set_linewidth(1.5)
    ax2.grid(False)


def save_for_paper(fig, filename):
    output_dirs = [PROJECT_ROOT / "paper_plots"]
    overleaf_dir = PROJECT_ROOT.parent / "let-s-estimate-some-resources" / "Plots"
    if SAVE_TO_OVERLEAF and overleaf_dir.exists():
        output_dirs.append(overleaf_dir)
    for out_dir in output_dirs:
        out_dir.mkdir(exist_ok=True)
        out_path = out_dir / filename
        fig.savefig(out_path, bbox_inches="tight", pad_inches=0.03)
        print(f"Saved {out_path}")


# ── AV plots ──────────────────────────────────────────────────────────────────

def _capacity_scaling_arrays(runs):
    capacities = np.array([r["capacity"] for r in runs])
    total_cycles = np.array([len(r["total_bp"]) for r in runs])
    speedup = total_cycles[0] / total_cycles
    mean_bp = np.array([r["total_bp"].mean() for r in runs])
    std_bp = np.array([r["total_bp"].std() for r in runs])
    memory_remaining = (capacities - mean_bp) / capacities
    memory_remaining_upper = (capacities - np.maximum(mean_bp - std_bp, 0)) / capacities
    memory_remaining_lower = np.maximum(capacities - (mean_bp + std_bp), 0) / capacities
    return capacities, memory_remaining, memory_remaining_lower, memory_remaining_upper, speedup


def plot_scaling_comparison(circuits=SCALING_CIRCUITS, normalization=AV_NORMALIZATION, figsize=(8.5, 3.23)):
    """Side-by-side memory-remaining + cycle-count vs workspace capacity."""
    n = len(circuits)
    fig, axes = plt.subplots(1, n, figsize=figsize, squeeze=False)

    mem_color = "#D55E00"
    cyc_color = "#0072B2"
    handles = None

    # First pass: collect data for all circuits to compute shared y-axis limits
    all_data = []
    for label, circuit in circuits.items():
        runs = load_sweep(DATA_DIR / f"bell_pairs_sweep_{circuit}.npz")
        capacities, memory_remaining, mem_lower, mem_upper, speedup = _capacity_scaling_arrays(runs)
        norm = normalization.get(label, 1)
        all_data.append((label, circuit, capacities / norm, memory_remaining, mem_lower, mem_upper, speedup))

    x_max = min(d[2].max() for d in all_data)
    x_min = min(d[2].min() for d in all_data)

    # Restrict each circuit's data to the shared x range before computing y limits
    clipped = []
    for label, circuit, norm_capacities, memory_remaining, mem_lower, mem_upper, speedup in all_data:
        mask = norm_capacities <= x_max
        clipped.append((label, circuit, norm_capacities[mask], memory_remaining[mask],
                        mem_lower[mask], mem_upper[mask], speedup[mask]))

    mem_min = min(d[4].min() for d in clipped)
    mem_max = max(d[5].max() for d in clipped)
    cyc_min = min(d[6].min() for d in clipped)
    cyc_max = max(d[6].max() for d in clipped)

    for col, (label, circuit, norm_capacities, memory_remaining, mem_lower, mem_upper, speedup) in enumerate(clipped):
        ax = axes[0, col]
        ax2 = ax.twinx()

        l1, = ax.plot(norm_capacities, memory_remaining,
                      marker="o", linestyle="-", color=mem_color,
                      lw=1.8, markersize=4.5, label="Memory remaining")
        ax.fill_between(norm_capacities, mem_lower, mem_upper,
                        color=mem_color, alpha=0.15)
        l2, = ax2.plot(norm_capacities, speedup,
                       marker="s", linestyle="-", color=cyc_color,
                       lw=1.6, markersize=4.2, label="Speedup")
        if handles is None:
            handles = (l1, l2)

        ax.set_yscale("log")
        ax2.set_yscale("log")
        ax.set_ylim(max(mem_min * 0.85, 1e-3), mem_max * 1.15)
        ax2.set_ylim(cyc_min * 0.85, cyc_max * 1.15)
        ax.yaxis.set_major_locator(mticker.LogLocator(base=10, subs=[1, 2, 5], numticks=8))
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:g}"))
        ax.yaxis.set_minor_locator(mticker.NullLocator())
        ax2.yaxis.set_major_locator(mticker.LogLocator(base=10, subs=[1, 2, 5], numticks=8))
        ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: _norm_tick_label(x)))
        ax2.yaxis.set_minor_locator(mticker.NullLocator())

        ax.set_title(label)
        _format_capacity_axis(ax, norm_capacities)
        ax.set_xlim(x_min / 1.1, x_max * 1.1)
        ax.grid(axis="both", which="major", color="0.88", linewidth=0.7)
        ax.set_axisbelow(True)
        _style_twin_ax(ax, ax2)

        ax.set_xlabel("Workspace capacity", labelpad=2)
        if col == 0:
            ax.set_ylabel("Remaining memory", labelpad=2)
        if col == n - 1:
            ax2.set_ylabel("Speedup", labelpad=2)

    fig.legend(
        handles, [h.get_label() for h in handles],
        loc="upper center", bbox_to_anchor=(0.5, 0.02),
        ncol=2, frameon=False, handlelength=1.8, columnspacing=1.0,
    )
    fig.tight_layout(rect=[0, 0.08, 1, 1], pad=0.18, w_pad=0.8)
    return fig


# ── T-AV plots ────────────────────────────────────────────────────────────────

def _transversal_scaling_arrays(runs):
    t_counts = np.array([r["t_count"] for r in runs])
    total_cycles = np.array([len(r["total_bp"]) for r in runs])
    speedup = total_cycles[0] / total_cycles
    mean_bp = np.array([r["total_bp"].mean() for r in runs])
    std_bp = np.array([r["total_bp"].std() for r in runs])
    return t_counts, mean_bp, mean_bp - std_bp, mean_bp + std_bp, speedup


# Factory production rates (code cycles per produced T state, per factory).
# Sourced from resource_estimators/qubits_runtime_estimates*.py.
TAV_VARIANT_PERIODS = {
    "Trans-MSD": 8.0,  # T_STATE_FACTORY_PERIOD
}
TAV_VARIANT_COLORS = {
    "Trans-MSD": "#0072B2",
}


def _factory_sweep_for_variant(mean_bp_by_t, std_bp_by_t, total_cycles_by_t, period):
    """Sweep num_factories from 1 to the saturation point (t_count = max).

    Returns ns (ints), bridges (floats), bridges_lower, bridges_upper,
    speedups (floats).

    Magic-limited (supply = N/period < 1): bridges = 0, runtime = total_T*period/N.
    Compute-limited (supply >= 1): t_count = floor(supply), capped at max t_count
    in the sweep data; bridges and runtime come from the sweep.

    Speedup is anchored at the variant's own start-of-compute-limited regime
    (N = period, i.e. supply = 1 T/cycle), so all variants reach the same
    saturated speedup = total_cycles[1] / total_cycles[t_count_max]. This
    makes the across-variant comparison apples-to-apples.
    """
    t_counts = sorted(mean_bp_by_t.keys())
    t_count_max = t_counts[-1]
    total_T = total_cycles_by_t[1]
    n_max = int(np.ceil(t_count_max * period))
    ns = np.arange(1, n_max + 1)

    bridges = np.zeros_like(ns, dtype=float)
    bridges_lower = np.zeros_like(ns, dtype=float)
    bridges_upper = np.zeros_like(ns, dtype=float)
    runtimes = np.zeros_like(ns, dtype=float)
    for i, N in enumerate(ns):
        supply = N / period
        if supply < 1.0:
            bridges[i] = 0.0
            bridges_lower[i] = 0.0
            bridges_upper[i] = 0.0
            runtimes[i] = total_T * period / N
        else:
            t_count = min(int(supply), t_count_max)
            mu = mean_bp_by_t[t_count]
            sigma = std_bp_by_t[t_count]
            bridges[i] = mu
            bridges_lower[i] = max(mu - sigma, 0.0)
            bridges_upper[i] = mu + sigma
            runtimes[i] = total_cycles_by_t[t_count]

    runtime_anchor = total_cycles_by_t[1]  # compute-limited at t_count=1
    speedups = runtime_anchor / runtimes
    return ns, bridges, bridges_lower, bridges_upper, speedups


def plot_transversal_scaling_panels(circuits=TRANSVERSAL_SCALING_CIRCUITS, figsize=(8.5, 3.6)):
    """Bridge qubits and speedup vs number of factories, per t-AV variant.

    Per panel (one per circuit): for each variant, plot
      - left y (linear): mean bridge qubits per cycle
      - right y (log):   speedup, anchored at N=1
      - x (linear):      raw number of factories
    Variants share color; bridge-qubits curves are solid, speedup curves dashed.
    """
    n = len(circuits)
    fig, axes = plt.subplots(1, n, figsize=figsize, squeeze=False)

    panel_arrays = []
    for label, circuit in circuits.items():
        runs = load_transversal_sweep(DATA_DIR / f"bell_pairs_sweep_transversal_{circuit}.npz")
        mean_bp_by_t = {r["t_count"]: float(r["total_bp"].mean()) for r in runs}
        std_bp_by_t = {r["t_count"]: float(r["total_bp"].std()) for r in runs}
        total_cycles_by_t = {r["t_count"]: len(r["total_bp"]) for r in runs}
        per_variant = {}
        for variant_label, period in TAV_VARIANT_PERIODS.items():
            per_variant[variant_label] = _factory_sweep_for_variant(
                mean_bp_by_t, std_bp_by_t, total_cycles_by_t, period,
            )
        panel_arrays.append((label, per_variant))

    bp_max = max(
        bridges_upper.max()
        for _, per_variant in panel_arrays
        for _, _, _, bridges_upper, _ in per_variant.values()
    )
    sp_max = max(
        speedups.max()
        for _, per_variant in panel_arrays
        for _, _, _, _, speedups in per_variant.values()
    )
    sp_min = min(
        speedups.min()
        for _, per_variant in panel_arrays
        for _, _, _, _, speedups in per_variant.values()
    )
    n_max_global = max(
        ns.max()
        for _, per_variant in panel_arrays
        for ns, _, _, _, _ in per_variant.values()
    )

    for col, (label, per_variant) in enumerate(panel_arrays):
        ax = axes[0, col]
        ax2 = ax.twinx()

        for variant_label, (ns, bridges, bridges_lower, bridges_upper, speedups) in per_variant.items():
            color = TAV_VARIANT_COLORS[variant_label]
            ax.fill_between(ns, bridges_lower, bridges_upper,
                            color=color, alpha=0.15, linewidth=0)
            ax.plot(ns, bridges, color=color, linestyle="-", lw=1.5,
                    marker="o", markersize=3.0, markeredgewidth=0)
            ax2.plot(ns, speedups, color="#D62728", linestyle="-", lw=1.5,
                     marker="s", markersize=3.0, markeredgewidth=0)

        ax.set_xscale("log")
        ax.set_yscale("linear")
        ax2.set_yscale("log")
        ax.set_ylim(-bp_max * 0.06, bp_max * 1.10)
        ax2.set_ylim(max(sp_min * 0.85, 1e-2), sp_max * 1.15)
        ax.set_xlim(1, n_max_global)

        ax.set_title(label)
        ax.set_xlabel("Number of factories", labelpad=2)
        if col == 0:
            ax.set_ylabel("Mean bridge qubits", labelpad=2)
        if col == n - 1:
            ax2.set_ylabel("Speedup", labelpad=2)

        ax.grid(axis="both", which="major", color="0.88", linewidth=0.7)
        ax.set_axisbelow(True)
        _style_twin_ax(ax, ax2)

        ax.xaxis.set_major_locator(mticker.LogLocator(base=10, subs=[1, 2, 5], numticks=8))
        ax.xaxis.set_minor_locator(mticker.NullLocator())
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x)}"))
        ax2.yaxis.set_major_locator(mticker.LogLocator(base=10, subs=[1, 2, 5], numticks=8))
        ax2.yaxis.set_minor_locator(mticker.NullLocator())
        ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: _norm_tick_label(x)))

    fig.tight_layout(pad=0.18, w_pad=0.8)
    return fig


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    with plt.style.context(str(TWO_COLUMN_STYLE_FILE)):
        fig_av = plot_scaling_comparison()
        save_for_paper(fig_av, "bell_pairs_scaling_vs_capacity_tmm_and_fermi_hubbard.pdf")
        plt.close(fig_av)

        fig_tav = plot_transversal_scaling_panels()
        save_for_paper(fig_tav, "bell_pairs_scaling_vs_t_count_tmm_and_fermi_hubbard.pdf")
        plt.close(fig_tav)

        fig_tmm = plot_transversal_scaling_panels(
            circuits={"TMM": TRANSVERSAL_SCALING_CIRCUITS["TMM"]},
            figsize=(5.0, 3.5),
        )
        save_for_paper(fig_tmm, "bell_pair_scaling_vs_factory_tmm.pdf")
        plt.close(fig_tmm)


if __name__ == "__main__":
    main()
