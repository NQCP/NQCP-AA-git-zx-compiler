"""
PPR parallelization sweep — plot helper script.

Loads precomputed sweep data from data/sweep_<circuit>.npz (written by
ppr_parallelization_different_volume.py) and renders the overlay plot.

Regenerate the data only when the input file, scheduler, or capacity list changes:
    python logical_network_parallelization_plot_helpers/ppr_parallelization_different_volume.py
"""

from pathlib import Path
import sys

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
DATA_DIR = HERE / "data"
TWO_COLUMN_STYLE_FILE = PROJECT_ROOT / "plotstylefile_two_column.mplstyle"

SAVE_TO_OVERLEAF = False

sys.path.insert(0, str(HERE))
from ppr_parallelization import load_sequences, schedule_sequences

FERMI_CIRCUIT = "fermi_hubbard_2d_step_s4_universal_paulis_commuted"
TMM_CIRCUIT = "tmm_paulis_commuted"
N_TROTTER_STEPS = 1
AV_NORMALIZATION = {"TMM": 25, "Fermi-Hubbard": 413}
TMM_CAPACITIES = [25 * i for i in range(1, 13)]


def load_sweep(path):
    data = np.load(path)
    caps = data["capacities"].tolist()
    return [
        {
            "capacity": int(c),
            "parallel_ops": data[f"parallel_ops_{int(c)}"],
            "util": data[f"util_{int(c)}"],
        }
        for c in caps
    ]


def load_or_compute_tmm_sweep():
    data_file = DATA_DIR / f"sweep_{TMM_CIRCUIT}_n{N_TROTTER_STEPS}.npz"
    if not data_file.exists():
        print(f"{data_file} not found — computing TMM sweep...")
        base_sequences = load_sequences(
            str(PROJECT_ROOT / "logical_network_files" / f"logical_blocks_{TMM_CIRCUIT}.json")
        )
        sequences = base_sequences * N_TROTTER_STEPS
        print(
            f"Scheduling {len(base_sequences):,} sequences/step x "
            f"{N_TROTTER_STEPS} steps = {len(sequences):,} total"
        )
        arrays = {
            "capacities": np.array(TMM_CAPACITIES, dtype=np.int64),
            "n_trotter_steps": np.array([N_TROTTER_STEPS], dtype=np.int64),
        }
        for cap in TMM_CAPACITIES:
            schedule = schedule_sequences(sequences, total_capacity=cap)
            po = np.asarray([s[1] for s in schedule], dtype=np.int32)
            used = np.asarray([s[2] for s in schedule], dtype=np.float32)
            arrays[f"parallel_ops_{cap}"] = po
            arrays[f"util_{cap}"] = (used / cap).astype(np.float32)
            print(f"  capacity={cap:>4}: {len(po):>8,} cycles")
        DATA_DIR.mkdir(exist_ok=True)
        np.savez_compressed(data_file, **arrays)
        print(f"Saved {data_file}")
    return load_sweep(data_file)


def _summary_arrays(runs, norm=1):
    runs = sorted(runs, key=lambda r: r["capacity"])
    capacities = np.array([r["capacity"] for r in runs], dtype=float) / norm
    cycles = np.array([len(r["parallel_ops"]) for r in runs], dtype=float)
    speedup = cycles[0] / cycles
    mean_pprs = np.array([float(r["parallel_ops"].mean()) for r in runs])
    std_pprs = np.array([float(r["parallel_ops"].std()) for r in runs])
    mean_usage = np.array([float(r["util"].mean()) for r in runs])
    return capacities, speedup, mean_pprs, mean_pprs + std_pprs, mean_usage


def _norm_tick_label(x):
    return f"{int(x)}x" if x == int(x) else f"{x:.1f}x"


def _format_capacity_axis(ax, capacities, max_ticks=6):
    capacities = np.asarray(capacities)
    ax.set_xscale("log")
    if len(capacities) > max_ticks:
        targets = np.geomspace(capacities.min(), capacities.max(), max_ticks)
        tick_idx = np.unique([np.abs(capacities - t).argmin() for t in targets])
        ticks = capacities[tick_idx]
    else:
        ticks = capacities
    minor_ticks = np.array([c for c in capacities if c not in ticks])
    ax.set_xticks(ticks)
    ax.set_xticks(minor_ticks, minor=True)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: _norm_tick_label(x)))
    ax.xaxis.set_minor_formatter(mticker.NullFormatter())
    ax.tick_params(axis="x", which="major", top=True, labeltop=False)
    ax.tick_params(axis="x", which="minor", top=True, labeltop=False,
                   length=3.5, width=0.9)


def _plot_capacity_summary_panel(ax, runs, norm=1, title=None, show_left_ylabel=False, show_right_ylabel=False):
    capacities, speedup, mean_pprs, upper_pprs, mean_usage = _summary_arrays(runs, norm=norm)
    ax2 = ax.twinx()

    cycle_color = "#0072B2"
    ppr_color = "#D55E00"
    usage_color = "#666666"

    l1, = ax.plot(
        capacities, mean_pprs,
        marker="s", markersize=4.2, linewidth=1.6, color=ppr_color, label="Mean PPRs/cycle",
    )
    ax.fill_between(capacities, mean_pprs, upper_pprs, color=ppr_color, alpha=0.12, linewidth=0)
    l2, = ax2.plot(
        capacities, speedup,
        marker="o", markersize=4.5, linewidth=1.8, color=cycle_color, label="Speedup",
    )

    for x, usage in zip(capacities, mean_usage):
        ax.axvline(x, ymin=0.02, ymax=min(0.98, 0.02 + 0.18 * usage),
                   color=usage_color, alpha=0.18, linewidth=3)

    _format_capacity_axis(ax, capacities)
    ax.set_xlabel("Workspace capacity", labelpad=2)
    if show_left_ylabel:
        ax.set_ylabel("PPRs per cycle", labelpad=2)
    if show_right_ylabel:
        ax2.set_ylabel("Speedup", labelpad=2)
    ax.tick_params(axis="both", which="major", direction="in", width=1.5, length=5.5)
    ax2.tick_params(axis="y", which="major", direction="in", width=1.5, length=5.5)
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
    for spine in ax2.spines.values():
        spine.set_linewidth(1.5)
    ax.set_ylim(0, max(upper_pprs.max() * 1.12, mean_pprs.max() * 1.2))
    ax2.set_ylim(0, speedup.max() * 1.12)
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: _norm_tick_label(x)))
    ax.yaxis.set_minor_locator(mticker.AutoMinorLocator(2))
    ax2.yaxis.set_minor_locator(mticker.AutoMinorLocator(2))
    ax.tick_params(axis="y", which="minor", direction="in", width=0.9, length=3.5, left=True)
    ax2.tick_params(axis="y", which="minor", direction="in", width=0.9, length=3.5, right=True)
    ax.grid(alpha=0.07)
    ax.set_axisbelow(True)
    ax2.grid(False)
    if title:
        ax.set_title(title)
    return l1, l2


def plot_capacity_summary_comparison(runs_tmm, runs_fermi, figsize=(8.5, 3.23)):
    fig, axes = plt.subplots(1, 2, figsize=figsize, sharey=True)
    handles = _plot_capacity_summary_panel(
        axes[0], runs_tmm, norm=AV_NORMALIZATION["TMM"],
        title="TMM", show_left_ylabel=True, show_right_ylabel=False
    )
    _plot_capacity_summary_panel(
        axes[1], runs_fermi, norm=AV_NORMALIZATION["Fermi-Hubbard"],
        title="Fermi-Hubbard", show_left_ylabel=False, show_right_ylabel=True
    )
    fig.legend(
        handles,
        [h.get_label() for h in handles],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.02),
        ncol=2,
        frameon=False,
        handlelength=1.8,
        columnspacing=1.0,
    )
    fig.tight_layout(rect=[0, 0.08, 1, 1], pad=0.18, w_pad=0.8)
    return fig


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


def main():
    fermi_data_file = DATA_DIR / f"sweep_{FERMI_CIRCUIT}.npz"
    runs_fermi = load_sweep(fermi_data_file)
    print(f"Loaded {len(runs_fermi)} capacities from {fermi_data_file}")
    for r in runs_fermi:
        po = r["parallel_ops"]
        print(f"  capacity={r['capacity']:>6}: {len(po):>10,} cycles  mean PPRs={po.mean():.2f}  mean util={r['util'].mean():.2f}")

    runs_tmm = load_or_compute_tmm_sweep()
    print(f"Loaded {len(runs_tmm)} TMM capacities")
    for r in runs_tmm:
        po = r["parallel_ops"]
        print(f"  capacity={r['capacity']:>6}: {len(po):>8,} cycles  mean PPRs={po.mean():.2f}  mean util={r['util'].mean():.2f}")

    with plt.style.context(str(TWO_COLUMN_STYLE_FILE)):
        fig = plot_capacity_summary_comparison(runs_tmm, runs_fermi)
        save_for_paper(fig, "ppr_parallelization_different_volume_tmm_and_fermi_hubbard.pdf")
        plt.close(fig)


if __name__ == "__main__":
    main()
