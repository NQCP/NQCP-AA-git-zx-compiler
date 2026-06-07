"""Helpers for reaction-limited plotting and figure generation.

This module consolidates the plotting logic that previously lived in
`reaction_time_analysis/plot_helpers.ipynb` into plain Python.
"""

from pathlib import Path
import csv
import importlib.util

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.colors as mcolors
import numpy as np
from matplotlib.lines import Line2D


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
STYLE_FILE = PROJECT_ROOT / "plotstylefile.mplstyle"
TWO_COLUMN_STYLE_FILE = PROJECT_ROOT / "plotstylefile_two_column.mplstyle"
OVERLAY_LIGHTER_ERROR_RATE_BLEND = 0.58
USE_CASE_LABEL_OVERRIDES = {
    "Trotter full QPE": "QPE-Abs",
    "Statistical QPE": "Stat-QPE-Abs",
    "Statistical QPE gap": "Stat-QPE-gap",
}

DEFAULT_FERMI_AV_CAPACITIES = [413, 1000, 1500]
DEFAULT_TMM_AV_CAPACITIES = [25, 50, 75]
DEFAULT_TMM_T_COUNTS = [1, 2, 3]
AV_CMAP_NAME = "viridis"
TAV_CMAP_NAME = "plasma"


def _load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


rldv = _load_module(
    "reaction_limited_different_volume",
    PROJECT_ROOT / "reaction_time_analysis" / "reaction-limited-different-volume.py",
)
tav_rldv = _load_module(
    "t_av_reaction_limited_different_volume",
    PROJECT_ROOT / "reaction_time_analysis" / "t-av-reaction-limited-different-volume.py",
)


def apply_default_style():
    if STYLE_FILE.exists():
        plt.style.use(str(STYLE_FILE))


apply_default_style()


def compute_stalling_grid(reaction_depths, d, tau_r, tau_c):
    """Return total stalling time on a tau_r x tau_c grid."""
    reaction_depths = np.asarray(reaction_depths, dtype=np.int32)
    values, counts = np.unique(reaction_depths, return_counts=True)
    grid = np.zeros((len(tau_r), len(tau_c)), dtype=float)
    for rd, count in zip(values, counts):
        if rd <= 0:
            continue
        grid += count * np.maximum(0.0, rd * tau_r[:, None] - d * tau_c[None, :])
    return grid


def get_no_stalling_boundary(reaction_depths, d, tau_c):
    max_rd = max(reaction_depths) if reaction_depths else 0
    if max_rd == 0:
        return np.full_like(tau_c, np.nan)
    return d * tau_c / max_rd


def select_runs(results, capacities):
    by_capacity = {int(run["capacity"]): run for run in results}
    missing = [capacity for capacity in capacities if capacity not in by_capacity]
    if missing:
        raise ValueError(f"Missing cached capacities: {missing}")
    return [by_capacity[capacity] for capacity in capacities]


def capacity_color_map(capacities, cmap_name="viridis"):
    cmap = plt.get_cmap(cmap_name)
    if len(capacities) == 1:
        return {capacities[0]: cmap(0.65)}
    return {
        capacity: cmap(0.16 + 0.72 * idx / (len(capacities) - 1))
        for idx, capacity in enumerate(capacities)
    }


def blend_color_toward_white(color, frac):
    red, green, blue = mcolors.to_rgb(color)
    return (
        red + frac * (1.0 - red),
        green + frac * (1.0 - green),
        blue + frac * (1.0 - blue),
    )


def display_use_case_label(label):
    return USE_CASE_LABEL_OVERRIDES.get(label, label)


def format_log_axes(ax, tau_c, tau_r):
    x_ticks = 10.0 ** np.arange(
        np.ceil(np.log10(tau_c.min())), np.floor(np.log10(tau_c.max())) + 1
    )
    y_ticks = 10.0 ** np.arange(
        np.ceil(np.log10(tau_r.min())), np.floor(np.log10(tau_r.max())) + 1
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks(x_ticks[(x_ticks >= tau_c.min()) & (x_ticks <= tau_c.max())])
    ax.set_yticks(y_ticks[(y_ticks >= tau_r.min()) & (y_ticks <= tau_r.max())])
    formatter = mticker.LogFormatterMathtext(base=10)
    ax.xaxis.set_major_formatter(formatter)
    ax.yaxis.set_major_formatter(formatter)
    ax.xaxis.set_minor_formatter(mticker.NullFormatter())
    ax.yaxis.set_minor_formatter(mticker.NullFormatter())
    ax.tick_params(axis="x", which="major", direction="in", pad=4, top=True, length=6, width=1.6)
    ax.tick_params(axis="y", which="major", direction="in", pad=1, right=True, length=6, width=1.6)
    ax.tick_params(axis="both", which="minor", direction="in", top=True, right=True, length=4, width=1.2)
    ax.grid(axis="both", which="major", color="0.88", linewidth=0.55)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_color(plt.rcParams["axes.edgecolor"])
        spine.set_linewidth(max(plt.rcParams["axes.linewidth"], 1.5))


def draw_contour_if_present(ax, X, Y, grid, level, color, linestyle):
    if grid.min() <= level <= grid.max():
        ax.contour(
            X,
            Y,
            grid,
            levels=[level],
            colors=[color],
            linewidths=1.65,
            linestyles=linestyle,
            zorder=2,
        )


def plot_reaction_limited(
    results,
    tau_r,
    tau_c,
    d,
    title="Fermi-Hubbard",
    figsize=(6.5, 4.5),
    capacity_label="blocks",
    cmap_name="viridis",
):
    ids = [int(run.get("capacity", run.get("t_count"))) for run in results]
    colors = capacity_color_map(ids, cmap_name=cmap_name)
    tau_c_line = np.logspace(np.log10(tau_c.min()), np.log10(tau_c.max()), 240)
    X, Y = np.meshgrid(tau_c, tau_r)

    fig, ax = plt.subplots(figsize=figsize)
    format_log_axes(ax, tau_c, tau_r)

    for run in results:
        run_id = int(run.get("capacity", run.get("t_count")))
        color = colors[run_id]
        reaction_depths = run["reaction_depths_list"]
        stalling_grid = compute_stalling_grid(reaction_depths, d, tau_r, tau_c)
        tau_r_boundary = get_no_stalling_boundary(reaction_depths, d, tau_c_line)
        mask = (tau_r_boundary >= tau_r.min()) & (tau_r_boundary <= tau_r.max())
        draw_contour_if_present(ax, X, Y, stalling_grid, 1e4, color, ":")
        draw_contour_if_present(ax, X, Y, stalling_grid, 1e5, color, "--")
        ax.plot(tau_c_line[mask], tau_r_boundary[mask], color=color, zorder=3)

    ax.set_title(title, pad=3)
    ax.set_xlabel(r"$\tau_c$ ($\mu$s)", labelpad=2)
    ax.set_ylabel(r"$\tau_r$ ($\mu$s)", labelpad=2)
    ax.set_xlim(tau_c.min(), tau_c.max())
    ax.set_ylim(tau_r.min(), tau_r.max())

    contour_handles = [
        Line2D([0], [0], color="black", linestyle="-", label="No stalling boundary"),
        Line2D([0], [0], color="black", linestyle=":", linewidth=1.65, label=r"Stall = $10^4$ $\mu$s"),
        Line2D([0], [0], color="black", linestyle="--", linewidth=1.65, label=r"Stall = $10^5$ $\mu$s"),
    ]
    capacity_handles = [
        Line2D(
            [0],
            [0],
            color=colors[int(run.get("capacity", run.get("t_count")))],
            label=run.get(
                "legend_label",
                f"{int(run.get('capacity', run.get('t_count')))} {capacity_label}",
            ),
        )
        for run in results
    ]

    leg_stall = fig.legend(
        contour_handles,
        [h.get_label() for h in contour_handles],
        loc="upper center",
        bbox_to_anchor=(0.30, 0.14),
        ncol=1,
        frameon=False,
        handlelength=1.8,
    )
    fig.add_artist(leg_stall)
    fig.legend(
        capacity_handles,
        [h.get_label() for h in capacity_handles],
        loc="upper center",
        bbox_to_anchor=(0.70, 0.14),
        ncol=1,
        frameon=False,
        handlelength=1.7,
    )
    fig.tight_layout(rect=[0, 0.16, 1, 1], pad=0.15)
    return fig


def save_for_paper(fig, filename):
    output_dirs = [PROJECT_ROOT / "paper_plots"]
    for out_dir in output_dirs:
        out_dir.mkdir(exist_ok=True)
        out_path = out_dir / filename
        fig.savefig(out_path, bbox_inches="tight", pad_inches=0.03)
        print(f"Saved {out_path}")


def load_tav_results(sequence_file, code_distance_override=None):
    data_file = tav_rldv.data_file_for_sequences_file(sequence_file)
    results_tav, code_distance_tav = tav_rldv.load_sweep_data(data_file)
    if code_distance_override is not None:
        code_distance_tav = int(code_distance_override)
    print(f"Loaded {len(results_tav)} T-count settings from {data_file}")
    print(f"  Using code distance d = {code_distance_tav}")
    for run in results_tav:
        rd = np.asarray(run["reaction_depths_list"])
        print(
            f"  T/cycle={run['t_count']:>2}: "
            f"cycles={run['n_cycles']:>7,}  max rd={rd.max():>3}  "
            f"mean rd={rd[rd > 0].mean():.2f}  total bp={run['total_bell_pairs_sum']:,}"
        )
    return results_tav, code_distance_tav, data_file


def plot_reaction_limited_on_ax(
    ax,
    results,
    tau_r,
    tau_c,
    d,
    title,
    capacity_label="blocks",
    stall_scale=1,
    color_transform=None,
    cmap_name="viridis",
):
    ids = [int(run.get("capacity", run.get("t_count"))) for run in results]
    colors = capacity_color_map(ids, cmap_name=cmap_name)
    tau_c_line = np.logspace(np.log10(tau_c.min()), np.log10(tau_c.max()), 240)
    X, Y = np.meshgrid(tau_c, tau_r)

    format_log_axes(ax, tau_c, tau_r)

    for run in results:
        run_id = int(run.get("capacity", run.get("t_count")))
        color = colors[run_id]
        if color_transform is not None:
            color = color_transform(color)
        reaction_depths = run["reaction_depths_list"]
        stalling_grid = stall_scale * compute_stalling_grid(reaction_depths, d, tau_r, tau_c)
        tau_r_boundary = get_no_stalling_boundary(reaction_depths, d, tau_c_line)
        mask = (tau_r_boundary >= tau_r.min()) & (tau_r_boundary <= tau_r.max())
        draw_contour_if_present(ax, X, Y, stalling_grid, 1e4, color, ":")
        draw_contour_if_present(ax, X, Y, stalling_grid, 1e5, color, "--")
        ax.plot(tau_c_line[mask], tau_r_boundary[mask], color=color, zorder=3)

    ax.set_title(title, pad=3)
    ax.set_xlabel(r"$\tau_c$ ($\mu$s)", labelpad=2)
    ax.set_ylabel(r"$\tau_r$ ($\mu$s)", labelpad=2)
    ax.set_xlim(tau_c.min(), tau_c.max())
    ax.set_ylim(tau_r.min(), tau_r.max())

    return [
        Line2D(
            [0],
            [0],
            color=(
                color_transform(colors[int(run.get("capacity", run.get("t_count")))])
                if color_transform is not None
                else colors[int(run.get("capacity", run.get("t_count")))]
            ),
            label=run.get(
                "legend_label",
                f"{int(run.get('capacity', run.get('t_count')))} {capacity_label}",
            ),
        )
        for run in results
    ]


def shared_stalling_handles():
    return [
        Line2D([0], [0], color="black", linestyle="-", label="No stalling boundary"),
        Line2D([0], [0], color="black", linestyle=":", linewidth=1.65, label=r"Stall = $10^4$ $\mu$s"),
        Line2D([0], [0], color="black", linestyle="--", linewidth=1.65, label=r"Stall = $10^5$ $\mu$s"),
    ]


def plot_reaction_limited_side_by_side(
    plot_specs,
    tau_r,
    tau_c,
    capacity_label,
    figsize=(10.5, 4.5),
    shared_capacity_legend=False,
    cmap_name="viridis",
):
    fig, axes = plt.subplots(1, len(plot_specs), figsize=figsize, sharex=True, sharey=True)
    if len(plot_specs) == 1:
        axes = [axes]

    shared_handles = None
    for ax, spec in zip(axes, plot_specs):
        handles = plot_reaction_limited_on_ax(
            ax,
            spec["results"],
            tau_r=tau_r,
            tau_c=tau_c,
            d=spec["code_distance"],
            title=spec["title"],
            capacity_label=capacity_label,
            stall_scale=spec.get("stall_scale", 1),
            cmap_name=spec.get("cmap_name", cmap_name),
        )
        if shared_capacity_legend:
            if shared_handles is None:
                shared_handles = handles
        else:
            ax.legend(
                handles,
                [h.get_label() for h in handles],
                loc="upper center",
                bbox_to_anchor=(0.5, -0.24),
                ncol=1,
                frameon=False,
                handlelength=1.7,
            )

    for ax in axes[1:]:
        ax.set_ylabel("")

    stall_handles = shared_stalling_handles()
    leg_stall = fig.legend(
        stall_handles,
        [h.get_label() for h in stall_handles],
        loc="upper center",
        bbox_to_anchor=(0.42, 0.08),
        ncol=1,
        frameon=False,
        handlelength=1.8,
    )
    fig.add_artist(leg_stall)

    if shared_capacity_legend and shared_handles is not None:
        fig.legend(
            shared_handles,
            [h.get_label() for h in shared_handles],
            loc="upper center",
            bbox_to_anchor=(0.60, 0.08),
            ncol=1,
            frameon=False,
            handlelength=1.7,
        )
        fig.tight_layout(rect=[0, 0.16, 1, 1], pad=0.15)
    else:
        fig.tight_layout(rect=[0, 0.14, 1, 1], pad=0.15)
    return fig


def render_tav_log_plot(sequence_file, title, code_distance_override=None):
    results_tav, code_distance_tav, data_file = load_tav_results(
        sequence_file, code_distance_override=code_distance_override
    )
    tau_r_tav = np.logspace(0, 2, 81)
    tau_c_tav = np.logspace(0, 2, 81)
    fig_tav = plot_reaction_limited(
        results_tav,
        tau_r=tau_r_tav,
        tau_c=tau_c_tav,
        d=code_distance_tav,
        title=title,
        capacity_label="T/cycle",
        cmap_name=TAV_CMAP_NAME,
    )
    output_name = f"t_av_reaction_limited_different_volume_{Path(sequence_file).stem}.pdf"
    plt.show()
    save_for_paper(fig_tav, output_name)
    return fig_tav, data_file


def render_tav_side_by_side_plot(sequence_specs):
    tau_r_tav = np.logspace(1, 3, 81)
    tau_c_tav = np.logspace(1, 3, 81)
    plot_specs = []
    data_files = []
    for spec in sequence_specs:
        results_tav, code_distance_tav, data_file = load_tav_results(
            spec["sequence_file"], code_distance_override=spec.get("code_distance")
        )
        plot_specs.append(
            {
                "results": results_tav,
                "code_distance": code_distance_tav,
                "title": spec["title"],
                "cmap_name": TAV_CMAP_NAME,
            }
        )
        data_files.append(data_file)

    fig_tav = plot_reaction_limited_side_by_side(
        plot_specs,
        tau_r=tau_r_tav,
        tau_c=tau_c_tav,
        capacity_label="T/cycle",
        shared_capacity_legend=True,
    )
    plt.show()
    save_for_paper(fig_tav, "t_av_reaction_limited_different_volume_side_by_side.pdf")
    return fig_tav, data_files


def load_av_results(circuit):
    data_file = rldv.data_file_for_circuit(circuit)
    results_av, code_distance_av = rldv.load_sweep_data(data_file)
    print(f"Loaded {len(results_av)} capacities from {data_file}")
    for run in results_av:
        rd = np.asarray(run["reaction_depths_list"])
        print(
            f"  capacity={run['capacity']:>5}: "
            f"cycles={run['n_cycles']:>9,}  max rd={rd.max():>3}  "
            f"mean rd={rd.mean():.2f}  total bp={run['total_bell_pairs_sum']:,}"
        )
    return results_av, code_distance_av, data_file


def render_av_log_plot(circuit, title, capacities):
    results_av, code_distance_av, data_file = load_av_results(circuit)
    results_plot = select_runs(results_av, capacities)
    tau_r_av = np.logspace(0, 2, 81)
    tau_c_av = np.logspace(0, 2, 81)
    fig_av = plot_reaction_limited(
        results_plot,
        tau_r=tau_r_av,
        tau_c=tau_c_av,
        d=code_distance_av,
        title=title,
        capacity_label="blocks",
        cmap_name=AV_CMAP_NAME,
    )
    output_name = f"reaction_limited_different_volume_{circuit}.pdf"
    plt.show()
    save_for_paper(fig_av, output_name)
    return fig_av, data_file


def render_av_side_by_side_plot(circuit_specs):
    tau_r_av = np.logspace(0, 2, 81)
    tau_c_av = np.logspace(0, 2, 81)
    plot_specs = []
    data_files = []
    for spec in circuit_specs:
        results_av, code_distance_av, data_file = load_av_results(spec["circuit"])
        plot_specs.append(
            {
                "results": select_runs(results_av, spec["capacities"]),
                "code_distance": code_distance_av,
                "title": spec["title"],
                "cmap_name": AV_CMAP_NAME,
            }
        )
        data_files.append(data_file)

    fig_av = plot_reaction_limited_side_by_side(
        plot_specs,
        tau_r=tau_r_av,
        tau_c=tau_c_av,
        capacity_label="blocks",
    )
    plt.show()
    save_for_paper(fig_av, "reaction_limited_different_volume_side_by_side.pdf")
    return fig_av, data_files


def av_use_case_specs_from_distance_table(circuit, architecture, error_rate, capacities):
    distance_table = PROJECT_ROOT / "resource_estimators" / "distance_table.csv"
    grouped = {}
    with distance_table.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["circuit"] != circuit:
                continue
            if row["architecture"] != architecture:
                continue
            if row["error_model"] != "circuit":
                continue
            if row["error_rate"] != str(error_rate):
                continue
            capacity = int(row["workspace_capacity"])
            if capacity not in capacities:
                continue
            grouped.setdefault(row["use_case"], []).append(row)

    specs = []
    for use_case, rows in grouped.items():
        min_distance = min(int(row["min_distance"]) for row in rows)
        specs.append(
            {
                "use_case": use_case,
                "use_case_label": display_use_case_label(rows[0]["use_case_label"]),
                "trotter_steps": int(rows[0]["trotter_steps"]),
                "code_distance": min_distance,
            }
        )

    specs.sort(key=lambda spec: spec["trotter_steps"], reverse=True)
    return specs


def render_av_tmm_use_cases_plot(capacities, error_rate="0.001"):
    results_av, _, data_file = load_av_results("tmm_paulis_commuted")
    results_plot = select_runs(results_av, capacities)
    tau_r_av = np.logspace(0, 2, 81)
    tau_c_av = np.logspace(0, 2, 81)

    use_case_specs = av_use_case_specs_from_distance_table(
        circuit="TMM",
        architecture="av",
        error_rate=error_rate,
        capacities=capacities,
    )
    plot_specs = [
        {
            "results": results_plot,
            "code_distance": spec["code_distance"],
            "title": f"{spec['use_case_label']}\n(k = {spec['trotter_steps']}, d = {spec['code_distance']})",
            "stall_scale": spec["trotter_steps"],
        }
        for spec in use_case_specs
    ]

    with plt.style.context(str(TWO_COLUMN_STYLE_FILE)):
        fig_av = plot_reaction_limited_side_by_side(
            plot_specs,
            tau_r=tau_r_av,
            tau_c=tau_c_av,
            capacity_label="blocks",
            figsize=(15.0, 4.5),
            shared_capacity_legend=True,
        )
        plt.show()
        save_for_paper(
            fig_av,
            "reaction_limited_different_volume_tmm_av_use_cases_side_by_side.pdf",
        )
    return fig_av, data_file, use_case_specs


def render_av_tmm_use_cases_two_rate_grid_plot(
    capacities,
    error_rates=("0.001", "0.0001"),
):
    results_av, _, data_file = load_av_results("tmm_paulis_commuted")
    results_plot = select_runs(results_av, capacities)
    tau_r_av = np.logspace(0, 2, 81)
    tau_c_av = np.logspace(0, 2, 81)

    use_case_specs_by_rate = {
        error_rate: av_use_case_specs_from_distance_table(
            circuit="TMM",
            architecture="av",
            error_rate=error_rate,
            capacities=capacities,
        )
        for error_rate in error_rates
    }

    with plt.style.context(str(TWO_COLUMN_STYLE_FILE)):
        fig, axes = plt.subplots(2, 3, figsize=(10.5, 7.4), sharex=True, sharey=True)
        shared_handles = None
        for row, error_rate in enumerate(error_rates):
            color_transform = None
            for col, spec in enumerate(use_case_specs_by_rate[error_rate]):
                ax = axes[row, col]
                if row == 0:
                    title = (
                        f"{spec['use_case_label']}\n"
                        f"($r$ = {spec['trotter_steps']}, $d$ = {spec['code_distance']})"
                    )
                else:
                    title = f"($r$ = {spec['trotter_steps']}, $d$ = {spec['code_distance']})"
                handles = plot_reaction_limited_on_ax(
                    ax,
                    results_plot,
                    tau_r=tau_r_av,
                    tau_c=tau_c_av,
                    d=spec["code_distance"],
                    title=title,
                    capacity_label="blocks",
                    stall_scale=spec["trotter_steps"],
                    color_transform=color_transform,
                    cmap_name=AV_CMAP_NAME,
                )
                if shared_handles is None and error_rate == "0.0001":
                    shared_handles = handles
                if row == 1:
                    ax.set_xlabel(r"$\tau_c$ ($\mu$s)", labelpad=2)
                if col == 0:
                    ax.set_ylabel(r"$\tau_r$ ($\mu$s)", labelpad=2)
                else:
                    ax.set_ylabel("")

        error_rate_handles = [
            Line2D(
                [0],
                [0],
                color=blend_color_toward_white("black", OVERLAY_LIGHTER_ERROR_RATE_BLEND),
                linewidth=1.6,
                label="0.001",
            ),
            Line2D([0], [0], color="black", linewidth=1.6, label="0.0001"),
        ]
        stall_handles = shared_stalling_handles()

        leg_stall = fig.legend(
            stall_handles,
            [h.get_label() for h in stall_handles],
            loc="upper center",
            bbox_to_anchor=(0.28, 0.06),
            ncol=1,
            title="Stalling",
            frameon=False,
            handlelength=1.8,
        )
        leg_stall.get_title().set_fontweight("bold")
        fig.add_artist(leg_stall)
        leg_capacity = fig.legend(
            shared_handles,
            [h.get_label() for h in shared_handles],
            loc="upper center",
            bbox_to_anchor=(0.52, 0.06),
            ncol=1,
            title="Workspace capacity",
            frameon=False,
            handlelength=1.7,
        )
        leg_capacity.get_title().set_fontweight("bold")
        fig.add_artist(leg_capacity)
        leg_error = fig.legend(
            error_rate_handles,
            [h.get_label() for h in error_rate_handles],
            loc="upper center",
            bbox_to_anchor=(0.77, 0.06),
            ncol=1,
            title="Error rate",
            frameon=False,
            handlelength=1.8,
        )
        leg_error.get_title().set_fontweight("bold")
        fig.tight_layout(rect=[0, 0.12, 1, 1], pad=0.18)
        save_for_paper(
            fig,
            "reaction_limited_different_volume_tmm_av_use_cases_two_rate_grid_0.0001_0.001.pdf",
        )
    return fig, data_file, use_case_specs_by_rate


def render_av_fermi_hubbard_two_rate_panels_plot(
    capacities=DEFAULT_FERMI_AV_CAPACITIES,
    error_rates=("0.001", "0.0001"),
):
    results_av, _, data_file = load_av_results("fermi_hubbard_2d_step_s4_universal_paulis_commuted")
    results_plot = select_runs(results_av, capacities)
    tau_r_av = np.logspace(0, 2, 81)
    tau_c_av = np.logspace(0, 2, 81)

    specs_by_rate = {
        error_rate: av_use_case_specs_from_distance_table(
            circuit="Fermi-Hubbard",
            architecture="av",
            error_rate=error_rate,
            capacities=capacities,
        )
        for error_rate in error_rates
    }

    with plt.style.context(str(TWO_COLUMN_STYLE_FILE)):
        fig, axes = plt.subplots(1, 2, figsize=(10.5, 5.2), sharex=True, sharey=True)
        if not isinstance(axes, np.ndarray):
            axes = np.array([axes])

        shared_handles = None
        for ax, error_rate in zip(axes, error_rates):
            color_transform = None
            spec = specs_by_rate[error_rate][0]
            handles = plot_reaction_limited_on_ax(
                ax,
                results_plot,
                tau_r=tau_r_av,
                tau_c=tau_c_av,
                d=spec["code_distance"],
                title=f"Fermi-Hubbard\n($r$ = 1, $d$ = {spec['code_distance']})",
                capacity_label="blocks",
                stall_scale=1,
                color_transform=color_transform,
                cmap_name=AV_CMAP_NAME,
            )
            if shared_handles is None and error_rate == "0.0001":
                shared_handles = handles
            ax.set_xlabel(r"$\tau_c$ ($\mu$s)", labelpad=2)
            ax.set_ylabel(r"$\tau_r$ ($\mu$s)", labelpad=2)

        axes[1].set_ylabel("")

        error_rate_handles = [
            Line2D(
                [0],
                [0],
                color=blend_color_toward_white("black", OVERLAY_LIGHTER_ERROR_RATE_BLEND),
                linewidth=1.6,
                label="0.001",
            ),
            Line2D([0], [0], color="black", linewidth=1.6, label="0.0001"),
        ]
        stall_handles = shared_stalling_handles()

        leg_stall = fig.legend(
            stall_handles,
            [h.get_label() for h in stall_handles],
            loc="upper center",
            bbox_to_anchor=(0.22, 0.06),
            ncol=1,
            title="Stalling",
            frameon=False,
            handlelength=1.8,
        )
        leg_stall.get_title().set_fontweight("bold")
        fig.add_artist(leg_stall)

        leg_capacity = fig.legend(
            shared_handles,
            [h.get_label() for h in shared_handles],
            loc="upper center",
            bbox_to_anchor=(0.52, 0.06),
            ncol=1,
            title="Workspace capacity",
            frameon=False,
            handlelength=1.7,
        )
        leg_capacity.get_title().set_fontweight("bold")
        fig.add_artist(leg_capacity)

        leg_error = fig.legend(
            error_rate_handles,
            [h.get_label() for h in error_rate_handles],
            loc="upper center",
            bbox_to_anchor=(0.82, 0.06),
            ncol=1,
            title="Error rate",
            frameon=False,
            handlelength=1.8,
        )
        leg_error.get_title().set_fontweight("bold")

        fig.tight_layout(rect=[0, 0.12, 1, 1], pad=0.18)
        save_for_paper(
            fig,
            "reaction_limited_different_volume_fermi_hubbard_av_two_rate_panels_0.0001_0.001.pdf",
        )
    return fig, data_file, specs_by_rate


def render_tav_two_rate_panels_plot(
    circuit,
    sequence_file,
    title_prefix,
    t_counts=DEFAULT_TMM_T_COUNTS,
    architecture="t-av",
    error_rates=("0.001", "0.0001"),
):
    results_tav, _, data_file = load_tav_results(sequence_file, code_distance_override=None)
    by_t_count = {int(run["t_count"]): run for run in results_tav}
    results_plot = [by_t_count[int(t_count)] for t_count in t_counts]
    tau_r_tav = np.logspace(1, 3, 81)
    tau_c_tav = np.logspace(1, 3, 81)

    specs_by_rate = {
        error_rate: tav_use_case_specs_from_distance_table(
            circuit=circuit,
            architecture=architecture,
            error_rate=error_rate,
            t_counts=t_counts,
        )
        for error_rate in error_rates
    }

    with plt.style.context(str(TWO_COLUMN_STYLE_FILE)):
        n_use_cases = max(len(specs) for specs in specs_by_rate.values())
        error_rate_handles = [
            Line2D(
                [0],
                [0],
                color=blend_color_toward_white("black", OVERLAY_LIGHTER_ERROR_RATE_BLEND),
                linewidth=1.6,
                label="0.001",
            ),
            Line2D([0], [0], color="black", linewidth=1.6, label="0.0001"),
        ]
        stall_handles = shared_stalling_handles()

        if n_use_cases > 1:
            fig, axes = plt.subplots(2, n_use_cases, figsize=(10.5, 7.4), sharex=True, sharey=True)
            shared_handles = None
            for row, error_rate in enumerate(error_rates):
                color_transform = None
                if error_rate == "0.001":
                    color_transform = lambda c: blend_color_toward_white(
                        c, OVERLAY_LIGHTER_ERROR_RATE_BLEND
                    )
                for col, spec in enumerate(specs_by_rate[error_rate]):
                    ax = axes[row, col]
                    title = (
                        f"{spec['use_case_label']}\n"
                        f"($r$ = {spec['trotter_steps']}, $d$ = {spec['code_distance']})"
                        if row == 0
                        else f"($r$ = {spec['trotter_steps']}, $d$ = {spec['code_distance']})"
                    )
                    handles = plot_reaction_limited_on_ax(
                        ax,
                        results_plot,
                        tau_r=tau_r_tav,
                        tau_c=tau_c_tav,
                        d=spec["code_distance"],
                        title=title,
                        capacity_label="T/cycle",
                        stall_scale=spec["trotter_steps"],
                        color_transform=color_transform,
                        cmap_name=TAV_CMAP_NAME,
                    )
                    if shared_handles is None and error_rate == "0.0001":
                        shared_handles = handles
                    if row == 1:
                        ax.set_xlabel(r"$\tau_c$ ($\mu$s)", labelpad=2)
                    if col == 0:
                        ax.set_ylabel(r"$\tau_r$ ($\mu$s)", labelpad=2)
                    else:
                        ax.set_ylabel("")

            leg_stall = fig.legend(
                stall_handles,
                [h.get_label() for h in stall_handles],
                loc="upper center",
                bbox_to_anchor=(0.22, 0.06),
                ncol=1,
                title="Stalling",
                frameon=False,
                handlelength=1.8,
            )
            leg_stall.get_title().set_fontweight("bold")
            fig.add_artist(leg_stall)

            leg_capacity = fig.legend(
                shared_handles,
                [h.get_label() for h in shared_handles],
                loc="upper center",
                bbox_to_anchor=(0.52, 0.06),
                ncol=1,
                title="T/cycle",
                frameon=False,
                handlelength=1.7,
            )
            leg_capacity.get_title().set_fontweight("bold")
            fig.add_artist(leg_capacity)

            leg_error = fig.legend(
                error_rate_handles,
                [h.get_label() for h in error_rate_handles],
                loc="upper center",
                bbox_to_anchor=(0.82, 0.06),
                ncol=1,
                title="Error rate",
                frameon=False,
                handlelength=1.8,
            )
            leg_error.get_title().set_fontweight("bold")
            fig.tight_layout(rect=[0, 0.12, 1, 1], pad=0.18)
            safe_prefix = title_prefix.lower().replace(" ", "_").replace("-", "_")
            save_for_paper(
                fig,
                f"reaction_limited_different_volume_{safe_prefix}_t_av_two_rate_grid_0.0001_0.001.pdf",
            )
        else:
            fig, axes = plt.subplots(1, 2, figsize=(10.5, 5.2), sharex=True, sharey=True)
            if not isinstance(axes, np.ndarray):
                axes = np.array([axes])

            shared_handles = None
            for ax, error_rate in zip(axes, error_rates):
                color_transform = None
                if error_rate == "0.001":
                    color_transform = lambda c: blend_color_toward_white(
                        c, OVERLAY_LIGHTER_ERROR_RATE_BLEND
                    )
                spec = specs_by_rate[error_rate][0]
                handles = plot_reaction_limited_on_ax(
                    ax,
                    results_plot,
                    tau_r=tau_r_tav,
                    tau_c=tau_c_tav,
                    d=spec["code_distance"],
                    title=f"{title_prefix}\n($r$ = {spec['trotter_steps']}, $d$ = {spec['code_distance']})",
                    capacity_label="T/cycle",
                    stall_scale=spec["trotter_steps"],
                    color_transform=color_transform,
                    cmap_name=TAV_CMAP_NAME,
                )
                if shared_handles is None and error_rate == "0.0001":
                    shared_handles = handles
                ax.set_xlabel(r"$\tau_c$ ($\mu$s)", labelpad=2)
                ax.set_ylabel(r"$\tau_r$ ($\mu$s)", labelpad=2)

            axes[1].set_ylabel("")

            leg_stall = fig.legend(
                stall_handles,
                [h.get_label() for h in stall_handles],
                loc="upper center",
                bbox_to_anchor=(0.22, 0.06),
                ncol=1,
                title="Stalling",
                frameon=False,
                handlelength=1.8,
            )
            leg_stall.get_title().set_fontweight("bold")
            fig.add_artist(leg_stall)

            leg_capacity = fig.legend(
                shared_handles,
                [h.get_label() for h in shared_handles],
                loc="upper center",
                bbox_to_anchor=(0.52, 0.06),
                ncol=1,
                title="T/cycle",
                frameon=False,
                handlelength=1.7,
            )
            leg_capacity.get_title().set_fontweight("bold")
            fig.add_artist(leg_capacity)

            leg_error = fig.legend(
                error_rate_handles,
                [h.get_label() for h in error_rate_handles],
                loc="upper center",
                bbox_to_anchor=(0.82, 0.06),
                ncol=1,
                title="Error rate",
                frameon=False,
                handlelength=1.8,
            )
            leg_error.get_title().set_fontweight("bold")

            fig.tight_layout(rect=[0, 0.12, 1, 1], pad=0.18)
            safe_prefix = title_prefix.lower().replace(" ", "_").replace("-", "_")
            save_for_paper(
                fig,
                f"reaction_limited_different_volume_{safe_prefix}_t_av_two_rate_panels_0.0001_0.001.pdf",
            )
    return fig, data_file, specs_by_rate


def render_tmm_av_and_tav_two_rate_grids_side_by_side_plot(
    av_capacities=DEFAULT_TMM_AV_CAPACITIES,
    tav_t_counts=DEFAULT_TMM_T_COUNTS,
    error_rates=("0.001", "0.0001"),
    av_cmap="Blues_r",
    tav_cmap="Oranges_r",
):
    av_results, _, av_data_file = load_av_results("tmm_paulis_commuted")
    av_results_plot = select_runs(av_results, av_capacities)
    base_capacity = av_capacities[0]
    for run in av_results_plot:
        run["legend_label"] = f"{run['capacity'] / base_capacity:.2f}x"
    tav_results, _, tav_data_file = load_tav_results(
        "transversal-logical-blocks-tmm.json", code_distance_override=None
    )
    tav_by_t_count = {int(run["t_count"]): run for run in tav_results}
    tav_results_plot = [tav_by_t_count[int(t_count)] for t_count in tav_t_counts]
    for run in tav_results_plot:
        run["legend_label"] = str(int(run["t_count"]))

    tau_r_av = np.logspace(0, 2, 81)
    tau_c_av = np.logspace(0, 3, 121)
    tau_r_tav = tau_r_av
    tau_c_tav = tau_c_av

    av_specs_by_rate = {
        error_rate: av_use_case_specs_from_distance_table(
            circuit="TMM",
            architecture="av",
            error_rate=error_rate,
            capacities=av_capacities,
        )
        for error_rate in error_rates
    }
    tav_specs_by_rate = {
        error_rate: tav_use_case_specs_from_distance_table(
            circuit="TMM",
            architecture="t-av",
            error_rate=error_rate,
            t_counts=tav_t_counts,
        )
        for error_rate in error_rates
    }

    with plt.style.context(str(TWO_COLUMN_STYLE_FILE)):
        fig, axes = plt.subplots(2, 6, figsize=(17.5, 6.17), sharex=True, sharey=True)

        av_shared_handles = None
        tav_shared_handles = None

        for row, error_rate in enumerate(error_rates):
            color_transform = None

            for col, spec in enumerate(av_specs_by_rate[error_rate]):
                ax = axes[row, col]
                title = (
                    f"{spec['use_case_label']}\n"
                    f"($r$ = {spec['trotter_steps']}, $d$ = {spec['code_distance']})"
                    if row == 0
                    else f"($r$ = {spec['trotter_steps']}, $d$ = {spec['code_distance']})"
                )
                handles = plot_reaction_limited_on_ax(
                    ax,
                    av_results_plot,
                    tau_r=tau_r_av,
                    tau_c=tau_c_av,
                    d=spec["code_distance"],
                    title=title,
                    capacity_label="blocks",
                    stall_scale=spec["trotter_steps"],
                    color_transform=color_transform,
                    cmap_name=av_cmap,
                )
                if av_shared_handles is None and error_rate == "0.0001":
                    av_shared_handles = handles
                if row == 1:
                    ax.set_xlabel(r"$\tau_c$ ($\mu$s)", labelpad=2)
                else:
                    ax.set_xlabel("")
                if col == 0:
                    ax.set_ylabel(r"$\tau_r$ ($\mu$s)", labelpad=2)
                else:
                    ax.set_ylabel("")

            for offset, spec in enumerate(tav_specs_by_rate[error_rate], start=3):
                ax = axes[row, offset]
                title = (
                    f"{spec['use_case_label']}\n"
                    f"($r$ = {spec['trotter_steps']}, $d$ = {spec['code_distance']})"
                    if row == 0
                    else f"($r$ = {spec['trotter_steps']}, $d$ = {spec['code_distance']})"
                )
                handles = plot_reaction_limited_on_ax(
                    ax,
                    tav_results_plot,
                    tau_r=tau_r_tav,
                    tau_c=tau_c_tav,
                    d=spec["code_distance"],
                    title=title,
                    capacity_label="T/cycle",
                    stall_scale=spec["trotter_steps"],
                    color_transform=color_transform,
                    cmap_name=tav_cmap,
                )
                if tav_shared_handles is None and error_rate == "0.0001":
                    tav_shared_handles = handles
                if row == 1:
                    ax.set_xlabel(r"$\tau_c$ ($\mu$s)", labelpad=2)
                else:
                    ax.set_xlabel("")
                if offset == 3:
                    ax.set_ylabel(r"$\tau_r$ ($\mu$s)", labelpad=2)
                else:
                    ax.set_ylabel("")

            exp = int(round(np.log10(float(error_rate))))
            axes[row, 5].yaxis.set_label_position("right")
            axes[row, 5].set_ylabel(f"$p = 10^{{{exp}}}$", rotation=90, labelpad=8)

        stall_handles = shared_stalling_handles()

        leg_stall = fig.legend(
            stall_handles,
            [h.get_label() for h in stall_handles],
            loc="upper center",
            bbox_to_anchor=(0.22, 0.06),
            ncol=3,
            title="Stalling",
            frameon=False,
            handlelength=1.8,
        )
        leg_stall.get_title().set_fontweight("bold")
        fig.add_artist(leg_stall)

        leg_av = fig.legend(
            av_shared_handles,
            [h.get_label() for h in av_shared_handles],
            loc="upper center",
            bbox_to_anchor=(0.60, 0.06),
            ncol=3,
            title="Workspace",
            frameon=False,
            handlelength=1.7,
        )
        leg_av.get_title().set_fontweight("bold")
        fig.add_artist(leg_av)

        leg_tav = fig.legend(
            tav_shared_handles,
            [h.get_label() for h in tav_shared_handles],
            loc="upper center",
            bbox_to_anchor=(0.84, 0.06),
            ncol=3,
            title="T/cycle",
            frameon=False,
            handlelength=1.7,
        )
        leg_tav.get_title().set_fontweight("bold")

        fig.tight_layout(rect=[0, 0.08, 1, 1], pad=0.18)
        save_for_paper(
            fig,
            "reaction_limited_different_volume_tmm_av_and_t_av_two_rate_grids_side_by_side_0.0001_0.001.pdf",
        )
    return fig, {"av": av_data_file, "tav": tav_data_file}, {
        "av": av_specs_by_rate,
        "tav": tav_specs_by_rate,
    }


def render_fermi_hubbard_av_and_tav_two_rate_grids_side_by_side_plot(
    av_capacities=DEFAULT_FERMI_AV_CAPACITIES,
    tav_t_counts=DEFAULT_TMM_T_COUNTS,
    error_rates=("0.001", "0.0001"),
    av_cmap="Blues_r",
    tav_cmap="Oranges_r",
):
    av_results, _, av_data_file = load_av_results(
        "fermi_hubbard_2d_step_s4_universal_paulis_commuted"
    )
    av_results_plot = select_runs(av_results, av_capacities)
    base_capacity = av_capacities[0]
    for run in av_results_plot:
        ratio = run["capacity"] / base_capacity
        run["legend_label"] = f"{ratio:.2f}x"
    tav_results, _, tav_data_file = load_tav_results(
        "transversal-logical-blocks-fermi-hubbard.json", code_distance_override=None
    )
    tav_by_t_count = {int(run["t_count"]): run for run in tav_results}
    tav_results_plot = [tav_by_t_count[int(t_count)] for t_count in tav_t_counts]
    for run in tav_results_plot:
        run["legend_label"] = str(int(run["t_count"]))

    tau_r = np.logspace(0, 2, 81)
    tau_c = np.logspace(0, 3, 121)

    av_specs_by_rate = {
        error_rate: av_use_case_specs_from_distance_table(
            circuit="Fermi-Hubbard",
            architecture="av",
            error_rate=error_rate,
            capacities=av_capacities,
        )
        for error_rate in error_rates
    }
    tav_specs_by_rate = {
        error_rate: tav_use_case_specs_from_distance_table(
            circuit="Fermi-Hubbard",
            architecture="t-av",
            error_rate=error_rate,
            t_counts=tav_t_counts,
        )
        for error_rate in error_rates
    }

    with plt.style.context(str(TWO_COLUMN_STYLE_FILE)):
        fig, axes = plt.subplots(2, 2, figsize=(8.5, 9.2), sharex=True, sharey=True)

        av_shared_handles = None
        tav_shared_handles = None

        for row, error_rate in enumerate(error_rates):
            color_transform = None

            av_spec = av_specs_by_rate[error_rate][0]
            title_av = (
                f"AV Fermi-Hubbard\n"
                f"($r$ = {av_spec['trotter_steps']}, $d$ = {av_spec['code_distance']})"
                if row == 0
                else f"($r$ = {av_spec['trotter_steps']}, $d$ = {av_spec['code_distance']})"
            )
            handles_av = plot_reaction_limited_on_ax(
                axes[row, 0],
                av_results_plot,
                tau_r=tau_r,
                tau_c=tau_c,
                d=av_spec["code_distance"],
                title=title_av,
                capacity_label="blocks",
                stall_scale=av_spec["trotter_steps"],
                color_transform=color_transform,
                cmap_name=av_cmap,
            )
            if av_shared_handles is None and error_rate == "0.0001":
                av_shared_handles = handles_av

            tav_spec = tav_specs_by_rate[error_rate][0]
            title_tav = (
                f"T-AV Fermi-Hubbard\n"
                f"($r$ = {tav_spec['trotter_steps']}, $d$ = {tav_spec['code_distance']})"
                if row == 0
                else f"($r$ = {tav_spec['trotter_steps']}, $d$ = {tav_spec['code_distance']})"
            )
            handles_tav = plot_reaction_limited_on_ax(
                axes[row, 1],
                tav_results_plot,
                tau_r=tau_r,
                tau_c=tau_c,
                d=tav_spec["code_distance"],
                title=title_tav,
                capacity_label="T/cycle",
                stall_scale=tav_spec["trotter_steps"],
                color_transform=color_transform,
                cmap_name=tav_cmap,
            )
            if tav_shared_handles is None and error_rate == "0.0001":
                tav_shared_handles = handles_tav

            if row == 1:
                axes[row, 0].set_xlabel(r"$\tau_c$ ($\mu$s)", labelpad=2)
                axes[row, 1].set_xlabel(r"$\tau_c$ ($\mu$s)", labelpad=2)
            axes[row, 0].set_ylabel(r"$\tau_r$ ($\mu$s)", labelpad=2)
            exp = int(round(np.log10(float(error_rate))))
            axes[row, 1].yaxis.set_label_position("right")
            axes[row, 1].set_ylabel(f"$p = 10^{{{exp}}}$", rotation=90, labelpad=8)

        stall_handles = shared_stalling_handles()

        leg_stall = fig.legend(
            stall_handles,
            [h.get_label() for h in stall_handles],
            loc="upper center",
            bbox_to_anchor=(0.26, 0.22),
            ncol=1,
            title="Stalling",
            frameon=False,
            handlelength=1.8,
        )
        leg_stall.get_title().set_fontweight("bold")
        fig.add_artist(leg_stall)

        leg_av = fig.legend(
            av_shared_handles,
            [h.get_label() for h in av_shared_handles],
            loc="upper center",
            bbox_to_anchor=(0.62, 0.22),
            ncol=1,
            title="Workspace",
            frameon=False,
            handlelength=1.7,
        )
        leg_av.get_title().set_fontweight("bold")
        fig.add_artist(leg_av)

        leg_tav = fig.legend(
            tav_shared_handles,
            [h.get_label() for h in tav_shared_handles],
            loc="upper center",
            bbox_to_anchor=(0.88, 0.22),
            ncol=1,
            title="T/cycle",
            frameon=False,
            handlelength=1.7,
        )
        leg_tav.get_title().set_fontweight("bold")

        fig.tight_layout(rect=[0, 0.26, 1, 1], pad=0.18)
        save_for_paper(
            fig,
            "reaction_limited_different_volume_fermi_hubbard_av_and_t_av_two_rate_grids_side_by_side_0.0001_0.001.pdf",
        )
    return fig, {"av": av_data_file, "tav": tav_data_file}, {
        "av": av_specs_by_rate,
        "tav": tav_specs_by_rate,
    }


def render_tmm_gap_av_and_tav_two_rate_grids_side_by_side_plot(
    av_capacities=DEFAULT_TMM_AV_CAPACITIES,
    tav_t_counts=DEFAULT_TMM_T_COUNTS,
    error_rates=("0.001", "0.0001"),
    av_cmap="Blues_r",
    tav_cmap="Oranges_r",
):
    """4-panel reaction-limited plot for TMM Stat-QPE-gap only:
    rows = error rates (1e-3, 1e-4), cols = architecture (AV, t-AV).
    Mirrors the Fermi-Hubbard side-by-side layout.
    """
    av_results, _, av_data_file = load_av_results("tmm_paulis_commuted")
    av_results_plot = select_runs(av_results, av_capacities)
    base_capacity = av_capacities[0]
    for run in av_results_plot:
        ratio = run["capacity"] / base_capacity
        run["legend_label"] = f"{ratio:.2f}x"
    tav_results, _, tav_data_file = load_tav_results(
        "transversal-logical-blocks-tmm.json", code_distance_override=None
    )
    tav_by_t_count = {int(run["t_count"]): run for run in tav_results}
    tav_results_plot = [tav_by_t_count[int(t_count)] for t_count in tav_t_counts]
    for run in tav_results_plot:
        run["legend_label"] = str(int(run["t_count"]))

    tau_r = np.logspace(0, 2, 81)
    tau_c = np.logspace(0, 3, 121)

    target_use_case = "stat_qpe_gap"

    def pick_gap(specs):
        for spec in specs:
            if spec["use_case"] == target_use_case:
                return spec
        raise ValueError(
            f"use_case {target_use_case!r} not found among {[s['use_case'] for s in specs]}"
        )

    av_specs_by_rate = {
        error_rate: pick_gap(
            av_use_case_specs_from_distance_table(
                circuit="TMM",
                architecture="av",
                error_rate=error_rate,
                capacities=av_capacities,
            )
        )
        for error_rate in error_rates
    }
    tav_specs_by_rate = {
        error_rate: pick_gap(
            tav_use_case_specs_from_distance_table(
                circuit="TMM",
                architecture="t-av",
                error_rate=error_rate,
                t_counts=tav_t_counts,
            )
        )
        for error_rate in error_rates
    }

    with plt.style.context(str(TWO_COLUMN_STYLE_FILE)):
        fig, axes = plt.subplots(2, 2, figsize=(8.5, 9.2), sharex=True, sharey=True)

        av_shared_handles = None
        tav_shared_handles = None

        for row, error_rate in enumerate(error_rates):
            color_transform = None

            av_spec = av_specs_by_rate[error_rate]
            title_av = (
                f"AV TMM Stat-QPE-gap\n"
                f"($r$ = {av_spec['trotter_steps']}, $d$ = {av_spec['code_distance']})"
                if row == 0
                else f"($r$ = {av_spec['trotter_steps']}, $d$ = {av_spec['code_distance']})"
            )
            handles_av = plot_reaction_limited_on_ax(
                axes[row, 0],
                av_results_plot,
                tau_r=tau_r,
                tau_c=tau_c,
                d=av_spec["code_distance"],
                title=title_av,
                capacity_label="blocks",
                stall_scale=av_spec["trotter_steps"],
                color_transform=color_transform,
                cmap_name=av_cmap,
            )
            if av_shared_handles is None and error_rate == "0.0001":
                av_shared_handles = handles_av

            tav_spec = tav_specs_by_rate[error_rate]
            title_tav = (
                f"T-AV TMM Stat-QPE-gap\n"
                f"($r$ = {tav_spec['trotter_steps']}, $d$ = {tav_spec['code_distance']})"
                if row == 0
                else f"($r$ = {tav_spec['trotter_steps']}, $d$ = {tav_spec['code_distance']})"
            )
            handles_tav = plot_reaction_limited_on_ax(
                axes[row, 1],
                tav_results_plot,
                tau_r=tau_r,
                tau_c=tau_c,
                d=tav_spec["code_distance"],
                title=title_tav,
                capacity_label="T/cycle",
                stall_scale=tav_spec["trotter_steps"],
                color_transform=color_transform,
                cmap_name=tav_cmap,
            )
            if tav_shared_handles is None and error_rate == "0.0001":
                tav_shared_handles = handles_tav

            if row == 1:
                axes[row, 0].set_xlabel(r"$\tau_c$ ($\mu$s)", labelpad=2)
                axes[row, 1].set_xlabel(r"$\tau_c$ ($\mu$s)", labelpad=2)
            axes[row, 0].set_ylabel(r"$\tau_r$ ($\mu$s)", labelpad=2)
            exp = int(round(np.log10(float(error_rate))))
            axes[row, 1].yaxis.set_label_position("right")
            axes[row, 1].set_ylabel(f"$p = 10^{{{exp}}}$", rotation=90, labelpad=8)

        stall_handles = shared_stalling_handles()

        leg_stall = fig.legend(
            stall_handles,
            [h.get_label() for h in stall_handles],
            loc="upper center",
            bbox_to_anchor=(0.26, 0.22),
            ncol=1,
            title="Stalling",
            frameon=False,
            handlelength=1.8,
        )
        leg_stall.get_title().set_fontweight("bold")
        fig.add_artist(leg_stall)

        leg_av = fig.legend(
            av_shared_handles,
            [h.get_label() for h in av_shared_handles],
            loc="upper center",
            bbox_to_anchor=(0.62, 0.22),
            ncol=1,
            title="Workspace",
            frameon=False,
            handlelength=1.7,
        )
        leg_av.get_title().set_fontweight("bold")
        fig.add_artist(leg_av)

        leg_tav = fig.legend(
            tav_shared_handles,
            [h.get_label() for h in tav_shared_handles],
            loc="upper center",
            bbox_to_anchor=(0.88, 0.22),
            ncol=1,
            title="T/cycle",
            frameon=False,
            handlelength=1.7,
        )
        leg_tav.get_title().set_fontweight("bold")

        fig.tight_layout(rect=[0, 0.26, 1, 1], pad=0.18)
        save_for_paper(
            fig,
            "reaction_limited_different_volume_tmm_stat_qpe_gap_av_and_t_av_two_rate_grids_side_by_side_0.0001_0.001.pdf",
        )
    return fig, {"av": av_data_file, "tav": tav_data_file}, {
        "av": av_specs_by_rate,
        "tav": tav_specs_by_rate,
    }


def render_tmm_gap_tav_two_rates_side_by_side_plot(
    tav_t_counts=DEFAULT_TMM_T_COUNTS,
    error_rates=("0.001", "0.0001"),
    tav_cmap="Oranges_r",
):
    """1x2 reaction-limited plot, TMM Stat-QPE-gap, t-AV only.
    Columns = error rates (1e-3 left, 1e-4 right). Same data layer as the
    right column of the 4-panel TMM-gap plot."""
    tav_results, _, tav_data_file = load_tav_results(
        "transversal-logical-blocks-tmm.json", code_distance_override=None
    )
    tav_by_t_count = {int(run["t_count"]): run for run in tav_results}
    tav_results_plot = [tav_by_t_count[int(t_count)] for t_count in tav_t_counts]
    for run in tav_results_plot:
        run["legend_label"] = str(int(run["t_count"]))

    tau_r = np.logspace(0, 2, 81)
    tau_c = np.logspace(0, 3, 121)

    target_use_case = "stat_qpe_gap"

    def pick_gap(specs):
        for spec in specs:
            if spec["use_case"] == target_use_case:
                return spec
        raise ValueError(
            f"use_case {target_use_case!r} not found among {[s['use_case'] for s in specs]}"
        )

    tav_specs_by_rate = {
        error_rate: pick_gap(
            tav_use_case_specs_from_distance_table(
                circuit="TMM",
                architecture="t-av",
                error_rate=error_rate,
                t_counts=tav_t_counts,
            )
        )
        for error_rate in error_rates
    }

    with plt.style.context(str(TWO_COLUMN_STYLE_FILE)):
        fig, axes = plt.subplots(1, len(error_rates), figsize=(8.5, 4.6),
                                  sharex=True, sharey=True)

        tav_shared_handles = None

        for col, error_rate in enumerate(error_rates):
            ax = axes[col]
            spec = tav_specs_by_rate[error_rate]
            handles = plot_reaction_limited_on_ax(
                ax,
                tav_results_plot,
                tau_r=tau_r,
                tau_c=tau_c,
                d=spec["code_distance"],
                title="",
                capacity_label="T/cycle",
                stall_scale=spec["trotter_steps"],
                color_transform=None,
                cmap_name=tav_cmap,
            )
            if tav_shared_handles is None and error_rate == "0.0001":
                tav_shared_handles = handles

            exp = int(round(np.log10(float(error_rate))))
            ax.set_title(rf"$p = 10^{{{exp}}}$", pad=3)

            ax.set_xlabel(r"$\tau_c$ ($\mu$s)", labelpad=2)
            if col == 0:
                ax.set_ylabel(r"$\tau_r$ ($\mu$s)", labelpad=2)
            else:
                ax.set_ylabel("")

        stall_handles = shared_stalling_handles()

        leg_stall = fig.legend(
            stall_handles,
            [h.get_label() for h in stall_handles],
            loc="upper center",
            bbox_to_anchor=(0.30, 0.10),
            ncol=1,
            title="Stalling",
            frameon=False,
            handlelength=1.8,
        )
        leg_stall.get_title().set_fontweight("bold")
        fig.add_artist(leg_stall)

        leg_tav = fig.legend(
            tav_shared_handles,
            [h.get_label() for h in tav_shared_handles],
            loc="upper center",
            bbox_to_anchor=(0.72, 0.10),
            ncol=1,
            title="T/cycle",
            frameon=False,
            handlelength=1.7,
        )
        leg_tav.get_title().set_fontweight("bold")

        fig.tight_layout(rect=[0, 0.16, 1, 1], pad=0.18)
        save_for_paper(
            fig,
            "reaction_limited_tmm_stat_qpe_gap_t_av_two_rates_side_by_side_0.0001_0.001.pdf",
        )
    return fig, {"tav": tav_data_file}, {"tav": tav_specs_by_rate}


def tav_use_case_specs_from_distance_table(circuit, architecture, error_rate, t_counts):
    distance_table = PROJECT_ROOT / "resource_estimators" / "distance_table.csv"
    error_model_by_architecture = {"t-av": "atoms", "transversal": "circuit"}
    error_model = error_model_by_architecture[architecture]
    grouped = {}
    with distance_table.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["circuit"] != circuit:
                continue
            if row["architecture"] != architecture:
                continue
            if row["error_model"] != error_model:
                continue
            if row["error_rate"] != str(error_rate):
                continue
            t_count = int(row["t_count_per_cycle"])
            if t_count not in t_counts:
                continue
            grouped.setdefault(row["use_case"], []).append(row)

    specs = []
    for use_case, rows in grouped.items():
        min_distance = min(int(row["min_distance"]) for row in rows)
        specs.append(
            {
                "use_case": use_case,
                "use_case_label": display_use_case_label(rows[0]["use_case_label"]),
                "trotter_steps": int(rows[0]["trotter_steps"]),
                "code_distance": min_distance,
            }
        )
    specs.sort(key=lambda spec: spec["trotter_steps"], reverse=True)
    return specs


def render_tav_tmm_use_cases_plot(t_counts, error_rate="0.001", architecture="t-av"):
    results_tav, _, data_file = load_tav_results(
        "transversal-logical-blocks-tmm.json", code_distance_override=None
    )
    by_t_count = {int(run["t_count"]): run for run in results_tav}
    results_plot = [by_t_count[t_count] for t_count in t_counts]
    tau_r_tav = np.logspace(1, 3, 81)
    tau_c_tav = np.logspace(1, 3, 81)

    use_case_specs = tav_use_case_specs_from_distance_table(
        circuit="TMM",
        architecture=architecture,
        error_rate=error_rate,
        t_counts=t_counts,
    )
    plot_specs = [
        {
            "results": results_plot,
            "code_distance": spec["code_distance"],
            "title": f"{spec['use_case_label']}\n(k = {spec['trotter_steps']}, d = {spec['code_distance']})",
            "stall_scale": spec["trotter_steps"],
        }
        for spec in use_case_specs
    ]

    fig_tav = plot_reaction_limited_side_by_side(
        plot_specs,
        tau_r=tau_r_tav,
        tau_c=tau_c_tav,
        capacity_label="T/cycle",
        figsize=(15.0, 4.5),
        shared_capacity_legend=True,
    )
    plt.show()
    output_name = f"t_av_reaction_limited_different_volume_tmm_{architecture}_use_cases_side_by_side.pdf"
    save_for_paper(fig_tav, output_name)
    return fig_tav, data_file, use_case_specs


def plot_tmm_use_case_summary(architecture="transversal", error_rate="0.001", t_counts=(1, 2, 3, 4, 5)):
    use_case_specs = tav_use_case_specs_from_distance_table(
        circuit="TMM",
        architecture=architecture,
        error_rate=error_rate,
        t_counts=t_counts,
    )
    labels = [spec["use_case_label"] for spec in use_case_specs]
    multipliers = [spec["trotter_steps"] for spec in use_case_specs]
    distances = [spec["code_distance"] for spec in use_case_specs]
    x = np.arange(len(use_case_specs))

    fig, ax_mult = plt.subplots(figsize=(6.8, 3.4))
    bar_color = plt.get_cmap("viridis")(0.25)
    line_color = plt.get_cmap("viridis")(0.82)
    ax_mult.bar(x, multipliers, width=0.58, color=bar_color, alpha=0.92)
    ax_mult.set_xticks(x)
    ax_mult.set_xticklabels(labels)
    ax_mult.set_ylabel("Trotter-step multiplier")
    ax_mult.set_yscale("log")
    ax_mult.yaxis.set_major_formatter(mticker.LogFormatterMathtext(base=10))
    ax_mult.grid(axis="y", which="major", color="0.88", linewidth=0.55)
    ax_mult.set_axisbelow(True)

    ax_d = ax_mult.twinx()
    ax_d.plot(x, distances, color=line_color, marker="o", linewidth=1.4, markersize=5)
    ax_d.set_ylabel("Chosen code distance d")
    ax_d.set_ylim(0, max(distances) + 3)

    for xi, multiplier, distance in zip(x, multipliers, distances):
        ax_mult.text(xi, multiplier * 1.08, f"k={multiplier}", ha="center", va="bottom", fontsize=8)
        ax_d.text(xi, distance + 0.35, f"d={distance}", ha="center", va="bottom", fontsize=8, color=line_color)

    ax_mult.set_title(f"TMM use-case summary ({architecture}, error rate = {error_rate})", pad=3)
    fig.tight_layout(pad=0.2)
    output_name = f"tmm_use_case_summary_{architecture}_error_rate_{str(error_rate).replace('.', 'p')}.pdf"
    save_for_paper(fig, output_name)
    return fig, use_case_specs


def plot_tmm_no_stalling_summary(architecture="transversal", error_rate="0.001", t_counts=(1, 2, 3, 4, 5)):
    results_tav, _, _ = load_tav_results("transversal-logical-blocks-tmm.json", code_distance_override=None)
    by_t_count = {int(run["t_count"]): run for run in results_tav}
    use_case_specs = tav_use_case_specs_from_distance_table(
        circuit="TMM",
        architecture=architecture,
        error_rate=error_rate,
        t_counts=t_counts,
    )

    fig, ax = plt.subplots(figsize=(6.8, 3.6))
    cmap = plt.get_cmap("viridis")
    colors = [cmap(v) for v in np.linspace(0.2, 0.85, len(use_case_specs))]
    summary_rows = []
    for color, spec in zip(colors, use_case_specs):
        slopes = []
        for t_count in t_counts:
            run = by_t_count[int(t_count)]
            max_rd = max(run["reaction_depths_list"])
            slope = spec["code_distance"] / max_rd
            slopes.append(slope)
            summary_rows.append(
                {
                    "use_case": spec["use_case"],
                    "use_case_label": spec["use_case_label"],
                    "trotter_steps": spec["trotter_steps"],
                    "code_distance": spec["code_distance"],
                    "t_count": int(t_count),
                    "max_rd": int(max_rd),
                    "boundary_slope": float(slope),
                }
            )
        ax.plot(
            t_counts,
            slopes,
            marker="o",
            linewidth=1.4,
            markersize=5,
            color=color,
            label=f"{spec['use_case_label']} (k = {spec['trotter_steps']}, d = {spec['code_distance']})",
        )

    ax.set_xticks(list(t_counts))
    ax.set_xlabel("T/cycle")
    ax.set_ylabel(r"No-stalling slope $d / \max(\mathrm{rd})$")
    ax.set_title(f"TMM no-stalling boundary summary ({architecture}, error rate = {error_rate})", pad=3)
    ax.grid(axis="both", which="major", color="0.88", linewidth=0.55)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(pad=0.2)
    output_name = f"tmm_no_stalling_summary_{architecture}_error_rate_{str(error_rate).replace('.', 'p')}.pdf"
    save_for_paper(fig, output_name)
    return fig, summary_rows


def render_default_fermi_av_plot():
    results_av, code_distance_av, data_file = load_av_results(rldv.CIRCUIT)
    results_plot = select_runs(results_av, DEFAULT_FERMI_AV_CAPACITIES)
    tau_r = np.logspace(0, 2, 81)
    tau_c = np.logspace(0, 2, 81)
    fig = plot_reaction_limited(results_plot, tau_r=tau_r, tau_c=tau_c, d=code_distance_av)
    save_for_paper(fig, f"reaction_limited_different_volume_{rldv.CIRCUIT}.pdf")
    return fig, data_file


def render_default_tav_fermi_plot():
    return render_tav_log_plot(
        "transversal-logical-blocks-fermi-hubbard.json",
        "T-AV Fermi-Hubbard",
        code_distance_override=19,
    )


def render_default_tav_tmm_plot():
    return render_tav_log_plot(
        "transversal-logical-blocks-tmm.json",
        "T-AV TMM",
        code_distance_override=13,
    )


def render_default_av_tmm_plot():
    return render_av_log_plot("tmm_paulis_commuted", "AV TMM", capacities=DEFAULT_TMM_AV_CAPACITIES)


def render_default_av_side_by_side_plot():
    return render_av_side_by_side_plot(
        [
            {
                "circuit": "fermi_hubbard_2d_step_s4_universal_paulis_commuted",
                "title": "AV Fermi-Hubbard",
                "capacities": DEFAULT_FERMI_AV_CAPACITIES,
            },
            {
                "circuit": "tmm_paulis_commuted",
                "title": "AV TMM",
                "capacities": DEFAULT_TMM_AV_CAPACITIES,
            },
        ]
    )


def render_default_tav_side_by_side_plot():
    return render_tav_side_by_side_plot(
        [
            {
                "sequence_file": "transversal-logical-blocks-fermi-hubbard.json",
                "title": "T-AV Fermi-Hubbard",
                "code_distance": 19,
            },
            {
                "sequence_file": "transversal-logical-blocks-tmm.json",
                "title": "T-AV TMM",
                "code_distance": 13,
            },
        ]
    )


def main():
    render_default_fermi_av_plot()
    render_default_tav_fermi_plot()
    render_default_tav_tmm_plot()
    render_default_av_tmm_plot()
    render_default_av_side_by_side_plot()
    render_default_tav_side_by_side_plot()
    render_av_tmm_use_cases_plot(DEFAULT_TMM_AV_CAPACITIES, error_rate="0.001")
    render_tav_tmm_use_cases_plot(DEFAULT_TMM_T_COUNTS, error_rate="0.001", architecture="t-av")
    render_tav_tmm_use_cases_plot([1, 2, 3], error_rate="0.001", architecture="transversal")
    plot_tmm_use_case_summary(architecture="transversal", error_rate="0.001", t_counts=DEFAULT_TMM_T_COUNTS)
    plot_tmm_no_stalling_summary(architecture="transversal", error_rate="0.001", t_counts=DEFAULT_TMM_T_COUNTS)


if __name__ == "__main__":
    main()
