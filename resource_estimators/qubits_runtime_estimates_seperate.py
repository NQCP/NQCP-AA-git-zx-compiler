"""Separate runtime-vs-qubits figures: TMM (3 panels) and Fermi–Hubbard (1 panel)."""

import colorsys
import math
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.path import Path as MplPath

import qubits_runtime_estimates as qre


PROJECT_ROOT = Path(__file__).resolve().parent.parent
TWO_COLUMN_STYLE_FILE = PROJECT_ROOT / "plotstylefile_two_column.mplstyle"
OUTPUT_TMM_PDF = (
    PROJECT_ROOT / "paper_plots" / "runtime_vs_qubits_tmm_separate_0.0001.pdf"
)
OUTPUT_FERMI_HUBBARD_PDF = (
    PROJECT_ROOT / "paper_plots" / "runtime_vs_qubits_fermi_hubbard_separate_0.0001.pdf"
)
OUTPUT_TMM_OVERLAY_PDF = (
    PROJECT_ROOT
    / "paper_plots"
    / "runtime_vs_qubits_tmm_separate_overlay_0.0001_0.001.pdf"
)
OUTPUT_TMM_TWO_RATE_GRID_PDF = (
    PROJECT_ROOT
    / "paper_plots"
    / "runtime_vs_qubits_tmm_separate_two_rate_grid_0.0001_0.001.pdf"
)
OUTPUT_FERMI_HUBBARD_OVERLAY_PDF = (
    PROJECT_ROOT
    / "paper_plots"
    / "runtime_vs_qubits_fermi_hubbard_separate_overlay_0.0001_0.001.pdf"
)
OUTPUT_FERMI_HUBBARD_TWO_RATE_PDF = (
    PROJECT_ROOT
    / "paper_plots"
    / "runtime_vs_qubits_fermi_hubbard_separate_two_rate_panels_0.0001_0.001.pdf"
)
OUTPUT_TMM_USE_CASE_TWO_RATE_PDFS = {
    "QPE-Abs": PROJECT_ROOT
    / "paper_plots"
    / "runtime_vs_qubits_tmm_qpe_abs_two_rate_panels_0.0001_0.001.pdf",
    "Stat-QPE": PROJECT_ROOT
    / "paper_plots"
    / "runtime_vs_qubits_tmm_stat_qpe_two_rate_panels_0.0001_0.001.pdf",
    "Stat-QPE-gap": PROJECT_ROOT
    / "paper_plots"
    / "runtime_vs_qubits_tmm_stat_qpe_gap_two_rate_panels_0.0001_0.001.pdf",
}

# Blend architecture colors toward white for the looser (0.001) overlay curves.
OVERLAY_LIGHTER_ERROR_RATE_BLEND = 0.58
# Draw 0.001 beneath 0.0001 (larger offset keeps layering unambiguous).
OVERLAY_ZORDER_OFFSET_LIGHTER = -80

TMM_USE_CASE_LABELS = ["QPE-Abs", "Stat-QPE", "Stat-QPE-gap"]
PLOT_MARKER_SIZE = 2.5
FOWLER_MARKER_SIZE = 9.0
DEFAULT_ARCHITECTURE_MARKER = "o"
FOWLER_ARCHITECTURE_LABELS = set()


def make_three_line_marker_path():
    angles = np.deg2rad([90, 210, 330])
    verts = []
    codes = []
    for angle in angles:
        verts.extend([(0.0, 0.0), (np.cos(angle), np.sin(angle))])
        codes.extend([MplPath.MOVETO, MplPath.LINETO])
    path = MplPath(verts, codes)
    vertices = np.array(path.vertices, dtype=float)
    vertices -= vertices.mean(axis=0)
    max_radius = np.max(np.linalg.norm(vertices, axis=1))
    vertices /= max_radius
    return MplPath(vertices, path.codes)


THREE_LINE_MARKER = make_three_line_marker_path()

NEUTRAL_ATOMS_LEGEND_LABELS = [
    "t-AV (atoms)",
    "t-AV (parity atoms)",
    "cult-tAV (atoms)",
]

PHOTONICS_LEGEND_LABELS = [
    "t-AV (Fowler)",
    "t-AV (parity Fowler)",
    "t-AV (LS factory)",
    "AV",
]

FERMI_HUBBARD_SIDE_LEGEND_GROUPS = [
    ("Neutral atoms", ["t-AV (atoms)", "t-AV (parity atoms)", "cult-tAV (atoms)"]),
    ("Superconducting", ["Compact", "Baseline"]),
    ("Photonics", ["t-AV (Fowler)", "t-AV (parity Fowler)", "t-AV (LS factory)", "AV"]),
]

# Internal architecture-key → legend display label.
# Internal keys stay stable (so dictionaries, colors, zorder don't move);
# only the visible legend text changes.
ARCHITECTURE_DISPLAY_LABELS = {
    "t-AV (atoms)": "t-AV (trans-dist)",
    "t-AV (Fowler)": "t-AV (trans-dist)",
    "t-AV (LS factory)": "t-AV (LS-dist)",
    "t-AV (parity atoms)": "t-AV (parity-dist)",
    "t-AV (parity Fowler)": "t-AV (parity-dist)",
    "cult-tAV (atoms)": "t-AV (cult)",
    "cult-tAV (Fowler)": "t-AV (cult)",
}


def display_label(architecture_label):
    return ARCHITECTURE_DISPLAY_LABELS.get(architecture_label, architecture_label)

FERMI_HUBBARD_EXCLUDED_ARCHITECTURES = {"cult-tAV (Fowler)"}

PLATFORM_BAND_COLORS = {
    "Neutral atoms": "#E69F00",
    "Superconducting": "#117733",
    "Photonics": "#0072B2",
}
PLATFORM_BAND_LIGHTEN = 0.65
PLATFORM_BAND_ALPHA = 0.28

ARCHITECTURE_ZORDER = {
    "Baseline": 1,
    "Compact": 2,
    "AV": 3,
    "t-AV (LS factory)": 3.5,
    "t-AV (Fowler)": 4,
    "t-AV (parity Fowler)": 4.5,
    "t-AV (atoms)": 5,
    "t-AV (parity atoms)": 5.5,
    "cult-tAV (Fowler)": 6,
    "cult-tAV (atoms)": 7,
}

T_AV_ATOMS_COLOR = "#922B21"
CULT_T_AV_ATOMS_COLOR = "#E69F00"


def with_hls(hex_color, saturation=None, lightness=None):
    red, green, blue = mcolors.to_rgb(hex_color)
    hue, current_lightness, current_saturation = colorsys.rgb_to_hls(red, green, blue)
    if saturation is not None:
        current_saturation = min(1.0, max(0.0, saturation))
    if lightness is not None:
        current_lightness = min(1.0, max(0.0, lightness))
    red, green, blue = colorsys.hls_to_rgb(
        hue, current_lightness, current_saturation
    )
    return mcolors.to_hex((red, green, blue))


ARCHITECTURE_COLOR_OVERRIDES = {
    "Baseline": "#A8E6B8",
    "Compact": "#117733",
    "t-AV (atoms)": with_hls(T_AV_ATOMS_COLOR, saturation=1.0, lightness=0.28),
    "t-AV (parity atoms)": with_hls(T_AV_ATOMS_COLOR, saturation=0.70, lightness=0.50),
    "t-AV (Fowler)": with_hls("#0072B2", saturation=1.0, lightness=0.22),
    "t-AV (parity Fowler)": with_hls("#0072B2", saturation=0.70, lightness=0.50),
    "t-AV (LS factory)": "#0072B2",
    "AV": with_hls("#0072B2", saturation=0.85, lightness=0.60),
    "cult-tAV (atoms)": with_hls(CULT_T_AV_ATOMS_COLOR, saturation=1.0, lightness=0.36),
    "cult-tAV (Fowler)": with_hls(CULT_T_AV_ATOMS_COLOR, saturation=0.42, lightness=0.56),
}


def apply_plot_style():
    if qre.STYLE_FILE.exists():
        plt.style.use(str(qre.STYLE_FILE))


def apply_two_column_plot_style():
    if TWO_COLUMN_STYLE_FILE.exists():
        plt.style.use(str(TWO_COLUMN_STYLE_FILE))
    else:
        apply_plot_style()


def architecture_color(architecture_label):
    if architecture_label in ARCHITECTURE_COLOR_OVERRIDES:
        return ARCHITECTURE_COLOR_OVERRIDES[architecture_label]
    return qre.ARCHITECTURE_STYLES[architecture_label]["color"]


def blend_color_toward_white(hex_color, frac):
    red, green, blue = mcolors.to_rgb(hex_color)
    return mcolors.to_hex(
        (
            red + frac * (1.0 - red),
            green + frac * (1.0 - green),
            blue + frac * (1.0 - blue),
        )
    )


def architecture_marker(architecture_label):
    if architecture_label in FOWLER_ARCHITECTURE_LABELS:
        return THREE_LINE_MARKER
    return DEFAULT_ARCHITECTURE_MARKER


def plot_markersize(architecture_label):
    if architecture_label in FOWLER_ARCHITECTURE_LABELS:
        return FOWLER_MARKER_SIZE
    return PLOT_MARKER_SIZE


def marker_plot_kwargs(architecture_label):
    if architecture_label in FOWLER_ARCHITECTURE_LABELS:
        return {"fillstyle": "none", "markeredgewidth": 1.4}
    return {}


def architecture_legend_handle(label):
    marker = architecture_marker(label)
    line_color = architecture_color(label)
    markersize = (
        FOWLER_MARKER_SIZE if label in FOWLER_ARCHITECTURE_LABELS else PLOT_MARKER_SIZE
    )
    handle_kwargs = {
        "color": line_color,
        "marker": marker,
        "linestyle": "-",
        "linewidth": 0.9,
        "markersize": markersize,
        "label": display_label(label),
    }
    if label in FOWLER_ARCHITECTURE_LABELS:
        handle_kwargs["fillstyle"] = "none"
        handle_kwargs["markeredgewidth"] = 1.4
    return Line2D([0], [0], **handle_kwargs)


def error_rate_legend_handles():
    return [
        Line2D(
            [0],
            [0],
            color=blend_color_toward_white("#000000", OVERLAY_LIGHTER_ERROR_RATE_BLEND),
            linestyle="-",
            linewidth=0.9,
            label="0.001",
        ),
        Line2D(
            [0],
            [0],
            color="black",
            linestyle="-",
            linewidth=0.9,
            label="0.0001",
        ),
    ]


def add_legend_panel(ax, title, handles, loc="upper left", bbox_to_anchor=None):
    ax.set_axis_off()
    legend_kwargs = dict(
        handles=handles,
        labels=[handle.get_label() for handle in handles],
        title=title,
        loc=loc,
        ncol=1,
        frameon=False,
        handlelength=1.8,
        handletextpad=0.5,
        labelspacing=0.35,
        borderaxespad=0.0,
    )
    if bbox_to_anchor is not None:
        legend_kwargs["bbox_to_anchor"] = bbox_to_anchor
    legend = ax.legend(**legend_kwargs)
    legend.get_title().set_fontweight("bold")


def add_grouped_legends(
    fig, gs, *, legend_layout="default", include_error_rate=False, legend_row=1
):
    if legend_layout == "center":
        # Equal-width legend columns spanning the full plot width.
        ncols = 4 if include_error_rate else 3
        gs_legends = gs[legend_row, :].subgridspec(
            1,
            ncols,
            wspace=0.18,
        )
        legend_cols = list(range(ncols))
        legend_styles = [{"loc": "upper center"}] * len(legend_cols)
        # Left to right: Superconducting, Photonics, Neutral atoms
        group_indices = [0, 1, 2]
        for slot, group_idx in enumerate(group_indices):
            title, arch_labels = qre.LEGEND_GROUPS[group_idx]
            legend_ax = fig.add_subplot(gs_legends[0, legend_cols[slot]])
            if title == "Neutral atoms":
                arch_labels = NEUTRAL_ATOMS_LEGEND_LABELS
            elif title == "Photonics":
                arch_labels = PHOTONICS_LEGEND_LABELS
            handles = [architecture_legend_handle(label) for label in arch_labels]
            add_legend_panel(legend_ax, title, handles, **legend_styles[slot])
        if include_error_rate:
            legend_ax = fig.add_subplot(gs_legends[0, legend_cols[3]])
            add_legend_panel(
                legend_ax,
                "Error rate",
                error_rate_legend_handles(),
                loc="upper center",
            )
    else:
        ncols = 4 if include_error_rate else 3
        gs_legends = gs[legend_row, :].subgridspec(
            1,
            ncols,
            wspace=0.18,
        )
        legend_cols = list(range(ncols))
        legend_styles = [{"loc": "upper center"}] * len(legend_cols)
        for col, (title, arch_labels) in enumerate(qre.LEGEND_GROUPS):
            legend_ax = fig.add_subplot(gs_legends[0, legend_cols[col]])
            if title == "Neutral atoms":
                arch_labels = NEUTRAL_ATOMS_LEGEND_LABELS
            elif title == "Photonics":
                arch_labels = PHOTONICS_LEGEND_LABELS
            handles = [architecture_legend_handle(label) for label in arch_labels]
            add_legend_panel(legend_ax, title, handles, **legend_styles[col])
        if include_error_rate:
            legend_ax = fig.add_subplot(gs_legends[0, legend_cols[3]])
            add_legend_panel(
                legend_ax,
                "Error rate",
                error_rate_legend_handles(),
                loc="upper center",
            )


def tmm_data_extents(series, x_pad=1.38, y_pad_bottom=1.72, y_pad_top=2.0):
    x_values = []
    y_values = []
    for item in series:
        if item["benchmark"] != "TMM":
            continue
        x_values.append(np.asarray(item["x"], dtype=float))
        y_values.append(np.asarray(item["y"], dtype=float))
        if "special_x" in item:
            x_values.append(np.asarray(item["special_x"], dtype=float))
            y_values.append(np.asarray(item["special_y"], dtype=float))
    x = np.concatenate(x_values)
    y = np.concatenate(y_values)
    x = x[x > 0]
    y = y[y > 0]
    return (x.min() / x_pad, x.max() * x_pad), (y.min() / y_pad_bottom, y.max() * y_pad_top)


def plot_series_on_ax(
    ax,
    series,
    use_case_label,
    *,
    color_transform=None,
    zorder_offset=0,
):
    subset = [
        item
        for item in series
        if item["benchmark"] == "TMM" and item["use_case_label"] == use_case_label
    ]
    subset.sort(key=lambda item: ARCHITECTURE_ZORDER[item["architecture_label"]])
    for item in subset:
        base_color = architecture_color(item["architecture_label"])
        color = (
            color_transform(base_color)
            if color_transform is not None
            else base_color
        )
        marker = architecture_marker(item["architecture_label"])
        arch_label = item["architecture_label"]
        zorder = ARCHITECTURE_ZORDER[arch_label] + zorder_offset
        marker_kwargs = marker_plot_kwargs(arch_label)
        if "special_x" in item:
            ax.plot(
                item["x"],
                item["y"],
                color=color,
                marker=marker,
                linestyle="-",
                linewidth=0.9,
                markersize=plot_markersize(arch_label),
                markevery=max(1, len(item["x"]) // 12),
                zorder=zorder,
                **marker_kwargs,
            )
            ax.plot(
                item["special_x"],
                item["special_y"],
                color=color,
                marker=marker,
                linestyle="None",
                markersize=plot_markersize(arch_label),
                zorder=zorder,
                **marker_kwargs,
            )
        else:
            ax.plot(
                item["x"],
                item["y"],
                color=color,
                marker=marker,
                linestyle="-",
                linewidth=0.9,
                markersize=plot_markersize(arch_label),
                zorder=zorder,
                **marker_kwargs,
            )


def plot_fermi_hubbard_on_ax(
    ax,
    series,
    *,
    color_transform=None,
    zorder_offset=0,
    plot_ken_cult=True,
):
    subset = [item for item in series if item["benchmark"] == "Fermi-Hubbard"]
    subset.sort(key=lambda item: ARCHITECTURE_ZORDER[item["architecture_label"]])
    for item in subset:
        base_color = architecture_color(item["architecture_label"])
        color = (
            color_transform(base_color)
            if color_transform is not None
            else base_color
        )
        marker = architecture_marker(item["architecture_label"])
        arch_label = item["architecture_label"]
        zorder = ARCHITECTURE_ZORDER[arch_label] + zorder_offset
        marker_kwargs = marker_plot_kwargs(arch_label)
        if "special_x" in item:
            ax.plot(
                item["x"],
                item["y"],
                color=color,
                marker=marker,
                linestyle="-",
                linewidth=0.9,
                markersize=plot_markersize(arch_label),
                markevery=max(1, len(item["x"]) // 12),
                zorder=zorder,
                **marker_kwargs,
            )
            ax.plot(
                item["special_x"],
                item["special_y"],
                color=color,
                marker=marker,
                linestyle="None",
                markersize=plot_markersize(arch_label),
                zorder=zorder,
                **marker_kwargs,
            )
        else:
            ax.plot(
                item["x"],
                item["y"],
                color=color,
                marker=marker,
                linestyle="-",
                linewidth=0.9,
                markersize=plot_markersize(arch_label),
                zorder=zorder,
                **marker_kwargs,
            )
    if plot_ken_cult and qre.PLOT_KEN:
        ax.plot(
            np.array(qre.KEN_X, dtype=float),
            np.array(qre.KEN_Y_DAYS, dtype=float) * qre.SECONDS_PER_DAY,
            color="#8B0000",
            marker="x",
            linestyle="-",
            linewidth=1.2,
            markersize=PLOT_MARKER_SIZE,
            markeredgewidth=1.2,
            zorder=6 + zorder_offset,
        )
    if plot_ken_cult and qre.PLOT_CULTIVATION:
        ax.plot(
            np.array(qre.CULT17_X, dtype=float),
            np.array(qre.CULT17_Y_DAYS, dtype=float) * qre.SECONDS_PER_DAY,
            color="black",
            marker="*",
            linestyle="-",
            linewidth=1.2,
            markersize=PLOT_MARKER_SIZE,
            markeredgewidth=1.2,
            zorder=5 + zorder_offset,
        )


def plot_tmm_figure(series):
    apply_plot_style()
    fig = plt.figure(figsize=(12.0, 5.2))
    gs = fig.add_gridspec(
        2,
        3,
        height_ratios=[1.0, 0.36],
        hspace=0.42,
        wspace=0.22,
        left=0.06,
        right=0.99,
        top=0.96,
        bottom=0.07,
    )
    gs_plots = gs[0, :].subgridspec(1, 3, wspace=0.28)
    xlim, ylim = tmm_data_extents(series)
    plot_axes = [fig.add_subplot(gs_plots[0, 0])]
    for col in range(1, 3):
        plot_axes.append(
            fig.add_subplot(gs_plots[0, col], sharex=plot_axes[0], sharey=plot_axes[0])
        )
    for col, use_case_label in enumerate(TMM_USE_CASE_LABELS):
        ax = plot_axes[col]
        plot_series_on_ax(ax, series, use_case_label)
        ax.set_title(use_case_label, pad=3)
        ax.set_xlabel("Physical qubits", labelpad=2)
        if col == 0:
            ax.set_ylabel("Runtime (s)", labelpad=2)
        qre.apply_axes_style(ax)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

    add_grouped_legends(fig, gs)
    return fig


def plot_tmm_figure_overlay(series_lighter, series_primary):
    """Overlay two distance-error-rate sweeps: lighter curves first, then primary on top."""
    apply_plot_style()
    combined = list(series_lighter) + list(series_primary)
    fig = plt.figure(figsize=(12.0, 5.2))
    gs = fig.add_gridspec(
        2,
        3,
        height_ratios=[1.0, 0.36],
        hspace=0.42,
        wspace=0.22,
        left=0.06,
        right=0.99,
        top=0.96,
        bottom=0.07,
    )
    gs_plots = gs[0, :].subgridspec(1, 3, wspace=0.28)
    xlim, ylim = tmm_data_extents(combined)
    plot_axes = [fig.add_subplot(gs_plots[0, 0])]
    for col in range(1, 3):
        plot_axes.append(
            fig.add_subplot(gs_plots[0, col], sharex=plot_axes[0], sharey=plot_axes[0])
        )
    lighten = lambda c: blend_color_toward_white(
        c, OVERLAY_LIGHTER_ERROR_RATE_BLEND
    )
    for col, use_case_label in enumerate(TMM_USE_CASE_LABELS):
        ax = plot_axes[col]
        plot_series_on_ax(
            ax,
            series_lighter,
            use_case_label,
            color_transform=lighten,
            zorder_offset=OVERLAY_ZORDER_OFFSET_LIGHTER,
        )
        plot_series_on_ax(ax, series_primary, use_case_label)
        ax.set_title(use_case_label, pad=3)
        ax.set_xlabel("Physical qubits", labelpad=2)
        if col == 0:
            ax.set_ylabel("Runtime (s)", labelpad=2)
        qre.apply_axes_style(ax)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

    add_grouped_legends(fig, gs, include_error_rate=True)
    return fig


def plot_fermi_hubbard_figure(series):
    apply_plot_style()
    fig = plt.figure(figsize=(10.5, 5.2))
    gs = fig.add_gridspec(
        2,
        3,
        height_ratios=[1.0, 0.36],
        hspace=0.42,
        wspace=0.22,
        left=0.06,
        right=0.99,
        top=0.96,
        bottom=0.07,
    )
    gs_plot = gs[0, :].subgridspec(1, 5, width_ratios=[0.38, 1.0, 1.0, 1.0, 0.38])
    ax = fig.add_subplot(gs_plot[0, 1:4])
    plot_fermi_hubbard_on_ax(ax, series)
    ax.set_title("Fermi-Hubbard", pad=3)
    ax.set_xlabel("Physical qubits", labelpad=2)
    ax.set_ylabel("Runtime (s)", labelpad=2)
    qre.apply_axes_style(ax)

    add_grouped_legends(fig, gs, legend_layout="center")
    return fig


def plot_fermi_hubbard_figure_overlay(series_lighter, series_primary):
    apply_plot_style()
    fig = plt.figure(figsize=(10.5, 5.2))
    gs = fig.add_gridspec(
        2,
        3,
        height_ratios=[1.0, 0.36],
        hspace=0.42,
        wspace=0.22,
        left=0.06,
        right=0.99,
        top=0.96,
        bottom=0.07,
    )
    gs_plot = gs[0, :].subgridspec(1, 5, width_ratios=[0.38, 1.0, 1.0, 1.0, 0.38])
    ax = fig.add_subplot(gs_plot[0, 1:4])
    lighten = lambda c: blend_color_toward_white(
        c, OVERLAY_LIGHTER_ERROR_RATE_BLEND
    )
    plot_fermi_hubbard_on_ax(
        ax,
        series_lighter,
        color_transform=lighten,
        zorder_offset=OVERLAY_ZORDER_OFFSET_LIGHTER,
        plot_ken_cult=False,
    )
    plot_fermi_hubbard_on_ax(ax, series_primary, plot_ken_cult=True)
    ax.set_title("Fermi-Hubbard", pad=3)
    ax.set_xlabel("Physical qubits", labelpad=2)
    ax.set_ylabel("Runtime (s)", labelpad=2)
    qre.apply_axes_style(ax)

    add_grouped_legends(fig, gs, legend_layout="center", include_error_rate=True)
    return fig


def plot_tmm_two_rate_grid(series_lighter, series_primary):
    apply_two_column_plot_style()
    combined = list(series_lighter) + list(series_primary)
    fig = plt.figure(figsize=(8.5, 5.99))
    gs = fig.add_gridspec(
        3,
        3,
        height_ratios=[1.0, 1.0, 0.36],
        hspace=0.55,
        wspace=0.22,
        left=0.07,
        right=0.99,
        top=0.95,
        bottom=0.07,
    )
    gs_plots = gs[:2, :].subgridspec(2, 3, wspace=0.24, hspace=0.26)
    xlim, ylim = tmm_data_extents(combined)
    lighten = lambda c: blend_color_toward_white(
        c, OVERLAY_LIGHTER_ERROR_RATE_BLEND
    )

    plot_axes = []
    for row in range(2):
        row_axes = []
        for col in range(3):
            sharex = plot_axes[0][0] if plot_axes else None
            sharey = plot_axes[0][0] if plot_axes else None
            ax = fig.add_subplot(gs_plots[row, col], sharex=sharex, sharey=sharey)
            row_axes.append(ax)
        plot_axes.append(row_axes)

    rate_specs = [
        ("0.001", series_lighter, None),
        ("0.0001", series_primary, None),
    ]
    for row, (rate_label, series, color_transform) in enumerate(rate_specs):
        for col, use_case_label in enumerate(TMM_USE_CASE_LABELS):
            ax = plot_axes[row][col]
            plot_series_on_ax(ax, series, use_case_label, color_transform=color_transform)
            if row == 0:
                ax.set_title(use_case_label, pad=3)
            if row == 1:
                ax.set_xlabel("Physical qubits", labelpad=2)
            if col == 0:
                ax.set_ylabel("Runtime (s)", labelpad=2)
            if col == len(TMM_USE_CASE_LABELS) - 1:
                exp = int(round(np.log10(float(rate_label))))
                ax.yaxis.set_label_position("right")
                ax.set_ylabel(f"$p = 10^{{{exp}}}$", rotation=90, labelpad=8)
            qre.apply_axes_style(ax)
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)

    add_grouped_legends(fig, gs, include_error_rate=False, legend_row=2)
    return fig


def platform_y_extent(series, arch_labels, benchmark="Fermi-Hubbard", use_case_label=None):
    ys = []
    for item in series:
        if item["benchmark"] != benchmark:
            continue
        if use_case_label is not None and item["use_case_label"] != use_case_label:
            continue
        if item["architecture_label"] not in arch_labels:
            continue
        ys.append(np.asarray(item["y"], dtype=float))
        if "special_y" in item:
            ys.append(np.asarray(item["special_y"], dtype=float))
    if not ys:
        return None
    y_all = np.concatenate(ys)
    y_all = y_all[y_all > 0]
    if y_all.size == 0:
        return None
    return float(y_all.min()), float(y_all.max())


def add_platform_bands(ax, series, benchmark="Fermi-Hubbard", use_case_label=None):
    for platform, arch_labels in FERMI_HUBBARD_SIDE_LEGEND_GROUPS:
        extent = platform_y_extent(
            series, arch_labels, benchmark=benchmark, use_case_label=use_case_label
        )
        if extent is None:
            continue
        y_min, y_max = extent
        band_color = blend_color_toward_white(
            PLATFORM_BAND_COLORS[platform], PLATFORM_BAND_LIGHTEN
        )
        ax.axhspan(
            y_min,
            y_max,
            color=band_color,
            alpha=PLATFORM_BAND_ALPHA,
            linewidth=0,
            zorder=0,
        )


def add_platform_side_legends(fig, gs_cell, include_khan=False):
    height_ratios = [6, 3, 2] if include_khan else [5, 3, 2]
    gs_legends = gs_cell.subgridspec(3, 1, height_ratios=height_ratios, hspace=0.0)
    for slot, (title, arch_labels) in enumerate(FERMI_HUBBARD_SIDE_LEGEND_GROUPS):
        legend_ax = fig.add_subplot(gs_legends[slot, 0])
        handles = [architecture_legend_handle(label) for label in arch_labels]
        if include_khan and title == "Neutral atoms":
            handles.append(
                Line2D(
                    [0],
                    [0],
                    color="#8B0000",
                    marker="x",
                    linestyle="-",
                    linewidth=1.2,
                    markersize=PLOT_MARKER_SIZE,
                    markeredgewidth=1.2,
                    label="Khan et al.",
                )
            )
        add_legend_panel(legend_ax, title, handles, loc="upper left")
        legend = legend_ax.get_legend()
        if legend is not None:
            legend._legend_box.align = "left"


def plot_fermi_hubbard_two_rate_panels(series_lighter, series_primary):
    apply_plot_style()
    fig = plt.figure(figsize=(7.6, 4.2))
    gs = fig.add_gridspec(
        1,
        2,
        width_ratios=[2.3, 0.85],
        wspace=0.05,
        left=0.09,
        right=0.99,
        top=0.93,
        bottom=0.12,
    )
    gs_plots = gs[0, 0].subgridspec(1, 2, wspace=0.10)
    ax_left = fig.add_subplot(gs_plots[0, 0])
    ax_right = fig.add_subplot(gs_plots[0, 1], sharex=ax_left, sharey=ax_left)

    series_lighter = [
        s for s in series_lighter
        if s["architecture_label"] not in FERMI_HUBBARD_EXCLUDED_ARCHITECTURES
    ]
    series_primary = [
        s for s in series_primary
        if s["architecture_label"] not in FERMI_HUBBARD_EXCLUDED_ARCHITECTURES
    ]

    add_platform_bands(ax_left, series_lighter)
    add_platform_bands(ax_right, series_primary)

    plot_fermi_hubbard_on_ax(ax_left, series_lighter, plot_ken_cult=False)
    plot_fermi_hubbard_on_ax(ax_right, series_primary, plot_ken_cult=False)
    if qre.PLOT_KEN:
        ax_left.plot(
            np.array(qre.KEN_X, dtype=float),
            np.array(qre.KEN_Y_DAYS, dtype=float) * qre.SECONDS_PER_DAY,
            color="#8B0000",
            marker="x",
            linestyle="-",
            linewidth=1.2,
            markersize=PLOT_MARKER_SIZE,
            markeredgewidth=1.2,
            zorder=10,
            label="Ken",
        )

    ax_left.set_title("$p = 10^{-3}$", pad=3)
    ax_right.set_title("$p = 10^{-4}$", pad=3)
    ax_left.set_xlabel("Physical qubits", labelpad=2)
    ax_right.set_xlabel("Physical qubits", labelpad=2)
    ax_left.set_ylabel("Runtime (s)", labelpad=2)
    qre.apply_axes_style(ax_left)
    qre.apply_axes_style(ax_right)
    plt.setp(ax_right.get_yticklabels(), visible=False)

    add_platform_side_legends(fig, gs[0, 1], include_khan=False)
    return fig


def plot_tmm_use_case_two_rate_panels(series_lighter, series_primary, use_case_label):
    apply_plot_style()
    fig = plt.figure(figsize=(7.6, 4.2))
    gs = fig.add_gridspec(
        1,
        2,
        width_ratios=[2.3, 0.85],
        wspace=0.05,
        left=0.09,
        right=0.99,
        top=0.93,
        bottom=0.12,
    )
    gs_plots = gs[0, 0].subgridspec(1, 2, wspace=0.10)
    ax_left = fig.add_subplot(gs_plots[0, 0])
    ax_right = fig.add_subplot(gs_plots[0, 1], sharex=ax_left, sharey=ax_left)

    series_lighter = [
        s for s in series_lighter
        if s["architecture_label"] not in FERMI_HUBBARD_EXCLUDED_ARCHITECTURES
    ]
    series_primary = [
        s for s in series_primary
        if s["architecture_label"] not in FERMI_HUBBARD_EXCLUDED_ARCHITECTURES
    ]

    add_platform_bands(
        ax_left, series_lighter, benchmark="TMM", use_case_label=use_case_label
    )
    add_platform_bands(
        ax_right, series_primary, benchmark="TMM", use_case_label=use_case_label
    )

    plot_series_on_ax(ax_left, series_lighter, use_case_label)
    plot_series_on_ax(ax_right, series_primary, use_case_label)

    ax_left.set_title("$p = 10^{-3}$", pad=3)
    ax_right.set_title("$p = 10^{-4}$", pad=3)
    ax_left.set_xlabel("Physical qubits", labelpad=2)
    ax_right.set_xlabel("Physical qubits", labelpad=2)
    ax_left.set_ylabel("Runtime (s)", labelpad=2)
    qre.apply_axes_style(ax_left)
    qre.apply_axes_style(ax_right)
    plt.setp(ax_right.get_yticklabels(), visible=False)

    add_platform_side_legends(fig, gs[0, 1], include_khan=False)
    return fig


def save_figure(fig, output_path):
    output_path.parent.mkdir(exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0.03, dpi=200)
    plt.close(fig)
    print(f"Saved {output_path.resolve()}")


def build_tav_rows_with_magic_prefix(
    spec,
    *,
    distance_architecture,
    distance_error_model,
    runtime_label,
    code_cycle_seconds=None,
):
    """qre.build_tav_rows + a magic-limited prefix (0 BP, num_factories=1..period-1).

    In the magic-limited regime T-supply < 1 per cycle, so there is no parallel
    T-injection and the Bell-pair workspace is unused (set to 0).
    """
    rows, source = qre.build_tav_rows(
        spec,
        distance_architecture=distance_architecture,
        distance_error_model=distance_error_model,
        runtime_label=runtime_label,
        code_cycle_seconds=code_cycle_seconds,
    )
    code_distance = rows[0]["code_distance"]
    system_qubits = qre.SYSTEM_QUBITS_T_AV_BY_CIRCUIT[spec["circuit"]]
    cycle_seconds = (
        code_cycle_seconds
        if code_cycle_seconds is not None
        else qre.NEUTRAL_ATOMS_CODE_CYCLE_SECONDS
    )
    variants, _ = qre.load_transversal_variants(spec)
    total_t_gates = variants[0]["total_cycles"] * variants[0]["t_count_per_cycle"]

    prefix = []
    for num_factories in range(1, qre.T_STATE_FACTORY_PERIOD):
        num_buffer_buses = qre.buffer_bus_count(num_factories)
        total_logical_qubits = (
            system_qubits
            + qre.T_STATE_FACTORY_TILES * num_factories
            + qre.BUFFER_BUS_TILES * num_buffer_buses
        )
        magic_limited_cycles = (
            total_t_gates * qre.T_STATE_FACTORY_PERIOD / num_factories
        )
        physical_qubits = int(
            round(total_logical_qubits * 2 * (code_distance ** 2))
        )
        prefix.append(
            {
                "t_count_per_cycle": 1,
                "avg_bell_pairs": 0.0,
                "num_factories": num_factories,
                "num_buffer_buses": num_buffer_buses,
                "total_logical_qubits": total_logical_qubits,
                "total_cycles": int(round(magic_limited_cycles)),
                "code_distance": int(code_distance),
                "physical_qubits": physical_qubits,
                "runtime_seconds": magic_limited_cycles * cycle_seconds,
                "runtime_label": runtime_label,
            }
        )
    return prefix + rows, source


# Cultivation cycles per produced T state, indexed by physical error rate
# (the value of qre.DISTANCE_ERROR_RATE during plot construction).
#
# Each entry = (rounds_per_attempt) * (expected_attempts).
# rounds_per_attempt = 143 ms / 10 ms measurement = 14.3 (Khan Table 6, citing Puri/Sahay et al.).
#
# expected_attempts:
#   p = 1e-3 : Khan §6.2 picks discard rate δ_r = 80% → 5 expected attempts
#              (Puri f=3 / LER≈2e-6 regime). 14.3 × 5 = 71.5.
#   p = 1e-4 : extrapolated analytically via the single-fault-dominant Poisson model
#                  1 - δ_r = exp(-N · p),
#              with N calibrated so δ_r(1e-3) = 0.8 → N ≈ -ln(0.2)/1e-3 ≈ 1609.
#              At p = 1e-4: δ_r ≈ 1 - exp(-0.1609) ≈ 0.149 → ≈1.18 expected attempts.
#              Rounded up to 1.5 for safety margin → 14.3 × 1.5 ≈ 21.45.
#              Neither Puri nor Khan publish p=1e-4 numbers, so this is the best
#              analytical extrapolation available; the resulting cult-tAV curves
#              are still on the conservative side (true Poisson gives ~17 cycles).
CULTIVATION_CYCLES_PER_STATE_BY_RATE = {
    "0.001": 14.3 * 5,
    "0.0001": 14.3 * 1.5,
}


# Trans-parity distillation: small, slow t-AV factory variant.
# 7 logical qubits per factory, 1 magic state per 71 code cycles, no buffer bus.
# Same Bell-pair workspace accounting as the standard t-AV (uses the variant's avg_bp
# in the compute-limited regime, 0 BP in the magic-limited regime).
TAV_PARITY_FACTORY_TILES = 7
TAV_PARITY_FACTORY_PERIOD = 71
# Magic-limited n_fac samples (1..period-1 is too many points; pick a few well-spaced ones).
TAV_PARITY_MAGIC_LIMITED_N = [1, 2, 4, 8, 16, 24, 32, 40]


def build_tav_parity_dist_rows(
    spec,
    *,
    distance_architecture,
    distance_error_model,
    runtime_label,
    code_cycle_seconds=None,
):
    """Trans-parity-distillation t-AV variant.

    Per factory: TAV_PARITY_FACTORY_TILES (=7) logical qubits, 1 T-state per
    TAV_PARITY_FACTORY_PERIOD (=71) code cycles, no buffer bus.

    Rows:
      - magic-limited prefix at TAV_PARITY_MAGIC_LIMITED_N sample points, 0 BP.
      - compute-limited point per variant t/c (matched n_fac = period * t/c),
        using the variant's avg_bell_pairs.
    """
    variants, source = qre.load_transversal_variants(spec)
    code_distance = qre.min_distance_from_csv(
        spec["circuit"],
        spec["use_case"],
        distance_architecture,
        distance_error_model,
        qre.DISTANCE_ERROR_RATE,
    )
    system_qubits = qre.SYSTEM_QUBITS_T_AV_BY_CIRCUIT[spec["circuit"]]
    cycle_seconds = (
        code_cycle_seconds
        if code_cycle_seconds is not None
        else qre.NEUTRAL_ATOMS_CODE_CYCLE_SECONDS
    )
    total_t_gates = variants[0]["total_cycles"] * variants[0]["t_count_per_cycle"]

    rows = []
    for num_factories in TAV_PARITY_MAGIC_LIMITED_N:
        total_logical_qubits = (
            system_qubits + TAV_PARITY_FACTORY_TILES * num_factories
        )
        magic_limited_cycles = (
            total_t_gates * TAV_PARITY_FACTORY_PERIOD / num_factories
        )
        physical_qubits = int(round(total_logical_qubits * 2 * (code_distance ** 2)))
        rows.append(
            {
                "t_count_per_cycle": 1,
                "avg_bell_pairs": 0.0,
                "num_factories": num_factories,
                "total_logical_qubits": total_logical_qubits,
                "total_cycles": int(round(magic_limited_cycles)),
                "code_distance": int(code_distance),
                "physical_qubits": physical_qubits,
                "runtime_seconds": magic_limited_cycles * cycle_seconds,
                "runtime_label": runtime_label,
            }
        )

    for variant in variants:
        t_count_per_cycle = variant["t_count_per_cycle"]
        avg_bell_pairs = variant["avg_bell_pairs"]
        total_cycles = variant["total_cycles"]
        num_factories = math.ceil(TAV_PARITY_FACTORY_PERIOD * t_count_per_cycle)
        total_logical_qubits = (
            system_qubits
            + avg_bell_pairs
            + TAV_PARITY_FACTORY_TILES * num_factories
        )
        physical_qubits = int(round(total_logical_qubits * 2 * (code_distance ** 2)))
        runtime_seconds = total_cycles * cycle_seconds
        rows.append(
            {
                "t_count_per_cycle": t_count_per_cycle,
                "avg_bell_pairs": avg_bell_pairs,
                "num_factories": num_factories,
                "total_logical_qubits": total_logical_qubits,
                "total_cycles": total_cycles,
                "code_distance": int(code_distance),
                "physical_qubits": physical_qubits,
                "runtime_seconds": runtime_seconds,
                "runtime_label": runtime_label,
            }
        )
    return rows, source


def build_cultivation_rows_zero_bp_magic_limited(
    spec,
    *,
    distance_architecture,
    distance_error_model,
    runtime_label,
    code_cycle_seconds=None,
):
    """qre.build_cultivation_rows with avg_bell_pairs zeroed when magic-limited.

    The Bell-pair workspace is unused whenever T-supply < 1 per cycle, so its
    contribution to physical_qubits must be subtracted in that regime.

    Also overrides qre.CULTIVATION_CYCLES_PER_STATE per physical error rate
    using CULTIVATION_CYCLES_PER_STATE_BY_RATE.
    """
    cycles_override = CULTIVATION_CYCLES_PER_STATE_BY_RATE.get(qre.DISTANCE_ERROR_RATE)
    previous_cycles_per_state = qre.CULTIVATION_CYCLES_PER_STATE
    try:
        if cycles_override is not None:
            qre.CULTIVATION_CYCLES_PER_STATE = cycles_override
        rows, source = qre.build_cultivation_rows(
            spec,
            distance_architecture=distance_architecture,
            distance_error_model=distance_error_model,
            runtime_label=runtime_label,
            code_cycle_seconds=code_cycle_seconds,
        )
        for row in rows:
            total_t_gates = row["total_cycles"] * row["t_count_per_cycle"]
            magic_limited_cycles = (
                total_t_gates * qre.CULTIVATION_CYCLES_PER_STATE / row["num_factories"]
            )
            if magic_limited_cycles > row["total_cycles"]:
                d = row["code_distance"]
                row["physical_qubits"] -= int(
                    round(row["avg_bell_pairs"] * 2 * (d ** 2))
                )
                row["avg_bell_pairs"] = 0.0
        return rows, source
    finally:
        qre.CULTIVATION_CYCLES_PER_STATE = previous_cycles_per_state


def build_all_plot_series(distance_error_rate):
    """Build plot series for a logical error rate column; restores qre.DISTANCE_ERROR_RATE."""
    previous = qre.DISTANCE_ERROR_RATE
    try:
        qre.DISTANCE_ERROR_RATE = distance_error_rate
        plot_series = []
        for spec in qre.USE_CASES:
            av_rows, _ = qre.build_av_rows(
                spec, include_t_factories=True
            )
            tav_atoms_rows, _ = build_tav_rows_with_magic_prefix(
                spec,
                distance_architecture="t-av",
                distance_error_model="atoms",
                runtime_label="atoms error model",
            )
            tav_circuit_rows, _ = build_tav_rows_with_magic_prefix(
                spec,
                distance_architecture="transversal",
                distance_error_model="circuit",
                runtime_label="surface-code Fowler model",
                code_cycle_seconds=qre.PHOTONICS_CODE_CYCLE_SECONDS,
            )
            cult_tav_rows, _ = build_cultivation_rows_zero_bp_magic_limited(
                spec,
                distance_architecture="t-av",
                distance_error_model="atoms",
                runtime_label="cultivation atoms error model",
            )
            cult_tav_fowler_rows, _ = build_cultivation_rows_zero_bp_magic_limited(
                spec,
                distance_architecture="transversal",
                distance_error_model="circuit",
                runtime_label="cultivation surface-code Fowler model",
                code_cycle_seconds=qre.PHOTONICS_CODE_CYCLE_SECONDS,
            )
            tav_ls_factory_rows, _ = qre.build_tav_ls_factory_rows(
                spec,
                code_cycle_seconds=qre.PHOTONICS_CODE_CYCLE_SECONDS,
            )
            tav_parity_atoms_rows, _ = build_tav_parity_dist_rows(
                spec,
                distance_architecture="t-av",
                distance_error_model="atoms",
                runtime_label="parity atoms error model",
            )
            tav_parity_fowler_rows, _ = build_tav_parity_dist_rows(
                spec,
                distance_architecture="transversal",
                distance_error_model="circuit",
                runtime_label="parity surface-code Fowler model",
                code_cycle_seconds=qre.PHOTONICS_CODE_CYCLE_SECONDS,
            )
            plot_series.extend(
                qre.build_plot_series(
                    spec,
                    av_rows,
                    tav_atoms_rows,
                    tav_circuit_rows,
                    cult_tav_rows,
                    cult_tav_fowler_rows,
                    tav_ls_factory_rows=tav_ls_factory_rows,
                )
            )
            for label, parity_rows in [
                ("t-AV (parity atoms)", tav_parity_atoms_rows),
                ("t-AV (parity Fowler)", tav_parity_fowler_rows),
            ]:
                plot_series.append(
                    {
                        "benchmark": spec["circuit"],
                        "use_case_label": spec["label"],
                        "architecture_label": label,
                        "x": np.array(
                            [row["physical_qubits"] for row in parity_rows],
                            dtype=float,
                        ),
                        "y": np.array(
                            [row["runtime_seconds"] for row in parity_rows],
                            dtype=float,
                        ),
                    }
                )
        return plot_series
    finally:
        qre.DISTANCE_ERROR_RATE = previous


def main():
    plot_series_0001 = build_all_plot_series("0.0001")
    save_figure(plot_tmm_figure(plot_series_0001), OUTPUT_TMM_PDF)
    save_figure(
        plot_fermi_hubbard_figure(plot_series_0001), OUTPUT_FERMI_HUBBARD_PDF
    )

    plot_series_001 = build_all_plot_series("0.001")
    save_figure(
        plot_tmm_figure_overlay(plot_series_001, plot_series_0001),
        OUTPUT_TMM_OVERLAY_PDF,
    )
    save_figure(
        plot_tmm_two_rate_grid(plot_series_001, plot_series_0001),
        OUTPUT_TMM_TWO_RATE_GRID_PDF,
    )
    save_figure(
        plot_fermi_hubbard_figure_overlay(plot_series_001, plot_series_0001),
        OUTPUT_FERMI_HUBBARD_OVERLAY_PDF,
    )
    save_figure(
        plot_fermi_hubbard_two_rate_panels(plot_series_001, plot_series_0001),
        OUTPUT_FERMI_HUBBARD_TWO_RATE_PDF,
    )
    for use_case_label in TMM_USE_CASE_LABELS:
        save_figure(
            plot_tmm_use_case_two_rate_panels(
                plot_series_001, plot_series_0001, use_case_label
            ),
            OUTPUT_TMM_USE_CASE_TWO_RATE_PDFS[use_case_label],
        )


if __name__ == "__main__":
    main()
