"""CNOT connectivity graph for the Fermi-Hubbard 4th-order Trotter QASM circuit.

Parses fermi_hubbard_2d_step_s4_universal.qasm, counts CNOTs between each
ordered pair of qubits, builds a weighted (undirected) graph where each edge
weight = number of CNOTs between the two endpoints, and saves a PDF.

Run:
    python circuit_compilation_helpers/connectivity_fermi.py
"""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
QASM_PATH = PROJECT_ROOT / "ppr_circuits" / "fermi_hubbard_2d_step_s4_universal.qasm"
OUTPUT_PDF = PROJECT_ROOT / "paper_plots" / "connectivity_fermi.pdf"
OUTPUT_LATTICE_PDF = PROJECT_ROOT / "paper_plots" / "connectivity_fermi_lattice.pdf"
OUTPUT_LINEAR_PDF = PROJECT_ROOT / "paper_plots" / "connectivity_fermi_linear.pdf"
OUTPUT_HEATMAP_PDF = PROJECT_ROOT / "paper_plots" / "connectivity_fermi_heatmap.pdf"
STYLE_FILE = PROJECT_ROOT / "plotstylefile.mplstyle"

# Row-major Jordan-Wigner mapping for the 10x10 Fermi-Hubbard lattice with two
# spin sectors. Inferred from the heavy-edge structure (q[i]--q[i+1] @ 440
# along consecutive indices within each row of 10, with breaks at the row
# boundary i=10k-1 ↔ i=10k). Sites 0-99 = spin up, 100-199 = spin down.
LATTICE_ROWS = 10
LATTICE_COLS = 10
N_SITES = LATTICE_ROWS * LATTICE_COLS  # 100


def lattice_position(qubit_index: int) -> tuple[float, float]:
    """Map qubit index → (x, y) on a two-layer 10×10 lattice.
    qubits 0..99 (spin up) form the left grid; 100..199 (spin down) form
    the right grid, offset along x. Row 0 is on top."""
    site = qubit_index % N_SITES
    spin = qubit_index // N_SITES  # 0 = up, 1 = down
    row = site // LATTICE_COLS
    col = site % LATTICE_COLS
    x_offset = spin * (LATTICE_COLS + 3.0)  # gap between the two spin sectors
    return (col + x_offset, (LATTICE_ROWS - 1) - row)

CX_RE = re.compile(r"^\s*cx\s+q\[(\d+)\]\s*,\s*q\[(\d+)\]\s*;")


def count_cnot_edges(qasm_path: Path) -> tuple[Counter[tuple[int, int]], int]:
    """Return (edge_counts, n_qubits) where edge_counts maps unordered
    qubit pairs (i, j) with i < j to the number of CNOTs between them."""
    edge_counts: Counter[tuple[int, int]] = Counter()
    n_qubits = 0
    with qasm_path.open() as f:
        for line in f:
            m = CX_RE.match(line)
            if m is None:
                if line.startswith("qreg"):
                    qreg_match = re.search(r"qreg\s+\w+\[(\d+)\]", line)
                    if qreg_match is not None:
                        n_qubits = int(qreg_match.group(1))
                continue
            a, b = int(m.group(1)), int(m.group(2))
            if a == b:
                continue
            key = (a, b) if a < b else (b, a)
            edge_counts[key] += 1
    return edge_counts, n_qubits


def build_graph(edge_counts: Counter[tuple[int, int]], n_qubits: int) -> nx.Graph:
    g = nx.Graph()
    g.add_nodes_from(range(n_qubits))
    for (a, b), w in edge_counts.items():
        g.add_edge(a, b, weight=int(w))
    return g


def print_summary(g: nx.Graph, edge_counts: Counter[tuple[int, int]]) -> None:
    weights = np.array([w for _, w in edge_counts.items()])
    total_cnots = int(weights.sum())
    print(f"Qubits:         {g.number_of_nodes()}")
    print(f"Unique edges:   {g.number_of_edges()}")
    print(f"Total CNOTs:    {total_cnots:,}")
    print(f"Weight min:     {int(weights.min())}")
    print(f"Weight median:  {int(np.median(weights))}")
    print(f"Weight max:     {int(weights.max())}")
    top = sorted(edge_counts.items(), key=lambda kv: -kv[1])[:10]
    print("Top 10 heaviest edges:")
    for (a, b), w in top:
        print(f"  q[{a}] -- q[{b}]  weight = {w}")


def plot_graph(g: nx.Graph, output_path: Path) -> None:
    weights = np.array([g[u][v]["weight"] for u, v in g.edges()], dtype=float)
    w_min, w_max = float(weights.min()), float(weights.max())

    # Edge widths scale on log so very heavy edges don't blow out the figure.
    log_w = np.log1p(weights - w_min)
    log_w_norm = log_w / log_w.max() if log_w.max() > 0 else log_w
    edge_widths = 0.4 + 3.6 * log_w_norm

    # Edge colors by weight (viridis on log scale).
    edge_colors = plt.cm.viridis(log_w_norm)

    # Spring layout seeded for reproducibility.
    pos = nx.spring_layout(g, seed=42, k=1.0 / np.sqrt(g.number_of_nodes()))

    with plt.style.context(str(STYLE_FILE)):
        fig, ax = plt.subplots(figsize=(8.0, 7.5))

        nx.draw_networkx_edges(
            g, pos, ax=ax,
            width=edge_widths,
            edge_color=edge_colors,
            alpha=0.7,
        )
        nx.draw_networkx_nodes(
            g, pos, ax=ax,
            node_size=42,
            node_color="#222222",
            linewidths=0.5,
            edgecolors="white",
        )
        # Label every qubit; small font.
        nx.draw_networkx_labels(
            g, pos, ax=ax,
            font_size=4.5,
            font_color="white",
        )

        sm = plt.cm.ScalarMappable(
            cmap=plt.cm.viridis,
            norm=plt.matplotlib.colors.LogNorm(
                vmin=max(w_min, 1),
                vmax=w_max,
            ),
        )
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, shrink=0.7, pad=0.02)
        cbar.set_label("CNOTs between qubit pair (log scale)", labelpad=4)

        ax.set_title(
            f"CNOT connectivity, Fermi-Hubbard 2D s4 Trotter step\n"
            f"{g.number_of_nodes()} qubits, {g.number_of_edges()} unique edges, "
            f"{int(weights.sum()):,} total CNOTs"
        )
        ax.set_axis_off()
        fig.tight_layout()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)
        print(f"Saved {output_path}")


def plot_graph_lattice(g: nx.Graph, output_path: Path) -> None:
    """Place qubits on a two-layer 10×10 lattice (spin up | spin down) using
    ``lattice_position``. Edges drawn between physical positions; weight
    encoded in line width and viridis color."""
    weights = np.array([g[u][v]["weight"] for u, v in g.edges()], dtype=float)
    w_min, w_max = float(weights.min()), float(weights.max())
    log_w = np.log1p(weights - w_min)
    log_w_norm = log_w / log_w.max() if log_w.max() > 0 else log_w
    edge_widths = 0.4 + 3.6 * log_w_norm
    edge_colors = plt.cm.viridis(log_w_norm)

    pos = {node: lattice_position(node) for node in g.nodes()}

    with plt.style.context(str(STYLE_FILE)):
        fig, ax = plt.subplots(figsize=(11.0, 6.0))

        nx.draw_networkx_edges(
            g, pos, ax=ax,
            width=edge_widths,
            edge_color=edge_colors,
            alpha=0.75,
        )
        nx.draw_networkx_nodes(
            g, pos, ax=ax,
            node_size=140,
            node_color="#222222",
            linewidths=0.5,
            edgecolors="white",
        )
        nx.draw_networkx_labels(
            g, pos, ax=ax,
            font_size=5.0,
            font_color="white",
        )

        # Layer labels
        ax.text(
            (LATTICE_COLS - 1) / 2.0, LATTICE_ROWS + 0.4,
            r"spin $\uparrow$  (q[0..99])",
            ha="center", va="bottom", fontsize=11,
        )
        ax.text(
            (LATTICE_COLS - 1) / 2.0 + (LATTICE_COLS + 3.0),
            LATTICE_ROWS + 0.4,
            r"spin $\downarrow$  (q[100..199])",
            ha="center", va="bottom", fontsize=11,
        )

        sm = plt.cm.ScalarMappable(
            cmap=plt.cm.viridis,
            norm=plt.matplotlib.colors.LogNorm(
                vmin=max(w_min, 1),
                vmax=w_max,
            ),
        )
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, shrink=0.8, pad=0.02)
        cbar.set_label("CNOTs between qubit pair (log scale)", labelpad=4)

        ax.set_title(
            f"CNOT connectivity on 10×10 Fermi-Hubbard lattice (2 spin sectors)\n"
            f"{g.number_of_nodes()} qubits, {g.number_of_edges()} unique edges, "
            f"{int(weights.sum()):,} total CNOTs"
        )
        ax.set_aspect("equal")
        ax.set_xlim(-1, 2 * LATTICE_COLS + 3)
        ax.set_ylim(-1, LATTICE_ROWS + 1)
        ax.set_axis_off()
        fig.tight_layout()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)
        print(f"Saved {output_path}")


def plot_graph_linear(g: nx.Graph, output_path: Path) -> None:
    """Place qubits on a horizontal line in JW index order; draw each edge
    as a semicircular arc above the line. Short arcs = JW-chain neighbors;
    tall arcs = long-range edges (e.g. on-site U n↑n↓ connecting q[i] to
    q[100+i]). Arc thickness and color scale with edge weight."""
    n = g.number_of_nodes()
    weights = np.array([g[u][v]["weight"] for u, v in g.edges()], dtype=float)
    w_min, w_max = float(weights.min()), float(weights.max())
    log_w = np.log1p(weights - w_min)
    log_w_norm = log_w / log_w.max() if log_w.max() > 0 else log_w
    edge_widths = 0.5 + 2.5 * log_w_norm
    edge_colors = plt.cm.viridis(log_w_norm)

    # Sort edges so heavy arcs draw last (on top of light ones)
    edges_sorted = sorted(
        g.edges(),
        key=lambda e: g[e[0]][e[1]]["weight"],
    )

    with plt.style.context(str(STYLE_FILE)):
        fig, ax = plt.subplots(figsize=(13.0, 6.0))

        max_arc_height = 0.0
        for u, v in edges_sorted:
            i, j = (min(u, v), max(u, v))
            w = g[u][v]["weight"]
            log_w_e = np.log1p(w - w_min) / max(log_w.max(), 1e-9)
            cx = (i + j) / 2.0
            r = (j - i) / 2.0
            max_arc_height = max(max_arc_height, r)
            # Parametric semicircle (theta from pi to 0 traces top half)
            theta = np.linspace(np.pi, 0.0, 60)
            x = cx + r * np.cos(theta)
            y = r * np.sin(theta)
            ax.plot(
                x, y,
                color=plt.cm.viridis(log_w_e),
                lw=0.5 + 2.5 * log_w_e,
                alpha=0.6,
                solid_capstyle="round",
            )

        # Spin-sector divider at x = 99.5 (boundary between q[0..99] and q[100..199])
        ax.axvline(N_SITES - 0.5, color="0.65", linewidth=0.7, linestyle="--", zorder=0)
        ax.text(
            (N_SITES - 1) / 2.0, -max_arc_height * 0.10,
            r"spin $\uparrow$  (q[0..99])",
            ha="center", va="top", fontsize=10,
        )
        ax.text(
            N_SITES + (N_SITES - 1) / 2.0, -max_arc_height * 0.10,
            r"spin $\downarrow$  (q[100..199])",
            ha="center", va="top", fontsize=10,
        )

        # Qubit markers along the baseline
        xs = np.arange(n)
        ax.scatter(xs, np.zeros(n), s=6, color="#222222", zorder=10)

        sm = plt.cm.ScalarMappable(
            cmap=plt.cm.viridis,
            norm=plt.matplotlib.colors.LogNorm(
                vmin=max(w_min, 1),
                vmax=w_max,
            ),
        )
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, shrink=0.85, pad=0.02)
        cbar.set_label("CNOTs between qubit pair (log scale)", labelpad=4)

        ax.set_xlim(-2, n + 1)
        ax.set_ylim(-max_arc_height * 0.18, max_arc_height * 1.05)
        ax.set_xlabel("Qubit index (Jordan-Wigner order)", labelpad=4)
        ax.set_yticks([])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.set_title(
            f"CNOT connectivity, Fermi-Hubbard 2D s4 — linear JW layout\n"
            f"{g.number_of_nodes()} qubits, {g.number_of_edges()} edges, "
            f"{int(weights.sum()):,} total CNOTs"
        )
        fig.tight_layout()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)
        print(f"Saved {output_path}")


def plot_graph_heatmap(g: nx.Graph, output_path: Path) -> None:
    """Render the 200×200 adjacency matrix as a log-scale heat map.
    Cell (i, j) is colored by CNOT count between q[i] and q[j]; zeros
    are masked (rendered white). Structural features expected:
      - tridiagonal band along main diagonal  → JW chain q[i]-q[i+1]
      - secondary band at offset ±100 (between spin sectors)  → on-site U
      - sparse scattered cells at larger offsets  → vertical hopping JW strings
    """
    n = g.number_of_nodes()
    A = np.zeros((n, n), dtype=float)
    for u, v, data in g.edges(data=True):
        A[u, v] = data["weight"]
        A[v, u] = data["weight"]
    masked = np.ma.masked_where(A == 0, A)

    with plt.style.context(str(STYLE_FILE)):
        fig, ax = plt.subplots(figsize=(7.0, 6.5))
        cmap = plt.cm.viridis.copy()
        cmap.set_bad("white")
        im = ax.imshow(
            masked,
            cmap=cmap,
            norm=plt.matplotlib.colors.LogNorm(
                vmin=max(A[A > 0].min(), 1),
                vmax=A.max(),
            ),
            origin="upper",
            interpolation="nearest",
        )
        # Spin-sector divider lines
        ax.axhline(N_SITES - 0.5, color="0.65", linewidth=0.7, linestyle="--")
        ax.axvline(N_SITES - 0.5, color="0.65", linewidth=0.7, linestyle="--")
        ax.set_xlabel("q[j]")
        ax.set_ylabel("q[i]")
        ax.set_xticks([0, 50, 100, 150, 199])
        ax.set_yticks([0, 50, 100, 150, 199])
        ax.set_title(
            f"CNOT adjacency matrix, Fermi-Hubbard 2D s4\n"
            f"{n} qubits, {g.number_of_edges()} edges, "
            f"{int(A.sum() / 2):,} total CNOTs"
        )
        cbar = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
        cbar.set_label("CNOTs between q[i] and q[j] (log scale)", labelpad=4)
        fig.tight_layout()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)
        print(f"Saved {output_path}")


def main() -> None:
    print(f"Reading {QASM_PATH}")
    edge_counts, n_qubits = count_cnot_edges(QASM_PATH)
    g = build_graph(edge_counts, n_qubits)
    print_summary(g, edge_counts)
    plot_graph(g, OUTPUT_PDF)
    plot_graph_lattice(g, OUTPUT_LATTICE_PDF)
    plot_graph_linear(g, OUTPUT_LINEAR_PDF)
    plot_graph_heatmap(g, OUTPUT_HEATMAP_PDF)


if __name__ == "__main__":
    main()
