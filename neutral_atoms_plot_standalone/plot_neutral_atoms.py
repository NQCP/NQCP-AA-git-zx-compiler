"""
Standalone reproduction of the neutral-atom runtime-vs-qubit plot for the
TMM Stat-QPE-gap benchmark at physical error rate p = 1e-3.

Three t-AV magic-state-factory variants on a neutral-atom platform
(code-cycle time = 10 ms):
  - t-AV (trans-dist)  : transversal 15-to-1, 8 code cycles per T state
  - t-AV (parity-dist) : parity-ancilla 15-to-1, 7 tiles, 71 code cycles per T
  - t-AV (cult)        : fold-transversal cultivation, 562 + d_eff^2 qubits

Self-contained: depends only on numpy + matplotlib and the two files in data/.
    python plot_neutral_atoms.py
produces  runtime_vs_qubits_tmm_stat_qpe_gap_neutral_atoms_0.001.pdf
"""

import colorsys
import csv
import math
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
STYLE_FILE = HERE / "plotstylefile.mplstyle"
OUTPUT_PDF = HERE / "runtime_vs_qubits_tmm_stat_qpe_gap_neutral_atoms_0.001.pdf"

# ---------------------------------------------------------------- parameters --
# Benchmark: TMM, Statistical-QPE-gap, 20 Trotter steps; neutral-atom platform.
CIRCUIT = "TMM"
USE_CASE = "stat_qpe_gap"
TROTTER_STEPS = 20
ERROR_RATE = "0.001"
TRANSVERSAL_NPZ = DATA / "bell-pairs-sweep-transversal-logical-blocks-tmm.npz"
DISTANCE_TABLE_CSV = DATA / "distance_table.csv"

# Distance lookup convention for the neutral-atom t-AV variants.
DISTANCE_ARCHITECTURE = "t-av"
DISTANCE_ERROR_MODEL = "atoms"

CODE_CYCLE_SECONDS = 1e-2          # neutral-atom code-cycle time (10 ms)
SYSTEM_QUBITS = 8                  # TMM logical (data) qubits

# Transversal 15-to-1 factory.
T_STATE_FACTORY_TILES = 15
BUFFER_BUS_TILES = 15
T_STATE_FACTORY_PERIOD = 8         # code cycles per produced T state

# Parity-ancilla factory.
PARITY_FACTORY_TILES = 7
PARITY_FACTORY_PERIOD = 71         # code cycles per produced T state
PARITY_MAGIC_LIMITED_N = [1, 2, 4, 8, 16, 24, 32, 40]

# Fold-transversal cultivation factory.
CULT_QUBIT_BASE = 562     # footprint = 562 + d_eff^2, d_eff = min(d, 11)
CULT_MAX_DISTANCE = 11
CULT_CYCLES_PER_STATE = 14.3 * 5   # = 71.5 code cycles per state

# --------------------------------------------------------------- data loaders --
def load_transversal_variants():
    """Per-(T-per-cycle) bell-pair statistics from the transversal sweep npz."""
    data = np.load(TRANSVERSAL_NPZ)
    variants = []
    for t_count in data["t_counts"]:
        t_count = int(t_count)
        total_bp = data[f"total_bp_{t_count}"]
        variants.append(
            {
                "t_count_per_cycle": t_count,
                "avg_bell_pairs": float(total_bp.mean()),
                "total_cycles": int(len(total_bp) * TROTTER_STEPS),
            }
        )
    return variants


def min_distance():
    """Minimum sufficient code distance for this benchmark/architecture."""
    distances = []
    with DISTANCE_TABLE_CSV.open() as f:
        for row in csv.DictReader(f):
            if (
                row["circuit"] == CIRCUIT
                and row["use_case"] == USE_CASE
                and row["architecture"] == DISTANCE_ARCHITECTURE
                and row["error_model"] == DISTANCE_ERROR_MODEL
                and row["error_rate"] == ERROR_RATE
                and row["min_distance"]
            ):
                distances.append(int(row["min_distance"]))
    if not distances:
        raise ValueError("No matching distance rows for this benchmark.")
    return min(distances)


# ----------------------------------------------------------- factory models --
def trans_dist_rows(variants, d):
    """Transversal 15-to-1: a magic-limited prefix (T-supply < 1 per cycle, so
    the Bell-pair workspace is unused) followed by the compute-limited points."""
    rows = []
    total_t_gates = variants[0]["total_cycles"] * variants[0]["t_count_per_cycle"]
    # magic-limited prefix: 1..period-1 factories, no bell-pair workspace.
    for num_factories in range(1, T_STATE_FACTORY_PERIOD):
        num_buffer_buses = math.ceil(3 * num_factories / 8)
        tiles = (
            SYSTEM_QUBITS
            + T_STATE_FACTORY_TILES * num_factories
            + BUFFER_BUS_TILES * num_buffer_buses
        )
        cycles = total_t_gates * T_STATE_FACTORY_PERIOD / num_factories
        rows.append(
            {
                "physical_qubits": tiles * 2 * d ** 2,
                "runtime_seconds": cycles * CODE_CYCLE_SECONDS,
            }
        )
    # compute-limited points: one per T-per-cycle setting.
    for v in variants:
        t = v["t_count_per_cycle"]
        num_factories = math.ceil(T_STATE_FACTORY_PERIOD * t)
        num_buffer_buses = math.ceil(3 * num_factories / 8)
        tiles = (
            SYSTEM_QUBITS
            + v["avg_bell_pairs"]
            + T_STATE_FACTORY_TILES * num_factories
            + BUFFER_BUS_TILES * num_buffer_buses
        )
        rows.append(
            {
                "physical_qubits": tiles * 2 * d ** 2,
                "runtime_seconds": v["total_cycles"] * CODE_CYCLE_SECONDS,
            }
        )
    return rows


def parity_dist_rows(variants, d):
    """Parity-ancilla 15-to-1: magic-limited prefix + compute-limited points."""
    rows = []
    total_t_gates = variants[0]["total_cycles"] * variants[0]["t_count_per_cycle"]
    for num_factories in PARITY_MAGIC_LIMITED_N:
        tiles = SYSTEM_QUBITS + PARITY_FACTORY_TILES * num_factories
        cycles = total_t_gates * PARITY_FACTORY_PERIOD / num_factories
        rows.append(
            {
                "physical_qubits": tiles * 2 * d ** 2,
                "runtime_seconds": cycles * CODE_CYCLE_SECONDS,
            }
        )
    for v in variants:
        num_factories = math.ceil(PARITY_FACTORY_PERIOD * v["t_count_per_cycle"])
        tiles = SYSTEM_QUBITS + v["avg_bell_pairs"] + PARITY_FACTORY_TILES * num_factories
        rows.append(
            {
                "physical_qubits": tiles * 2 * d ** 2,
                "runtime_seconds": v["total_cycles"] * CODE_CYCLE_SECONDS,
            }
        )
    return rows


def cultivation_rows(variants, d):
    """Cultivation: sweep factory count, keep the best (runtime, qubits) point."""
    cult_factory_qubits = CULT_QUBIT_BASE + min(d, CULT_MAX_DISTANCE) ** 2
    max_factories = max(
        math.ceil(CULT_CYCLES_PER_STATE * v["t_count_per_cycle"]) for v in variants
    )
    rows = []
    for num_factories in range(1, max_factories + 1):
        best = None
        for v in variants:
            t = v["t_count_per_cycle"]
            total_t_gates = v["total_cycles"] * t
            magic_limited_cycles = total_t_gates * CULT_CYCLES_PER_STATE / num_factories
            runtime_cycles = max(v["total_cycles"], magic_limited_cycles)
            physical_qubits = (
                SYSTEM_QUBITS * 2 * d ** 2
                + v["avg_bell_pairs"] * 2 * d ** 2
                + cult_factory_qubits * num_factories
            )
            cand = {
                "physical_qubits": physical_qubits,
                "runtime_seconds": runtime_cycles * CODE_CYCLE_SECONDS,
                "t_count_per_cycle": t,
                "bp_qubits": v["avg_bell_pairs"] * 2 * d ** 2,
                "magic_limited": magic_limited_cycles > v["total_cycles"],
            }
            key = (cand["runtime_seconds"], cand["physical_qubits"], -t)
            if best is None or key < (
                best["runtime_seconds"], best["physical_qubits"], -best["t_count_per_cycle"]
            ):
                best = cand
        # In the magic-limited regime T-supply < 1 per cycle, so the Bell-pair
        # workspace is unused and its qubits are subtracted.
        if best["magic_limited"]:
            best["physical_qubits"] -= best["bp_qubits"]
        rows.append(best)
    return rows


# ------------------------------------------------------------------- styling --
def with_hls(hex_color, saturation=None, lightness=None):
    r, g, b = mcolors.to_rgb(hex_color)
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    if saturation is not None:
        s = min(1.0, max(0.0, saturation))
    if lightness is not None:
        l = min(1.0, max(0.0, lightness))
    return mcolors.to_hex(colorsys.hls_to_rgb(h, l, s))


# (label, builder, color) in legend/zorder order.
CURVES = [
    ("t-AV (trans-dist)", trans_dist_rows, with_hls("#922B21", 1.0, 0.28)),
    ("t-AV (parity-dist)", parity_dist_rows, with_hls("#922B21", 0.70, 0.50)),
    ("t-AV (cult)", cultivation_rows, with_hls("#E69F00", 1.0, 0.36)),
]


def main():
    if STYLE_FILE.exists():
        plt.style.use(str(STYLE_FILE))

    variants = load_transversal_variants()
    d = min_distance()

    fig, ax = plt.subplots(figsize=(5.2, 3.1))
    for label, builder, color in CURVES:
        rows = sorted(builder(variants, d), key=lambda r: r["physical_qubits"])
        x = [r["physical_qubits"] for r in rows]
        y = [r["runtime_seconds"] for r in rows]
        ax.plot(x, y, "-o", color=color, markersize=4, linewidth=1.6, label=label)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Physical qubits")
    ax.set_ylabel("Runtime (s)")
    ax.legend(title="Neutral atoms", loc="center left", bbox_to_anchor=(1.02, 0.5),
              frameon=False)
    fig.tight_layout()
    fig.savefig(OUTPUT_PDF, bbox_inches="tight")
    print(f"code distance d = {d}")
    print(f"Wrote {OUTPUT_PDF}")


if __name__ == "__main__":
    main()
