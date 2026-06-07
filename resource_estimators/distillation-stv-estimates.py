"""
Per-factory space-time volume of the four magic-state distillation schemes,
as a function of code distance d.

For each scheme we report the space-time volume required to produce ONE
distilled T state, in physical qubit * code-cycle units. A logical patch is
2 d^2 physical qubits (rotated surface code, incl. measurement ancillas) and one
logical cycle is d code cycles, so one logical block = 2 d^3.

Scheme models (physical qubit * code-cycle per T state):
  trans-dist  : transversal 15-to-1. Active t-AV = 132.5/d blocks per T
                -> 132.5/d * 2 d^3 = 265 d^2. The buffer bus (15 tiles used for
                3 of the 8 code cycles) adds 15 * 3 * 2 d^2 = 90 d^2, giving
                355 d^2 with the bus included.
  LS-dist     : concatenated (15-to-1) x (8-to-CCZ) lattice-surgery factory
                (Litinski AV): 35 blocks per CCZ + 16.5 blocks for the
                CCZ -> 2T conversion = 25.75 blocks per T state
                -> 25.75 * 2 d^3 = 51.5 d^3.
  parity-dist : parity-ancilla factory, 7 tiles, 1 T per 71 code cycles, no bus
                -> 7 * 2 d^2 * 71 = 994 d^2.
  cultivation : 562 + d_eff^2 physical qubits with d_eff = min(d, 11) (the
                cultivation patch distance saturates at 11), 1 T per
                14.3*5 = 71.5 code cycles -> rises as (562 + d^2) * 71.5 up to
                d = 11, then flat at 683 * 71.5 = 4.88e4.

Run from the project root:
    python resource_estimators/distillation-stv-estimates.py
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TWO_COLUMN_STYLE_FILE = PROJECT_ROOT / "plotstylefile_two_column.mplstyle"

# --- factory constants (mirrors qubits_runtime_estimates.py) ---
TAV_15TO1_BLOCKS = 132.5            # active t-AV of transversal 15-to-1, per T
TRANS_BUFFER_TILES = 15             # buffer-bus register size (tiles)
TRANS_BUFFER_CYCLES = 3             # code cycles a factory holds the bus per T

# Concatenated (15-to-1) x (8-to-CCZ) lattice-surgery factory (Litinski AV):
# 35 active-volume blocks per CCZ state + 16.5 blocks for the CCZ -> 2T
# conversion, i.e. (35 + 16.5)/2 = 25.75 blocks per T state.
LS_BLOCKS_PER_T = (35 + 16.5) / 2

PARITY_TILES = 7
PARITY_CYCLES_PER_T = 71

CULTIVATION_QUBIT_BASE = 562        # footprint = 562 + d_eff^2 physical qubits
CULTIVATION_MAX_DISTANCE = 11       # patch distance saturates: d_eff = min(d, 11)
CULTIVATION_CYCLES_PER_T = 14.3 * 5  # = 71.5 code cycles per state

PATCH_QUBITS = lambda d: 2 * d ** 2  # physical qubits per logical patch

# Representative benchmark code distances (transversal / photonic, both rates).
BENCHMARK_DISTANCES = [7, 9, 11, 13, 15, 17, 19]


# --- space-time volume per T state (physical qubit * code-cycle) ---
def stv_trans_core(d):
    """Transversal 15-to-1, active volume only (132.5/d blocks -> 265 d^2)."""
    return (TAV_15TO1_BLOCKS / d) * 2 * d ** 3


def stv_trans_buffer(d):
    """Transversal 15-to-1 including the buffer bus (265 d^2 + 90 d^2)."""
    buffer = TRANS_BUFFER_TILES * TRANS_BUFFER_CYCLES * PATCH_QUBITS(d)
    return stv_trans_core(d) + buffer


def stv_ls(d):
    """Concatenated 15-to-1 x 8-to-CCZ LS factory: 25.75 blocks/T (51.5 d^3)."""
    return LS_BLOCKS_PER_T * 2 * np.asarray(d, dtype=float) ** 3


def stv_parity(d):
    """Parity-ancilla factory: 7 tiles, 71 code cycles per T (994 d^2)."""
    return PARITY_TILES * PATCH_QUBITS(d) * PARITY_CYCLES_PER_T


def stv_cultivation(d):
    """Cultivation: 562 + d_eff^2 qubits (d_eff = min(d, 11)), 71.5 code cycles/T."""
    d_eff = np.minimum(np.asarray(d, dtype=float), CULTIVATION_MAX_DISTANCE)
    qubits = CULTIVATION_QUBIT_BASE + d_eff ** 2
    return qubits * CULTIVATION_CYCLES_PER_T


CURVES = [
    ("trans-dist", stv_trans_buffer, "-", "#0072B2"),
    ("LS-dist", stv_ls, "-", "#D55E00"),
    ("parity-dist", stv_parity, "-", "#009E73"),
    ("cultivation", stv_cultivation, "-", "#CC79A7"),
]


def crossover(f, g, lo=3.0, hi=60.0):
    """First distance in [lo, hi] where f(d) - g(d) changes sign, else None."""
    ds = np.linspace(lo, hi, 20000)
    diff = f(ds) - g(ds)
    sign_change = np.where(np.diff(np.sign(diff)) != 0)[0]
    return float(ds[sign_change[0]]) if len(sign_change) else None


def print_crossovers():
    print("Crossover distances (space-time volume per T state):")
    pairs = [
        ("trans-dist (core)", stv_trans_core, "LS-dist", stv_ls),
        ("trans-dist (+buffer)", stv_trans_buffer, "LS-dist", stv_ls),
        ("cultivation", stv_cultivation, "trans-dist (+buffer)", stv_trans_buffer),
        ("cultivation", stv_cultivation, "LS-dist", stv_ls),
        ("cultivation", stv_cultivation, "parity-dist", stv_parity),
    ]
    for na, fa, nb, fb in pairs:
        d = crossover(fa, fb)
        loc = f"d = {d:.1f}" if d is not None else "no crossover in range"
        print(f"  {na:<22} vs {nb:<22}: {loc}")


def main():
    print_crossovers()

    if TWO_COLUMN_STYLE_FILE.exists():
        plt.style.use(str(TWO_COLUMN_STYLE_FILE))

    d = np.linspace(5, 25, 400)
    fig, ax = plt.subplots(figsize=(8.0, 5.5))

    for label, fn, ls, color in CURVES:
        ax.plot(d, fn(d), ls, color=color, lw=2, label=label)

    for db in BENCHMARK_DISTANCES:
        ax.axvline(db, color="0.8", lw=0.6, zorder=0)
    ax.text(0.02, 0.04, "vertical lines: benchmark code distances",
            transform=ax.transAxes, fontsize=12, color="0.4")

    ax.set_yscale("log")
    ax.set_xlabel("Code distance $d$")
    ax.set_ylabel("Space-time volume\nper $T$ state")
    ax.set_xlim(5, 25)
    ax.grid(False)  # override the style file's grid for this plot only
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2)
    fig.tight_layout()

    out = PROJECT_ROOT / "paper_plots" / "distillation_stv_vs_distance.pdf"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
