"""
Compute the minimum sufficient rotated-surface-code distance for three
architectures (baseline, compact, AV) at two physical error rates (1e-3,
1e-4), targeting total logical error <= 0.01, for multiple circuits.

For baseline & compact: tile count and cycles are derived from the PPR CSV.
For AV: each workspace-capacity variant from a sweep NPZ is one row;
total_tiles_av = 2 * workspace_capacity (per user specification).
For transversal/t-av: each t_count_per_cycle variant from a transversal sweep
NPZ is one row; workspace capacity is empty, avg_bell_pairs is the per-cycle
mean, total_tiles = system_qubits + mean(bell_tiles), cycles = len(total_bp_{t}).

Outputs
-------
  - Markdown table printed to stdout (one block per circuit)
  - CSV at resource_estimators/distance_table.csv

Error model:
  p_L(p, d) = 0.1 * (100*p) ** ((d+1)/2)
Distance criterion:
  total_tiles * total_cycles * d * p_L(p, d) < 0.01
Only odd distances are considered, as appropriate for rotated surface-code
patches.
"""

import json
import sys
import os
import csv
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_CSV = Path(__file__).resolve().parent / "distance_table.csv"

ERROR_RATES = [1e-3, 1e-4]
TARGET_ERROR_PROB = 0.01

# Baseline / compact constants (mirror baseline_estimates.py)
MAGIC_STATE_TILES = 11
MSD_PRODUCTION_RATE = 11
TOTAL_MSD_TILES_BASELINE = MAGIC_STATE_TILES * MSD_PRODUCTION_RATE  # 121
TOTAL_MSD_TILES_COMPACT = MAGIC_STATE_TILES                         # 11

CLOCK_CYCLES_PER_PPR_BASELINE = 1
CLOCK_CYCLES_PER_PPR_COMPACT = 11

D_MAX = 100

TROTTER_STEPS_FULL_QPE = 1680
TROTTER_STEPS_STAT_QPE = 40
TROTTER_STEPS_STAT_QPE_GAP = 20

TMM_USE_CASES = [
    {
        "use_case": "trotter_full_qpe",
        "label": "Trotter full QPE",
        "trotter_steps": TROTTER_STEPS_FULL_QPE,
    },
    {
        "use_case": "stat_qpe",
        "label": "Statistical QPE",
        "trotter_steps": TROTTER_STEPS_STAT_QPE,
    },
    {
        "use_case": "stat_qpe_gap",
        "label": "Statistical QPE gap",
        "trotter_steps": TROTTER_STEPS_STAT_QPE_GAP,
    },
]

# Circuit specs: each one provides a CSV (for baseline/compact dimensions) and
# an AV sweep NPZ (for cycle counts per workspace capacity). The npz field
# prefix differs across our two pipelines (parallel_ops vs total_bp); both
# names point at arrays whose LENGTH is the cycle count for that capacity.
CIRCUITS = [
    {
        "name": "TMM",
        "ppr_file": PROJECT_ROOT / "ppr_circuits" / "trotter_circuit_v2_paulis_commuted.csv",
        "av_npz": PROJECT_ROOT / "bell_pair_analysis_helpers" / "data"
                  / "bell_pairs_sweep_tmm_paulis_commuted.npz",
        "av_field_prefix": "total_bp",
        "av_capacities_subset": [25, 50, 100, 200, 500, 1000],
        "transversal_npz": PROJECT_ROOT / "bell_pair_analysis_helpers" / "data"
                             / "bell_pairs_sweep_transversal_transversal-logical-blocks-tmm.npz",
        "transversal_json": PROJECT_ROOT / "logical_network_files"
                            / "transversal-logical-blocks-tmm.json",
        "use_cases": TMM_USE_CASES,
    },
    {
        "name": "Fermi-Hubbard",
        "ppr_file": PROJECT_ROOT / "ppr_circuits" / "fermi_hubbard_2d_step_s4_universal_paulis_commuted.csv",
        "av_npz": PROJECT_ROOT / "logical_network_parallelization_plot_helpers" / "data"
                  / "sweep_fermi_hubbard_2d_step_s4_universal_paulis_commuted.npz",
        "av_field_prefix": "parallel_ops",
        "av_capacities_subset": None,  # use all capacities in the npz
        "transversal_npz": PROJECT_ROOT / "bell_pair_analysis_helpers" / "data"
                             / "bell_pairs_sweep_transversal_transversal-logical-blocks-fermi-hubbard.npz",
        "transversal_json": PROJECT_ROOT / "logical_network_files"
                            / "transversal-logical-blocks-fermi-hubbard.json",
        "use_cases": [
            {
                "use_case": "single_trotter_step",
                "label": "Single Trotter step",
                "trotter_steps": 1,
            },
        ],
    },
]


# ---------------------------------------------------------------------------
# Error-budget model
# ---------------------------------------------------------------------------

def logical_error_rate(physical_error_rate, distance, atoms=False):
    if atoms: 
        return 0.03 * (physical_error_rate/0.04) ** ((distance + 1) / 2)
    else:
        return 0.1 * (100 * physical_error_rate) ** ((distance + 1) / 2)


def min_distance(total_tiles, total_cycles, physical_error_rate,
                 target_error=TARGET_ERROR_PROB, d_max=D_MAX, atoms=False):
    """Smallest d such that total_tiles * total_cycles * d * p_L(p, d) < target_error."""
    for d in range(1, d_max + 1, 2):
        budget = total_tiles * total_cycles * d * logical_error_rate(
            physical_error_rate, d, atoms=atoms
        )
        if budget < target_error:
            return d
    return None


# ---------------------------------------------------------------------------
# Per-circuit dimension extraction
# ---------------------------------------------------------------------------

def baseline_compact_dimensions(ppr_file, trotter_steps):
    """Return (tt_baseline, tc_baseline, tt_compact, tc_compact, N, T)."""
    with open(ppr_file) as f:
        first = f.readline().split(",")
        n_logical = len(first[2:-2])
    with open(ppr_file) as f:
        t_count = sum(1 for _ in f)

    num_data_tiles = 2 * n_logical
    num_workspace_tiles = num_data_tiles
    total_tiles_baseline = (
        num_data_tiles + num_workspace_tiles 
    )
    total_cycles_baseline = (
        t_count * CLOCK_CYCLES_PER_PPR_BASELINE * trotter_steps
    )

    num_data_tiles_compact = 1.5 * n_logical + 3
    total_tiles_compact = num_data_tiles_compact 
    total_cycles_compact = (
        t_count * CLOCK_CYCLES_PER_PPR_COMPACT * trotter_steps
    )

    return (
        int(total_tiles_baseline), int(total_cycles_baseline),
        int(total_tiles_compact), int(total_cycles_compact),
        n_logical, t_count,
    )


def transversal_system_qubits(json_path):
    """Number of distinct logical qubits in a transversal logical-blocks JSON."""
    with open(json_path) as f:
        sequences = json.load(f)
    qubits = set()
    for seq in sequences:
        for circle in seq.get("circles", []):
            ports = circle.get("ports", {})
            for key in ("up", "down", "connect"):
                q = ports.get(key)
                if q is not None:
                    qubits.add(q)
    return len(qubits)


def load_transversal_variants(npz_path, system_qubits, trotter_steps):
    """Return list of (t_count_per_cycle, avg_bell_pairs, total_tiles, total_cycles)."""
    data = np.load(npz_path)
    variants = []
    for t in data["t_counts"]:
        t = int(t)
        total_bp = data[f"total_bp_{t}"]
        bell_tiles = data[f"bell_tiles_{t}"]
        avg_bp = float(total_bp.mean())
        avg_bell_tiles = float(bell_tiles.mean())
        total_tiles = system_qubits + avg_bell_tiles
        total_cycles = len(total_bp) * trotter_steps
        variants.append((t, avg_bp, total_tiles, total_cycles))
    return variants


def load_av_variants(npz_path, field_prefix, trotter_steps, subset=None):
    """Return list of (workspace_capacity, total_tiles_av, total_cycles).
    total_tiles_av = 2 * workspace_capacity.
    `subset` filters to those capacities (in order); None = use all.
    """
    data = np.load(npz_path)
    all_caps = data["capacities"].tolist()
    caps_to_use = subset if subset is not None else all_caps
    variants = []
    for cap in caps_to_use:
        cap = int(cap)
        if cap not in [int(c) for c in all_caps]:
            raise ValueError(
                f"Capacity {cap} not in npz {npz_path.name}; available = {all_caps}"
            )
        n_cycles = len(data[f"{field_prefix}_{cap}"]) * trotter_steps
        variants.append((cap, 2 * cap, n_cycles))
    return variants


# ---------------------------------------------------------------------------
# Build the table
# ---------------------------------------------------------------------------

def build_rows():
    rows = []
    for circuit in CIRCUITS:
        name = circuit["name"]
        ppr = circuit["ppr_file"]
        for case in circuit["use_cases"]:
            case_name = case["use_case"]
            case_label = case["label"]
            trotter_steps = case["trotter_steps"]
            (tt_base, tc_base, tt_compact, tc_compact,
             n_logical, t_count) = baseline_compact_dimensions(
                ppr, trotter_steps
            )
            print(f"\n{name} ({case_label}, k={trotter_steps}):")
            print(f"  PPR file:  {ppr.name}")
            print(f"  N_logical = {n_logical}, T_count = {t_count:,}")
            print(f"  baseline: tiles={tt_base}, cycles={tc_base:,}")
            print(f"  compact:  tiles={tt_compact}, cycles={tc_compact:,}")

            av = load_av_variants(
                circuit["av_npz"], circuit["av_field_prefix"], trotter_steps,
                subset=circuit["av_capacities_subset"],
            )
            print(f"  AV npz:   {circuit['av_npz'].name}")
            print(f"  AV variants: capacities = {[c for c, _, _ in av]}")

            n_sys = transversal_system_qubits(circuit["transversal_json"])
            trans = load_transversal_variants(
                circuit["transversal_npz"], n_sys, trotter_steps
            )
            print(f"  Transversal npz: {circuit['transversal_npz'].name}")
            print(f"  system_qubits = {n_sys}, t_counts = {[t for t, _, _, _ in trans]}")

            base = {
                "circuit": name,
                "use_case": case_name,
                "use_case_label": case_label,
                "trotter_steps": trotter_steps,
            }
            empty = {
                "workspace_capacity": "",
                "t_count_per_cycle": "",
                "avg_bell_pairs": "",
            }
            for p in ERROR_RATES:
                rows.append({
                    **base,
                    "architecture": "baseline",
                    "error_model": "circuit",
                    **empty,
                    "total_tiles": tt_base, "total_cycles": tc_base,
                    "error_rate": p,
                    "min_distance": min_distance(tt_base, tc_base, p),
                })
            for p in ERROR_RATES:
                rows.append({
                    **base,
                    "architecture": "compact",
                    "error_model": "circuit",
                    **empty,
                    "total_tiles": tt_compact, "total_cycles": tc_compact,
                    "error_rate": p,
                    "min_distance": min_distance(tt_compact, tc_compact, p),
                })
            for cap, tt_av, tc_av in av:
                for p in ERROR_RATES:
                    rows.append({
                        **base,
                        "architecture": "av",
                        "error_model": "circuit",
                        "workspace_capacity": cap,
                        "t_count_per_cycle": "",
                        "avg_bell_pairs": "",
                        "total_tiles": tt_av, "total_cycles": tc_av,
                        "error_rate": p,
                        "min_distance": min_distance(tt_av, tc_av, p),
                    })
            for t_count, avg_bp, tt_trans, tc_trans in trans:
                for p in ERROR_RATES:
                    rows.append({
                        **base,
                        "architecture": "transversal",
                        "error_model": "circuit",
                        "workspace_capacity": "",
                        "t_count_per_cycle": t_count,
                        "avg_bell_pairs": avg_bp,
                        "total_tiles": tt_trans, "total_cycles": tc_trans,
                        "error_rate": p,
                        "min_distance": min_distance(tt_trans, tc_trans, p),
                    })
                    rows.append({
                        **base,
                        "architecture": "t-av",
                        "error_model": "atoms",
                        "workspace_capacity": "",
                        "t_count_per_cycle": t_count,
                        "avg_bell_pairs": avg_bp,
                        "total_tiles": tt_trans, "total_cycles": tc_trans,
                        "error_rate": p,
                        "min_distance": min_distance(
                            tt_trans, tc_trans, p, atoms=True
                        ),
                    })
    return rows


def print_markdown_table(rows):
    header = ("Circuit", "Use case", "k", "Architecture", "Error model",
              "Workspace cap.", "T/cycle", "Avg bell pairs", "Total tiles",
              "Total cycles", "Phys. error rate", "Min distance")
    print()
    print(f"| {' | '.join(header)} |")
    print(f"|{'|'.join('---' for _ in header)}|")
    for r in rows:
        cap_str = "—" if r["workspace_capacity"] == "" else str(r["workspace_capacity"])
        t_str = "—" if r["t_count_per_cycle"] == "" else str(r["t_count_per_cycle"])
        bp_str = "—" if r["avg_bell_pairs"] == "" else f"{r['avg_bell_pairs']:.2f}"
        d_str = "n/a" if r["min_distance"] is None else str(r["min_distance"])
        tt = r["total_tiles"]
        tt_str = f"{tt:,}" if isinstance(tt, int) else f"{tt:,.1f}"
        print(
            f"| {r['circuit']:<14} "
            f"| {r['use_case_label']:<18} "
            f"| {r['trotter_steps']:>3} "
            f"| {r['architecture']:<12} "
            f"| {r['error_model']:<11} "
            f"| {cap_str:>14} "
            f"| {t_str:>7} "
            f"| {bp_str:>14} "
            f"| {tt_str:>11} "
            f"| {r['total_cycles']:>12,} "
            f"| {r['error_rate']:>16.0e} "
            f"| {d_str:>12} |"
        )


def save_csv(rows, path=OUTPUT_CSV):
    fieldnames = ["circuit", "use_case", "use_case_label", "trotter_steps",
                  "architecture", "error_model", "workspace_capacity",
                  "t_count_per_cycle", "avg_bell_pairs",
                  "total_tiles", "total_cycles", "error_rate", "min_distance"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"\nCSV saved to {path}")


if __name__ == "__main__":
    rows = build_rows()
    print_markdown_table(rows)
    save_csv(rows)
