import csv
import math
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DISTANCE_TABLE_CSV = Path(__file__).resolve().parent / "distance_table.csv"
STYLE_FILE = PROJECT_ROOT / "plotstylefile.mplstyle"
OUTPUT_PDF = PROJECT_ROOT / "paper_plots" / "runtime_vs_qubits_tmm_and_fermi_hubbard.pdf"

PHOTONICS_CODE_CYCLE_SECONDS = 1e-6
NEUTRAL_ATOMS_CODE_CYCLE_SECONDS = 1e-2
DISTANCE_ERROR_RATE = "0.0001"

SYSTEM_QUBITS_T_AV_BY_CIRCUIT = {
    "Fermi-Hubbard": 200,
    "TMM": 8,
}
SUPERCONDUCTING_FACTORY_TILES = 11
T_STATE_FACTORY_TILES = 15
BUFFER_BUS_TILES = 15
T_STATE_FACTORY_PERIOD = 8  # code cycles per produced T state
# Cultivation factory footprint = 562 + d_eff^2 physical qubits, where the
# cultivation patch distance saturates at 11 (d_eff = min(code_distance, 11)).
CULTIVATION_QUBIT_BASE = 562
CULTIVATION_MAX_DISTANCE = 11

# LS-dist factory distance per benchmark (App. tab:lsdist-configs):
# configuration A (d_fac = 20) for Stat-QPE and Stat-QPE-gap, configuration B
# (d_fac = 24) for QPE-Abs and Fermi-Hubbard. The factory distance is set by
# the output-error requirement and is decoupled from the data-qubit distance,
# so factory tiles are 2*d_fac^2 physical qubits each and a factory supplies
# 2 T states per d_fac code cycles.
LS_FACTORY_DISTANCE = {
    "single_trotter_step": 24,
    "trotter_full_qpe": 24,
    "stat_qpe": 20,
    "stat_qpe_gap": 20,
}
CULTIVATION_CYCLES_PER_STATE = 14.3 * 5
PLOT_KEN = False
PLOT_CULTIVATION = True
SECONDS_PER_DAY = 24 * 60 * 60

KEN_X = [116387, 117174, 117961, 119535, 123470, 127405, 135275, 154950, 170690,
    228141,
    284805,
    340682]
KEN_Y_DAYS = [9.4, 4.12, 2.95, 2.02, 1.34, 1.13, 0.98, 0.88, 0.885, 0.883, 0.875, 0.87]

CULT17_X = [
    116387,
    117174,
    117961,
    119535,
    123470,
    131340,
    139997,
    155737,
    170690,
    228141,
    284805,
    340682,
]
CULT17_Y_DAYS = [
    9.4,
    4.12,
    2.95,
    2.02,
    0.826388889,
    0.413194444,
    0.275462963,
    0.165277778,
    0.118055556,
    0.058759491,
    0.039172917,
    0.029379861,
]

USE_CASES = [
    {
        "label": "Fermi-Hubbard",
        "circuit": "Fermi-Hubbard",
        "use_case": "single_trotter_step",
        "trotter_steps": 1,
        "ppr_file": PROJECT_ROOT
        / "ppr_circuits"
        / "fermi_hubbard_2d_step_s4_universal_paulis_commuted.csv",
        "av_reaction_npz": PROJECT_ROOT
        / "reaction_time_analysis"
        / "data"
        / "reaction_limited_sweep_fermi_hubbard_2d_step_s4_universal_paulis_commuted.npz",
        "av_bell_npz": PROJECT_ROOT
        / "bell_pair_analysis_helpers"
        / "data"
        / "bell_pairs_sweep_fermi_hubbard_2d_step_s4_universal_paulis_commuted.npz",
        "transversal_npz": PROJECT_ROOT
        / "bell_pair_analysis_helpers"
        / "data"
        / "bell_pairs_sweep_transversal_transversal-logical-blocks-fermi-hubbard.npz",
        "transversal_json": PROJECT_ROOT
        / "logical_network_files"
        / "transversal-logical-blocks-fermi-hubbard.json",
    },
    {
        "label": "QPE-Abs",
        "circuit": "TMM",
        "use_case": "trotter_full_qpe",
        "trotter_steps": 1680,
        "ppr_file": PROJECT_ROOT
        / "ppr_circuits"
        / "trotter_circuit_v2_paulis_commuted.csv",
        "av_reaction_npz": PROJECT_ROOT
        / "reaction_time_analysis"
        / "data"
        / "reaction_limited_sweep_tmm_paulis_commuted.npz",
        "av_bell_npz": PROJECT_ROOT
        / "bell_pair_analysis_helpers"
        / "data"
        / "bell_pairs_sweep_tmm_paulis_commuted.npz",
        "transversal_npz": PROJECT_ROOT
        / "bell_pair_analysis_helpers"
        / "data"
        / "bell_pairs_sweep_transversal_transversal-logical-blocks-tmm.npz",
        "transversal_json": PROJECT_ROOT
        / "logical_network_files"
        / "transversal-logical-blocks-tmm.json",
    },
    {
        "label": "Stat-QPE",
        "circuit": "TMM",
        "use_case": "stat_qpe",
        "trotter_steps": 40,
        "ppr_file": PROJECT_ROOT
        / "ppr_circuits"
        / "trotter_circuit_v2_paulis_commuted.csv",
        "av_reaction_npz": PROJECT_ROOT
        / "reaction_time_analysis"
        / "data"
        / "reaction_limited_sweep_tmm_paulis_commuted.npz",
        "av_bell_npz": PROJECT_ROOT
        / "bell_pair_analysis_helpers"
        / "data"
        / "bell_pairs_sweep_tmm_paulis_commuted.npz",
        "transversal_npz": PROJECT_ROOT
        / "bell_pair_analysis_helpers"
        / "data"
        / "bell_pairs_sweep_transversal_transversal-logical-blocks-tmm.npz",
        "transversal_json": PROJECT_ROOT
        / "logical_network_files"
        / "transversal-logical-blocks-tmm.json",
    },
    {
        "label": "Stat-QPE-gap",
        "circuit": "TMM",
        "use_case": "stat_qpe_gap",
        "trotter_steps": 20,
        "ppr_file": PROJECT_ROOT
        / "ppr_circuits"
        / "trotter_circuit_v2_paulis_commuted.csv",
        "av_reaction_npz": PROJECT_ROOT
        / "reaction_time_analysis"
        / "data"
        / "reaction_limited_sweep_tmm_paulis_commuted.npz",
        "av_bell_npz": PROJECT_ROOT
        / "bell_pair_analysis_helpers"
        / "data"
        / "bell_pairs_sweep_tmm_paulis_commuted.npz",
        "transversal_npz": PROJECT_ROOT
        / "bell_pair_analysis_helpers"
        / "data"
        / "bell_pairs_sweep_transversal_transversal-logical-blocks-tmm.npz",
        "transversal_json": PROJECT_ROOT
        / "logical_network_files"
        / "transversal-logical-blocks-tmm.json",
    },
]


def load_distance_rows():
    with DISTANCE_TABLE_CSV.open() as f:
        return list(csv.DictReader(f))


DISTANCE_ROWS = load_distance_rows()

ARCHITECTURE_STYLES = {
    "Baseline": {"color": "#CC79A7"},
    "Compact": {"color": "#56B4E9"},
    "AV": {"color": "#0072B2"},
    "t-AV (atoms)": {"color": "#D55E00"},
    "t-AV (Fowler)": {"color": "#009E73"},
    "t-AV (LS factory)": {"color": "#88CCEE"},
    "cult-tAV (atoms)": {"color": "#E69F00"},
    "cult-tAV (Fowler)": {"color": "#6A3D9A"},
}

ARCHITECTURE_LEGEND_LABELS = {
    "t-AV (atoms)": "t-AV(trans-dist)",
    "cult-tAV (atoms)": "t-AV (cult)",
    "t-AV (Fowler)": "t-AV (trans-dist)",
    "t-AV (LS factory)": "t-AV (LS-dist)",
}

USE_CASE_MARKERS = {
    "QPE-Abs": "o",
    "Stat-QPE": "s",
    "Stat-QPE-gap": "^",
    "Fermi-Hubbard": "p",
}
# Blend architecture colors toward white: lighter QPE-Abs, darkest Stat-QPE-gap.
USE_CASE_WHITE_BLEND = {
    "QPE-Abs": 0.50,
    "Stat-QPE": 0.25,
    "Stat-QPE-gap": 0.0,
    "Fermi-Hubbard": 0.0,
}
USE_CASE_LEGEND_COLORS = {
    "QPE-Abs": "#C4C4C4",
    "Stat-QPE": "#6E6E6E",
    "Stat-QPE-gap": "#1A1A1A",
    "Fermi-Hubbard": "#1A1A1A",
}

LEGEND_GROUPS = [
    ("Superconducting", ["Compact", "Baseline"]),
    ("Photonics", ["AV"]),
    (
        "Neutral atoms",
        ["t-AV (atoms)", "t-AV (Fowler)", "cult-tAV (atoms)", "cult-tAV (Fowler)"],
    ),
]
def min_distance_from_csv(circuit, use_case, architecture, error_model, error_rate):
    distances = []
    for row in DISTANCE_ROWS:
        if row["circuit"] != circuit:
            continue
        if row["use_case"] != use_case:
            continue
        if row["architecture"] != architecture:
            continue
        if row["error_model"] != error_model:
            continue
        if row["error_rate"] != error_rate:
            continue
        if row["min_distance"]:
            distances.append(int(row["min_distance"]))
    if not distances:
        raise ValueError(
            "No matching distance rows for "
            f"circuit={circuit}, use_case={use_case}, architecture={architecture}, "
            f"error_model={error_model}, error_rate={error_rate}."
        )
    return min(distances)


def load_av_capacities_and_cycles(spec):
    reaction_npz = spec["av_reaction_npz"]
    bell_npz = spec["av_bell_npz"]

    if reaction_npz.exists():
        data = np.load(reaction_npz)
        capacities = data["capacities"].astype(int)
        total_cycles = data["n_cycles"].astype(int)
        return capacities, total_cycles, reaction_npz

    if bell_npz.exists():
        data = np.load(bell_npz)
        capacities = data["capacities"].astype(int)
        total_cycles = np.array(
            [len(data[f"total_bp_{int(capacity)}"]) for capacity in capacities],
            dtype=int,
        )
        return capacities, total_cycles, bell_npz

    raise FileNotFoundError(
        f"Could not find AV sweep data for {spec['label']}. "
        f"Tried {reaction_npz} and {bell_npz}."
    )


def load_transversal_variants(spec):
    data = np.load(spec["transversal_npz"])
    variants = []
    for t_count_per_cycle in data["t_counts"]:
        t_count_per_cycle = int(t_count_per_cycle)
        total_bp = data[f"total_bp_{t_count_per_cycle}"]
        variants.append(
            {
                "t_count_per_cycle": t_count_per_cycle,
                "avg_bell_pairs": float(total_bp.mean()),
                "total_cycles": int(len(total_bp) * spec["trotter_steps"]),
            }
        )
    return variants, spec["transversal_npz"]


def baseline_compact_dimensions(spec):
    with open(spec["ppr_file"]) as f:
        first = f.readline().split(",")
        n_logical = len(first[2:-2])
    with open(spec["ppr_file"]) as f:
        t_count = sum(1 for _ in f)

    baseline_total_cycles = t_count * spec["trotter_steps"]
    baseline_logical_qubits = 4 * n_logical
    compact_logical_qubits = 1.5 * n_logical + 3
    return {
        "baseline_total_cycles": int(baseline_total_cycles),
        "baseline_logical_qubits": int(baseline_logical_qubits),
        "compact_logical_qubits": float(compact_logical_qubits),
    }


def buffer_bus_count(num_factories):
    return math.ceil(3 * num_factories / 8)


def build_av_rows(
    spec,
    include_t_factories=False,
    t_per_factory_per_cycle=2,
    # Concatenated (15-to-1) x (8-to-CCZ) LS factory (Litinski AV): 35 blocks
    # per CCZ + 16.5 blocks CCZ->2T conversion = 51.5 tiles per 2-T-per-cycle
    # factory (25.75 blocks per T state).
    blocks_per_factory=51.5,
):
    capacities, base_cycles, source = load_av_capacities_and_cycles(spec)
    code_distance = min_distance_from_csv(
        spec["circuit"], spec["use_case"], "av", "circuit", DISTANCE_ERROR_RATE
    )
    total_cycles = base_cycles * spec["trotter_steps"]

    if include_t_factories:
        with open(spec["ppr_file"]) as f:
            t_count_per_step = sum(1 for _ in f)
        pprs_per_cycle = t_count_per_step / base_cycles
        # Factory runs at its own distance d_fac: tiles are 2*d_fac^2 physical
        # qubits each, and a factory supplies t_per_factory_per_cycle T states
        # per d_fac code cycles, i.e. t_per_factory_per_cycle * d / d_fac per
        # algorithm logical cycle.
        d_fac = LS_FACTORY_DISTANCE[spec["use_case"]]
        supply_per_cycle = t_per_factory_per_cycle * code_distance / d_fac
        num_factories = np.ceil(pprs_per_cycle / supply_per_cycle).astype(int)
        factory_tiles = blocks_per_factory * num_factories
        factory_physical_qubits = factory_tiles * 2 * (d_fac ** 2)
    else:
        num_factories = np.zeros_like(capacities)
        factory_tiles = np.zeros_like(capacities)
        factory_physical_qubits = np.zeros_like(capacities)

    total_tiles = 2 * capacities + factory_tiles
    total_physical_qubits = (
        2 * (code_distance ** 2) * (2 * capacities) + factory_physical_qubits
    )
    runtime_seconds = (
        code_distance * total_cycles * PHOTONICS_CODE_CYCLE_SECONDS
    )

    rows = []
    for capacity, cycles, tiles, qubits, runtime, fac in zip(
        capacities, total_cycles, total_tiles, total_physical_qubits,
        runtime_seconds, num_factories,
    ):
        rows.append(
            {
                "workspace_capacity": int(capacity),
                "total_tiles": int(tiles),
                "num_t_factories": int(fac),
                "total_cycles": int(cycles),
                "code_distance": int(code_distance),
                "physical_qubits": int(qubits),
                "runtime_seconds": float(runtime),
            }
        )
    return rows, source


def build_superconducting_rows(spec, architecture_label):
    dims = baseline_compact_dimensions(spec)
    if architecture_label == "Baseline":
        code_distance = min_distance_from_csv(
            spec["circuit"], spec["use_case"], "baseline", "circuit", DISTANCE_ERROR_RATE
        )
        base_cycles = dims["baseline_total_cycles"]
        base_logical_qubits = dims["baseline_logical_qubits"]
        runtime_factor = lambda n: max(11 / n, 1)
    elif architecture_label == "Compact":
        code_distance = min_distance_from_csv(
            spec["circuit"], spec["use_case"], "compact", "circuit", DISTANCE_ERROR_RATE
        )
        base_cycles = dims["baseline_total_cycles"]
        base_logical_qubits = dims["compact_logical_qubits"]
        runtime_factor = lambda n: max(11 / n, 9)
    else:
        raise ValueError(f"Unknown superconducting architecture {architecture_label}")

    rows = []
    for num_factories in range(1, 21):
        total_logical_qubits = (
            base_logical_qubits + SUPERCONDUCTING_FACTORY_TILES * num_factories
        )
        total_cycles = int(round(base_cycles * runtime_factor(num_factories)))
        physical_qubits = int(round(total_logical_qubits * 2 * (code_distance ** 2)))
        runtime_seconds = code_distance * total_cycles * PHOTONICS_CODE_CYCLE_SECONDS
        rows.append(
            {
                "num_factories": num_factories,
                "total_logical_qubits": total_logical_qubits,
                "total_cycles": total_cycles,
                "code_distance": int(code_distance),
                "physical_qubits": physical_qubits,
                "runtime_seconds": runtime_seconds,
            }
        )
    return rows


def build_tav_rows(
    spec,
    distance_architecture,
    distance_error_model,
    runtime_label,
    code_cycle_seconds=None,
):
    variants, source = load_transversal_variants(spec)
    code_distance = min_distance_from_csv(
        spec["circuit"],
        spec["use_case"],
        distance_architecture,
        distance_error_model,
        DISTANCE_ERROR_RATE,
    )
    system_qubits = SYSTEM_QUBITS_T_AV_BY_CIRCUIT[spec["circuit"]]
    if code_cycle_seconds is None:
        code_cycle_seconds = NEUTRAL_ATOMS_CODE_CYCLE_SECONDS

    rows = []
    for variant in variants:
        t_count_per_cycle = variant["t_count_per_cycle"]
        avg_bell_pairs = variant["avg_bell_pairs"]
        total_cycles = variant["total_cycles"]
        num_factories = math.ceil(T_STATE_FACTORY_PERIOD * t_count_per_cycle)
        num_buffer_buses = buffer_bus_count(num_factories)
        total_logical_qubits = (
            system_qubits
            + avg_bell_pairs
            + T_STATE_FACTORY_TILES * num_factories
            + BUFFER_BUS_TILES * num_buffer_buses
        )
        physical_qubits = int(round(total_logical_qubits * 2 * (code_distance ** 2)))
        runtime_seconds = total_cycles * code_cycle_seconds
        rows.append(
            {
                "t_count_per_cycle": t_count_per_cycle,
                "avg_bell_pairs": avg_bell_pairs,
                "num_factories": num_factories,
                "num_buffer_buses": num_buffer_buses,
                "total_logical_qubits": total_logical_qubits,
                "total_cycles": total_cycles,
                "code_distance": int(code_distance),
                "physical_qubits": physical_qubits,
                "runtime_seconds": runtime_seconds,
                "runtime_label": runtime_label,
            }
        )
    return rows, source


def build_tav_ls_factory_rows(
    spec,
    distance_architecture="transversal",
    distance_error_model="circuit",
    runtime_label="LS factory",
    code_cycle_seconds=None,
    t_per_factory_per_cycle=2,
    # Concatenated (15-to-1) x (8-to-CCZ) LS factory (Litinski AV): 35 blocks
    # per CCZ + 16.5 blocks CCZ->2T conversion = 51.5 tiles per 2-T-per-cycle
    # factory (25.75 blocks per T state).
    blocks_per_factory=51.5,
):
    """t-AV layout with AV-style lattice-surgery T-factories.

    Data tiles = system + bell-pair workspace (same as t-AV Fowler).
    Factory tiles = blocks_per_factory * ceil(t_count_per_cycle / t_per_factory_per_cycle).
    Pacing is lattice-surgery: 1 logical cycle = d code cycles, so
    runtime = total_cycles * d * code_cycle_seconds.
    """
    variants, source = load_transversal_variants(spec)
    code_distance = min_distance_from_csv(
        spec["circuit"],
        spec["use_case"],
        distance_architecture,
        distance_error_model,
        DISTANCE_ERROR_RATE,
    )
    system_qubits = SYSTEM_QUBITS_T_AV_BY_CIRCUIT[spec["circuit"]]
    if code_cycle_seconds is None:
        code_cycle_seconds = NEUTRAL_ATOMS_CODE_CYCLE_SECONDS

    rows = []
    for variant in variants:
        t_count_per_cycle = variant["t_count_per_cycle"]
        avg_bell_pairs = variant["avg_bell_pairs"]
        total_cycles = variant["total_cycles"]
        # Factory runs at its own distance d_fac (App. tab:lsdist-configs):
        # tiles are 2*d_fac^2 physical qubits each, and a factory supplies
        # t_per_factory_per_cycle T states per d_fac code cycles.
        d_fac = LS_FACTORY_DISTANCE[spec["use_case"]]
        supply_per_cycle = t_per_factory_per_cycle * code_distance / d_fac
        num_factories = math.ceil(t_count_per_cycle / supply_per_cycle)
        factory_tiles = blocks_per_factory * num_factories
        total_logical_qubits = system_qubits + avg_bell_pairs + factory_tiles
        physical_qubits = int(
            round(
                (system_qubits + avg_bell_pairs) * 2 * (code_distance ** 2)
                + factory_tiles * 2 * (d_fac ** 2)
            )
        )
        runtime_seconds = total_cycles * code_distance * code_cycle_seconds
        rows.append(
            {
                "t_count_per_cycle": t_count_per_cycle,
                "avg_bell_pairs": avg_bell_pairs,
                "num_factories": num_factories,
                "factory_tiles": factory_tiles,
                "total_logical_qubits": total_logical_qubits,
                "total_cycles": total_cycles,
                "code_distance": int(code_distance),
                "physical_qubits": physical_qubits,
                "runtime_seconds": runtime_seconds,
                "runtime_label": runtime_label,
            }
        )
    return rows, source


def build_cultivation_rows(
    spec,
    distance_architecture,
    distance_error_model,
    runtime_label,
    code_cycle_seconds=None,
):
    variants, source = load_transversal_variants(spec)
    code_distance = min_distance_from_csv(
        spec["circuit"],
        spec["use_case"],
        distance_architecture,
        distance_error_model,
        DISTANCE_ERROR_RATE,
    )
    system_qubits = SYSTEM_QUBITS_T_AV_BY_CIRCUIT[spec["circuit"]]
    if code_cycle_seconds is None:
        code_cycle_seconds = NEUTRAL_ATOMS_CODE_CYCLE_SECONDS
    cult_factory_qubits = (
        CULTIVATION_QUBIT_BASE + min(code_distance, CULTIVATION_MAX_DISTANCE) ** 2
    )
    variants = sorted(variants, key=lambda item: item["t_count_per_cycle"] if isinstance(item, dict) else item[0])
    threshold_factories = {
        int((variant["t_count_per_cycle"] if isinstance(variant, dict) else variant[0])): math.ceil(
            CULTIVATION_CYCLES_PER_STATE * int((variant["t_count_per_cycle"] if isinstance(variant, dict) else variant[0]))
        )
        for variant in variants
    }
    max_factories = max(threshold_factories.values())

    rows = []
    for num_factories in range(1, max_factories + 1):
        best_row = None
        for variant in variants:
            if isinstance(variant, dict):
                t_count_per_cycle = variant["t_count_per_cycle"]
                avg_bell_pairs = variant["avg_bell_pairs"]
                total_cycles = variant["total_cycles"]
            else:
                t_count_per_cycle, avg_bell_pairs, _, total_cycles = variant
            total_t_gates = total_cycles * t_count_per_cycle
            production_cycles_per_state = CULTIVATION_CYCLES_PER_STATE / num_factories
            magic_limited_cycles = total_t_gates * production_cycles_per_state
            runtime_cycles = max(total_cycles, magic_limited_cycles)
            physical_qubits = int(
                round(
                    system_qubits * 2 * (code_distance ** 2)
                    + avg_bell_pairs * 2 * (code_distance ** 2)
                    + cult_factory_qubits * num_factories
                )
            )
            candidate = {
                "t_count_per_cycle": t_count_per_cycle,
                "avg_bell_pairs": avg_bell_pairs,
                "num_factories": num_factories,
                "production_cycles_per_state": production_cycles_per_state,
                "production_rate": num_factories / CULTIVATION_CYCLES_PER_STATE,
                "total_t_gates": total_t_gates,
                "total_cycles": total_cycles,
                "runtime_cycles": runtime_cycles,
                "code_distance": int(code_distance),
                "physical_qubits": physical_qubits,
                "runtime_seconds": runtime_cycles * code_cycle_seconds,
                "runtime_label": runtime_label,
                "threshold_target": "",
            }
            if best_row is None or (
                candidate["runtime_seconds"],
                candidate["physical_qubits"],
                -candidate["t_count_per_cycle"],
            ) < (
                best_row["runtime_seconds"],
                best_row["physical_qubits"],
                -best_row["t_count_per_cycle"],
            ):
                best_row = candidate
        for target_t_count, target_factories in threshold_factories.items():
            if num_factories == target_factories:
                best_row["threshold_target"] = str(target_t_count)
                break
        rows.append(best_row)
    return rows, source


def print_av_table(rows, source, label, trotter_steps):
    print(f"{label} | AV")
    print(f"Source NPZ: {source}")
    print(f"Trotter steps: {trotter_steps}")
    print()
    print(
        f"{'capacity':>10}  {'tiles':>8}  {'cycles':>12}  "
        f"{'distance':>8}  {'phys qubits':>14}  {'runtime (s)':>12}"
    )
    print("-" * 76)
    for row in rows:
        print(
            f"{row['workspace_capacity']:>10}  "
            f"{row['total_tiles']:>8}  "
            f"{row['total_cycles']:>12,}  "
            f"{row['code_distance']:>8}  "
            f"{row['physical_qubits']:>14,}  "
            f"{row['runtime_seconds']:>12.6f}"
        )
    print()


def print_tav_table(rows, source, label, runtime_label):
    print(f"{label} | t-AV | {runtime_label}")
    print(f"Source NPZ: {source}")
    print(f"Neutral-atoms code cycle: {NEUTRAL_ATOMS_CODE_CYCLE_SECONDS} s")
    print()


def apply_plot_style():
    if STYLE_FILE.exists():
        plt.style.use(str(STYLE_FILE))


def blend_toward_white(color, blend_fraction):
    rgb = np.array(mcolors.to_rgb(color))
    return tuple(rgb + blend_fraction * (1.0 - rgb))


def series_color(architecture_label, use_case_label):
    base_color = ARCHITECTURE_STYLES[architecture_label]["color"]
    blend_fraction = USE_CASE_WHITE_BLEND.get(use_case_label, 0.0)
    return blend_toward_white(base_color, blend_fraction)


def architecture_legend_handle(label):
    style = ARCHITECTURE_STYLES[label]
    return Line2D(
        [0],
        [0],
        color=style["color"],
        marker="o",
        linestyle="-",
        linewidth=0.9,
        markersize=4.5,
        label=ARCHITECTURE_LEGEND_LABELS.get(label, label),
    )


def use_case_legend_handle(label):
    return Line2D(
        [0],
        [0],
        color=USE_CASE_LEGEND_COLORS[label],
        marker=USE_CASE_MARKERS[label],
        linestyle="None",
        markersize=4.5,
        label=label,
    )


def benchmark_legend_handles():
    handles = [use_case_legend_handle(label) for label in USE_CASE_MARKERS]
    if PLOT_KEN:
        handles.append(
            Line2D(
                [0],
                [0],
                color="#8B0000",
                marker="x",
                linestyle="-",
                linewidth=1.2,
                markersize=5.0,
                markeredgewidth=1.2,
                label="Ken",
            )
        )
    if PLOT_CULTIVATION:
        handles.append(
            Line2D(
                [0],
                [0],
                color="black",
                marker="*",
                linestyle="-",
                linewidth=1.2,
                markersize=7.0,
                markeredgewidth=1.2,
                label="cult-tav-ken",
            )
        )
    return handles


def add_legend_panel(ax, title, handles):
    ax.set_axis_off()
    legend = ax.legend(
        handles=handles,
        labels=[handle.get_label() for handle in handles],
        title=title,
        loc="upper left",
        ncol=1,
        frameon=False,
        handlelength=1.8,
        handletextpad=0.5,
        labelspacing=0.35,
        borderaxespad=0.0,
    )
    legend.get_title().set_fontweight("bold")


def apply_axes_style(ax):
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(axis="both", which="major", color="0.88", linewidth=0.7)
    ax.tick_params(
        axis="x",
        which="major",
        direction="in",
        pad=4,
        top=True,
        length=4,
        width=0.7,
    )
    ax.tick_params(
        axis="y",
        which="major",
        direction="in",
        pad=1,
        right=True,
        length=4,
        width=0.7,
    )
    ax.tick_params(
        axis="both",
        which="minor",
        direction="in",
        top=True,
        right=True,
        length=2.5,
        width=0.5,
    )
    for spine in ax.spines.values():
        spine.set_color(plt.rcParams["axes.edgecolor"])
        spine.set_linewidth(plt.rcParams["axes.linewidth"])


def build_plot_series(
    spec,
    av_rows,
    tav_atoms_rows,
    tav_circuit_rows,
    cult_tav_rows,
    cult_tav_fowler_rows,
    tav_ls_factory_rows=None,
):
    series = [
        {
            "benchmark": spec["circuit"],
            "use_case_label": spec["label"],
            "architecture_label": "AV",
            "x": np.array([row["physical_qubits"] for row in av_rows], dtype=float),
            "y": np.array([row["runtime_seconds"] for row in av_rows], dtype=float),
        },
        {
            "benchmark": spec["circuit"],
            "use_case_label": spec["label"],
            "architecture_label": "t-AV (atoms)",
            "x": np.array([row["physical_qubits"] for row in tav_atoms_rows], dtype=float),
            "y": np.array([row["runtime_seconds"] for row in tav_atoms_rows], dtype=float),
        },
        {
            "benchmark": spec["circuit"],
            "use_case_label": spec["label"],
            "architecture_label": "t-AV (Fowler)",
            "x": np.array([row["physical_qubits"] for row in tav_circuit_rows], dtype=float),
            "y": np.array([row["runtime_seconds"] for row in tav_circuit_rows], dtype=float),
        },
        {
            "benchmark": spec["circuit"],
            "use_case_label": spec["label"],
            "architecture_label": "cult-tAV (atoms)",
            "x": np.array([row["physical_qubits"] for row in cult_tav_rows], dtype=float),
            "y": np.array([row["runtime_seconds"] for row in cult_tav_rows], dtype=float),
            "special_x": np.array(
                [row["physical_qubits"] for row in cult_tav_rows if row["threshold_target"]],
                dtype=float,
            ),
            "special_y": np.array(
                [row["runtime_seconds"] for row in cult_tav_rows if row["threshold_target"]],
                dtype=float,
            ),
        },
        {
            "benchmark": spec["circuit"],
            "use_case_label": spec["label"],
            "architecture_label": "cult-tAV (Fowler)",
            "x": np.array([row["physical_qubits"] for row in cult_tav_fowler_rows], dtype=float),
            "y": np.array([row["runtime_seconds"] for row in cult_tav_fowler_rows], dtype=float),
            "special_x": np.array(
                [row["physical_qubits"] for row in cult_tav_fowler_rows if row["threshold_target"]],
                dtype=float,
            ),
            "special_y": np.array(
                [row["runtime_seconds"] for row in cult_tav_fowler_rows if row["threshold_target"]],
                dtype=float,
            ),
        },
    ]
    if tav_ls_factory_rows is not None:
        series.append(
            {
                "benchmark": spec["circuit"],
                "use_case_label": spec["label"],
                "architecture_label": "t-AV (LS factory)",
                "x": np.array(
                    [row["physical_qubits"] for row in tav_ls_factory_rows],
                    dtype=float,
                ),
                "y": np.array(
                    [row["runtime_seconds"] for row in tav_ls_factory_rows],
                    dtype=float,
                ),
            }
        )
    baseline_rows = build_superconducting_rows(spec, "Baseline")
    compact_rows = build_superconducting_rows(spec, "Compact")
    series.extend(
        [
            {
                "benchmark": spec["circuit"],
                "use_case_label": spec["label"],
                "architecture_label": "Baseline",
                "x": np.array([row["physical_qubits"] for row in baseline_rows], dtype=float),
                "y": np.array([row["runtime_seconds"] for row in baseline_rows], dtype=float),
            },
            {
                "benchmark": spec["circuit"],
                "use_case_label": spec["label"],
                "architecture_label": "Compact",
                "x": np.array([row["physical_qubits"] for row in compact_rows], dtype=float),
                "y": np.array([row["runtime_seconds"] for row in compact_rows], dtype=float),
            },
        ]
    )
    return series


def plot_runtime_vs_qubits(series):
    apply_plot_style()

    fig = plt.figure(figsize=(10.0, 5.2))
    gs = fig.add_gridspec(
        2,
        4,
        height_ratios=[1.0, 0.36],
        hspace=0.42,
        wspace=0.18,
        left=0.07,
        right=0.99,
        top=0.96,
        bottom=0.07,
    )
    gs_plots = gs[0, :].subgridspec(1, 2, wspace=0.32)
    plot_axes = [fig.add_subplot(gs_plots[0, col]) for col in range(2)]
    benchmark_order = [("TMM", "TMM"), ("Fermi-Hubbard", "Fermi-Hubbard")]

    for col, (benchmark_key, title) in enumerate(benchmark_order):
        ax = plot_axes[col]
        subset = [item for item in series if item["benchmark"] == benchmark_key]
        for item in subset:
            color = series_color(item["architecture_label"], item["use_case_label"])
            marker = USE_CASE_MARKERS[item["use_case_label"]]
            if "special_x" in item:
                ax.plot(
                    item["x"],
                    item["y"],
                    color=color,
                    marker=marker,
                    linestyle="-",
                    linewidth=0.9,
                    markersize=3.2,
                    markevery=max(1, len(item["x"]) // 12),
                )
                ax.plot(
                    item["special_x"],
                    item["special_y"],
                    color=color,
                    marker=marker,
                    linestyle="None",
                    markersize=3.2,
                )
            else:
                ax.plot(
                    item["x"],
                    item["y"],
                    color=color,
                    marker=marker,
                    linestyle="-",
                    linewidth=0.9,
                    markersize=4.5,
                )
        if PLOT_KEN and benchmark_key == "Fermi-Hubbard":
            ax.plot(
                np.array(KEN_X, dtype=float),
                np.array(KEN_Y_DAYS, dtype=float) * SECONDS_PER_DAY,
                color="#8B0000",
                marker="x",
                linestyle="-",
                linewidth=1.2,
                markersize=5.0,
                markeredgewidth=1.2,
                zorder=6,
            )
        if PLOT_CULTIVATION and benchmark_key == "Fermi-Hubbard":
            ax.plot(
                np.array(CULT17_X, dtype=float),
                np.array(CULT17_Y_DAYS, dtype=float) * SECONDS_PER_DAY,
                color="black",
                marker="*",
                linestyle="-",
                linewidth=1.2,
                markersize=7.0,
                markeredgewidth=1.2,
                zorder=5,
            )

        ax.set_title(title, pad=3)
        ax.set_xlabel("Physical qubits", labelpad=2)
        if col == 0:
            ax.set_ylabel("Runtime (s)", labelpad=2)
        apply_axes_style(ax)

    for col, (title, arch_labels) in enumerate(LEGEND_GROUPS):
        legend_ax = fig.add_subplot(gs[1, col])
        handles = [architecture_legend_handle(label) for label in arch_labels]
        add_legend_panel(legend_ax, title, handles)
    benchmark_ax = fig.add_subplot(gs[1, 3])
    add_legend_panel(benchmark_ax, "Benchmarks", benchmark_legend_handles())
    return fig


def save_figure(fig, output_path):
    output_path.parent.mkdir(exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0.03)
    print(f"Saved {output_path}")


def main():
    plot_series = []
    for spec in USE_CASES:
        av_rows, av_source = build_av_rows(spec)
        print_av_table(av_rows, av_source, spec["label"], spec["trotter_steps"])

        tav_atoms_rows, tav_atoms_source = build_tav_rows(
            spec,
            distance_architecture="t-av",
            distance_error_model="atoms",
            runtime_label="atoms error model",
        )
        print_tav_table(tav_atoms_rows, tav_atoms_source, spec["label"], "atoms error model")

        tav_circuit_rows, tav_circuit_source = build_tav_rows(
            spec,
            distance_architecture="transversal",
            distance_error_model="circuit",
            runtime_label="surface-code Fowler model",
        )
        print_tav_table(
            tav_circuit_rows,
            tav_circuit_source,
            spec["label"],
            "surface-code Fowler model",
        )
        cult_tav_rows, cult_tav_source = build_cultivation_rows(
            spec,
            distance_architecture="t-av",
            distance_error_model="atoms",
            runtime_label="cultivation atoms error model",
        )
        print_tav_table(
            cult_tav_rows,
            cult_tav_source,
            spec["label"],
            "cultivation atoms error model",
        )

        cult_tav_fowler_rows, cult_tav_fowler_source = build_cultivation_rows(
            spec,
            distance_architecture="transversal",
            distance_error_model="circuit",
            runtime_label="cultivation surface-code Fowler model",
        )
        print_tav_table(
            cult_tav_fowler_rows,
            cult_tav_fowler_source,
            spec["label"],
            "cultivation surface-code Fowler model",
        )
        plot_series.extend(
            build_plot_series(
                spec,
                av_rows,
                tav_atoms_rows,
                tav_circuit_rows,
                cult_tav_rows,
                cult_tav_fowler_rows,
            )
        )

    fig = plot_runtime_vs_qubits(plot_series)
    # plt.show()
    save_figure(fig, OUTPUT_PDF)


if __name__ == "__main__":
    main()
