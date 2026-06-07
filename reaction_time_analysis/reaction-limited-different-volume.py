"""
Reaction-limited stalling contours for AV compilation across workspace capacities.

Runs the same workspace-capacity sweep used in bell_pairs_count.py and computes
reaction-depth-dependent stalling contours for one logical-network circuit.
"""

import argparse
import gzip
import json
import warnings
from collections import Counter
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
DATA_DIR = HERE / "data"

CIRCUIT = "fermi_hubbard_2d_step_s4_universal_paulis_commuted"
DEFAULT_CAPACITIES_BY_CIRCUIT = {
    "fermi_hubbard_2d_step_s4_universal_paulis_commuted": [
        413,
        500,
        750,
        1000,
        1500,
        2000,
        2500,
        3000,
        3500,
        4000,
        4500,
        5000,
    ],
    "tmm_paulis_commuted": [25, 50, 75, 100, 125, 150],
}
T_STATE_OVERHEAD = 0
CODE_DISTANCE = 21

STYLE_FILE = PROJECT_ROOT / "plotstylefile.mplstyle"
DATA_FILE = DATA_DIR / f"reaction_limited_sweep_{CIRCUIT}.npz"


def input_file_candidates_for_circuit(circuit):
    return [
        PROJECT_ROOT / "logical_network_files" / f"logical_blocks_{circuit}.jsonl.gz",
        PROJECT_ROOT / "logical_network_files" / f"logical_blocks_{circuit}.json",
    ]


def data_file_for_circuit(circuit):
    return DATA_DIR / f"reaction_limited_sweep_{circuit}.npz"


def default_capacities_for_circuit(circuit):
    try:
        return DEFAULT_CAPACITIES_BY_CIRCUIT[circuit]
    except KeyError as exc:
        raise ValueError(f"No default capacities configured for circuit {circuit!r}.") from exc


def resolve_input_file(circuit=CIRCUIT):
    candidates = input_file_candidates_for_circuit(circuit)
    for path in candidates:
        if path.exists():
            return path
    tried = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"Could not find logical-blocks input. Tried: {tried}")


def _extract_internal_bp(obj):
    bell_pairs = set()
    for hexagon in obj.get("hexagons", []):
        for port_val in hexagon.get("ports", {}).values():
            if isinstance(port_val, str) and port_val.startswith("b"):
                bell_pairs.add(port_val)
    return len(bell_pairs)


def _extract_qubits(obj):
    qubits = set()
    for hexagon in obj.get("hexagons", []):
        for port_val in hexagon.get("ports", {}).values():
            if isinstance(port_val, str) and port_val.startswith("q"):
                qubits.add(port_val)
    return tuple(qubits)


def iter_compact_sequences(filename):
    """Yield only the per-sequence fields needed for the reaction-limited sweep."""
    filename = Path(filename)
    if filename.suffix == ".gz":
        open_fn = gzip.open
    else:
        open_fn = open

    with open_fn(filename, "rt") as f:
        if filename.suffix == ".gz":
            iterable = (json.loads(line) for line in f if line.strip())
        else:
            iterable = iter(json.load(f))
        for obj in iterable:
            input_sequence = obj.get("input_sequence", [])
            if isinstance(input_sequence, list):
                input_sequence = "".join(input_sequence)
            yield {
                "cost": obj["active_volume"] + T_STATE_OVERHEAD,
                "input_sequence": input_sequence,
                "internal_bp": _extract_internal_bp(obj),
                "qubits": _extract_qubits(obj),
            }


def pauli_strings_anticommute(p1, p2):
    n = min(len(p1), len(p2))
    diff = 0
    for i in range(n):
        a = p1[i]
        b = p2[i]
        if a != "i" and b != "i" and a != b:
            diff += 1
    return (diff & 1) == 1


def reaction_depth_chain_length(ppr_strings):
    if not ppr_strings:
        return 0
    n = len(ppr_strings)
    dp = [1] * n
    for i in range(1, n):
        for j in range(i):
            if pauli_strings_anticommute(ppr_strings[j], ppr_strings[i]):
                dp[i] = max(dp[i], dp[j] + 1)
    return max(dp)


def _new_run_state(capacity):
    return {
        "capacity": capacity,
        "current_used": 0.0,
        "current_internal_bp": 0,
        "current_qubit_counts": Counter(),
        "current_ppr_strings": [],
        "reaction_depths_list": [],
        "n_cycles": 0,
        "total_bell_pairs_sum": 0,
        "max_used_blocks": 0.0,
    }


def _finalize_cycle(run_state):
    if not run_state["current_ppr_strings"]:
        return
    cross_sequence = sum(
        count - 1 for count in run_state["current_qubit_counts"].values() if count > 1
    )
    total_bell_pairs = run_state["current_internal_bp"] + cross_sequence
    reaction_depth = reaction_depth_chain_length(run_state["current_ppr_strings"])

    run_state["reaction_depths_list"].append(reaction_depth)
    run_state["n_cycles"] += 1
    run_state["total_bell_pairs_sum"] += total_bell_pairs
    run_state["max_used_blocks"] = max(run_state["max_used_blocks"], run_state["current_used"])
    run_state["current_used"] = 0.0
    run_state["current_internal_bp"] = 0
    run_state["current_qubit_counts"].clear()
    run_state["current_ppr_strings"].clear()


def _add_sequence_to_cycle(run_state, sequence):
    run_state["current_used"] += sequence["cost"]
    run_state["current_internal_bp"] += sequence["internal_bp"]
    run_state["current_ppr_strings"].append(sequence["input_sequence"])
    for qubit in sequence["qubits"]:
        run_state["current_qubit_counts"][qubit] += 1


def run_capacity_sweep(input_file, capacities):
    run_states = [_new_run_state(capacity) for capacity in capacities]
    sequence_count = 0
    for sequence_count, sequence in enumerate(iter_compact_sequences(input_file), start=1):
        if sequence_count % 100000 == 0:
            print(f"  processed {sequence_count:,} sequences", flush=True)
        for run_state in run_states:
            if (
                run_state["current_ppr_strings"]
                and run_state["current_used"] + sequence["cost"] > run_state["capacity"]
            ):
                _finalize_cycle(run_state)
            _add_sequence_to_cycle(run_state, sequence)

    for run_state in run_states:
        _finalize_cycle(run_state)

    print(f"Loaded and scheduled {sequence_count:,} sequences from {input_file}")
    return run_states


def save_sweep_data(results, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(exist_ok=True)

    arrays = {
        "capacities": np.array([int(run["capacity"]) for run in results], dtype=np.int32),
        "n_cycles": np.array([int(run["n_cycles"]) for run in results], dtype=np.int32),
        "total_bell_pairs_sum": np.array(
            [int(run["total_bell_pairs_sum"]) for run in results], dtype=np.int64
        ),
        "max_used_blocks": np.array(
            [float(run["max_used_blocks"]) for run in results], dtype=np.float64
        ),
        "code_distance": np.array([CODE_DISTANCE], dtype=np.int32),
    }
    for run in results:
        arrays[f"reaction_depths_{int(run['capacity'])}"] = np.array(
            run["reaction_depths_list"], dtype=np.int16
        )

    np.savez_compressed(output_path, **arrays)
    print(f"Saved {output_path}")


def load_sweep_data(npz_path):
    data = np.load(npz_path)
    capacities = data["capacities"].tolist()
    results = []
    for idx, capacity in enumerate(capacities):
        results.append(
            {
                "capacity": int(capacity),
                "reaction_depths_list": data[f"reaction_depths_{int(capacity)}"].tolist(),
                "n_cycles": int(data["n_cycles"][idx]),
                "total_bell_pairs_sum": int(data["total_bell_pairs_sum"][idx]),
                "max_used_blocks": float(data["max_used_blocks"][idx]),
            }
        )
    code_distance = int(data["code_distance"][0])
    return results, code_distance


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--circuit", default=CIRCUIT)
    parser.add_argument(
        "--capacities",
        nargs="+",
        type=int,
        help="Workspace capacities to sweep. Defaults depend on the circuit.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    circuit = args.circuit
    capacities = args.capacities or default_capacities_for_circuit(circuit)
    input_file = resolve_input_file(circuit)
    output_path = data_file_for_circuit(circuit)
    results = run_capacity_sweep(input_file, [float(capacity) for capacity in capacities])

    for run in results:
        capacity = int(run["capacity"])
        reaction_depths = run["reaction_depths_list"]
        print(f"\n--- capacity = {capacity} logical blocks ---")
        print(f"  Logical cycles: {run['n_cycles']:,}")
        print(f"  Total bell pairs (all cycles): {run['total_bell_pairs_sum']:,}")
        print(
            "  Reaction depth: "
            f"min={min(reaction_depths)}, max={max(reaction_depths)}, avg={np.mean(reaction_depths):.2f}"
        )

    save_sweep_data(results, output_path)

    print(f"\nSaved sweep data only: {output_path}")


if __name__ == "__main__":
    main()
