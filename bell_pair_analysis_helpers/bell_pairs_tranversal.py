"""
Bell pair count per code cycle for transversal logical-block circuits.

Sweep `t_count_per_cycle` values and save per-cycle arrays to
data/bell_pairs_sweep_transversal_<circuit>.npz.

Run from the project root:

    python bell_pair_analysis_helpers/bell_pairs_tranversal.py

Parallelization rule:
- Iterate sequences in JSON order and pack them into rows (code cycles).
- Each row holds at most `t_count_per_cycle` T gates. When that limit is hit,
  the row is closed and a new one is started.

Bell pair rule (per row):
- Walk the sequences inside the row serially. For each consecutive pair
  (seq_i, seq_{i+1}), every qubit that appears as an 'up' port of seq_i and
  also as a 'down' port of seq_{i+1} contributes one bell pair.

Tile/data format:
- S, H, Sdg: 1 tile / 1 code cycle
- CNOT: 2 tiles / 1 code cycle
- T: 2 tiles / 1 code cycle (no extra waiting cycle inserted)
- Each bell pair = 2 tiles
"""

import json
from collections import Counter
from pathlib import Path

import numpy as np

# T gates allowed per code cycle (sweep values).
T_COUNTS_PER_CYCLE = [1, 2, 3, 4, 5]
t_count_per_cycle = 4  # default for ad-hoc calls

# CIRCUIT = "transversal-logical-blocks-fermi-hubbard"
CIRCUIT = "transversal-logical-blocks-tmm"

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
INPUT_FILE = PROJECT_ROOT / "logical_network_files" / f"{CIRCUIT}.json"
DATA_DIR = HERE / "data"
OUTPUT_FILE = DATA_DIR / f"bell_pairs_sweep_transversal_{CIRCUIT}.npz"


TILE = {"s": 1, "h": 1, "cx": 2, "t": 2, "sdg": 1}
BELL_PAIR_TILES = 1


def load_sequences(filename=INPUT_FILE):
    with open(filename) as f:
        return json.load(f)


def get_up_qubits(sequence):
    """Qubit indices appearing in the 'up' ports of a sequence."""
    qubits = []
    for circle in sequence.get("circles", []):
        up = circle.get("ports", {}).get("up")
        if up is not None:
            qubits.append(up)
    return qubits


def get_down_qubits(sequence):
    """Qubit indices appearing in the 'down' ports of a sequence."""
    qubits = []
    for circle in sequence.get("circles", []):
        down = circle.get("ports", {}).get("down")
        if down is not None:
            qubits.append(down)
    return qubits


def tiles_for_gate(gate):
    return TILE.get(gate.lower(), 1)


def count_bell_pairs_serial(row_up_qubits, row_down_qubits):
    """
    Bell pairs in a row: for each consecutive pair (i, i+1), every qubit that is
    in seq_i's up ports and seq_{i+1}'s down ports contributes one bell pair.
    """
    bell_pairs = 0
    for i in range(len(row_up_qubits) - 1):
        ups = set(row_up_qubits[i])
        downs = set(row_down_qubits[i + 1])
        bell_pairs += len(ups & downs)
    return bell_pairs


def reaction_depth(row_up_qubits, row_gates):
    """Longest sequential chain of T gates on any single qubit within the row."""
    counts = Counter()
    for qubits, gate in zip(row_up_qubits, row_gates):
        if gate == "t":
            for q in qubits:
                counts[q] += 1
    return max(counts.values()) if counts else 0


def schedule_and_count_bell_pairs(sequences, t_per_cycle=t_count_per_cycle):
    """
    Pack sequences into rows constrained by `t_per_cycle` T gates per row.

    Returns:
        List of (cycle, bell_pairs, num_sequences, comp_tiles, bell_tiles,
                 reaction_depth, seq_indices)
    """
    result = []
    current_cycle = 0

    pack_indices = []
    pack_up_qubits = []
    pack_down_qubits = []
    pack_gates = []
    comp_tiles = 0
    t_in_row = 0

    def flush_row():
        nonlocal pack_indices, pack_up_qubits, pack_down_qubits, pack_gates
        nonlocal comp_tiles, t_in_row, current_cycle
        if not pack_indices:
            return
        bell_pairs = count_bell_pairs_serial(pack_up_qubits, pack_down_qubits)
        bell_tiles = BELL_PAIR_TILES * bell_pairs
        rd = reaction_depth(pack_up_qubits, pack_gates)
        result.append(
            (
                current_cycle,
                bell_pairs,
                len(pack_indices),
                comp_tiles,
                bell_tiles,
                rd,
                pack_indices,
            )
        )
        current_cycle += 1
        pack_indices = []
        pack_up_qubits = []
        pack_down_qubits = []
        pack_gates = []
        comp_tiles = 0
        t_in_row = 0

    for seq_idx, seq in enumerate(sequences):
        gate = seq.get("gate", "").lower()
        ups = get_up_qubits(seq)
        downs = get_down_qubits(seq)

        pack_indices.append(seq_idx)
        pack_up_qubits.append(ups)
        pack_down_qubits.append(downs)
        pack_gates.append(gate)
        comp_tiles += tiles_for_gate(gate)
        if gate == "t":
            t_in_row += 1
            if t_in_row >= t_per_cycle:
                flush_row()

    flush_row()
    return result


def _results_to_arrays(results):
    """Extract per-cycle numpy arrays from schedule_and_count_bell_pairs output."""
    return {
        "total_bp": np.array([r[1] for r in results], dtype=np.int32),
        "parallel_ops": np.array([r[2] for r in results], dtype=np.int32),
        "comp_tiles": np.array([r[3] for r in results], dtype=np.int32),
        "bell_tiles": np.array([r[4] for r in results], dtype=np.int32),
        "reaction_depth": np.array([r[5] for r in results], dtype=np.int32),
    }


def sweep_t_counts(sequences, t_counts):
    """Run scheduler at each t_count_per_cycle; return per-setting dicts."""
    runs = []
    for t in t_counts:
        print(f"\n--- t_count_per_cycle = {t} ---")
        results = schedule_and_count_bell_pairs(sequences, t_per_cycle=t)
        arrays = _results_to_arrays(results)
        runs.append({"t_count": t, **arrays})
        print(
            f"  {len(results):,} cycles, "
            f"mean bp/cycle = {arrays['total_bp'].mean():.2f}"
        )
    return runs


def save_sweep_data(runs, t_counts, output_file=OUTPUT_FILE):
    """Save per-cycle arrays for each t_count_per_cycle to a compressed .npz."""
    output_file.parent.mkdir(exist_ok=True)
    arrays = {"t_counts": np.array(t_counts, dtype=np.int64)}
    for run in runs:
        t = int(run["t_count"])
        arrays[f"total_bp_{t}"] = run["total_bp"]
        arrays[f"parallel_ops_{t}"] = run["parallel_ops"]
        arrays[f"comp_tiles_{t}"] = run["comp_tiles"]
        arrays[f"bell_tiles_{t}"] = run["bell_tiles"]
        arrays[f"reaction_depth_{t}"] = run["reaction_depth"]
    np.savez_compressed(output_file, **arrays)
    print(f"\nSaved sweep data to {output_file}")
    for run in runs:
        t = int(run["t_count"])
        n = len(run["total_bp"])
        bp_kb = arrays[f"total_bp_{t}"].nbytes / 1024
        print(f"  t_count={t}: {n:,} cycles  ({bp_kb:.1f} KB total_bp)")
    print(f"  Compressed file size: {output_file.stat().st_size / (1024 ** 2):.2f} MB")


def print_summary(runs):
    print("\n" + "=" * 78)
    print("SUMMARY: bell pairs per cycle vs t_count_per_cycle")
    print("=" * 78)
    print(
        f"{'t_count':>8}  {'cycles':>12}  {'mean bp':>8}  "
        f"{'p95 bp':>7}  {'max bp':>6}  {'total bp':>12}"
    )
    for r in runs:
        t = r["t_count"]
        bp = r["total_bp"]
        print(
            f"{t:>8}  {len(bp):>12,}  {bp.mean():>8.2f}  "
            f"{np.percentile(bp, 95):>7.1f}  {bp.max():>6.0f}  {int(bp.sum()):>12,}"
        )
    print("=" * 78)


if __name__ == "__main__":
    print(f"Loading sequences from {INPUT_FILE}...")
    sequences = load_sequences()
    print(f"T counts per cycle: {T_COUNTS_PER_CYCLE}")

    runs = sweep_t_counts(sequences, T_COUNTS_PER_CYCLE)
    save_sweep_data(runs, T_COUNTS_PER_CYCLE)
    print_summary(runs)
