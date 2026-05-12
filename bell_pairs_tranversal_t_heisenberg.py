"""
Bell pair count per code cycle for transversal-logical-blocks-heisenberg.json.

Parallelization rule:
- Iterate sequences in JSON order and pack them into rows (code cycles).
- Each row holds at most `t_count_per_cycle` T gates. When that limit is hit,
  the row is closed and a new one is started.

Bell pair rule (per row):
- Walk the sequences inside the row serially. For each consecutive pair
  (seq_i, seq_{i+1}), every qubit that appears as an 'up' port of seq_i and
  also as a 'down' port of seq_{i+1} contributes one bell pair.

Tile/data format mirrors bell_pairs_count_t_av.py:
- S, H, Sdg: 1 tile / 1 code cycle
- CNOT: 2 tiles / 1 code cycle
- T: 2 tiles / 1 code cycle (no extra waiting cycle inserted)
- Each bell pair = 2 tiles
"""

import csv
import json
from collections import Counter

# Parameters
t_count_per_cycle = 4
INPUT_FILE = "transversal-logical-blocks-fermi-hubbard.json"
OUTPUT_CSV = "bell_pairs_per_cycle_fermi_hubbard.csv"

TILE = {"s": 1, "h": 1, "cx": 2, "t": 2, "sdg": 1}
BELL_PAIR_TILES = 2


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


def reaction_depth(row_up_qubits):
    """Max occurrences of any qubit across the row's up ports."""
    counts = Counter()
    for qubits in row_up_qubits:
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
    comp_tiles = 0
    t_in_row = 0

    def flush_row():
        nonlocal pack_indices, pack_up_qubits, pack_down_qubits
        nonlocal comp_tiles, t_in_row, current_cycle
        if not pack_indices:
            return
        bell_pairs = count_bell_pairs_serial(pack_up_qubits, pack_down_qubits)
        bell_tiles = BELL_PAIR_TILES * bell_pairs
        rd = reaction_depth(pack_up_qubits)
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
        comp_tiles = 0
        t_in_row = 0

    for seq_idx, seq in enumerate(sequences):
        gate = seq.get("gate", "").lower()
        ups = get_up_qubits(seq)
        downs = get_down_qubits(seq)

        pack_indices.append(seq_idx)
        pack_up_qubits.append(ups)
        pack_down_qubits.append(downs)
        comp_tiles += tiles_for_gate(gate)
        if gate == "t":
            t_in_row += 1
            if t_in_row >= t_per_cycle:
                flush_row()

    flush_row()
    return result


def main():
    sequences = load_sequences()
    print(f"Loaded {len(sequences)} sequences from {INPUT_FILE}")
    print(f"t_count_per_cycle: {t_count_per_cycle}")
    print()

    results = schedule_and_count_bell_pairs(sequences)

    print("=" * 95)
    print("BELL PAIRS PER CODE CYCLE")
    print("=" * 95)
    print(
        f"{'Cycle':>6}  {'Bell pairs':>10}  {'Parallel ops':>11}  "
        f"{'Reaction depth':>14}  {'Comp tiles':>11}  {'Bell tiles':>10}  Total tiles"
    )
    print("-" * 95)

    for cycle, bell_pairs, n_ops, comp_tiles, bell_tiles, rd, _ in results:
        total = comp_tiles + bell_tiles
        print(
            f"{cycle:>6}  {bell_pairs:>10}  {n_ops:>11}  {rd:>14}  "
            f"{comp_tiles:>11}  {bell_tiles:>10}  {total}"
        )

    print("=" * 95)
    print(f"Total code cycles: {len(results)}")
    print(f"Total bell pairs (all cycles): {sum(r[1] for r in results)}")
    if results:
        avg = sum(r[1] for r in results) / len(results)
        print(f"Average bell pairs per cycle: {avg:.1f}")
    active_results = [r for r in results if r[2] > 0]
    if active_results:
        rds = [r[5] for r in active_results]
        print(
            f"Reaction depth: min={min(rds)}, max={max(rds)}, "
            f"avg={sum(rds)/len(rds):.1f}"
        )

    with open(OUTPUT_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "cycle",
                "bell_pairs",
                "parallel_ops",
                "reaction_depth",
                "comp_tiles",
                "bell_tiles",
                "total_tiles",
                "seq_indices",
            ]
        )
        for cycle, bell_pairs, n_ops, comp_tiles, bell_tiles, rd, seq_indices in results:
            total = comp_tiles + bell_tiles
            w.writerow(
                [cycle, bell_pairs, n_ops, rd, comp_tiles, bell_tiles, total, seq_indices]
            )
    print(f"\nData saved to {OUTPUT_CSV}")

    bell_pairs_per_cycle = [r[1] for r in results]
    parallel_ops_per_cycle = [r[2] for r in results]
    reaction_depth_per_cycle = [r[5] for r in results]
    print(
        "\nbell_pairs_per_cycle =",
        bell_pairs_per_cycle[:20],
        "..." if len(bell_pairs_per_cycle) > 20 else "",
    )
    print(
        "parallel_ops_per_cycle =",
        parallel_ops_per_cycle[:20],
        "..." if len(parallel_ops_per_cycle) > 20 else "",
    )
    print(
        "reaction_depth_per_cycle =",
        reaction_depth_per_cycle[:20],
        "..." if len(reaction_depth_per_cycle) > 20 else "",
    )


if __name__ == "__main__":
    main()
