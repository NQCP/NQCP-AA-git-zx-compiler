"""
Bell pair count per code cycle for transversal-logical-blocks.json.

Schedules sequences in JSON order (no reordering).
Capacity = available_patches.

Gate requirements:
- S, H: 1 tile, 1 code cycle
- CNOT: 2 tiles, 1 code cycle
- T: 2 tiles, 2 code cycles (1 extra waiting cycle for all T gates in parallel)

Bell pairs: if qubit q_i appears k_i times in 'up' ports across parallel sequences
in a code cycle, need b_i = k_i - 1 bell pairs for that qubit. Each bell pair = 2 tiles.

Reaction depth: max(k_1, ..., k_i, ...) over all qubits in that code cycle.
"""

import csv
import json
from collections import Counter, deque

cultivation = False
distillation = True


if distillation:
    # Capacity parameters
    d = 17
    capacity = 175 - 50
    num_factories = 4
    tiles_per_factory = 15
    buffer_tiles = 15
    available_patches = capacity - (tiles_per_factory * num_factories + buffer_tiles)
    print(available_patches)
    # exit()
    # Gate tile requirements
    TILE = {"s": 1, "h": 1, "cx": 2, "t": 2, "sdg": 1}
    BELL_PAIR_TILES = 2

if cultivation: 
    d = 17
    tiles_per_factory = 1
    available_patches = 50
    TILE = {"s": 1, "h": 1, "cx": 2, "t": 2, "sdg": 1}
    BELL_PAIR_TILES = 2



def load_sequences(filename="transversal-logical-blocks-heisenberg.json"):
    with open(filename) as f:
        return json.load(f)


def get_up_qubits(sequence):
    """Extract qubit indices from 'up' ports of all circles in a sequence."""
    qubits = []
    for circle in sequence.get("circles", []):
        up = circle.get("ports", {}).get("up")
        if up is not None:
            qubits.append(up)
    return qubits


def tiles_for_gate(gate):
    return TILE.get(gate.lower(), 1)


def get_qubit_counts(up_qubits_list):
    """Count occurrences of each qubit across parallel sequences."""
    counts = Counter()
    for qubits in up_qubits_list:
        for q in qubits:
            counts[q] += 1
    return counts


def count_bell_pairs(up_qubits_list):
    """
    Given list of up-qubit lists from parallel sequences, count total bell pairs.
    For each qubit q with count k across all sequences: b_i = k_i - 1.
    """
    counts = get_qubit_counts(up_qubits_list)
    return sum(max(0, c - 1) for c in counts.values())


def reaction_depth(up_qubits_list):
    """Reaction depth = max(k_1, ..., k_i, ...) over all qubits in the cycle."""
    counts = get_qubit_counts(up_qubits_list)
    return max(counts.values()) if counts else 0


def schedule_and_count_bell_pairs(sequences, capacity=available_patches):
    """
    Schedule sequences in order, pack by capacity, compute bell pairs per cycle.

    T gates can run in parallel with other gates on the same qubit; use bell pairs
    for parallelization (same logic as Clifford gates). T takes 1 extra waiting cycle
    globally but does not block packing.

    Returns:
        List of (cycle, bell_pairs, num_sequences, comp_tiles, bell_tiles, reaction_depth, seq_indices)
    """
    remaining = deque(enumerate(sequences))
    result = []
    current_cycle = 0

    while remaining:
        pack_indices = []
        pack_up_qubits = []
        comp_tiles = 0

        while remaining:
            seq_idx, seq = remaining[0]
            up_qubits = get_up_qubits(seq)
            gate_tiles = tiles_for_gate(seq.get("gate", "").lower())

            # Tentative pack: current + this sequence
            trial_up = pack_up_qubits + [up_qubits]
            trial_bell = count_bell_pairs(trial_up)
            trial_comp = comp_tiles + gate_tiles
            trial_total = trial_comp + BELL_PAIR_TILES * trial_bell

            if trial_total <= capacity:
                remaining.popleft()
                pack_indices.append(seq_idx)
                pack_up_qubits.append(up_qubits)
                comp_tiles += gate_tiles
            else:
                break

        bell_pairs = count_bell_pairs(pack_up_qubits)
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

        # T gates take 1 extra waiting cycle; insert it after any cycle that has T gates
        has_t_gate = any(sequences[i].get("gate", "").lower() == "t" for i in pack_indices)
        if has_t_gate:
            result.append(
                (current_cycle, 0, 0, 0, 0, 0, [])  # T waiting cycle
            )
            current_cycle += 1

    return result


def main():
    sequences = load_sequences()
    print(f"Loaded {len(sequences)} sequences from transversal-logical-blocks-heisenberg.json")
    print(f"Available patches (capacity): {available_patches}")
    print()

    results = schedule_and_count_bell_pairs(sequences)

    print("=" * 80)
    print("BELL PAIRS PER CODE CYCLE")
    print("=" * 80)
    print(
        f"{'Cycle':>6}  {'Bell pairs':>10}  {'Parallel ops':>11}  {'Reaction depth':>14}  {'Comp tiles':>11}  {'Bell tiles':>10}  Total tiles"
    )
    print("-" * 95)

    for cycle, bell_pairs, n_ops, comp_tiles, bell_tiles, rd, _ in results:
        total = comp_tiles + bell_tiles
        print(f"{cycle:>6}  {bell_pairs:>10}  {n_ops:>11}  {rd:>14}  {comp_tiles:>11}  {bell_tiles:>10}  {total}")

    print("=" * 80)
    print(f"Total code cycles: {len(results)}")
    print(f"Total bell pairs (all cycles): {sum(r[1] for r in results)}")
    if results:
        avg = sum(r[1] for r in results) / len(results)
        print(f"Average bell pairs per cycle: {avg:.1f}")
    active_results = [r for r in results if r[2] > 0]  # exclude waiting cycles
    if active_results:
        rds = [r[5] for r in active_results]
        print(f"Reaction depth: min={min(rds)}, max={max(rds)}, avg={sum(rds)/len(rds):.1f}")

    # Save to file
    output_file = "bell_pairs_per_cycle_data.csv"
    with open(output_file, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["cycle", "bell_pairs", "parallel_ops", "reaction_depth", "comp_tiles", "bell_tiles", "total_tiles", "seq_indices"])
        for cycle, bell_pairs, n_ops, comp_tiles, bell_tiles, rd, seq_indices in results:
            total = comp_tiles + bell_tiles
            w.writerow([cycle, bell_pairs, n_ops, rd, comp_tiles, bell_tiles, total, seq_indices])
    print(f"\nData saved to {output_file}")

    # Export for downstream use
    bell_pairs_per_cycle = [r[1] for r in results]
    parallel_ops_per_cycle = [r[2] for r in results]
    reaction_depth_per_cycle = [r[5] for r in results]
    print("\nbell_pairs_per_cycle =", bell_pairs_per_cycle[:20], "..." if len(bell_pairs_per_cycle) > 20 else "")
    print("parallel_ops_per_cycle =", parallel_ops_per_cycle[:20], "..." if len(parallel_ops_per_cycle) > 20 else "")
    print("reaction_depth_per_cycle =", reaction_depth_per_cycle[:20], "..." if len(reaction_depth_per_cycle) > 20 else "")


if __name__ == "__main__":
    main()
