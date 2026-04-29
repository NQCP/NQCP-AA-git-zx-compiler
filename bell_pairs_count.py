"""
Count bell pairs required per logical cycle.

Total logical blocks per cycle: 86.
Schedules sequences in JSON file order (no reordering by cost).
Cost per sequence = active_volume + T_state_overhead.

Bell pair sources:
1. Internal: Within a sequence, ports with 'b' values (e.g. b4, b5) indicate bell pairs
   needed when a qubit is used twice in the same sequence.
2. Cross-sequence: When multiple sequences run in parallel and share a qubit (e.g. q3),
   each duplicate use beyond the first requires a bell pair for teleportation.
"""

import json
from collections import deque, Counter

# Constants (match reaction_depth.py)
TOTAL_LOGICAL_BLOCKS = 25#86
T_STATE_OVERHEAD = 0#35 / 2
LOGICAL_BLOCKS_FILE = "logical_blocks.json"


def load_sequences(filename=LOGICAL_BLOCKS_FILE):
    """Load sequences from JSON file."""
    with open(filename) as f:
        sequences = json.load(f)
    return sequences


def calculate_sequence_cost(sequence):
    """Total logical blocks = active_volume + T_state_overhead."""
    return sequence["active_volume"] + T_STATE_OVERHEAD


def count_internal_bell_pairs(sequence):
    """
    Count bell pairs within a sequence from hexagon ports.
    Port values starting with 'b' (e.g. b4, b5) are bell pair identifiers.
    Each unique 'b' value = one bell pair.
    """
    bell_pairs = set()
    for hexagon in sequence.get("hexagons", []):
        for port_val in hexagon.get("ports", {}).values():
            if isinstance(port_val, str) and port_val.startswith("b"):
                bell_pairs.add(port_val)
    return len(bell_pairs)


def get_qubits_used(sequence):
    """
    Extract set of qubit identifiers used by a sequence.
    Port values starting with 'q' (e.g. q1, q3, q8) are qubits.
    """
    qubits = set()
    for hexagon in sequence.get("hexagons", []):
        for port_val in hexagon.get("ports", {}).values():
            if isinstance(port_val, str) and port_val.startswith("q"):
                qubits.add(port_val)
    return qubits


def schedule_sequences_with_indices(sequences, total_capacity=TOTAL_LOGICAL_BLOCKS):
    """
    Schedule sequences to run in parallel in JSON order, returning which sequence
    indices run each cycle.

    Returns:
        List of (cycle, list_of_seq_indices, used_blocks) tuples
    """
    sequence_costs = [(i, calculate_sequence_cost(seq)) for i, seq in enumerate(sequences)]
    schedule = []
    remaining = deque(sequence_costs)

    current_cycle = 0
    while remaining:
        current_used = 0
        scheduled_indices = []
        temp_queue = deque()

        while remaining:
            seq_idx, cost = remaining.popleft()
            if current_used + cost <= total_capacity:
                current_used += cost
                scheduled_indices.append(seq_idx)
            else:
                temp_queue.append((seq_idx, cost))

        schedule.append((current_cycle, scheduled_indices, current_used))
        remaining = temp_queue
        current_cycle += 1

    return schedule


def count_bell_pairs_per_cycle(sequences):
    """
    Count total bell pairs required per logical cycle.

    Returns:
        List of (cycle, total_bell_pairs, internal_bell_pairs, cross_sequence_bell_pairs)
    """
    # Precompute per-sequence data
    internal_bell_pairs = [count_internal_bell_pairs(seq) for seq in sequences]
    qubits_per_sequence = [get_qubits_used(seq) for seq in sequences]

    schedule = schedule_sequences_with_indices(sequences)
    result = []

    for cycle, seq_indices, used_blocks in schedule:
        # Internal bell pairs: sum over all sequences in this cycle
        internal = sum(internal_bell_pairs[i] for i in seq_indices)

        # Cross-sequence bell pairs: qubits used in more than one sequence need teleportation
        qubit_counts = Counter()
        for seq_idx in seq_indices:
            for q in qubits_per_sequence[seq_idx]:
                qubit_counts[q] += 1

        cross_sequence = sum(count - 1 for count in qubit_counts.values() if count > 1)

        total = internal + cross_sequence
        result.append((cycle, total, internal, cross_sequence, len(seq_indices)))

    return result


def get_reaction_depth_per_cycle(bell_pairs_result):
    """
    Reaction depth for each logical cycle.

    If internal bell pair requirement is 0: reaction_depth = l - 1
    Otherwise (internal > 0): reaction_depth = l

    where l = number of sequences performed in parallel that cycle.

    Args:
        bell_pairs_result: Output from count_bell_pairs_per_cycle (with 5-tuples including l)

    Returns:
        List of (cycle, reaction_depth) tuples
    """
    result = []
    for cycle, total, internal, cross_sequence, l in bell_pairs_result:
        if internal == 0:
            reaction_depth = l - 1
        else:
            reaction_depth = l
        result.append((cycle, reaction_depth))
    return result


def main():
    sequences = load_sequences()
    print(f"Loaded {len(sequences)} sequences from {LOGICAL_BLOCKS_FILE}")

    bell_pairs = count_bell_pairs_per_cycle(sequences)
    reaction_depths = get_reaction_depth_per_cycle(bell_pairs)

    print("\n" + "=" * 60)
    print("BELL PAIRS PER LOGICAL CYCLE")
    print("=" * 60)
    for (cycle, total, internal, cross, l), (_, rd) in zip(bell_pairs, reaction_depths):
        print(f"Cycle {cycle:3d}: total={total:3d}  (internal={internal:3d}, cross={cross:3d})  l={l}  reaction_depth={rd}")
    print("=" * 60)
    # save total to a list
    total_bell_pairs = [r[1] for r in bell_pairs]
    reaction_depths_list = [r[1] for r in reaction_depths]
    print(total_bell_pairs)
    print(reaction_depths_list)


    print(f"Total cycles: {len(bell_pairs)}")
    print(f"Total bell pairs (all cycles): {sum(r[1] for r in bell_pairs)}")
    print(f"Average bell pairs per cycle: {sum(r[1] for r in bell_pairs) / len(bell_pairs):.1f}")
    print(f"\nReaction depth: min={min(r[1] for r in reaction_depths)}, max={max(r[1] for r in reaction_depths)}, avg={sum(r[1] for r in reaction_depths) / len(reaction_depths):.1f}")


if __name__ == "__main__":
    main()
