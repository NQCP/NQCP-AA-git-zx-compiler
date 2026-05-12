"""
Count bell pairs required per logical cycle.

Workspace capacity per cycle: TOTAL_LOGICAL_BLOCKS = 25 logical blocks
(remaining capacity after reserving 35 of the 86 total blocks for MSD; matches
the convention in reaction_depth.py).

Cost per sequence = active_volume + T_state_overhead.

Scheduling: strict FIFO. Sequences are taken from the front of the JSON in
order; a sequence is placed in the current cycle if it fits in the remaining
capacity, otherwise the cycle ends and that sequence becomes the first of the
next cycle. No overtaking: a small sequence later in the JSON is NEVER
scheduled before a bigger sequence earlier in the JSON. This is the correct
behavior for inputs whose rows are only block-commuting (as in fermi-hubbard).
Lone exception: if a single sequence's cost exceeds total_capacity, it is
packed alone in its own cycle to avoid an infinite loop.

Bell pair sources:
1. Internal: Within a sequence, ports with 'b' values (e.g. b4, b5) indicate bell pairs
   needed when a qubit is used twice in the same sequence.
2. Cross-sequence: When multiple sequences run in parallel and share a qubit (e.g. q3),
   each duplicate use beyond the first requires a bell pair for teleportation.
"""

import gzip
import json
from collections import deque, Counter

# Constants (match reaction_depth.py)
TOTAL_LOGICAL_BLOCKS = 25#86
T_STATE_OVERHEAD = 0#35 / 2
LOGICAL_BLOCKS_FILE = "logical_blocks.json"


def load_sequences(filename=LOGICAL_BLOCKS_FILE):
    """Load sequences from a JSON file, or a gzipped JSON-Lines file (.jsonl.gz)."""
    if filename.endswith('.gz'):
        with gzip.open(filename, 'rt') as f:
            return [json.loads(line) for line in f if line.strip()]
    with open(filename) as f:
        return json.load(f)


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
    Schedule sequences in strict JSON order (no overtaking).
    Stops filling a cycle as soon as the next sequence doesn't fit.

    Returns:
        List of (cycle, list_of_seq_indices, used_blocks) tuples
    """
    remaining = deque(
        (i, calculate_sequence_cost(seq)) for i, seq in enumerate(sequences)
    )
    schedule = []
    current_cycle = 0

    while remaining:
        current_used = 0
        scheduled_indices = []
        while remaining:
            seq_idx, cost = remaining[0]
            # Pack if it fits, or if this is the first item in the cycle
            # (handles a single oversized sequence without infinite-looping).
            if current_used + cost <= total_capacity or not scheduled_indices:
                remaining.popleft()
                current_used += cost
                scheduled_indices.append(seq_idx)
            else:
                break
        schedule.append((current_cycle, scheduled_indices, current_used))
        current_cycle += 1

    return schedule


def count_bell_pairs_per_cycle(sequences, schedule=None):
    """
    Count total bell pairs required per logical cycle.

    Args:
        sequences: List of sequence dicts.
        schedule:  Optional precomputed schedule (avoids recomputing it).

    Returns:
        List of (cycle, total_bell_pairs, internal_bell_pairs, cross_sequence_bell_pairs, l)
    """
    # Precompute per-sequence data
    internal_bell_pairs = [count_internal_bell_pairs(seq) for seq in sequences]
    qubits_per_sequence = [get_qubits_used(seq) for seq in sequences]

    if schedule is None:
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


def pauli_strings_anticommute(p1, p2):
    """Return True iff the two positional Pauli product strings anticommute.

    Each input is a list of single-char tokens ('i', 'x', 'y', 'z') with
    positional meaning (p[i] = Pauli on qubit i). Two Pauli products
    anticommute iff the number of qubit positions where their non-identity
    Paulis differ is odd.
    """
    n = min(len(p1), len(p2))
    diff = 0
    for i in range(n):
        a = p1[i]
        b = p2[i]
        if a != 'i' and b != 'i' and a != b:
            diff += 1
    return (diff & 1) == 1


def reaction_depth_chain_length(ppr_strings):
    """Length of the longest chain of pairwise anticommutations in CSV order.

    Each PPR's reaction must wait for any earlier-in-cycle anticommuting PPR's
    measurement outcome (the correction would otherwise flip the next PPR's
    rotation axis). The reaction depth of a cycle is the longest such chain.

    For pairwise-anticommuting PPRs: returns k (every pair adds an edge).
    For pairwise-commuting PPRs: returns 1 (no edges).
    Mixed: somewhere in between.
    """
    if not ppr_strings:
        return 0
    n = len(ppr_strings)
    dp = [1] * n
    for i in range(1, n):
        for j in range(i):
            if pauli_strings_anticommute(ppr_strings[j], ppr_strings[i]):
                if dp[j] + 1 > dp[i]:
                    dp[i] = dp[j] + 1
    return max(dp)


def get_reaction_depth_per_cycle(sequences, schedule):
    """
    Exact reaction depth per cycle from pairwise anticommutation of the
    scheduled PPRs' Pauli strings (from each sequence's `input_sequence`).

    Args:
        sequences: List of sequence dicts (must contain `input_sequence`).
        schedule:  Output of schedule_sequences_with_indices(sequences) —
                   list of (cycle, [seq_indices], used_blocks) tuples.

    Returns:
        List of (cycle, reaction_depth) tuples.
    """
    result = []
    for cycle, seq_indices, _used in schedule:
        ppr_strings = [sequences[i]["input_sequence"] for i in seq_indices]
        rd = reaction_depth_chain_length(ppr_strings)
        result.append((cycle, rd))
    return result


def count_qubit_uses_per_sequence(sequence):
    """Counter of q* port appearances within a single sequence (not deduped)."""
    return Counter(
        p for h in sequence.get("hexagons", [])
        for p in h.get("ports", {}).values()
        if isinstance(p, str) and p.startswith("q")
    )


def diagnose_undertagged_sequences(sequences, threshold=3):
    """
    Report sequences where some qubit appears `threshold`+ times in the PPR.
    logical_blocks.py only tags internal bell pairs when an `up` value is shared
    between exactly 2 hexagons, so qubits reused 3+ times within one PPR are
    silently under-tagged (count_internal_bell_pairs will miss them).
    """
    flagged = []
    for i, seq in enumerate(sequences):
        counts = count_qubit_uses_per_sequence(seq)
        max_use = max(counts.values(), default=0)
        # Each hexagon connects a qubit on up AND down ports in the degree-1
        # case, so a qubit that appears on 2 hexagons shows up 4 times here.
        # The "exactly 2 hexagons" tag-trigger corresponds to max_use == 4.
        # Reuses beyond that (>= 3 distinct hexagons) appear as max_use >= 6 or
        # interleaved patterns. Flag anything above the 2-hexagon case.
        if max_use > 4:
            flagged.append((i, seq.get("sequence_id"), max_use, dict(counts)))
    return flagged


def main():
    sequences = load_sequences()
    print(f"Loaded {len(sequences)} sequences from {LOGICAL_BLOCKS_FILE}")

    flagged = diagnose_undertagged_sequences(sequences)
    print(f"\nDIAGNOSTIC: {len(flagged)}/{len(sequences)} sequences have a qubit reused on 3+ hexagons")
    print("  (these may be missing internal bell-pair tags from logical_blocks.py)")
    if flagged:
        print("  First few examples:")
        for i, sid, max_use, counts in flagged[:5]:
            top = sorted(counts.items(), key=lambda kv: -kv[1])[:3]
            print(f"    seq idx={i} id={sid}  max port-count={max_use}  top qubits={top}")

    schedule = schedule_sequences_with_indices(sequences)
    bell_pairs = count_bell_pairs_per_cycle(sequences, schedule=schedule)
    reaction_depths = get_reaction_depth_per_cycle(sequences, schedule)

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
