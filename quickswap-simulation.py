"""
Quickswap simulation: count quickswap layers per logical cycle.

Grid: 2 × 86 qubit modules = 172 positions.
- Memory row: positions 1, 3, 5, ..., 171 (odd)
- Workspace row: positions 2, 4, 6, ..., 172 (even)

Connection rule: positions i and j are quickswappable iff |i - j| = 2^k,
where k ∈ {0, 1, ..., floor(log2(N))}, N = 172.
Valid distances: 1, 2, 4, 8, 16, 32, 64, 128.
"""

import json
import math
from collections import deque

# Constants
N_MODULES = 2 * 86  # 172
TOTAL_LOGICAL_BLOCKS = 86
T_STATE_OVERHEAD = 35 / 2
LOGICAL_BLOCKS_FILE = "logical_blocks.json"

# Valid quickswap distances: |i-j| = 2^k, k = 0..floor(log2(172))
MAX_K = int(math.floor(math.log2(N_MODULES)))
VALID_DISTANCES = frozenset(2**k for k in range(MAX_K + 1))  # {1, 2, 4, 8, 16, 32, 64, 128}


def load_sequences(filename=LOGICAL_BLOCKS_FILE):
    with open(filename) as f:
        return json.load(f)


def calculate_sequence_cost(sequence):
    return sequence["active_volume"] + T_STATE_OVERHEAD


def schedule_sequences_with_indices(sequences, total_capacity=TOTAL_LOGICAL_BLOCKS):
    """Schedule sequences in JSON order. Returns (cycle, list_of_seq_indices, used_blocks)."""
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


def get_qubit_id(name):
    """Extract qubit index from name like 'q1', 'q3'."""
    if isinstance(name, str) and name.startswith("q"):
        try:
            return int(name[1:])
        except ValueError:
            pass
    return None


def build_qubit_targets_for_cycle(sequences, seq_indices):
    """
    Build (qubit_id -> target_position) for sequences in this cycle.

    Workspace positions: 2, 4, 6, ..., 172 (slot 0, 1, 2, ...).
    Each hexagon gets a workspace slot. Hexagons that reference a qubit (up/down = q*)
    define where that qubit must be. Use first occurrence as target.
    """
    qubit_to_target = {}  # qubit_id -> position (1..172)
    slot = 0
    for seq_idx in seq_indices:
        seq = sequences[seq_idx]
        for hexagon in seq.get("hexagons", []):
            pos = 2 * (slot + 1)  # workspace position (even)
            for port_val in hexagon.get("ports", {}).values():
                qid = get_qubit_id(port_val)
                if qid is not None and qid not in qubit_to_target:
                    qubit_to_target[qid] = pos
            slot += 1
    return qubit_to_target


def are_connected(i, j):
    """Positions i and j are quickswappable iff |i-j| in {1,2,4,8,16,32,64,128}."""
    d = abs(i - j)
    return d in VALID_DISTANCES


def get_connected_positions(pos):
    """Positions k such that |pos - k| in {1, 2, 4, 8, 16, 32, 64, 128}."""
    result = []
    for d in VALID_DISTANCES:
        if pos - d >= 1:
            result.append(pos - d)
        if pos + d <= N_MODULES:
            result.append(pos + d)
    return result


def find_valid_swap(current_pos, target_pos, eligible):
    """
    Find eligible position k such that:
    - current_pos and k are connected
    - After swap, qubit would be closer to target
    - k is eligible
    - k is as close to target as possible (greedy)
    """
    best_k = None
    best_new_dist = abs(current_pos - target_pos)  # no swap is worse

    for k in get_connected_positions(current_pos):
        if k not in eligible or k == current_pos:
            continue
        new_dist = abs(k - target_pos)
        if new_dist < best_new_dist:
            best_new_dist = new_dist
            best_k = k

    return best_k


def run_quickswap_layers(qubit_positions, qubit_targets, max_layers=1000):
    """
    Run greedy quickswap algorithm. Returns number of layers until all qubits
    at target locations.

    qubit_positions: dict qubit_id -> current position (1..172)
    qubit_targets: dict qubit_id -> target position (1..172)
    """
    positions = dict(qubit_positions)  # mutable copy

    for layer in range(max_layers):
        # Check if done
        all_at_target = True
        for qid, target in qubit_targets.items():
            if positions.get(qid) != target:
                all_at_target = False
                break
        if all_at_target:
            return layer

        # Mark all positions eligible
        eligible = set(range(1, N_MODULES + 1))

        # Process qubits in some order (use sorted for determinism)
        qubits_to_process = sorted(qubit_targets.keys())

        for qid in qubits_to_process:
            target = qubit_targets[qid]
            current = positions.get(qid)
            if current is None:
                continue

            if current == target:
                # Mark target ineligible
                eligible.discard(target)
            else:
                # Try valid quickswap to k as close to target as possible
                k = find_valid_swap(current, target, eligible)
                if k is not None:
                    # Perform swap: qubit at current goes to k, whatever was at k goes to current
                    # We track qubit qid; we need to swap with whoever is at k
                    other_qubit = None
                    for q, pos in positions.items():
                        if pos == k:
                            other_qubit = q
                            break

                    positions[qid] = k
                    if other_qubit is not None:
                        positions[other_qubit] = current

                    # Mark both locations ineligible
                    eligible.discard(current)
                    eligible.discard(k)

    return max_layers  # did not converge


def simulate_quickswap_per_cycle(sequences):
    """
    For each logical cycle, run quickswap algorithm and return layers count.

    Returns:
        List of (cycle, num_quickswap_layers) tuples
    """
    schedule = schedule_sequences_with_indices(sequences)
    result = []

    # Initial qubit positions: qubit i at memory position 2*i - 1 (odd)
    qubit_positions = {}

    for cycle, seq_indices, _ in schedule:
        qubit_targets = build_qubit_targets_for_cycle(sequences, seq_indices)

        # Ensure we have positions for all qubits that need targets
        for qid in qubit_targets:
            if qid not in qubit_positions:
                # Default: memory position 2*qid - 1, clamped to valid range
                qubit_positions[qid] = min(2 * qid - 1, N_MODULES)

        if not qubit_targets:
            result.append((cycle, 0))
            continue

        layers = run_quickswap_layers(qubit_positions, qubit_targets)
        result.append((cycle, layers))

        # Update positions for next cycle: qubits end at their targets
        for qid, target in qubit_targets.items():
            qubit_positions[qid] = target

    return result


def main():
    print("Loading...")
    sequences = load_sequences()
    print(f"Loaded {len(sequences)} sequences from {LOGICAL_BLOCKS_FILE}")
    print(f"N = {N_MODULES} modules, valid swap distances: {sorted(VALID_DISTANCES)}")

    print("Running quickswap simulation...")
    quickswap_layers = simulate_quickswap_per_cycle(sequences)
    print(f"Simulation done. Total cycles: {len(quickswap_layers)}")

    # Plot
    cycles_list = [r[0] for r in quickswap_layers]
    layers_list = [r[1] for r in quickswap_layers]
    print(f"Max layers: {max(layers_list)}, Avg: {sum(layers_list)/len(layers_list):.1f}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        try:
            plt.style.use("plotstylefile.mplstyle")
        except OSError:
            pass
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(cycles_list, layers_list, "o-", markersize=3)
        ax.set_xlabel("Logical cycle")
        ax.set_ylabel("Quickswap layers")
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)
        ax.grid(True, alpha=0.3)
        textstr = f"Total cycles: {len(cycles_list)}\nMax layers: {max(layers_list)}\nAvg layers: {sum(layers_list)/len(layers_list):.1f}"
        ax.text(0.98, 0.98, textstr, transform=ax.transAxes, verticalalignment="top", horizontalalignment="right")
        plt.tight_layout()
        plt.savefig("quickswap_layers_vs_cycles.pdf", dpi=150, bbox_inches="tight")
        print("Saved quickswap_layers_vs_cycles.pdf")
    except Exception as e:
        print("Plot skipped:", e)


if __name__ == "__main__":
    main()
