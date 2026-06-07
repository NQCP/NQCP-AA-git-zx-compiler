"""
Compute per-cycle bell-pair counts across multiple TOTAL_LOGICAL_BLOCKS
budgets and save to data/bell_pairs_sweep_<circuit>.npz.

Uses a streaming loader with per-sequence summaries precomputed at load time,
so heavy `hexagons` data is discarded after extraction. This avoids the ~30 GB
OOM that hexagon-heavy loading would hit on the 1M-row fermi-hubbard file.

Re-run this script when:
  - The input file logical_blocks_*.jsonl.gz changes
  - You want different capacities (edit CAPACITIES below)
  - The bell-pair counting logic or scheduler changes

Run from the project root:

    python bell_pair_analysis_helpers/bell_pairs_count.py

Scheduling: strict FIFO. Sequences are taken from the front of the JSON in
order; a sequence is placed in the current cycle if it fits in the remaining
capacity, otherwise the cycle ends and that sequence becomes the first of the
next cycle. No overtaking. Lone exception: if a single sequence's cost exceeds
total_capacity, it is packed alone in its own cycle.

Bell pair sources:
1. Internal: Within a sequence, ports with 'b' values (e.g. b4, b5) indicate
   bell pairs needed when a qubit is used twice in the same sequence.
2. Cross-sequence: When multiple sequences run in parallel and share a qubit,
   each duplicate use beyond the first requires a bell pair for teleportation.
"""

import gzip
import json
import sys
from collections import deque, Counter
from pathlib import Path

import numpy as np


# Workspace capacities (logical blocks) to sweep.
CAPACITIES = [413, 500, 750, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000]
# CAPACITIES = [25, 50, 75, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950, 1000]
T_STATE_OVERHEAD = 0

# CIRCUIT = "tmm_paulis_commuted"
CIRCUIT = "fermi_hubbard_2d_step_s4_universal_paulis_commuted"

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
INPUT_FILE = PROJECT_ROOT / "logical_network_files" / f"logical_blocks_{CIRCUIT}.jsonl.gz"
# INPUT_FILE = PROJECT_ROOT / "logical_network_files" / f"logical_blocks_{CIRCUIT}.json"
DATA_DIR = HERE / "data"
OUTPUT_FILE = DATA_DIR / f"bell_pairs_sweep_{CIRCUIT}.npz"


# ---------------------------------------------------------------------------
# Streaming loader: extract only what's needed per sequence, discard hexagons
# ---------------------------------------------------------------------------

def _extract_internal_bp(obj):
    """Count unique 'b*' tags in this sequence's hexagon ports."""
    bell_pairs = set()
    for h in obj.get("hexagons", []):
        for v in h.get("ports", {}).values():
            if isinstance(v, str) and v.startswith("b"):
                bell_pairs.add(v)
    return len(bell_pairs)


def _extract_qubits(obj):
    """Tuple of unique 'q*' qubit labels (interned strings, low memory)."""
    qubits = set()
    for h in obj.get("hexagons", []):
        for v in h.get("ports", {}).values():
            if isinstance(v, str) and v.startswith("q"):
                qubits.add(sys.intern(v))
    return tuple(qubits)


def load_compact_sequences(filename):
    """Stream-load a sequence file, keeping only fields needed for downstream
    bell-pair / reaction-depth analysis. The heavy `hexagons` field is
    consumed once to compute summaries (internal-bp count, qubit set) and
    then discarded; `input_sequence` is collapsed from list-of-1-char strings
    to a single str.

    Memory: ~700 bytes per sequence (vs ~30 KB if hexagons are kept).
    """
    sequences = []
    filename = str(filename)

    if filename.endswith(".gz"):
        f_ctx = gzip.open(filename, "rt")
    else:
        f_ctx = open(filename, "r")

    with f_ctx as f:
        if filename.endswith(".gz"):
            iterable = (json.loads(line) for line in f if line.strip())
        else:
            iterable = iter(json.load(f))
        for obj in iterable:
            sequences.append({
                "active_volume": obj["active_volume"],
                "input_sequence": "".join(obj.get("input_sequence", [])),
                "_internal_bp": _extract_internal_bp(obj),
                "_qubits": _extract_qubits(obj),
            })

    print(f"Loaded {len(sequences)} sequences from {filename}")
    return sequences


# Backward-compatible alias used by older callers.
load_sequences = load_compact_sequences


# ---------------------------------------------------------------------------
# Per-sequence summaries (use precomputed values if present)
# ---------------------------------------------------------------------------

def calculate_sequence_cost(sequence):
    """Total logical blocks = active_volume + T_state_overhead."""
    return sequence["active_volume"] + T_STATE_OVERHEAD


def count_internal_bell_pairs(sequence):
    """Number of internal bell pairs (unique 'b*' tags) in this sequence."""
    if "_internal_bp" in sequence:
        return sequence["_internal_bp"]
    bell_pairs = set()
    for hexagon in sequence.get("hexagons", []):
        for port_val in hexagon.get("ports", {}).values():
            if isinstance(port_val, str) and port_val.startswith("b"):
                bell_pairs.add(port_val)
    return len(bell_pairs)


def get_qubits_used(sequence):
    """Tuple/set of unique qubit identifiers used by this sequence."""
    if "_qubits" in sequence:
        return sequence["_qubits"]
    qubits = set()
    for hexagon in sequence.get("hexagons", []):
        for port_val in hexagon.get("ports", {}).values():
            if isinstance(port_val, str) and port_val.startswith("q"):
                qubits.add(port_val)
    return qubits


# ---------------------------------------------------------------------------
# Scheduler (strict FIFO) and per-cycle bell-pair counting
# ---------------------------------------------------------------------------

def schedule_sequences_with_indices(sequences, total_capacity):
    """Strict-FIFO scheduling. See module docstring for full semantics."""
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
            if current_used + cost <= total_capacity or not scheduled_indices:
                remaining.popleft()
                current_used += cost
                scheduled_indices.append(seq_idx)
            else:
                break
        schedule.append((current_cycle, scheduled_indices, current_used))
        current_cycle += 1
    return schedule


def count_bell_pairs_per_cycle(sequences, schedule):
    """Return per-cycle total bell pairs and used-blocks as int32 numpy arrays.

    total_bp_per_cycle[c] = (internal within each scheduled PPR)
                          + (cross-sequence: qubits shared across PPRs in cycle c)
    """
    internal_bp = [count_internal_bell_pairs(seq) for seq in sequences]
    qubits = [get_qubits_used(seq) for seq in sequences]

    n_cycles = len(schedule)
    total_bp = np.zeros(n_cycles, dtype=np.int32)
    used = np.zeros(n_cycles, dtype=np.int32)

    for ci, (cycle, seq_indices, used_blocks) in enumerate(schedule):
        internal = sum(internal_bp[i] for i in seq_indices)
        if len(seq_indices) > 1:
            qubit_counts = Counter()
            for seq_idx in seq_indices:
                for q in qubits[seq_idx]:
                    qubit_counts[q] += 1
            cross = sum(c - 1 for c in qubit_counts.values() if c > 1)
        else:
            cross = 0
        total_bp[ci] = internal + cross
        used[ci] = used_blocks

    return total_bp, used


# ---------------------------------------------------------------------------
# Reaction-depth helpers (kept for ad-hoc analysis; not used by main sweep)
# ---------------------------------------------------------------------------

def pauli_strings_anticommute(p1, p2):
    """True iff two positional Pauli product strings anticommute.

    Each input may be a list of single-char tokens or a single string
    ('i', 'x', 'y', 'z' per qubit position). Anticommutation = odd count of
    positions where both are non-identity and differ.
    """
    n = min(len(p1), len(p2))
    diff = 0
    for i in range(n):
        a = p1[i]
        b = p2[i]
        if a != "i" and b != "i" and a != b:
            diff += 1
    return (diff & 1) == 1


def reaction_depth_chain_length(ppr_strings):
    """Length of the longest chain of pairwise anticommutations in input order."""
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
    """Exact reaction depth per cycle from pairwise anticommutation of input_sequence."""
    result = []
    for cycle, seq_indices, _used in schedule:
        ppr_strings = [sequences[i]["input_sequence"] for i in seq_indices]
        result.append((cycle, reaction_depth_chain_length(ppr_strings)))
    return result


# ---------------------------------------------------------------------------
# Sweep + save
# ---------------------------------------------------------------------------

def sweep_capacities(sequences, capacities):
    """Run scheduler + bell-pair counter at each capacity; return per-capacity dicts."""
    runs = []
    for cap in capacities:
        print(f"\n--- Scheduling at capacity = {cap} ---")
        schedule = schedule_sequences_with_indices(sequences, total_capacity=cap)
        total_bp, used = count_bell_pairs_per_cycle(sequences, schedule)
        runs.append({
            "capacity": cap,
            "total_bp": total_bp,
            "used_blocks": used,
        })
        print(f"  {len(schedule):,} cycles, mean bp/cycle = {total_bp.mean():.2f}")
    return runs


def save_sweep_data(runs, capacities, output_file=OUTPUT_FILE):
    """Save total_bp and used_blocks per cycle per capacity to a compressed .npz."""
    output_file.parent.mkdir(exist_ok=True)
    arrays = {"capacities": np.array(capacities, dtype=np.int64)}
    for run in runs:
        cap = int(run["capacity"])
        arrays[f"total_bp_{cap}"] = run["total_bp"]
        arrays[f"used_blocks_{cap}"] = run["used_blocks"]
    np.savez_compressed(output_file, **arrays)
    print(f"\nSaved sweep data to {output_file}")
    for run in runs:
        cap = int(run["capacity"])
        n = len(run["total_bp"])
        bp_kb = arrays[f"total_bp_{cap}"].nbytes / 1024
        ub_kb = arrays[f"used_blocks_{cap}"].nbytes / 1024
        print(f"  capacity={cap}: {n:,} cycles  ({bp_kb:.1f} KB total_bp + {ub_kb:.1f} KB used_blocks)")
    print(f"  Compressed file size: {output_file.stat().st_size / (1024 ** 2):.2f} MB")


def print_summary(runs):
    print("\n" + "=" * 78)
    print("SUMMARY: bell pairs per cycle vs capacity")
    print("=" * 78)
    print(f"{'capacity':>10}  {'cycles':>12}  {'mean bp':>8}  {'p95 bp':>7}  {'max bp':>6}  {'total bp':>12}")
    for r in runs:
        cap = r["capacity"]
        bp = r["total_bp"]
        print(f"{cap:>10}  {len(bp):>12,}  {bp.mean():>8.2f}  "
              f"{np.percentile(bp, 95):>7.1f}  {bp.max():>6.0f}  {int(bp.sum()):>12,}")
    print("=" * 78)


if __name__ == "__main__":
    print(f"Loading sequences from {INPUT_FILE}...")
    sequences = load_compact_sequences(INPUT_FILE)
    print(f"T_state_overhead: {T_STATE_OVERHEAD}")

    runs = sweep_capacities(sequences, CAPACITIES)
    save_sweep_data(runs, CAPACITIES)
    print_summary(runs)
