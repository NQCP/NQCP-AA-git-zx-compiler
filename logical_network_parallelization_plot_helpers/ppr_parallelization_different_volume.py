"""
Compute per-cycle parallelism and utilization data across multiple
TOTAL_LOGICAL_BLOCKS budgets and save to data/sweep_<circuit>.npz.

The strict-FIFO scheduler (in ppr_parallelization.py) is run once per capacity.
Outputs are cached so plot tweaks (see plot_sweep.ipynb) don't re-run the
scheduler.

Re-run this script when:
  - The input file logical_blocks_*.jsonl.gz changes
  - You want different capacities (edit CAPACITIES below)
  - The scheduler logic in ppr_parallelization.py changes

Run from the project root:

    python logical_network_parallelization_plot_helpers/ppr_parallelization_different_volume.py
"""

from pathlib import Path
import numpy as np

from ppr_parallelization import (
    load_sequences,
    schedule_sequences,
    T_STATE_OVERHEAD,
)


# Workspace capacities (logical blocks) to sweep.
CAPACITIES = [250, 500, 750, 1000, 1500, 2000]

CIRCUIT = "fermi_hubbard_2d_step_s4_universal_paulis_commuted"

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
INPUT_FILE = PROJECT_ROOT / "logical_network_files" / f"logical_blocks_{CIRCUIT}.jsonl.gz"
DATA_DIR = HERE / "data"
OUTPUT_FILE = DATA_DIR / f"sweep_{CIRCUIT}.npz"


def run_for_capacity(sequences, capacity):
    """Schedule once at the given capacity; return parallel_ops and util arrays."""
    schedule = schedule_sequences(sequences, total_capacity=capacity)
    parallel_ops = np.asarray([s[1] for s in schedule])
    used_blocks = np.asarray([s[2] for s in schedule])
    util = used_blocks / capacity
    return schedule, parallel_ops, util


def sweep_capacities(sequences, capacities):
    """Run scheduler at each capacity once; return list of per-capacity dicts."""
    runs = []
    for cap in capacities:
        print(f"\n--- Scheduling at capacity = {cap} ---")
        schedule, po, util = run_for_capacity(sequences, cap)
        runs.append({
            "capacity": cap,
            "schedule": schedule,
            "parallel_ops": po,
            "util": util,
        })
    return runs


def save_sweep_data(runs, capacities, output_file=OUTPUT_FILE):
    """Save per-capacity parallel_ops and util arrays to a compressed .npz."""
    output_file.parent.mkdir(exist_ok=True)
    arrays = {"capacities": np.array(capacities, dtype=np.int64)}
    for run in runs:
        cap = int(run["capacity"])
        arrays[f"parallel_ops_{cap}"] = np.asarray(run["parallel_ops"], dtype=np.int32)
        arrays[f"util_{cap}"] = np.asarray(run["util"], dtype=np.float32)
    np.savez_compressed(output_file, **arrays)
    print(f"\nSaved sweep data to {output_file}")
    for run in runs:
        cap = int(run["capacity"])
        n = len(run["parallel_ops"])
        po_kb = arrays[f"parallel_ops_{cap}"].nbytes / 1024
        ut_kb = arrays[f"util_{cap}"].nbytes / 1024
        print(f"  capacity={cap}: {n:,} cycles  ({po_kb:.1f} KB parallel_ops + {ut_kb:.1f} KB util)")
    print(f"  Compressed file size: {output_file.stat().st_size / (1024 ** 2):.2f} MB")


def print_summary(runs):
    print("\n" + "=" * 70)
    print("SUMMARY: parallelism / utilization vs capacity")
    print("=" * 70)
    print(f"{'capacity':>10}  {'cycles':>12}  {'mean l':>8}  {'p95 l':>6}  {'max l':>6}  {'mean util':>10}")
    for r in runs:
        cap = r["capacity"]
        po = r["parallel_ops"]
        util = r["util"]
        print(f"{cap:>10}  {len(po):>12,}  {po.mean():>8.2f}  "
              f"{np.percentile(po, 95):>6.1f}  {po.max():>6.0f}  {util.mean():>10.2f}")
    print("=" * 70)


if __name__ == "__main__":
    print(f"Loading sequences from {INPUT_FILE}...")
    sequences = load_sequences(str(INPUT_FILE))
    print(f"T_state_overhead (inherited from ppr_parallelization.py): {T_STATE_OVERHEAD}")

    runs = sweep_capacities(sequences, CAPACITIES)
    save_sweep_data(runs, CAPACITIES)
    print_summary(runs)
