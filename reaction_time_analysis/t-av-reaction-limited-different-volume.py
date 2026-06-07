"""
Reaction-limited stalling contours for T-AV compilation across T-count settings.

Runs a sweep over T gates allowed per code cycle for one transversal
logical-network circuit, caches the per-cycle reaction depths, and leaves
plotting to the notebook helper.
"""

import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
DATA_DIR = HERE / "data"

sys.path.insert(0, str(PROJECT_ROOT))

SEQUENCES_FILE = "transversal-logical-blocks-fermi-hubbard.json"
T_COUNTS_PER_CYCLE = [1, 2, 3, 4, 5]
CODE_DISTANCE = 17


from bell_pair_analysis_helpers.bell_pairs_tranversal import (  # noqa: E402
    load_sequences,
    schedule_and_count_bell_pairs,
)


def input_file_for_sequences_file(sequences_file):
    return PROJECT_ROOT / "logical_network_files" / sequences_file


def data_file_for_sequences_file(sequences_file):
    return DATA_DIR / f"reaction_limited_sweep_t_av_{Path(sequences_file).stem}.npz"


INPUT_FILE = input_file_for_sequences_file(SEQUENCES_FILE)
DATA_FILE = data_file_for_sequences_file(SEQUENCES_FILE)


def run_for_t_count(sequences, t_count_per_cycle):
    """
    Compute reaction depths for a given T-count-per-cycle setting.
    Returns dict with reaction_depths_list, n_cycles, total_bell_pairs_sum.
    """
    results = schedule_and_count_bell_pairs(sequences, t_per_cycle=int(t_count_per_cycle))
    reaction_depths_list = [int(r[5]) for r in results]
    total_bell_pairs_sum = int(sum(r[1] for r in results))
    return {
        "reaction_depths_list": reaction_depths_list,
        "n_cycles": len(results),
        "total_bell_pairs_sum": total_bell_pairs_sum,
    }


def run_t_count_sweep(input_file=INPUT_FILE, t_counts=T_COUNTS_PER_CYCLE):
    sequences = load_sequences(str(input_file))

    results = []
    for t_count in t_counts:
        run = run_for_t_count(sequences, t_count)
        run["t_count"] = int(t_count)
        run["legend_label"] = f"T/cycle = {int(t_count)}"
        results.append(run)
    print(f"Loaded and scheduled {len(sequences):,} transversal sequences from {input_file}")
    return results


def compute_stalling_grid(reaction_depths, d, tau_r, tau_c):
    """
    Return total stalling time on a tau_r x tau_c grid.

    Stalling = sum_k max(0, tau_r * reaction_depth_k - d * tau_c)
    """
    reaction_depths = np.asarray(reaction_depths, dtype=np.int32)
    values, counts = np.unique(reaction_depths, return_counts=True)
    grid = np.zeros((len(tau_r), len(tau_c)), dtype=float)
    for rd, count in zip(values, counts):
        if rd <= 0:
            continue
        grid += count * np.maximum(0.0, rd * tau_r[:, None] - d * tau_c[None, :])
    return grid


def get_no_stalling_boundary(reaction_depths, d, tau_c):
    max_rd = max(reaction_depths) if reaction_depths else 0
    if max_rd == 0:
        return np.full_like(tau_c, np.nan)
    return d * tau_c / max_rd


def save_sweep_data(results, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(exist_ok=True)

    arrays = {
        "t_counts": np.array([int(run["t_count"]) for run in results], dtype=np.int32),
        "n_cycles": np.array([int(run["n_cycles"]) for run in results], dtype=np.int32),
        "total_bell_pairs_sum": np.array(
            [int(run["total_bell_pairs_sum"]) for run in results], dtype=np.int64
        ),
        "code_distance": np.array([int(CODE_DISTANCE)], dtype=np.int32),
    }
    for run in results:
        arrays[f"reaction_depths_{int(run['t_count'])}"] = np.array(
            run["reaction_depths_list"], dtype=np.int16
        )

    np.savez_compressed(output_path, **arrays)
    print(f"Saved {output_path}")


def load_sweep_data(npz_path):
    data = np.load(npz_path)
    t_counts = data["t_counts"].tolist()
    results = []
    for idx, t_count in enumerate(t_counts):
        results.append(
            {
                "t_count": int(t_count),
                "legend_label": f"T/cycle = {int(t_count)}",
                "reaction_depths_list": data[f"reaction_depths_{int(t_count)}"].tolist(),
                "n_cycles": int(data["n_cycles"][idx]),
                "total_bell_pairs_sum": int(data["total_bell_pairs_sum"][idx]),
            }
        )
    code_distance = int(data["code_distance"][0])
    return results, code_distance


def main():
    results = run_t_count_sweep()

    for run in results:
        rd = np.asarray(run["reaction_depths_list"])
        active = rd[rd > 0]
        print(f"\n--- T/cycle = {run['t_count']} ---")
        print(f"  Logical cycles: {run['n_cycles']:,}")
        print(f"  Total bell pairs (all cycles): {run['total_bell_pairs_sum']:,}")
        if active.size:
            print(
                "  Reaction depth: "
                f"min={active.min()}, max={active.max()}, avg={active.mean():.2f}"
            )

    save_sweep_data(results, DATA_FILE)
    print(f"\nSaved sweep data only: {DATA_FILE}")


if __name__ == "__main__":
    main()
