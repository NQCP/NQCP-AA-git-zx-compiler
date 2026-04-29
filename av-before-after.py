"""
Compute AV before and after replacing repeating Pauli segments with cheap gadgets.
"""

import importlib.util
import json
import sys

# Load ppr-optimization (hyphen in filename, so use importlib)
spec = importlib.util.spec_from_file_location("ppr_opt", "ppr-optimization.py")
ppr_opt = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ppr_opt)
load_rotations_full = ppr_opt.load_rotations_full
find_maximal_commuting_segments = ppr_opt.find_maximal_commuting_segments


def load_av_by_index(json_path: str) -> dict:
    """Load logical_blocks.json and return {index: active_volume}."""
    with open(json_path) as f:
        data = json.load(f)
    av = {}
    for entry in data:
        sid = entry["sequence_id"]
        idx = int(sid.split("_")[1])
        av[idx] = entry["active_volume"]
    return av


def bases_to_key(bases) -> str:
    if isinstance(bases, str):
        return bases.upper()
    return "".join(b.upper() for b in bases)


def compute_av_before_after(
    segments: list,
    rotations: list,
    av_by_index: dict,
) -> list:
    """
    For each segment (P, indices):
    - AV before: sum of AV of all ops in the segment span [min(indices), max(indices)]
    - AV after: replace AV of repeating P ops with 1, sum all, then add 2*(original AV of P sequence)
    """
    results = []
    for p_key, p_indices in segments:
        i_min = min(p_indices)
        i_max = max(p_indices)
        # Span is inclusive [i_min, i_max]
        span = range(i_min, i_max + 1)
        p_set = set(p_indices)
        av_before = sum(av_by_index.get(i, 0) for i in span)
        av_of_p_ops = sum(av_by_index.get(i, 0) for i in p_indices)

        # Replace AV of P ops with 1, sum all, then add 2 * (original AV per P op)
        av_replaced = sum(1 if i in p_set else av_by_index.get(i, 0) for i in span)
        av_after = av_replaced + 2 * (av_of_p_ops / len(p_indices))
        results.append({
            "P": p_key,
            "indices": p_indices,
            "len": len(p_indices),
            "span": (i_min, i_max),
            "av_before": av_before,
            "av_after": av_after,
            "av_of_p_ops": av_of_p_ops,
        })
    return results


def main():
    csv_path = "ppr_circuits/trotter_circuit_v2_paulis_commuted.csv"
    json_path = "logical_blocks.json"

    print("Loading rotations and segments...")
    rotations = load_rotations_full(csv_path)
    segments = find_maximal_commuting_segments(rotations, min_p_per_segment=2)

    print("Loading AV from logical_blocks.json...")
    av_by_index = load_av_by_index(json_path)

    results = compute_av_before_after(segments, rotations, av_by_index)
    results_sorted = sorted(results, key=lambda r: -r["len"])

    print("\n" + "=" * 80)
    print("AV BEFORE vs AFTER (repeating P → cheap gadget)")
    print("=" * 80)
    print(f"{'P':<12} {'len':>4} {'span':>12} {'AV before':>10} {'AV after':>10} {'saved':>8}  % gain")
    print("-" * 80)

    total_av_before = 0
    total_av_after = 0
    for r in results_sorted:
        saved = r["av_before"] - r["av_after"]
        total_av_before += r["av_before"]
        total_av_after += r["av_after"]
        print(
            f"{r['P']:<12} {r['len']:>4} "
            f"({r['span'][0]:4d}-{r['span'][1]:4d}) "
            f"{r['av_before']:>10} {r['av_after']:>10} {saved:>8}"
        )

    print("-" * 80)
    saved_total = total_av_before - total_av_after
    pct_total = 100 * saved_total / total_av_before if total_av_before else 0
    print(f"{'TOTAL':<12} {'':<4} {'':<12} {total_av_before:>10} {total_av_after:>10} "
          f"{saved_total:>8}  ({pct_total:.1f}%)")

    # Totals neglecting rows with negative gain
    total_before_positive = sum(r["av_before"] for r in results_sorted if r["av_before"] - r["av_after"] >= 0)
    total_after_positive = sum(r["av_after"] for r in results_sorted if r["av_before"] - r["av_after"] >= 0)
    saved_positive = total_before_positive - total_after_positive
    pct_positive = 100 * saved_positive / total_before_positive if total_before_positive else 0
    print(f"{'TOTAL (≥0 gain)':<12} {'':<4} {'':<12} {total_before_positive:>10} {total_after_positive:>10} "
          f"{saved_positive:>8}  ({pct_positive:.1f}%)")


if __name__ == "__main__":
    main()
