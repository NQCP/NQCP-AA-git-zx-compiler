"""
Spacetime volume (STV) estimator.

For each (benchmark, architecture, physical error rate), sweeps the
architecture's tuning parameter (workspace capacity for AV, T-per-cycle
for t-AV / cult-tAV, factory count for Baseline / Compact) and reports
the minimum STV in qubit*code-cycle units.

Run from the project root:
    python resource_estimators/space-time-volume-estimator.py
"""

import qubits_runtime_estimates_v2 as q
from qubits_runtime_estimates_v2 import (
    USE_CASES,
    LS_FACTORY_DISTANCE,
    build_av_rows,
    build_superconducting_rows,
    build_tav_rows,
    build_cultivation_rows,
    min_distance_from_csv,
)
from av_compilation import (
    C_T,
    C_T_SINGLE_STAGE,
    ZERO_DIST_EXTRA_T_BLOCKS,
    split_active_volume_from_csv,
    total_tav_blocks_and_t_count,
)

# FH / QPE-Abs need two-level (concatenated) distillation only at p = 1e-3; at
# p = 1e-4 a single-stage 15-to-1 suffices. Stat-QPE / Stat-QPE-gap are always
# single-stage. (App. tab:lsdist-configs / tab:15-to-1-comparison.)
CONCAT_FACTORY_USE_CASES = {"single_trotter_step", "trotter_full_qpe"}

ARCHITECTURES = [
    "Baseline",
    "Compact",
    "AV",
    "t-AV (atoms)",
    "t-AV (photonics)",
    "cult-tAV (atoms)",
    "cult-tAV (photonics)",
]

# Rows that report (transversal) active volume instead of STV in tab:stv-summary.
ACTIVE_VOL_LABELS = {"AV", "t-AV (atoms)", "t-AV (photonics)"}

ERROR_RATES = ("0.001", "0.0001")

# Cultivation cycles per state, per physical error rate (App. fold-transversal
# f=5 stage count: r_att = 12 code cycles/attempt; 5 attempts at 1e-3, 1.5 at 1e-4).
CULTIVATION_CYCLES_PER_STATE_BY_RATE = {"0.001": 12 * 5, "0.0001": 12 * 1.5}


def runtime_in_code_cycles(row, arch):
    if arch in ("AV", "Baseline", "Compact"):
        return row["code_distance"] * row["total_cycles"]
    if arch in ("t-AV (atoms)", "t-AV (photonics)"):
        return row["total_cycles"]
    return row["runtime_cycles"]  # cult-tAV variants


def best_stv(rows, arch):
    if not rows:
        return None
    best = None
    for row in rows:
        runtime_cc = runtime_in_code_cycles(row, arch)
        stv = row["physical_qubits"] * runtime_cc
        if best is None or stv < best[2]:
            best = (row["physical_qubits"], runtime_cc, stv)
    return best


def rows_for(spec, arch):
    if arch == "AV":
        return build_av_rows(spec)[0]
    if arch in ("Baseline", "Compact"):
        # Wider factory sweep than the runtime figures (which use the default
        # 20): with the concatenated factory at p = 1e-3 the Baseline STV
        # minimum sits at n = 4*dm2 = 52 factories (one T per clock cycle).
        return build_superconducting_rows(spec, arch, max_factories=60)
    if arch in ("t-AV (atoms)", "t-AV (photonics)"):
        arch_key = "t-av" if arch == "t-AV (atoms)" else "transversal"
        err_model = "atoms" if arch_key == "t-av" else "circuit"
        return build_tav_rows(spec, arch_key, err_model, arch)[0]
    arch_key = "t-av" if "atoms" in arch else "transversal"
    err_model = "atoms" if arch_key == "t-av" else "circuit"
    return build_cultivation_rows(spec, arch_key, err_model, arch)[0]


# Active-volume metric per architecture, with the STV key it maps to:
#   (STV label, csv architecture, csv error model, is_transversal)
ACTIVE_VOL_ARCHS = [
    ("AV", "av", "circuit", False),
    ("t-AV (atoms)", "t-av", "atoms", True),
    ("t-AV (photonics)", "transversal", "circuit", True),
]


def active_volumes_physical(spec, rate):
    """Active volume (AV) and transversal active volume (t-AV) per architecture,
    converted to physical qubit * code-cycle units so they are directly
    comparable to the STV reported above.

    Metrics are computed in logical-block (qubit * logical-cycle) units and
    converted with 1 logical block = 2 d^2 physical qubits held for d code
    cycles = 2 d^3. AV uses a split-distance conversion: the Clifford (C_m)
    blocks live at the algorithm distance, while the distillation blocks
    (C_T = 25.75 per T state) live at the LS-factory distance d_fac
    (App. tab:lsdist-configs), which is set by the output-error requirement.
    t-AV blocks all run in-fabric at the algorithm distance and are
    distance-independent counts divided by d (t-AV physical = blocks * 2 d^2).
    t-AV is reported under both the neutral-atom and the photonic (circuit) noise
    models, so it can be compared against AV at the same noise model.
    """
    steps = spec["trotter_steps"]
    cm_blocks, n_rotations = split_active_volume_from_csv(spec["ppr_file"])
    tav_blocks_step, tav_t_count_step = total_tav_blocks_and_t_count(
        spec["transversal_json"]
    )
    tav_blocks = tav_blocks_step * steps
    # 0-dist + trans-dist for photonics (circuit model) at p = 1e-3 on the
    # benchmarks single-stage 15-to-1 cannot supply (FH, QPE-Abs): each T
    # carries an extra ZERO_DIST_EXTRA_T_BLOCKS. Atoms (erasure model) keep
    # plain trans-dist via injected-error post-selection.
    tav_blocks_zero_dist = (
        tav_blocks_step + tav_t_count_step * ZERO_DIST_EXTRA_T_BLOCKS
    ) * steps
    d_fac = LS_FACTORY_DISTANCE[spec["use_case"]]

    out = {}
    for label, arch, err_model, transversal in ACTIVE_VOL_ARCHS:
        d = min_distance_from_csv(
            spec["circuit"], spec["use_case"], arch, err_model, rate
        )
        if transversal:
            blocks = tav_blocks
            if (
                err_model == "circuit"
                and spec["use_case"] in CONCAT_FACTORY_USE_CASES
                and rate == "0.001"
            ):
                blocks = tav_blocks_zero_dist
            phys = (blocks / d) * 2 * d ** 3
        else:
            # AV distillation: concatenated factory (25.75 blocks/T at d_fac)
            # only for FH / QPE-Abs at p = 1e-3; otherwise single-stage 15-to-1
            # (17.5 blocks/T) run in-fabric at the algorithm distance d.
            if spec["use_case"] in CONCAT_FACTORY_USE_CASES and rate == "0.001":
                distill = n_rotations * C_T * 2 * d_fac ** 3
            else:
                distill = n_rotations * C_T_SINGLE_STAGE * 2 * d ** 3
            phys = (cm_blocks * 2 * d ** 3 + distill) * steps
        out[label] = {"d": d, "phys": phys}
    return out


def reported_cost(label, arch, results_rate, active_rate):
    """Quantity tabulated in tab:stv-summary: active volume for the AV and t-AV
    rows, minimum STV for every other architecture (physical qubit*code-cycle)."""
    if arch in ACTIVE_VOL_LABELS:
        return active_rate[label][arch]["phys"]
    res = results_rate[label].get(arch)
    return res[2] if res else float("nan")


def main():
    results = {}
    active = {}
    for rate in ERROR_RATES:
        q.DISTANCE_ERROR_RATE = rate
        q.CULTIVATION_CYCLES_PER_STATE = CULTIVATION_CYCLES_PER_STATE_BY_RATE[rate]
        results[rate] = {}
        active[rate] = {}
        for spec in USE_CASES:
            results[rate][spec["label"]] = {
                arch: best_stv(rows_for(spec, arch), arch) for arch in ARCHITECTURES
            }
            active[rate][spec["label"]] = active_volumes_physical(spec, rate)

    for rate in ERROR_RATES:
        print()
        print(f"=== p = {float(rate):.0e}  (min STV in qubit * code-cycle) ===")
        print(f"{'Benchmark':<18} {'Architecture':<22} {'Qubits':>12} {'Cycles':>14} {'STV':>12}")
        print("-" * 82)
        for spec in USE_CASES:
            label = spec["label"]
            for arch in ARCHITECTURES:
                res = results[rate][label].get(arch)
                if res is None:
                    continue
                q_count, cycles, stv = res
                print(f"{label:<18} {arch:<22} {q_count:>12,} {cycles:>14,} {stv:>12.2e}")
            print()

    for rate in ERROR_RATES:
        print()
        print(
            f"=== p = {float(rate):.0e}  active volume vs space-time volume "
            f"(physical qubit * code-cycle) ==="
        )
        print(
            f"{'Benchmark':<18} {'Architecture':<16} {'d':>4} "
            f"{'active vol':>12} {'STV':>12} {'STV/active':>11}"
        )
        print("-" * 76)
        for spec in USE_CASES:
            label = spec["label"]
            for arch, *_ in ACTIVE_VOL_ARCHS:
                act = active[rate][label][arch]
                stv = results[rate][label].get(arch)
                stv_v = stv[2] if stv else float("nan")
                ratio = stv_v / act["phys"] if act["phys"] else float("nan")
                print(
                    f"{label:<18} {arch:<16} {act['d']:>4} "
                    f"{act['phys']:>12.2e} {stv_v:>12.2e} {ratio:>11.2f}"
                )
            print()

    # tab:stv-summary table: AV and t-AV rows report active volume, all other
    # architectures report minimum STV (physical qubit * code-cycle units).
    for rate in ERROR_RATES:
        print()
        print(
            f"=== p = {float(rate):.0e}  tab:stv-summary  (AV & t-AV: active "
            f"volume; others: min STV; physical qubit * code-cycle) ==="
        )
        header = f"{'Architecture':<22}" + "".join(
            f"{spec['label']:>16}" for spec in USE_CASES
        )
        print(header)
        print("-" * len(header))
        for arch in ARCHITECTURES:
            row = f"{arch:<22}"
            for spec in USE_CASES:
                row += f"{reported_cost(spec['label'], arch, results[rate], active[rate]):>16.2e}"
            print(row)
        print()


if __name__ == "__main__":
    main()
