"""
Energy error vs Trotter dt and gridsynth precision — standalone plot helper.

Computes two sweeps for the TMM molecule:
  1. GS / excited-state / gap energy error vs Trotter time step dt
  2. GS / excited-state / gap energy error vs gridsynth approximation precision

Results are cached in data/ so the slow gridsynth precision sweep only runs
once.  Delete the cache files to force a recompute.

Usage:
    python circuit_compilation_helpers/energy_error_plot_helper.py
"""

from pathlib import Path
import pickle

import mpmath
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import logm, eigh
from qiskit import QuantumCircuit, qasm3
from qiskit.circuit.library import UnitaryGate
from qiskit.exceptions import MissingOptionalLibraryError
from qiskit.qasm2 import dumps as qasm2_dumps
from qiskit.quantum_info import Operator
from openfermion import FermionOperator
from openfermion.linalg import get_sparse_operator
from pygridsynth.gridsynth import gridsynth_gates

# ── paths ──────────────────────────────────────────────────────────────────────

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
MISC_DIR = PROJECT_ROOT / "miscellaneous"
QASM_PATH = PROJECT_ROOT / "ppr_circuits" / "trotter_circuit_tmm_v3_logical.qasm"
TWO_COLUMN_STYLE = PROJECT_ROOT / "plotstylefile_two_column.mplstyle"
DATA_DIR = HERE / "data"

# ── simulation config ──────────────────────────────────────────────────────────

N_QUBITS = 8
N_STEPS = 1
TAU = 2.4
CHEM_ACCURACY = 0.04354 / 3

DT_VALUES = np.array([0.01, 0.02, 0.03, 0.04, 0.05, 0.06])
FIXED_DT = 0.04
PRECISION_VALUES = np.array([1e-2, 5e-3, 1e-3, 5e-4, 1e-4, 5e-5, 1e-5])
PRECISION_LABELS = ["1e-2", "5e-3", "1e-3", "5e-4", "1e-4", "5e-5", "1e-5"]

# ── circuit construction ───────────────────────────────────────────────────────

def total_number_operator(n_spin_orbitals):
    N_op = FermionOperator()
    for i in range(n_spin_orbitals):
        N_op += FermionOperator(((i, 1), (i, 0)), 1.0)
    return N_op


def construct_V_time_evolution_circuit(qc, V_jw, dt):
    for term, coeff in V_jw.terms.items():
        theta = 2 * coeff.real
        if len(term) == 1:
            qc.rz(theta * dt, term[0][0])
        elif len(term) == 2:
            q1, q2 = term[0][0], term[1][0]
            qc.cx(q1, q2)
            qc.rz(theta * dt, q2)
            qc.cx(q1, q2)
    return qc


def F_ij_circuit(qc, q1, q2):
    qc.s(q2); qc.h(q2); qc.h(q1)
    qc.cx(q1, q2); qc.tdg(q2); qc.h(q1); qc.s(q1)
    qc.cx(q1, q2); qc.t(q2); qc.h(q1)
    qc.cx(q1, q2); qc.h(q2); qc.h(q1); qc.s(q1)
    return qc


def XXYY_ij_circuit(qc, theta, t, q1, q2):
    qc.h(q1); qc.h(q2); qc.s(q1); qc.s(q2)
    qc.h(q1); qc.h(q2); qc.cx(q1, q2); qc.h(q1)
    qc.rz(-theta * t, q1); qc.rz(-theta * t, q2)
    qc.h(q1); qc.cx(q1, q2)
    qc.s(q1); qc.s(q2); qc.h(q1); qc.h(q2)
    qc.s(q1); qc.s(q2)


def construct_T_time_evolution_TMM(qc, theta, dt):
    XXYY_ij_circuit(qc, theta, dt / 2, 0, 1)
    XXYY_ij_circuit(qc, theta, dt / 2, 4, 5)
    F_ij_circuit(qc, 2, 3)
    XXYY_ij_circuit(qc, np.sqrt(2) * theta, dt, 1, 2)
    F_ij_circuit(qc, 2, 3)
    F_ij_circuit(qc, 6, 7)
    XXYY_ij_circuit(qc, np.sqrt(2) * theta, dt, 5, 6)
    F_ij_circuit(qc, 6, 7)
    XXYY_ij_circuit(qc, theta, dt / 2, 0, 1)
    XXYY_ij_circuit(qc, theta, dt / 2, 4, 5)
    return qc


def trotter_circuit_tmm(qc, V_jw, n_steps, dt, tau=TAU):
    construct_V_time_evolution_circuit(qc, V_jw, dt / 2)
    for _ in range(n_steps):
        construct_T_time_evolution_TMM(qc, tau, dt)
        construct_V_time_evolution_circuit(qc, V_jw, dt)
    construct_V_time_evolution_circuit(qc, V_jw, -dt / 2)


# ── eigenstate helpers ─────────────────────────────────────────────────────────

def _eigenstate_indices(states, num_op_matrix, n_eigenstates=20):
    N_e = N_QUBITS // 2
    return [
        i for i in range(n_eigenstates)
        if np.round(
            np.vdot(states[:, i], num_op_matrix @ states[:, i]).real,
            decimals=1,
        ) == N_e
    ]


def _effective_hamiltonian(unitary, dt, n_steps):
    global_phase = (-1) ** n_steps
    u = np.eye(2 ** N_QUBITS) * global_phase @ unitary
    H = logm(u) / (-1j * dt)
    return (H + H.conj().T) / 2  # symmetrize to remove numerical noise


# ── gridsynth / logical-to-universal ──────────────────────────────────────────

_W_GATE_MATRIX = np.diag([np.exp(1j * np.pi / 4), np.exp(1j * np.pi / 4)])


def rot_decompose(qc, theta, qubit_idx, precision_str):
    mpmath.mp.dps = 256
    gates = gridsynth_gates(
        theta=mpmath.mpmathify(str(theta)),
        epsilon=mpmath.mpmathify(precision_str),
    )[::-1]
    W_gate = UnitaryGate(_W_GATE_MATRIX, label="W")
    _gate_map = {
        "H": qc.h, "T": qc.t, "S": qc.s,
        "X": qc.x, "Y": qc.y, "Z": qc.z,
    }
    for g in gates:
        if g == "W":
            qc.append(W_gate, [qubit_idx])
        elif g in _gate_map:
            _gate_map[g](qubit_idx)


def _circuit_from_qasm(path):
    text = Path(path).read_text()
    for raw in text.splitlines():
        s = raw.strip()
        if not s or s.startswith("//"):
            continue
        if s.upper().startswith("OPENQASM") and "3" in s:
            try:
                return qasm3.load(Path(path))
            except MissingOptionalLibraryError:
                return qasm3.load_experimental(Path(path))
        break
    return QuantumCircuit.from_qasm_str(text)


def logical_to_universal(path, precision_str):
    qc_base = _circuit_from_qasm(path)
    qc = QuantumCircuit(qc_base.num_qubits)
    for instr in qc_base.data:
        op = instr.operation
        qidxs = [qc_base.find_bit(q).index for q in instr.qubits]
        if op.name == "rz":
            rot_decompose(qc, op.params[0], qidxs[0], precision_str)
        else:
            qc.append(op, qidxs)
    return qc


# ── sweeps ─────────────────────────────────────────────────────────────────────

def _energy_errors(energies, states, num_op_matrix, exact_gs, exact_excited):
    idx = _eigenstate_indices(states, num_op_matrix)
    approx_gs = energies[idx[0]]
    approx_ex = energies[idx[-2]]
    gs_err = abs(approx_gs - float(exact_gs))
    ex_err = abs(approx_ex - float(exact_excited))
    gap_approx = approx_ex - approx_gs
    gap_exact = float(exact_excited) - float(exact_gs)
    gap_err = abs(gap_approx - gap_exact)
    return gs_err, ex_err, gap_err


def compute_dt_sweep(V_jw, num_op_matrix, exact_gs, exact_excited):
    cache = DATA_DIR / "energy_error_dt_sweep.npz"
    if cache.exists():
        d = np.load(cache)
        return d["gs_errors"], d["ex_errors"], d["gap_errors"]

    gs_errors, ex_errors, gap_errors = [], [], []
    for dt in DT_VALUES:
        print(f"  dt={dt:.2f} ...", flush=True)
        qc = QuantumCircuit(N_QUBITS)
        trotter_circuit_tmm(qc, V_jw, N_STEPS, dt)
        H_eff = _effective_hamiltonian(Operator(qc).data, dt, N_STEPS)
        energies, states = eigh(H_eff)
        gs_e, ex_e, gap_e = _energy_errors(energies, states, num_op_matrix, exact_gs, exact_excited)
        gs_errors.append(gs_e); ex_errors.append(ex_e); gap_errors.append(gap_e)
        print(f"    gs={gs_e:.4f}  ex={ex_e:.4f}  gap={gap_e:.4f}")

    result = {
        "gs_errors": np.array(gs_errors),
        "ex_errors": np.array(ex_errors),
        "gap_errors": np.array(gap_errors),
    }
    DATA_DIR.mkdir(exist_ok=True)
    np.savez(cache, **result)
    return result["gs_errors"], result["ex_errors"], result["gap_errors"]


def compute_precision_sweep(V_jw, num_op_matrix, exact_gs, exact_excited):
    cache = DATA_DIR / "energy_error_precision_sweep.npz"
    if cache.exists():
        d = np.load(cache)
        return d["gs_errors"], d["ex_errors"], d["gap_errors"], d["t_per_rot"]

    qc_base = QuantumCircuit(N_QUBITS)
    trotter_circuit_tmm(qc_base, V_jw, N_STEPS, FIXED_DT)
    QASM_PATH.parent.mkdir(exist_ok=True)
    with open(QASM_PATH, "w") as f:
        f.write(qasm2_dumps(qc_base))
    print(f"  Written {QASM_PATH}")

    n_rz = qc_base.count_ops().get("rz", 0)
    print(f"  Logical circuit has {n_rz} rz gates")

    gs_errors, ex_errors, gap_errors, t_per_rot = [], [], [], []
    for prec_str in PRECISION_LABELS:
        print(f"  precision={prec_str} ...", flush=True)
        qc_u = logical_to_universal(QASM_PATH, prec_str)
        ops = qc_u.count_ops()
        t_count = ops.get("t", 0) + ops.get("tdg", 0)
        t_per_rot.append(t_count / n_rz)
        H_eff = _effective_hamiltonian(Operator(qc_u).data, FIXED_DT, N_STEPS)
        energies, states = eigh(H_eff)
        gs_e, ex_e, gap_e = _energy_errors(energies, states, num_op_matrix, exact_gs, exact_excited)
        gs_errors.append(gs_e); ex_errors.append(ex_e); gap_errors.append(gap_e)
        print(f"    T/rot={t_count/n_rz:.1f}  gs={gs_e:.4f}  ex={ex_e:.4f}  gap={gap_e:.4f}")

    result = {
        "gs_errors": np.array(gs_errors),
        "ex_errors": np.array(ex_errors),
        "gap_errors": np.array(gap_errors),
        "t_per_rot": np.array(t_per_rot),
    }
    DATA_DIR.mkdir(exist_ok=True)
    np.savez(cache, **result)
    return result["gs_errors"], result["ex_errors"], result["gap_errors"], result["t_per_rot"]


# ── plotting ───────────────────────────────────────────────────────────────────

def _style_axes(ax):
    ax.tick_params(axis="both", direction="in", width=1.5, length=5.5,
                   top=True, right=True)
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)


def plot_energy_errors(gs_dt, ex_dt, gap_dt, gs_prec, ex_prec, gap_prec, t_counts):
    gs_color = "#0072B2"
    ex_color = "#D55E00"
    gap_color = "#009E73"

    with plt.style.context(str(TWO_COLUMN_STYLE)):
        fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.8), sharey=True)

        # left: dt sweep
        ax = axes[0]
        l1, = ax.plot(DT_VALUES, gs_dt, marker="o", linewidth=1.8, markersize=4.5,
                      color=gs_color, label="GS error")
        l2, = ax.plot(DT_VALUES, ex_dt, marker="s", linewidth=1.8, markersize=4.5,
                      color=ex_color, label="ES error")
        l3, = ax.plot(DT_VALUES, gap_dt, marker="^", linewidth=1.8, markersize=4.5,
                      color=gap_color, label="Gap error")
        ax.axhline(CHEM_ACCURACY, color="black", linestyle="--", linewidth=1.0, alpha=0.7)
        ax.set_xlabel(r"$dt$ ($\mathrm{eV}^{-1}$)")
        ax.set_ylabel("Energy error (eV)")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xticks(DT_VALUES)
        ax.xaxis.set_major_formatter(plt.matplotlib.ticker.FuncFormatter(lambda x, _: f"{x:g}"))
        ax.xaxis.set_minor_formatter(plt.matplotlib.ticker.NullFormatter())
        ax.tick_params(axis="x", rotation=30)
        ax.grid(color="0.88", linewidth=0.7)
        ax.set_axisbelow(True)
        _style_axes(ax)
        ax.tick_params(axis="both", labelsize=16)

        # right: T count sweep
        ax = axes[1]
        ax.plot(t_counts, gs_prec, marker="o", linewidth=1.8, markersize=4.5,
                color=gs_color, label="GS error")
        ax.plot(t_counts, ex_prec, marker="s", linewidth=1.8, markersize=4.5,
                color=ex_color, label="ES error")
        ax.plot(t_counts, gap_prec, marker="^", linewidth=1.8, markersize=4.5,
                color=gap_color, label="Gap error")
        ax.axhline(CHEM_ACCURACY, color="black", linestyle="--", linewidth=1.0, alpha=0.7)
        ax.set_xlabel(r"T count per $R_z$")
        ax.set_yscale("log")
        ax.set_xticks(t_counts)
        ax.xaxis.set_major_formatter(plt.matplotlib.ticker.FuncFormatter(lambda x, _: f"{x:.0f}"))
        ax.tick_params(axis="x", rotation=30)
        ax.grid(color="0.88", linewidth=0.7)
        ax.set_axisbelow(True)
        ax.text(0.97, 0.90, f"$dt = {FIXED_DT}\\,\\mathrm{{eV}}^{{-1}}$", transform=ax.transAxes,
                ha="right", va="top", fontsize=15)
        _style_axes(ax)
        ax.tick_params(axis="both", labelsize=16)

        chem_line = plt.matplotlib.lines.Line2D(
            [], [], color="black", linestyle="--", linewidth=1.0, alpha=0.7,
            label="Chemical accuracy",
        )
        fig.legend(
            [l1, l2, l3, chem_line],
            ["GS error", "ES error", "Gap error", r"$\epsilon_{\mathrm{chem}}/3$"],
            loc="upper center", bbox_to_anchor=(0.5, 0.02),
            ncol=4, frameon=False, handlelength=1.8, columnspacing=1.0,
        )
        fig.tight_layout(rect=[0, 0.08, 1, 1], pad=0.18, w_pad=0.8)
    return fig


def save_for_paper(fig, filename):
    out_dirs = [PROJECT_ROOT / "paper_plots"]
    overleaf = PROJECT_ROOT.parent / "let-s-estimate-some-resources" / "Plots"
    if overleaf.exists():
        out_dirs.append(overleaf)
    for d in out_dirs:
        d.mkdir(exist_ok=True)
        path = d / filename
        fig.savefig(path, bbox_inches="tight", pad_inches=0.03)
        print(f"Saved {path}")


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    with open(MISC_DIR / "V_terms.pkl", "rb") as f:
        V_jw = pickle.load(f)

    exact_gs = np.load(MISC_DIR / "Gs_energy.npy")
    exact_excited = np.load(MISC_DIR / "Excited_state_energy.npy")

    num_op = total_number_operator(N_QUBITS)
    num_op_matrix = get_sparse_operator(num_op, n_qubits=N_QUBITS).toarray()

    print("Computing dt sweep...")
    gs_dt, ex_dt, gap_dt = compute_dt_sweep(V_jw, num_op_matrix, exact_gs, exact_excited)

    print("Computing precision sweep (slow — uses gridsynth)...")
    gs_prec, ex_prec, gap_prec, t_counts = compute_precision_sweep(V_jw, num_op_matrix, exact_gs, exact_excited)

    fig = plot_energy_errors(gs_dt, ex_dt, gap_dt, gs_prec, ex_prec, gap_prec, t_counts)
    save_for_paper(fig, "energy_error_tmm.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
