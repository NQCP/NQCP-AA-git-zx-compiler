"""Qiskit construction of one 2D long-range TFIM benchmark Trotter step.

Reference:
    Sahil Khan et al., "Architecting Early Fault Tolerant Neutral Atoms
    Systems with Quantum Advantage", arXiv:2604.19735.

Benchmark configuration (paper Section 2.4):
    - 100 qubits on a 10x10 lattice
    - transverse field B = 1
    - power-law-decay ZZ coupling J_{ij} = 1 / r_{ij}^alpha with alpha = 2
    - one fourth-order Suzuki step
    - delta = 1 Trotter step over total evolution time T = 10

The Hamiltonian splits into two mutually commuting groups:
    H_ZZ = sum_{i<j} J_{ij} Z_i Z_j
    H_X  = B sum_i X_i

so each symmetric second-order Suzuki step is just three sub-layers
(X, ZZ, X). The fourth-order step is built from S2 by Yoshida recursion,
matching the structure used in HubbardCircuit.py.
"""

from math import pow, sqrt
from typing import Dict, List, Sequence, Tuple

from qiskit import QuantumCircuit, transpile

PairCoupling = Tuple[int, int, float]
LayerSpec = Tuple[str, float]

ROWS = 10
COLS = 10
NUM_QUBITS = ROWS * COLS

B_FIELD = 1.0
ALPHA = 2.0

TOTAL_EVOLUTION_TIME = 10.0
TROTTER_STEPS = 1
SUZUKI_ORDER = 4
DELTA_T = TOTAL_EVOLUTION_TIME / TROTTER_STEPS
DEFAULT_BASIS_GATES = ("cx", "t", "tdg", "rz", "sx", "s", "sdg", "h")


def benchmark_configuration() -> Dict[str, object]:
    """Return the paper configuration used for the benchmark circuit."""
    return {
        "reference": "arXiv:2604.19735",
        "model": "2D long-range TFIM",
        "rows": ROWS,
        "cols": COLS,
        "num_qubits": NUM_QUBITS,
        "boundary_conditions": "open",
        "transverse_field": B_FIELD,
        "alpha": ALPHA,
        "total_evolution_time": TOTAL_EVOLUTION_TIME,
        "trotter_steps": TROTTER_STEPS,
        "delta_t": DELTA_T,
        "suzuki_order": SUZUKI_ORDER,
    }


def lattice_index(row: int, col: int, cols: int = COLS) -> int:
    """Map a lattice site to its row-major qubit index."""
    return row * cols + col


def site_position(idx: int, cols: int = COLS) -> Tuple[int, int]:
    """Inverse of lattice_index."""
    return idx // cols, idx % cols


def build_zz_couplings(
    rows: int = ROWS,
    cols: int = COLS,
    alpha: float = ALPHA,
) -> List[PairCoupling]:
    """Return [(i, j, J_ij), ...] for all i<j with J_ij = 1/r_ij^alpha."""
    pairs: List[PairCoupling] = []
    n = rows * cols
    for i in range(n):
        ri, ci = site_position(i, cols)
        for j in range(i + 1, n):
            rj, cj = site_position(j, cols)
            r = sqrt((ri - rj) ** 2 + (ci - cj) ** 2)
            J = 1.0 / (r ** alpha)
            pairs.append((i, j, J))
    return pairs


def strang_layer_sequence(tau: float) -> List[LayerSpec]:
    """Return one symmetric second-order Suzuki step S2(tau) for H = H_ZZ + H_X."""
    half_tau = tau / 2.0
    return [
        ("X", half_tau),
        ("ZZ", tau),
        ("X", half_tau),
    ]


def suzuki_layer_sequence(order: int = SUZUKI_ORDER, tau: float = DELTA_T) -> List[LayerSpec]:
    """Return the unmerged layer sequence for S_order(tau)."""
    if order % 2 != 0 or order < 2:
        raise ValueError("Suzuki order must be an even integer >= 2.")
    if order == 2:
        return strang_layer_sequence(tau)

    coeff = 1.0 / (4.0 - pow(4.0, 1.0 / (order - 1)))
    scaled = coeff * tau
    middle = (1.0 - 4.0 * coeff) * tau
    lower = suzuki_layer_sequence(order - 2, scaled)

    return lower + lower + suzuki_layer_sequence(order - 2, middle) + lower + lower


def apply_zz_evolution(circuit: QuantumCircuit, q0: int, q1: int, tau: float) -> None:
    """Apply exp[-i tau Z⊗Z]."""
    circuit.cx(q0, q1)
    circuit.rz(2.0 * tau, q1)
    circuit.cx(q0, q1)


def apply_x_evolution(circuit: QuantumCircuit, q: int, tau: float) -> None:
    """Apply exp[-i tau X] = H · RZ(2 tau) · H, kept in the chosen basis explicitly."""
    circuit.h(q)
    circuit.rz(2.0 * tau, q)
    circuit.h(q)


def apply_zz_layer(
    circuit: QuantumCircuit,
    pairs: Sequence[PairCoupling],
    tau: float,
) -> None:
    """Apply exp[-i tau sum_{i<j} J_ij Z_i Z_j]. All ZZ terms commute, so order is free."""
    for i, j, J in pairs:
        apply_zz_evolution(circuit, i, j, J * tau)


def apply_x_layer(
    circuit: QuantumCircuit,
    num_qubits: int,
    tau: float,
    b_field: float = B_FIELD,
) -> None:
    """Apply exp[-i tau B sum_i X_i]."""
    angle = b_field * tau
    for q in range(num_qubits):
        apply_x_evolution(circuit, q, angle)


def build_lrtfim_trotter_step(
    rows: int = ROWS,
    cols: int = COLS,
    tau: float = DELTA_T,
    order: int = SUZUKI_ORDER,
    include_barriers: bool = False,
) -> QuantumCircuit:
    """Build one benchmark Trotter step with the paper configuration."""
    if rows != ROWS or cols != COLS:
        raise ValueError("This benchmark step is configured for the 100-qubit 10x10 lattice.")

    pairs = build_zz_couplings(rows=rows, cols=cols, alpha=ALPHA)
    layer_sequence = suzuki_layer_sequence(order=order, tau=tau)
    circuit = QuantumCircuit(rows * cols, name="lrtfim_2d_step_s{0}".format(order))

    circuit.metadata = benchmark_configuration()
    circuit.metadata["delta_t"] = tau
    circuit.metadata["suzuki_order"] = order
    circuit.metadata["layer_sequence"] = [(label, angle) for label, angle in layer_sequence]
    circuit.metadata["num_layers"] = len(layer_sequence)
    circuit.metadata["num_zz_pairs"] = len(pairs)

    for label, layer_tau in layer_sequence:
        if label == "X":
            apply_x_layer(circuit, rows * cols, layer_tau)
        elif label == "ZZ":
            apply_zz_layer(circuit, pairs, layer_tau)
        else:
            raise ValueError("Unknown layer label: {0}".format(label))
        if include_barriers:
            circuit.barrier()

    return circuit


def count_gates_in_basis(
    circuit: QuantumCircuit,
    basis_gates: Sequence[str] = DEFAULT_BASIS_GATES,
) -> Tuple[Dict[str, int], int]:
    """Transpile to a target basis and return per-gate and total counts."""
    transpiled_circuit = transpile(
        circuit,
        basis_gates=list(basis_gates),
        optimization_level=0,
    )
    gate_counts = dict(transpiled_circuit.count_ops())
    total_gates = sum(gate_counts.values())
    return gate_counts, total_gates


if __name__ == "__main__":
    qc = build_lrtfim_trotter_step()
    basis_gate_counts, total_basis_gates = count_gates_in_basis(qc)
    # rz count 
    rz_count = basis_gate_counts.get("rz", 0)
    t_count = rz_count * 100
    print("Estimated T count (assuming 100 rz gates per T):", t_count)

    print("Built circuit:", qc.name)
    print("Qubits:", qc.num_qubits)
    print("Layers in unmerged Suzuki expansion:", qc.metadata["num_layers"])
    print("ZZ pairs per ZZ layer:", qc.metadata["num_zz_pairs"])
    print("Delta t:", qc.metadata["delta_t"])
    print("Basis gates:", list(DEFAULT_BASIS_GATES))
    print("Basis gate counts:", basis_gate_counts)
    print("Total basis gates:", total_basis_gates)

    from qiskit.qasm2 import dumps

    with open('ppr_circuits/lrtfim_2d_step_s{0}.qasm'.format(SUZUKI_ORDER), 'w') as f:
        f.write(dumps(qc))
