"""Qiskit construction of one 2D Fermi-Hubbard benchmark Trotter step.

Reference:
    Sahil Khan et al., "Architecting Early Fault Tolerant Neutral Atoms
    Systems with Quantum Advantage", arXiv:2604.19735.

Benchmark configuration (paper Section 2.4):
    - 10x10 spinful lattice = 100 sites = 200 qubits (Jordan-Wigner)
    - hopping t = 1, on-site interaction U = 1
    - one fourth-order Suzuki step
    - delta = 1 Trotter step over total evolution time T = 1

Hamiltonian:
    H = -t sum_{<i,j>, sigma} (c^†_{i,sigma} c_{j,sigma} + h.c.)
        + U sum_i n_{i,up} n_{i,down}

Jordan-Wigner ordering: spin-block (all up qubits first, then all down qubits),
row-major within each block. With this layout the JW string for a horizontal
nearest-neighbour hop is empty, and for a vertical hop spans 9 qubits.

Construction pipeline:
    1. Build H as a SparsePauliOp with all JW-mapped hopping and on-site terms.
    2. Wrap it in a PauliEvolutionGate with SuzukiTrotter(order=4, reps=1).
    3. Transpile to the same basis used by the other benchmark scripts.
"""

from typing import Dict, List, Sequence, Tuple

from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.quantum_info import SparsePauliOp
from qiskit.synthesis import SuzukiTrotter


PauliTerm = Tuple[str, List[int], float]

ROWS = 10
COLS = 10
N_SITES = ROWS * COLS
NUM_QUBITS = 2 * N_SITES

T_HOPPING = 1.0
U_ONSITE = 8.0

TOTAL_EVOLUTION_TIME = 1.0
TROTTER_STEPS = 1
SUZUKI_ORDER = 4
DELTA_T = TOTAL_EVOLUTION_TIME / TROTTER_STEPS
DEFAULT_BASIS_GATES = ("cx", "t", "tdg", "rz", "sx", "s", "sdg", "h")


def benchmark_configuration() -> Dict[str, object]:
    """Return the paper configuration used for the benchmark circuit."""
    return {
        "reference": "arXiv:2604.19735",
        "model": "2D Fermi-Hubbard",
        "rows": ROWS,
        "cols": COLS,
        "num_sites": N_SITES,
        "num_qubits": NUM_QUBITS,
        "boundary_conditions": "open",
        "t_hopping": T_HOPPING,
        "U_onsite": U_ONSITE,
        "total_evolution_time": TOTAL_EVOLUTION_TIME,
        "trotter_steps": TROTTER_STEPS,
        "delta_t": DELTA_T,
        "suzuki_order": SUZUKI_ORDER,
        "jw_ordering": "spin-block (all up first, then all down), row-major within each block",
    }


def site_qubit(row: int, col: int, spin: int, rows: int = ROWS, cols: int = COLS) -> int:
    """Map (row, col, spin in {0=up, 1=down}) to its JW qubit index."""
    return spin * (rows * cols) + row * cols + col


def hopping_pauli_terms(p: int, q: int) -> List[PauliTerm]:
    """Return JW-mapped Pauli terms for c^†_p c_q + h.c. with coefficients 1/2 each.

    c^†_p c_q + h.c. = (1/2) [ X_p (prod Z) X_q + Y_p (prod Z) Y_q ].
    """
    if p > q:
        p, q = q, p
    string_qubits = list(range(p + 1, q))
    string_len = len(string_qubits)
    label_x = "X" + "Z" * string_len + "X"
    label_y = "Y" + "Z" * string_len + "Y"
    qubits = [p] + string_qubits + [q]
    return [
        (label_x, qubits, 0.5),
        (label_y, qubits, 0.5),
    ]


def build_hubbard_hamiltonian(
    rows: int = ROWS,
    cols: int = COLS,
    t: float = T_HOPPING,
    u: float = U_ONSITE,
) -> SparsePauliOp:
    """Build the 2D Fermi-Hubbard Hamiltonian under Jordan-Wigner."""
    n_qubits = 2 * rows * cols
    terms: List[PauliTerm] = []

    for spin in (0, 1):
        for r in range(rows):
            for c in range(cols - 1):
                p = site_qubit(r, c, spin, rows, cols)
                q = site_qubit(r, c + 1, spin, rows, cols)
                for label, qubits, coeff in hopping_pauli_terms(p, q):
                    terms.append((label, qubits, -t * coeff))
        for r in range(rows - 1):
            for c in range(cols):
                p = site_qubit(r, c, spin, rows, cols)
                q = site_qubit(r + 1, c, spin, rows, cols)
                for label, qubits, coeff in hopping_pauli_terms(p, q):
                    terms.append((label, qubits, -t * coeff))

    # On-site U n_up n_down. Using n = (I - Z) / 2 expands to
    #   U n_up n_down = (U/4) * (I - Z_up - Z_down + Z_up Z_down).
    # We drop the identity (global phase) and keep the three nontrivial terms.
    for r in range(rows):
        for c in range(cols):
            p_up = site_qubit(r, c, 0, rows, cols)
            p_dn = site_qubit(r, c, 1, rows, cols)
            # terms.append(("Z", [p_up], -u / 4.0))
            # terms.append(("Z", [p_dn], -u / 4.0))
            terms.append(("ZZ", [p_up, p_dn], u / 4.0 ))

    return SparsePauliOp.from_sparse_list(terms, num_qubits=n_qubits)



def build_fermihubbard_trotter_step(
    rows: int = ROWS,
    cols: int = COLS,
    tau: float = DELTA_T,
    order: int = SUZUKI_ORDER,
    reps: int = 1,
) -> QuantumCircuit:
    """Build one benchmark Trotter step with the paper configuration."""
    if rows != ROWS or cols != COLS:
        raise ValueError("This benchmark step is configured for the 200-qubit 10x10 spinful lattice.")

    hamiltonian = build_hubbard_hamiltonian(rows=rows, cols=cols)
    n_qubits = 2 * rows * cols

    synthesis = SuzukiTrotter(order=order, reps=reps)
    
    evo_gate = PauliEvolutionGate(hamiltonian, time=tau, synthesis=synthesis)
    # print(synthesis.expand(evo_gate))
    

    circuit = QuantumCircuit(n_qubits, name="fermi_hubbard_2d_step_s{0}".format(order))
    circuit.append(evo_gate, range(n_qubits))

    circuit.metadata = benchmark_configuration()
    circuit.metadata["delta_t"] = tau
    circuit.metadata["suzuki_order"] = order
    circuit.metadata["num_pauli_terms"] = len(hamiltonian)

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
    return gate_counts, total_gates, transpiled_circuit


if __name__ == "__main__":
    qc = build_fermihubbard_trotter_step()
    basis_gate_counts, total_basis_gates, transpiled_circuit = count_gates_in_basis(qc)
    rz_count = basis_gate_counts.get("rz", 0)
    t_count = rz_count * 100
    print("Estimated T count (assuming 100 rz gates per T):", t_count)
    # into powers of 10

    print("Built circuit:", qc.name)
    print("Qubits:", qc.num_qubits)
    print("Pauli terms in H:", qc.metadata["num_pauli_terms"])
    print("Delta t:", qc.metadata["delta_t"])
    print("Basis gates:", list(DEFAULT_BASIS_GATES))
    print("Basis gate counts:", basis_gate_counts)
    print("Total basis gates:", total_basis_gates)
    # print(build_hubbard_hamiltonian())

    from qiskit.qasm2 import dumps

    with open('ppr_circuits/fermi_hubbard_2d_step_s{0}.qasm'.format(SUZUKI_ORDER), 'w') as f:
        f.write(dumps(transpiled_circuit))
