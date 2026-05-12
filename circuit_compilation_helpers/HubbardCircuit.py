"""Qiskit construction of one 2D Heisenberg benchmark Trotter step.

Reference:
    Sahil Khan et al., "Architecting Early Fault Tolerant Neutral Atoms
    Systems with Quantum Advantage", arXiv:2604.19735.

This file keeps the benchmark configuration explicit:
    - 50 qubits on a 5x10 lattice
    - isotropic couplings Jx = Jy = Jz = 1
    - one sixth-order Suzuki step
    - 300 total Trotter steps over total evolution time T = 50
    - open-boundary nearest-neighbour lattice, matching the 85-bond summary
      recorded in AGENT.md

The paper discusses commuting-group partitioning and simultaneous
diagonalization. Here we build the same logical unitary directly in Qiskit
using explicit XX, YY and ZZ evolutions per bond, without transpilation or
angle synthesis.
"""

from math import pow
from typing import Dict, List, Sequence, Tuple

from qiskit import QuantumCircuit, transpile

Bond = Tuple[int, int]
LayerSpec = Tuple[str, float]

ROWS = 5
COLS = 10
NUM_QUBITS = ROWS * COLS

JX = 1.0
JY = 1.0
JZ = 1.0

TOTAL_EVOLUTION_TIME = float(NUM_QUBITS)
TROTTER_STEPS = 300
SUZUKI_ORDER = 6
DELTA_T = TOTAL_EVOLUTION_TIME / TROTTER_STEPS
DEFAULT_BASIS_GATES = ("cx", "t", "tdg", "rz", "sx", "s", "sdg", "h")


def benchmark_configuration() -> Dict[str, object]:
    """Return the paper configuration used for the benchmark circuit."""
    return {
        "reference": "arXiv:2604.19735",
        "model": "2D Heisenberg",
        "rows": ROWS,
        "cols": COLS,
        "num_qubits": NUM_QUBITS,
        "boundary_conditions": "open",
        "couplings": {"Jx": JX, "Jy": JY, "Jz": JZ},
        "total_evolution_time": TOTAL_EVOLUTION_TIME,
        "trotter_steps": TROTTER_STEPS,
        "delta_t": DELTA_T,
        "suzuki_order": SUZUKI_ORDER,
    }


def lattice_index(row: int, col: int, cols: int = COLS) -> int:
    """Map a lattice site to its row-major qubit index."""
    return row * cols + col


def build_edge_matchings(rows: int = ROWS, cols: int = COLS) -> Dict[str, List[Bond]]:
    """Partition the 5x10 open lattice into the four commuting matchings.

    The labels follow the AGENT.md summary:
        A: odd-indexed horizontal bonds
        B: even-indexed horizontal bonds
        C: odd-indexed vertical bonds
        D: even-indexed vertical bonds

    We implement this with 1-based bond parity along each row/column.
    """
    matchings = {"A": [], "B": [], "C": [], "D": []}

    for row in range(rows):
        for col in range(cols - 1):
            left = lattice_index(row, col, cols)
            right = lattice_index(row, col + 1, cols)
            key = "A" if col % 2 == 0 else "B"
            matchings[key].append((left, right))

    for row in range(rows - 1):
        for col in range(cols):
            top = lattice_index(row, col, cols)
            bottom = lattice_index(row + 1, col, cols)
            key = "C" if row % 2 == 0 else "D"
            matchings[key].append((top, bottom))

    return matchings


def strang_layer_sequence(tau: float) -> List[LayerSpec]:
    """Return one symmetric second-order Suzuki step S2(tau)."""
    half_tau = tau / 2.0
    return [
        ("A", half_tau),
        ("B", half_tau),
        ("C", half_tau),
        ("D", tau),
        ("C", half_tau),
        ("B", half_tau),
        ("A", half_tau),
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


def apply_xx_evolution(circuit: QuantumCircuit, q0: int, q1: int, tau: float) -> None:
    """Apply exp[-i tau X⊗X]."""
    circuit.h(q0)
    circuit.h(q1)
    apply_zz_evolution(circuit, q0, q1, tau)
    circuit.h(q0)
    circuit.h(q1)


def apply_yy_evolution(circuit: QuantumCircuit, q0: int, q1: int, tau: float) -> None:
    """Apply exp[-i tau Y⊗Y]."""
    circuit.sdg(q0)
    circuit.sdg(q1)
    circuit.h(q0)
    circuit.h(q1)
    apply_zz_evolution(circuit, q0, q1, tau)
    circuit.h(q0)
    circuit.h(q1)
    circuit.s(q0)
    circuit.s(q1)


def apply_heisenberg_bond(
    circuit: QuantumCircuit,
    q0: int,
    q1: int,
    tau: float,
    jx: float = JX,
    jy: float = JY,
    jz: float = JZ,
) -> None:
    """Apply exp[-i tau (jx XX + jy YY + jz ZZ)] for one bond."""
    if jx != 0.0:
        apply_xx_evolution(circuit, q0, q1, jx * tau)
    if jy != 0.0:
        apply_yy_evolution(circuit, q0, q1, jy * tau)
    if jz != 0.0:
        apply_zz_evolution(circuit, q0, q1, jz * tau)


def apply_layer(
    circuit: QuantumCircuit,
    bonds: List[Bond],
    tau: float,
    jx: float = JX,
    jy: float = JY,
    jz: float = JZ,
) -> None:
    """Apply one commuting layer exp[-i H_X tau]."""
    for q0, q1 in bonds:
        apply_heisenberg_bond(circuit, q0, q1, tau, jx=jx, jy=jy, jz=jz)


def build_heisenberg_trotter_step(
    rows: int = ROWS,
    cols: int = COLS,
    tau: float = DELTA_T,
    order: int = SUZUKI_ORDER,
    include_barriers: bool = False,
) -> QuantumCircuit:
    """Build one benchmark Trotter step with the paper configuration."""
    if rows != ROWS or cols != COLS:
        raise ValueError("This benchmark step is configured for the 50-qubit 5x10 lattice.")

    matchings = build_edge_matchings(rows=rows, cols=cols)
    layer_sequence = suzuki_layer_sequence(order=order, tau=tau)
    circuit = QuantumCircuit(rows * cols, name="heisenberg_2d_step_s{0}".format(order))

    circuit.metadata = benchmark_configuration()
    circuit.metadata["delta_t"] = tau
    circuit.metadata["suzuki_order"] = order
    circuit.metadata["layer_sequence"] = [(label, angle) for label, angle in layer_sequence]
    circuit.metadata["num_layers"] = len(layer_sequence)

    for label, layer_tau in layer_sequence:
        apply_layer(circuit, matchings[label], layer_tau)
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
        optimization_level=0
    )
    gate_counts = dict(transpiled_circuit.count_ops())
    total_gates = sum(gate_counts.values())
    return gate_counts, total_gates


if __name__ == "__main__":
    qc = build_heisenberg_trotter_step()
    basis_gate_counts, total_basis_gates = count_gates_in_basis(qc)

    print("Built circuit:", qc.name)
    print("Qubits:", qc.num_qubits)
    print("Layers in unmerged Suzuki expansion:", qc.metadata["num_layers"])
    print("Delta t:", qc.metadata["delta_t"])
    print("Basis gates:", list(DEFAULT_BASIS_GATES))
    print("Basis gate counts:", basis_gate_counts)
    print("Total basis gates:", total_basis_gates)

    from qiskit.qasm2 import dumps 

    with open('ppr_circuits/heisenberg_2d_step_s{0}.qasm'.format(SUZUKI_ORDER), 'w') as f:
        f.write(dumps(qc))
