from pathlib import Path

from pygridsynth.gridsynth import gridsynth_gates
from pygridsynth.myplot import plot_sol
import mpmath
import numpy as np
from qiskit import QuantumCircuit, qasm3
from qiskit.circuit.library import UnitaryGate
from qiskit.exceptions import MissingOptionalLibraryError
from qiskit.qasm2 import dumps

precision = "1e-10"
mpmath.mp.dps = 256

w = np.exp(1j*np.pi/4)
W = np.array([[w, 0], [0, w]])
W_gate = UnitaryGate(W, label='W')

def rot_decompose(qc, theta, qubit):
    epsilon = mpmath.mpmathify(str(precision))
    theta = mpmath.mpmathify(str(theta))
    # gates = gridsynth_gates(theta=2*theta, epsilon=epsilon)
    gates = gridsynth_gates(theta=theta, epsilon=epsilon) # claude improve tmm

    gate_dcomp_list = gates    
    gate_dcomp_list = gate_dcomp_list[::-1]

    for gate in gate_dcomp_list:
        if gate == 'H':
            qc.h(qubit)
        elif gate == 'T':
            qc.t(qubit)
        elif gate == 'S':
            qc.s(qubit)
        elif gate == 'W':
            # qc.append(W_gate, [qubit])
            pass
        elif gate == 'X':
            qc.x(qubit)
        elif gate == 'Y':
            qc.y(qubit)
        elif gate == 'Z':
            qc.z(qubit)


def _is_openqasm3_header(line: str) -> bool:
    s = line.strip().upper()
    if not s.startswith("OPENQASM"):
        return False
    # e.g. OPENQASM 3.0; or OPENQASM 3;
    return "OPENQASM 3" in s or s.startswith("OPENQASM3")


def circuit_from_qasm_file(path_to_qasm: str) -> QuantumCircuit:
    """Load OpenQASM 2 or 3 into a QuantumCircuit.

    QASM 3 uses ``qiskit.qasm3.load`` when ``qiskit_qasm3_import`` is installed,
    otherwise the experimental importer (may emit ExperimentalWarning).
    """
    path = Path(path_to_qasm)
    text = path.read_text()
    is_v3 = False
    for raw in text.splitlines():
        stripped = raw.strip()
        if not stripped or stripped.startswith("//"):
            continue
        if _is_openqasm3_header(stripped):
            is_v3 = True
        break
    if is_v3:
        try:
            return qasm3.load(path)
        except MissingOptionalLibraryError:
            return qasm3.load_experimental(path)
    return QuantumCircuit.from_qasm_str(text)


def logical_to_universal(path_to_qasm):
    qc = circuit_from_qasm_file(path_to_qasm)
    qc_universal = QuantumCircuit(qc.num_qubits)
    for gate in qc.data:
        if gate[0].name == 'rz':
            rot_decompose(qc_universal, gate[0].params[0], gate[1][0])
        else:
            qc_universal.append(gate[0], gate[1])
    return qc_universal

# print(logical_to_universal('ppr_circuits/trotter_circuit_v2_logical.qasm'))
# save as qasm file 
qc_universal = logical_to_universal('ppr_circuits/fermi_hubbard_2d_step_s4.qasm')
qasm_str = dumps(qc_universal)
with open('ppr_circuits/fermi_hubbard_2d_step_s4_universal.qasm', 'w') as f:
    f.write(qasm_str)