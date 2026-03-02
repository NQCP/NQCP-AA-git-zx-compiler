from pygridsynth.gridsynth import gridsynth_gates
from pygridsynth.myplot import plot_sol
import mpmath
import numpy as np
from qiskit.circuit.library import UnitaryGate
from qiskit import QuantumCircuit
from qiskit.qasm2 import dumps

precision = "1e-3"
mpmath.mp.dps = 256

w = np.exp(1j*np.pi/4)
W = np.array([[w, 0], [0, w]])
W_gate = UnitaryGate(W, label='W')

def rot_decompose(qc, theta, qubit):
    epsilon = mpmath.mpmathify(str(precision))
    theta = mpmath.mpmathify(str(theta))
    gates = gridsynth_gates(theta=2*theta, epsilon=epsilon)

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

def logical_to_universal(path_to_qasm):
    
    with open(path_to_qasm, 'r') as f:
        qasm_str = f.read()
    qc = QuantumCircuit.from_qasm_str(qasm_str)
    qc_universal = QuantumCircuit(qc.num_qubits)
    for gate in qc.data:
        if gate[0].name == 'rz':
            rot_decompose(qc_universal, gate[0].params[0], gate[1][0])
        else:
            qc_universal.append(gate[0], gate[1])
    return qc_universal

# print(logical_to_universal('ppr_circuits/trotter_circuit_v2_logical.qasm'))
# save as qasm file 
qc_universal = logical_to_universal('ppr_circuits/trotter_circuit_v2_logical.qasm')
qasm_str = dumps(qc_universal)
with open('ppr_circuits/trotter_circuit_v2_universal.qasm', 'w') as f:
    f.write(qasm_str)