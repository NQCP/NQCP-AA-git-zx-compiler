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

qasm_str = 'circacomp.qasm'
qc = QuantumCircuit.from_qasm_file(qasm_str)

# Create a new circuit with the same number of qubits and classical bits
new_qc = QuantumCircuit(qc.num_qubits, qc.num_clbits)

# Iterate through all operations in the original circuit
for circuit_instruction in qc.data:
    # Use the new API to get operation, qubits, and clbits
    instruction = circuit_instruction.operation
    qargs = circuit_instruction.qubits
    cargs = circuit_instruction.clbits
    
    # Check if the instruction is an RZ gate
    if instruction.name == 'rz':
        # Extract the rotation angle (theta) and qubit index
        theta = float(instruction.params[0])
        # Get the qubit index by finding its position in the circuit's qubits
        qubit = qc.find_bit(qargs[0])[0]
        # Decompose the RZ rotation
        rot_decompose(new_qc, theta, qubit)
    else:
        # For all other gates, convert qubits and clbits to indices and append
        # Convert qubits from old circuit to indices, then use new circuit's qubits
        new_qargs = [new_qc.qubits[qc.find_bit(qarg)[0]] for qarg in qargs]
        new_cargs = [new_qc.clbits[qc.find_bit(carg)[0]] for carg in cargs] if cargs else []
        new_qc.append(instruction, new_qargs, new_cargs)

# Dump the decomposed circuit to a new QASM file
output_qasm_str = dumps(new_qc)
with open('circacomp_decomposed.qasm', 'w') as f:
    f.write(output_qasm_str)

print(f"Decomposed circuit saved to circacomp_decomposed.qasm")
print(f"Original circuit had {qc.num_qubits} qubits and {len(qc.data)} operations")
print(f"New circuit has {new_qc.num_qubits} qubits and {len(new_qc.data)} operations")
