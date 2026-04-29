from qiskit import QuantumCircuit, QuantumRegister
import pyzx as zx
import matplotlib.pyplot as plt

clock_cycle_s = 1
distance = 7

def distillation_circuit(n_qubits):
    qreg = QuantumRegister(n_qubits)
    qc = QuantumCircuit(qreg)

    qc.cx(1, 9)
    qc.cx(2, 10)
    qc.cx(4, 12)
    ##
    qc.cx(0, 4)
    qc.cx(1, 5)
    qc.cx(2, 6)

    qc.cx(8, 12)
    qc.cx(9, 13)
    qc.cx(10, 14)
    ##
    qc.cx(0, 2)
    qc.cx(1, 3)
    qc.cx(4, 6)
    qc.cx(5, 7)
    qc.cx(8, 10)
    qc.cx(9, 11)
    qc.cx(12, 14)
    qc.cx(13, 15)
    ##
    qc.cx(0, 1)
    qc.cx(2, 3)
    qc.cx(4, 5)
    qc.cx(6, 7)
    qc.cx(8, 9)
    qc.cx(10, 11)
    qc.cx(12, 13)
    qc.cx(14, 15)
    
    return qc

def compute_av_distillation_circuit(n_qubits):
    qc = distillation_circuit(n_qubits)
    # get cnot count
    cnot_count = 0
    for gate in qc.data:
        if gate[0].name == 'cx':
            cnot_count += 1
    av_cnots = (cnot_count * 2)/distance
    print(f'av_cnots: {av_cnots}')
    t_states = n_qubits - 1
    s_av = (1*clock_cycle_s/distance) * t_states
    s_correction_av = s_av/2
    total_av = av_cnots + s_correction_av + t_states
    return total_av

av = compute_av_distillation_circuit(16)
print(f'av: {av}')



