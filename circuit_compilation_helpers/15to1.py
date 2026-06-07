from qiskit import QuantumCircuit

qc = QuantumCircuit(16)

# code cycle 1
qc.h(0)
qc.h(1)
qc.h(2)
qc.h(4)
qc.h(8)

# code cycle 2
qc.cx(1, 9)
qc.cx(2, 10)
qc.cx(4, 12)

# code cycle 3
qc.cx(0, 4)
qc.cx(1, 5)
qc.cx(2, 6)

qc.cx(8, 12)
qc.cx(9, 13)
qc.cx(10, 14)

# code cycle 4
qc.cx(0, 2)
qc.cx(1, 3)

qc.cx(4, 6)
qc.cx(5, 7)

qc.cx(8, 10)
qc.cx(9, 11)

qc.cx(12, 14)
qc.cx(13, 15)

# code cycle 5
qc.cx(0, 1)
qc.cx(2, 3)
qc.cx(4, 5)
qc.cx(6, 7)
qc.cx(8, 9)
qc.cx(10, 11)
qc.cx(12, 13)
qc.cx(14, 15)
