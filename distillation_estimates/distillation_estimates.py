import numpy as np 
from qiskit.quantum_info import DensityMatrix, Statevector, Operator, Pauli
from qiskit import QuantumCircuit


rotation_list = [('I', 'Z', 'I', 'I', 'I'), # 1
                 ('I', 'I', 'Z', 'I', 'I'), # 2
                 ('I', 'I', 'I', 'Z', 'I'), # 3
                 ('I', 'I', 'I', 'I', 'Z'), # 4
                 ('I', 'Z', 'Z', 'Z', 'I'), # 5
                 ('Z', 'Z', 'Z', 'I', 'I'), # 6
                 ('Z', 'Z', 'I', 'Z', 'I'), # 7
                 ('Z', 'I', 'Z', 'Z', 'I'), # 8
                 ('Z', 'I', 'I', 'Z', 'Z'), # 9
                 ('Z', 'Z', 'I', 'I', 'Z'), # 10
                 ('Z', 'I', 'Z', 'I', 'Z'), # 11
                 ('Z', 'Z', 'Z', 'Z', 'Z'), # 12
                 ('I', 'I', 'Z', 'Z', 'Z'), # 13
                 ('I', 'Z', 'I', 'Z', 'Z'), # 14
                 ('I', 'Z', 'Z', 'I', 'Z')] # 15 
    
# write 10^(-4) as a float
p = 10**(-3)

circuit = QuantumCircuit(5)
circuit.h(range(5))

rho_init = DensityMatrix.from_instruction(circuit)


def pauli_product_rot(rotation_list, index):
    """
    Build a Pauli product operator from a tuple like ('I', 'Z', 'I', 'I', 'I').
    Returns the matrix representation of the tensor product.
    """
    # Define single-qubit Pauli matrices
    I = np.array([[1, 0], [0, 1]])
    Z = np.array([[1, 0], [0, -1]])
    
    # Map Pauli labels to matrices
    pauli_map = {'I': I, 'Z': Z}
    
    # Start with the first qubit's Pauli matrix
    pauli_string = rotation_list[index]
    op = pauli_map[pauli_string[0]]
    
    # Kronecker product with each subsequent qubit
    for i in range(1, len(pauli_string)):
        op = np.kron(op, pauli_map[pauli_string[i]])
    
    return op

# Alternative using Qiskit's Pauli class (cleaner approach)
def pauli_product_qiskit(rotation_list, index):
    """
    Build a Pauli product operator using Qiskit's Pauli class.
    Returns the Operator object.
    """
    pauli_string = ''.join(rotation_list[index])
    pauli = Pauli(pauli_string)
    return Operator(pauli)

def pauli_rotation_matrix(rotation_list, index, phi):
    """
    Compute the matrix form of P_phi = e^(-i*P*phi) where:
    - P is the Pauli product operator from rotation_list[index]
    - phi is the rotation angle (in radians)
    
    For Pauli operators: e^(-i*P*phi) = cos(phi)*I - i*sin(phi)*P
    
    Returns: numpy array (complex matrix)
    """
    # Get the Pauli operator P
    P = pauli_product_rot(rotation_list, index)
    
    # Get the identity matrix of the same dimension
    n = P.shape[0]
    I = np.eye(n, dtype=complex)
    
    # Compute e^(-i*P*phi) = cos(phi)*I - i*sin(phi)*P
    P_phi = np.cos(phi) * I - 1j * np.sin(phi) * P
    
    return P_phi

def pauli_rotation_qiskit(rotation_list, index, phi):
    """
    Compute P_phi = e^(-i*P*phi) using Qiskit's Operator.
    Returns: Operator object
    """
    # Get the Pauli operator P
    P_op = pauli_product_qiskit(rotation_list, index)
    P_matrix = P_op.data
    
    # Get the identity matrix
    n = P_matrix.shape[0]
    I = np.eye(n, dtype=complex)
    
    # Compute e^(-i*P*phi) = cos(phi)*I - i*sin(phi)*P
    P_phi_matrix = np.cos(phi) * I - 1j * np.sin(phi) * P_matrix
    
    return Operator(P_phi_matrix)


def distillation_protocol(rho_init, p):
    rho = rho_init
    for i in range(len(rotation_list)):
        P_Pi8 = pauli_rotation_qiskit(rotation_list, i, np.pi / 8)
        P_5Pi8 = pauli_rotation_qiskit(rotation_list, i, 5 * np.pi / 8)
        rho_mapped = (1 - p) * rho.evolve(P_Pi8) + p * (rho.evolve(P_5Pi8))
        rho = rho_mapped
    return rho

rho = distillation_protocol(rho_init, p)

def projector_Pi_X():
    """
    Returns the operator I ⊗ (Π_x ⊗ Π_x ⊗ Π_x ⊗ Π_x)
    where Π_x = (I + X)/2 is the projector onto the |+> state.
    """
    I = np.array([[1, 0], [0, 1]], dtype=complex)
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Pi_x = (I + X) / 2
    
    # Compute Π_x ⊗ Π_x ⊗ Π_x ⊗ Π_x (tensor product of 4 copies)
    Pi_x_4 = Pi_x
    for _ in range(3):  # Already have one copy, need 3 more
        Pi_x_4 = np.kron(Pi_x_4, Pi_x)
    
    # Compute I ⊗ (Π_x ⊗ Π_x ⊗ Π_x ⊗ Π_x)
    result = np.kron(I, Pi_x_4)
    
    return Operator(result)

p_fail = 1 - rho.evolve(projector_Pi_X()).trace()
print('p_fail =', p_fail)
rho_out = (1/(1-p_fail)) * rho.evolve(projector_Pi_X())


def construct_rho_ideal():
    """
    Construct rho_ideal = |m⟩⟨m| ⊗ (|+⟩⟨+|)^⊗4
    where |m⟩ = (|0⟩ + e^(-iπ/4)|1⟩)/√2
    """
    # Define basis states
    ket_0 = np.array([1, 0], dtype=complex)
    ket_1 = np.array([0, 1], dtype=complex)
    
    # Construct |m⟩ = (|0⟩ + e^(-iπ/4)|1⟩)/√2
    phase = np.exp(-1j * np.pi / 4)
    ket_m = (ket_0 + phase * ket_1) / np.sqrt(2)
    
    # Compute |m⟩⟨m| (outer product)
    rho_m = np.outer(ket_m, ket_m.conj())
    
    # Construct |+⟩⟨+| = (I + X)/2
    I = np.array([[1, 0], [0, 1]], dtype=complex)
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Pi_plus = (I + X) / 2
    
    # Compute (|+⟩⟨+|)^⊗4
    Pi_plus_4 = Pi_plus
    for _ in range(3):  # Already have one copy, need 3 more
        Pi_plus_4 = np.kron(Pi_plus_4, Pi_plus)
    
    # Compute |m⟩⟨m| ⊗ (|+⟩⟨+|)^⊗4
    rho_ideal_matrix = np.kron(rho_m, Pi_plus_4)
    
    return DensityMatrix(rho_ideal_matrix)

rho_ideal = construct_rho_ideal()

# Compute p_out = 1 - tr(rho_ideal * rho_out)
# This is the complement of the overlap between rho_ideal and rho_out
overlap = np.trace(rho_ideal.data @ rho_out.data)
p_out = 1 - overlap
print(f"p_out = {p_out}")