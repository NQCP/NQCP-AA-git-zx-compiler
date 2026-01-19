import pyzx as zx
import matplotlib.pyplot as plt
# rotate,-8,z,x,x,x,z,y,y,y,x,y,
sequence  = ['z', 'x', 'x', 'x', 'x', 'z', 'y', 'y', 'y', 'x', 'y', 'rs', 'rs', 'rs', 'ancilla']
# we need 10 + 3 (y resource states) + 1 (ancilla) = 14 qubits 
cnot_edges_zy = []
cnot_edges_x = []
# add edges between all the 'z' vertices an ancilla and 'y' and ancilla
for i in range(len(sequence)):
    if sequence[i] == 'z' or sequence[i] == 'y' or sequence[i] == 'rs':

        cnot_edges_zy.append((i, len(sequence)-1))
    if sequence[i] == 'x' or sequence[i] == 'y':
        cnot_edges_x.append((len(sequence)-1, i))


print(cnot_edges_zy)
print(cnot_edges_x)

def pauli_product_rot(sequence):
    g = zx.Graph()

    # add boundary vertices
    for i in range(len(sequence)):
        g.add_vertex(zx.VertexType.BOUNDARY, i)

    # add edges between all the 'z' vertices an ancilla and 'y' and ancilla
    for i in range(len(sequence)):
        ctrl = g.add_vertex(zx.VertexType.Z, i, i)
        
        if sequence[i] == 'z' or sequence[i] == 'y' or sequence[i] == 'rs':
            target = g.add_vertex(zx.VertexType.X, len(sequence)-1, i)
            g.add_edge((ctrl, target))
        if sequence[i] == 'x' or sequence[i] == 'y':
            target = g.add_vertex(zx.VertexType.Z, len(sequence)-1, i)
            g.add_edge((target, ctrl))

    return g

fig = zx.draw_matplotlib(pauli_product_rot(sequence))
# save the figure
fig.savefig('pauli_product_rot.png')




