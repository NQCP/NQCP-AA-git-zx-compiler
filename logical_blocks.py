import pyzx as zx
import numpy as np
import networkx as nx
import json
import os
import csv
import gzip

def active_volume_exact(sequence):
    z_count = 0
    x_count = 0
    y_count = 0
    for i in range(len(sequence)):
        if sequence[i] == 'z':
            z_count += 1
        elif sequence[i] == 'x':
            x_count += 1
        elif sequence[i] == 'y':
            z_count  += 1
            x_count += 1
            y_count += 1
    if y_count % 2 != 0: 
        x_count += 1
        z_count += 1
    return np.ceil(3/2 * x_count) + np.ceil(3/2 * z_count) + 1


class Hexagon:
    def __init__(self, index):
        self.index = index
        self.ports = {d: None for d in ['up', 'down', 'north', 'south', 'east', 'west']}

    def connect(self, direction, neighbor_index):
        self.ports[direction] = neighbor_index



def process_sequence_to_hexagons(sequence_input):
    """
    Process a sequence and return hexagon data.
    
    Args:
        sequence_input: List of gate types (e.g., ['i', 'z', 'z', 'y', ...])
    
    Returns:
        List of dictionaries with hexagon data (index and ports)
    """
    # Make a copy 
    sequence = sequence_input.copy()
    
    # y count calculation 
    y_count = 0
    for i in sequence:
        if i == 'y':
            y_count += 1
    if y_count%2 != 0:
        sequence.append('rs')
    
    sequence.append('anc')
    
    cnot_edges_zy = []
    cnot_edges_xy = []
    # add edges between all the 'z' vertices an ancilla and 'y' and ancilla
    for i in range(len(sequence)):
        if sequence[i] == 'z' or sequence[i] == 'y' or sequence[i] == 'rs':
            cnot_edges_zy.append((i, len(sequence)-1))
        if sequence[i] == 'x' or sequence[i] == 'y' or sequence[i] == 'rs':
            cnot_edges_xy.append((len(sequence)-1, i))
    
    # one node should not be connected by more that 4 edges     
    c = zx.Circuit(len(sequence))
    for i in range(len(cnot_edges_zy)):
        c.add_gate('CNOT', cnot_edges_zy[i][0], cnot_edges_zy[i][1])
    c.add_gate('H', len(sequence)-1)
    for i in range(len(cnot_edges_xy)):
        c.add_gate('CNOT', cnot_edges_xy[i][0], cnot_edges_xy[i][1])
    g = c.to_basic_gates().to_graph()
    
    ancilla_nodes_left  = np.floor((len(cnot_edges_zy) + 1)/2)
    ancilla_nodes_right = np.floor((len(cnot_edges_xy) + 1)/2)
    qubit_nodes_left = len(cnot_edges_zy)
    qubit_nodes_right = len(cnot_edges_xy)
    connector_node = 1
    total_nodes = ancilla_nodes_left + ancilla_nodes_right + qubit_nodes_left + qubit_nodes_right + connector_node
    
    G = nx.Graph()
    G.add_nodes_from(range(int(total_nodes)))
    
    # add labels to the ancilla nodes left
    for i in range(int(ancilla_nodes_left)):
        G.nodes[i]['label'] = 'blue ancilla'
        G.nodes[i]['color'] = 'blue'
    
    # add labels to the ancilla nodes right
    for i in range(int(ancilla_nodes_left), int(ancilla_nodes_left + ancilla_nodes_right)):
        G.nodes[i]['label'] = 'yellow ancilla'
        G.nodes[i]['color'] = 'yellow'
    
    # add labels to the qubit nodes left
    for i in range(int(ancilla_nodes_left + ancilla_nodes_right), int(ancilla_nodes_left + ancilla_nodes_right + qubit_nodes_left)):
        G.nodes[i]['label'] = 'yellow qubit'
        G.nodes[i]['color'] = 'yellow'
        G.nodes[i]['name'] = 'q' + str(cnot_edges_zy[i-int(ancilla_nodes_left + ancilla_nodes_right)][0])
    
    # add labels to the qubit nodes right
    for i in range(int(ancilla_nodes_left + ancilla_nodes_right + qubit_nodes_left), int(ancilla_nodes_left + ancilla_nodes_right + qubit_nodes_left + qubit_nodes_right)):
        G.nodes[i]['label'] = 'blue qubit'
        G.nodes[i]['color'] = 'blue'
        G.nodes[i]['name'] = 'q' + str(cnot_edges_xy[i-int(ancilla_nodes_left + ancilla_nodes_right + qubit_nodes_left)][1])
    
    # add labels to the connector node and color it orange
    connector_idx = int(ancilla_nodes_left + ancilla_nodes_right + qubit_nodes_left + qubit_nodes_right)
    G.nodes[connector_idx]['label'] = 'connector'
    G.nodes[connector_idx]['color'] = 'orange'

     # Connect blue ancillas to each other
    # for i in range(int(ancilla_nodes_left) - 1):
    #     G.add_edge(i, i+1)
    
    # # Connect last blue ancilla to connector
    # G.add_edge(int(ancilla_nodes_left) - 1, connector_idx, color='red')
    
    # Connect blue ancillas to each other (only when there are blue ancillas)
    if ancilla_nodes_left > 0:
        for i in range(int(ancilla_nodes_left) - 1):
            G.add_edge(i, i+1)
        # Connect last blue ancilla to connector
        G.add_edge(int(ancilla_nodes_left) - 1, connector_idx, color='red')
    
    # Connect yellow ancillas to each other and finally to the connector
    start_yellow = int(ancilla_nodes_left)
    end_yellow = int(ancilla_nodes_left + ancilla_nodes_right)
    for i in range(start_yellow, end_yellow - 1):
        G.add_edge(i, i+1)
    if ancilla_nodes_right > 0:
        G.add_edge(end_yellow - 1, connector_idx)
    
    # Connect each left ancilla node to 2 yellow qubit nodes (one-to-two mapping)
    yellow_qubit_start = int(ancilla_nodes_left + ancilla_nodes_right)
    for i in range(int(ancilla_nodes_left)):
        yq1 = yellow_qubit_start + 2 * i
        yq2 = yellow_qubit_start + 2 * i + 1
        if yq1 < yellow_qubit_start + int(qubit_nodes_left):
            G.add_edge(i, yq1)
        if yq2 < yellow_qubit_start + int(qubit_nodes_left):
            G.add_edge(i, yq2)
    
    # Connect each right ancilla node to 2 blue qubit nodes (one-to-two mapping)
    blue_qubit_start = int(ancilla_nodes_left + ancilla_nodes_right + qubit_nodes_left)
    for i in range(int(ancilla_nodes_right)):
        bq1 = blue_qubit_start + 2 * i
        bq2 = blue_qubit_start + 2 * i + 1
        if bq1 < blue_qubit_start + int(qubit_nodes_right):
            G.add_edge(i + int(ancilla_nodes_left), bq1)
        if bq2 < blue_qubit_start + int(qubit_nodes_right):
            G.add_edge(i + int(ancilla_nodes_left), bq2)
    
    # Create hexagons
    direction = ['up', 'down', 'north', 'south', 'east', 'west']
    sequence_i_want = []
    for i in G.nodes:
        n_neighbours = list(G.neighbors(i))
        hexagon = Hexagon(i)
        name = G.nodes[i]['name'] if 'name' in G.nodes[i] else None
        if len(n_neighbours) == 1 and i <= len(G.nodes)//2:
            hexagon.connect('up', name)
            hexagon.connect('down', name)
            hexagon.connect('south', n_neighbours[0])
        elif len(n_neighbours) == 1 and i > len(G.nodes)//2:
            hexagon.connect('up', name)
            hexagon.connect('down', name)
            hexagon.connect('north', n_neighbours[0])
        elif len(n_neighbours) == 2 and i < len(G.nodes)//2:
            hexagon.connect('north', n_neighbours[0])
            hexagon.connect('south', n_neighbours[1])
        elif len(n_neighbours) == 2 and i > len(G.nodes)//2:
            hexagon.connect('south', n_neighbours[0])
            hexagon.connect('north', n_neighbours[1])
        elif len(n_neighbours) == 3 and i < len(G.nodes)//2:
            hexagon.connect('east', n_neighbours[0])
            hexagon.connect('north', n_neighbours[1])
            hexagon.connect('south', n_neighbours[2])
        elif len(n_neighbours) == 3 and i > len(G.nodes)//2:
            hexagon.connect('west', n_neighbours[0])
            hexagon.connect('south', n_neighbours[1])
            hexagon.connect('north', n_neighbours[2])
        elif len(n_neighbours) == 4 and i < len(G.nodes)//2:
            hexagon.connect('west', n_neighbours[0])
            hexagon.connect('east', n_neighbours[1])
            hexagon.connect('north', n_neighbours[2])
            hexagon.connect('south', n_neighbours[3])
        elif len(n_neighbours) == 4 and i > len(G.nodes)//2:
            hexagon.connect('east', n_neighbours[0])
            hexagon.connect('west', n_neighbours[1])
            hexagon.connect('south', n_neighbours[2])
            hexagon.connect('north', n_neighbours[3])
        sequence_i_want.append(hexagon)
        if i == len(G.nodes)//2 + 1:
            hexagon.connect('up', name)
    
    # Build a map of up node names to all hexagons referencing that 'up'
    up_map = {}
    for hexagon in sequence_i_want:
        up_val = hexagon.ports.get('up')
        if up_val is not None:
            up_map.setdefault(up_val, []).append(hexagon)
    
    # For each up value that appears twice, apply the replacements as specified
    for up_val, hexagons in up_map.items():
        if up_val is not None and len(hexagons) == 2:
            if 'down' in hexagons[0].ports and hexagons[0].ports['down'] == up_val:
                hexagons[0].ports['down'] = 'b' + str(hexagons[0].index)
            if 'up' in hexagons[1].ports and hexagons[1].ports['up'] == up_val:
                hexagons[1].ports['up'] = 'b' + str(hexagons[0].index)
    
    # Convert to list of dictionaries for JSON serialization
    hexagon_data = []
    for hexagon in sequence_i_want:
        hexagon_data.append({
            'index': hexagon.index,
            'ports': hexagon.ports
        })
    
    # Convert total_nodes to int for JSON serialization
    active_volume = int(total_nodes)

    # verify that the active volume matches the expected active volume
    # get the sequence
    
    expected_active_volume = active_volume_exact(sequence_input)
    if active_volume != expected_active_volume:
        print(f"Active volume mismatch: {active_volume} != {expected_active_volume}")
        print(f"Sequence: {sequence_input}")
        print(f"Hexagons: {hexagon_data}")
        raise ValueError(f"Active volume mismatch: {active_volume} != {expected_active_volume}")
    
    return hexagon_data, active_volume


def save_sequences_to_json(sequences_data, filename='sequences_data.json'):
    """
    Save multiple sequences to a JSON file.
    
    Args:
        sequences_data: List of dictionaries, each containing:
            - 'sequence_id': (optional) identifier for the sequence
            - 'input_sequence': original input sequence
            - 'hexagons': list of hexagon data from process_sequence_to_hexagons()
        filename: Output JSON filename
    """
    with open(filename, 'w') as f:
        json.dump(sequences_data, f, indent=2)
    print(f"Saved {len(sequences_data)} sequences to {filename}")


def load_sequences_from_json(filename='sequences_data.json'):
    """
    Load multiple sequences from a JSON file.
    
    Args:
        filename: Input JSON filename
    
    Returns:
        List of sequence dictionaries
    """
    if not os.path.exists(filename):
        print(f"File {filename} not found. Returning empty list.")
        return []
    
    with open(filename, 'r') as f:
        sequences_data = json.load(f)
    print(f"Loaded {len(sequences_data)} sequences from {filename}")
    return sequences_data


def append_sequence_to_json(sequence_input, sequence_id=None, filename='sequences_data.json'):
    """
    Process a sequence and append it to an existing JSON file (or create new one).
    
    Args:
        sequence_input: List of gate types (e.g., ['i', 'z', 'z', 'y', ...])
        sequence_id: Optional identifier for this sequence (defaults to index)
        filename: JSON filename to append to
    """
    # Load existing sequences or create new list
    sequences_data = load_sequences_from_json(filename)
    if sequences_data is None:
        sequences_data = []
    
    # Process the sequence
    hexagons, active_volume = process_sequence_to_hexagons(sequence_input)
    
    # Create sequence entry
    if sequence_id is None:
        sequence_id = len(sequences_data)
    
    sequence_entry = {
        'sequence_id': sequence_id,
        'input_sequence': sequence_input,
        'active_volume': active_volume,
        'hexagons': hexagons
    }
    
    sequences_data.append(sequence_entry)
    
    # Save back to file
    save_sequences_to_json(sequences_data, filename)


if __name__ == '__main__':
    csv_file = 'ppr_circuits/fermi_hubbard_2d_step_s4_universal_paulis_commuted.csv'
    out_file = 'logical_blocks_fermi_hubbard_2d_step_s4_universal_paulis_commuted.jsonl.gz'

    print(f"Reading sequences from {csv_file}...")
    print(f"Streaming output to {out_file}...")

    n_written = 0
    n_errors = 0
    with open(csv_file, 'r') as fin, gzip.open(out_file, 'wt') as fout:
        reader = csv.reader(fin)
        for row_idx, row in enumerate(reader):
            if not row or len(row) < 3:
                continue

            sequence = [gate.strip() for gate in row[2:] if gate.strip() != '']
            if not sequence:
                continue

            try:
                hexagons, active_volume = process_sequence_to_hexagons(sequence)
            except Exception as e:
                n_errors += 1
                print(f"Error processing sequence {row_idx}: {e}")
                print(f"  Sequence: {sequence}")
                import traceback
                traceback.print_exc()
                continue

            entry = {
                'sequence_id': f'seq_{row_idx:04d}',
                'input_sequence': sequence,
                'active_volume': active_volume,
                'hexagons': hexagons,
            }
            fout.write(json.dumps(entry, separators=(',', ':')) + '\n')
            n_written += 1

            if n_written % 5000 == 0:
                print(f"  Written {n_written} sequences...")

    print(f"\nDone! Wrote {n_written} sequences to {out_file} ({n_errors} errors).")
