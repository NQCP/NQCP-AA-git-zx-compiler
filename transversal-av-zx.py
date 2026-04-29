"""
Parse QASM file and generate transversal-logical-blocks.json.
Each sequence is a single gate (s, t, h, cx). Pauli x, y, z are ignored.
Circles have up/down ports (qubit indices). Two-qubit gates have two circles with 'connect' linking them.
"""
import re
import json

cultivation = False
distillation = True

# Gates to ignore (Pauli operators)
IGNORE_GATES = {'x', 'y', 'z'}

# Active volume per gate type (T_AV can be changed)
if distillation:
    d = 17 # heisenberg is 15
    time_to_distill = 117.5/d
    time_to_inject = 4.5/d  
    ACTIVE_VOLUME = {'s': 1/d, 
                     't': time_to_distill + time_to_inject, 
                     'h': 1/d, 
                     'cx': 2/d, 
                     'sdg': 1/d}

# cultivation assumptions:
if cultivation:
    time_to_cultivate = 14.3/d # code cycles
    time_to_inject = 4.5/d # code cyles 
    qubits_per_factory = 787
    ACTIVE_VOLUME = {'s': 1/d, 
                     't': (time_to_cultivate + time_to_inject), 
                     'h': 1/d, 
                     'cx': 2/d, 
                     'sdg': 1/d}



def parse_qasm_line(line):
    """Parse a QASM gate line. Returns (gate_name, qubits) or None if not a gate line."""
    line = line.strip()
    if not line or line.startswith('//') or line.startswith('OPENQASM') or line.startswith('include') or line.startswith('qreg'):
        return None

    parts = line.split()
    if not parts:
        return None

    gate = parts[0].lower().rstrip(';')
    if gate in IGNORE_GATES:
        return None

    if gate not in {'s', 't', 'h', 'cx', 'sdg'}:
        return None

    qubit_part = ' '.join(parts[1:]).rstrip(';')
    qubits = [int(m) for m in re.findall(r'q\[(\d+)\]', qubit_part)]

    return (gate, qubits)


def gate_to_sequence(gate, qubits, seq_id):
    """Convert a gate to the transversal logical block format (circles with up/down/connect ports)."""
    if gate == 'cx':
        control, target = qubits[0], qubits[1]
        circles = [
            {
                "index": 0,
                "ports": {
                    "up": control,
                    "down": control,
                    "connect": target
                }
            },
            {
                "index": 1,
                "ports": {
                    "up": target,
                    "down": target,
                    "connect": control
                }
            }
        ]
    else:
        # Single qubit gate: s, t, h, sdg
        q = qubits[0]
        circles = [
            {
                "index": 0,
                "ports": {
                    "up": q,
                    "down": q,
                    "connect": None
                }
            }
        ]

    phase = "pi/2" if gate == 's' or gate == 'sdg' else ""
    active_volume = ACTIVE_VOLUME.get(gate, 1)

    return {
        "sequence_id": seq_id,
        "gate": gate,
        "phase": phase,
        "active_volume": active_volume,
        "circles": circles
    }


def qasm_to_transversal_blocks(qasm_path, output_path='transversal-logical-blocks-heisenberg.json'):
    """Parse QASM file and write transversal-logical-blocks.json."""
    sequences = []
    seq_idx = 0

    with open(qasm_path, 'r') as f:
        for line in f:
            parsed = parse_qasm_line(line)
            if parsed is None:
                continue

            gate, qubits = parsed
            seq = gate_to_sequence(gate, qubits, f'seq_{seq_idx:04d}')
            sequences.append(seq)
            seq_idx += 1

    with open(output_path, 'w') as f:
        json.dump(sequences, f, indent=2)

    print(f"Wrote {len(sequences)} sequences to {output_path}")
    return sequences


if __name__ == '__main__':
    sequences = qasm_to_transversal_blocks('ppr_circuits/heisenberg_2d_step_s6_universal.qasm')
    total_av = sum(s['active_volume'] for s in sequences)
    print(f"Sum of active volume: {total_av}")
