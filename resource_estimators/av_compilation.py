import csv
import math
import re

rotate_sequence = ['z', 'z', 'z', 'z', 'z', 'z', 'z', 'z', 'z', 'y']

# Distillation active volume per T state: concatenated (15-to-1) x (8-to-CCZ)
# LS factory (Litinski AV), 35 blocks per CCZ + 16.5 blocks CCZ->2T conversion
# = 25.75 blocks per T state. These blocks live at the factory distance d_fac
# (App. tab:lsdist-configs), not the algorithm distance. Used for the benchmarks
# that need two-level distillation: FH / QPE-Abs at p = 1e-3.
C_T = 25.75

# Single-stage (15-to-1) factory: 35 blocks per 2-T factory = 17.5 blocks per T
# state, run in-fabric at the algorithm distance d (not d_fac). A single stage
# suffices wherever the T-error budget allows it: Stat-QPE / Stat-QPE-gap at both
# rates, and FH / QPE-Abs at p = 1e-4 (single-stage 35 p^3 covers n_T up to ~1e9).
C_T_SINGLE_STAGE = 17.5

# Distance-independent transversal-active-volume block counts per gate.
# A single-qubit Clifford occupies 1 patch for 1 code cycle; a transversal CX
# occupies 2 patches; a T gate carries the 15-to-1 distillation (133.5, active
# t-AV with dirty-state init free and buffer bus folded into injection) plus
# injection (3.5) block budget. Dividing the summed blocks by the code distance
# d recovers the per-gate t-AV costs used in the paper (1/d, 2/d, 137/d).
TAV_BLOCKS = {"s": 1, "sdg": 1, "h": 1, "cx": 2, "t": 137, "tdg": 137}

# 0-dist + trans-dist: zero-level distillation performed on the buffer-bus
# qubits via code conversion adds 15 patches x 13 code cycles = 195 blocks per
# T state (App. 0-dist+trans-dist: 133.5 + 195 = 328.5, +3.5 injection = 332).
# Charged for photonics (circuit noise model) at p = 1e-3 on the benchmarks
# whose T-error budget single-stage 15-to-1 cannot meet (FH, QPE-Abs).
ZERO_DIST_EXTRA_T_BLOCKS = 195

_GATE_RE = re.compile(r'"gate":\s*"([a-z]+)"')


def total_tav_blocks(json_file_path):
    """Sum distance-independent t-AV block counts over a transversal-logical-blocks
    JSON for one Trotter step.

    The file is streamed line-by-line (it is pretty-printed, one ``"gate": "x"``
    per line) so multi-hundred-MB benchmarks stay within memory budget. The
    returned block total is distance-independent; divide by the code distance d
    to obtain the t-AV in logical-block (qubit * logical-cycle) units.
    """
    total = 0
    with open(json_file_path) as file:
        for line in file:
            match = _GATE_RE.search(line)
            if match:
                total += TAV_BLOCKS.get(match.group(1), 0)
    return total


def total_tav_blocks_and_t_count(json_file_path):
    """Single-pass variant of total_tav_blocks that also returns the T-gate
    count, so callers can add per-T overheads (e.g. ZERO_DIST_EXTRA_T_BLOCKS)
    without streaming the multi-hundred-MB JSON a second time."""
    total = 0
    t_count = 0
    with open(json_file_path) as file:
        for line in file:
            match = _GATE_RE.search(line)
            if match:
                gate = match.group(1)
                total += TAV_BLOCKS.get(gate, 0)
                if gate in ("t", "tdg"):
                    t_count += 1
    return total, t_count

def active_volume_ppr(rotate_sequence):
    z_count = 0
    x_count = 0
    y_count = 0
    for i in range(len(rotate_sequence)):
        if rotate_sequence[i] == 'z':
            z_count += 1
        elif rotate_sequence[i] == 'x':
            x_count += 1
        elif rotate_sequence[i] == 'y':
            y_count += 1
    if y_count % 2 == 0 and y_count != 0: 
        if x_count != 0: 
            x_count = x_count + y_count
        if z_count != 0:
            z_count = z_count + y_count
    elif y_count % 2 == 1 and y_count != 0:
        if x_count != 0:
            x_count = x_count + y_count + 1
        if z_count != 0:
            z_count = z_count + y_count + 1
    
    C_m = math.floor(3/2 * x_count) + math.floor(3/2 * (z_count + 1)) + 1

    # PPM = math.floor(3/2 * x_count) + math.floor(3/2 * (z_count)) + 1
    # return PPM
    return C_m + 1.5 + C_T
    

def split_active_volume_from_csv(csv_file_path):
    """Return (clifford_blocks, n_rotations) for one Trotter step of the
    commuted PPR CSV: the C_m + 1.5 contribution summed over all rotations,
    and the rotation (T) count. The distillation contribution is
    n_rotations * C_T, kept separate so it can be converted to physical units
    at the factory distance d_fac rather than the algorithm distance."""
    clifford_blocks = 0.0
    n_rotations = 0
    with open(csv_file_path, 'r') as file:
        for row in csv.reader(file):
            if not row or len(row) < 2 or row[0] != 'rotate':
                continue
            rotation_sequence = [c for c in row[2:] if c and c != 'i']
            if rotation_sequence:
                n_rotations += 1
                clifford_blocks += active_volume_ppr(rotation_sequence) - C_T
    return clifford_blocks, n_rotations


def calculate_total_active_volume_from_csv(csv_file_path):
    """
    Calculate active volume for each row in the commuted CSV and sum them up.
    
    Args:
        csv_file_path (str): Path to the commuted CSV file
    
    Returns:
        int: Total active volume across all rows
    """
    total_active_volume = 0
    processed_count = 0
    active_volume_list = []
    with open(csv_file_path, 'r') as file:
        csv_reader = csv.reader(file)
        
        for row_num, row in enumerate(csv_reader, 1):
            # Skip empty rows
            if not row or len(row) < 2:
                continue
                
            # Check if this is a rotate operation
            if row[0] == 'rotate':
                # Extract rotation sequence from the row (skip first two elements: 'rotate' and angle)
                rotation_sequence = []
                for i in range(2, len(row)):
                    if row[i] and row[i] != 'i':  # Skip empty cells and 'i' (identity)
                        rotation_sequence.append(row[i])
                
                if rotation_sequence:  # Only process if we have a valid sequence
                    active_vol = active_volume_ppr(rotation_sequence) 
                    active_volume_list.append(active_vol)
                    total_active_volume += active_vol
                    processed_count += 1
                    
                    # Only print first 10 rotation sequences
                    # if processed_count <= 1000:
                        # print(f"Row {row_num}: {rotation_sequence} -> Active Volume: {active_vol}")
    # print maximum active volume
    # print(f"Maximum active volume: {max(active_volume_list)}")
    print(f"\nProcessed {processed_count} rotation sequences total.")
    return total_active_volume



