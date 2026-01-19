import csv
import math

rotate_sequence = ['z', 'z', 'z', 'z', 'z', 'z', 'z', 'z', 'z', 'y']

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
    C_T = 25

    # PPM = math.floor(3/2 * x_count) + math.floor(3/2 * (z_count)) + 1
    # return PPM
    return C_m + 1.5 + C_T
    

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



