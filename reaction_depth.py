import json
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from collections import deque

# 35 full logical blocks produce 2 msd per cycle 
# Constants
# TOTAL_LOGICAL_BLOCKS = 86  # Quantum computer capacity
# T_STATE_OVERHEAD = 35 / 4  # Logical blocks required per T state (17.5)
T_STATE_OVERHEAD = 0
TOTAL_LOGICAL_BLOCKS = 25  #(here i am assuming, i am already using 35 blocks for msd production)
LOGICAL_BLOCKS_FILE = 'logical_blocks.json'


def load_sequences(filename=LOGICAL_BLOCKS_FILE):
    """Load sequences from JSON file."""
    with open(filename, 'r') as f:
        sequences = json.load(f)
    print(f"Loaded {len(sequences)} sequences from {filename}")
    return sequences


def calculate_sequence_cost(sequence):
    """
    Calculate total logical blocks needed for a sequence.
    
    Args:
        sequence: Dictionary with 'active_volume' field
    
    Returns:
        Total logical blocks = active_volume + T_state_overhead
    """
    active_volume = sequence['active_volume']
    total_cost = active_volume + T_STATE_OVERHEAD
    return total_cost


def schedule_sequences(sequences, total_capacity=TOTAL_LOGICAL_BLOCKS):
    """
    Schedule sequences to run in parallel, respecting logical block constraints.
    
    Args:
        sequences: List of sequence dictionaries
        total_capacity: Total available logical blocks
    
    Returns:
        List of (cycle, num_parallel_ops) tuples
    """
    # Calculate cost for each sequence (process in JSON order, no reordering)
    sequence_costs = [(i, calculate_sequence_cost(seq)) for i, seq in enumerate(sequences)]

    # Track scheduling
    schedule = []  # List of (cycle, num_parallel_ops, used_blocks)
    remaining_sequences = deque(sequence_costs)
    current_cycle = 0
    
    while remaining_sequences:
        current_used = 0
        parallel_ops = 0
        scheduled_this_cycle = []
        
        # Try to fit as many sequences as possible in this cycle
        temp_queue = deque()
        
        while remaining_sequences:
            seq_idx, cost = remaining_sequences.popleft()
            
            if current_used + cost <= total_capacity:
                # Can fit this sequence
                current_used += cost
                parallel_ops += 1
                scheduled_this_cycle.append((seq_idx, cost))
            else:
                # Can't fit, save for next cycle
                temp_queue.append((seq_idx, cost))
        
        # Record this cycle
        schedule.append((current_cycle, parallel_ops, current_used))
        print(f"Cycle {current_cycle}: {parallel_ops} operations in parallel, "
              f"{current_used:.1f}/{total_capacity} logical blocks used")
        
        # Move remaining sequences back to queue for next cycle
        remaining_sequences = temp_queue
        current_cycle += 1
    
    return schedule


def plot_parallel_operations(schedule, output_file='reaction_depth_analysis.pdf'):
    """
    Plot number of operations performed in parallel vs logical cycles.
    
    Args:
        schedule: List of (cycle, num_parallel_ops, used_blocks) tuples
        output_file: Output filename for the plot
    """
    # Apply simple style file
    plt.style.use('plotstylefile.mplstyle')
    
    cycles = [s[0] for s in schedule]
    parallel_ops = [s[1] for s in schedule]
    used_blocks = [s[2] for s in schedule]
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot 1: Parallel operations vs cycles (MAIN PLOT - what user requested)
    ax1.plot(cycles, parallel_ops, 'o-')
    ax1.set_xlabel('Logical Cycles')
    ax1.set_ylabel('Number of PPRs')
    # ax1.set_title('Parallel Operations vs Logical Cycles\n(Until Computation Finishes)')
    ax1.set_xlim(left=0)
    ax1.set_ylim(bottom=0, top=max(parallel_ops) * 1.15)
    
    # Add text box with summary
    total_cycles = len(cycles)
    textstr = f'Total Cycles: {total_cycles}\nTotal Operations: {sum(parallel_ops)}'
    ax1.text(0.98, 0.98, textstr, transform=ax1.transAxes,
             verticalalignment='top', horizontalalignment='right')
    
    # Plot 2: Logical blocks utilization vs cycles (additional insight)
    # Normalize used_blocks to 0-1 scale
    normalized_blocks = [b / TOTAL_LOGICAL_BLOCKS for b in used_blocks]
    ax2.plot(cycles, normalized_blocks, 's-', alpha=0.7)
    ax2.axhline(y=1.0, color='r', linestyle='--', label='Total Capacity')
    ax2.fill_between(cycles, 0, normalized_blocks, alpha=0.3, color='green')
    ax2.set_xlabel('Logical Cycles')
    ax2.set_ylabel('Usage of workspace')
    # ax2.set_title('Logical Blocks Utilization vs Logical Cycles')
    ax2.set_xlim(left=0)
    ax2.set_ylim(bottom=0, top=1.1)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(output_file, bbox_inches='tight', pad_inches=0.1)
    print(f"\nPlot saved to {output_file}")
    
    # Use non-interactive backend to avoid display issues
    plt.close('all')
    
    return fig


def print_statistics(schedule, sequences):
    """Print summary statistics."""
    total_cycles = len(schedule)
    total_sequences = len(sequences)
    parallel_ops = [s[1] for s in schedule]
    used_blocks = [s[2] for s in schedule]
    
    print("\n" + "="*60)
    print("SCHEDULING STATISTICS")
    print("="*60)
    print(f"Total sequences: {total_sequences}")
    print(f"Total logical cycles: {total_cycles}")
    print(f"Average operations per cycle: {np.mean(parallel_ops):.2f}")
    print(f"Maximum operations per cycle: {max(parallel_ops)}")
    print(f"Minimum operations per cycle: {min(parallel_ops)}")
    print(f"Average logical blocks used per cycle: {np.mean(used_blocks):.2f}")
    print(f"Average utilization: {np.mean(used_blocks) / TOTAL_LOGICAL_BLOCKS * 100:.2f}%")
    print(f"T state overhead per sequence: {T_STATE_OVERHEAD} logical blocks")
    print("="*60)


def analyze_sequence_costs(sequences):
    """Analyze the distribution of sequence costs."""
    costs = [calculate_sequence_cost(seq) for seq in sequences]
    active_volumes = [seq['active_volume'] for seq in sequences]
    
    print("\n" + "="*60)
    print("SEQUENCE COST ANALYSIS")
    print("="*60)
    print(f"Active volume range: {min(active_volumes)} - {max(active_volumes)}")
    print(f"Active volume mean: {np.mean(active_volumes):.2f}")
    print(f"Total cost range: {min(costs):.2f} - {max(costs):.2f}")
    print(f"Total cost mean: {np.mean(costs):.2f}")
    print(f"Sequences that fit in one cycle (cost <= {TOTAL_LOGICAL_BLOCKS}): "
          f"{sum(1 for c in costs if c <= TOTAL_LOGICAL_BLOCKS)}/{len(costs)}")
    print("="*60)


# sum of all the active volumes
def sum_of_active_volumes(sequences):
    return sum(seq['active_volume'] + T_STATE_OVERHEAD for seq in sequences)


if __name__ == '__main__':
    # Load sequences
    sequences = load_sequences()
    
    # Analyze sequence costs
    analyze_sequence_costs(sequences)
    
    # Schedule sequences
    print("\nScheduling sequences...")
    schedule = schedule_sequences(sequences)
    
    # Print statistics
    print_statistics(schedule, sequences)
    
    # Plot results
    plot_parallel_operations(schedule)
    print(f"Sum of active volumes: {sum_of_active_volumes(sequences)}")
