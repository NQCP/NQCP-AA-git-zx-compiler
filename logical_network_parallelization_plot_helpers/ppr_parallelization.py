import gzip
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
TOTAL_LOGICAL_BLOCKS =  1000  #(here i am assuming, i am already using 35 blocks for msd production)
circuit = 'fermi_hubbard_2d_step_s4_universal_paulis_commuted'  # Identifier for the dataset
input_dir = 'logical_network_files'  # Directory where the input file is located
output_dir = 'paper_plots'  # Directory to save the output plot
LOGICAL_BLOCKS_FILE = f'{input_dir}/logical_blocks_{circuit}.jsonl.gz'  # Input file with sequences


def load_sequences(filename=LOGICAL_BLOCKS_FILE):
    """Stream-load only the active_volume of each sequence.

    The full hexagon/input_sequence data is huge (~30+ GB resident for the
    fermi-hubbard 1M-row dataset) and reaction_depth.py only needs
    active_volume, so we discard the rest as we read. Returns a list of small
    dicts {'active_volume': int} so the rest of the script doesn't change.
    """
    sequences = []
    if filename.endswith('.gz'):
        opener = lambda: gzip.open(filename, 'rt')
        line_mode = True
    else:
        opener = lambda: open(filename, 'r')
        line_mode = False

    with opener() as f:
        if line_mode:
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                sequences.append({'active_volume': obj['active_volume']})
        else:
            for obj in json.load(f):
                sequences.append({'active_volume': obj['active_volume']})

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
    Schedule sequences in strict JSON order (no overtaking). The cycle ends
    as soon as the next sequence doesn't fit. This is the correct behavior
    when the input rows are only block-commuting (e.g. fermi-hubbard).
    A single oversized sequence is packed alone in its own cycle.

    Args:
        sequences: List of sequence dictionaries
        total_capacity: Total available logical blocks

    Returns:
        List of (cycle, num_parallel_ops, used_blocks) tuples
    """
    remaining_sequences = deque(
        (i, calculate_sequence_cost(seq)) for i, seq in enumerate(sequences)
    )
    schedule = []
    current_cycle = 0

    while remaining_sequences:
        current_used = 0
        parallel_ops = 0
        while remaining_sequences:
            seq_idx, cost = remaining_sequences[0]
            if current_used + cost <= total_capacity or parallel_ops == 0:
                remaining_sequences.popleft()
                current_used += cost
                parallel_ops += 1
            else:
                break

        schedule.append((current_cycle, parallel_ops, current_used))
        if current_cycle % 50000 == 0:
            print(f"Cycle {current_cycle}: {parallel_ops} operations in parallel, "
                  f"{current_used:.1f}/{total_capacity} logical blocks used")
        current_cycle += 1

    return schedule


def _rolling_stats(arr, window):
    """Chunk-based rolling stats: mean and 5/25/75/95th percentiles per window."""
    n_full = (len(arr) // window) * window
    if n_full == 0:
        x = np.array([len(arr) / 2.0])
        mean = np.array([arr.mean()])
        return x, mean, mean.copy(), mean.copy(), mean.copy(), mean.copy()
    chunks = arr[:n_full].reshape(-1, window)
    x = (np.arange(chunks.shape[0]) + 0.5) * window
    return (
        x,
        chunks.mean(axis=1),
        np.percentile(chunks, 5, axis=1),
        np.percentile(chunks, 25, axis=1),
        np.percentile(chunks, 75, axis=1),
        np.percentile(chunks, 95, axis=1),
    )


def plot_parallel_operations(schedule, output_file=f'{output_dir}/ppr_parallelization_{circuit}.pdf'):
    """Rolling-window summary plot: mean + percentile bands across logical cycles."""
    try:
        plt.style.use('plotstylefile.mplstyle')
    except OSError:
        pass

    parallel_ops = np.asarray([s[1] for s in schedule])
    used_blocks = np.asarray([s[2] for s in schedule])
    n_cycles = len(schedule)
    # Aim for ~500 windows; at least 1 cycle per window
    window = max(1, n_cycles // 500)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    # Panel 1: parallel PPRs per cycle
    x, mean, p05, p25, p75, p95 = _rolling_stats(parallel_ops, window)
    ax1.fill_between(x, p05, p95, alpha=0.20, color='steelblue', label='5-95th %ile')
    ax1.fill_between(x, p25, p75, alpha=0.40, color='steelblue', label='25-75th %ile')
    ax1.plot(x, mean, color='navy', lw=1.5, label='rolling mean')
    ax1.set_ylabel('PPRs per cycle')
    ax1.set_ylim(bottom=0)
    ax1.legend(loc='upper left')

    textstr = (
        f'Cycles: {n_cycles:,}\n'
        f'Window: {window} cycles\n'
        f'Mean PPRs/cycle: {parallel_ops.mean():.2f}\n'
        f'Max PPRs/cycle: {parallel_ops.max()}'
    )
    ax1.text(0.98, 0.98, textstr, transform=ax1.transAxes,
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))

    # Panel 2: workspace utilization
    util = used_blocks / TOTAL_LOGICAL_BLOCKS
    x, mean, p05, p25, p75, p95 = _rolling_stats(util, window)
    ax2.fill_between(x, p05, p95, alpha=0.20, color='seagreen')
    ax2.fill_between(x, p25, p75, alpha=0.40, color='seagreen')
    ax2.plot(x, mean, color='darkgreen', lw=1.5)
    ax2.axhline(y=1.0, color='r', linestyle='--', alpha=0.6, label='capacity')
    ax2.set_xlabel('Logical cycle')
    ax2.set_ylabel('Workspace utilization')
    ax2.set_xlim(0, n_cycles)
    ax2.set_ylim(0, 1.1)
    ax2.legend(loc='upper right')

    plt.tight_layout()
    plt.savefig(output_file, bbox_inches='tight', pad_inches=0.1)
    print(f"\nPlot saved to {output_file}")
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
