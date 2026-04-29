import sys
import os
# Add parent directory to path to allow imports when running as script
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np 
import matplotlib.pyplot as plt
from resource_estimators.av_estimates import av_estimator
from resource_estimators.av_compilation import calculate_total_active_volume_from_csv

def baseline_estimator(ppr_file):
    distance_list_av, time_sc_list_av, physical_qubits_sc_list_av = av_estimator(ppr_file)
    # number of qubits needed, for baseline one qubit is encoded in 2 patches 
    
    error_prob = 0.01
    logical_qubits_system = len(open(ppr_file).readline().split(',')[2:-2])

    num_data_tiles = 2 * logical_qubits_system
    num_workspace_tiles = num_data_tiles 

    num_data_tiles_compact = 1.5 * logical_qubits_system + 3

    magic_state_tiles = 11
    msd_production_rate = 11 
    total_msd_tiles = magic_state_tiles * msd_production_rate # to have production rate of 1 msd per cycle
    total_tiles = num_data_tiles + num_workspace_tiles + total_msd_tiles
    print('total_tiles: ', total_tiles)

    total_msd_tiles_compact = magic_state_tiles
    total_tiles_compact = num_data_tiles_compact + total_msd_tiles_compact

    T_count = len(open(ppr_file).readlines())
    clock_cycle_per_ppr = 1 
    clock_cycle_value_sc = 1 # us
    total_clock_cycles = T_count * clock_cycle_per_ppr

    clock_cycle_per_ppr_compact = 11 
    total_clock_cycles_compact = T_count * clock_cycle_per_ppr_compact

    def msd_error_rate_required():
        return error_prob / T_count
    # goal is to stay below 1% error prob

    def magic_state_error_rate_15to1(physical_error_rate):
        return 35*physical_error_rate**3

    def logical_error_rate_pr_lq_pr_code_cycle(physical_error_rate, distance):
        return 0.1*(100*physical_error_rate)**((distance+1)/2)



    # determining code distance:  total_tiles * total_clock_cycles * d * logical_error_rate_pr_lq_pr_code_cycle(physical_error_rate, distance) < 0.01
    physical_error_rate_list = [ 10**(-3), 10**(-4), 10**(-5), 10**(-6), 10**(-7), 10**(-8), 10**(-9), 10**(-10)]
    for physical_error_rate in physical_error_rate_list:
        if msd_error_rate_required() < magic_state_error_rate_15to1(physical_error_rate):
            break

    distance_list = []
    distance_list_compact = []
    for physical_error_rate in physical_error_rate_list:
        for distance in range(1, 50):
            if total_tiles * total_clock_cycles * distance * logical_error_rate_pr_lq_pr_code_cycle(physical_error_rate, distance) < error_prob:
                distance_list.append(distance)
                
                break
            else:
                print(f"Distance {distance} is not sufficient for physical error rate {physical_error_rate}")

    for physical_error_rate in physical_error_rate_list:
        for distance in range(1, 50):            
            if total_tiles_compact * total_clock_cycles_compact * distance * logical_error_rate_pr_lq_pr_code_cycle(physical_error_rate, distance) < error_prob:
                distance_list_compact.append(distance)
                break
            else:
                print(f"Distance {distance} is not sufficient for physical error rate {physical_error_rate}")

    # circuit volumes 
    
    circ_volume_baseline = total_tiles * total_clock_cycles
    print('total_tiles: ', total_tiles)
    print('total_clock_cycles: ', total_clock_cycles)
    circ_volume_compact = total_tiles_compact * total_clock_cycles_compact
    circ_volume_av = calculate_total_active_volume_from_csv(ppr_file)
    print('circuit volume baseline: ', circ_volume_baseline, 'circuit volume compact: ', circ_volume_compact, 'circuit volume av: ', circ_volume_av)

    # now estimates for the superconducting: 
    time_sc_list = []
    physical_qubits_sc_list = []

    time_sc_list_compact = []
    physical_qubits_sc_list_compact = []
    for distance in distance_list:
        time_sc = distance *  total_clock_cycles * clock_cycle_value_sc * 10**(-6) # seconds
        physical_qubits_sc = 2* (distance)**2 * total_tiles
        time_sc_list.append(time_sc)
        physical_qubits_sc_list.append(physical_qubits_sc)


    for distance in distance_list_compact:
        time_sc_compact = distance *  total_clock_cycles_compact * clock_cycle_value_sc * 10**(-6) # seconds
        physical_qubits_sc_compact = 2* (distance)**2 * total_tiles_compact
        time_sc_list_compact.append(time_sc_compact)
        physical_qubits_sc_list_compact.append(physical_qubits_sc_compact)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].loglog(physical_error_rate_list, time_sc_list, label='baseline', linewidth=2, marker='o')
    axes[0].loglog(physical_error_rate_list, time_sc_list_compact, label='compact', linewidth=2, marker='s')
    axes[0].loglog(physical_error_rate_list, time_sc_list_av, label='av', linewidth=2, marker='^')
    axes[0].set_xlabel('Physical Error Rate')
    axes[0].set_ylabel('Time (seconds)')
    axes[0].legend()
    axes[0].grid(True)

    axes[1].loglog(physical_error_rate_list, physical_qubits_sc_list, label='baseline', linewidth=2, marker='o')
    axes[1].loglog(physical_error_rate_list, physical_qubits_sc_list_compact, label='compact', linewidth=2, marker='s')
    axes[1].loglog(physical_error_rate_list, physical_qubits_sc_list_av, label='av', linewidth=2, marker='^')
    axes[1].set_xlabel('Physical Error Rate')
    axes[1].set_ylabel('Physical Qubits')
    # axes[1].legend()
    axes[1].grid(True)

    axes[2].loglog(physical_error_rate_list, distance_list, label='baseline', linewidth=2, marker='o')
    axes[2].loglog(physical_error_rate_list, distance_list_compact, label='compact', linewidth=2, marker='s')
    axes[2].loglog(physical_error_rate_list, distance_list_av, label='av', linewidth=2, marker='^')
    axes[2].set_xlabel('Physical Error Rate')   
    axes[2].set_ylabel('Distance')
    # axes[2].legend()
    axes[2].grid(True)

    plt.tight_layout()

    plt.savefig('plots/resource_estimates_'+ppr_file.split('/')[-1].split('.')[0]+'.pdf')
    plt.show()
    print('total_tiles:' , total_tiles)

if __name__ == '__main__':
    ppr_file = 'ppr_circuits/trotter_circuit_v2_paulis_commuted.csv'
    baseline_estimator(ppr_file)

