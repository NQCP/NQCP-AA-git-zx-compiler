import numpy as np 
import matplotlib.pyplot as plt
from resource_estimators.av_compilation import calculate_total_active_volume_from_csv



def av_estimator(ppr_file):
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

    total_msd_tiles_compact = magic_state_tiles
    total_tiles_compact = num_data_tiles_compact + total_msd_tiles_compact

    total_av_tiles = total_tiles 
    workspace_av_tiles = int(total_av_tiles/2)
    data_av_tiles = workspace_av_tiles
    logical_blocks_per_cycle = workspace_av_tiles
    T_count_av = calculate_total_active_volume_from_csv(ppr_file) / logical_blocks_per_cycle

    T_count = len(open(ppr_file).readlines())
    clock_cycle_per_ppr = 1 
    clock_cycle_value_sc = 1 # us
    total_clock_cycles = T_count * clock_cycle_per_ppr
    total_clock_cycles_av = T_count_av * clock_cycle_per_ppr

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
        #print(msd_error_rate_required())
        #print(magic_state_error_rate_15to1(physical_error_rate))
        if msd_error_rate_required() < magic_state_error_rate_15to1(physical_error_rate):
            #print("MSD error rate required is less than magic state error rate")
            #print('msd error rate required: ', msd_error_rate_required())
            #print('magic state error rate: ', magic_state_error_rate_15to1(physical_error_rate))
            #print('physical error rate: ', physical_error_rate)
            break

    distance_list_av = []
    for physical_error_rate in physical_error_rate_list:
        for distance in range(1, 50):
            if total_tiles * total_clock_cycles_av * distance * logical_error_rate_pr_lq_pr_code_cycle(physical_error_rate, distance) < error_prob:
                distance_list_av.append(distance)
                # print('distance: ', distance, 'physical error rate: ', physical_error_rate)
                break
            else:
                print(f"Distance {distance} is not sufficient for physical error rate {physical_error_rate}")


    # log scale for physical error rate


    # now estimates for the superconducting: 
    time_sc_list_av = []
    physical_qubits_sc_list_av = []


    for distance in distance_list_av:
        time_sc_av = distance *  total_clock_cycles_av * clock_cycle_value_sc * 10**(-6) # seconds
        physical_qubits_sc_av = 2* (distance)**2 * total_tiles
        time_sc_list_av.append(time_sc_av)
        physical_qubits_sc_list_av.append(physical_qubits_sc_av)

    return distance_list_av, time_sc_list_av, physical_qubits_sc_list_av

