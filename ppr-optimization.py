
def check_commutativity(pauli_string1, pauli_string2):
    # x, y 
    # x, z
    # z, y do not commute
    commute_count = 0
    for i in range(len(pauli_string1)):
        if pauli_string1[i] == 'x' and pauli_string2[i] == 'y' or pauli_string1[i] == 'y' and pauli_string2[i] == 'x':
            commute_count += 1
        if pauli_string1[i] == 'x' and pauli_string2[i] == 'z' or pauli_string1[i] == 'z' and pauli_string2[i] == 'x':
            commute_count += 1
        if pauli_string1[i] == 'z' and pauli_string2[i] == 'y' or pauli_string1[i] == 'y' and pauli_string2[i] == 'z':
            commute_count += 1
    if commute_count % 2 == 0:
        return True
    else:
        return False

ppr_file = 'ppr_circuits/trotter_circuit_v2_paulis_commuted.csv'

# extract the pauli strings from the ppr file
commutable_pairs = []
pauli_strings = []
with open(ppr_file, 'r') as file:
    for line in file:
        pauli_strings.append(line.split(',')[3:-2])

for i in range(len(pauli_strings) - 1):
    if check_commutativity(pauli_strings[i], pauli_strings[i+1]) == True:
        commutable_pairs.append((i+1, i+2))



merged_pairs = []
for i in range(len(pauli_strings) - 3):
    if (i+1, i+2) in commutable_pairs:
        if pauli_strings[i+1] == pauli_strings[i+3]:
            merged_pairs.append((i+3, i+5))  # 1-based CSV row numbers

print((merged_pairs))

    

# plot first value of pair  against index of pair
# import matplotlib.pyplot as plt
# plt.scatter([pair[0] for pair in commutable_pairs], range(len(commutable_pairs)))
# plt.show()
