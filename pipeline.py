from resource_estimators.baseline_estimates import baseline_estimator
from logical_to_ppr.ppr_functions import qasm_to_paulis
from logical_to_ppr.commute_ppr import commuted_ppr

circuit_name = 'trotter_circuit_v2'

qasm_to_paulis(f'ppr_circuits/{circuit_name}.qasm')
commuted_ppr(f'ppr_circuits/{circuit_name}_paulis.csv')
baseline_estimator(f'ppr_circuits/{circuit_name}_paulis_commuted.csv')