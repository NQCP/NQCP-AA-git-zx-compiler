from resource_estimators.baseline_estimates import baseline_estimator
from circuit_compilation_helpers.ppr_functions import qasm_to_paulis
from circuit_compilation_helpers.commute_ppr_tableau import commuted_ppr

circuit_name = 'fermi_hubbard_2d_step_s4_universal'

qasm_to_paulis(f'ppr_circuits/{circuit_name}.qasm')
commuted_ppr(f'ppr_circuits/{circuit_name}_paulis.csv')
baseline_estimator(f'ppr_circuits/{circuit_name}_paulis_commuted.csv')