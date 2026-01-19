# NQCP-AA-git-zx-compiler

This repository contains code to get resource estimates with active volume compilation.

## Repository Structure

```
NQCP-AA-git-zx-compiler/
├── pipeline.py                          
├── resource_estimators/                 
│   ├── baseline_estimates.py           
│   ├── av_estimates.py                 
│   └── av_compilation.py               
├── logical_to_ppr/                     
│   ├── ppr_functions.py                
│   └── commute_ppr.py                  
├── ppr_circuits/                       
│   └── *.qasm, *.csv                   
├── plots/                              
│   └── resource_estimates_*.pdf        
├── requirements.txt                    
└── README.md                           
```

## Running the Pipeline

The `pipeline.py` script processes quantum circuits through three main steps:
1. **QASM to Pauli Rotations**: Converts QASM circuit files to Pauli rotation format
2. **Clifford Commutation**: Optimizes the circuit by commuting Clifford gates
3. **Resource Estimation**: Generates resource estimates for baseline and active volume compilation

### Prerequisites

Install the required dependencies:

```bash
pip install -r requirements.txt
```

### Usage

1. First, use the `logical_circuit.ipynb` notebook to generate the logical circuit in `ppr_circuits/` directory
2. Open `pipeline.py` and set the `circuit_name` variable to your circuit filename (without extension):

```python
circuit_name = 'your_circuit_name'
```

3. Run the pipeline:

```bash
python pipeline.py
```

### Example

To process a circuit named `trotter_circuit_v2.qasm`:

```python
circuit_name = 'trotter_circuit_v2'
```



### Output

The pipeline generates a resource estimate plot saved as:

```
plots/resource_estimates_{circuit_name}_paulis_commuted.pdf
```

For example, if `circuit_name = 'trotter_circuit_v2'`, the output will be:
```
plots/resource_estimates_trotter_circuit_v2_paulis_commuted.pdf
```

This plot contains resource estimates comparing baseline, compact, and active volume compilation methods across different physical error rates.
