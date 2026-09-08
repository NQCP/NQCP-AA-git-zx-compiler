# Platform-aware fault-tolerant compilation and resource estimation

Compilers and resource estimators for fault-tolerant quantum algorithms across three
hardware connectivity classes, together with the scripts that produce the figures and
tables of the accompanying manuscript.

Given a logical Clifford+$T$ circuit, the code re-compiles it into a hardware-compatible
fault-tolerant instruction set and reports physical-qubit count, runtime, bridge-qubit
demand for parallelization, space-time volume, and reaction depth. Three compilation
targets are supported:

| Connectivity | Architecture | Instruction set |
| --- | --- | --- |
| Nearest-neighbour | baseline / compact lattice surgery | Pauli product rotations executed via Pauli product measurements |
| Limited non-local ($\log N$) | active volume (AV) | networks of logical blocks (oriented ZX diagrams) |
| Effectively all-to-all | transversal active volume (t-AV) | networks of *taubles* (transversal logical blocks) |

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Python 3.10+. Only `numpy`, `scipy` and `matplotlib` are needed for the estimators and
plots; `qiskit`, `pyzx`, `networkx` and `gridsynth` are used by the circuit-generation
helpers.

## Repository layout

```
circuit_compilation_helpers/        logical circuit -> PPR circuit
  ppr_functions.py                    QASM -> Pauli rotations (qasm_to_paulis)
  commute_ppr_tableau.py              commute Cliffords to the end (commuted_ppr)
  FermiHubbardCircuit.py              symmetry-shifted Fermi-Hubbard Trotter circuits
  energy_error_plot_helper.py         TMM Trotter/synthesis energy-error study

logical_network_compilation_helpers/
  logical_blocks.py                   PPR circuit -> AV logical-block network
  transversal-av-zx.py                QASM -> t-AV tauble network (JSON)

resource_estimators/
  qubits_runtime_estimates_v2.py      core: distances, per-architecture row builders
  qubits_runtime_estimates_seperate_v2.py   runtime-vs-qubits paper figures
  space-time-volume-estimator.py      per-benchmark STV / active-volume table
  distance_estimator.py               minimum code distances -> distance_table.csv
  distillation-stv-estimates.py       per-factory space-time volume vs distance
  av_compilation.py                   active-volume and t-AV block accounting
  baseline_estimates.py, av_estimates.py    lattice-surgery estimators
  distance_table.csv                  generated distance table (tracked)

bell_pair_analysis_helpers/           bridge-qubit (Bell-pair) demand
  bell_pairs_count.py                   AV scheduler sweep over workspace capacity
  bell_pairs_tranversal.py              t-AV scheduler sweep over T-per-cycle
  plot_helpers.py                       bridge-qubit / speedup figures

reaction_time_analysis/               reaction depth and stalling phase diagrams
logical_network_parallelization_plot_helpers/   PPR parallelization vs workspace capacity
transversal-limited-non-local/        standalone architecture simulator (placement,
                                      routing, scheduling, factories)

ppr_circuits/                         input QASM and generated PPR circuits
hamiltonians/                         TMM-PPP fermionic and Jordan-Wigner Hamiltonians
notebooks/logical_circuit.ipynb       logical circuit construction
plotstylefile*.mplstyle               matplotlib styles
```

## Quick start

`pipeline.py` runs the lattice-surgery path end to end: QASM to Pauli rotations, commute
Cliffords to the end, then estimate. Set `circuit_name` at the top of the file and run:

```bash
python pipeline.py
```

## Reproducing the manuscript figures and tables

Scripts that import sibling modules by bare name must be run from their own directory.

| Output | Command |
| --- | --- |
| Runtime vs physical qubits (TMM Stat-QPE-gap, Fermi-Hubbard, Stat-QPE, QPE-Abs) | `cd resource_estimators && python qubits_runtime_estimates_seperate_v2.py` |
| Space-time volume / active-volume summary table | `cd resource_estimators && python space-time-volume-estimator.py` |
| Minimum code distances (rebuilds `distance_table.csv`) | `python resource_estimators/distance_estimator.py` |
| Per-factory space-time volume vs code distance, with crossovers | `cd resource_estimators && python distillation-stv-estimates.py` |
| Bridge qubits and speedup vs factory count (t-AV); memory vs workspace capacity (AV) | `python bell_pair_analysis_helpers/plot_helpers.py` |
| PPR parallelization vs workspace capacity | `python logical_network_parallelization_plot_helpers/plot_helpers.py` |
| TMM Trotter and synthesis energy error | `python circuit_compilation_helpers/energy_error_plot_helper.py` |
| Reaction-time stalling phase diagram (TMM Stat-QPE-gap, t-AV, two error rates) | `cd reaction_time_analysis && python -c "import plot_helpers as p; p.render_tmm_gap_tav_two_rates_side_by_side_plot()"` |

Figures are written to `paper_plots/`. `reaction_time_analysis/plot_helpers.py`'s own
`main()` renders a larger set of exploratory variants rather than the manuscript figure,
hence the explicit function call above.

The physical error rate is selected by `DISTANCE_ERROR_RATE` in
`resource_estimators/qubits_runtime_estimates_v2.py` (`"0.001"` or `"0.0001"`); the
runtime-vs-qubits driver sets it per panel itself.

## Regenerating intermediate data

Compiled networks, scheduler sweeps and figures are not tracked (see `.gitignore`), so a
fresh clone must regenerate them in this order. The TMM inputs are in the repository; the
Fermi-Hubbard universal circuit is generated locally.

1. **PPR circuit** — `python pipeline.py` (or call `qasm_to_paulis` then `commuted_ppr`),
   producing `ppr_circuits/<name>_paulis_commuted.csv`.
2. **Logical networks** — `python logical_network_compilation_helpers/transversal-av-zx.py`
   for the t-AV tauble network and `python logical_network_compilation_helpers/logical_blocks.py`
   for the AV logical-block network, both written to `logical_network_files/`.
3. **Scheduler sweeps** — `python bell_pair_analysis_helpers/bell_pairs_tranversal.py`
   (t-AV, sweeps T-per-cycle) and `python bell_pair_analysis_helpers/bell_pairs_count.py`
   (AV, sweeps workspace capacity), written to each helper's `data/` directory as `.npz`.
4. **Distance table** — `python resource_estimators/distance_estimator.py`.
5. **Figures and tables** — the commands in the previous section.

The target circuit is a module-level constant (`circuit_name`, `CIRCUIT`) at the top of
each script in steps 1–3; the sweep grids (`CAPACITIES`, `T_COUNTS_PER_CYCLE`) are set the
same way. Fermi-Hubbard networks are large (hundreds of MB) and stream line by line, so
expect long runtimes on that benchmark.

## Conventions

- A logical qubit is a rotated surface-code patch of $2d^2$ physical qubits (measurement
  ancillas included).
- One logical cycle is $d$ code cycles. Lattice-surgery operations cost one logical cycle;
  transversal operations cost one code cycle.
- Block counts in `av_compilation.py` are distance-independent; dividing by $d$ gives the
  per-gate cost in logical-block units, and multiplying a patch count by $2d^2$ converts a
  logical space-time volume to physical qubit $\times$ code cycles.
- Magic-state factory models and their per-state periods and footprints are named
  constants at the top of `resource_estimators/qubits_runtime_estimates_v2.py` and
  `distillation-stv-estimates.py`.
- Noise models: circuit-level $p_L = 0.1(100p)^{(d+1)/2}$, and an erasure-conversion form
  $p_L = 0.03(25p)^{(d+1)/2}$ for neutral atoms; distances are chosen as the smallest odd
  $d$ with $n_Q n_C d\, p_L < \epsilon$ for a target failure budget $\epsilon = 0.01$.
