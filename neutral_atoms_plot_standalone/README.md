# Neutral-atom runtime-vs-qubit plot (TMM Stat-QPE-gap, p = 1e-3)

Self-contained reproduction of the runtime-vs-physical-qubit trade-off for three
t-AV magic-state-factory variants on a neutral-atom platform.

## Run

```bash
pip install numpy matplotlib
python plot_neutral_atoms.py
```

Produces `runtime_vs_qubits_tmm_stat_qpe_gap_neutral_atoms_0.001.pdf`.

## Contents

- `plot_neutral_atoms.py` — computes the three curves from the data files and plots them.
- `data/bell-pairs-sweep-transversal-logical-blocks-tmm.npz` —
  per-(T-per-cycle) Bell-pair / cycle-count statistics from the transversal scheduler.
- `data/distance_table.csv` — minimum sufficient code distances per benchmark.
- `plotstylefile.mplstyle` — matplotlib style (optional; the script falls back to defaults if absent).

## What it shows

Runtime (s) vs physical-qubit count, sweeping each factory's scheduling parameter:

- **t-AV (trans-dist)** — transversal 15-to-1 distillation (8 code cycles per T state).
- **t-AV (parity-dist)** — parity-ancilla 15-to-1 (7 tiles, 71 code cycles per T state, no buffer bus).
- **t-AV (cult)** — fold-transversal cultivation (562 + d_eff² physical qubits, d_eff = min(d, 11)).

Code-cycle time is fixed at 10 ms (neutral atoms); the benchmark is the TMM
Statistical-QPE-gap circuit (20 Trotter steps) at physical error rate 1e-3.
All factory parameters are named constants at the top of `plot_neutral_atoms.py`.
