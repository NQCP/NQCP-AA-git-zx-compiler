"""Correctness tests for commute_ppr_tableau.commuted_ppr.

The tableau implementation walks the rotation list forward and maintains a
Clifford "frame" F such that, for each surviving T-rotation T with Pauli P_T,
the emitted output is F . P_T . F^dagger (the conjugation of P_T by the
accumulated Clifford F).

This script verifies the tableau output against a numpy ground truth computed
by direct matrix exponentiation (scipy.linalg.expm).  No external data is
needed; all test cases are constructed inline.

Run:
    python verify_tableau.py
"""

from __future__ import annotations
import os
import numpy as np
from scipy.linalg import expm

from logical_to_ppr.commute_ppr_tableau import commuted_ppr

# ----------------------------------------------------------------------------
# Pauli matrices and helpers
# ----------------------------------------------------------------------------

I2 = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
PMAT = {"i": I2, "x": X, "y": Y, "z": Z}


def kron_pauli(letters):
    """Build n-qubit Pauli matrix from letter list (letters[0] = qubit 0)."""
    out = PMAT[letters[-1]]
    for letter in reversed(letters[:-1]):
        out = np.kron(out, PMAT[letter])
    return out


def _all_paulis(n):
    if n == 0:
        yield ""
        return
    for rest in _all_paulis(n - 1):
        for p in "ixyz":
            yield p + rest


def identify_pauli(mat, n):
    """Given a matrix that should be +-1 . (Pauli word), return (sign, letters)."""
    for letters in _all_paulis(n):
        candidate = kron_pauli(list(letters))
        for sign in (+1, -1):
            if np.allclose(mat, sign * candidate, atol=1e-9):
                return sign, list(letters)
    raise AssertionError(f"matrix does not match any signed Pauli word:\n{mat}")


# ----------------------------------------------------------------------------
# Test harness: build a tiny CSV, run tableau, compare to numpy
# ----------------------------------------------------------------------------

def run_tableau(rotations, n_qubits, tmpfile="/tmp/verify_tableau_in.csv"):
    """rotations = list of (angle, [letters]) tuples in time order.
    Writes a paulis.csv, runs commute_ppr_tableau.commuted_ppr, returns the
    emitted T-rotation rows as a list of (angle, [letters])."""
    with open(tmpfile, "w") as f:
        for angle, bases in rotations:
            assert len(bases) == n_qubits
            f.write(f"rotate,{angle},{','.join(bases)},,\n")
    commuted_ppr(tmpfile)
    out_path = tmpfile.replace(".csv", "_commuted.csv")
    rows = []
    with open(out_path) as f:
        for line in f:
            parts = line.rstrip("\n").split(",")
            angle = int(parts[1])
            pauli = parts[2 : 2 + n_qubits]
            rows.append((angle, pauli))
    os.remove(tmpfile)
    os.remove(out_path)
    return rows


def numpy_truth(cliffords, T_pauli):
    """Compute F . P_T . F^dagger for a list of Clifford rotations.

    F is built by right-multiplying each new Clifford as we walk forward in
    time -- this matches commute_ppr.commuted_ppr's convention (verified
    against the existing algorithm on small test circuits).
    """
    n = len(T_pauli)
    F = np.eye(2**n, dtype=complex)
    for angle_C, P_C in cliffords:
        C = expm(1j * np.pi / angle_C * kron_pauli(P_C))
        F = F @ C
    result = F @ kron_pauli(T_pauli) @ F.conj().T
    return identify_pauli(result, n)


# ----------------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------------

CASES = [
    # (description, cliffords, T-rotation)
    # cliffords = [(angle, [letters]), ...]
    # T-rotation = (angle, [letters])

    ("[a] single qubit: S past T(Z)",
     [(4, ["z"])], (8, ["z"])),

    ("[b] single qubit: Sdg past T(Z)",
     [(-4, ["z"])], (8, ["z"])),

    ("[c] single qubit: H (3 rotations) past T(Z)",
     [(4, ["x"]), (4, ["z"]), (4, ["x"])], (8, ["z"])),

    ("[d] two-qubit: X(x)Y rotation past T(X(x)X) -- the bug counter-example",
     [(4, ["x", "y"])], (8, ["x", "x"])),

    ("[e] two-qubit: CX-like sequence past T(Z) on target",
     [(4, ["z", "x"]), (-4, ["z", "i"]), (-4, ["i", "x"])], (8, ["i", "z"])),

    ("[f] 4-qubit: layered Cliffords past T",
     [(4, ["x", "i", "i", "i"]),
      (4, ["i", "z", "i", "i"]),
      (4, ["z", "i", "x", "i"]),
      (4, ["i", "x", "i", "y"])],
     (8, ["z", "y", "z", "x"])),

    ("[g] diagonal X-X overall-anti-commute case (would have triggered the bug)",
     [(4, ["x", "y"]), (4, ["i", "x"])], (8, ["x", "i"])),
]


def main():
    print("Tableau correctness tests (commute_ppr_tableau vs scipy.linalg.expm)")
    print("=" * 78)
    n_pass = 0
    n_fail = 0
    for desc, cliffords, T in CASES:
        n_q = len(T[1])
        # Build full rotation list: cliffords in order, then T.
        rotations = list(cliffords) + [T]
        try:
            actual_rows = run_tableau(rotations, n_q)
            assert len(actual_rows) == 1, (
                f"expected 1 surviving T-rotation, got {len(actual_rows)}"
            )
            actual_angle, actual_pauli = actual_rows[0]
            exp_sign, exp_pauli = numpy_truth(cliffords, T[1])
            expected_angle = T[0] * exp_sign
            ok = actual_angle == expected_angle and actual_pauli == exp_pauli
        except Exception as e:
            ok = False
            actual_angle, actual_pauli = None, None
            expected_angle, exp_pauli = None, None
            err = repr(e)
        if ok:
            n_pass += 1
            print(f"  PASS  {desc}")
            print(f"          {actual_angle:+d} * {','.join(actual_pauli)}")
        else:
            n_fail += 1
            print(f"  FAIL  {desc}")
            print(f"          numpy:    {expected_angle} * {exp_pauli}")
            print(f"          tableau:  {actual_angle} * {actual_pauli}")
            if "err" in dir():
                print(f"          err:      {err}")

    print()
    print(f"{n_pass} passed, {n_fail} failed")
    if n_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
