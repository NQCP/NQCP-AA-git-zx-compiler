"""Bit-exact tableau-based commute_cliffords_to_end.

Walks the rotation list forward in time. Maintains a Clifford "frame" F
such that for each non-Clifford rotation T (or measurement), we emit
T_new with Pauli = F · T.Pauli · F† and angle's sign multiplied by the
sign of that Pauli image.

For purely unitary, unconditional rotation lists (the universal QASM
inputs in this repo), this produces byte-for-byte equivalent output to
the existing commute_ppr.commuted_ppr — verified by diffing against the
baseline on small circuits before running anything large.

Out-of-scope (will raise): conditional rotations, measurement rows.
None of the universal QASM inputs in this repo exercise either branch.
"""

from __future__ import annotations
from typing import Dict, List, Sequence, Tuple

from logical_to_ppr.ppr_functions import write_condition
from logical_to_ppr.commute_ppr_optimized import (
    parse_paulis_file,
    PAULI_TO_CODE,
    CODE_TO_PAULI,
)


# Per-qubit Pauli letter multiplication a·b → letter.
# Codes: 0=I, 1=X, 2=Y, 3=Z.
LETTER_PRODUCT: Tuple[Tuple[int, ...], ...] = (
    (0, 1, 2, 3),   # I·*
    (1, 0, 3, 2),   # X·X=I, X·Y=Z, X·Z=Y
    (2, 3, 0, 1),   # Y·X=Z, Y·Y=I, Y·Z=X
    (3, 2, 1, 0),   # Z·X=Y, Z·Y=X, Z·Z=I
)

# Phase factor of a·b in units of i (count mod 4: 0→1, 1→i, 2→-1, 3→-i).
LETTER_PRODUCT_PHASE: Tuple[Tuple[int, ...], ...] = (
    (0, 0, 0, 0),   # I·*
    (0, 0, 1, 3),   # X·X=I, X·Y=iZ (k=1), X·Z=-iY (k=3)
    (0, 3, 0, 1),   # Y·X=-iZ (k=3), Y·Z=iX (k=1)
    (0, 1, 3, 0),   # Z·X=iY (k=1), Z·Y=-iX (k=3)
)

# {p, q} = 0 (anti-commute) lookup. 1 if anti-commute, 0 if commute.
ANTI_COMMUTES: Tuple[Tuple[int, ...], ...] = (
    (0, 0, 0, 0),
    (0, 0, 1, 1),
    (0, 1, 0, 1),
    (0, 1, 1, 0),
)


def pauli_mul(letters1: List[int], phase1: int,
              letters2: List[int], phase2: int) -> Tuple[List[int], int]:
    """Multiply two phased Paulis.  Phases are i^k counts (0..3).
    Returns (result_letters, result_phase)."""
    n = len(letters1)
    result = [0] * n
    phase = (phase1 + phase2) % 4
    for q in range(n):
        a, b = letters1[q], letters2[q]
        result[q] = LETTER_PRODUCT[a][b]
        phase = (phase + LETTER_PRODUCT_PHASE[a][b]) % 4
    return result, phase


class CliffordFrame:
    """Action of a Clifford F on the n Pauli generators (X_q, Z_q)."""

    def __init__(self, n_qubits: int) -> None:
        self.n = n_qubits
        # F(X_q) and F(Z_q) as dense letter lists + i^k phase.
        # For a valid Clifford map of a real Pauli, phase is always 0 or 2 (=±1).
        self.x_img: List[List[int]] = [[0] * n_qubits for _ in range(n_qubits)]
        self.z_img: List[List[int]] = [[0] * n_qubits for _ in range(n_qubits)]
        self.x_phase: List[int] = [0] * n_qubits
        self.z_phase: List[int] = [0] * n_qubits
        for q in range(n_qubits):
            self.x_img[q][q] = 1   # X_q -> X_q
            self.z_img[q][q] = 3   # Z_q -> Z_q

    def _y_image(self, q: int) -> Tuple[List[int], int]:
        """F(Y_q) = i · F(X_q) · F(Z_q)."""
        letters, phase = pauli_mul(
            self.x_img[q], self.x_phase[q],
            self.z_img[q], self.z_phase[q],
        )
        return letters, (phase + 1) % 4

    def _generator_image(self, q: int, code: int) -> Tuple[List[int], int]:
        """Image of single-qubit Pauli (q, code) under F."""
        if code == 0:
            return [0] * self.n, 0
        if code == 1:
            return list(self.x_img[q]), self.x_phase[q]
        if code == 2:
            return self._y_image(q)
        if code == 3:
            return list(self.z_img[q]), self.z_phase[q]
        raise ValueError(code)

    def apply_to_pauli(self, sparse_pauli: Sequence[Tuple[int, int]]
                       ) -> Tuple[List[int], int]:
        """Compute F · P · F† for sparse P (list of (qubit, code))."""
        result = [0] * self.n
        phase = 0
        for q, c in sparse_pauli:
            if c == 0:
                continue
            contrib_letters, contrib_phase = self._generator_image(q, c)
            result, phase = pauli_mul(result, phase, contrib_letters, contrib_phase)
        return result, phase

    def apply_clifford(self, angle: int,
                       sparse_pauli_C: Sequence[Tuple[int, int]]) -> None:
        """Update F to F · C, where C = exp(iπ/angle · P_C)."""
        abs_angle = abs(angle)
        if abs_angle not in (2, 4):
            raise ValueError(f"apply_clifford requires |angle| in {{2,4}}, got {angle}")

        # Map qubit -> Pauli code on that qubit (0 if absent).
        c_pauli_at: Dict[int, int] = {q: c for q, c in sparse_pauli_C if c != 0}
        if not c_pauli_at:
            return  # identity Clifford, nothing to do

        # F(P_C) is needed for the |angle|=4 update.  Compute once with old F.
        if abs_angle == 4:
            f_pc_letters, f_pc_phase = self.apply_to_pauli(sparse_pauli_C)
        else:
            f_pc_letters = f_pc_phase = None  # not needed

        # Collect (target_field, q, new_letters, new_phase) updates using OLD F,
        # then write them all back so concurrent updates don't see each other.
        updates: List[Tuple[str, int, List[int], int]] = []
        for q, p_c_q in c_pauli_at.items():
            for code, field in ((1, "x"), (3, "z")):
                if not ANTI_COMMUTES[p_c_q][code]:
                    continue  # this generator commutes with C; no change

                if field == "x":
                    f_g_letters = self.x_img[q]
                    f_g_phase = self.x_phase[q]
                else:
                    f_g_letters = self.z_img[q]
                    f_g_phase = self.z_phase[q]

                if abs_angle == 2:
                    # exp(iπ/(±2)·P_C) ≈ ±iP_C; for anti-commuting Q the
                    # conjugation evaluates to -Q regardless of sign(angle).
                    new_letters = list(f_g_letters)
                    new_phase = (f_g_phase + 2) % 4
                else:  # abs_angle == 4
                    # C(G) = (sign(angle)·i) · P_C · G.  +i for angle>0, -i
                    # for angle<0 (= -i factor encoded as i^3).  This matches
                    # commute_ppr.py's `phase = 1 if angle > 0 else -1` seed.
                    new_letters, new_phase = pauli_mul(
                        f_pc_letters, f_pc_phase, f_g_letters, f_g_phase
                    )
                    new_phase = (new_phase + (1 if angle > 0 else 3)) % 4
                updates.append((field, q, new_letters, new_phase))

        for field, q, new_letters, new_phase in updates:
            if field == "x":
                self.x_img[q] = new_letters
                self.x_phase[q] = new_phase
            else:
                self.z_img[q] = new_letters
                self.z_phase[q] = new_phase


def commuted_ppr(pauli_file_name: str) -> None:
    """Bit-exact tableau implementation of commute_ppr.commuted_ppr.

    Currently rejects measurements and conditional rotations; the universal
    QASM inputs in this repo never exercise those code paths.
    """
    n, rotations = parse_paulis_file(pauli_file_name)
    F = CliffordFrame(n)
    output_rows: List[Tuple[str, int, List[int], str, Dict[str, str]]] = []

    for instruction, angle, bases_sparse, cbit, conditions in rotations:
        if instruction != "rotate":
            raise NotImplementedError("tableau impl: 'measure' rows not yet handled")
        if conditions:
            raise NotImplementedError("tableau impl: conditional rotations not yet handled")

        if 2 <= abs(angle) <= 4:
            F.apply_clifford(angle, bases_sparse)
            continue

        # Non-Clifford rotation (T or otherwise): apply F, emit.
        new_letters, new_phase = F.apply_to_pauli(bases_sparse)
        if new_phase == 0:
            sign = 1
        elif new_phase == 2:
            sign = -1
        else:
            raise AssertionError(
                f"F · P · F† produced i^{new_phase} phase (expected 0 or 2). "
                f"Tableau is corrupted."
            )
        output_rows.append((instruction, angle * sign, new_letters, cbit, conditions))

    output_name = f"{'.'.join(pauli_file_name.split('.')[:-1])}_commuted.csv"
    with open(output_name, "w") as out:
        for instruction, angle, letters, cbit, conditions in output_rows:
            out.write(
                f"{instruction},{angle},"
                + ",".join(CODE_TO_PAULI[c] for c in letters)
                + f",{cbit},{write_condition(conditions)}\n"
            )
