"""QASM to Pauli-product-rotation (PPR) conversion.

Reads an OpenQASM 2.0 circuit over the Clifford+T gate set and writes one row
per Pauli product rotation to ``<name>_paulis.csv``. Each gate is rewritten
using the standard identities (see e.g. Litinski, arXiv:1808.02892):

    X = X_{pi/2}                      S = Z_{pi/4}      T   = Z_{pi/8}
    Z = Z_{pi/2}                      Sdg = Z_{-pi/4}   Tdg = Z_{-pi/8}
    H = X_{pi/4} Z_{pi/4} X_{pi/4}    SX  = X_{pi/4}
    CX(c,t) = (Z_c X_t)_{pi/4} (Z_c)_{-pi/4} (X_t)_{-pi/4}

Row format (one line per rotation):

    <instruction>,<angle>,<p_0>,...,<p_{n-1}>,<cindex>,<condition>

where ``angle = n`` denotes a rotation by pi/n (negative for the inverse),
``p_i`` is the Pauli letter on qubit i (``i`` for identity), ``cindex`` is the
classical bit index for measurements, and ``condition`` renders an ``if(...)``
guard. ``angle`` and ``cindex`` are left empty where they do not apply, so
rotation rows end in two empty fields.

Downstream, ``commute_ppr_tableau.commuted_ppr`` commutes the Clifford
rotations to the end of the circuit, leaving the pi/8 rotations that the
resource estimators consume.

Gate dispatch
-------------
``LEGACY_PREFIX_DISPATCH`` selects how a QASM line is matched to a gate rule:

``True`` (default)
    First-prefix match against the ordered rule list, reproducing the original
    implementation exactly, including three cases where a longer gate name is
    shadowed by a shorter one earlier in the list:

      * ``sx``  is matched by ``s``   -> Z_{pi/4}  instead of X_{pi/4}
      * ``sdg`` is matched by ``s``   -> Z_{pi/4}  instead of Z_{-pi/4}
      * ``tdg`` is matched by ``t``   -> Z_{pi/8}  instead of Z_{-pi/8}

    Kept as the default so previously generated CSVs reproduce byte for byte.

``False``
    Exact match on the gate token, handling ``sx``, ``sdg`` and ``tdg``
    correctly. This changes the emitted circuit wherever those gates occur.
"""

from typing import Dict, List, Optional, TextIO, Tuple

LEGACY_PREFIX_DISPATCH = True

Rotation = Tuple[str, int]  # (Pauli letter, angle denominator; pi/angle)

# Single-qubit gates as a sequence of (Pauli, angle) rotations on the target.
SINGLE_QUBIT_GATES: Dict[str, List[Rotation]] = {
    "x": [("x", 2)],
    "z": [("z", 2)],
    "s": [("z", 4)],
    "sdg": [("z", -4)],
    "sx": [("x", 4)],
    "h": [("x", 4), ("z", 4), ("x", 4)],
    "t": [("z", 8)],
    "tdg": [("z", -8)],
}

# Order in which prefixes are tested under LEGACY_PREFIX_DISPATCH; this is the
# original dispatch order and is what shadows sx/sdg/tdg (see module docstring).
LEGACY_DISPATCH_ORDER = ("x", "z", "s", "sdg", "h", "t", "tdg", "cx", "measure")

HEADER_PREFIXES = ("//", "OPENQASM", "include", "creg")


def write_condition(condition) -> str:
    """Render an ``if(...)`` guard as ``(cbit==value)&&(...)``; empty if none."""
    if not condition:
        return ""
    return "&&".join(f"({cindex}=={value})" for cindex, value in condition.items())


def write_instruction(
    pauli_file: TextIO,
    num_qubits: int,
    rotations: List[Tuple[int, str]],
    instruction: str = "rotate",
    angle=" ",
    cindex=" ",
    condition=" ",
) -> None:
    """Write one row: `instruction`, angle, the Pauli string, cbit, condition."""
    paulis = ["i"] * num_qubits
    for index, basis in rotations:
        paulis[index] = basis
    angle = "" if angle == " " else angle
    cindex = "" if cindex == " " else cindex
    condition = "" if condition == " " else condition
    pauli_file.write(
        f"{instruction},{angle},{','.join(paulis)},{cindex},"
        f"{write_condition(condition)}\n"
    )


def get_index(string: str, is_measurement_cbit: bool = False) -> int:
    """Qubit/cbit index from either ``q[3]`` or ``q3`` style operands."""
    if "[" in string:
        return int(string.split("[", 1)[1].split("]", 1)[0])
    return int(string.strip()[1:])


def _gate_name(line: str) -> str:
    """Leading gate token of a QASM line (``cx q[0],q[1];`` -> ``cx``)."""
    return line.strip().split(" ", 1)[0].rstrip(";")


def _resolve_gate(line: str) -> Optional[str]:
    """Gate rule a line dispatches to, or None if it matches no rule."""
    if LEGACY_PREFIX_DISPATCH:
        for name in LEGACY_DISPATCH_ORDER:
            if line.startswith(name):
                return name
        return None
    name = _gate_name(line)
    if name in SINGLE_QUBIT_GATES or name in ("cx", "measure"):
        return name
    return None


def _operands(line: str) -> List[str]:
    """Operand tokens of a gate line, e.g. ``['q[0]', 'q[1]']`` for a CX."""
    body = line.strip().rstrip(";")
    _, _, operands = body.partition(" ")
    return [token.strip() for token in operands.split(",") if token.strip()]


def parse_instruction(
    pauli_file: TextIO,
    num_qubits: int,
    line: str,
    condition=" ",
) -> None:
    """Expand one QASM gate line into its PPR rows and write them."""
    if not line.strip():
        return

    gate = _resolve_gate(line)
    if gate is None:
        raise ValueError(line)

    if gate in SINGLE_QUBIT_GATES:
        index = get_index(_operands(line)[0])
        for basis, angle in SINGLE_QUBIT_GATES[gate]:
            write_instruction(
                pauli_file,
                num_qubits,
                [(index, basis)],
                angle=angle,
                condition=condition,
            )
        return

    if gate == "cx":
        control, target = (get_index(token) for token in _operands(line)[:2])
        for rotations, angle in (
            ([(control, "z"), (target, "x")], 4),
            ([(control, "z")], -4),
            ([(target, "x")], -4),
        ):
            write_instruction(
                pauli_file, num_qubits, rotations, angle=angle, condition=condition
            )
        return

    # measure q[i] -> c[j]: a Z measurement recording into classical bit j.
    qubit, cbit = line.split(" -> ")
    write_instruction(
        pauli_file,
        num_qubits,
        [(get_index(qubit), "z")],
        instruction="measure",
        cindex=get_index(cbit, True),
        condition=condition,
    )


def _parse_conditional(line: str) -> Tuple[Dict[int, int], str]:
    """Split ``if(c[0]==1) gate ...`` into ({cbit: value}, gate line)."""
    guard, _, body = line.partition(") ")
    cbit, _, value = guard.split("(", 1)[1].partition("==")
    return {get_index(cbit): int(value)}, body


def qasm_to_paulis(qasm_file_name: str) -> str:
    """Convert a QASM file to ``<name>_paulis.csv``; returns the output path."""
    output_name = f"{qasm_file_name.rsplit('.', 1)[0]}_paulis.csv"
    num_qubits = None

    with open(qasm_file_name) as qasm_file, open(output_name, "w") as pauli_file:
        for line in qasm_file:
            stripped = line.strip()
            if not stripped or stripped.startswith(HEADER_PREFIXES):
                continue
            if stripped.startswith("qreg"):
                num_qubits = get_index(stripped)
                continue
            if num_qubits is None:
                raise ValueError(
                    f"gate encountered before qreg declaration: {stripped!r}"
                )
            if stripped.startswith("if"):
                condition, body = _parse_conditional(line)
                parse_instruction(pauli_file, num_qubits, body, condition)
            else:
                parse_instruction(pauli_file, num_qubits, line)

    return output_name
