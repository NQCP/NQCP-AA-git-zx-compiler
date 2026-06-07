from __future__ import annotations

from typing import Dict, Iterable, List, Sequence, Tuple

from circuit_compilation_helpers.ppr_functions import write_condition

PAULI_TO_CODE = {"i": 0, "x": 1, "y": 2, "z": 3}
CODE_TO_PAULI = ("i", "x", "y", "z")

# These tables preserve the exact per-qubit logic in commute_ppr.py while
# letting the implementation use compact integer codes instead of strings.
COMMUTES = (
    (True, True, True, True),
    (True, True, False, False),
    (True, False, True, False),
    (True, False, False, True),
)
PHASES = (
    (1, 1, 1, 1),
    (1, 1, -1, 1),
    (1, 1, 1, -1),
    (1, -1, 1, 1),
)
COMMUTED_CODES = (
    (0, 1, 2, 3),
    (1, 0, 3, 2),
    (2, 3, 0, 1),
    (3, 2, 1, 0),
)

SparsePauli = Tuple[Tuple[int, int], ...]
Rotation = Tuple[str, int, SparsePauli, str, Dict[str, str]]
CommutedUpdate = Tuple[int, SparsePauli, Dict[str, str]]

EMPTY_CONDITIONS: Dict[str, str] = {}


def parse_conditions(conditions: str) -> Dict[str, str]:
    if not conditions:
        return {}

    parsed: Dict[str, str] = {}
    for clause in conditions.split("&&"):
        body = clause[1:-1]
        bit, value = body.split("==", 1)
        parsed[bit] = value
    return parsed


def encode_paulis(bases: Sequence[str]) -> SparsePauli:
    return tuple(
        (index, PAULI_TO_CODE[basis])
        for index, basis in enumerate(bases)
        if basis != "i"
    )


def decode_paulis(paulis: SparsePauli, num_qubits: int) -> List[str]:
    bases = ["i"] * num_qubits
    for index, code in paulis:
        bases[index] = CODE_TO_PAULI[code]
    return bases


def parse_paulis_file(file_path: str) -> Tuple[int, List[Rotation]]:
    rotations: List[Rotation] = []
    num_qubits = 0
    with open(file_path) as paulis_file:
        for line in paulis_file:
            terms = line.rstrip("\n").split(",")
            instruction = terms[0]
            try:
                angle = int(terms[1])
            except (ValueError, IndexError):
                angle = 1
            bases = terms[2:-2]
            if not num_qubits:
                num_qubits = len(bases)
            cbit = terms[-2]
            conditions = parse_conditions(terms[-1])
            rotations.append(
                (instruction, angle, encode_paulis(bases), cbit, conditions)
            )
    return num_qubits, rotations


def has_condition_conflict(
    first_conditions: Dict[str, str], second_conditions: Dict[str, str]
) -> bool:
    for bit, value in first_conditions.items():
        other_value = second_conditions.get(bit)
        if other_value is not None and other_value != value:
            return True
    return False


def is_condition_subset(
    first_conditions: Dict[str, str], second_conditions: Dict[str, str]
) -> bool:
    return first_conditions.items() <= second_conditions.items()


def flipped_conditions(conditions: Dict[str, str]) -> Dict[str, str]:
    return {bit: str((int(value) + 1) % 2) for bit, value in conditions.items()}


def merge_conditions(
    base_conditions: Dict[str, str], updates: Dict[str, str]
) -> Dict[str, str]:
    if not updates:
        return base_conditions
    if not base_conditions:
        return updates
    merged = dict(base_conditions)
    merged.update(updates)
    return merged


def iter_commuted_paulis(
    first_paulis: SparsePauli, second_paulis: SparsePauli
) -> Tuple[bool, int, SparsePauli]:
    commute = True
    phase = 1
    commuted_paulis: List[Tuple[int, int]] = []

    first_index = 0
    second_index = 0

    while first_index < len(first_paulis) or second_index < len(second_paulis):
        if second_index == len(second_paulis) or (
            first_index < len(first_paulis)
            and first_paulis[first_index][0] < second_paulis[second_index][0]
        ):
            qubit = first_paulis[first_index][0]
            first_code = first_paulis[first_index][1]
            second_code = 0
            first_index += 1
        elif first_index == len(first_paulis) or (
            second_paulis[second_index][0] < first_paulis[first_index][0]
        ):
            qubit = second_paulis[second_index][0]
            first_code = 0
            second_code = second_paulis[second_index][1]
            second_index += 1
        else:
            qubit = first_paulis[first_index][0]
            first_code = first_paulis[first_index][1]
            second_code = second_paulis[second_index][1]
            first_index += 1
            second_index += 1

        if not COMMUTES[first_code][second_code]:
            commute = not commute
        phase *= PHASES[first_code][second_code]

        commuted_code = COMMUTED_CODES[first_code][second_code]
        if commuted_code:
            commuted_paulis.append((qubit, commuted_code))

    return commute, phase, tuple(commuted_paulis)


def commuted_pauli_rotations(
    angle: int,
    first_paulis: SparsePauli,
    second_paulis: SparsePauli,
    first_conditions: Dict[str, str],
    second_conditions: Dict[str, str],
) -> Tuple[CommutedUpdate, ...]:
    if has_condition_conflict(first_conditions, second_conditions):
        return ((1, second_paulis, EMPTY_CONDITIONS),)

    commute, phase_delta, commuted_paulis = iter_commuted_paulis(
        first_paulis, second_paulis
    )
    phase = phase_delta if angle > 0 else -phase_delta

    if commute:
        return ((1, second_paulis, EMPTY_CONDITIONS),)

    if abs(angle) == 2:
        if first_conditions and not is_condition_subset(
            first_conditions, second_conditions
        ):
            return (
                (-1, second_paulis, first_conditions),
                (1, second_paulis, flipped_conditions(first_conditions)),
            )
        return ((-1, second_paulis, EMPTY_CONDITIONS),)

    if first_conditions and not is_condition_subset(first_conditions, second_conditions):
        return (
            (phase, commuted_paulis, first_conditions),
            (1, second_paulis, flipped_conditions(first_conditions)),
        )
    return ((phase, commuted_paulis, EMPTY_CONDITIONS),)


def commute_cliffords_to_end(rotations: List[Rotation]) -> List[Rotation]:
    for clifford_index in range(len(rotations) - 1, -1, -1):
        instruction, angle, bases, cbit, conditions = rotations[clifford_index]
        if instruction == "rotate" and 2 <= abs(angle) <= 4:
            updated_rotations = rotations[:clifford_index]
            for t_index in range(clifford_index + 1, len(rotations)):
                (
                    t_instruction,
                    t_angle,
                    t_bases,
                    t_cbit,
                    t_conditions,
                ) = rotations[t_index]
                commuted_updates = commuted_pauli_rotations(
                    angle, bases, t_bases, conditions, t_conditions
                )
                # commute_ppr.py inserts each update at the same index, so a
                # two-row split lands in reverse order. Preserve that exactly.
                for phase_delta, commuted_bases, condition_updates in reversed(
                    commuted_updates
                ):
                    updated_rotations.append(
                        (
                            t_instruction,
                            t_angle * phase_delta,
                            commuted_bases,
                            t_cbit,
                            merge_conditions(t_conditions, condition_updates),
                        )
                    )
            rotations = updated_rotations
    return rotations


def write_rotations(
    output_file, rotations: Iterable[Rotation], num_qubits: int
) -> None:
    for instruction, angle, bases, cbit, conditions in rotations:
        output_file.write(
            f"{instruction},{angle},{','.join(decode_paulis(bases, num_qubits))},"
            f"{cbit},{write_condition(conditions)}\n"
        )


def commuted_ppr(pauli_file_name: str) -> None:
    num_qubits, rotations = parse_paulis_file(pauli_file_name)
    commuted_rotations = commute_cliffords_to_end(rotations)

    output_name = f"{'.'.join(pauli_file_name.split('.')[:-1])}_commuted.csv"

    with open(output_name, "w") as output_file:
        write_rotations(output_file, commuted_rotations, num_qubits)
