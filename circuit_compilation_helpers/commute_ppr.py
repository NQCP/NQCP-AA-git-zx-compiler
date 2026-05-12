from typing import Dict, List, Tuple
from logical_to_ppr.ppr_functions import write_condition  # riverlane 

COMMUTING_RULES = {
    ("i", "i"): (True, 1, "i"),
    ("i", "x"): (True, 1, "x"),
    ("i", "y"): (True, 1, "y"),
    ("i", "z"): (True, 1, "z"),
    ("x", "i"): (True, 1, "x"),
    ("x", "x"): (True, 1, "i"),
    ("x", "y"): (False, -1, "z"),
    ("x", "z"): (False, 1, "y"),
    ("y", "i"): (True, 1, "y"),
    ("y", "x"): (False, 1, "z"),
    ("y", "y"): (True, 1, "i"),
    ("y", "z"): (False, -1, "x"),
    ("z", "i"): (True, 1, "z"),
    ("z", "x"): (False, -1, "y"),
    ("z", "y"): (False, 1, "x"),
    ("z", "z"): (True, 1, "i"),
}


def parse_conditions(conditions: str) -> Dict[str, str]:
    if conditions:
        return {
            clause.split("==")[0][1:]: clause.split("==")[1][:-1]
            for clause in conditions.split("&&")
        }
    else:
        return {}


def parse_paulis_file(file_path):
    rotations = []
    with open(file_path) as paulis_file:
        for line in paulis_file:
            terms = line[:-1].split(",")
            instruction = terms[0]
            try:
                angle = int(terms[1])
            except (ValueError, IndexError):
                angle = 1
            bases = terms[2:-2]
            cbit = terms[-2]
            conditions = parse_conditions(terms[-1])
            rotations.append((instruction, angle, bases, cbit, conditions))
    return rotations


def commuted_pauli_rotations(
    angle: int,
    first_paulis: List[str],
    second_paulis: List[str],
    first_conditions: Dict[str, str],
    second_conditions: Dict[str, str],
) -> List[Tuple[int, List[str], Dict[str, str]]]:
    commute = True
    commuted_paulis = []
    phase = 1 if angle > 0 else -1
    for bit, value in first_conditions.items():
        if bit in second_conditions and second_conditions[bit] != value:
            return [(1, second_paulis, {})]
    for first_pauli, second_pauli in zip(first_paulis, second_paulis):
        commutation_rule = COMMUTING_RULES[(first_pauli, second_pauli)]
        commute ^= not commutation_rule[0]
        phase *= commutation_rule[1]
        commuted_paulis.append(commutation_rule[2])
    if commute:
        return [(1, second_paulis, {})]
    elif abs(angle) == 2:
        if first_conditions and not (
            first_conditions.items() <= second_conditions.items()
        ):
            return [
                (-1, second_paulis, first_conditions),
                (
                    1,
                    second_paulis,
                    {
                        bit: str((int(value) + 1) % 2)
                        for bit, value in first_conditions.items()
                    },
                ),
            ]
        else:
            return [(-1, second_paulis, {})]
    else:
        if first_conditions and not (
            first_conditions.items() <= second_conditions.items()
        ):
            return [
                (phase, commuted_paulis, first_conditions),
                (
                    1,
                    second_paulis,
                    {
                        bit: str((int(value) + 1) % 2)
                        for bit, value in first_conditions.items()
                    },
                ),
            ]
        else:
            return [(phase, commuted_paulis, {})]


def commute_cliffords_to_end(rotations):
    for clifford_index in range(len(rotations) - 1, -1, -1):
        instruction, angle, bases, cbit, conditions = rotations[clifford_index]
        if instruction == "rotate" and 2 <= abs(angle) <= 4:
            del rotations[clifford_index]
            t_index = clifford_index
            while t_index < len(rotations):
                t_instruction, t_angle, t_bases, t_cbit, t_conditions = rotations[
                    t_index
                ]
                del rotations[t_index]
                commuted_updates = commuted_pauli_rotations(
                    angle, bases, t_bases, conditions, t_conditions
                )
                for phase, commuted_bases, condition_updates in commuted_updates:
                    new_angle = t_angle * phase
                    rotations.insert(
                        t_index,
                        (
                            t_instruction,
                            new_angle,
                            commuted_bases,
                            t_cbit,
                            {**t_conditions, **condition_updates},
                        ),
                    )
                t_index += len(commuted_updates)
    return rotations


def commuted_ppr(pauli_file_name):
    rotations = parse_paulis_file(pauli_file_name)
    commuted_rotations = commute_cliffords_to_end(rotations)

    output_name = f"{'.'.join(pauli_file_name.split('.')[:-1])}_commuted.csv"

    with open(output_name, "w") as output_file:
        for instruction, angle, bases, cbit, conditions in commuted_rotations:
            output_file.write(
                f"{instruction},{angle},{','.join(bases)},{cbit},{write_condition(conditions)}\n"
            )
