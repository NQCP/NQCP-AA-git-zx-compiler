from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple

from logical_to_ppr.commute_ppr_optimized import commuted_ppr as fallback_commuted_ppr

Pauli = Tuple[int, int, int]
EXACT_STREAMING_FAST_PATH = False

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
    (1, 1, 3, 2),
    (2, 3, 2, 1),
    (3, 2, 1, 3),
)


def iter_set_bits(mask: int) -> Iterable[int]:
    while mask:
        lsb = mask & -mask
        yield lsb.bit_length() - 1
        mask ^= lsb


def pauli_from_bases(bases: Sequence[str]) -> Pauli:
    x_mask = 0
    z_mask = 0
    phase = 0

    for index, basis in enumerate(bases):
        bit = 1 << index
        if basis == "x":
            x_mask |= bit
        elif basis == "y":
            x_mask |= bit
            z_mask |= bit
            phase = (phase + 1) & 3
        elif basis == "z":
            z_mask |= bit

    return x_mask, z_mask, phase


def multiply_paulis(left: Pauli, right: Pauli) -> Pauli:
    left_x, left_z, left_phase = left
    right_x, right_z, right_phase = right
    return (
        left_x ^ right_x,
        left_z ^ right_z,
        (left_phase + right_phase + 2 * (left_z & right_x).bit_count()) & 3,
    )


def anticommutes(left: Pauli, right: Pauli) -> bool:
    left_x, left_z, _ = left
    right_x, right_z, _ = right
    return ((left_x & right_z).bit_count() + (left_z & right_x).bit_count()) & 1 == 1


def pauli_code_at(x_mask: int, z_mask: int, index: int) -> int:
    bit = 1 << index
    has_x = bool(x_mask & bit)
    has_z = bool(z_mask & bit)
    if has_x and has_z:
        return 2
    if has_x:
        return 1
    if has_z:
        return 3
    return 0


def pauli_from_sparse_codes(indices_and_codes: List[Tuple[int, int]], sign: int) -> Pauli:
    x_mask = 0
    z_mask = 0
    phase = 0

    for index, code in indices_and_codes:
        bit = 1 << index
        if code == 1:
            x_mask |= bit
        elif code == 2:
            x_mask |= bit
            z_mask |= bit
            phase = (phase + 1) & 3
        elif code == 3:
            z_mask |= bit

    if sign < 0:
        phase = (phase + 2) & 3

    return x_mask, z_mask, phase


def conjugate_by_clifford(axis: Pauli, angle: int, target: Pauli) -> Pauli:
    if not anticommutes(axis, target):
        return target

    target_x, target_z, target_phase = target

    if abs(angle) == 2:
        return target_x, target_z, (target_phase + 2) & 3

    if abs(angle) == 4:
        axis_x, axis_z, _ = axis
        target_x, target_z, _ = target
        sign = 1 if angle > 0 else -1
        support = (axis_x | axis_z | target_x | target_z)
        indices_and_codes: List[Tuple[int, int]] = []

        for index in iter_set_bits(support):
            axis_code = pauli_code_at(axis_x, axis_z, index)
            target_code = pauli_code_at(target_x, target_z, index)
            sign *= PHASES[axis_code][target_code]
            code = COMMUTED_CODES[axis_code][target_code]
            if code:
                indices_and_codes.append((index, code))

        return pauli_from_sparse_codes(indices_and_codes, sign)

    raise ValueError(f"Unsupported Clifford angle: {angle}")


def transform_pauli(pauli: Pauli, x_images: Sequence[Pauli], z_images: Sequence[Pauli]) -> Pauli:
    x_mask, z_mask, phase = pauli
    result: Pauli = (0, 0, phase)

    for index in iter_set_bits(x_mask):
        result = multiply_paulis(result, x_images[index])

    for index in iter_set_bits(z_mask):
        result = multiply_paulis(result, z_images[index])

    return result


def pauli_to_bases_and_sign(pauli: Pauli, num_qubits: int) -> Tuple[List[str], int]:
    x_mask, z_mask, phase = pauli
    bases = ["i"] * num_qubits

    support = x_mask | z_mask
    for index in iter_set_bits(support):
        bit = 1 << index
        has_x = bool(x_mask & bit)
        has_z = bool(z_mask & bit)
        if has_x and has_z:
            bases[index] = "y"
        elif has_x:
            bases[index] = "x"
        else:
            bases[index] = "z"

    bare_phase = (x_mask & z_mask).bit_count() & 3
    delta = (phase - bare_phase) & 3
    if delta == 0:
        sign = 1
    elif delta == 2:
        sign = -1
    else:
        raise ValueError("Encountered a non-Hermitian Pauli while commuting Cliffords")

    return bases, sign


def supports_streaming_fast_path(file_path: str) -> Tuple[bool, int]:
    num_qubits = 0
    with open(file_path) as paulis_file:
        for line in paulis_file:
            terms = line.rstrip("\n").split(",")
            if len(terms) < 4:
                return False, 0

            instruction = terms[0]
            cbit = terms[-2]
            conditions = terms[-1]
            bases = terms[2:-2]

            if instruction != "rotate" or cbit or conditions:
                return False, 0

            if num_qubits == 0:
                num_qubits = len(bases)
            elif len(bases) != num_qubits:
                return False, 0

    return True, num_qubits


def experimental_commuted_ppr(pauli_file_name: str) -> None:
    supports_fast_path, num_qubits = supports_streaming_fast_path(pauli_file_name)
    if not supports_fast_path:
        fallback_commuted_ppr(pauli_file_name)
        return

    original_x = [(1 << index, 0, 0) for index in range(num_qubits)]
    original_z = [(0, 1 << index, 0) for index in range(num_qubits)]
    x_images = list(original_x)
    z_images = list(original_z)

    output_name = f"{'.'.join(pauli_file_name.split('.')[:-1])}_commuted.csv"

    with open(pauli_file_name) as input_file, open(output_name, "w") as output_file:
        for line in input_file:
            terms = line.rstrip("\n").split(",")
            angle = int(terms[1])
            pauli = pauli_from_bases(terms[2:-2])

            if 2 <= abs(angle) <= 4:
                x_updates = []
                z_updates = []

                for index in iter_set_bits(pauli[1]):
                    transformed_generator = transform_pauli(
                        conjugate_by_clifford(pauli, angle, original_x[index]),
                        x_images,
                        z_images,
                    )
                    x_updates.append((index, transformed_generator))

                for index in iter_set_bits(pauli[0]):
                    transformed_generator = transform_pauli(
                        conjugate_by_clifford(pauli, angle, original_z[index]),
                        x_images,
                        z_images,
                    )
                    z_updates.append((index, transformed_generator))

                for index, transformed_generator in x_updates:
                    x_images[index] = transformed_generator

                for index, transformed_generator in z_updates:
                    z_images[index] = transformed_generator
            else:
                transformed_pauli = transform_pauli(pauli, x_images, z_images)
                bases, sign = pauli_to_bases_and_sign(transformed_pauli, num_qubits)
                output_file.write(f"rotate,{angle * sign},{','.join(bases)},,\n")


def commuted_ppr(pauli_file_name: str) -> None:
    # The streaming shortcut above is not exact for the current commute_ppr.py
    # semantics. Keep this module safe by delegating to the exact optimized
    # implementation until an exact linear-time formulation is derived.
    fallback_commuted_ppr(pauli_file_name)
