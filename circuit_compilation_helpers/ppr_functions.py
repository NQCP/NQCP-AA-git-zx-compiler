from typing import Dict, List, TextIO, Tuple  # riverlane

def write_condition(condition):
    if condition == "":
        return ""
    else:
        return "&&".join(
            [f"({cindex}=={value})" for cindex, value in condition.items()]
        )

def write_instruction(
    pauli_file: TextIO,
    num_qubits: int,
    rotations: List[Tuple[int, str]],
    instruction: str = "rotate",
    angle: int = "",
    cindex: int = "",
    condition: Dict[int, int] = "",
):
    joint = ["i" for _ in range(num_qubits)]
    for index, basis in rotations:
        joint[index] = basis
    pauli_file.write(
        f"{instruction},{angle},{','.join(joint)},{cindex},{write_condition(condition)}\n"
    )


def get_index(string: str, is_measurement_cbit: bool = False) -> int:
    if "[" in string:
        parsed = string.split("[")
        return int(parsed[1].split("]")[0])
    else:
        return int(string[1:])


def parse_instruction(
    pauli_file: TextIO,
    num_qubits: int,
    line: str,
    condition: Dict[int, int] = "",
):
    if not line.strip():
        return
    if line.startswith("x"):
        index = get_index(line)
        write_instruction(
            pauli_file,
            num_qubits,
            [(index, "x")],
            angle=2,
            condition=condition,
        )
    elif line.startswith("z"):
        index = get_index(line)
        write_instruction(
            pauli_file,
            num_qubits,
            [(index, "z")],
            angle=2,
            condition=condition,
        )
    elif line.startswith("s"):
        index = get_index(line)
        write_instruction(
            pauli_file,
            num_qubits,
            [(index, "z")],
            angle=4,
            condition=condition,
        )
    elif line.startswith("sdg"):
        index = get_index(line)
        write_instruction(
            pauli_file,
            num_qubits,
            [(index, "z")],
            angle=-4,
            condition=condition,
        )
    elif line.startswith("h"):
        index = get_index(line)
        write_instruction(
            pauli_file,
            num_qubits,
            [(index, "x")],
            angle=4,
            condition=condition,
        )
        write_instruction(
            pauli_file,
            num_qubits,
            [(index, "z")],
            angle=4,
            condition=condition,
        )
        write_instruction(
            pauli_file,
            num_qubits,
            [(index, "x")],
            angle=4,
            condition=condition,
        )
    elif line.startswith("t"):
        index = get_index(line)
        write_instruction(
            pauli_file,
            num_qubits,
            [(index, "z")],
            angle=8,
            condition=condition,
        )
    elif line.startswith("tdg"):
        index = get_index(line)
        write_instruction(
            pauli_file, num_qubits, [(index, "z")], angle=-8, condition=condition
        )
    elif line.startswith("cx"):
        qubits = line[3:-2].split(",")
        control = get_index(qubits[0])
        target = get_index(qubits[1])
        write_instruction(
            pauli_file,
            num_qubits,
            [(control, "z"), (target, "x")],
            angle=4,
            condition=condition,
        )
        write_instruction(
            pauli_file,
            num_qubits,
            [(control, "z")],
            angle=-4,
            condition=condition,
        )
        write_instruction(
            pauli_file,
            num_qubits,
            [(target, "x")],
            angle=-4,
            condition=condition,
        )
    elif line.startswith("measure"):
        qubit, cbit = line.split(" -> ")
        qindex = get_index(qubit)
        cindex = get_index(cbit, True)
        write_instruction(
            pauli_file,
            num_qubits,
            [(qindex, "z")],
            instruction="measure",
            cindex=cindex,
            condition=condition,
        )
    else:
        raise ValueError(line)


def qasm_to_paulis(qasm_file_name):
    with open(qasm_file_name) as qasm_file:
        pauli_operations = []

        if qasm_file.readline().startswith("//"):
            next(qasm_file)
            next(qasm_file)
        next(qasm_file)
        with open(
            f"{'.'.join(qasm_file_name.split('.')[:-1])}_paulis.csv", "w"
        ) as pauli_file:
            for line in qasm_file:
                if line.startswith("qreg"):
                    num_qubits = get_index(line)
                elif line.startswith("creg"):
                    pass
                elif line.startswith("if"):
                    parts = line.split(") ")
                    condition = parts[0].split("(")[1]
                    cbit, value = condition.split("==")
                    cindex = get_index(cbit)
                    parse_instruction(
                        pauli_file, num_qubits, parts[1], {cindex: int(value)}
                    )
                else:
                    parse_instruction(pauli_file, num_qubits, line)
