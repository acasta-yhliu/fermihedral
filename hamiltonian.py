from itertools import product
from sys import argv
from fermihedral import HamiltonianSolver
from fermihedral.satutil import Kissat

# parse hamiltonian from file
[_, hamiltonian_filename] = argv

print(f"> parsing Hamiltonian from {hamiltonian_filename}")

with open(hamiltonian_filename, "r", newline='') as hamiltonian_file:
    lines = list(map(str.strip, hamiltonian_file.readlines()))
    lines = list(filter(lambda x: len(x) > 0, lines))


def parse_hamiltonian(lines: list[str]):
    # parse the header first
    def parse_header(line: str):
        case_name, n_modes, input_format = line.split(' ')
        return case_name, int(n_modes), input_format

    case_name, n_modes, input_format = parse_header(lines[0])

    # parse the annihilation and creation operators at each line
    occurence = []

    if input_format == "ac":
        def expand_op(op: str):
            op = abs(int(op))
            return (2 * op, 2 * op - 1)

        def delete_duplicate(item: list[int]):
            result = []
            index = 0
            while True:
                if index >= len(item):
                    return result

                if index == len(item) - 1:
                    result.append(item[index])
                    return result

                if item[index + 1] == item[index]:
                    index += 2
                else:
                    result.append(item[index])
                    index += 1

        # transform into list of creation and annihilation ops
        for line in lines[1:]:
            line = map(expand_op, line.split(" "))
            occurence.extend(product(*line))

        # filter duplicated I
        while True:
            last_length = len(occurence)
            occurence = list(filter(lambda x: len(x) > 0,
                                    map(delete_duplicate, occurence)))
            if len(occurence) == last_length:
                break

    elif input_format == "mj":
        # directly obtain all the majoranas
        for line in lines[1:]:
            line = line.strip().split(" ")
            occurence.append(tuple(map(int, line)))

    return HamiltonianSolver(case_name, n_modes, occurence)


solver = parse_hamiltonian(lines)

# build a valid solution
print(
    f"> solving with Hamiltonian Pauli weight, problem = '{solver.name}' ({solver.n_modes} modes), bk = {solver.get_bk_weight()}")

solution, weight = solver.solve(
    progress=True, solver_init=Kissat, solver_args=[24 * 60 * 60])

# calculate hamiltonian pauli weight under bravyi-kitaev transformation
bk_weight = solver.get_bk_weight()

print(
    f"{solver.name};{solver.n_modes};{weight};{bk_weight};{solution}")
