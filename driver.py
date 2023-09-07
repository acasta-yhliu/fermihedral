#!/bin/env python3

from fermihedral import MajoranaModel
from sys import argv
from os import system
from time import time

if __name__ == "__main__":
    [_, nqubits, max_weight, model_file, result_file] = argv
    print(f"> {nqubits} qubits, max weight restricted to {max_weight}")

    model = MajoranaModel(int(nqubits))

    with open(model_file, "w+") as f:
        model.restrict_weight(max_weight)
        f.write(model.solve("dimacs"))

    # print("> z3")
    # start = time()
    # system(f"z3 parallel.enable=true -dimacs {model_file}")
    # z3_elapsed_time = time() - start

    print("> kissat")
    start = time()
    system(f"kissat -q {model_file}")
    kissat_elapsed_time = time() - start

    with open(result_file, "w+") as f:
        f.write(f"z3 {z3_elapsed_time} s\n")
        f.write(f"kissat {kissat_elapsed_time} s\n")

    # with open("result.log", "w+") as f:
    #     for i in range(1, 10):
    #         solver = DecentSolver(i)
    #         mapping, pauli_weight = solver.solve("z3")
    #         f.write(f"{i} => {pauli_weight}\n")
    #         print(i, "=>", pauli_weight)
