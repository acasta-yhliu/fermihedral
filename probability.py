from functools import reduce
from typing import List
from fermihedral import MajoranaModel
from fermihedral.iterators import PowerSet
from fermihedral.pauli import check_algebraic_independent
from fermihedral.satutil import Kissat

from operator import and_, xor

# parse descent solver result
n_modes_list = []
weight_list = []
with open('imgs/descent.csv', "r") as decent_result:
    for line in decent_result.readlines():

        line = line.strip()
        if len(line) == 0:
            continue

        mode, weight = map(int, line.split(','))
        n_modes_list.append(mode)
        weight_list.append(weight)


with open("imgs/probability.csv", "w+") as csv:
    for n_mode, weight in zip(n_modes_list, weight_list):
        model = MajoranaModel(n_mode, False)
        model.restrict_weight(weight, relationship="==")
        print(f"> solving solutions for {n_mode} modes")
        solutions = model.solve_forall(100, progess=True,
                                       solver_init=Kissat, solver_args=[30*60, True])
        print(f"  checking algebraic independence")
        percentage = len(
            list(filter(check_algebraic_independent, solutions))) / len(solutions)
        print(f"  collision probability = {percentage}")
        print(n_mode, percentage, file=csv)
