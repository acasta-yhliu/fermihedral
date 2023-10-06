from tqdm import tqdm
from fermihedral import MajoranaModel
from fermihedral.majorana import get_pauli_weight
from fermihedral.pauli import check_algebraic_independent
from fermihedral.satutil import Kissat


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


with open("imgs/distribution.csv", "w+") as csv:
    for n_mode, weight in zip(n_modes_list, weight_list):
        model = MajoranaModel(n_mode, False)
        model.restrict_weight(weight, relationship="<=")

        print(f"> solving solutions for {n_mode} modes")
        solutions = model.solve_forall(100, progess=True,
                                       solver_init=Kissat, solver_args=[30*60, True])

        TAIL = 5

        print(
            f"  checking algebraic independence for {len(solutions)} solutions")
        ndep = 0
        dists = [[] for _ in range(TAIL)]
        for solution in tqdm(solutions):
            dep, _dists = check_algebraic_independent(solution, TAIL)
            if dep:
                ndep += 1
            for id, dist in enumerate(dists):
                dist.append(_dists[id])
        probability = ndep / len(solutions)
        dists = map(lambda x: 0 if len(x) == 0 else sum(x) / len(x), dists)
        print(n_mode, probability, *dists, file=csv)
