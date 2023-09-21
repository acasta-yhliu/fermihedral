from fermihedral import ProgessiveSolver, DescentSolver, get_pauli_weight, get_bk_weight
from fermihedral.satutil import Kissat

import matplotlib.pyplot as plt

kissat = Kissat(30 * 60)

descent_solver = DescentSolver(4, 0, -1)
final_solution, weight = descent_solver.solve("dimacs", external_solver=kissat)

x = []
y = []

while len(final_solution) <= 2 * 30:
    previous_weight = get_pauli_weight(final_solution)

    print("> solving weight", len(final_solution) // 2)

    progessive_solver = ProgessiveSolver(final_solution, 2)
    final_solution = (solution := progessive_solver.solve(kissat))

    while True:
        diff_weight = get_pauli_weight(solution) - previous_weight
        progessive_solver.restrict_weight(diff_weight)
        solution = progessive_solver.solve(kissat)

        if solution != None:
            final_solution = solution

            break

    x.append(len(final_solution) // 2)
    y.append(get_pauli_weight(final_solution))

z = [get_bk_weight(i) for i in x]
plt.plot(x, y, label="progessive")
plt.plot(x, z, label="bravyi-kitaev")
plt.legend()
plt.savefig("progessive.png")