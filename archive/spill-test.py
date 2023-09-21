from sys import argv

from matplotlib.pyplot import clf, plot, xlabel, ylabel, grid, title, savefig

from fermihedral import DescentSolver
from fermihedral.satutil import Kissat

kissat = Kissat(timeout=30*60)

n_modes = int(argv[1])

with open("spill-test.csv", "w+") as f:
    for i in range(0, n_modes + 1):
        solver = DescentSolver(n_modes, i, -1)
        solution, weight = solver.solve("dimacs", external_solver=kissat)
        print(f"{n_modes},{i},{weight}", file=f)

    print("> plot spill test result")

    exp_n_modes = 0
    exp_n_spills = []
    exp_weight = []

    with open("spill-test.csv", "r") as spill_result:
        for line in spill_result.readlines():

            line = line.strip()
            if len(line) == 0:
                continue

            exp_n_modes, n_spills, weight = map(int, line.split(','))
            exp_n_spills.append(n_spills)
            exp_weight.append(weight)

    clf()
    plot(exp_n_spills, exp_weight, color="blue",
         marker="x", markerfacecolor='none')
    xlabel("extra qubits/n")
    ylabel("total pauli weight/n")
    grid()
    title(f"total pauli weight with/out extra qubits ({exp_n_modes} modes)")
    savefig("spill-test.png")
