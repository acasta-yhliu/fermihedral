from sys import argv

from fermihedral import DecentSolver
from fermihedral.satutil import Kissat

kissat = Kissat(timeout=30*60)

n_modes = int(argv[1])

with open("spill-test.csv", "w+") as f:
    for i in range(0, n_modes + 1):
        solver = DecentSolver(n_modes, i, -1)
        solution, weight = solver.solve("dimacs", external_solver=kissat)
        print(f"{n_modes},{i},{weight}", file=f)
