from sys import argv
from time import time

from fermihedral import DescentSolver
from fermihedral.satutil import Kissat, Cadical


[_, nmodes] = argv
print(
    f"> solving decenting model for {nmodes} modes, {nmodes} qubits")

nmodes = int(nmodes)

# without algebraic independence

start = time()

solver = DescentSolver(nmodes, False)

nodep_construct = time()

solution, weight = solver.solve(progress=True,
                                solver_init=Kissat, solver_args=[72 * 60 * 60])

nodep_solve = time()

nodep_solve = nodep_solve - nodep_construct
nodep_construct = nodep_construct - start

# with algebraic independence

start = time()

solver = DescentSolver(nmodes, True)

dep_construct = time()

solution, weight = solver.solve(progress=True,
                                solver_init=Kissat, solver_args=[72 * 60 * 60])

dep_solve = time()

dep_solve = dep_solve - dep_construct
dep_construct = dep_construct - start

with open("imgs/benchmark.log", "a+") as f:
    print(f"{nmodes};without;{nodep_construct};{nodep_solve}", file=f)
    print(f"{nmodes};with;{dep_construct};{dep_solve}", file=f)
