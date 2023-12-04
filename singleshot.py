from sys import argv

from fermihedral import DescentSolver
from fermihedral.satutil import Kissat, Cadical


[_, nmodes] = argv
print(
    f"> solving decenting model for {nmodes} modes, {nmodes} qubits")

nmodes = int(nmodes)

solver = DescentSolver(nmodes, independence=False, vacuum=True)
print("> start solving")
solution, weight = solver.solve(progress=True,
                                solver_init=Kissat, solver_args=[24 * 60 * 60])

print(f"{nmodes};{weight};{solution}")
