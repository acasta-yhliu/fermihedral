from sys import argv

from fermihedral import DecentSolver
from fermihedral.satutil import Kissat

[_, nmodes, spill] = argv
print(
    f"> solving decenting model for {nmodes} modes, {nmodes} + {spill} qubits")

solver = DecentSolver(int(nmodes), int(spill))

kissat = Kissat()

solution, weight = solver.solve("dimacs", external_solver=kissat)

print(solution, weight)
