from sys import argv

from fermihedral import DecentSolver
from fermihedral.satutil import Kissat

[_, nqubits] = argv
print(f"> solving decenting model for {nqubits} qubits")

solver = DecentSolver(int(nqubits))

kissat = Kissat()

solution, weight = solver.solve(kissat)

print(solution, weight)
