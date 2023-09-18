from sys import argv

from fermihedral import DecentSolver
from fermihedral.satutil import Kissat, AmazonSolver

[_, nmodes, spill] = argv
print(
    f"> solving decenting model for 2 - {nmodes} modes, {nmodes} + {spill} qubits")

kissat = Kissat(timeout=30*60)  # 30 mins, or we'll pass
amz = AmazonSolver()

with open("decent-test.csv", "w+") as f:
    for i in range(1, int(nmodes) + 1):
        if i <= 8:
            satsolver = kissat
        else:
            satsolver = amz

        solver = DecentSolver(i, int(spill), -1)
        solution, weight = solver.solve("dimacs", external_solver=satsolver)
        
        relax_solver = DecentSolver(i, int(spill))
        _, relax_weight = relax_solver.solve("dimacs", external_solver=satsolver)
        print(f"{i},{weight},{relax_weight}", file=f)
