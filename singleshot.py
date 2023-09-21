from sys import argv

from fermihedral import DescentSolver
from fermihedral.satutil import Kissat, Cadical




[_, nmodes, independency] = argv
print(
    f"> solving decenting model for {nmodes} modes, {nmodes} qubits, independency = {independency}")


def parse_bool(string: str):
    if string == "True" or string == "true" or string == "1":
        return True
    elif string == "False" or string == "false" or string == "0":
        return False


independency = parse_bool(independency)
nmodes = int(nmodes)

solver = DescentSolver(nmodes, independency)
print("> start solving")
solution, weight = solver.solve(progress=True,
                                solver_init=Kissat, solver_args=[30*60])

print(f"{nmodes},{weight},{solution}")
