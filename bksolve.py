from os import system
from sys import argv

from fermihedral import MajoranaModel, get_bk_weight, get_pauli_weight
from fermihedral.satutil import Kissat

n_qubits = int(argv[1])

kissat = Kissat()
model = MajoranaModel(n_qubits)

model.restrict_weight(get_bk_weight(n_qubits))
solution_dimacs = model.solve("dimacs", external_solver=kissat)
solution_z3 = model.solve("z3")
print(get_pauli_weight(solution_dimacs), get_pauli_weight(solution_z3))