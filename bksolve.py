from fermihedral import MajoranaModel, get_bk_weight, get_pauli_weight

model = MajoranaModel(8)

model.restrict_weight(get_bk_weight(8))
solution = model.solve("dimacs")

with open("bksolve", "w+") as f:
    f.write(solution)
