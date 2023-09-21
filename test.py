from fermihedral.pauli import check_algebraic_independent

assert check_algebraic_independent(["XX", "XY", "YX", "XX"])