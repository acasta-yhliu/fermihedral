from typing import Literal
from z3 import Goal, And, Or, Xor, Not, BoolRef, BitVecVal
from z3 import If, Then, Solver, sat
from functools import reduce
from operator import add
from openfermion import bravyi_kitaev, FermionOperator, QubitOperator

from .pauli import Pauli, PauliOp
from .iterators import PowerSet, Combination


def get_pauli_weight(model: list[str]):
    return sum(map(lambda x: len(x) - x.count("_"), model))


def get_bk_weight(n_modes: int):
    # accquire the weight provided by bk transformation, n_modes -> n_qubits
    def get_weight(op: QubitOperator):
        return len(list(op.terms.keys())[0])  # weird

    # create corresponding operators
    majoranas = [FermionOperator(f"[{i}] + [{i}^]") for i in range(n_modes)] + [
        FermionOperator(f"[{i}] - [{i}^]") for i in range(n_modes)]

    majoranas = [bravyi_kitaev(i, n_modes) for i in majoranas]

    return sum(map(get_weight, majoranas))


class MajoranaModel:
    def __init__(self, nqubits: int, max_independent_length: int = 5) -> None:
        self.nqubits = nqubits
        self.majoranas = [Pauli(nqubits) for _ in range(2 * nqubits)]
        self.goal = Goal()

        print("> generating constraints for partial algebraic independent")
        for comb in PowerSet(self.majoranas, 2, max_independent_length):
            zipped = map(lambda x: reduce(Xor, x), zip(
                *(i.iter_bits() for i in comb)))
            self.goal.add(Or(*zipped))

        print("> generating constraints for anti-commutativity")
        for sa, sb in Combination(self.majoranas, 2):
            xors = []
            for a, b in zip(sa, sb):
                xors.append(Or(And(a.bit0, Not(b.bit0), b.bit1), And(a.bit1, b.bit0, Not(
                    b.bit1)), And(b.bit0, Not(a.bit0), a.bit1), And(b.bit1, a.bit0, Not(a.bit1))))
            self.goal.add(reduce(Xor, xors))

        print("> generating constraints for vaccuum state preservation")
        for m1, m2 in zip(self.majoranas[::2], self.majoranas[1::2]):
            def pairing(op_pair: tuple[PauliOp, PauliOp]):
                op1, op2 = op_pair
                return Or(And(op1.bit0, Not(op1.bit1), Not(op2.bit0), op2.bit1), And(op1.bit0, op1.bit1, Not(op2.bit0), Not(op2.bit1)))

            self.goal.add(reduce(Or, map(pairing, zip(m1, m2))))

    def restrict_weight(self, weight: int):
        def to_bitvec(b: BoolRef):
            return If(b, BitVecVal(1, 4 + self.nqubits), BitVecVal(0, 4 + self.nqubits))

        print(f"> generating constraints for total pauli weight < {weight}")
        weight_constraints = []
        for string in self.majoranas:
            for op in string:
                weight_constraints.append(to_bitvec(Or(op.bit0, op.bit1)))
        self.goal.add(reduce(add, weight_constraints) < weight)

    def solve(self, method: Literal["z3", "dimacs"]):
        solver = Solver()
        solver.add(Then('simplify', 'bit-blast', 'tseitin-cnf')(self.goal)[0])

        if method == "z3":
            print("> solving via z3")

            if solver.check() == sat:
                model = solver.model()
                return [op.decode(model) for op in self.majoranas]
            else:
                return None
        elif method == "dimacs":
            print("> solving via invoking external sat solver")

            return solver.dimacs(False)


class DecentSolver:
    def __init__(self, n: int) -> None:
        self.n = n
        self.model = MajoranaModel(n)

    def solve(self, method: Literal["z3", "sat"]):

        optimal_model, optimal_weight = None, 2 * self.n ** 2 + 1

        while True:
            print(
                f"solving {self.n} < {optimal_weight} weight")
            self.model.restrict_weight(optimal_weight)
            if (solution := self.model.solve(method)) is not None:
                optimal_model, optimal_weight = solution, calculate_weight(
                    solution)
            else:
                return optimal_model, optimal_weight
