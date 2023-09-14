from functools import reduce
from operator import add
from typing import Literal

from openfermion import FermionOperator, QubitOperator, bravyi_kitaev
from z3 import (And, BitVecVal, BoolRef, Goal, If, Not, Or, Solver, Then, Xor,
                sat)

from .iterators import Combination, PowerSet
from .pauli import Pauli, PauliOp
from .satutil import SATModel, SATSolver


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
    def __init__(self, n_modes: int, spill: int = 0, max_independent_length: int = 4) -> None:
        self.nqubits = n_modes
        self.majoranas = [Pauli(n_modes + spill) for _ in range(2 * n_modes)]
        self.goal = Goal()

        # print("> generating constraints for partial algebraic independent")
        for comb in PowerSet(self.majoranas, 2, max_independent_length):
            zipped = map(lambda x: reduce(Xor, x), zip(
                *(i.iter_bits() for i in comb)))
            self.goal.add(Or(*zipped))

        # print("> generating constraints for anti-commutativity")
        for sa, sb in Combination(self.majoranas, 2):
            xors = []
            for a, b in zip(sa, sb):
                xors.append(Or(And(a.bit0, Not(b.bit0), b.bit1), And(a.bit1, b.bit0, Not(
                    b.bit1)), And(b.bit0, Not(a.bit0), a.bit1), And(b.bit1, a.bit0, Not(a.bit1))))
            self.goal.add(reduce(Xor, xors))

        # print("> generating constraints for vaccuum state preservation")
        for m1, m2 in zip(self.majoranas[::2], self.majoranas[1::2]):
            def pairing(op_pair: tuple[PauliOp, PauliOp]):
                op1, op2 = op_pair
                return Or(And(op1.bit0, Not(op1.bit1), Not(op2.bit0), op2.bit1), And(op1.bit0, op1.bit1, Not(op2.bit0), Not(op2.bit1)))

            self.goal.add(reduce(Or, map(pairing, zip(m1, m2))))

    def restrict_weight(self, weight: int):
        def to_bitvec(b: BoolRef):
            return If(b, BitVecVal(1, 4 + self.nqubits), BitVecVal(0, 4 + self.nqubits))

        # print(f"> generating constraints for total pauli weight < {weight}")
        weight_constraints = []
        for string in self.majoranas:
            for op in string:
                weight_constraints.append(to_bitvec(Or(op.bit0, op.bit1)))
        self.goal.add(reduce(add, weight_constraints) < weight)

    def solve(self, method: Literal["z3", "dimacs"], *, external_solver: SATSolver | None = None):
        solver = Solver()
        solver.add(Then('simplify', 'bit-blast', 'tseitin-cnf')(self.goal)[0])

        if method == "z3":
            print("> solving via z3")

            if solver.check() == sat:
                model = solver.model()
                return [op.decode_z3(model) for op in self.majoranas]
            else:
                return None
        elif method == "dimacs":
            # print("> solving via invoking external sat solver")

            LOCAL_GOAL_PATH = "./__goal.cnf"
            LOCAL_MODEL_PATH = "./__model.sol"

            with open(LOCAL_GOAL_PATH, "w+") as goal_file:
                goal_file.write(solver.dimacs())

            external_solver(LOCAL_GOAL_PATH, LOCAL_MODEL_PATH)

            sat_model = SATModel.from_file(
                LOCAL_MODEL_PATH, renaming=LOCAL_GOAL_PATH)

            if sat_model.sat:
                return [op.decode_model(sat_model) for op in self.majoranas]
            else:
                return None


class DecentSolver:
    def __init__(self, n: int, spill: int = 0, max_independent: int = 4) -> None:
        self.n = n
        self.model = MajoranaModel(n, spill, max_independent)
        self.relaxation = max_independent

    def solve(self, method: Literal["z3", "dimacs"], *, external_solver: SATSolver | None = None):

        optimal_model, optimal_weight = None, get_bk_weight(self.n) + 1

        while True:
            print(
                f"solving {self.n} < {optimal_weight} weight, relax = {self.relaxation}")
            self.model.restrict_weight(optimal_weight)
            if (solution := self.model.solve(method, external_solver=external_solver)) is not None:
                optimal_model, optimal_weight = solution, get_pauli_weight(
                    solution)
            else:
                return optimal_model, optimal_weight
