from dataclasses import dataclass
from functools import reduce
from itertools import combinations
from operator import add, ge, gt, le, lt, eq, mul
from typing import Literal, Type

import math
from tqdm import tqdm
from openfermion import FermionOperator, QubitOperator, bravyi_kitaev
from z3 import (And, BitVecVal, BoolRef, Goal, If, Not, Or, Solver, Then, Xor)

from .iterators import PowerSet
from .pauli import PAULIOP_MULT, Pauli
from .satutil import SATSolver


def get_pauli_weight(model: list[str]):
    return sum(map(lambda x: len(x) - x.count("_"), model))


def get_approx_weight(n_modes: int):
    return int((0.65 * math.log2(n_modes) + 0.95) * 2 * n_modes)


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
    def __init__(self, n_modes: int, independence: bool = True) -> None:
        self.nqubits = n_modes
        self.majoranas = [Pauli(n_modes) for _ in range(2 * n_modes)]
        self.goal = Goal()

        # print("> generating constraints for partial algebraic independent")
        n_ops = len(self.majoranas)
        assert n_ops == 2 * n_modes
        n_pairs = math.comb(n_ops, 2)

        if independence:
            for comb in tqdm(PowerSet(self.majoranas, 3), total=(pow(2, 2 * n_modes) - n_ops - n_pairs - 1)):
                zipped = map(lambda x: reduce(Xor, x), zip(
                    *(i.iter_bits() for i in comb)))
                self.goal.add(Or(*zipped))

        # print("> generating constraints for anti-commutativity")
        for sa, sb in tqdm(combinations(self.majoranas, 2), total=n_pairs):
            xors = []
            for a, b in zip(sa, sb):
                xors.append(Or(And(a.bit0, Not(b.bit0), b.bit1), And(a.bit1, b.bit0, Not(
                    b.bit1)), And(b.bit0, Not(a.bit0), a.bit1), And(b.bit1, a.bit0, Not(a.bit1))))
            self.goal.add(reduce(Xor, xors))

        # print("> generating constraints for vaccuum state preservation")
        # for m1, m2 in zip(self.majoranas[::2], self.majoranas[1::2]):
        #     def pairing(op_pair: tuple[PauliOp, PauliOp]):
        #         op1, op2 = op_pair
        #         return Or(And(op1.bit0, Not(op1.bit1), Not(op2.bit0), op2.bit1), And(op1.bit0, op1.bit1, Not(op2.bit0), Not(op2.bit1)))

        #     self.goal.add(reduce(Or, map(pairing, zip(m1, m2))))

    def restrict_weight(self, weight: int, *, relationship: Literal["<", "<=", ">", ">=", "=="] = "<"):
        def to_bitvec(b: BoolRef):
            return If(b, BitVecVal(1, 4 + self.nqubits), BitVecVal(0, 4 + self.nqubits))

        # print(f"> generating constraints for total pauli weight < {weight}")
        weight_constraints = []
        for string in self.majoranas:
            for op in string:
                weight_constraints.append(to_bitvec(Or(op.bit0, op.bit1)))

        relationship_op = {"<": lt, "<=": le,
                           ">": gt, ">=": ge, "==": eq}[relationship]
        self.goal.add(relationship_op(reduce(add, weight_constraints), weight))

    def restrict_hamiltonian_weight(self, occurence: list[tuple[int, ...]], weight: int, nbits: int, *, relationship: Literal["<", "<=", ">", ">=", "=="] = "<"):
        def to_bitvec(b: BoolRef):
            return If(b, BitVecVal(1, nbits), BitVecVal(0, nbits))

        weight_constraints = []
        for product in occurence:
            comb_ops = [self.majoranas[i - 1] for i in product]
            comb_bits = map(lambda x: reduce(Xor, x), zip(
                *(i.iter_bits() for i in comb_ops)))
            for bit in comb_bits:
                bit0, bit1 = bit, next(comb_bits)
                weight_constraints.append(to_bitvec(Or(bit0, bit1)))
        hamiltonian_weight_constraint = reduce(add, weight_constraints)

        relationship_op = {"<": lt, "<=": le,
                           ">": gt, ">=": ge, "==": eq}[relationship]
        self.goal.add(relationship_op(hamiltonian_weight_constraint, weight))

    def _solver_setup(self, *, solver_init: Type[SATSolver], solver_args):
        z3_solver = Solver()
        z3_solver.add(Then('simplify', 'bit-blast',
                      'tseitin-cnf')(self.goal)[0])

        return solver_init(z3_solver.dimacs(), *solver_args)

    def solve_exists(self, *, solver_init: Type[SATSolver], solver_args):
        solver = self._solver_setup(
            solver_init=solver_init, solver_args=solver_args)

        if solver.check():
            model = solver.model()
            return [op.decode_model(model) for op in self.majoranas]
        else:
            return None

    def solve_forall(self, max: int = 30, *, progess: bool = False, solver_init: Type[SATSolver], solver_args):
        solver = self._solver_setup(
            solver_init=solver_init, solver_args=solver_args)

        solutions = []

        while solver.check() and len(solutions) < max:
            if progess:
                print(
                    f"\rfound {len(solutions)}/{max} solutions for {self.nqubits} modes", end="")
            model = solver.model()
            solutions.append([op.decode_model(model)
                              for op in self.majoranas])
            solver.block()
        if progess:
            print("\r                                                                                                           \r", end="")
        return solutions


class DescentSolver:
    def __init__(self, n: int, independence: bool = True) -> None:
        self.n = n
        self.model = MajoranaModel(n, independence)

    def solve(self, *, progress: bool = False, solver_init: Type[SATSolver], solver_args):
        optimal_model, optimal_weight = None, get_approx_weight(self.n) + 1

        while True:
            if progress:
                print(
                    f"\rfound {optimal_weight} weight for {self.n} modes", end="")
            self.model.restrict_weight(optimal_weight)
            if (solution := self.model.solve_exists(solver_init=solver_init, solver_args=solver_args)) is not None:
                optimal_model, optimal_weight = solution, get_pauli_weight(
                    solution)
            else:
                if progress:
                    print(
                        "\r                                                                                                           \r", end="")
                return optimal_model, optimal_weight


@dataclass
class HamiltonianSolver:
    name: str
    n_modes: int
    occurence: list[tuple[int, ...]]

    def solve(self, *, progress: bool = False, solver_init: Type[SATSolver], solver_args):
        def get_hamiltonian_pauli_weight(solution: list[str]):
            def pauli_string_mult(a: str, b: str):
                return "".join(PAULIOP_MULT[i] for i in zip(a, b))

            h_weight = 0
            for product in self.occurence:
                result_op = reduce(pauli_string_mult,
                                   (solution[i - 1] for i in product))
                h_weight += len(result_op) - result_op.count("_")
            return h_weight

        optimal_model, _ = DescentSolver(self.n_modes, False).solve(
            progress=progress, solver_init=solver_init, solver_args=solver_args)
        optimal_weight = get_hamiltonian_pauli_weight(optimal_model)
        suggested_bits = math.floor(math.log2(optimal_weight))

        model = MajoranaModel(self.n_modes, False)

        while True:
            if progress:
                print(
                    f"found {optimal_weight} Hamiltonian Pauli weight ({get_pauli_weight(optimal_model)}) for '{self.name}' ({self.n_modes} modes)")

            model.restrict_hamiltonian_weight(
                self.occurence, optimal_weight, suggested_bits + 2)
            if (solution := model.solve_exists(solver_init=solver_init, solver_args=solver_args)) is not None:
                optimal_model = solution
                optimal_weight = get_hamiltonian_pauli_weight(solution)
            else:
                return optimal_model, optimal_weight

    def get_bk_weight(self):
        # accquire the weight provided by bk transformation, n_modes -> n_qubits
        def get_weight(op: QubitOperator):
            return len(list(op.terms.keys())[0])  # weird

        # create corresponding operators
        majoranas = [FermionOperator(f"[{i}] + [{i}^]") for i in range(self.n_modes)] + [
            FermionOperator(f"[{i}] - [{i}^]") for i in range(self.n_modes)]
        majoranas = [bravyi_kitaev(i, self.n_modes) for i in majoranas]

        weight = 0

        for product in self.occurence:
            result_op = reduce(mul, (majoranas[i - 1] for i in product))
            weight += get_weight(result_op)

        return weight

# class ProgessiveSolver:
#     def __init__(self, pred_model: list[str], extra_Z: int = 0) -> None:
#         # calculate pred modes
#         pred_2n_modes = len(pred_model)

#         assert pred_2n_modes % 2 == 0
#         assert extra_Z <= pred_2n_modes

#         pred_n_modes = pred_2n_modes // 2
#         self.n_modes = pred_n_modes + 1
#         self.extra_Z = extra_Z

#         # append extra Z and I
#         pred_model = [pred_model[i] + ("Z" if i <
#                       extra_Z else "_") for i in range(0, len(pred_model))]

#         self.pred_model = pred_model

#         # encode pred model into boolean
#         ENCODE = {"_": (False, False), "X": (
#             False, True), "Y": (True, False), "Z": (True, True)}
#         pred_model: List[List[Tuple[bool, bool]]] = [[ENCODE[op] for op in i]
#                                                      for i in pred_model]

#         # construct extra pauli strings
#         self.extra_op1 = Pauli(pred_n_modes)  # ****X
#         self.extra_op2 = Pauli(pred_n_modes)  # ****Y

#         # construct constraints
#         self.goal = Goal()

#         # generate anti commutativity constraint
#         def gen_anti_comm(pauli: Pauli):
#             for op in pred_model:
#                 xors = []
#                 for a, b in zip(pauli, op[:-1]):  # except the extra one
#                     # print(a, b)
#                     xors.append(Or(And(a.bit0, Not(b[0]), b[1]), And(a.bit1, b[0], Not(
#                         b[1])), And(b[0], Not(a.bit0), a.bit1), And(b[1], a.bit0, Not(a.bit1))))

#                 if op[-1] == (True, True):  # additional Z
#                     xors.append(True)

#                 self.goal.add(reduce(Xor, xors))

#         gen_anti_comm(self.extra_op1)
#         gen_anti_comm(self.extra_op2)

#         # anti commutativity for extra op1 and op2, note that the last op is different
#         xors = [True]
#         for a, b in zip(self.extra_op1, self.extra_op2):
#             xors.append(Or(And(a.bit0, Not(b.bit0), b.bit1), And(a.bit1, b.bit0, Not(
#                 b.bit1)), And(b.bit0, Not(a.bit0), a.bit1), And(b.bit1, a.bit0, Not(a.bit1))))
#         self.goal.add(reduce(Xor, xors))

#         # generate algebraic independence along extra Z sets

#         extra_Z_group = pred_model[:extra_Z]

#         def flatten_trim(matrix):
#             return matrix[-1]

#         for comb in PowerSet(extra_Z_group, 1):
#             comb = (*map(flatten_trim, comb), list(self.extra_op1.iter_bits()),
#                     list(self.extra_op2.iter_bits()))
#             zipped = map(lambda x: reduce(Xor, x), zip(*comb))
#             self.goal.add(Or(*zipped))

#     def restrict_weight(self, weight: int):
#         assert weight >= self.extra_Z + 2  # Z, ****X, ****Y

#         remain_weight = weight - self.extra_Z - 2

#         def to_bitvec(b: BoolRef):
#             return If(b, BitVecVal(1, 4 + self.n_modes), BitVecVal(0, 4 + self.n_modes))

#         weight_constraints = [to_bitvec(Or(op.bit0, op.bit1))
#                               for op in self.extra_op1]
#         weight_constraints.extend([to_bitvec(Or(op.bit0, op.bit1))
#                                    for op in self.extra_op2])
#         self.goal.add(reduce(add, weight_constraints) < remain_weight)

#     def solve(self, external_solver: SATSolver):
#         solver = Solver()
#         solver.add(Then('simplify', 'bit-blast', 'tseitin-cnf')(self.goal)[0])

#         LOCAL_GOAL_PATH = "./__goal.cnf"
#         LOCAL_MODEL_PATH = "./__model.sol"

#         with open(LOCAL_GOAL_PATH, "w+") as goal_file:
#             goal_file.write(solver.dimacs())

#         external_solver(LOCAL_GOAL_PATH, LOCAL_MODEL_PATH)

#         sat_model = SATModel.from_file(
#             LOCAL_MODEL_PATH, renaming=LOCAL_GOAL_PATH)

#         if sat_model.sat:
#             op1 = self.extra_op1.decode_model(sat_model) + "X"
#             op2 = self.extra_op2.decode_model(sat_model) + "Y"
#             return [*self.pred_model, op1, op2]
#         else:
#             return None
