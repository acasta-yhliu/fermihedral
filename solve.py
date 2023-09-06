#!/bin/env python3

from math import comb
from sys import stderr
from typing import Generic, Iterable
from functools import reduce
from itertools import combinations
from typing import Literal, TypeVar
from z3 import Solver, Goal, Xor, Or, And, Not, Then, BoolRef, If, BitVecVal, sat, Bool, ModelRef, set_param
from operator import add
from time import time


T = TypeVar("T")


class PartialPowerSet(Generic[T]):
    def __init__(self, iterable: Iterable[T], threshold_min: int, threshold_max: int) -> None:
        super().__init__()
        self.iterable = iterable
        self.threshold_min = threshold_min
        self.threshold_max = min(threshold_max, len(iterable)) + 1

    def __iter__(self):
        for i in range(self.threshold_min, self.threshold_max):
            for comb in combinations(self.iterable, i):
                yield comb

    def __len__(self):
        return sum((comb(len(self.iterable), i) for i in range(self.threshold_min, self.threshold_max)))


class EncPauliOp:
    _DEC_STRATEGY = {
        (False, False): "_",
        (False, True): "X",
        (True, False): "Y",
        (True, True): "Z"
    }

    _ID = 0

    @staticmethod
    def _id():
        EncPauliOp._ID += 1
        return str(EncPauliOp._ID)

    def __init__(self):
        self.bit0, self.bit1 = Bool(
            EncPauliOp._id()), Bool(EncPauliOp._id())

    def __getitem__(self, key: int):
        return (self.bit0, self.bit1)[key]

    def decode(self, model: ModelRef):
        gate = bool(model[self.bit0]), bool(model[self.bit1])
        return EncPauliOp._DEC_STRATEGY[gate]


class EncPauliStr(Iterable[EncPauliOp]):
    def __init__(self, length: int):
        self.ops = [EncPauliOp() for _ in range(length)]

    def __len__(self):
        return len(self.ops)

    # special method for iterate all bits of all encoded Pauli operators
    def iter_bits(self):
        for op in self.ops:
            yield op.bit0
            yield op.bit1

    def __iter__(self):
        return iter(self.ops)

    def decode(self, model: ModelRef):
        return ''.join((i.decode(model) for i in self.ops))


class MajoranaModel:
    def __init__(self, nqubits: int, max_independent_length: int = 5) -> None:
        self.nqubits = nqubits
        self.majoranas = [EncPauliStr(nqubits) for _ in range(2 * nqubits)]
        self.goal = Goal()

        print("> generating constraints for partial algebraic independent")
        for comb in PartialPowerSet(self.majoranas, 2, max_independent_length):
            zipped = map(lambda x: reduce(Xor, x), zip(
                *(i.iter_bits() for i in comb)))
            self.goal.add(Or(*zipped))

        print("> generating constraints for anti-commutativity")
        for sa, sb in combinations(self.majoranas, 2):
            xors = []
            for a, b in zip(sa, sb):
                xors.append(Or(And(a.bit0, Not(b.bit0), b.bit1), And(a.bit1, b.bit0, Not(
                    b.bit1)), And(b.bit0, Not(a.bit0), a.bit1), And(b.bit1, a.bit0, Not(a.bit1))))
            self.goal.add(reduce(Xor, xors))

        print("> generating constraints for vaccuum state preservation")
        for m1, m2 in zip(self.majoranas[::2], self.majoranas[1::2]):
            def pairing(op_pair: tuple[EncPauliOp, EncPauliOp]):
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
        def calculate_weight(model: list[str]):
            return sum(map(lambda x: len(x) - x.count("_"), model))

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


if __name__ == "__main__":
    set_param("parallel.enable", True)

    model = MajoranaModel(10)

    # with open("model.cnf", "w+") as f:
    #     model.restrict_weight(120)
    #     f.write(model.solve("dimacs"))
     
    model.restrict_weight(120)
    

    start = time()
    model.solve("z3")
    print(time() - start, "s", file=stderr)

    # with open("result.log", "w+") as f:
    #     for i in range(1, 10):
    #         solver = DecentSolver(i)
    #         mapping, pauli_weight = solver.solve("z3")
    #         f.write(f"{i} => {pauli_weight}\n")
    #         print(i, "=>", pauli_weight)
