from functools import reduce
from itertools import combinations
from typing import Iterable, TypeVar
from z3 import Solver, Xor, Or, And, Not, IntSort, sat
from operator import add

from .util.encode import EncPauliStr, EncPauliOp
from .util.collection import PowerSet

T = TypeVar("T")


class MajoranaModel:
    def __init__(self, nqubits: int, max_weight: int, progess: bool = False) -> None:
        self.majoranas = [EncPauliStr(nqubits) for _ in range(2 * nqubits)]
        self.max_weight = max_weight
        self.solver = Solver()

        # format model

        if progess:
            from tqdm import tqdm

            def progess(iter: Iterable[T], desc: str):
                return tqdm(iter, desc=desc, leave=False)
        else:
            def progess(iter: Iterable[T], _desc):
                return iter

        for comb in progess(PowerSet(self.majoranas), "algebraic independent"):
            if len(comb) > 1:
                zipped = map(lambda x: reduce(Xor, x), zip(
                    *(i.iter_bits() for i in comb)))
                self.solver.add(Or(*zipped))

        for sa, sb in progess(combinations(self.majoranas, 2), "anti-commutativity"):
            xors = []
            for a, b in zip(sa, sb):
                xors.append(Or(And(a.bit0, Not(b.bit0), b.bit1), And(a.bit1, b.bit0, Not(
                    b.bit1)), And(b.bit0, Not(a.bit0), a.bit1), And(b.bit1, a.bit0, Not(a.bit1))))
            self.solver.add(reduce(Xor, xors))

        def p_xyz(p: EncPauliOp):
            return Or(p.bit0, p.bit1)

        def exists_xyz(x):
            return reduce(Or, map(p_xyz, x)) if len(x) != 0 else False

        weights = []
        for string in progess(self.majoranas, "weight"):
            for i in range(len(string.ops)):
                left, op, right = string.ops[:i], string.ops[i], string.ops[i:]
                weights.append(IntSort().cast(Or(p_xyz(op), And(
                    exists_xyz(left), exists_xyz(right)))))
        self.solver.add(reduce(add, weights) <= self.max_weight)

        for m1, m2 in progess(zip(self.majoranas[::2], self.majoranas[1::2]), "vacuum state preservation"):
            def pairing(op_pair: tuple[EncPauliOp, EncPauliOp]):
                op1, op2 = op_pair
                return Or(And(op1.bit0, Not(op1.bit1), Not(op2.bit0), op2.bit1), And(op1.bit0, op1.bit1, Not(op2.bit0), Not(op2.bit1)))

            self.solver.add(reduce(Or, map(pairing, zip(m1, m2))))

    def solve(self):
        if self.solver.check() == sat:
            model = self.solver.model()
            self.solver.add(Or([f() != model[f]
                            for f in model.decls() if f.arity() == 0]))
            return [op.decode(model) for op in self.majoranas]
        else:
            return None


class FermionModel:
    def __init__(self, nfermions: int, nqubits: int, max_weight: int) -> None:
        assert nqubits >= nfermions

        self.majorana_model = MajoranaModel(nqubits, max_weight)
