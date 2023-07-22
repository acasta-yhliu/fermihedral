from operator import add
import z3
from functools import reduce
from itertools import combinations
from tqdm import tqdm
from itertools import combinations
from typing import Generic, Iterable, TypeVar
from z3 import ModelRef, Solver, sat, Or

T = TypeVar("T")


class powerset(Generic[T]):
    def __init__(self, iterable: Iterable[T]) -> None:
        self.iterable = iterable

    def __iter__(self):
        for i in range(0, len(self.iterable) + 1):
            for comb in combinations(self.iterable, i):
                yield comb

    def __len__(self):
        return 2 ** len(self.iterable)


def block_model(s: Solver, m: ModelRef):
    s.add(Or([f() != m[f] for f in m.decls() if f.arity() == 0]))


def all_smt(s: Solver):
    while sat == s.check():
        model = s.model()
        yield model
        block_model(s, model)


class PauliOp:
    def __init__(self, name: str) -> None:
        self.name = name
        self.encoding = (z3.Bool(name + ".0"), z3.Bool(name + ".1"))

    @property
    def _0(self) -> z3.BoolRef:
        return self.encoding[0]

    @property
    def _1(self) -> z3.BoolRef:
        return self.encoding[1]

    def format(self, model: z3.ModelRef) -> str:
        gate = bool(model[self._0]), bool(model[self._1])
        return {(False, False): "I", (False, True): "X",
                (True, False): "Y", (True, True): "Z"}[gate]


class PauliString:
    def __init__(self, length: int, name: str) -> None:
        self.length = length
        self.name = name
        self.ops = [PauliOp(f"{name}[{i}]") for i in range(length)]

    def iter_vars(self):
        for op in self.ops:
            yield op._0
            yield op._1

    def iter_ops(self):
        return iter(self.ops)

    def format(self, model: z3.ModelRef):
        return ''.join(map(lambda x: x.format(model), self.ops))


class FermionModel:
    def __init__(self, n: int, initial_weight: int | None = None) -> None:
        self.n = n
        self.majoranas = [PauliString(n, str(i)) for i in range(2 * n)]

        self.initial_weight = initial_weight

        self.solver = z3.Solver()

    def apply(self):
        self.apply_independent()
        self.apply_anticomm()
        self.apply_weight()

    def apply_independent(self):
        for comb in tqdm(powerset(self.majoranas), desc="algebraic independent constraints", leave=False):
            if len(comb) > 1:
                zipped = zip(*map(lambda x: x.iter_vars(), comb))
                zipped = map(lambda x: reduce(z3.Xor, x), zipped)
                self.solver.add(z3.Or(*zipped))

    def apply_anticomm(self):
        for sa, sb in tqdm(combinations(self.majoranas, 2), desc="anti-commutativity constraints", leave=False):
            xors = []
            for a, b in zip(sa.iter_ops(), sb.iter_ops()):
                xors.append(z3.Or(z3.And(a._0, z3.Not(b._0), b._1), z3.And(a._1, b._0, z3.Not(
                    b._1)), z3.And(b._0, z3.Not(a._0), a._1), z3.And(b._1, a._0, z3.Not(a._1))))
            self.solver.add(reduce(z3.Xor, xors))

    def apply_weight(self):
        def p_xyz(p: PauliOp):
            return z3.Or(p._0, p._1)

        def exists_xyz(x):
            return reduce(z3.Or, map(p_xyz, x)) if len(x) != 0 else False

        weights = []
        for string in tqdm(self.majoranas, desc="weight constraints", leave=False):
            for i in range(len(string.ops)):
                left, op, right = string.ops[:i], string.ops[i], string.ops[i:]
                weights.append(z3.IntSort().cast(z3.Or(p_xyz(op), z3.And(
                    exists_xyz(left), exists_xyz(right)))))
        self.solver.add(reduce(add, weights) <= self.initial_weight)

    def solve(self):
        for model in all_smt(self.solver):
            majorana_str = [op.format(model) for op in self.majoranas]
            yield majorana_str


n_qubits = int(input("Please provide with N quibts: "))
initial_weight = int(input("Please provide the maximum total weight: "))

model = FermionModel(n_qubits, initial_weight)

model.apply()

solutions = list(model.solve())

if len(solutions) != 0:
    print(f"Totally {len(solutions)} solutions. First 10 solutions are:")
    for i in range(10):
        if i < len(solutions):
            print(solutions[i])
else:
    print("Not found.")
