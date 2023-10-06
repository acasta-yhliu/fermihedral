from functools import reduce
from itertools import combinations
from math import comb
from operator import and_, xor
from typing import Iterable, List

from z3 import Bools, ModelRef

from .iterators import PowerSet
from .satutil import SATModel

_GLOBAL_BOOLEAN_ID = 0


def _new_id():
    global _GLOBAL_BOOLEAN_ID
    _GLOBAL_BOOLEAN_ID += 1
    return _GLOBAL_BOOLEAN_ID


_PAULIOP_DECODE = {
    (False, False): "_",
    (False, True): "X",
    (True, False): "Y",
    (True, True): "Z", }


class PauliOp:
    def __init__(self):
        self.id1, self.id2 = _new_id(), _new_id()
        self.bit0, self.bit1 = Bools(f"{self.id1} {self.id2}")

    def __getitem__(self, key: int):
        return (self.bit0, self.bit1)[key]

    def decode_z3(self, model: ModelRef):
        gate = bool(model[self.bit0]), bool(model[self.bit1])
        return _PAULIOP_DECODE[gate]

    def decode_model(self, model: SATModel):
        gate = model[self.id1], model[self.id2]
        return _PAULIOP_DECODE[gate]


class Pauli(Iterable[PauliOp]):
    def __init__(self, length: int):
        self.ops = [PauliOp() for _ in range(length)]

    def __len__(self):
        return len(self.ops)

    # special method for iterate all bits of all encoded Pauli operators
    def iter_bits(self):
        for op in self.ops:
            yield op.bit0
            yield op.bit1

    def __iter__(self):
        return iter(self.ops)

    def __getitem__(self, key) -> PauliOp:
        return self.ops[key]

    def decode_z3(self, model: ModelRef):
        return ''.join((i.decode_z3(model) for i in self.ops))

    def decode_model(self, model: SATModel):
        return ''.join((i.decode_model(model) for i in self.ops))


def check_algebraic_independent(solution: List[str], tail: int):
    def test_group(group: List[List[bool]]):
        deps = []
        for row in zip(*group):
            x, y, z = row.count("X") % 2, row.count(
                "Y") % 2, row.count("Z") % 2
            deps.append((x == 1 and y == 1 and z == 1)
                        or (x == 0 and y == 0 and z == 0))
        return reduce(and_, deps), deps

    def count(iterable, obj):
        n = 0
        for i in iterable:
            if i == obj:
                n += 1
        return n

    def mean(iterable):
        return 0 if len(iterable) == 0 else sum(iterable) / len(iterable)

    dependent = False
    dists = [[] for _ in range(tail)]
    for i in PowerSet(solution, 3):
        dependent_, truth_table = test_group(i)
        dependent = dependent or dependent_
        nlen = len(truth_table)
        for i in range(1, tail + 1):
            dists[i - 1].append(0 if comb(nlen, i) == 0 else count(map(lambda x: reduce(and_, x),
                                combinations(truth_table, i)), True) / comb(nlen, i))

    return dependent, [mean(i) for i in dists]
