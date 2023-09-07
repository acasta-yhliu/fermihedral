from typing import Iterable

from z3 import Bools, ModelRef

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

    def decode_z3(self, model: ModelRef):
        return ''.join((i.decode_z3(model) for i in self.ops))

    def decode_model(self, model: SATModel):
        return ''.join((i.decode_model(model) for i in self.ops))
