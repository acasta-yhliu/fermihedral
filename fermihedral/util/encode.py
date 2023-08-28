from typing import Iterable
from z3 import Bool, ModelRef


class EncPauliOp:
    _DEC_STRATEGY = {
        (False, False): "I",
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
