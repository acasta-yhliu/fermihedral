from itertools import combinations
from typing import Generic, Iterable, TypeVar


T = TypeVar("T")


class PowerSet(Generic[T]):
    def __init__(self, iterable: Iterable[T]) -> None:
        self.iterable = iterable

    def __iter__(self):
        for i in range(0, len(self.iterable) + 1):
            for comb in combinations(self.iterable, i):
                yield comb

    def __len__(self):
        return 2 ** len(self.iterable)
