from itertools import combinations
from math import comb, pow
from typing import Generic, Iterable, TypeVar

T = TypeVar("T")


class PowerSet(Generic[T]):
    def __init__(self, iterable: Iterable[T], threshold_min: int) -> None:
        super().__init__()
        self.iterable = iterable
        self.threshold_min = threshold_min

    def __iter__(self):
        for i in range(self.threshold_min, len(self.iterable) + 1):
            for comb in combinations(self.iterable, i):
                yield comb
