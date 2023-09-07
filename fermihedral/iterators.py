from typing import Generic, Iterable, TypeVar
from itertools import combinations
from math import comb, pow

T = TypeVar("T")


class Combination(Generic[T]):
    def __init__(self, iterable: Iterable[T], items: int) -> None:
        super().__init__()
        self.iterable = iterable
        self.n_items = items

    def __iter__(self):
        return combinations(self.iterable, self.n_items)

    def __len__(self):
        return pow(2, len(self.iterable))


class PowerSet(Generic[T]):
    def __init__(self, iterable: Iterable[T], threshold_min: int, threshold_max: int) -> None:
        super().__init__()
        self.iterable = iterable
        self.threshold_min = threshold_min
        self.threshold_max = min(threshold_max, len(iterable)) + 1

    def __iter__(self):
        for i in range(self.threshold_min, self.threshold_max):
            for comb in Combination(self.iterable, i):
                yield comb

    def __len__(self):
        return sum((comb(len(self.iterable), i) for i in range(self.threshold_min, self.threshold_max)))
