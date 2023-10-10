from itertools import combinations, combinations_with_replacement, product
from sys import argv

_, nmodes = argv
nmodes = int(nmodes)

print(f"electronic-structure {nmodes}")

ops = list(range(1, nmodes + 1))

for i, j in product(ops, repeat=2):
    print(f"-{i} {j}")

for i, j, k, l in product(ops, repeat=4):
    print(f"-{i} -{j} {k} {l}")
