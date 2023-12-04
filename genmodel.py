from itertools import product
from sys import argv

_, model, nmodes = argv
nmodes = int(nmodes)

if model == "fermi-hubbard":
    assert nmodes % 2 == 0, "must be even modes"

    print(f"fermi-hubbard-periodic {nmodes} ac")  # annhilation, creation

    nodes = list(range(nmodes // 2))

    for spin in (1, 2):  # 1 for down, 2 for up
        for i, j in product(nodes, nodes):
            print(f"-{2 * i + spin} {2 * j + spin}")

    for i in nodes:
        up, down = 2 * i + 2, 2 * i
        print(f"-{up} {up} -{down} {down}")
elif model == "syk":
    print(f"syk {nmodes} mj")  # Majorana
    ops = list(range(1, nmodes + 1))
    for i, j, k, l in product(ops, ops, ops, ops):
        print(i, j, k, l)
