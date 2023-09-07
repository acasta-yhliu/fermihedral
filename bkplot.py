"""
plot the average pauli weight of bravyi-kitaev transformation
"""

from math import log2

from matplotlib.pyplot import plot, savefig
from scipy.stats import linregress

from fermihedral import get_bk_weight

MIN_QUBITS = 1
MAX_QUBITS = 20

x = [i for i in range(MIN_QUBITS, MAX_QUBITS + 1)]
y = [get_bk_weight(i) / i for i in x]
log_x = [log2(i) for i in x]
a, b, _, _, _ = linregress(log_x, y)

plot(x, y)
plot(x, [a * i + b for i in log_x])
savefig("bkplot.png")

print(get_bk_weight(10))