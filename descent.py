from sys import argv

from fermihedral import DescentSolver
from fermihedral.satutil import Kissat

from math import log2

from matplotlib.pyplot import plot, savefig, legend, grid, title, xlabel, ylabel, clf, bar, xticks, xscale
from scipy.stats import linregress

from fermihedral import get_bk_weight

[_, nmodes, independency] = argv
print(
    f"> solving decenting model for 2 - {nmodes} modes, {nmodes} qubits, independency = {independency}")


def parse_bool(string: str):
    if string == "True" or string == "true" or string == "1":
        return True
    elif string == "False" or string == "false" or string == "0":
        return False


independency = parse_bool(independency)


with open("imgs/descent.csv", "w+") as f:
    with open("logs/descent.log", "w+") as log:
        for i in range(1, int(nmodes) + 1):
            print(f"> solving {i} modes")
            solver = DescentSolver(i, independency)
            print("> start solving")
            solution, weight = solver.solve(progress=True,
                                            solver_init=Kissat, solver_args=[30*60])

            print(f"{i},{weight}", file=f)
            print(f"{i},{solution}", file=log)

print("> plot descent test result")

exp_n_modes = []
exp_fermihedral_weight = []
exp_avg_fermihedral_weight = []

with open('imgs/descent.csv', "r") as decent_result:
    for line in decent_result.readlines():

        line = line.strip()
        if len(line) == 0:
            continue

        mode, weight = map(int, line.split(','))
        exp_n_modes.append(mode)
        exp_fermihedral_weight.append(weight)
        exp_avg_fermihedral_weight.append(weight / mode)

exp_bk_weight = [get_bk_weight(i) for i in exp_n_modes]
exp_avg_bk_weight = [get_bk_weight(i) / i for i in exp_n_modes]


def log_regress(x, y):
    log_x = [log2(i) for i in x]
    a, b, _, _, _ = linregress(log_x, y)
    return [a * i + b for i in log_x]


# plot average first
clf()
plot(exp_n_modes, exp_avg_bk_weight, label="Bravyi-Kitaev",
     marker="o", markerfacecolor='none', color="red")
plot(exp_n_modes, exp_avg_fermihedral_weight, label="Fermihedral",
     marker="x", markerfacecolor='none', color="blue")
xscale('log')
#     plot(exp_n_modes, log_regress(exp_n_modes, exp_avg_bk_weight),
#          label="O(log N) - Bravyi-Kitaev", color="red", ls="--")
#     plot(exp_n_modes, log_regress(exp_n_modes, exp_avg_fermihedral_weight),
#          label="O(log N) - Fermihedral", color="blue", ls="--")

xlabel("modes/n")
xticks([i for i in range(1, exp_n_modes[-1] + 1)])
ylabel("average pauli weight/n")
grid()
legend()
title("average (per op) pauli weight")
savefig("imgs/descent-average.png")

clf()
plot(exp_n_modes, exp_bk_weight, label="Bravyi-Kitaev",
     marker="o", markerfacecolor="none", color="red")
plot(exp_n_modes, exp_fermihedral_weight, label="Fermihedral",
     marker="x", markerfacecolor="none", color="blue")
xlabel("modes/n")
xticks([i for i in range(1, exp_n_modes[-1] + 1)])
ylabel("total pauli weight")
grid()
legend()
title("total pauli weight")
savefig("imgs/descent-total.png")

clf()

percentage = [(exp_bk_weight[i] - exp_fermihedral_weight[i]
               ) * 100 / exp_bk_weight[i] for i in range(len(exp_bk_weight))]

bar(exp_n_modes, percentage, color="blue")
plot(exp_n_modes, [sum(percentage) / len(percentage)
                   for i in range(len(percentage))], color="red")
xlabel("modes/n")
xticks([i for i in range(1, exp_n_modes[-1] + 1)])
ylabel("percentage/%")
grid()
title("reduction of pauli weight")
savefig("imgs/descent-percentage.png")
