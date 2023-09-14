"""
plot the average pauli weight of bravyi-kitaev transformation
"""

from math import log2
from os.path import exists

from matplotlib.pyplot import plot, savefig, legend, grid, title, xlabel, ylabel, clf, bar, xticks, xscale
from scipy.stats import linregress

from fermihedral import get_bk_weight

if exists("spill-test.csv"):
    print("> plot spill test result")

    exp_n_modes = 0
    exp_n_spills = []
    exp_weight = []

    with open("spill-test.csv", "r") as spill_result:
        for line in spill_result.readlines():

            line = line.strip()
            if len(line) == 0:
                continue

            exp_n_modes, n_spills, weight = map(int, line.split(','))
            exp_n_spills.append(n_spills)
            exp_weight.append(weight)

    clf()
    plot(exp_n_spills, exp_weight, color="blue",
         marker="x", markerfacecolor='none')
    xlabel("extra qubits/n")
    ylabel("total pauli weight/n")
    grid()
    title(f"total pauli weight with/out extra qubits ({exp_n_modes} modes)")
    savefig("spill-test.png")

if exists("decent-test.csv"):
    print("> plot decent test result")

    exp_n_modes = []
    exp_fermihedral_weight = []
    exp_avg_fermihedral_weight = []
    exp_fermihedral_relax_weight = []
    exp_avg_fermihedral_relax_weight = []

    with open('decent-test.csv', "r") as decent_result:
        for line in decent_result.readlines():

            line = line.strip()
            if len(line) == 0:
                continue

            mode, weight, relax_weight = map(int, line.split(','))
            exp_n_modes.append(mode)
            exp_fermihedral_weight.append(weight)
            exp_avg_fermihedral_weight.append(weight / mode)
            exp_fermihedral_relax_weight.append(relax_weight)
            exp_avg_fermihedral_relax_weight.append(relax_weight / mode)

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
    plot(exp_n_modes, exp_avg_fermihedral_relax_weight, label="Fermihedral (Relaxed)",
         marker="x", markerfacecolor="none", color="green")
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
    savefig("decent-test-avg.png")

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
    savefig("decent-test-ttl.png")

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
    savefig("decent-test-per.png")
