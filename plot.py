from math import log2

import matplotlib.pyplot as plt
import matplotlib
from scipy.stats import linregress

from fermihedral import get_bk_weight

plt.style.use('classic')
matplotlib.rcParams.update({'font.size': 20})

print("> plot small scale cases, to maximum 8 modes")
print("    - loading solutions, required file: imgs/solution.log")

exp_n_modes = []
exp_fermihedral_weight = []
exp_avg_fermihedral_weight = []


with open('imgs/solution.log', "r") as decent_result:
    for line in decent_result.readlines():

        line = line.strip()
        if len(line) == 0:
            continue

        mode, weight = map(int, line.split(';')[:-1])
        exp_n_modes.append(mode)
        exp_fermihedral_weight.append(weight)
        exp_avg_fermihedral_weight.append(weight / (2 * mode))

exp_bk_weight = [get_bk_weight(i) for i in exp_n_modes]
exp_avg_bk_weight = [get_bk_weight(i) / (2 * i) for i in exp_n_modes]


def log_regress(x, y):
    log_x = [log2(i) for i in x]
    a, b, _, _, _ = linregress(log_x, y)
    return a, b, [a * i + b for i in log_x]


print("    - plotting")

plt.clf()
plt.figure(figsize=(8, 6.2))
plt.plot(exp_n_modes[:8], exp_avg_bk_weight[:8], label="Bravyi-Kitaev",
         marker="o", markerfacecolor='none', markeredgecolor='red', color='red', markersize=10, linewidth=1.5)
plt.plot(exp_n_modes[:8], exp_avg_fermihedral_weight[:8],
         label="Fermihedral", marker="x", markerfacecolor='none', markeredgecolor='blue', color='blue', markersize=10, linewidth=1.5)
bk_a, bk_b, bk_data = log_regress(exp_n_modes[:8], exp_avg_bk_weight[:8])
plt.plot(exp_n_modes[:8], bk_data,
         label=f"{bk_a:.2f}log$_2$(N)+{bk_b:.2f}", color="red", ls="--", linewidth=1)
fh_a, fh_b, fh_data = log_regress(
    exp_n_modes[:8], exp_avg_fermihedral_weight[:8])
plt.plot(exp_n_modes[:8], fh_data,
         label=f"{fh_a:.2f}log$_2$(N)+{fh_b:.2f}", color="blue", ls="--", linewidth=1)
plt.legend(loc='lower right', bbox_to_anchor=(1, 0),
           ncol=1, fontsize=20)
plt.grid()
plt.ylim(0.5, 4)
plt.xlim(0.5, 8.5)
plt.xlabel("Modes/n")
plt.ylabel("Pauli Weight/n")
plt.savefig("imgs/small-scale-average.pdf")

print("> plot algebraic indepence probability and distribution")

print("    - loading distribution, required file: imgs/distribution.csv")

nmodes = []
total_deps_prob = []
ndeps_prob = []

with open("imgs/distribution.csv") as log:
    for line in log.readlines():
        line = line.strip()
        if len(line) == 0:
            continue

        nmodes_, deps_, *probs_ = map(float, line.split(' '))
        nmodes.append(nmodes_)
        total_deps_prob.append(deps_)
        ndeps_prob.append(probs_)

print("    - plotting")
plt.clf()
plt.figure(figsize=(8.5, 6.2))
for n, data in enumerate(zip(*ndeps_prob)):
    n = n + 1
    plt_color = (0.1, 0.8 - n / (len(ndeps_prob[0]) + 5), 0.1)

    def filter_indices(predicate, iterable):
        datas = []
        indices = []
        for id, item in enumerate(iterable):
            if predicate(item):
                datas.append(item)
                indices.append(id)
        return datas, indices

    def slice_out(iterable, indices):
        data = []
        for id in indices:
            data.append(iterable[id])
        return data

    data, indices = filter_indices(lambda x: x > 0, data)

    plt.plot(slice_out(nmodes, indices), data, marker="x", markerfacecolor='none', color=plt_color,
             markersize=10, linewidth=1.5)
    plt.plot(nmodes, [0.25 ** n for _ in range(len(nmodes))],
             color=plt_color, linestyle='--', linewidth=1.5)
plt.grid()
plt.yscale('log')
plt.xlabel("Modes/n")
plt.ylabel("Probability")
plt.savefig("imgs/small-scale-distribution.pdf")

# def get_sol_weight(i):
#     return int(fh_a * log2(i) + fh_b)

# plt.clf()
# n = list(range(10, 50))
# orgi_diff = [(get_bk_weight(i) - get_sol_weight(i)) for i in n]
# now_diff = [(get_approx_weight(i) - get_sol_weight(i)) for i in n]
# plt.plot(n, orgi_diff, color="blue")
# plt.plot(n, now_diff, color="red")
# plt.savefig("imgs/approx_weight_benefit.pdf")

# plot average first
# clf()
# plot(exp_n_modes, exp_avg_bk_weight, label="Bravyi-Kitaev",
#      marker="o", markerfacecolor='none', color="red")
# plot(exp_n_modes, exp_avg_fermihedral_weight, label="Fermihedral",
#      marker="x", markerfacecolor='none', color="blue")
# plot(exp_n_modes, log_regress(exp_n_modes, exp_avg_bk_weight),
#      label="O(log N) - Bravyi-Kitaev", color="red", ls="--")
# plot(exp_n_modes, log_regress(exp_n_modes, exp_avg_fermihedral_weight),
#      label="O(log N) - Fermihedral", color="blue", ls="--")
# plot(exp_n_modes, [1.35 * log2(i) + 1.8 for i in exp_n_modes])

# xlabel("modes/n")
# xticks([i for i in range(1, exp_n_modes[-1] + 1)])
# ylabel("average pauli weight/n")
# grid()
# legend()
# title("average (per op) pauli weight")
# savefig("imgs/solution-average.png")

# clf()
# plot(exp_n_modes, exp_bk_weight, label="Bravyi-Kitaev",
#      marker="o", markerfacecolor="none", color="red")
# plot(exp_n_modes, exp_fermihedral_weight, label="Fermihedral",
#      marker="x", markerfacecolor="none", color="blue")
# xlabel("modes/n")
# xticks([i for i in range(1, exp_n_modes[-1] + 1)])
# ylabel("total pauli weight")
# grid()
# legend()
# title("total pauli weight")
# savefig("imgs/solution-total.png")

# clf()

# percentage = [(exp_bk_weight[i] - exp_fermihedral_weight[i]
#                ) * 100 / exp_bk_weight[i] for i in range(len(exp_bk_weight))]

# bar(exp_n_modes, percentage, color="blue")
# plot(exp_n_modes, [sum(percentage) / len(percentage)
#                    for i in range(len(percentage))], color="red")
# xlabel("modes/n")
# xticks([i for i in range(1, exp_n_modes[-1] + 1)])
# ylabel("percentage/%")
# grid()
# title("reduction of pauli weight")
# savefig("imgs/solution-percentage.png")
