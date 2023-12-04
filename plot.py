from math import log2

import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy.stats import linregress
import numpy as np

from fermihedral import get_bk_weight

plt.style.use('classic')
plt.rc("font", size=28, family="serif")

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


print("    - plotting small scale")

plt.clf()
plt.figure(figsize=(7, 6))
plt.subplots_adjust(left=0.12, right=0.98, top=0.95, bottom=0.15)
plt.plot(exp_n_modes[:8], exp_avg_bk_weight[:8], label="Bravyi-Kitaev",
         marker="o", markerfacecolor='none', markeredgecolor='brown', color='brown', markersize=12, linewidth=2, markeredgewidth=2)
plt.plot(exp_n_modes[:8], exp_avg_fermihedral_weight[:8],
         label="Full SAT", marker="x", markerfacecolor='none', markeredgecolor='blue', color='blue', markersize=12, linewidth=2, markeredgewidth=2)

improvement = [(bk - fh) / bk for bk,
               fh in zip(exp_avg_bk_weight[:8], exp_avg_fermihedral_weight[:8])]
improvement = sum(improvement) / len(improvement)
print(f"  > average improvement on small scale !!! = {improvement * 100:.2f}%")

bk_a, bk_b, bk_data = log_regress(exp_n_modes[:8], exp_avg_bk_weight[:8])
plt.plot(exp_n_modes[:8], bk_data,
         label=f"{bk_a:.2f}log$_2$(N)+{bk_b:.2f}", color="brown", ls="--", linewidth=1)
fh_a, fh_b, fh_data = log_regress(
    exp_n_modes[:8], exp_avg_fermihedral_weight[:8])
plt.plot(exp_n_modes[:8], fh_data,
         label=f"{fh_a:.2f}log$_2$(N)+{fh_b:.2f}", color="blue", ls="--", linewidth=1)
plt.legend(loc='lower right', bbox_to_anchor=(1, 0),
           ncol=1, fontsize=21)
plt.grid()
plt.ylim(0.5, 3.7)
plt.xlim(0.5, 8.5)
plt.yticks([1, 2, 3])
plt.xlabel("Modes/n")
plt.ylabel("Pauli Weight/n")
plt.savefig("imgs/small-scale-average.pdf")

print("    - plotting plain pauli weight")

exp_n_modes = exp_n_modes[8:]
exp_avg_bk_weight = exp_avg_bk_weight[8:]
exp_avg_fermihedral_weight = exp_avg_fermihedral_weight[8:]

plt.clf()
plt.figure(figsize=(14, 6))
plt.subplots_adjust(left=0.07, right=0.98, top=0.95, bottom=0.15, wspace=0.27)

plt.subplot(1, 2, 1)
plt.plot(exp_n_modes, exp_avg_bk_weight, label="Bravyi-Kitaev",
         marker="o", markerfacecolor='none', markeredgecolor='brown', color='brown', markersize=12, markeredgewidth=2, linewidth=2)
plt.plot(exp_n_modes, exp_avg_fermihedral_weight,
         label="SAT w/o Alg.", marker="x", markerfacecolor='none', markeredgecolor='blue', color='blue', markersize=12, markeredgewidth=2, linewidth=2)
bk_a, bk_b, bk_data = log_regress(exp_n_modes, exp_avg_bk_weight)
plt.plot(exp_n_modes, bk_data, color="brown", ls="--", linewidth=1)
fh_a, fh_b, fh_data = log_regress(exp_n_modes, exp_avg_fermihedral_weight)
plt.plot(exp_n_modes, fh_data, color="blue", ls="--", linewidth=1)
plt.legend(loc="upper left", fontsize=23)
plt.grid()
plt.xlabel("Modes/n")
plt.ylabel("Pauli Weight/n")
plt.yticks([3.0, 4.0, 5.0])
print(
    f"    - !! copy !! expected improvement : {(bk_a - fh_a) * 100 / bk_a:.2f}%")


plt.subplot(1, 2, 2)
improvement = [(exp_avg_bk_weight[i] - exp_avg_fermihedral_weight[i])
               * 100 / exp_avg_bk_weight[i] for i in range(len(exp_n_modes))]
avg_improvement = sum(improvement) / len(improvement)
plt.bar(exp_n_modes, improvement, color="white",
        edgecolor="black", linewidth=1.5)
plt.plot([8] + exp_n_modes + [20], [avg_improvement for _ in range(
    len(exp_n_modes) + 2)], color='brown', linewidth=3)
plt.grid()
plt.xlabel("Modes/n")
plt.ylabel("Improvement/%")
plt.yticks([5, 15, 25])
plt.savefig("imgs/plain-pauli-weight.pdf")

print(f"    - !! copy !! general improvment : {avg_improvement:.2f}%")

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
plt.figure(figsize=(8, 6))
plt.subplots_adjust(left=0.2, right=0.95, top=0.95, bottom=0.15)
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
             markersize=12, linewidth=2, markeredgewidth=2)
    plt.plot(nmodes, [0.25 ** n for _ in range(len(nmodes))],
             color=plt_color, linestyle='--', linewidth=1.5)
    plt.annotate(
        f"$n={n}$", (9.7, data[-1] * 1.19), fontsize=32, color=plt_color)
plt.grid()
plt.yscale('log')
plt.xlabel("Modes/n")
plt.ylabel("Probability")
plt.savefig("imgs/small-scale-distribution.pdf")

print("> generating table for benchmark")

nmodes = []
t_solve_with = []
t_construct_with = []
t_solve_wo = []
t_construct_wo = []

with open("imgs/benchmark.log", "r") as benchmark:
    for line in benchmark.readlines():
        line = line.strip()

        if len(line) == 0:
            continue

        [mode, tag, t_construct, t_solve] = line.split(";")

        mode = int(mode)
        t_construct = float(t_construct)
        t_solve = float(t_solve)

        if tag == "with":
            t_construct_with.append(t_construct)
            t_solve_with.append(t_solve)
            nmodes.append(mode)
        elif tag == "without":
            t_construct_wo.append(t_construct)
            t_solve_wo.append(t_solve)

nsample = len(nmodes)

print("    - plotting")

plt.clf()
fig=plt.figure(figsize=(16, 8))
plt.subplots_adjust(left=0.08, right=0.92, top=0.91, bottom=0.2, wspace=0.35)

plt.subplot(1, 2, 2)
plt.title("(b) Solving", pad=15)
plt.plot(nmodes, t_solve_with, marker="x", markerfacecolor='none', color="brown",
         markersize=15, linewidth=2, label="w/", markeredgewidth=2)
plt.plot(nmodes, t_solve_wo, marker="o", markerfacecolor='none',  markeredgecolor="blue", color="blue",
         markersize=15, linewidth=2, label="w/o", markeredgewidth=2)
plt.legend(loc="lower center", bbox_to_anchor=(
    0.5, -0.32), frameon=False, ncols=2)

plt.yscale("log")
plt.grid()

improve_axis = plt.twinx()
improve_axis.bar(nmodes, [(t_solve_with[i] - t_solve_wo[i]) * 100 / t_solve_with[i]
                 for i in range(nsample)], label="Improvement", color="none", linewidth=1.5)
improve_axis.set_ylabel("Improvement/%", rotation=270, labelpad=35)
improve_axis.set_xlabel("Modes/n")

plt.xticks(nmodes)
plt.xlim(1.5, nmodes[-1] + 0.5)
plt.xlabel("Modes/n")

plt.subplot(1, 2, 1)
plt.title("(a) Constructing", pad=15)
plt.plot(nmodes, t_construct_with, marker="x", markerfacecolor='none', color="brown",
         markersize=15, linewidth=2, label="w/", markeredgewidth=2)
plt.plot(nmodes, t_construct_wo, marker="o", markerfacecolor='none',  markeredgecolor="blue", color="blue",
         markersize=15, linewidth=2, label="w/o", markeredgewidth=2)

plt.ylabel("Time/s", labelpad=-10)
plt.yscale("log")
plt.grid()


def relu(x):
    return x if x > 0 else 0


improve_axis = plt.twinx()
improve_axis.bar(nmodes, [relu((t_construct_with[i] - t_construct_wo[i]) * 100 / t_construct_with[i])
                 for i in range(nsample)], label="Improvement", color="none", linewidth=1.5)
improve_axis.legend(loc="lower center",
                    bbox_to_anchor=(0.5, -0.32), frameon=False)
improve_axis.set_xlabel("Modes/n")

plt.xlabel("Modes/n")
plt.xticks(nmodes)
fig.supxlabel("Modes/n", y=0.1)
plt.xlim(1.5, nmodes[-1] + 0.5)
plt.savefig("imgs/benchmarking.pdf")

print("> plot noisy simulation for fermi-hubbard experiment")


def plot_fermi_hubbard(*pairs: tuple[int, int]):
    fig = plt.figure(figsize=(12, 8))
    fig.subplots_adjust(left=0.1, right=0.96, top=0.95,
                        bottom=0.25, wspace=0.33, hspace=0.2)

    axis_all = fig.subplots(nrows=len(pairs), ncols=2, sharex=True)

    for id, (nrows, ncols) in enumerate(pairs):

        figure_axis = axis_all[id]
        ax1 = figure_axis[0]
        ax2 = figure_axis[1]

        with open(f"imgs/noisy-fermi-hubbard-{nrows}-{ncols}.log", "r") as log:
            lines = log.readlines()

            def read_line(lineno: int):
                return np.array(list(map(float, lines[lineno].strip().split(' '))))

            x = read_line(0)
            standard = read_line(1)
            bk_exp = read_line(2)
            bk_var = read_line(3)

            jw_exp = read_line(6)
            jw_var = read_line(7)

            fh_exp = read_line(10)
            fh_var = read_line(11)

        def plot_with_variance(x, exp, var, label, color, *, sigma=1):
            low = exp - var
            high = exp + var

            ax1.plot(x, exp, color=color, linewidth=2, label=label)
            ax1.plot(x, exp,
                     linewidth=1.5, color=color, linestyle="--")
            ax1.fill_between(x, low, high, color=color, alpha=0.1)
            ax1.plot(x, low, color=color, linewidth=1, alpha=0.4)
            ax1.plot(x, high, color=color, linewidth=1, alpha=0.4)

        plot_with_variance(x, bk_exp, bk_var, "Bravyi-Kitaev", "brown")
        plot_with_variance(x, jw_exp, jw_var, "Jordan-Wigner", "orange")
        plot_with_variance(x, fh_exp, fh_var, "Our Method", "blue")

        std_value = standard[0]

        ax1.annotate(f"$E_0$", (1.5 * 10e-5, std_value + 0.4))

        ax1.set_title(f"${nrows}\\times{ncols}$")
        ax2.set_title(f"${nrows}\\times{ncols}$")

        ax1.plot(x, standard, color="black", linewidth=2)

        ax1.grid()
        ax1.set_xscale("log")
        ax1.tick_params(axis='x', labelrotation=20)
        # ax1.set_ylim(ylims[eigenstate])
        ax1.set_yticks([-2, 2, 6])

        if id == 0:
            ax1.set_ylabel("Energy")
            ax1.yaxis.set_label_coords(-0.17, -0.07)

        def plot_smooth(x, y, color, label):
            ax2.plot(x, y, color=color, linewidth=2, label=label, alpha=0.7)
            ax2.plot(x, gaussian_filter1d(y, 5),
                     linewidth=1.5, color=color, linestyle="-.")

        plot_smooth(x, bk_var, label="Bravyi-Kitaev", color="brown")
        plot_smooth(x, jw_var, label="Jordan-Wigner", color="orange")
        plot_smooth(x, fh_var, label="Full SAT", color="blue")
        ax2.grid()
        ax2.set_xscale("log")
        ax2.tick_params(axis='x', labelrotation=20)
        # ax2.set_ylim(ylims[eigenstate])

        yticks = {(3, 1): [3.6, 3.8, 4.0], (2, 2): [4.3, 4.5, 4.7]}

        ax2.set_yticks(yticks[(nrows, ncols)])
        if id == 0:
            ax2.set_ylabel("$\sigma$ ($\leftarrow$better)")
            ax2.yaxis.set_label_coords(-0.17, -0.07)

    lines, labels = fig.axes[-1].get_legend_handles_labels()
    fig.legend(lines, labels, loc="lower center",
               fontsize=25, ncol=3, bbox_to_anchor=(0.5, 0))
    fig.supxlabel("2-Qubits Gate Error Rate", y=0.11)
    fig.savefig(f"imgs/noisy-fermi-hubbard.pdf")


plot_fermi_hubbard((3, 1), (2, 2))

print("> print noisy simulation result for H2")

fig = plt.figure(figsize=(12, 13))
fig.subplots_adjust(left=0.125, right=0.96, top=0.99,
                    bottom=0.15, wspace=0.4, hspace=0.07)
fig_axis = fig.subplots(nrows=4, ncols=2, sharex=True)

bk_diffs = []
jw_diffs = []
fh_diffs = []

bk_vars = []
fh_vars = []

for eigenstate in range(4):
    bk = []
    jw = []
    fh = []

    with open(f"imgs/noisy-H2-{eigenstate}.log", "r") as log:
        lines = log.readlines()

        def read_line(lineno: int):
            return np.array(list(map(float, lines[lineno].strip().split(' '))))

        x = read_line(0)
        standard = read_line(1)
        bk_exp = read_line(2)
        bk_var = read_line(3)

        jw_exp = read_line(6)
        jw_var = read_line(7)

        fh_exp = read_line(10)
        fh_var = read_line(11)

    ax1, ax2 = fig_axis[eigenstate][0], fig_axis[eigenstate][1]

    def plot_with_variance(x, exp, var, label, color, *, sigma=1):
        low = exp - var
        high = exp + var

        ax1.plot(x, exp, color=color, linewidth=2, label=label)
        ax1.plot(x, exp,
                 linewidth=1.5, color=color, linestyle="--")
        ax1.fill_between(x, low, high, color=color, alpha=0.1)
        ax1.plot(x, low, color=color, linewidth=1, alpha=0.4)
        ax1.plot(x, high, color=color, linewidth=1, alpha=0.4)

    plot_with_variance(x, bk_exp, bk_var, "Bravyi-Kitaev", "brown")
    plot_with_variance(x, jw_exp, jw_var, "Jordan-Wigner", "orange")
    plot_with_variance(x, fh_exp, fh_var, "Full SAT", "blue")

    bk_diffs.extend(bk_exp - standard)
    jw_diffs.extend(jw_exp - standard)
    fh_diffs.extend(fh_exp - standard)

    bk_vars.extend(bk_var)
    fh_vars.extend(fh_var)

    std_value = standard[0]
    exp_value = bk_exp[-5]

    ax1.annotate(f"$E_{eigenstate}$", (0.005, std_value +
                 [0.03, -0.15, -0.15, 0.03][eigenstate]))

    ax1.plot(x, standard, color="black", linewidth=2)

    ax1.grid()
    ax1.set_xscale("log")
    ax1.tick_params(axis='x', labelrotation=20)

    ylims = [(-2.05, -0.95), (-1.7, -0.5), (-1.4, -0.3), (-1, 0)]
    yticks = [[-2.0, -1.5, -1.0], [-1.6, -1.2, -0.8],
              [-1.2, -0.9, -0.6], [-0.9, -0.5, -0.1]]

    ax1.set_ylim(ylims[eigenstate])
    ax1.set_yticks(yticks[eigenstate])

    if eigenstate == 1:
        ax1.set_ylabel("Energy")
        ax1.yaxis.set_label_coords(-0.25, 0.05)

    def plot_smooth(x, y, color, label):
        ax2.plot(x, y, color=color, linewidth=2, label=label, alpha=0.7)
        ax2.plot(x, gaussian_filter1d(y, 5),
                 linewidth=1.5, color=color, linestyle="-.")

    plot_smooth(x, bk_var, label="Bravyi-Kitaev", color="brown")
    plot_smooth(x, jw_var, label="Jordan-Wigner", color="orange")
    plot_smooth(x, fh_var, label="Full SAT", color="blue")
    ax2.grid()
    ax2.set_xscale("log")
    ax2.tick_params(axis='x', labelrotation=20)
    ylims = [(0.12, 0.47), (0.4, 0.52), (0.4, 0.52), (0.1, 0.47)]
    yticks = [[0.2, 0.3, 0.4], [0.4, 0.45, 0.5],
              [0.4, 0.45, 0.5], [0.2, 0.3, 0.4]]
    ax2.set_ylim(ylims[eigenstate])
    ax2.set_yticks(yticks[eigenstate])
    if eigenstate == 1:
        ax2.set_ylabel("$\sigma$ ($\leftarrow$better)")
        ax2.yaxis.set_label_coords(-0.25, 0.05)

lines, labels = fig.axes[-1].get_legend_handles_labels()
fig.legend(lines, labels, loc="lower center",
           fontsize=25, ncol=3, bbox_to_anchor=(0.5, 0))
fig.supxlabel("2-Qubits Gate Error Rate", y=0.065)
fig.savefig(f"imgs/noisy-simulation-H2.pdf")

aveg_bk_diffs = sum(bk_diffs) / len(bk_diffs)
aveg_jw_diffs = sum(jw_diffs) / len(jw_diffs)
aveg_fh_diffs = sum(fh_diffs) / len(fh_diffs)

avg_bk_var = sum(bk_vars) / len(bk_vars)
avg_fh_var = sum(fh_vars) / len(fh_vars)

print((aveg_bk_diffs - aveg_fh_diffs) * 100 / aveg_bk_diffs)
print((avg_bk_var - avg_fh_var) * 100 / avg_bk_var)