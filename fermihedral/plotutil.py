from typing import Tuple
import matplotlib.pyplot as plt
import matplotlib
from numpy import arange

plt.style.use('classic')
matplotlib.rcParams.update({'font.size': 18})

X_INCHES = 8.2
RATIO = 16 / 11
Y_INCHES = X_INCHES / RATIO

class Figure:
    def __init__(self, title: str) -> None:
        self.title = title
        plt.clf()
        plt.figure(figsize=(X_INCHES, Y_INCHES))

    def save(self, filename: str):
        plt.legend()
        plt.grid()
        plt.xlabel(self.xlabel)
        plt.xticks(arange(*self.xscale))
        plt.ylabel(self.ylabel)
        plt.yticks(arange(*self.yscale))
        plt.title(self.title)
        plt.savefig(filename)

    def set_x_axis(self, x, label: str, scale: Tuple[int, int, int]):
        self.x = x
        self.xlabel = label
        self.xscale = scale

    def set_y_axis(self, label: str, scale: Tuple[int, int, int]):
        self.ylabel = label
        self.yscale = scale

    def plot(self, y, label: str, marker: str, color: str):
        plt.plot(self.x, y, label=label, marker=marker,
                 color=color, markerfacecolor='none', markersize=8, linewidth=2, markeredgewidth=2, markeredgecolor=color)

    def line(self, y):
        labels = list(arange(*self.xscale))
        plt.plot(labels, [y for _ in range(len(labels))],
                 color="black", linewidth=1.6, linestyle='--')
