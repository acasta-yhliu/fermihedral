import z3

from .solver import DecentSolver
from .model import MajoranaModel

if __name__ == '__main__':
    # enable the parallel support, just try
    z3.set_param("parallel.enable", "true")

    model = MajoranaModel(10, 2 * 10 * 10, True)
    print(model.solve())

    # z3.set_param("timeout", 12000)
    # for i in range(1, 10):
    #     solver = DecentSolver(i)
    #     print(i, "=>", solver.solve())