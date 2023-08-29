import z3

from .solver import DecentSolver

if __name__ == '__main__':
    # enable the parallel support, just try
    z3.set_param("parallel.enable", "true")
    z3.set_param("timeout", 12000)
    for i in range(1, 10):
        solver = DecentSolver(i)
        print(i, "=>", solver.solve())