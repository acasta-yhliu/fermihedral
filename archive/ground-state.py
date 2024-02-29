import matplotlib.pyplot as plt
import numpy as np
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit.algorithms.minimum_eigensolvers import NumPyMinimumEigensolver
from qiskit_nature.second_q.algorithms import GroundStateEigensolver
from qiskit_nature.second_q.mappers.fermionic_mapper import FermionicMapper
from qiskit_nature.second_q.mappers import BravyiKitaevMapper, JordanWignerMapper
from qiskit.quantum_info import Pauli
from qiskit_nature.second_q.hamiltonians import FermiHubbardModel
from qiskit_nature.second_q.hamiltonians.lattices import BoundaryCondition, SquareLattice
from qiskit_nature.second_q.problems import LatticeModelProblem

import qiskit_nature
qiskit_nature.settings.use_pauli_sum_op = False

plt.style.use('classic')
plt.rc("font", size=28, family="serif")


def get_problem_groundstate(problem, mapper: FermionicMapper):
    calc = GroundStateEigensolver(mapper, NumPyMinimumEigensolver())
    ground_state = calc.solve(problem)
    return ground_state.eigenvalues[0]


class FermiHubbardFermihedral(FermionicMapper):
    SOLUTIONS = {6: ['__Y___', '__X___', 'ZZZYZZ', 'ZZZXZZ', 'Y_Z__Z',
                     'X_Z__Z', 'ZYZ_ZZ', 'ZXZ_ZZ', '__Z__Y', '__Z__X', 'Z_Z_YZ', 'Z_Z_XZ'],
                 8: ['_ZZ__YZ_', '_ZZ__XZ_', '_ZZ_YZZ_', '_ZZ_XZZ_', '_YZ___Z_', '_XZ___Z_', 'ZZZYZZZ_', 'ZZZXZZZ_', '__Y_____', '__X_____', 'YZZ_ZZZ_', 'XZZ_ZZZ_', '__Z___Y_', '__Z___X_', 'ZZZZZZZY', 'ZZZZZZZX']}

    @classmethod
    def get_solution(cls, nmodes: int):
        assert nmodes in cls.SOLUTIONS, f"undefined fermion-to-qubit mapping of {nmodes} modes"

        return [string.replace("_", "I") for string in cls.SOLUTIONS[nmodes]]

    @classmethod
    def pauli_table(
        cls, register_length: int, *, _: int | None = None
    ) -> list[tuple[Pauli, Pauli]]:
        solution = cls.get_solution(register_length)

        tables = []
        for j in range(register_length):
            tables.append((Pauli(solution[2 * j]), Pauli(solution[2 * j + 1])))

        return tables


class MoleculeFermihedral(FermionicMapper):
    SOLUTIONS = {8: ['_Z__Z_Z_', 'Z___Z_X_', '___ZYY__', '____X__X', '_Y__Z_Z_', '____X__Z', 'Y___Z_X_', 'X___Z_X_', '_X__Z_Z_', '__Y_X__Y', '__Z_X__Y', '___ZYZ__', '___ZYX__', '____Z_Y_', '___YY___', '___XY___'],
                 6: ['Z__Y__', 'Y____Y', 'Y____Z', 'Z__X__', 'Z__Z__', 'XZX___', 'XYX___', 'Y____X', 'X_Y_Y_', 'XXX___', 'X_Z___', 'X_Y_Z_'],
                 4: ['XZ_X', 'YZ_X', 'ZZZX', '_ZZY', 'ZZX_', 'Z_Y_', 'ZXXZ', 'ZYXZ'],
                 2: ['YX', 'XX', '_Z', '_Y'],
                 12: ['____Y____Z__', 'YY__X_______', 'Z___X__Y____', '____YX___X__', '__ZZZ_______', '__X_Z_X_____', 'YZ__X_______', '__Y_Z_____X_', '____YZ___X__', '__ZXZ_______', 'X___X______Y', 'X___X______X', '__Y_Z_____Z_', 'YX__X_______', '____Y___YY__', '____Y___XY__', '____Y___ZY__', '____YY___X__', '__Y_Z_____Y_', '__X_Z_Y_____', '__X_Z_Z_____', 'X___X______Z', '__ZYZ_______', 'Z___X__X____'],
                 10: ['_____XY__X', '___X_XZ___', '_____XY__Y', '_____XY__Z', '_____XX_X_', '_____XX_Y_', '_____Z_Z__', '_____XX_Z_', '___Y_XZ___', '___Z_XZ___', '_X__XY____', '_ZY__Y____', 'YY___Y____', 'ZY___Y____', '_ZX__Y____', '_ZZ__Y____', '_____Z_X__', '_____Z_Y__', '_X__YY____', 'XY___Y____']}

    @classmethod
    def get_solution(cls, nmodes: int):
        assert nmodes in cls.SOLUTIONS, f"undefined fermion-to-qubit mapping of {nmodes} modes"

        return [string.replace("_", "I") for string in cls.SOLUTIONS[nmodes]]

    @classmethod
    def pauli_table(
        cls, register_length: int, *, _: int | None = None
    ) -> list[tuple[Pauli, Pauli]]:
        solution = cls.get_solution(register_length)

        tables = []
        for j in range(register_length):
            tables.append((Pauli(solution[2 * j]), Pauli(solution[2 * j + 1])))

        return tables


class Problem:
    def __init__(self, name: str) -> None:
        self.mappers = [(BravyiKitaevMapper(), "bk"),
                        (JordanWignerMapper(), "jw")]
        self.ground_state = {}
        self.problem = None
        self.name = name

    def solve(self):
        for mapper, name in self.mappers:
            self.ground_state[name] = get_problem_groundstate(
                self.problem, mapper)
        return self.ground_state


class MoleculeProblem(Problem):
    def __init__(self, name: str, molecule: str, charge: int = 0) -> None:
        super().__init__(name)
        self.problem = PySCFDriver(molecule, charge=charge).run()
        self.mappers.append((MoleculeFermihedral(), "fh"))


class FermiHubbardProblem(Problem):
    def __init__(self, nrows: int, ncols: int) -> None:
        super().__init__(f"${nrows}\\times{ncols}$")

        boundary_condition = BoundaryCondition.PERIODIC
        t = -1.0
        v = 0.0
        u = 5.0

        square_lattice = SquareLattice(
            rows=nrows, cols=ncols, boundary_condition=boundary_condition)
        fhm = FermiHubbardModel(
            square_lattice.uniform_parameters(
                uniform_interaction=t,
                uniform_onsite_potential=v,
            ),
            onsite_interaction=u,
        )

        self.problem = LatticeModelProblem(fhm)
        self.mappers.append((FermiHubbardFermihedral(), "fh"))


problem_H2 = MoleculeProblem("$H_2$", "H 0 0 0; H 0 0 0.735")
problem_LiH = MoleculeProblem("$LiH$", "Li 0 0 0; H 0 0 1.6")
problem_Li = MoleculeProblem("$Li+$", "Li 0 0 0", charge=1)
problem_Na = MoleculeProblem("$He$", "He 0 0 0")
problem_31 = FermiHubbardProblem(3, 1)
problem_22 = FermiHubbardProblem(2, 2)


def save_problems(*problems: Problem):
    with open("imgs/groundstate.log", "w+") as log:
        for problem in problems:
            print("currently solving", problem.name)
            eigen_e = problem.solve()
            e_bk = eigen_e["bk"].real
            e_jw = eigen_e["jw"].real
            e_fh = eigen_e["fh"].real

            nearest_decimal = round(e_bk * 10) / 10
            print(f"{problem.name} {e_bk} {e_jw} {e_fh} {nearest_decimal}", file=log)


save_problems(problem_H2, problem_LiH, problem_Li,
              problem_Na, problem_31, problem_22)
