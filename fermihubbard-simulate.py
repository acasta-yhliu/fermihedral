from qiskit_nature.second_q.mappers import BravyiKitaevMapper, JordanWignerMapper
from qiskit_nature.second_q.problems import LatticeModelProblem
from qiskit_nature.second_q.mappers.fermionic_mapper import FermionicMapper
from qiskit_nature.second_q.hamiltonians.lattices import BoundaryCondition, SquareLattice
from qiskit_nature.second_q.hamiltonians import FermiHubbardModel
from qiskit.quantum_info import Pauli
from fermihedral.fock import configure_noise, run_qiskit_circuit, compile_fermi_hubbard
from tqdm import tqdm
from dataclasses import dataclass
import numpy as np


class FermihedralMapper(FermionicMapper):
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


bk = BravyiKitaevMapper()
fh = FermihedralMapper()
jw = JordanWignerMapper()


@dataclass
class SimulationData:
    error_q2: list[float]
    actual: list[float]
    estimate: list[float]
    variance: list[float]


def perform_simulation(nrows: int, ncols: int, nshots: int, *, prob1: float = 0.0001, prob2s: list = list(np.arange(0.0001, 0.01, 0.0001))):
    exp_result = {"BK": SimulationData([], [], [], []), "JW": SimulationData([],
                                                                             [], [], []), "Our Method": SimulationData([], [], [], [])}

    # fixed parameter
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

    for mapper, name in [(bk, "BK"), (jw, "JW"), (fh, "Our Method")]:
        circuit, observable, initial_energy = compile_fermi_hubbard(LatticeModelProblem(
            fhm), mapper)
        for prob2 in tqdm(prob2s):
            configure_noise(prob1, prob2)

            estimation, variance = run_qiskit_circuit(
                circuit, observable, nshots)

            exp_result[name].error_q2.append(prob2)
            exp_result[name].actual.append(initial_energy)
            exp_result[name].estimate.append(estimation)
            exp_result[name].variance.append(variance)

    # save data
    with open(f"imgs/noisy-fermi-hubbard-{nrows}-{ncols}.log", "w+") as log:
        for name in ("BK", "JW", "Our Method"):
            print(' '.join(map(str, exp_result[name].error_q2)), file=log)
            print(' '.join(map(str, exp_result[name].actual)), file=log)
            print(' '.join(map(str, exp_result[name].estimate)), file=log)
            print(' '.join(map(str, exp_result[name].variance)), file=log)


perform_simulation(2, 2, 1000)
# perform_simulation(3, 1, 500)
