from dataclasses import dataclass
from typing import Literal
from fermihedral.fock import compile_molecule, configure_noise, get_eigenstates, run_qiskit_circuit
from qiskit_nature.second_q.mappers import BravyiKitaevMapper, JordanWignerMapper
from qiskit_nature.second_q.mappers.fermionic_mapper import FermionicMapper
from qiskit.quantum_info import Pauli
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

plt.style.use('classic')
plt.rc("font", size=28)


class FermihedralMapper(FermionicMapper):
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


bk = BravyiKitaevMapper()
fh = FermihedralMapper()
jw = JordanWignerMapper()


@dataclass
class SimulationData:
    error_q2: list[float]
    actual: list[float]
    estimate: list[float]
    variance: list[float]


def perform_simulation(molecule: Literal["H2", "LiH", "Li"], nshots: int, state: int = 0, *, prob1: float = 0.0001, prob2s: list = list(np.arange(0.0001, 0.01, 0.0001))):
    exp_result = {"BK": SimulationData([], [], [], []), "JW": SimulationData([],
                                                                             [], [], []), "Our Method": SimulationData([], [], [], [])}

    atom = {"H2": "H 0 0 0; H 0 0 0.735",
            "LiH": "Li 0.0 0.0 0.0; H 0.0 0.0 2.5", "Li": "Li 0 0 0"}[molecule]

    remove_orbits = {"H2": None, "LiH": [-3, -2], "Li": []}[molecule]

    for prob2 in tqdm(prob2s):
        for mapper, name in [(bk, "BK"), (jw, "JW"), (fh, "Our Method")]:
            circuit, observable, initial_energy = compile_molecule(
                atom, mapper, state, remove_orbits=remove_orbits, charge=(1 if molecule == "Li" else 0))

            configure_noise(prob1, prob2)

            estimation, variance = run_qiskit_circuit(
                circuit, observable, nshots)

            exp_result[name].error_q2.append(prob2)
            exp_result[name].actual.append(initial_energy)
            exp_result[name].estimate.append(estimation)
            exp_result[name].variance.append(variance)

    # save data
    with open(f"imgs/noisy-{molecule}-{state}.log", "w+") as log:
        for name in ("BK", "JW", "Our Method"):
            print(' '.join(map(str, exp_result[name].error_q2)), file=log)
            print(' '.join(map(str, exp_result[name].actual)), file=log)
            print(' '.join(map(str, exp_result[name].estimate)), file=log)
            print(' '.join(map(str, exp_result[name].variance)), file=log)


def perform_solving(molecule: Literal["H2", "LiH"]):
    atom = {"H2": "H 0 0 0; H 0 0 0.735",
            "LiH": "Li 0.0 0.0 0.0; H 0.0 0.0 1.6"}[molecule]
    with open(f"imgs/eigenstate-{molecule}.log", "w+") as log:
        for mapper, name in [(bk, "BK"), (jw, "JW"), (fh, "Our Method")]:
            eigenstates = get_eigenstates(atom, mapper)
            print(" ".join(map(str, eigenstates)), file=log)


# perform_simulation("Li", 10, prob1=0, prob2s=[0], iters=1)
# perform_simulation("LiH", 10, prob1=0, prob2s=[0], iters=1)
for i in range(4):
    print(f">>>>> simulating with eigenstate {i} of H2")
    perform_simulation("H2", 3000, i)
