from qiskit_nature.second_q.problems import LatticeModelProblem
from math import sqrt
from typing import Literal
from qiskit.algorithms.eigensolvers import NumPyEigensolver
from qiskit.algorithms.minimum_eigensolvers import NumPyMinimumEigensolver
from qiskit_nature.second_q.algorithms import GroundStateEigensolver
from qiskit_nature.second_q.algorithms import ExcitedStatesEigensolver
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import PauliEvolutionGate

from qiskit.quantum_info import Statevector
from qiskit_nature.second_q.mappers.fermionic_mapper import FermionicMapper
from qiskit_nature.second_q.operators import FermionicOp
from qiskit_nature.second_q.transformers import FreezeCoreTransformer
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit.synthesis import QDrift
import qiskit_aer.noise as noise
from qiskit_aer.primitives.estimator import Estimator

noise_model = None


def configure_noise(prob_1: float, prob_2: float):
    global noise_model

    # Depolarizing quantum errors
    error_1 = noise.depolarizing_error(prob_1, 1)
    error_2 = noise.depolarizing_error(prob_2, 2)

    # Add errors to noise model
    noise_model = noise.NoiseModel()
    noise_model.add_all_qubit_quantum_error(error_1, ['u3'])
    noise_model.add_all_qubit_quantum_error(error_2, ['cx'])


def run_qiskit_circuit(circuit, observable, shots):
    backend = Estimator(backend_options={"noise_model": noise_model, "device": "GPU"}, transpile_options={
                        "optimization_level": 3, "basis_gates": ["u3", "cx"]}, run_options={"shots": shots})
    result = backend.run(circuit, observable).result()
    return result.values[0], sqrt(result.metadata[0]["variance"])


def fockbasis2fermionic(fockbasis: str, nmodes: str):
    op_list = []
    for id, state in enumerate(fockbasis):
        if state == "1":
            op_list.append(f"+_{id}")

    return FermionicOp({" ".join(op_list): 1.0}, num_spin_orbitals=nmodes)


def qubit_to_fock(nmodes: int, mapper: FermionicMapper):
    # define the matrix, 2^N * 2^N
    entries = 2 ** nmodes
    q2f_map = {"0" * nmodes: "0" * nmodes}

    # fill up the matrix
    for i in range(1, entries):
        # for each Fock basis, get its creation operator form
        fock_basis_str = bin(i)[2:].zfill(nmodes)[::-1]
        fermionic_op = fockbasis2fermionic(fock_basis_str, nmodes)

        # map the creation operators to Pauli strings, then transform the zero state
        pauli_op = mapper.map(fermionic_op)
        new_state = Statevector.from_label(
            "0" * nmodes).evolve(pauli_op).to_dict()

        assert len(new_state) == 1

        q2f_map[fock_basis_str] = list(new_state.keys())[0]

    return q2f_map


def fock_to_qubit(nmodes: int, mapper: FermionicMapper):
    return dict([(value, key) for key, value in qubit_to_fock(nmodes, mapper).items()])


# set up the eigenstate solvers


def filter_criterion(eigenstate, eigenvalue, aux_values):
    return np.isclose(aux_values["ParticleNumber"][0], 2.0) and np.isclose(
        aux_values["Magnetization"][0], 0.0
    )


def generate_solver(mapper, k=4):
    numpy_solver = NumPyEigensolver(k=k, filter_criterion=filter_criterion)
    return ExcitedStatesEigensolver(mapper, numpy_solver)


def get_eigenstates(molecule: str, mapper: FermionicMapper):
    problem = PySCFDriver(molecule).run()
    energy_state = generate_solver(mapper, k=4).solve(problem)
    return energy_state.eigenvalues


def get_problem_groundstate(problem, mapper: FermionicMapper):
    calc = GroundStateEigensolver(mapper, NumPyMinimumEigensolver())
    ground_state = calc.solve(problem)
    return ground_state.eigenvalues[0]


MOLECULE_CACHE = {}


def compile_molecule(molecule: str, mapper: FermionicMapper, state: int = 0, *, time: float = 1.0, remove_orbits: list[int] | None = None, charge: int = 0):
    problem = PySCFDriver(molecule, charge=charge).run()

    if remove_orbits is not None:
        freeze_transformer = FreezeCoreTransformer(
            remove_orbitals=remove_orbits)
        problem = freeze_transformer.transform(problem)

    # print(problem.hamiltonian.second_q_op())
    hamiltonian = mapper.map(problem.hamiltonian.second_q_op())
    nqubits = hamiltonian.num_qubits

    # f2q_mapper = fock_to_qubit(nqubits, mapper)
    # initial_state = f2q_mapper[initial_state]

    prompt = f"{mapper.__class__.__name__}_{molecule}_{state}"
    if prompt not in MOLECULE_CACHE:
        print(prompt, "not found solved result")
        calc = generate_solver(mapper, state + 1)
        ground_state = calc.solve(problem)
        initial_circuit = ground_state.eigenstates[-1][0]
        initial_energy = ground_state.eigenvalues[-1]
        MOLECULE_CACHE[prompt] = (initial_circuit, initial_energy)
        print(
            f"> solved {molecule}, {state} state, {mapper.__class__.__name__}, E = {initial_energy}")
    else:
        initial_circuit, initial_energy = MOLECULE_CACHE[prompt]

    circuit = QuantumCircuit(nqubits, nqubits)
    circuit.append(initial_circuit, range(nqubits))
    circuit.append(PauliEvolutionGate(hamiltonian, time), range(nqubits))
    circuit.measure(range(nqubits), range(nqubits))

    return circuit, hamiltonian, initial_energy


FERMI_HUBBARD_CACHE = {}


def compile_fermi_hubbard(problem: LatticeModelProblem, mapper: FermionicMapper, *, time: float = 1.0):
    hamiltonian = mapper.map(problem.hamiltonian.second_q_op())
    nqubits = hamiltonian.num_qubits

    prompt = f"{mapper.__class__.__name__}"
    if prompt not in FERMI_HUBBARD_CACHE:
        print(prompt, "not found solved result")
        calc = GroundStateEigensolver(mapper, NumPyMinimumEigensolver())
        ground_state = calc.solve(problem)
        initial_circuit = ground_state.eigenstates[-1][0]
        initial_energy = ground_state.eigenvalues[-1]
        FERMI_HUBBARD_CACHE[prompt] = (initial_circuit, initial_energy)
        print(
            f"> solved fermi hubbard, ground state, {mapper.__class__.__name__}, E = {initial_energy}")
    else:
        initial_circuit, initial_energy = FERMI_HUBBARD_CACHE[prompt]

    circuit = QuantumCircuit(nqubits, nqubits)
    circuit.append(initial_circuit, range(nqubits))
    circuit.append(PauliEvolutionGate(hamiltonian, time), range(nqubits))
    circuit.measure(range(nqubits), range(nqubits))

    return circuit, hamiltonian, initial_energy
