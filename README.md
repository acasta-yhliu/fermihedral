# Fermihedral

Code for paper _Fermihedral: On the Optimal Compilation for Fermion-to-Qubit Encoding_

## System Requirements

#### Hardware

A Nvidia GPU is required and configured to run noisy simulation using GPU backend. To reduce the time to solve the SAT problems, we recommend to use a CPU with powerful single-core performance since the SAT solver uses only one core. 

#### Operating System

Ubuntu 22.04 tested.

## Dependency

To replicate our experiments, please at least have

1. Python 3.9+ for this project
2. For compiling the SAT solver `kissat`: `gcc` or `clang` and `make` are required. If your are using Ubuntu, then `sudo apt install build-essential`.

## Checklist

* `README.md`: this file
* `requirements.txt`: required Python packages
* `fermihedral/`: the main code of our framework
* `model/`: Hamiltonian patterns of different physical models
* `prepare.py`: script to prepare the environment
* `simulation.ipynb`, `singleshot.ipynb`, `hamiltonian-weight.ipynb`: notebooks to reproduce the result.

## Execute

1. To prepare the environment, execute `python3 prepare.py`, it will create the virtual environment, install packages and `kissat`.

2. For experiments including Figure 3, 4 and 5, please refer to `singleshot.ipynb`. The executation time could be extremely long.

3. For experiments including all noisy simulation, please refer to `simulation.ipynb`.

4. For experiments including Hamiltonian Pauli weight using **Full SAT** or **SAT+Anl.** method, please refer to `hamiltonian-weight.ipynb`.

Note: please make sure `./data` exists so that all generated data and cache files could be placed correctly.

## Cite

```bibtex
@inproceedings{liu2024fermihedral,
author = {Liu, Yuhao and Che, Shize and Zhou, Junyu and Shi, Yunong and Li, Gushu},
title = {Fermihedral: On the Optimal Compilation for Fermion-to-Qubit Encoding},
year = {2024},
isbn = {9798400703867},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3620666.3651371},
doi = {10.1145/3620666.3651371},
abstract = {This paper introduces Fermihedral, a compiler framework focusing on discovering the optimal Fermion-to-qubit encoding for targeted Fermionic Hamiltonians. Fermion-to-qubit encoding is a crucial step in harnessing quantum computing for efficient simulation of Fermionic quantum systems. Utilizing Pauli algebra, Fermihedral redefines complex constraints and objectives of Fermion-to-qubit encoding into a Boolean Satisfiability problem which can then be solved with high-performance solvers. To accommodate larger-scale scenarios, this paper proposed two new strategies that yield approximate optimal solutions mitigating the overhead from the exponentially large number of clauses. Evaluation across diverse Fermionic systems highlights the superiority of Fermihedral, showcasing substantial reductions in implementation costs, gate counts, and circuit depth in the compiled circuits. Real-system experiments on IonQ's device affirm its effectiveness, notably enhancing simulation accuracy.},
booktitle = {Proceedings of the 29th ACM International Conference on Architectural Support for Programming Languages and Operating Systems, Volume 3},
pages = {382–397},
numpages = {16},
keywords = {quantum computing, fermion-to-qubit encoding, formal methods, boolean satisfiability},
location = {La Jolla, CA, USA},
series = {ASPLOS '24}
}
```
