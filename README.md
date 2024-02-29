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

