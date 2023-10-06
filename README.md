# Fermihedral

## Introduction

## Run

First, setup a virtual environment and activate:

```bash
python3 -m venv venv
source venv/bin/activate
```

Then, install necessary Python packages:

```bash
pip3 install -r requirements.txt
```

Run the experiment, `singleshot.py` to solve under a certain mode with _plain Pauli weight_ constraint and `hamiltonian.py` with _Hamiltonian Pauli weight_ constraint. `probability.py` uses the data to test the distribution of $A_k$ described in the paper.

Finally, use `plot.py` to produce figures.

## SAT Solver

To run the experiment, you have to have `kissat` in your path. You could build it compiling from source code or binary distribution via github.
