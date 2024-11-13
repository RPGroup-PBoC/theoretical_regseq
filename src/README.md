## `tregseq`

This is a python software module which includes all custom written code needed for this work. The module can be installed locally after downloading or cloning the repository. Therefore navigate into this folder in the command line on your machine and type

`pip install -e .`

Afterwards the package can simply be imported in any python code. 

The package contains the following scripts:
- `wregseq.py`: functions to produce a mutant library with customizable mutation rate and mutational spectrum.
- `simulate.py`: functions to simulate a synthetic dataset, including functions to calculate the probability of RNAP being bound for all six common regulatory architectures.
- `footprint.py`: functions to calculate and plot summary statistics including information footprint and expression shift matrices
- `utils.py`: utility functions.
- `seq_utils.py`: utility functions specifically for handling promoter sequences.
- `solve_utils.py`: utility functions for solving the polynomial needed to calculate chemical potential.
- `mpl_pboc.py`: Sets the Physical Biology of the Cell plotting style for matplotlib plots. Curtesy of Griffin Chure.

The package also contains the following test scripts: `test_simulate.py`, `test_footprint.py`, `test_utils.py`, and `test_seq_utils.py`.