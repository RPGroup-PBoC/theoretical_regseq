This is the Github repository for all the code used in the following paper:

Pan RW, Roeschinger T, Faizi K, Garcia H, Phillips R. Deciphering regulatory architectures from synthetic single-cell expression patterns. bioRxiv. 2024. p. 2024.01.28.577658. doi:10.1101/2024.01.28.577658

## Repository structure

### **`data`** 
Input data needed for constructing synthetic gene expression datasets, including the required energy matrices.

### **`notebooks`** 
The notebooks contain all the code needed to generate the plots for the figures in the paper.

### **`src`**
Source code for custom Python package `tregs`, which contains functions for generating synthetic datasets of various regulatory architectures and plotting summary statistics including information footprints and expression shift matrices.

## Setting up the computational environment

To run the code in this repository, the package `tregs` will need to be installed first. We provide a `conda` environment that contains all the libraries needed for installing and running `tregs`. The environment can be installed by running

```
conda env create -f environment.yml
```

Subsequently, the environment needs to be activated in order to be used. This can be done by

```
conda activate theoretical_regseq
```

In the `theoretical_regseq` environment, navigate to the `src` directory and the `tregs` package can be installed by running

```
pip install -e .
```
