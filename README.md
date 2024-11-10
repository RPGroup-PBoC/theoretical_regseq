#

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
conda env create -f src/environment.yml
```

Subsequently, the environment needs to be activated in order to be used. This can be done by

```
conda activate tregs
```

and the environment and package is ready to use. If Jupyter is used to run the notebooks, 
a kernel will have to be installed as well. Run

```
python -m ipykernel install --name tregs_kernel
```

and choose `tregs_kernel` as kernel for the notebooks.

# License Information
<img src="https://licensebuttons.net/l/by/3.0/88x31.png"> This work is
licensed under a [Creative Commons CC-BY 4.0 Attribution license](https://creativecommons.org/licenses/by/4.0/). All
software is issued under the standard MIT license which is as follows:

```
Copyright 2024, The authors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
