# Structure-Preserving Nonlinear Sufficient Dimension Reduction

This repository contains the implementation of the Structure-Preserving Nonlinear Sufficient Dimension Reduction. The method introduces tensor-based sufficient dimension reduction techniques using Tucker and CP decompositions to preserve the intrinsic structure of tensor predictors while effectively reducing dimensionality.

## Features
- Implements Nonlinear Dimension Folding (NDF) using kernel-based sufficient dimension reduction.
- Supports different kernel types including Gaussian and Discrete kernels.
- Provides comparison with other dimension reduction techniques such as PCA, UMAP, t-SNE, LDA, SIR, and GSIR.
- Includes classification accuracy computations for various datasets.

## Dependencies
The following R packages are required:
- `matrixcalc`
- `MASS`
- `class`
- `umap`
- `Rtsne`
- `Rcpp`
- `kernlab`
- `expm`
- `doParallel`
- `foreach`

## Installation
1. Clone the repository:
   ```sh
   git clone https://github.com/DJLin0219/NonlinearTensorSDR.git
   cd NonlinearTensorSDR
   ```

2. Install required R packages:
   ```r
   install.packages(c("matrixcalc", "MASS", "class", "umap", "Rtsne", "Rcpp", "kernlab", "expm", "doParallel", "foreach"))
   ```

3. Compile the C++ source files:
   ```r
   file_path <- "/path/to/repository"
   sourceCpp(paste(file_path, "NDF.cpp", sep="/"))
   ```


## Citation
If you use this code in your research, please cite (The paper will be posted to arxiv soon).


## License
This project is licensed under the MIT License.
