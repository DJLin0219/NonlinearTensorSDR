# Structure-Preserving Nonlinear Sufficient Dimension Reduction

This repository contains the implementation of the paper Structure-Preserving Nonlinear Sufficient Dimension Reduction for Tensor Regression. The paper introduces novel nonlinear sufficient dimension reduction (SDR) methods specifically designed for tensor regression and classification problems. It employs a Tensor Product Space framework within multiple Reproducing Kernel Hilbert Spaces (RKHS) and proposes Tucker and CANDECOMP/PARAFAC (CP) Tensor Envelope frameworks.

## Features
- Implements Nonlinear Dimension Folding (NDF) using kernel-based sufficient dimension reduction.
- Establishes a relationship between the Conventional SDR Subspace and the Tensor Envelope Subspace.
- Develops two optimization algorithms leveraging Tucker and CP Tensor Decomposition.
- Implements both population-level and sample-level estimations using a Coordinate Mapping approach.
- Provides comparison with other dimension reduction techniques such as PCA, UMAP, t-SNE, LDA, SIR, and GSIR.
- Evaluates the performance through simulations and real-world EEG & CSIQ data analysis.




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
## Results
- Simulation Studies: Demonstrate the effectiveness of our methods in estimating the true subspace structure and outperforming existing approaches.
- EEG Data Analysis: Shows the improved classification accuracy of our method in distinguishing alcoholic vs. non-alcoholic subjects.
- CSIQ Data Analysis: Provides better Pearson correlation values compared to GSIR when assessing image quality distortions.

## Citation
If you use this code in your research, please cite (The paper will be posted to arxiv soon).

## Contact

For any questions or issues, please open an issue in this repository or contact dzl5618@psu.edu.

## License
This project is licensed under the MIT License.
