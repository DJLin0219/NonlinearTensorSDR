#' Generalized Sliced Inverse Regression (GSIR)
#'
#' This script implements the Generalized Sliced Inverse Regression (GSIR) method 
#' based on the work of Lee, Kuang-Yao, Bing Li, and Francesca Chiaromonte:
#' 
#' *"A general theory for nonlinear sufficient dimension reduction: Formulation and estimation."* 
#' Annals of Statistics (2013): 221-249.
#'
#' The GSIR method is designed to handle nonlinear sufficient dimension reduction
#' by using kernel-based covariance operators, allowing for greater flexibility in 
#' capturing nonlinear relationships between predictors and response variables.
#'
#' This implementation includes:
#' - **GSIR estimation**: Computes the first `r` eigenvectors of the GSIR matrix.
#' - **GSIR prediction**: Projects new test data onto the estimated sufficient dimensions.
#' - **Matrix operations**: Utilities for computing symmetric matrix power and kernel matrices.
#' - **Kernel computations**: Gaussian kernel and discrete kernel methods for continuous and categorical responses.
#'
#' The functions provided can be used to perform sufficient dimension reduction (SDR) 
#' while preserving nonlinear structures in high-dimensional data.
#'######################################################################################################
#' Ensure Matrix Symmetry
#'
#' @param a A square matrix.
#' @return A symmetrized matrix.
symmetry <- function(a) {
  return((a + t(a)) / 2)
}

#' Compute the Operator Norm
#'
#' @param a A square matrix.
#' @return The operator norm (largest eigenvalue).
onorm <- function(a) {
  return(max(abs(eigen(round(symmetry(a), 8))$values)))
}

#' Compute Symmetric Matrix Power
#'
#' @param a A square symmetric matrix.
#' @param alpha The power exponent.
#' @return The powered matrix.
matpower = function(a,alpha){
  a = (a + t(a))/2
  tmp = eigen(a)
  if(length(tmp$values) == 1){
    m <- tmp$vectors%*%(abs(tmp$values)^alpha)%*%t(tmp$vectors)
  } else{
    m <- tmp$vectors%*%diag(abs(tmp$values)^alpha)%*%t(tmp$vectors)
  }
  return(m)}

#' Compute Matrix Power with Eigenvalue Thresholding
#'
#' @param matrix A symmetric matrix.
#' @param power The exponent for matrix power.
#' @param ignore Threshold for small eigenvalues.
#' @return The powered matrix.
mppower <- function(matrix, power, ignore) {
  eig <- eigen(matrix)
  valid_idx <- abs(eig$values) > ignore
  if (sum(valid_idx) == 1) {
    return(eig$vectors[, valid_idx] %*% t(eig$vectors[, valid_idx]) * eig$values[valid_idx]^power)
  } else {
    return(eig$vectors[, valid_idx] %*% diag(eig$values[valid_idx]^power) %*% t(eig$vectors[, valid_idx]))
  }
}

#' Compute Gaussian Kernel Matrix
#'
#' @param x A numeric matrix (n x p).
#' @param x_new A numeric matrix for prediction (m x p).
#' @param complexity Kernel bandwidth parameter.
#' @return The Gaussian kernel matrix.
gram_gauss=function(x,x.new,complexity){
  x=as.matrix(x);x.new=as.matrix(x.new)
  n=dim(x)[1];m=dim(x.new)[1]
  k2=x%*%t(x);k1=t(matrix(diag(k2),n,n));k3=t(k1);k=k1-2*k2+k3
  sigma=sum(sqrt(k))/(2*choose(n,2));gamma=complexity/(2*sigma^2)
  k.new.1=matrix(diag(x%*%t(x)),n,m)
  k.new.2=x%*%t(x.new)
  k.new.3=matrix(diag(x.new%*%t(x.new)),m,n)
  return(exp(-gamma*(k.new.1-2*k.new.2+t(k.new.3))))
}
#' Compute Discrete Kernel Matrix
#'
#' @param y A numeric vector or matrix.
#' @return The discrete kernel matrix.
gram_dis <- function(y) {
  return(outer(y, y, FUN = "==") * 1)
}

#' Generalized Sliced Inverse Regression (GSIR)
#'
#' @param x Predictor matrix (n x p).
#' @param y Response variable (numeric vector or matrix).
#' @param ytype Response type ('continuous' or 'categorical').
#' @param ex Regularization parameter for `Gx` inversion.
#' @param ey Regularization parameter for `Gy` inversion.
#' @param complex_x Kernel complexity parameter for `x`.
#' @param complex_y Kernel complexity parameter for `y`.
#' @param r Number of sufficient dimensions.
#' @return First `r` eigenvectors of GSIR matrix.
gsir <- function(x, y, ytype, ex, ey, complex_x, complex_y, r) {
  
  n <- nrow(x)
  Q <- diag(n) - matrix(1, n, n) / n
  
  # Compute kernel matrices
  Kx <- gram_gauss(x, x, complex_x)
  Ky <- if (ytype == "continuous") gram_gauss(y, y, complex_y) else gram_dis(y)
  
  # Centering kernel matrices
  Gx <- Q %*% Kx %*% Q
  Gy <- Q %*% Ky %*% Q
  
  # Compute inverse matrices
  Gx_inv <- matpower(symmetry(Gx + ex * onorm(Gx) * diag(n)), -1)
  Gy_inv <- if (ytype == "categorical") {
    mppower(symmetry(Gy), -1, 1e-9)
  } else {
    matpower(symmetry(Gy + ey * onorm(Gy) * diag(n)), -1)
  }
  
  # Compute GSIR matrix
  gsir_matrix <- Gx_inv %*% Gx %*% Gy %*% Gy_inv %*% t(Gx_inv %*% Gx)
  
  # Eigen decomposition
  return(eigen(symmetry(gsir_matrix))$vectors[, 1:r])
}

#' Generalized Sliced Inverse Regression (GSIR) Prediction
#'
#' @param x Training data matrix (n x p).
#' @param x_new Test data matrix (m x p).
#' @param y Response variable.
#' @param ytype Response type ('continuous' or 'categorical').
#' @param ex Regularization parameter for `Gx`.
#' @param ey Regularization parameter for `Gy`.
#' @param complex_x Kernel complexity for `x`.
#' @param complex_y Kernel complexity for `y`.
#' @param r Number of sufficient dimensions.
#' @return Projection of new data onto first `r` GSIR directions.
gsir.predict <- function(x, x_new, y, ytype, ex, ey, complex_x, complex_y, r) {
  
  n <- nrow(x)
  Q <- diag(n) - matrix(1, n, n) / n
  
  # Compute kernel matrices
  Kx <- gram_gauss(x, x, complex_x)
  Ky <- if (ytype == "continuous") gram_gauss(y, y, complex_y) else gram_dis(y)
  
  # Centering kernel matrices
  Gx <- Q %*% Kx %*% Q
  Gy <- Q %*% Ky %*% Q
  
  # Compute inverse matrices
  Gx_inv <- matpower(symmetry(Gx + ex * onorm(Gx) * diag(n)), -1)
  Gy_inv <- if (ytype == "categorical") {
    mppower(symmetry(Gy), -1, 1e-9)
  } else {
    matpower(symmetry(Gy + ey * onorm(Gy) * diag(n)), -1)
  }
  
  # Compute GSIR matrix
  gsir_matrix <- Gx_inv %*% Gx %*% Gy %*% Gy_inv %*% t(Gx_inv %*% Gx)
  
  # Compute eigenvectors
  v <- eigen(symmetry(gsir_matrix))$vectors[, 1:r]
  
  # Compute prediction kernel
  Kx_new <- gram_gauss(x, x_new, complex_x)
  
  # Return projected new data
  return(t(t(v) %*% Gx_inv %*% Q %*% Kx_new))
}

