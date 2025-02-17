#' Kernel Sliced Inverse Regression (KSIR)
#'
#' This script implements the Kernel Sliced Inverse Regression (KSIR) method, 
#' based on the work of Wu, Han-Ming:
#' 
#' *"Kernel sliced inverse regression with applications to classification."*  
#' Journal of Computational and Graphical Statistics 17.3 (2008): 590-610.
#'
#' The KSIR method extends traditional Sliced Inverse Regression (SIR) by 
#' incorporating kernel methods to capture nonlinear structures in the data. 
#' It is particularly useful in classification and regression problems where 
#' the predictor-response relationship is highly nonlinear.
#'
#' This implementation includes:
#' - **Centering and standardization**: Ensures predictors are properly normalized.
#' - **Gaussian kernel computation**: Uses a Gaussian kernel to construct 
#'   the nonlinear covariance operator.
#' - **KSIR estimation**: Computes the first `r` sufficient dimensions using 
#'   eigen-decomposition of the kernel-based covariance matrix.
#'
#' The KSIR method provides a powerful framework for nonlinear sufficient 
#' dimension reduction and has applications in classification and regression tasks.
#' ######################################################################################################
library(stats)
#' Center a Matrix
#'
#' @param x A numeric matrix.
#' @return A centered matrix.
#' @examples
#' mat <- matrix(1:9, 3, 3)
#' centered_mat <- center(mat)
center <- function(x) {
  return(scale(x, center = TRUE, scale = FALSE))
}

#' Compute Matrix Power
#'
#' @param a A symmetric matrix.
#' @param alpha A numeric value representing the power.
#' @return A matrix raised to the given power.
#' @examples
#' mat <- matrix(c(4, 2, 2, 3), 2, 2)
#' powered_mat <- matpower(mat, 0.5)
matpower <- function(a, alpha) {
  
  # Ensure symmetry
  a <- (a + t(a)) / 2
  eig <- eigen(a)
  
  # Compute matrix power
  if (length(eig$values) == 1) {
    m <- eig$vectors %*% (abs(eig$values)^alpha) %*% t(eig$vectors)
  } else {
    m <- eig$vectors %*% diag(abs(eig$values)^alpha) %*% t(eig$vectors)
  }
  
  return(m)
}

#' Kernel Sliced Inverse Regression (KSIR)
#'
#' @param x A numeric matrix of predictors (n x p).
#' @param y A numeric vector or matrix of response variable.
#' @param b Tuning parameter for the Gaussian kernel.
#' @param eps Small positive value to prevent division by zero.
#' @param r The number of sufficient dimensions chosen.
#' @return A matrix containing the first r KSIR directions.
#' @examples
#' x <- matrix(rnorm(1000), nrow = 100, ncol = 10)
#' y <- rnorm(100)
#' ksir_directions <- kir(x, y, b = 1, eps = 1e-6, r = 2)
kir <- function(x, y, b, eps, r) {
  
  n <- length(y)
  p <- ncol(x)
  
  # Center X
  x_centered <- center(x)
  
  # Compute Sigma_{XX}^{-1/2}
  sigma_inv_sqrt <- matpower(var(x), -1/2)
  
  # Standardize X and Y
  x_standardized <- x_centered %*% sigma_inv_sqrt
  y_standardized <- scale(y)
  
  # Compute Gaussian Kernel Gram matrix
  gker=function(b,y){
    n=length(y);k1=y%*%t(y);k2=matrix(diag(k1),n,n)
    return((1/b)*exp(-(k2+t(k2)-2*k1)/(2*b^2)))}
  
  kernel_matrix <- gker(b, y_standardized)
  
  # Compute weight matrix W
  mean_kern <- mean(kernel_matrix %*% rep(1, n))
  denominator <- pmax(kernel_matrix %*% rep(1, n), eps * mean_kern)
  
  # Compute vector V
  exy <- (kernel_matrix %*% x_standardized) / as.numeric(denominator)
  
  # Compute KSIR matrix
  ksir_mat <- t(exy) %*% exy
  
  # Compute the first r eigenvectors
  sir_directions <- sigma_inv_sqrt %*% eigen(ksir_mat)$vectors[, 1:r]
  
  return(sir_directions)
}
