#' Sliced Inverse Regression (SIR)
#'
#' This script implements the Sliced Inverse Regression (SIR) method, 
#' originally proposed by Li, Ker-Chau:
#' 
#' *"Sliced inverse regression for dimension reduction."*  
#' Journal of the American Statistical Association 86.414 (1991): 316-327.
#'
#' The SIR method is a classical approach to sufficient dimension reduction (SDR),
#' which estimates the central subspace by computing the principal eigenvectors 
#' of the covariance matrix of the conditional expectations of predictors.
#'
#' This implementation includes:
#' - **Response variable discretization**: Converts continuous responses into h slices.
#' - **SIR computation**: Estimates the first `r` sufficient dimensions.
#' - **Standardization**: Ensures predictors are standardized before applying SIR.
#'
#' The SIR method is particularly useful when the relationship between the predictors 
#' and response variable is nonlinear but remains in a lower-dimensional subspace.
#' It is widely used in regression analysis, feature selection, and dimensionality reduction.
#' 
#' ######################################################################################################
#' Discretize a response variable into h categories
#'
#' @param y A numeric vector, matrix, or data.frame.
#' @param h Integer, number of categories (slices).
#' @return A numeric vector with discretized values.
#' @examples
#' y <- rnorm(100)
#' discretized_y <- discretize(y, 5)
discretize <- function(y, h) {
  if (!is.numeric(y)) stop("Input y must be numeric.")
  if (!is.integer(h) && h <= 1) stop("h must be an integer greater than 1.")
  
  n <- length(y)
  if (!is.vector(y)) {
    n <- nrow(y)
  }
  m <- floor(n / h)
  
  # Add small noise to break ties
  y <- y + 0.00001 * mean(y) * rnorm(n)
  
  # Sort values and define breakpoints
  y_sorted <- sort(y)
  divpt <- numeric(h - 1)
  
  for (i in 1:(h - 1)) {
    divpt[i] <- y_sorted[i * m + 1]
  }
  
  # Discretize y
  y_discrete <- rep(0, n)
  y_discrete[y < divpt[1]] <- 1
  y_discrete[y >= divpt[h - 1]] <- h
  
  for (i in 2:(h - 1)) {
    y_discrete[(y >= divpt[i - 1]) & (y < divpt[i])] <- i
  }
  
  return(y_discrete)
}

#' Perform Sliced Inverse Regression (SIR)
#'
#' @param x A numeric matrix of predictors (n x p).
#' @param y A numeric vector or matrix of response variable.
#' @param h Integer, number of slices for discretization.
#' @param r Integer, number of sufficient dimensions.
#' @param ytype Character, type of response variable ('continuous' or 'categorical').
#' @return A matrix containing the first r SIR directions.
#' @examples
#' x <- matrix(rnorm(1000), nrow = 100, ncol = 10)
#' y <- rnorm(100)
#' sir_directions <- sir(x, y, h = 5, r = 2)
sir <- function(x, y, h, r, ytype = "continuous") {
  p <- ncol(x)
  n <- nrow(x)
  
  # Standardize x using inverse square root of covariance
  signrt <- mppower(var(x), -1/2, 1e-6)
  xc <- scale(x, center = TRUE, scale = FALSE)
  xst <- xc %*% signrt
  
  # Discretize y if continuous
  if (ytype == "continuous") {
    y_discrete <- discretize(y, h)
  } else {
    y_discrete <- y
  }
  
  # Compute slice means and probabilities
  y_labels <- unique(y_discrete)
  slice_prob <- numeric(h)
  slice_means <- matrix(0, nrow = h, ncol = p)
  
  for (i in 1:h) {
    slice_prob[i] <- mean(y_discrete == y_labels[i])
    slice_means[i, ] <- colMeans(x[y_discrete == y_labels[i], , drop = FALSE])
  }
  
  # Compute SIR covariance matrix
  sirmat <- t(slice_means) %*% diag(slice_prob) %*% slice_means
  sirmat <- (sirmat + t(sirmat)) / 2  # Ensure symmetry
  
  # Compute eigenvectors for dimension reduction
  eig_vals <- eigen(sirmat)
  sir_directions <- signrt %*% eig_vals$vectors[, 1:r]
  
  return(sir_directions)
}
