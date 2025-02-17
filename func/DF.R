#' Linear Dimension Folding (LDF)
#'
#' This script implements the **Linear Dimension Folding (LDF)** method, 
#' which is designed for reducing the dimensionality of matrix-valued predictors 
#' while preserving their intrinsic structure. 
#'
#' The method finds two projection matrices that map the original predictors 
#' to a lower-dimensional subspace while maximizing the association with 
#' the response variable.
#'
#' ## Reference:
#' This method is motivated by dimension folding techniques in sufficient 
#' dimension reduction and matrix regression models.
#'
#' ## Features:
#' - **Iterative estimation of left and right projection matrices** to reduce 
#'   dimensionality while maintaining statistical efficiency.
#' - **Uses covariance-based transformations** to ensure stable estimation.
#' - **Matrix power computations** for numerical stability.
#'
#' ## Included Functions:
#' - `matpower()`: Computes matrix power for symmetric matrices.
#' - `df()`: Main function to perform dimension folding, iteratively updating
#'   left and right reduction matrices until convergence.
#'
#' ## Applications:
#' - **Matrix-valued data analysis** in regression and classification.
#' - **Tensor dimension reduction** for structured predictors.
#' - **Improving computational efficiency** in high-dimensional problems.
#'
#' This implementation follows an alternating optimization strategy,
#' iteratively updating projection matrices until convergence.
#' ########################################################################
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
#######################LINEAR DIMENSION FOLDING#########################################
#' Linear Dimension Folding
#'
#' This function performs linear dimension folding on input matrices `X` and `Y` to reduce their dimensions.
#'
#' @param X A matrix with dimensions `pl * pr * n`, where `n` is the sample size.
#' @param Y A matrix with dimensions `n * d`, where `n` is the sample size.
#' @param index A list containing indices for each slice.
#' @param p A vector of proportions of elements in each slice relative to `n`.
#' @param h The number of slices.
#' @param epsilon A numeric value specifying the convergence threshold.
#' @param max_iteration The maximum number of iterations.
#' @param pl The number of rows in `X`.
#' @param pr The number of columns in `X`.
#' @param dl The target row dimension for reduction.
#' @param dr The target column dimension for reduction.
#' #'
#' @return A list containing:
#' \item{a}{The left dimension reduction matrix with dimensions `pl * dl`.}
#' \item{b}{The right dimension reduction matrix with dimensions `pr * dr`.}
#'
df <- function(X,Y,index,p,h,epsilon,iteration,pl,pr,dl,dr){
  Sigmavec <- cov(t(X))
  ##Step 1 Initialization##
  a0 <- matrix(data = rnorm(n = pl*dl),nrow = pl,ncol = dl)
  f0 <- matrix(data = rnorm(n = dl*dr*h),nrow = dl*dr,ncol = h)
  a <- a0
  f <- f0
  B <- 1e3
  for (k in 1:iteration) {
  ##Step 2 Compute first vec(b)#
    V1 <- matrix(data = 0,nrow = pl*pr,ncol = h)
    V2 <- list()
    tV2V2 <- matrix(data = 0,nrow = pr*dr,ncol = pr*dr)
    tV2V1 <- matrix(data = 0,nrow = pr*dr,ncol = 1)
    for (i in 1:h) {
      V1[,i] <- 1/p[i]*(mppower(Sigmavec,-0.5,1e-1)%*%(rowMeans(X[,index[[i]]])))
      if(dr > 1){V2[[i]] <- mppower(Sigmavec,0.5,1e-1)%*%(diag(1,pr)%x%(a%*%matrix(data = f[,i],nrow = dl,ncol = dr)))%*%commutation.matrix(pr,dr)}
      if(dr == 1){V2[[i]] <- mppower(Sigmavec,0.5,1e-1)%*%(diag(1,pr)%x%(a%*%matrix(data = f[,i],nrow = dl,ncol = dr)))}
      tV2V1 <- tV2V1+p[i]*t(V2[[i]])%*%V1[,i]
      tV2V2 <- tV2V2+p[i]*t(V2[[i]])%*%V2[[i]]
    }
    b1 <- matrix(data = mppower(as.matrix(tV2V2),-1,1e-1)%*%tV2V1,nrow = pr,ncol = dr)
    ## Step 3 Compute second vec(a)
    tV2V1 <- matrix(data = 0,nrow = pl*dl,ncol = 1)
    tV2V2 <- matrix(data = 0,nrow = pl*dl,ncol = pl*dl)
    V2 <- list()
    for (i in 1:h) {
      V2[[i]] <- mppower(Sigmavec,0.5,1e-1)%*%((b1%*%t(matrix(f[,i],nrow = dl,ncol = dr)))%x%diag(1,pl))
      tV2V2 <- tV2V2+p[i]*t(V2[[i]])%*%V2[[i]]
      tV2V1 <- tV2V1+p[i]*t(V2[[i]])%*%V1[,i]
    }
    a1 <- matrix(data = mppower(as.matrix(tV2V2),-1,1e-1)%*%tV2V1,nrow = pl,ncol = dl)
    ## Step 4 Compute second f
    V2 <- (mppower(Sigmavec,0.5,1e-6)%*%(b1%x%a1))
    f1 <- matrix(data = mppower(t(V2)%*%V2,-1,1e-6)%*%(t(V2)%*%V1),nrow = dl*dr,ncol = h)
    ## Step 5 Stop criterion
    A <- 0
    for (i in 1:h) {
      A <- A+p[i]*norm(mppower(Sigmavec,-0.5,1e-1)%*%(rowMeans(X[,index[[i]]])) - mppower(Sigmavec,0.5,1e-1)%*%(b1%x%a1)%*%f1[,i],"2")^2
    }
    if (abs(A - B) < epsilon) {
      B <- A;a <- a1;b <- b1;f <- f1
      return(list("a" <- a,"b" <- b))
      break
    }
    B <- A;a <- a1;b <- b1;f <- f1
  }
  return(list("a" <- a,"b" <- b))
}