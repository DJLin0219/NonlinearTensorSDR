##### Simulation: Classification of Example in Dimension Folding #####

# Load required R packages
library(matrixcalc)  # Matrix operations
library(MASS)        # Multivariate normal distribution functions
library(class)       # k-Nearest Neighbors (k-NN) classification
library(Rcpp)        # C++ integration with R

#### Function to compute trace of a matrix ####
tr <- function(a) {
  return(sum(diag(a)))  # Sum of diagonal elements
}

#### Load C++ Files (Our Method: Nonlinear Dimension Folding) ####
file_path <- "/Users/djlin/Desktop/TensorSDR/func"
source(paste(file_path, "GSIR.R", sep="/"))  # Load Generalized SIR
source(paste(file_path, "SIR.R", sep="/"))   # Load SIR method
source(paste(file_path, "DF.R", sep="/"))    # Load Dimension Folding method
sourceCpp(paste(file_path, "NDF.cpp", sep="/"))  # Load Nonlinear Dimension Folding C++ implementation

########## Figure 1: Simulation with n = 100, p = 5 ##########

# Model parameters
pi <- 0.5          # Class probability
sigma <- sqrt(0.1) # Standard deviation for class 0
tau <- sqrt(1.5)   # Standard deviation for class 1
mu <- 2            # Mean shift for class 1
iteration <- 200   # Number of iterations for optimization

# Sample size and feature dimension
n <- 100           # Number of samples
pl <- pr <- p <- 5 # Feature dimensions
dl <- dr <- d <- 2 # Reduced dimensions

# Generate response variable Y (binary classification)
set.seed(2024)  # Set random seed for reproducibility
Y <- rbinom(n = n, size = 1, prob = pi)  # Random binary labels (0 or 1)

# Generate feature matrix X (initialized as NA array)
X <- array(data = NA, dim = pl * pr)

# Generate features based on class label
for (i in 1:n) {
  if (Y[i] == 0) {  # Class 0
    mean1 <- array(data = 0, dim = pl * pr)  # Mean vector
    Sigma1 <- diag(array(data = 1, dim = pl * pr))  # Identity covariance matrix
    Sigma1[2,2] <- sigma^2  # Modify covariance
    Sigma1[p+1,p+1] <- sigma^2
    X <- cbind(X, mvrnorm(n = 1, mu = matrix(mean1), Sigma = Sigma1))  # Sample from normal distribution
  } else if (Y[i] == 1) {  # Class 1
    mean2 <- array(data = 0, dim = pl * pr)
    mean2[1] <- mu
    mean2[p+2] <- mu
    Sigma2 <- diag(array(data = 1, dim = pl * pr))
    Sigma2[2,2] <- tau^2
    Sigma2[p+1,p+1] <- tau^2
    X <- cbind(X, mvrnorm(n = 1, mu = matrix(mean2), Sigma = Sigma2))
  }
}

# Remove initial NA column
X <- X[,2:(n+1)]  # X is a p^2 x n matrix (each column represents a feature vector)

# Apply a nonlinear transformation
X[1,] <- X[1,] * X[(p+2),]  # Interaction transformation
Y <- Y  # Retain response variable Y (0 or 1)
######Dimension Folding##########
p0 <- sum(Y==0)/n
p1 <- sum(Y==1)/n
p.t <- c(p0,p1)
X0 <- X[,Y=="0"]
X1 <- X[,Y=="1"]
index <- list()
index[[1]] <- which(Y=="0")
index[[2]] <- which(Y=="1")
h <- 2
epsilon <- 1e-9

######
result <- df(X = X,Y = Y,index = index,p = p.t,h = h,epsilon = epsilon,iteration = iteration,pl = pl,pr = pr,dl = dl,dr = dr)
a <- result[[1]]
b <- result[[2]]

##### Visualization #####
# Transform data using Dimension Folding
Xn <- matrix(data = 0, nrow = dl * dr, ncol = n)
for (i in 1:(ncol(X) / 2)) {
  Xn[, i] <- matrix(data = t(a) %*% matrix(X[, i], nrow = pl, ncol = pr) %*% b, nrow = dl * dr)
}

# Create data frame for plotting
data <- data.frame(
  x11 = Xn[1,1:50],
  x21 = Xn[2,1:50],
  x12 = Xn[3,1:50],
  x22 = Xn[4,1:50]
)

# Define colors based on class labels
color_vector <- ifelse(Y > 0, "red", "black")

# Scatter plot of transformed data
pairs(data, col = color_vector)

##### Sufficient Dimension Reduction Methods #####

# SIR (Sliced Inverse Regression)
beta <- sir(x = t(X), y = Y, h = 2, r = 4, ytype = "categorical")
sir.t <- t(X) %*% beta

# Create data frame for visualization
data <- data.frame(
  x11 = sir.t[1:50,1],
  x21 = sir.t[1:50,2],
  x12 = sir.t[1:50,3],
  x22 = sir.t[1:50,4]
)

# Scatter plot
pairs(data, col = color_vector)

# GSIR (Generalized SIR)
gsir.t <- gsir(x = t(X), y = Y, ytype = "categorical", complex_x = 1, complex_y = 1, r = 4, ex = 0.05, ey = 0.05)

# Create data frame for visualization
data <- data.frame(
  x11 = gsir.t[1:50,1],
  x21 = gsir.t[1:50,2],
  x12 = gsir.t[1:50,3],
  x22 = gsir.t[1:50,4]
)

# Scatter plot
pairs(data, col = color_vector)

##### Nonlinear Dimension Folding (Tucker Form) #####

# Estimate kernel matrices
re <- KernelGenerator(X = X, Y = as.matrix(Y), pl = pl, pr = pr, thre = 0.01, iteration = 10)
A <- re$A
Gy <- re$Gy
Gu <- re$Gu
Gv <- re$Gv

# Tune regularization parameter
epsilon_x <- 10^seq(-15,-1,1)
gcv_x <- array(data = 0, dim = length(epsilon_x))
for (i in 1:length(epsilon_x)) {
  gcv_x[i] <- gcvx(As = A, GY = Gy, epsilon_x = epsilon_x[i])
}
epsilon.x <- epsilon_x[which.min(gcv_x)]

##### NSPGSIR Tucker Form #####
result <- NSPGSIRTu(X = X, Y = as.matrix(Y), pl = pl, pr = pr, dl = dl, dr = dr, thre=0.1, max_iter=10,
                    epsilon_u = 0.01, epsilon_v = 0.01, epsilon_x = epsilon.x, kernel_u="gaussian", kernel_v="gaussian",
                    kernel_Y="discrete")

# Extract transformations
f <- result$f
g <- result$g

# Project data onto reduced space
Xn <- NSPGSIR_predict_Tu(X = X, Y = as.matrix(Y), X_new = X, pl = pl, pr = pr, dl = dl, dr = dr, f = f, g = g)

# Visualization
data <- data.frame(
  x11 = Xn[1,1:50],
  x21 = Xn[2,1:50],
  x12 = Xn[3,1:50],
  x22 = Xn[4,1:50]
)

pairs(data, col = color_vector)

######################NDF CP Form################
###Estimation of f,g and h######
result <- NSPGSIRCP(X = X,Y = as.matrix(Y),pl = pl,pr = pr,d=4,thre=0.1,max_iter=10,
                    epsilon_u = epsilon.u,epsilon_v = epsilon.v,epsilon_x = epsilon.x,kernel_u="gaussian",kernel_v="gaussian",
                    kernel_Y="discrete")
f <- result$f
g <- result$g
Xn <- NSPGSIR_predict_CP(X = X,Y = as.matrix(Y),X_new = X,pl = pl,pr = pr,d = 4,f = f,g = g)
data <- data.frame(
  x11 = Xn[1,1:50],
  x21 = Xn[2,1:50],
  x12 = Xn[3,1:50],
  x22 = Xn[4,1:50]
)
# Scatter Plot
pairs(data,col = color_vector)



##########n=50 and p=10 ######
#Model Generating
pi <- 0.5
sigma <- sqrt(0.1)
tau <- sqrt(1.5)
mu <- 2
iteration <- 100
#n=100,200,300,500,800
n <- 50
#p=5,10
pl <- pr <- p <- 10
#d = 2
dl <- dr <- d <- 2
#Generate Y and X
set.seed(2023)
Y <- rbinom(n = n,size = 1,prob = pi)
X <- array(data = NA,dim = pl*pr)
for (i in 1:n) {
  if (Y[i] == 0) {
    mean1 <- array(data = 0,dim = pl*pr)
    Sigma1 <- diag(array(data = 1,dim = pl*pr))
    Sigma1[2,2] <- sigma^2
    Sigma1[p+1,p+1] <- sigma^2
    X <- cbind(X,mvrnorm(n = 1,mu = matrix(mean1),Sigma = Sigma1))
  } else if (Y[i] == 1) {
    mean2 <- array(data = 0,dim = pl*pr)
    mean2[1] <- mu; mean2[p+2] <- mu
    Sigma2 <- diag(array(data = 1,dim = pl*pr))
    Sigma2[2,2] <- tau^2
    Sigma2[p+1,p+1] <- tau^2
    X <- cbind(X,mvrnorm(n = 1,mu = matrix(mean2),Sigma = Sigma2))
  }
}
######
X <- X[,2:(n+1)] #X is a p^2*n matrix, with each column is vec(X_i)
X[1,] <- X[1,]*X[(p+2),]
Y <- Y #Y is a 0-1 number
#####
p0 <- sum(Y==0)/n
p1 <- sum(Y==1)/n
p <- c(p0,p1)
X0 <- X[,Y=="0"]
X1 <- X[,Y=="1"]
index <- list()
index[[1]] <- which(Y=="0")
index[[2]] <- which(Y=="1")
h <- 2
epsilon <- 1e-9

#########Dimension Folding########################
result <- df(X = X,Y = Y,index = index,p = p,h = h,epsilon = epsilon,iteration = iteration,pl = pl,pr = pr,dl = dl,dr = dr)
a <- result[[1]]
b <- result[[2]]
f <- result[[3]]
print(result[[4]])

#####Visualization###
Xn <- matrix(data = 0,nrow = dl*dr,ncol = n)
for (i in 1:ncol(X)) {
  Xn[,i] <- matrix(data = t(a)%*%matrix(X[,i],nrow = pl,ncol = pr)%*%b,nrow = dl*dr)
}
data <- data.frame(
  x11 = Xn[1,],
  x21 = Xn[2,],
  x12 = Xn[3,],
  x22 = Xn[4,]
)
color_vector <- ifelse(Y > 0, "red", "black")

# Scatter Plot
pairs(data,col = color_vector)

###########SIR##################
beta <- sir(x = t(X),y = Y,h = 2,r = 4,ytype = "categorical")
sir.t <- t(X)%*%beta
data <- data.frame(
  x11 = sir.t[1:50,1],
  x21 = sir.t[1:50,2],
  x12 = sir.t[1:50,3],
  x22 = sir.t[1:50,4]
)
# Scatter Plot
pairs(data,col = color_vector)

########GSIR####################
gsir.t <- gsir(x = t(X),y = Y,ytype = "categorical",complex_x = 1,complex_y = 1,r = 4,ex = 0.05,ey = 0.05)
data <- data.frame(
  x11 = gsir.t[1:50,1],
  x21 = gsir.t[1:50,2],
  x12 = gsir.t[1:50,3],
  x22 = gsir.t[1:50,4]
)
# Scatter Plot
pairs(data,col = color_vector)
######################NDF Tucker Form################
####Tuning##########
re <- KernelGenerator(X = X,Y = Y,pl = pl,pr = pr,dl = dl,dr = dr,
                      thre = 0.01,iteration = 10,kernel.u = "gaussian",kernel.v = "gaussian",kernel.Y = "gaussian")
A <- re$A
Gy <- re$Gy
Gu <- re$Gu
Gv <- re$Gv
epsilon_x <- 10^seq(-15,-1,1)
gcv_x <- array(data = 0,dim = length(epsilon_x))
for (i in 1:length(epsilon_x)) {
  gcv_x[i] <- gcvx(As = A,GY = Gy,epsilon_x = epsilon_x[i])
}
epsilon.x <- epsilon_x[which.min(gcv_x)]
epsilon_u <- 10^seq(-3,0,0.5)
epsilon_v <- 10^seq(-3,0,0.5)
gcv_uv <- matrix(data = 0,nrow = length(epsilon_u),ncol = length(epsilon_v))
for (i in 1:length(epsilon_u)) {
  for (j in 1:length(epsilon_v)) {
    gcv_uv[i,j] <- gcvuv(As = A,Gu = Gu,Gv = Gv,GY = Gy,epsilon_x = epsilon.x,epsilon_u = epsilon_u[i],epsilon_v = epsilon_v[j])
  }
}
epsilon.u <- epsilon_u[which(gcv_uv == min(gcv_uv), arr.ind = TRUE)[1]]
epsilon.v <- epsilon_v[which(gcv_uv == min(gcv_uv), arr.ind = TRUE)[2]]
###Estimation of f,g and h######
result <- NSPGSIRTu(X = X,Y = as.matrix(Y),pl = pl,pr = pr,dl = dl,dr = dr,thre=0.1,max_iter=10,
                  epsilon_u = epsilon.v,epsilon_v = epsilon.v,epsilon_x = epsilon.x,kernel.u="gaussian",kernel.v="gaussian",
                  kernel.Y="discrete")
print(result$error)
f <- result$f
g <- result$g
Xn <- NSPGSIR_predict_Tu(X = X,Y = as.matrix(Y),X.new = X,pl = pl,pr = pr,dl = dl,dr = dr,f = f,g = g)
data <- data.frame(
  x11 = Xn[1,],
  x21 = Xn[2,],
  x12 = Xn[3,],
  x22 = Xn[4,]
)
color_vector <- ifelse(Y > 0, "red", "black")
# Scatter Plot
pairs(data,col = color_vector)

######################NDF CP Form################
###Estimation of f,g and h######
result <- NSPGSIRCP(X = X,Y = as.matrix(Y),pl = pl,pr = pr,d=4,thre=0.1,max_iter =10,
                     epsilon_u = epsilon.u,epsilon_v = epsilon.v,epsilon_x = epsilon.x,kernel_u="gaussian",kernel_v="gaussian",
                     kernel_Y="discrete")
print(result$error)
f <- result$f
g <- result$g
Xn <- NSPGSIR_predict_CP(X = X,Y = as.matrix(Y),X_new = X,pl = pl,pr = pr,d = 4,f = f,g = g)
data <- data.frame(
  x11 = Xn[1,],
  x21 = Xn[2,],
  x12 = Xn[3,],
  x22 = Xn[4,]
)
color_vector <- ifelse(Y > 0, "red", "black")
# Scatter Plot
pairs(data,col = color_vector)


