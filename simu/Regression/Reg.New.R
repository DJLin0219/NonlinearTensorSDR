########Load R package
library(Rcpp)
library(foreach)
library(doParallel)
library(energy)
library(MASS)
library(matrixcalc)
library(kernlab)
library(np)
file_path <- "/Users/djlin/Desktop/TensorSDR/functions"
source(paste(file_path, "GSIR.R", sep="/"))
source(paste(file_path, "KSIR.R", sep="/"))
########## Simulation 1 #########
NDF.Reg <- function(pl,pr,n,Xtype,mu,sd=0.5,epsilon.x,epsilon.u,epsilon.v){
  #####Data Generation And Accuracy computation for Simulation Regression Part Model I #######
  #######Input#########
  ##pl: The target number of rows of the matrix
  ##pr: The target number of columns of the matrix
  ##n: The number of the matrix to generate
  ##Xtype: The type of distribution of X ("I","II" or "III" corresponding to "A","B" and "C" to the paper)
  ##mu: The mean of matrix X, which is a pl*pr matrix
  ##sd: The standard deviation of the noise of Y (controlling SNR)
  ##epsilon.x: The tuning parameter given in Section 6.5 for computing the generalized inverse of A^TA
  ##epsilon.u: The tuning parameter given in Section 6.5 for computing the Gram matrix Gu
  ##epsilon.v: The tuning parameter given in Section 6.5 for computing the Gram matrix Gv
  ##seed: Seed to generate different datasets
  #######Output###
  ##Compute four accuracy which is the distance correlation of predictors given by our methods (both methods share a same result), the distance
  ##correlation of responses given by our methods (both methods share a same result), the distance correlation of responses given by GSIR and 
  ##the distance correlation of responses given by KSIR.
  pl <- pl
  pr <- pr
  dl <- 1
  dr <- 1
  r <- min(pl,pr)
  n <- n
  #target dimensions to reduce, in Model 1, dl=dr=1
  n.t <- 100
  if (Xtype == "I") {
    Mu <- rep(0,pl*pr)
    Mu[1] <- mu
    X <- MASS::mvrnorm(n= (n+n.t),mu = Mu,Sigma = diag(1,pl*pr))
  } else if (Xtype == "II") {
    Mu1 <- rep(-1,pl*pr);
    Mu1[1] <- mu
    Mu2 <- rep(1,pl*pr);
    Mu2[1] <- mu
    X <- 1/2*MASS::mvrnorm(n=(n+n.t),mu = Mu1,Sigma = diag(2,pl*pr))+1/2*MASS::mvrnorm(n= (n+n.t),mu = Mu2,Sigma = diag(2,pl*pr))
  } else if (Xtype == "III"){
    Mu <- rep(0,pl*pr)
    Mu[1] <- mu
    X <- MASS::mvrnorm(n =(n+n.t),mu = Mu,Sigma = 0.8*diag(1,pl*pr)+0.2*rep(1,pl*pr)%*%t(rep(1,pl*pr)))
  }
  u <- matrix(nrow = pl,ncol = 0)
  v <- matrix(nrow = pr,ncol = 0)
  lambda <- c()
  X <- t(X)
  result.train <- SVDs(X = X,pl = pl,pr = pr)
  u.train <- result.train$u
  v.train <- result.train$v
  lambda.train <- result.train$lambda
  f.true <- ((u.train[1,])^3)
  g.true <- sign(v.train[1,])*abs(v.train[1,])^(5)
  Y.n <- array(data = 0,dim = ncol(X))
  for (i in 1:(n+n.t)) {
    #Y.n[i] <- (sum((lambda.train*f.true*g.true)[(r*(i-1)+1):(r*i)]))
    Y.n[i] <- log(1+sum((lambda.train*f.true*g.true)[(r*(i-1)+1):(r*i)]))
  }
  #Y.n <- log(1+Y.n)
  Y <- (Y.n)+rnorm((n+n.t),0,sd)*sd(Y.n)
  ####Training Data and Test Data##########
  X.train <- X[,1:n]
  X.test <- X[,(n+1):(n+n.t)]
  Y.train <- Y[1:n]
  Y.test <- Y[(n+1):(n+n.t)]
  #########################NSPGSIRTu/NSPGSIRCP(Same for two methods)##################
  ####Tuning##########
  epsilon.x <- epsilon.x
  epsilon.u <- epsilon.u
  epsilon.v <- epsilon.v
  ###Estimation of f,g and h######
  result <- NSPGSIRTu(X = X.train,Y = matrix(Y.train, ncol=1),pl = pl,pr = pr,dl = dl,dr = dr,thre=1e-4,max_iter=50,
                    epsilon_u = epsilon.v,epsilon_v = epsilon.v,epsilon_x = epsilon.x,kernel_u="gaussian",kernel_v="gaussian",
                    kernel_Y="gaussian")
  f <- result$f
  g <- result$g
  h <- result$h
  ##Compare f.pred with f.true, g.pred with g.true###
  f.pred <- as.vector(t(f)%*%gram_gauss(t(u.train)[1:(r*n),],t(u.train)[(r*n+1):(r*(n+n.t)),],1))
  g.pred <- as.vector(t(g)%*%gram_gauss(t(v.train)[1:(r*n),],t(v.train)[(r*n+1):(r*(n+n.t)),],1))  
  acc1 <- (abs(dcor(g.true[(r*n+1):(r*(n+n.t))]*f.true[(r*n+1):(r*(n+n.t))],g.pred*f.pred)))
  X.train.d <- NSPGSIR_predict_Tu(X = X.train,Y = matrix(Y.train, ncol=1),X_new = X.train,pl = pl,pr = pr,dl = dl,dr = dr,f = f,g = g)
  data.train <- data.frame("Y" = Y.train,"X" = t(X.train.d))
  kernel.model <- npreg(Y~X,data = data.train, regtype = "ll", bwmethod = "cv.aic", gradients = TRUE)
  X.test.d <- NSPGSIR_predict_Tu(X = X.train,Y = matrix(Y.train, ncol=1),X_new = X.test,pl = pl,pr = pr,dl = dl,dr = dr,f = f,g = g)
  Y.test.pred <- predict(kernel.model, newdata = data.frame("X" = t(X.test.d)))
  acc2 <- (dcor(as.matrix(Y.test.pred),as.matrix(Y[(n+1):(n+n.t)])))
  ############################GSIR#################
  gsir.train <- gsir(x = t(X.train),y = as.matrix(Y.train),ytype = "continuous",ex = 0.01,ey = 0.01,complex_x = 1,complex_y = 1,r = 1)
  gsir.test <- gsir.predict(x = t(X.train),y = as.matrix(Y.train),x_new = t(X.test),ytype = "continuous",ex = 0.01,ey = 0.01,complex_x = 1,complex_y = 1,r = 1)
  data.train.2 <- data.frame("Y" = Y.train,"X" = gsir.train)
  kernel.model.2 <- npreg(Y~X,data = data.train.2, regtype = "ll", bwmethod = "cv.aic", gradients = TRUE)
  Y.test.pred.2 <- predict(kernel.model.2, newdata = data.frame("X" = gsir.test))
  acc3 <- (dcor(as.matrix(Y.test.pred.2),as.matrix(Y[(n+1):(n+n.t)])))
  ############################KSIR#################
  ksir.beta <- kir(x = t(X.train),y = as.matrix(Y.train),b = 1,eps = 0.1,r = 1)
  ksir.train <- t(X.train)%*%ksir.beta
  ksir.test <- t(X.test)%*%ksir.beta
  data.train.3 <- data.frame("Y" = Y.train,"X" = ksir.train)
  kernel.model.3 <- npreg(Y~X,data = data.train.3, regtype = "ll", bwmethod = "cv.aic", gradients = TRUE)
  Y.test.pred.3 <- predict(kernel.model.3, newdata = data.frame("X" = ksir.test))
  acc4 <- (dcor(as.matrix(Y.test.pred.3),as.matrix(Y[(n+1):(n+n.t)])))
  return(c(acc1,acc2,acc3,acc4))
}

#Parallel Computing
cores <- detectCores()
cl <- makeCluster(cores)
registerDoParallel(cl)
iteration <- 200
args <- commandArgs(trailingOnly = TRUE)
iter <- as.numeric(args[1])
print(iter)
results <- foreach(l = 1:iteration, .combine = rbind) %dopar% {
  library(matrixcalc)
  library(MASS)
  library(Rcpp)
  library(foreach)
  library(energy)
  library(doParallel)
  library(kernlab)
  library(np)
  sourceCpp(paste(file_path, "NDFGSIR.cpp", sep="/"))
  epsilon.x <- 1e-6
  epsilon.u <- 0.05
  epsilon.v <- 0.05
  sd <- 0.5
  mu <- 5
  ####We performed manual tuning here. While it would be ideal to tune parameters using GCV,
  ##our results are already sufficiently good. The GCV functions can be found in our CPP files.
  ###########n = 100,pl= 5, pr = 5, Xtype = "I"##########
  acc1 <- NDF.Reg(5,5,100,"I",mu,sd,epsilon.x,epsilon.u,epsilon.v)
  ###########n = 100,pl= 5, pr = 5, Xtype = "II"##########
  acc2 <- NDF.Reg(5,5,100,"II",mu,sd,epsilon.x,epsilon.u,epsilon.v)
  ###########n = 100,pl= 5, pr = 5, Xtype = "III"##########
  acc3 <- NDF.Reg(5,5,100,"III",mu,sd,epsilon.x,epsilon.u,epsilon.v)
  ###########n = 200,pl= 5, pr = 5, Xtype = "I"##########
  acc4 <- NDF.Reg(5,5,200,"I",mu,sd,epsilon.x,epsilon.u,epsilon.v)
  ###########n = 200,pl= 5, pr = 5, Xtype = "II"##########
  acc5 <- NDF.Reg(5,5,200,"II",mu,sd,epsilon.x,epsilon.u,epsilon.v)
  ###########n = 200,pl= 5, pr = 5, Xtype = "III"##########
  acc6 <- NDF.Reg(5,5,200,"III",mu,sd,epsilon.x,epsilon.u,epsilon.v)
  ###########n = 100,pl= 10, pr = 10, Xtype = "I"##########
  acc7 <- NDF.Reg(10,10,101,"I",mu,sd,epsilon.x,epsilon.u,epsilon.v)
  ###########n = 100,pl= 10, pr = 10, Xtype = "II"##########
  acc8 <- NDF.Reg(10,10,101,"II",mu,sd,epsilon.x,epsilon.u,epsilon.v)
  ###########n = 100,pl= 10, pr = 10, Xtype = "III"##########
  acc9 <- NDF.Reg(10,10,101,"III",mu,sd,epsilon.x,epsilon.u,epsilon.v)
  ###########n = 200,pl= 10, pr = 10, Xtype = "I"##########
  acc10 <- NDF.Reg(10,10,200,"I",mu,sd,epsilon.x,epsilon.u,epsilon.v)
  ###########n = 200,pl= 10, pr = 10, Xtype = "II"##########
  acc11<- NDF.Reg(10,10,200,"II",mu,sd,epsilon.x,epsilon.u,epsilon.v)
  ###########n = 200,pl= 10, pr = 10, Xtype = "III"##########
  acc12 <- NDF.Reg(10,10,200,"III",mu,sd,epsilon.x,epsilon.u,epsilon.v)
  return(c(acc1,acc2,acc3,acc4,acc5,acc6,acc7,acc8,acc9,
             acc10,acc11,acc12))
}
# Stop Cluster
stopCluster(cl)
write.csv(results, file = "Reg1.csv")
