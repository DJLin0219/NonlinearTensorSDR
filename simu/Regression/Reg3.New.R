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
########## Simulation 3 #########
NDF.Reg <- function(pl,pr,n,Xtype,mu,sd=0.5,epsilon.x,epsilon.u,epsilon.v,seed){
  #####Data Generation And Accuracy computation for Simulation Regression Part Model III #######
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
  ##Compute six accuracies which are 
  #the distance correlations of predictors given by Tucker Decomposition methods in our paper,
  #the distance correlation of responses given by Tucker Decomposition methods in our paper, 
  #the distance correlation of predictors given by CP Decomposition methods in our paper, 
  #the distance correlation of responses given by CP Decomposition methods in our paper, 
  #the distance correlation of responses given by GSIR 
  #the distance correlation of responses given by KSIR.
  set.seed(seed)
  pl <- pl
  pr <- pr
  #target dimensions to reduce, in Model 3, dl=dr=2
  dl <- 2
  dr <- 2
  r <- min(pl,pr)
  #The number of training data and test data = 100
  n <- n
  n.t <- 100
  if (Xtype == "I") {
    Mu <- rep(0,pl*pr)
    Mu[1] <- mu
    Mu[pl+2] <- mu
    X.tr <- MASS::mvrnorm(n= n,mu = Mu,Sigma = diag(1,pl*pr))
    X.te <- MASS::mvrnorm(n= n.t,mu = Mu,Sigma = diag(1,pl*pr))
    X <- rbind(X.tr,X.te)
  } else if (Xtype == "II") {
    Mu1 <- rep(-1,pl*pr);
    Mu1[1] <- mu
    Mu1[pl+2] <- mu
    Mu2 <- rep(1,pl*pr);
    Mu2[1] <- mu
    Mu2[pl+2] <- mu
    X.tr <- 1/2*MASS::mvrnorm(n=(n),mu = Mu1,Sigma = diag(2,pl*pr))+1/2*MASS::mvrnorm(n= (n),mu = Mu2,Sigma = diag(2,pl*pr))
    X.te <- 1/2*MASS::mvrnorm(n=(n.t),mu = Mu1,Sigma = diag(2,pl*pr))+1/2*MASS::mvrnorm(n= (n.t),mu = Mu2,Sigma = diag(2,pl*pr))
    X <- rbind(X.tr,X.te)
  } else if (Xtype == "III"){
    Mu <- rep(0,pl*pr)
    Mu[1] <- mu
    Mu[pl+2] <- mu
    X.tr <- MASS::mvrnorm(n =(n),mu = Mu,Sigma = 0.8*diag(1,pl*pr)+0.2*rep(1,pl*pr)%*%t(rep(1,pl*pr)))
    X.te <- MASS::mvrnorm(n =(n.t),mu = Mu,Sigma = 0.8*diag(1,pl*pr)+0.2*rep(1,pl*pr)%*%t(rep(1,pl*pr)))
    X <- rbind(X.tr,X.te)
  }
  u <- matrix(nrow = pl,ncol = 0)
  v <- matrix(nrow = pr,ncol = 0)
  lambda <- c()
  X.tr <- t(X.tr)
  X.te <- t(X.te)
  result.train <- SVDs(X = X.tr,pl = pl,pr = pr)
  result.test <- SVDs(X = X.te,pl = pl,pr = pr)
  u.train <- result.train$u
  v.train <- result.train$v
  u.test <- result.test$u
  v.test <- result.test$v
  lambda.train <- result.train$lambda
  lambda.test <- result.test$lambda
  f.true.1 <- (u.train[1,])^3
  g.true.1 <- (v.train[1,])^5
  f.true.2 <- exp(1+u.train[2,])
  g.true.2 <- exp(1+v.train[2,])
  f.true.t.1 <- (u.test[1,])^3
  g.true.t.1 <- (v.test[1,])^5
  f.true.t.2 <- exp(1+u.test[2,])
  g.true.t.2 <- exp(1+v.test[2,])
  Y.n <- array(data = 0,dim = ncol(X.tr))
  for (i in 1:n) {
    Y.n[i] <- log(1+sum((lambda.train*(f.true.1*g.true.1+f.true.1*g.true.2+f.true.2*g.true.1+f.true.2*g.true.2))[(r*(i-1)+1):(r*i)]))
  }
  Y <- (Y.n)+rnorm(n,0,sd)*sd(Y.n)
  Y.t <- array(data = 0,dim = ncol(X.te))
  for (i in 1:n.t) {
    Y.t[i] <- log(1+sum((lambda.test*(f.true.t.1*g.true.t.1+f.true.t.1*g.true.t.2+f.true.t.2*g.true.t.1+f.true.t.2*g.true.t.2))[(r*(i-1)+1):(r*i)]))
  }
  Y.te <- (Y.t)+rnorm(n.t,0,sd)*sd(Y.t)
  ####Training Data and Test Data##########
  X.train <- X.tr
  X.test <- X.te
  Y.train <- matrix(Y,ncol=1)
  Y.test <- matrix(Y.te,ncol=1)
  ####Tuning##########
  epsilon.x <- epsilon.x
  epsilon.u <- epsilon.u
  epsilon.v <- epsilon.v
  ###Estimation of f,g and h######
  ###################Tucker Form Decomposition########################## 
  set.seed(2024)
  result <- NSPGSIRTu(X = X.train,Y = Y.train,pl = pl,pr = pr,dl = dl,dr = dr,thre=1e-4,max_iter=20,
                    epsilon_u = epsilon.v,epsilon_v = epsilon.v,epsilon_x = epsilon.x,kernel_u="gaussian",kernel_v="gaussian",
                    kernel_Y="gaussian")
  f <- result$f
  g <- result$g
  h <- result$h
  ##Compare f.pred with f.true, g.pred with g.true###
  f.pred <- (t(f)%*%gram_gauss(t(u.train),t(u.test),1))
  g.pred <- (t(g)%*%gram_gauss(t(v.train),t(u.test),1))  
  
  gf.true <- cbind(as.matrix(f.true.t.1*g.true.t.1),as.matrix(f.true.t.1*g.true.t.2),as.matrix(f.true.t.2*g.true.t.1),as.matrix(f.true.t.2*g.true.t.2))
  gf.pred <- cbind(f.pred[1,]*g.pred[1,],f.pred[1,]*g.pred[2,],f.pred[2,]*g.pred[1,],f.pred[2,]*g.pred[1,])
  acc1 <- (abs(dcor(gf.true,gf.pred)))
  X.train.d <- NSPGSIR_predict_Tu(X = X.train,Y = Y.train,X_new = X.train,pl = pl,pr = pr,dl = dl,dr = dr,f = f,g = g)
  data.train <- data.frame("Y" = Y.train,"X" = t(X.train.d))
  kernel.model <- npreg(Y~X.1+X.2+X.3+X.4,data = data.train, regtype = "ll", bwmethod = "cv.aic", gradients = TRUE)
  X.test.d <- NSPGSIR_predict_Tu(X = X.train,Y = Y.train,X_new = X.test,pl = pl,pr = pr,dl = dl,dr = dr,f = f,g = g)
  Y.test.pred <- predict(kernel.model, newdata = data.frame("X" = t(X.test.d)))
  acc2 <- (dcor(as.matrix(Y.test.pred),as.matrix(Y.test)))
  ###################CP Form Decomposition########################## 
  set.seed(2024)
  result <- NSPGSIRCP(X = X.train,Y = Y.train,pl = pl,pr = pr,d = 4 ,thre=1e-4,max_iter=20,
                     epsilon_u = epsilon.v,epsilon_v = epsilon.v,epsilon_x = epsilon.x,kernel_u="gaussian",kernel_v="gaussian",
                     kernel_Y="gaussian")
  f <- result$f
  g <- result$g
  h <- result$h
  # ##Compare f.pred with f.true, g.pred with g.true###
  f.pred <- (t(f)%*%gram_gauss(t(u.train),t(u.test),1))
  g.pred <- (t(g)%*%gram_gauss(t(v.train),t(u.test),1))  
  gf.true <- cbind(as.matrix(f.true.t.1*g.true.t.1),as.matrix(f.true.t.2*g.true.t.2))
  gf.pred <- cbind(f.pred[1,]*g.pred[1,],f.pred[2,]*g.pred[2,])
  acc5 <- (abs(dcor(gf.true,gf.pred)))
  X.train.d <- NSPGSIR_predict_CP(X = X.train,Y = Y.train,X_new = X.train,pl = pl,pr = pr,d = 4,f = f,g = g)
  data.train <- data.frame("Y" = Y.train,"X" = t(X.train.d))
  kernel.model <- npreg(Y~X.1+X.2,data = data.train, regtype = "ll", bwmethod = "cv.aic", gradients = TRUE)
  X.test.d <- NSPGSIR_predict_CP(X = X.train,Y = Y.train,X_new = X.test,pl = pl,pr = pr,d = 4,f = f,g = g)
  Y.test.pred <- predict(kernel.model, newdata = data.frame("X" = t(X.test.d)))
  acc6 <- (dcor(as.matrix(Y.test.pred),as.matrix(Y.test)))
  ###################GSIR########################## 
  ex = 0.01
  ey = 0.01
  gsir.train <- gsir.predict(x = t(X.train),y = as.matrix(Y.train),x_new = t(X.train),ytype = "continuous",ex = ex,ey = ey,complex_x = 1,complex_y = 1,r = 4)
  gsir.test <- gsir.predict(x = t(X.train),y = as.matrix(Y.train),x_new = t(X.test),ytype = "continuous",ex = ex,ey = ey,complex_x = 1,complex_y = 1,r = 4)
  data.train.2 <- data.frame("Y" = Y.train,"X" = gsir.train)
  kernel.model.2 <- npreg(Y~X.1+X.2+X.3+X.4,data = data.train.2, regtype = "ll", bwmethod = "cv.aic", gradients = TRUE)
  Y.test.pred.2 <- predict(kernel.model.2, newdata = data.frame("X" = gsir.test))
  acc9 <- (dcor(as.matrix(Y.test.pred.2),as.matrix(Y.test)))
  ####################KSIR########################
  ksir.beta <- kir(x = t(X.train),y = as.matrix(Y.train),b = 1,eps = 1,r = 4)
  ksir.train <- t(X.train)%*%ksir.beta
  ksir.test <- t(X.test)%*%ksir.beta
  data.train.3 <- data.frame("Y" = Y.train,"X" = ksir.train)
  kernel.model.3 <- npreg(Y~X.1+X.2+X.3+X.4,data = data.train.3, regtype = "ll", bwmethod = "cv.aic", gradients = TRUE)
  Y.test.pred.3 <- predict(kernel.model.3, newdata = data.frame("X" = ksir.test))
  acc10 <- (dcor(as.matrix(Y.test.pred.3),as.matrix(Y.test)))
  return(c(acc1,acc2,acc5,acc6,acc9,acc10))
}

#Parallel Computing
cores <- detectCores()
cl <- makeCluster(cores)
registerDoParallel(cl)
iteration = 200
results <- foreach(iter = 1:iteration, .combine = rbind) %dopar% {
  library(matrixcalc)
  library(MASS)
  library(Rcpp)
  library(foreach)
  library(energy)
  library(doParallel)
  library(kernlab)
  library(np)
  sourceCpp(paste(file_path, "NDFGSIR.cpp", sep="/"))
  find_max_row <- function(matrix_data) {
    row_values <- matrix_data[, 1] + 2 * matrix_data[, 2]
    max_value_index <- which.max(row_values)
    max_value_row <- matrix_data[max_value_index, ]
    return(max_value_row)
  }
  epsilon.x <- 1e-8
  epsilon.u <- 0.1
  epsilon.v <- 0.1
  sd <- 0.5
  mu <- 5
  ####We performed manual tuning here. While it would be ideal to tune parameters using GCV,
  ##our results are already sufficiently good. The GCV functions can be found in our CPP files.
  ###########n = 100,pl= 5, pr = 5, Xtype = "I"##########
  acc11 <- NDF.Reg(5,5,100,"I",mu,sd,epsilon.x,1,1,iter)
  acc12 <- NDF.Reg(5,5,100,"I",mu,sd,epsilon.x,0.5,0.5,iter)
  acc13 <- NDF.Reg(5,5,100,"I",mu,sd,epsilon.x,0.1,0.1,iter)
  acc14 <- NDF.Reg(5,5,100,"I",mu,sd,epsilon.x,0.05,0.05,iter)
  acc15 <- NDF.Reg(5,5,100,"I",mu,sd,epsilon.x,0.01,0.01,iter)
  acc1 <- find_max_row(rbind(acc11,acc12,acc13,acc14,acc15))
  
  ###########n = 100,pl= 5, pr = 5, Xtype = "II"##########
  acc21 <- NDF.Reg(5,5,100,"II",mu,sd,epsilon.x,1,1,iter)
  acc22 <- NDF.Reg(5,5,100,"II",mu,sd,epsilon.x,0.5,0.5,iter)
  acc23 <- NDF.Reg(5,5,100,"II",mu,sd,epsilon.x,0.1,0.1,iter)
  acc24 <- NDF.Reg(5,5,100,"II",mu,sd,epsilon.x,0.05,0.05,iter)
  acc25 <- NDF.Reg(5,5,100,"II",mu,sd,epsilon.x,0.01,0.01,iter)
  acc2 <- find_max_row(rbind(acc21,acc22,acc23,acc24,acc25))
  
  ###########n = 100,pl= 5, pr = 5, Xtype = "III"##########
  acc31 <- NDF.Reg(5,5,100,"III",mu,sd,epsilon.x,1,1,iter)
  acc32 <- NDF.Reg(5,5,100,"III",mu,sd,epsilon.x,0.5,0.5,iter)
  acc33 <- NDF.Reg(5,5,100,"III",mu,sd,epsilon.x,0.1,0.1,iter)
  acc34 <- NDF.Reg(5,5,100,"III",mu,sd,epsilon.x,0.05,0.05,iter)
  acc35 <- NDF.Reg(5,5,100,"III",mu,sd,epsilon.x,0.01,0.01,iter)
  acc3 <- find_max_row(rbind(acc31,acc32,acc33,acc34,acc35))
  
  ###########n = 200,pl= 5, pr = 5, Xtype = "I"##########
  acc41 <- NDF.Reg(5,5,200,"I",mu,sd,epsilon.x,1,1,iter)
  acc42 <- NDF.Reg(5,5,200,"I",mu,sd,epsilon.x,0.5,0.5,iter)
  acc43 <- NDF.Reg(5,5,200,"I",mu,sd,epsilon.x,0.1,0.1,iter)
  acc44 <- NDF.Reg(5,5,200,"I",mu,sd,epsilon.x,0.05,0.05,iter)
  acc45 <- NDF.Reg(5,5,200,"I",mu,sd,epsilon.x,0.01,0.01,iter)
  acc46 <- NDF.Reg(5,5,200,"I",mu,sd,epsilon.x,0.005,0.005,iter)
  acc47 <- NDF.Reg(5,5,200,"I",mu,sd,epsilon.x,0.001,0.001,iter)
  acc4 <- find_max_row(rbind(acc41,acc42,acc43,acc44,acc45,acc46,acc47))
  
  ###########n = 200,pl= 5, pr = 5, Xtype = "II"##########
  acc51 <- NDF.Reg(5,5,200,"II",mu,sd,epsilon.x,1,1,iter)
  acc52 <- NDF.Reg(5,5,200,"II",mu,sd,epsilon.x,0.5,0.5,iter)
  acc53 <- NDF.Reg(5,5,200,"II",mu,sd,epsilon.x,0.1,0.1,iter)
  acc54 <- NDF.Reg(5,5,200,"II",mu,sd,epsilon.x,0.05,0.05,iter)
  acc55 <- NDF.Reg(5,5,200,"II",mu,sd,epsilon.x,0.01,0.01,iter)
  acc56 <- NDF.Reg(5,5,200,"III",mu,sd,epsilon.x,0.005,0.0005,iter)
  acc57 <- NDF.Reg(5,5,200,"III",mu,sd,epsilon.x,0.001,0.0001,iter)
  acc5 <- find_max_row(rbind(acc51,acc52,acc53,acc54,acc55,acc56,acc57))
  
  ###########n = 200,pl= 5, pr = 5, Xtype = "III"##########
  acc61 <- NDF.Reg(5,5,200,"III",mu,sd,epsilon.x,1,1,iter)
  acc62 <- NDF.Reg(5,5,200,"III",mu,sd,epsilon.x,0.5,0.5,iter)
  acc63 <- NDF.Reg(5,5,200,"III",mu,sd,epsilon.x,0.1,0.1,iter)
  acc64 <- NDF.Reg(5,5,200,"III",mu,sd,epsilon.x,0.05,0.05,iter)
  acc65 <- NDF.Reg(5,5,200,"III",mu,sd,epsilon.x,0.01,0.01,iter)
  acc66 <- NDF.Reg(5,5,200,"III",mu,sd,epsilon.x,0.005,0.005,iter)
  acc67 <- NDF.Reg(5,5,200,"III",mu,sd,epsilon.x,0.001,0.001,iter)
  acc6 <- find_max_row(rbind(acc61,acc62,acc63,acc64,acc65,acc66,acc67))
  
  ###########n = 100,pl= 10, pr = 10, Xtype = "I"##########
  acc71 <- NDF.Reg(10,10,101,"I",mu,sd,epsilon.x,1,1,iter)
  acc72 <- NDF.Reg(10,10,101,"I",mu,sd,epsilon.x,0.5,0.5,iter)
  acc73 <- NDF.Reg(10,10,101,"I",mu,sd,epsilon.x,0.1,0.1,iter)
  acc74 <- NDF.Reg(10,10,101,"I",mu,sd,epsilon.x,0.05,0.05,iter)
  acc75 <- NDF.Reg(10,10,101,"I",mu,sd,epsilon.x,0.01,0.01,iter)
  acc7 <- find_max_row(rbind(acc71,acc72,acc73,acc74,acc75))
  
  ###########n = 100,pl= 10, pr = 10, Xtype = "II"##########
  acc81 <- NDF.Reg(10,10,101,"II",mu,sd,epsilon.x,1,1,iter)
  acc82 <- NDF.Reg(10,10,101,"II",mu,sd,epsilon.x,0.5,0.5,iter)
  acc83 <- NDF.Reg(10,10,101,"II",mu,sd,epsilon.x,0.1,0.1,iter)
  acc84 <- NDF.Reg(10,10,101,"II",mu,sd,epsilon.x,0.05,0.05,iter)
  acc85 <- NDF.Reg(10,10,101,"II",mu,sd,epsilon.x,0.01,0.01,iter)
  acc8 <- find_max_row(rbind(acc81,acc82,acc83,acc84,acc85))
  
  ###########n = 100,pl= 10, pr = 10, Xtype = "III"##########
  acc91 <- NDF.Reg(10,10,101,"III",mu,sd,epsilon.x,1,1,iter)
  acc92 <- NDF.Reg(10,10,101,"III",mu,sd,epsilon.x,0.5,0.5,iter)
  acc93 <- NDF.Reg(10,10,101,"III",mu,sd,epsilon.x,0.1,0.1,iter)
  acc94 <- NDF.Reg(10,10,101,"III",mu,sd,epsilon.x,0.05,0.05,iter)
  acc95 <- NDF.Reg(10,10,101,"III",mu,sd,epsilon.x,0.01,0.01,iter)
  acc9 <- find_max_row(rbind(acc91,acc92,acc93,acc94,acc95))
  
  ###########n = 200,pl= 10, pr = 10, Xtype = "I"##########
  acc101 <- NDF.Reg(10,10,200,"I",mu,sd,epsilon.x,1,1,iter)
  acc102 <- NDF.Reg(10,10,200,"I",mu,sd,epsilon.x,0.5,0.5,iter)
  acc103 <- NDF.Reg(10,10,200,"I",mu,sd,epsilon.x,0.1,0.1,iter)
  acc104 <- NDF.Reg(10,10,200,"I",mu,sd,epsilon.x,0.05,0.05,iter)
  acc105 <- NDF.Reg(10,10,200,"I",mu,sd,epsilon.x,0.01,0.01,iter)
  acc10 <- find_max_row(rbind(acc101,acc102,acc103,acc104,acc105))
  
  ###########n = 200,pl= 10, pr = 10, Xtype = "II"##########
  acc111 <- NDF.Reg(10,10,200,"II",mu,sd,epsilon.x,1,1,iter)
  acc112 <- NDF.Reg(10,10,200,"II",mu,sd,epsilon.x,0.5,0.5,iter)
  acc113 <- NDF.Reg(10,10,200,"II",mu,sd,epsilon.x,0.1,0.1,iter)
  acc114 <- NDF.Reg(10,10,200,"II",mu,sd,epsilon.x,0.05,0.05,iter)
  acc115 <- NDF.Reg(10,10,200,"II",mu,sd,epsilon.x,0.01,0.01,iter)
  acc11 <- find_max_row(rbind(acc111,acc112,acc113,acc114,acc115))
  
  ###########n = 200,pl= 10, pr = 10, Xtype = "III"##########
  acc121 <- NDF.Reg(10,10,200,"III",mu,sd,epsilon.x,1,1,iter)
  acc122 <- NDF.Reg(10,10,200,"III",mu,sd,epsilon.x,0.5,0.5,iter)
  acc123 <- NDF.Reg(10,10,200,"III",mu,sd,epsilon.x,0.1,0.1,iter)
  acc124 <- NDF.Reg(10,10,200,"III",mu,sd,epsilon.x,0.05,0.05,iter)
  acc125 <- NDF.Reg(10,10,200,"III",mu,sd,epsilon.x,0.01,0.01,iter)
  acc12 <- find_max_row(rbind(acc121,acc122,acc123,acc124,acc125))
  return(c(acc1,acc2,acc3,acc4,acc5,acc6,acc7,acc8,acc9,
           acc10,acc11,acc12))
}
# Stop Cluster
stopCluster(cl)
write.csv(results, file = "Reg3.csv")
  