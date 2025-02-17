#########################################################################################################################
##### Code for Comparison of Different Dimension Reduction Methods (Supervised/Unsupervised) for Classification ###########  
#########################################################################################################################

##### Load Required R Packages #####
library(matrixcalc)  # Matrix operations
library(MASS)        # Multivariate normal distribution sampling
library(class)       # k-Nearest Neighbors (k-NN)
library(umap)        # UMAP for nonlinear dimension reduction
library(Rtsne)       # t-SNE for nonlinear dimension reduction
library(Rcpp)        # Interface with C++ for computational efficiency
library(kernlab)     # Kernel PCA (KPCA) for non-linear dimensionality reduction
library(expm)        # Matrix exponential package

##### Load C++ Functions for Our Method: Nonlinear Dimension Folding #####
file_path <- "/Users/djlin/Desktop/TensorSDR/func"
source(paste(file_path, "GSIR.R", sep="/"))  # Load GSIR method
source(paste(file_path, "SIR.R", sep="/"))   # Load SIR method
source(paste(file_path, "DF.R", sep="/"))    # Load Dimension Folding (DF) method
sourceCpp(paste(file_path, "NDF.cpp", sep="/"))  # Load C++ implementation of Nonlinear Dimension Folding (NDF)

############## CLASSIFICATION MODEL GENERATION FUNCTION ######################
CMG <- function(n, pl, pr) {
  # Function to generate synthetic classification data for testing different dimension reduction methods
  
  pi <- 0.5  # Probability of class 1
  sigma <- sqrt(3.0)  # Standard deviation for class 0
  tau <- sqrt(1.5)    # Standard deviation for class 1
  mu <- 2.5  # Mean shift for class 1
  
  # Number of observations
  n <- n
  # Predictor dimensions (tensor dimensions)
  pl <- pl
  p <- pl
  pr <- pr
  
  # Target dimensions for reduction
  dl <- dr <- d <- 2
  
  # Generate response variable Y (binary classification: 0 or 1)
  Y <- rbinom(n = n, size = 1, prob = pi)
  
  # Initialize the predictor matrix X
  X <- array(data = NA, dim = pl * pr)
  
  for (i in 1:n) {
    if (Y[i] == 0) {
      # Generate X from a normal distribution for class 0
      mean1 <- array(data = 0, dim = pl * pr)
      Sigma1 <- diag(array(data = 1, dim = pl * pr))
      Sigma1[2,2] <- sigma^2
      Sigma1[p+1,p+1] <- sigma^2
      X <- cbind(X, mvrnorm(n = 1, mu = matrix(mean1), Sigma = Sigma1))
      
    } else if (Y[i] == 1) {
      # Generate X from a normal distribution for class 1 with shifted mean
      mean2 <- array(data = 0, dim = pl * pr)
      mean2[1] <- mu
      mean2[p+2] <- mu
      Sigma2 <- diag(array(data = 1, dim = pl * pr))
      Sigma2[2,2] <- tau^2
      Sigma2[p+1,p+1] <- tau^2
      X <- cbind(X, mvrnorm(n = 1, mu = matrix(mean2), Sigma = Sigma2))
    }
  }
  
  ###### Nonlinear Transformation ######
  X <- X[,2:(n+1)]  # Remove the first NA column
  
  a <- X[1,]
  X[1,] <- sign(X[1,]) * (abs(X[1,]))^(3) + X[(p+2),]
  X[(p+2),] <- sign(X[(p+2),]) * abs(X[(p+2),])^(5)
  X[2,] <- X[2,] * X[(p+1),]
  
  return(list("X" = X, "Y" = Y))
}
################MAIN FUNCTION: ACCURANCY COMPUTATUON FUNCTION##############
acc <- function( n = n, pl = pl, pr = pr){
##############DATA GENERATION###################
  ######Model Generating#######
  tra <- CMG(n = (n+100),pl = pl,pr = pr) #Generate  (plpr*(n+100)) matrix
  #tes <- CMG(n = 50,pl = pl,pr = pr)
  X.tra <- tra$X[,1:n]
  Y.tra <- tra$Y[1:n]
  X.tes <- tra$X[,((n+1):(n+100))]
  Y.tes <- tra$Y[((n+1):(n+100))]
  X.tra <- t(X.tra)
  X.tes <- t(X.tes)
  ##############################################################################################
  #####PCA###########
  PCA.classifier <- function(X.train,Y.train,X.test,Y.test,k){
    a <- prcomp(t(X.train))$x[,1:4]
    b <- prcomp(t(X.test))$x[,1:4]
    predicted_labels <- knn(train = a, test = b, cl = Y.train, k = k)
    acc <- sum(predicted_labels == Y.test)/length(Y.test)
    return(acc)
  }
  acc.pca.simu.1 <- PCA.classifier(X.train = t(X.tra),Y.train = Y.tra,X.test = t(X.tra),Y.test = Y.tra,k = 10)
  acc.pca.t.simu.1 <- PCA.classifier(X.train = t(X.tra),Y.train = Y.tra,X.test = t(X.tes),Y.test = Y.tes,k = 10)
    
  ####UMAP##########
    UMAP.classifier <- function(X.train,Y.train,X.test,Y.test,k){
      a <- umap(t(X.train))$layout
      b <- umap(t(X.test))$layout
      predicted_labels <- knn(train = a, test = b, cl = Y.train, k = k)
      acc <- sum(Y.test==predicted_labels)/length(Y.test)
      return(acc)
    }
    acc.umap.simu.1 <- UMAP.classifier(X.train = t(X.tra),Y.train = Y.tra,X.test = t(X.tra),Y.test = Y.tra,k = 10)
    acc.umap.t.simu.1 <- UMAP.classifier(X.train = t(X.tra),Y.train = Y.tra,X.test = t(X.tes),Y.test = Y.tes,k = 10)
    
  #######TSNE########
    TSNE.classifier<- function(X.train,Y.train,X.test,Y.test,k){
      a <- Rtsne(X = t(X.train),dims = 2)$Y
      predicted_labels <- knn(train = a, test = a, cl = Y.train, k = k)
      return(sum(Y.test==predicted_labels)/length(Y.test))
    }
    acc.tsne.simu.1 <- TSNE.classifier(X.train = t(X.tra),Y.train = Y.tra,X.test = t(X.tra),Y.test = Y.tra,k = 10)
    TSNE.classifier.t <- function(X.train,Y.train,X.test,Y.test,k){
      a <- Rtsne(X = t(X.train),dims = 2)$Y
      b <- Rtsne(X = t(X.test),dims = 2)$Y
      predicted_labels <- knn(train = a, test = b, cl = Y.train, k = k)
      return(sum(Y.test==predicted_labels)/length(Y.test))
    }
    acc.tsne.t.simu.1 <- TSNE.classifier.t(X.train = t(X.tra),Y.train = Y.tra,X.test = t(X.tes),Y.test = Y.tes,k = 10)
    
  ########KPCA#########
    KPCA.classifier <- function(X.train,Y.train,X.test,Y.test,k){
      kpca_result <- kpca(~., data=as.data.frame(X.train), kernel="rbfdot", kpar=list(sigma=0.2), features=2)
      pc_data <- as.data.frame(predict(kpca_result, as.data.frame(X.train)))
      pc_data_2 <- as.data.frame(predict(kpca_result, as.data.frame(X.test)))
      predicted_labels <- knn(train = pc_data,test = pc_data_2,cl = Y.train,k = k)
      return(sum(Y.test==predicted_labels)/length(Y.test))
    }
    
    acc.kpca.simu.1 <- KPCA.classifier(X.train = X.tra,Y.train = Y.tra,X.test = X.tra,Y.test = Y.tra,k = 10)
    acc.kpca.t.simu.1 <- KPCA.classifier(X.train = X.tra,Y.train = Y.tra,X.test = X.tes,Y.test = Y.tes,k = 10)
  ########GSIR##################
    GSIR.classifier <- function(X.train,Y.train,X.test,Y.test,k){
      gsir.train <- gsir(x = X.train,y = Y.train,ytype = "categorical",complex_x = 1,complex_y = 1,r = 4,ex = 0.05,ey = 0.05)
      gsir.test <- gsir.predict(x = X.train,x_new = X.test,y = Y.train,ytype = "categorical",ex = 0.05,ey = 0.05,complex_x = 1,complex_y = 1,r = 4)
      predicted_labels <- knn(train = gsir.train,test = gsir.test,cl = Y.train,k = k)
      return(sum(Y.test==predicted_labels)/length(Y.test))
    }
    acc.gsir.simu.1 <- GSIR.classifier(X.train = X.tra,Y.train = Y.tra,X.test = X.tra,Y.test = Y.tra,k = 10)
    acc.gsir.t.simu.1 <- GSIR.classifier(X.train = X.tra,Y.train = Y.tra,X.test = X.tes,Y.test = Y.tes,k = 10)
  #######SIR###################
    SIR.classifier <- function(X.train,Y.train,X.test,Y.test,k){
      beta <- sir(x = X.train,y = Y.train,h = 2,r = 4,ytype = "categorical")
      sir.train <- X.train%*%beta
      sir.test <- X.test%*%beta
      predicted_labels <- knn(train = sir.train,test = sir.test,cl = Y.train,k = k)
      return(sum(Y.test==predicted_labels)/length(Y.test))
    }
    acc.sir.simu.1 <- SIR.classifier(X.train = X.tra,Y.train = Y.tra,X.test = X.tra,Y.test = Y.tra,k = 10)
    acc.sir.t.simu.1 <- SIR.classifier(X.train = X.tra,Y.train = Y.tra,X.test = X.tes,Y.test = Y.tes,k = 10)
  ########LDA##################
    LDA.classifier <- function(X.train,Y.train,X.test,Y.test,k){
      train_data <- data.frame(X.train)
      train_data$Y <- Y.train
      lda_model <- lda(Y ~ ., data = train_data)
      lda_train <- predict(lda_model, newdata = train_data)$x
      lda_test <- predict(lda_model, newdata = data.frame(X.test))$x
      predicted_labels <- knn(train = lda_train,test = lda_test,cl = Y.train,k = k)
      return(sum(Y.test==predicted_labels)/length(Y.test))
    }
    acc.lda.simu.1 <- LDA.classifier(X.train = X.tra,Y.train = Y.tra,X.test = X.tra,Y.test = Y.tra,k = 10)
    acc.lda.t.simu.1 <- LDA.classifier(X.train = X.tra,Y.train = Y.tra,X.test = X.tes,Y.test = Y.tes,k = 10) 
    
  ########DIMENSION FOLDING#####
    DF.classifier <- function(X.train,Y.train,X.test,Y.test,index,p,h,epsilon,iteration,pl,pr,dl,dr,k){
      result <- df(X = X.train,Y = Y.train,index = index,p = p,h = h,epsilon = epsilon,iteration = iteration,pl = pl,pr = pr,dl = dl,dr = dr)
      a <- result[[1]]
      b <- result[[2]]
      #####KNN######
      Xn.test1 <- matrix(data = 0,nrow = dl*dr,ncol = ncol(X.test))
      for (i in 1:ncol(Xn.test1)) {
        Xn.test1[,i] <- matrix(data = t(a)%*%matrix(X.test[,i],nrow = pl,ncol = pr)%*%b,nrow = dl*dr)
      }
      Xn.train1 <- matrix(data = 0,nrow = dl*dr,ncol = ncol(X.train))
      for (i in 1:ncol(Xn.train1)) {
        Xn.train1[,i] <- matrix(data = t(a)%*%matrix(X.train[,i],nrow = pl,ncol = pr)%*%b,nrow = dl*dr)
      }
      predicted_labels.df <- knn(train = t(Xn.train1), test = t(Xn.test1), cl = Y.train, k = k)
      #print(predicted_labels.df)
      return(sum(Y.test==predicted_labels.df)/length(Y.test))
    }
    ######
    pl <- pl;pr <- pr;dl <- 2;dr <- 2;h <- 2
    p.t <- array(data = 0,dim = h)
    index <- list()
    for (i in 0:1) {
      p.t[i+1] <- sum(Y.tra==i)/length(Y.tra)
      index[[i+1]] <- which(Y.tra== i)
    }
    epsilon <- 1e-9
    acc.df.simu.1 <- DF.classifier(X.train = t(X.tra),Y.train = matrix(Y.tra,ncol=1),X.test = t(X.tra),Y.test = Y.tra,
                                   index = index,p = p.t,h = h,epsilon = epsilon,iteration = 1000,
                                   pl = pl,pr = pr,dl = dl,dr = dr,k=10)
    
    acc.df.t.simu.1 <- DF.classifier(X.train = t(X.tra),Y.train = matrix(Y.tra,ncol=1),X.test = t(X.tes),Y.test = Y.tes,
                                     index = index,p = p.t,h = h,epsilon = epsilon,iteration = 1000,
                                     pl = pl,pr = pr,dl = dl,dr = dr,k=10)
    
    
    ########NONLINEAR DIMENSION FOLDING TUCKER FORM#################
    pl <- pl;pr <- pr;dl <- 2;dr <- 2
    epsilon.x <- 1e-10
    epsilon.u <- 0.01
    epsilon.v <- 0.01
    ##################################Classification##########################################
    res <- NSPGSIRTu(X = t(X.tra) ,Y =  matrix(Y.tra,ncol=1),pl = pl,pr = pr,dl = dl,dr = dr,
                   thre = 0.01, max_iter= 10,kernel_u = "gaussian",kernel_v = "gaussian",kernel_Y = "discrete",
                   epsilon_u = epsilon.u,epsilon_v = epsilon.v,epsilon_x = epsilon.v)
    f <- res$f
    g <- res$g
    X.train.d <- NSPGSIR_predict_Tu(X = t(X.tra),Y =  matrix(Y.tra,ncol=1),X_new = t(X.tra),
                                 pl = pl,pr = pr,dl = dl,dr = dr,f = f,g = g)
    X.test.d <- NSPGSIR_predict_Tu(X = t(X.tra),Y =  matrix(Y.tra,ncol=1),X_new = t(X.tes),
                                pl = pl,pr = pr,dl = dl,dr = dr,f = f,g = g)
    
    
    label1 <- knn(train = t(X.train.d),test = t(X.train.d),cl = Y.tra,k = 10)
    acc.ndf.simu.1 <- sum(label1 == Y.tra)/length(Y.tra)
    
    label1.t <- knn(train = t(X.train.d),test = t(X.test.d),cl = Y.tra,k = 5)
    acc.ndf.t.simu.1 <- sum(label1.t == Y.tes)/length(Y.tes)
  ##################################Classification##########################################
  res <- NSPGSIRCP(X = t(X.tra) ,Y = matrix(Y.tra,ncol=1),pl = pl,pr = pr,d = 4,
                 thre = 0.01,max_iter = 10,kernel_u = "gaussian",kernel_v = "gaussian",kernel_Y = "discrete",
                 epsilon_u = epsilon.u,epsilon_v = epsilon.v,epsilon_x = epsilon.v)
  f <- res$f
  g <- res$g
  X.train.d <- NSPGSIR_predict_CP(X = t(X.tra),Y = matrix(Y.tra,ncol = 1),X_new = t(X.tra),
                               pl = pl,pr = pr,d = 4,f = f,g = g)
  X.test.d <- NSPGSIR_predict_CP(X = t(X.tra),Y = matrix(Y.tra,ncol = 1),X_new = t(X.tes),
                              pl = pl,pr = pr,d = 4,f = f,g = g)
  
  
  label2<- knn(train = t(X.train.d),test = t(X.train.d),cl = Y.tra,k = 5)
  acc.ndf.simu.2 <- sum(label2 == Y.tra)/length(Y.tra)
  
  label2.t <- knn(train = t(X.train.d),test = t(X.test.d),cl = Y.tra,k = 5)
  acc.ndf.t.simu.2 <- sum(label2.t == Y.tes)/length(Y.tes)
    
  return(matrix(data = c(acc.ndf.simu.1,acc.ndf.t.simu.1,acc.ndf.simu.2,acc.ndf.simu.2,acc.df.simu.1,acc.df.t.simu.1,acc.umap.simu.1,acc.umap.t.simu.1,acc.tsne.simu.1,acc.tsne.t.simu.1,
                           acc.kpca.simu.1,acc.kpca.t.simu.1,acc.lda.simu.1,acc.lda.t.simu.1,acc.sir.simu.1,acc.sir.t.simu.1,acc.gsir.simu.1,acc.gsir.t.simu.1),nrow=1))
}

################Get the result#########################
library(doParallel)
library(foreach)
cores <- detectCores()
cl <- makeCluster(cores) 
registerDoParallel(cl)
tasks <- foreach(i = 1:100, .combine = rbind) %dopar% {
  ####Import packages######
  library(matrixcalc)
  library(MASS)
  library(class)
  library(umap)  #UMAP
  library(Rtsne) #TSNE
  library(Rcpp)
  library(kernlab) #KPCA
  library(expm)
  ####Load Cpp Files (Our Method: Nonlinear Dimension Folding)
  sourceCpp(paste(file_path, "NDFGSIR.cpp", sep="/"))
  acc.100.2.10 <- acc(n = 100,pl = 2,pr = 10)
  acc.100.2.30 <- acc(n = 100,pl = 2,pr = 30)
  acc.100.2.50 <- acc(n = 101,pl = 2,pr = 50)
  acc.200.2.10 <- acc(n = 200,pl = 2,pr = 10)
  acc.200.2.30 <- acc(n = 200,pl = 2,pr = 30)
  acc.200.2.50 <- acc(n = 200,pl = 2,pr = 50)
  acc.200.2.100 <- acc(n = 201,pl = 2,pr = 100)
  acc.500.2.10 <- acc(n = 500,pl = 2,pr = 10)
  acc.500.2.30 <- acc(n = 500,pl = 2,pr = 30)
  acc.500.2.50 <- acc(n = 500,pl = 2,pr = 50)
  acc.500.2.100 <- acc(n = 500,pl = 2,pr = 100)
  acc.100.4.10 <- acc(n = 100,pl = 4,pr = 10)
  acc.200.4.10 <- acc(n = 200,pl = 4,pr = 10)
  acc.200.4.30 <- acc(n = 200,pl = 4,pr = 30)
  acc.200.4.50 <- acc(n = 201,pl = 4,pr = 50)
  acc.500.4.10 <- acc(n = 500,pl = 4,pr = 10)
  acc.500.4.30 <- acc(n = 500,pl = 4,pr = 30)
  acc.500.4.50 <- acc(n = 500,pl = 4,pr = 50)
  return(cbind(acc.100.2.10,acc.100.2.30,acc.100.2.50,acc.200.2.10,acc.200.2.30,acc.200.2.50,acc.200.2.100,
               acc.500.2.10,acc.500.2.30,acc.500.2.50,acc.500.2.100,acc.200.4.10,acc.200.4.30,acc.200.4.50,
               acc.500.4.10,acc.500.4.30,acc.500.4.50))
}
stopCluster(cl)





















