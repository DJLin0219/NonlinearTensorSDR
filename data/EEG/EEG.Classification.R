library(foreach)
library(doParallel)
library(matrixcalc)
library(randomForest)
library(MASS)
library(caret)
library(class)
library(Rcpp)
####Load Cpp Files (Our Method: Nonlinear Dimension Folding)
file_path <- "/Users/djlin/Desktop/TensorSDR/func"
source(paste(file_path, "GSIR.R", sep="/"))
source(paste(file_path, "SIR.R", sep="/"))
source(paste(file_path, "DF.R", sep="/"))
sourceCpp(paste(file_path, "NDF.cpp", sep="/"))
file_path2 <- "/Users/djlin/Desktop/TensorSDR/RealData/EEG"
X.ac <- read.table(paste(file_path2, "alcoholic_data.txt", sep="/"))
X.co <- read.table(paste(file_path2, "control_data.txt", sep="/"))
X <- rbind(X.ac,X.co)
########GSIR############
pl <- 256
pr <- 64
ml <- 10
mr <- 10
X.t <- t(X)
X.c <- X.t - rowMeans(X.t)
Sigma.c.w <- matrix(data = 0, pr, pr)
Sigma.c.v <- matrix(data = 0, pl, pl)
for (i in 1:ncol(X.c)) {
  X.c.c <- matrix(data = X.c[, i], nrow = pl, ncol = pr)
  Sigma.c.v <- Sigma.c.v + (X.c.c) %*% t(X.c.c) / ncol(X.c)
  Sigma.c.w <- Sigma.c.w + t(X.c.c) %*% X.c.c / ncol(X.c)
}
v.c <- eigen(Sigma.c.v)$vectors[, 1:ml]
w.c <- eigen(Sigma.c.w)$vectors[, 1:mr]
X.n <- matrix(data = 0, nrow = ml*mr, ncol = ncol(X.t))
for (i in 1:ncol(X.n)) {
  X.n[, i] <- as.vector(t(v.c) %*% matrix(data = X.t[, i], pl, pr) %*% (w.c))
}
#set.seed(2024)
#index <- sample(1:122)
index <- 1:122
X.n <- X.n[,index]
Y <- c(rep(1,77),rep(0,45))
Y <- Y[index]
acc1 <- 0
for (i in 1:122) {
  X_train <- as.matrix(X.n)[, -i]
  Y_train <- Y[-i]
  X_test <- as.matrix(X.n[, i])
  Y_test <- Y[i]
  GSIR.classifier <- function(X.train,Y.train,X.test,Y.test){
    r = 4
    set.seed(2024)
    gsir.train <- gsir(x = X.train,y = Y.train,ytype = "categorical",complex_x = 1,complex_y = 1,r = r,ex = 0.05,ey = 0.05)
    gsir.test <- gsir.predict(x = X.train,x_new = X.test,y = Y.train,ytype = "categorical",ex = 0.05,ey = 0.05,complex_x = 1,complex_y = 1,r = r)
    dat.tr <- data.frame("X"=gsir.train,"Y"=as.factor(Y.train))
    #qda_model <- caret::train(x = as.data.frame(gsir.train),as.factor(Y_train), method = "qda", trControl = trainControl(method = "cv"))
    #md <- qda(Y~.,data = dat.tr)
    rf_model <- randomForest(Y ~., data = dat.tr, ntree = 500, mtry = 2, importance = TRUE)
    dat.te <- data.frame("X"=gsir.test,"Y"=as.factor(Y.test))
    rf_predictions <- predict(rf_model, dat.te[,1:r])
    #predicted_labels <- knn(train = gsir.train,test = gsir.test,cl = Y.train,k = k)
    return(sum(Y_test==rf_predictions)/length(Y.test))
  }
  acc1 <- acc1+GSIR.classifier(X.train = t(X_train),Y.train = Y_train,X.test = t(X_test),Y.test = Y_test)
  print(i)
}
print(acc1)
########SIR############
acc2 <- 0
pl <- 256
pr <- 64
ml <- 16
mr <- 4
v.c <- eigen(Sigma.c.v)$vectors[, 1:ml]
w.c <- eigen(Sigma.c.w)$vectors[, 1:mr]
X.n.n <- matrix(data = 0, nrow = ml*mr, ncol = ncol(X.c))
for (i in 1:ncol(X.c)) {
  X.n.n[, i] <- as.vector(t(v.c) %*% matrix(data = X.t[, i], pl, pr) %*% (w.c))
}
#set.seed(2024)
#index <- sample(1:122)
index <- 1:122
X.n.n <- X.n.n[,index]
Y <- c(rep(1,77),rep(0,45))
Y <- Y[index]
for (i in 1:122) {
  X_train <- as.matrix(X.n.n)[, -i]
  Y_train <- Y[-i]
  X_test <- as.matrix(X.n.n[, i])
  Y_test <- Y[i]
  SIR.classifier <- function(X.train,Y.train,X.test,Y.test){
    r = 4
    set.seed(2024)
    beta <- sir(x = X.train,y = Y.train,h = 2,r = r,ytype = "categorical")
    sir.train <- X.train%*%beta
    sir.test <- X.test%*%beta
    set.seed(2024)
    dat.tr <- data.frame("X" = sir.train, "Y" = as.factor(Y_train))
    dat.te <- data.frame("X" = sir.test, "Y" = as.factor(Y_test))
    rf_model <- randomForest::randomForest(Y ~ ., data = dat.tr, ntree = 500, mtry = 2, importance = TRUE)
    rf_predictions <- predict(rf_model, dat.te[, 1:r])
    return(sum(Y_test == rf_predictions) / length(Y_test))
  }
  print(i)
  acc2 <- acc2+SIR.classifier(X.train = t(X_train),Y.train = Y_train,X.test = t(X_test),Y.test = Y_test)
}  
print(acc2)
#########DF##############
pl <- 256
pr <- 64
ml <- 15
mr <- 15
X.t <- t(X)
X.c <- X.t - rowMeans(X.t)
Sigma.c.w <- matrix(data = 0, pr, pr)
Sigma.c.v <- matrix(data = 0, pl, pl)
for (i in 1:ncol(X.c)) {
  X.c.c <- matrix(data = X.c[, i], nrow = pl, ncol = pr)
  Sigma.c.v <- Sigma.c.v + (X.c.c) %*% t(X.c.c) / ncol(X.c)
  Sigma.c.w <- Sigma.c.w + t(X.c.c) %*% X.c.c / ncol(X.c)
}
v.c <- eigen(Sigma.c.v)$vectors[, 1:ml]
w.c <- eigen(Sigma.c.w)$vectors[, 1:mr]
X.n <- matrix(data = 0, nrow = ml*mr, ncol = ncol(X.t))
for (i in 1:ncol(X.n)) {
  X.n[, i] <- as.vector(t(v.c) %*% matrix(data = X.t[, i], pl, pr) %*% (w.c))
}
#set.seed(2023)
index <- sample(1:122)
#index <- 1:122
X.n <- X.n[,index]
Y <- c(rep(1,77),rep(0,45))
Y <- Y[index]
acc3 <- 0
for (i in 1:122) {
  X_train <- as.matrix(X.n)[, -i]
  Y_train <- Y[-i]
  X_test <- as.matrix(X.n[, i])
  Y_test <- Y[i]
  DF.classifier <- function(X.train,Y.train,X.test,Y.test){
    dl = 1
    dr = 4
    pl <- ml
    pr <- mr
    h <- 2
    p.t <- array(data = 0,dim = h)
    index <- list()
    for (k in 0:1) {
      p.t[k+1] <- sum(Y.train==k)/length(Y.train)
      index[[k+1]] <- which(Y.train== k)
    }
    epsilon <- 1e-2
    result <- df(X = X.train,Y = Y.train,index = index,p = p.t,h = h,epsilon = epsilon,iteration = 100,pl = pl,pr = pr,dl = dl,dr = dr)
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
    dat.tr <- data.frame("X"=t(Xn.train1),"Y"=as.factor(Y.train))
    #rf_model <- randomForest(Y ~., data = dat.tr, ntree = 500, mtry = 2, importance = TRUE)
    #md <- qda(Y~.,data = dat.tr)
    #qda_model <- caret::train(x = as.data.frame(t(Xn.train1)),as.factor(Y_train), method = "qda", trControl = caret::trainControl(method = "cv"))
    dat.te <- data.frame("X"=t(Xn.test1),"Y"=Y.test)
    set.seed(2025)
    rf_model <- randomForest::randomForest(Y ~ ., data = dat.tr, ntree = 500, mtry = 2, importance = TRUE)
    rf_predictions <- predict(rf_model, dat.te)
    return(sum(Y_test == rf_predictions) / length(Y_test))
    #return(sum(Y.test==predict(md,dat.te)$class)/length(Y.test))
  }
  print(i)
  acc3 <- acc3+DF.classifier(X.train = X_train,Y.train = Y_train,X.test = X_test,Y.test = Y_test)
}  
print(acc3)
#####NDF###############
X <- rbind(X.ac,X.co)
Y <- c(rep(1,77),rep(0,45))
##########
pl <- 256
pr <- 64
ml <- 9
mr <- 9
X.t <- t(X)
X.c <- X.t - rowMeans(X.t)
Sigma.c.w <- matrix(data = 0, pr, pr)
Sigma.c.v <- matrix(data = 0, pl, pl)
for (k in 1:ncol(X.c)) {
  X.c.c <- matrix(data = X.c[, k], nrow = pl, ncol = pr)
  Sigma.c.v <- Sigma.c.v + (X.c.c) %*% t(X.c.c) / ncol(X.c)
  Sigma.c.w <- Sigma.c.w + t(X.c.c) %*% X.c.c / ncol(X.c)
}
v.c <- eigen(Sigma.c.v)$vectors[, 1:ml]
w.c <- eigen(Sigma.c.w)$vectors[, 1:mr]
X.n <- matrix(data = 0, nrow = ml*mr, ncol = ncol(X.t))
for (k in 1:ncol(X.n)) {
  X.n[, k] <- as.vector(t(v.c) %*% matrix(data = X.c[, k], pl, pr) %*% (w.c))
}
##
set.seed(1)
index <- sample(1:122)
X.n <- X.n[,index]
Y <- Y[index]
#####NDF###############
pl <- ml
pr <- mr
dl <- 1
dr <- 2
epsilon.x <- 1e-8
epsilon.u <- 0.1
epsilon.v <- 0.1
num_cores <- detectCores() - 1
cl <- makeCluster(num_cores)
registerDoParallel(cl)
results <- foreach(i = 1:122, .combine = c) %dopar% {
  library(foreach)
  library(doParallel)
  library(matrixcalc)
  library(randomForest)
  library(MASS)
  library(caret)
  library(class)
  library(Rcpp)
  sourceCpp("NDFGSIR.cpp")
  
  X_train <- as.matrix(X.n)[, -i]
  Y_train <- Y[-i]
  X_test <- as.matrix(X.n[, i])
  Y_test <- Y[i]
  res <- NSPGSIRTu(X = X_train, Y = Y_train, pl = pl, pr = pr, dl = dl, dr = dr,
                   thre = -1e10, max_iter= 5, kernel_u = "gaussian", kernel_v = "gaussian", kernel_Y = "discrete",
                   epsilon_u = epsilon.u, epsilon_v = epsilon.v, epsilon_x = epsilon.x)
  f <- res$f
  g <- res$g
  X.d.tr <- NSPGSIR_predict_Tu(X = X_train, Y = Y_train, X_new = X_train, pl = pl, pr = pr, dl = dl, dr = dr, f = f, g = g)
  X.d.te <- NSPGSIR_predict_Tu(X = X_train, Y = Y_train, X_new = X_test, pl = pl, pr = pr, dl = dl, dr = dr, f = f, g = g)
  dat.tr <- data.frame("X" = t(X.d.tr),"Y"=as.factor(Y_train))
  dat.te <- data.frame("X" = t(X.d.te), "Y" = as.factor(Y_test))
  md <- qda(Y ~ ., data = dat.tr)
  return(sum(Y_test == predict(md, dat.te)$class) / length(Y_test))
}
stopCluster(cl)

d = 4
# Perform the NSPGSIRCP algorithm
res <- NSPGSIRCP(X = X_train, Y = matrix(Y_train, length(Y_train), 1), pl = pl, pr = pr, d = d,
                 thre = 0.001, max_iter = 50, kernel_u = "gaussian", kernel_v = "gaussian", kernel_Y = "discrete",
                 epsilon_u = epsilon.u, epsilon_v = epsilon.v, epsilon_x = epsilon.x)
f <- res$f
g <- res$g

# Predict the new data
X.d.tr <- NSPGSIR_predict_CP(X = X_train, Y = matrix(Y_train, length(Y_train), 1), X_new = X_train, pl = pl, pr = pr, d = d, f = f, g = g)
X.d.te <- NSPGSIR_predict_CP(X = X_train, Y = matrix(Y_train, length(Y_train), 1), X_new = X_test, pl = pl, pr = pr, d = d, f = f, g = g)

dat.tr <- data.frame("X" = t(X.d.tr), "Y" = as.factor(Y_train))
dat.te <- data.frame("X" = t(X.d.te), "Y" = as.factor(Y_test))

# Train and test the model
md <- qda(Y ~ ., data = dat.tr)
accuracy <- sum(Y_test == predict(md, dat.te)$class) / length(Y_test)

# Save the accuracy result
write(accuracy, file = "results.txt", append = TRUE)
print(accuracy)






















