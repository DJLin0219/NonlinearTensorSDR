######################################
library(parallel)
args <- commandArgs(trailingOnly = TRUE)
k <- as.integer(args[1])
#######################################
library(Rcpp)
library(readxl)
library(ggplot2)
library(tools)
library(np)
sourceCpp("NDF.cpp")
#########################################
symmetry = function(a){
  return((a + t(a))/2)}
onorm=function(a) {
  return(eigen(round((a+t(a))/2,8))$values[1])
}
mppower=function(matrix,power,ignore){
  eig = eigen(matrix)
  eval = eig$values
  evec = eig$vectors
  m = length(eval[abs(eval)>ignore])
  if(m == 1){
    tmp = evec[,1:m]%*%t(evec[,1:m])*abs(eval[1:m]^power)
  }else
  {tmp = evec[,1:m]%*%diag(eval[1:m]^power)%*%t(evec[,1:m])}}
#########################################
data <- read_excel("Img/ImgValues.xlsx")
file_path = paste0("Img_Res/", toTitleCase(data$image[data$id == k][1]),".csv")
combined_matrix <- read.csv(file = file_path,header = T)
combined_matrix <- combined_matrix[,2:ncol(combined_matrix)]
Y <- data$dmos[data$id == k]
length(Y)
Y <- Y[c(1:5,21:25,26:ncol(combined_matrix),16:20,6:10,11:15)]
######################################################
pl <- 512
pr <- 512
ml <- 8
mr <- 8
X.t <- combined_matrix
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
######################################################
pl <- 8
pr <- 8
epsilon.u <- 5
epsilon.v <- 5
epsilon.x <- 1e-8
iteration <- 20
corr.gsir <- array(data = 0,dim = iteration)
corr.ndf <- array(data = 0,dim = iteration)
for (iter in 1:iteration) {
  set.seed(iter*2024)
  samp <- sample(1:length(Y),22)
  X.train <- X.n[,samp]
  X.test <- X.n[,-samp]
  Y.train <- Y[samp]
  Y.test <- Y[-samp]
  d <- 3
  set.seed(2023)
  result <- NSPGSIRCP(X = X.train,Y = matrix(Y.train,ncol=1),pl = pl,pr = pr,d = d,thre=1e-4,iteration=100,
                       epsilon_u = epsilon.v,epsilon_v = epsilon.v,epsilon_x = epsilon.x,kernel_u="gaussian",kernel_v="gaussian",
                       kernel_Y="gaussian")
  f <- result$f
  g <- result$g
  h <- result$h
  X.test.ndf.d <- NSPGSIR_predict_CP(X = X.train,Y = matrix(Y.train,ncol=1),X_new = X.test,pl = pl,pr = pr,d = d,f = f,g = g)
  X.train.ndf.d <- NSPGSIR_predict_CP(X = X.train,Y = matrix(Y.train,ncol=1),X_new = X.train,pl = pl,pr = pr,d = d,f = f,g = g)
  data.train.ndf <- data.frame("Y" = Y.train,"X" = t(X.train.ndf.d))
  kernel.model.ndf <- npreg(Y~X.1+X.2+X.3,data = data.train.ndf, regtype = "ll", bwmethod = "cv.aic", gradients = TRUE)
  Y.test.pred.ndf <- predict(kernel.model.ndf, newdata = data.frame("X" = t(X.test.ndf.d)))
  corr.ndf[iter] <- abs(cor(Y.test.pred.ndf,Y.test,method = "pearson"))
  #############################
  set.seed(2024)
  Y.gsir.pred <- (gsir.predict(x = t(X.train),y = matrix(Y.train,ncol=1),x_new = t(X.test),ytype = "continuous",ex = 1,ey = 1,complex_x = 1,complex_y = 1,r = 3))
  set.seed(2024)
  Y.gsir.train<- (gsir.predict(x = t(X.train),y = matrix(Y.train,ncol=1),x_new = t(X.train),ytype = "continuous",ex = 1,ey = 1,complex_x = 1,complex_y = 1,r = 3))
  data.train.gsir <- data.frame("Y" = Y.train,"X" = Y.gsir.train)
  kernel.model.gsir <- npreg(Y~X.1+X.2+X.3,data = data.train.gsir, regtype = "ll", bwmethod = "cv.aic", gradients = TRUE)
  Y.test.pred.gsir <- predict(kernel.model.gsir, newdata = data.frame("X" = Y.gsir.pred))
  corr.gsir[iter] <- abs(cor(Y.test.pred.gsir,Y.test,method = "pearson"))
  print(iter)
}
file_path4 <- paste0("/Users/djlin/Desktop/Corrgsir1/Corr", toTitleCase(data$image[data$id == k][1]),".csv")
file_path5 <- paste0("/Users/djlin/Desktop/Corrndf1/Corr", toTitleCase(data$image[data$id == k][1]),".csv")
write.csv(x = corr.gsir,file = file_path4)
write.csv(x = corr.ndf,file = file_path5)



