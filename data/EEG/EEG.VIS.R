library(foreach)
library(doParallel)
library(matrixcalc)
library(randomForest)
library(MASS)
library(caret)
library(class)
library(Rcpp)
file_path <- "/Users/djlin/Desktop/TensorSDR/func"
source(paste(file_path, "GSIR.R", sep="/"))
source(paste(file_path, "SIR.R", sep="/"))
source(paste(file_path, "DF.R", sep="/"))
sourceCpp(paste(file_path, "NDF.cpp", sep="/"))
file_path2 <- "/Users/djlin/Desktop/TensorSDR/RealData/EEG"
X.ac <- read.table(paste(file_path2, "alcoholic_data.txt", sep="/"))
X.co <- read.table(paste(file_path2, "control_data.txt", sep="/"))
X <- rbind(X.ac,X.co)
Y <- c(rep(1,77),rep(0,45))
##########GSIR############
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
X_train <- as.matrix(X.n)
Y_train <- Y
gsir.train <- gsir(x = t(X_train),y = Y_train,ytype = "categorical",complex_x = 0.1,complex_y = 0.1,r = 2,ex = 1,ey = 1)
label <- knn(train =gsir.train,test = gsir.train,cl = Y_train,k = 5)
print(label == Y)
print(sum(label == Y))
x11 = gsir.train[,1]
x21 = gsir.train[,2]
color_vector <- ifelse(Y == 0, "slategray","red")
# Scatter Plot
library(ggplot2)
Df <- data.frame(x11 = x11, x21 = x21, color_vector = color_vector)
p1 <- ggplot(Df, aes(x = x11, y = x21, color = color_vector)) +
  geom_point(size = 1.8, alpha = 0.5) +
  scale_color_manual(values =  c("red","slategray"), guide = "none") + 
  labs(title = "Generalized Slice Inverse Regression", x = "First Dimension", y = "Second Dimension") +
  theme_classic() +
  theme(
    plot.title = element_text(hjust = 0.5, size = 10, face = "bold"),  # Center and enlarge title
    axis.title = element_text(size = 10)  # Enlarge axis labels
  )
ggsave(paste(file_path2, "GSIR.png", sep="/"), plot = p1, width = 6, height = 6, dpi = 400)
############SIR#############
X_train <- as.matrix(X.n)
Y_train <- Y
beta <- sir(x = t(X_train),y = matrix(Y_train,ncol = 1),h = 2,r = 2,ytype = "categorical")
sir.train <- t(X_train)%*%beta
label <- knn(train =sir.train,test = sir.train,cl = Y_train,k = 5)
print(label == Y)
print(sum(label == Y))
x11 = sir.train[,1]
x21 = sir.train[,2]
color_vector <- ifelse(Y == 0, "slategray","red")
# Scatter Plot
library(ggplot2)
Df <- data.frame(x11 = x11, x21 = x21, color_vector = color_vector)
p2 <- ggplot(Df, aes(x = x11, y = x21, color = color_vector)) +
  geom_point(size = 1.8, alpha = 0.5) +
  scale_color_manual(values = c("red","slategray"), guide = "none") + 
  labs(title = "Slice Inverse Regression", x = "First Dimension", y = "Second Dimension") +
  theme_classic() +
  theme(
    plot.title = element_text(hjust = 0.5, size = 10, face = "bold"),  # Center and enlarge title
    axis.title = element_text(size = 10)  # Enlarge axis labels
  )
ggsave(paste(file_path2, "SIR.png", sep="/"), plot = p2, width = 6, height = 6, dpi = 400)
############DF#############
pl <- ml
pr <- mr
dl <- 2
dr <- 1
h <- 2
p.t <- array(data = 0,dim = h)
index <- list()
for (k in 0:1) {
  p.t[k+1] <- sum(Y_train==k)/length(Y_train)
  index[[k+1]] <- which(Y_train== k)
}
epsilon <- 1e-6
result <- df(X = X_train,Y = Y_train,index = index,p = p.t,h = h,epsilon = epsilon,iteration = 2000,pl = pl,pr = pr,dl = dl,dr = dr)
a <- result[[1]]
b <- result[[2]]
Xn.train1 <- matrix(data = 0,nrow = dl*dr,ncol = ncol(X_train))
for (i in 1:ncol(Xn.train1)) {
  Xn.train1[,i] <- matrix(data = t(a)%*%matrix(X_train[,i],nrow = pl,ncol = pr)%*%b,nrow = dl*dr)
}
label <- knn(train =t(Xn.train1),test = t(Xn.train1),cl = Y_train,k = 5)
print(label == Y)
print(sum(label == Y))
x11 = Xn.train1[1,]
x21 = Xn.train1[2,]
color_vector <- ifelse(Y == 0, "slategray","red")
# Scatter Plot
library(ggplot2)
Df <- data.frame(x11 = x11, x21 = x21, color_vector = color_vector)
p3 <- ggplot(Df, aes(x = x11, y = x21, color = color_vector)) +
  geom_point(size = 2, alpha = 0.5) +
  scale_color_manual(values =c("red","slategray"), guide = "none")  + 
  labs(title = "Dimension Folding", x = "First Dimension", y = "Second Dimension") +
  theme_classic() +
  theme(
    plot.title = element_text(hjust = 0.5, size = 10, face = "bold"),  # Center and enlarge title
    axis.title = element_text(size = 10)  # Enlarge axis labels
  )
p3
write.csv(Df,paste(file_path2, "DF1515.csv", sep="/"))
ggsave(paste(file_path2, "DF1515.png", sep="/"), plot = p3, width = 6, height = 6, dpi = 400)
#####NDF###############
Df <- read.csv("/Users/djlin/Desktop/NDFTu1010.csv")\
p4 <- ggplot(Df, aes(x = x11, y = x21, color = color_vector)) +
  geom_point(size = 2, alpha = 0.5) +
  scale_color_manual(values = c("red","slategray"), guide = "none") + 
  labs(title = "Nonlinear Dimension Folding Tucker Decomposition", x = "First Dimension", y = "Second Dimension") +
  theme_classic() +
  theme(
    plot.title = element_text(hjust = 0.5, size = 10, face = "bold"),  # Center and enlarge title
    axis.title = element_text(size = 10)  # Enlarge axis labels
  )
p4
write.csv(Df,paste(file_path2, "NDFTu1010.csv", sep="/"))
ggsave(paste(file_path2, "NDFTu1010.png", sep="/"), plot = p4, width = 6, height = 6, dpi = 400)


