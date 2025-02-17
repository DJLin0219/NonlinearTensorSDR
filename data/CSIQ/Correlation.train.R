#######################################
library(Rcpp)
library(readxl)
library(ggplot2)
library(tools)
library(np)
library(magick)
file_path = "/Users/djlin/Desktop/TensorSDR/func"
sourceCpp(paste(file_path,"NDF.cpp",sep="/"))
source(paste(file_path,"GSIR.R",sep="/"))
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
file_index <- sprintf("%02d", k)
file_path2 <- paste0("/Users/djlin/Desktop/TensorSDR/RealData/CSIQ/Img/Imgs", file_index,"/")
img_files <- list.files(file_path2, pattern = "\\.png$", full.names = TRUE)
img_matrices <- list()
for (i in seq_along(img_files)) {
  img <- image_read(img_files[i])
  img_gray <- image_convert(img, colorspace = "gray")
  img_data <- image_data(img_gray, channels = "gray")
  img_matrix <- matrix(as.double(img_data[1,,]), 512, 512)
  img_matrices[[i]] <- img_matrix
}
combined_matrix <- do.call(cbind, lapply(img_matrices, function(x) as.vector(x)))
print(seq_along(img_files))
data <- read_excel("/Users/djlin/Desktop/TensorSDR/RealData/CSIQ/Img/ImgValues.xlsx")
Y <- data$dmos[data$id == k]
length(Y)
Y <- Y[c(1:5,21:25,26:max(seq_along(img_files)),16:20,6:10,11:15)]
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
####################################################################
####################################################################
#set.seed(2024)
ex <- 100
ey <- 100
complex.x <- 0.001
complex.y <- 0.001
X.train <- X.test <-  X.n
Y.train <- Y.test <- Y
Y.gsir.pred <- (gsir.predict(x = t(X.train),y = matrix(Y.train,ncol=1),x_new = t(X.train),ytype = "continuous",ex = ex,ey = ey,complex_x = complex.x,complex_y = complex.y,r = 8))
data.train.gsir <- data.frame("Y" = Y.train,"X" = Y.gsir.pred)
kernel.model.gsir1 <- npreg(Y~X.1,data = data.train.gsir, regtype = "ll", bwmethod = "cv.aic", gradients = TRUE)
kernel.model.gsir2 <- npreg(Y~X.1+X.2,data = data.train.gsir, regtype = "ll", bwmethod = "cv.aic", gradients = TRUE)
kernel.model.gsir3 <- npreg(Y~X.1+X.2+X.3,data = data.train.gsir, regtype = "ll", bwmethod = "cv.aic", gradients = TRUE)
kernel.model.gsir4 <- npreg(Y~X.1+X.2+X.3+X.4,data = data.train.gsir, regtype = "ll", bwmethod = "cv.aic", gradients = TRUE)
kernel.model.gsir5 <- npreg(Y~X.1+X.2+X.3+X.4+X.5,data = data.train.gsir, regtype = "ll", bwmethod = "cv.aic", gradients = TRUE)
kernel.model.gsir6 <- npreg(Y~X.1+X.2+X.3+X.4+X.5+X.6,data = data.train.gsir, regtype = "ll", bwmethod = "cv.aic", gradients = TRUE)
kernel.model.gsir7 <- npreg(Y~X.1+X.2+X.3+X.4+X.5+X.6+X.7,data = data.train.gsir, regtype = "ll", bwmethod = "cv.aic", gradients = TRUE)
kernel.model.gsir8 <- npreg(Y~X.1+X.2+X.3+X.4+X.5+X.6+X.7+X.8,data = data.train.gsir, regtype = "ll", bwmethod = "cv.aic", gradients = TRUE)
####################################################################
pl <- 8
pr <- 8
epsilon.u <- 1e-6
epsilon.v <- 1e-6
epsilon.x <- 1e-8
samp <- 1:length(Y)
X.train <- X.test <-  X.n
Y.train <- Y.test <- Y
d <- 8
#set.seed(2024)
result <- NSPGSIRCP(X = X.train,Y = matrix(Y.train,ncol=1),pl = pl,pr = pr,d = d,thre=1e-4,iteration=20,
                     epsilon_u = epsilon.v,epsilon_v = epsilon.v,epsilon_x = epsilon.x,kernel_u="gaussian",kernel_v="gaussian",
                     kernel_Y="gaussian")
f <- result$f
g <- result$g
h <- result$h
X.train.d <- NSPGSIR_predict_CP(X = X.train,Y = matrix(Y.train,ncol=1),X_new = X.train,pl = pl,pr = pr,d = d,f = f,g = g)
data.train <- data.frame("Y" = Y.train,"X" = t(X.train.d))
cor(t(X.train.d)[,1],Y.train)
####################################################################
kernel.model1 <- npreg(Y~X.1,data = data.train, regtype = "ll", bwmethod = "cv.aic", gradients = TRUE)
kernel.model2 <- npreg(Y~X.1+X.2,data = data.train, regtype = "ll", bwmethod = "cv.aic", gradients = TRUE)
kernel.model3 <- npreg(Y~X.1+X.2+X.3,data = data.train, regtype = "ll", bwmethod = "cv.aic", gradients = TRUE)
kernel.model4 <- npreg(Y~X.1+X.2+X.3+X.4,data = data.train, regtype = "ll", bwmethod = "cv.aic", gradients = TRUE)
kernel.model5 <- npreg(Y~X.1+X.2+X.3+X.4+X.5,data = data.train, regtype = "ll", bwmethod = "cv.aic", gradients = TRUE)
kernel.model6 <- npreg(Y~X.1+X.2+X.3+X.4+X.5+X.6,data = data.train, regtype = "ll", bwmethod = "cv.aic", gradients = TRUE)
kernel.model7 <- npreg(Y~X.1+X.2+X.3+X.4+X.5+X.6+X.7,data = data.train, regtype = "ll", bwmethod = "cv.aic", gradients = TRUE)
kernel.model8 <- npreg(Y~X.1+X.2+X.3+X.4+X.5+X.6+X.7+X.8,data = data.train, regtype = "ll", bwmethod = "cv.aic", gradients = TRUE)
####################################################################
Rsq <- data.frame("d"=0:8,"rsq.ndf"=c(0,kernel.model1$R2,kernel.model2$R2,kernel.model3$R2,kernel.model4$R2,kernel.model5$R2,
                                      kernel.model6$R2,kernel.model7$R2,kernel.model8$R2),
                  "rsq.gsir" = c(0,kernel.model.gsir1$R2,kernel.model.gsir2$R2,kernel.model.gsir3$R2,kernel.model.gsir4$R2,kernel.model.gsir5$R2,
                                 kernel.model.gsir6$R2,kernel.model.gsir7$R2,kernel.model.gsir8$R2),
                  "cor.ndf" = c(0,kernel.model1$CORR,kernel.model2$CORR,kernel.model3$CORR,kernel.model4$CORR,kernel.model5$CORR,
                                kernel.model6$CORR,kernel.model7$CORR,kernel.model8$CORR),
                  "cor.gsir" = c(0,kernel.model.gsir1$CORR,kernel.model.gsir2$CORR,kernel.model.gsir3$CORR,kernel.model.gsir4$CORR,kernel.model.gsir5$CORR,
                                 kernel.model.gsir6$CORR,kernel.model.gsir7$CORR,kernel.model.gsir8$CORR))
#################################################################
p1 <-ggplot(data = Rsq, aes(x = d)) +
  geom_line(aes(y = rsq.ndf, color = "NDF.CP"), size = 0.8) +   
  geom_point(aes(y = rsq.ndf, color = "NDF.CP"), size = 2) +    
  geom_line(aes(y = rsq.gsir, color = "GSIR"), size = 0.5) + 
  geom_point(aes(y = rsq.gsir, color = "GSIR"), size = 2) +  
  scale_color_manual(values = c("NDF.CP" = "red", "GSIR" = "black")) +
  geom_hline(yintercept = 0.8, linetype = "dashed", size = 0.3, color = "black") +  
  geom_hline(yintercept = 0.9, linetype = "dashed", size = 0.3, color = "black") +  
  geom_hline(yintercept = 0.7, linetype = "dashed", size = 0.3, color = "black") +  
  geom_hline(yintercept = 1.0, linetype = "dashed", size = 0.3, color = "black") +  
  geom_text(aes(x = -Inf, y = 0.8, label = "0.8"), vjust = -0.5, hjust = -0.1, color = "black", size = 3) + 
  geom_text(aes(x = -Inf, y = 0.9, label = "0.9"), vjust = -0.5, hjust = -0.1, color = "black", size = 3) + 
  geom_text(aes(x = -Inf, y = 1.0, label = "1.0"), vjust = -0.5, hjust = -0.1, color = "black", size = 3) + 
  geom_text(aes(x = -Inf, y = 0.7, label = "0.7"), vjust = -0.5, hjust = -0.1, color = "black", size = 3) + 
  labs(title = toTitleCase(data$image[data$id == k][1]),
       x = "Reduced Dimension",
       y = "Rsquare",
       color = "Methods") +                       
  theme_light() +                             
  theme(
    plot.title = element_text(size = 15, hjust = 0.5),
    legend.position = c(0.85, 0.15)
  )
p1
#################################################################
p2 <-ggplot(data = Rsq, aes(x = d)) +
  geom_line(aes(y = cor.ndf, color = "NDF.CP"), size = 0.8) +   
  geom_point(aes(y = cor.ndf, color = "NDF.CP"), size = 2) +    
  geom_line(aes(y = cor.gsir, color = "GSIR"), size = 0.5) + 
  geom_point(aes(y = cor.gsir, color = "GSIR"), size = 2) +  
  scale_color_manual(values = c("NDF.CP" = "red", "GSIR" = "black")) +
  geom_hline(yintercept = 0.90, linetype = "dashed", size = 0.3, color = "black") +  
  geom_hline(yintercept = 0.95, linetype = "dashed", size = 0.3, color = "black") +  
  geom_hline(yintercept = 1.00, linetype = "dashed", size = 0.3, color = "black") +  
  geom_text(aes(x = -Inf, y = 0.90, label = "0.90"), vjust = -0.5, hjust = -0.1, color = "black", size = 3) + 
  geom_text(aes(x = -Inf, y = 0.95, label = "0.95"), vjust = -0.5, hjust = -0.1, color = "black", size = 3) + 
  geom_text(aes(x = -Inf, y = 1.00, label = "1.00"), vjust = -0.5, hjust = -0.1, color = "black", size = 3) + 
  labs(title = toTitleCase(data$image[data$id == k][1]),
       x = "Reduced Dimension",
       y = "Pearson Correlation",
       color = "Methods") +                       
  theme_light() +                             
  theme(
    plot.title = element_text(size = 15, hjust = 0.5),
    legend.position = c(0.85, 0.15)
  )
p2
file_path2 <-  paste0("/Users/djlin/Desktop/Rsq/", toTitleCase(data$image[data$id == k][1]),"R2.png")
file_path3 <-  paste0("/Users/djlin/Desktop/Corr/", toTitleCase(data$image[data$id == k][1]),"Corr.png")
ggsave(file_path2, plot = p1, width = 10, height = 6, units = "in", dpi = 300)
ggsave(file_path3, plot = p2, width = 10, height = 6, units = "in", dpi = 300)










 




