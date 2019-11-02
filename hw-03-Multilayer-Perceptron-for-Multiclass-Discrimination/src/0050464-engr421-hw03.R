##############################################################
####                                                      ####
####  Mehmet Samed Bicer - 0050464                        ####
####  ENGR421 Homework-03                                 ####
####  Multilayer Perceptron for Multiclass Discrimination ####
####  Koc University, Istanbul - 31-Oct-19                ####
####                                                      ####
##############################################################

# set working directory
#setwd("C:/Users/MBICER14/Desktop/ENGR421/hw-03")

# global variables for learning parameters
eta <- 0.0005
epsilon <- 1e-3 
max_iteration <- 500

# to avoid log(0) in case of returning inf, safelog is written as follows
safelog <- function(x) {
  return (log(x + 1e-100))
}

# sigmoid function 
sigmoid <- function(x) {
  return (1 / (1 + exp(-x)))
}

# softmax function 
softmax <- function(Z, V) {
  scores <- exp(Z %*% t(V))
  return(scores / matrix(rowSums(scores), nrow(scores), ncol(scores), byrow = FALSE))
}

# read data from csv files
data_set <- read.csv(file = "hw03_images.csv", header = FALSE)
labels <- read.csv(file = "hw03_labels.csv", header = FALSE)

# split given dataset into train and test sets
x_train <- as.matrix(data_set[1:500,])
x_test <- as.matrix(data_set[501:1000,])

labels_train <- labels[1:500,]
labels_test <- labels[501:1000,]

# number of classes and number of images for train and test sets
K <- max(labels)
N <- nrow(x_train)
D <- nrow(data_set)
H <- 20

# create y_train and y_test matrix
# these matrixes are expanded into their 5 columns variants
y_train <- matrix(0, N, K)
y_train[cbind(1:N, labels_train)] <- 1

y_test <- matrix(0, N, K)
y_test[cbind(1:N, labels_test)] <- 1

# read initial W and V parameters
W <- as.matrix(read.csv(file = "initial_W.csv", header = FALSE))
V <- as.matrix(read.csv(file = "initial_V.csv", header = FALSE))
V <- t(V[1:20,])

# calculate Z 
Z <- sigmoid(cbind(1, x_train) %*% W)

# predict y values using sigmoid function for train data
y_predicted <- softmax(Z,V) 

# create objective values matrix
objective_values <- -sum(y_train * safelog(y_predicted))

# initialize loop parameter
iteration <- 1

# get start time for logging
loop_start <- Sys.time()

while (1) {
  # logging
  iteration_start <- Sys.time()
  cat("iteration:", iteration, "\t") 
  
  # instead of 1:N, sample(N) is used since 1:N uses 500 iteration
  # sample(N) might use less iteration to reach desired objective value 
  for (i in sample(N)) {
    # calculate hidden nodes
    Z[i,] <- sigmoid(c(1, x_train[i,]) %*% W)
    
    # calculate output node
    y_predicted[i,] <- softmax(Z[i,],V)
    
    # update V values
    V <- V + eta * (y_train[i,] - y_predicted[i,]) * c(1, Z[i,])
    
    # update W values
    for (h in 1:20) {
      W[,h] <- W[,h] + eta * sum((y_train[i,] - y_predicted[i,]) * V[,h]) * Z[i, h] * (1 - Z[i, h]) * c(1, x_train[i,])
    }
    # for loop perform better than sapply in terms of execution time 
    #W <-sapply(X = 1:20, FUN = function(c) {W[,c] + eta * sum((y_train[i,] - y_predicted[i,]) * V[,c]) * Z[i, c] * (1 - Z[i, c]) * c(1, x_train[i,])})
    
  }
  
  # calculate predicted values and add objective value
  Z <- sigmoid(cbind(1, x_train) %*% W)
  y_predicted <- softmax(Z,V) 
  objective_values <- c(objective_values, -sum(y_train * safelog(y_predicted)))
    
  # exit condition for while loop
  if (abs(objective_values[iteration + 1] - objective_values[iteration]) < epsilon || iteration >= max_iteration) {
    iteration_end <- Sys.time()
    print(iteration_end - iteration_start)
    break
  }
  
  # update iteration counter  
  iteration <- iteration + 1
  
  # logging
  iteration_end <- Sys.time()
  print(iteration_end - iteration_start)
  
}

# logging  
loop_end <- Sys.time()
cat("\nTotal Time: ")
print(loop_end - loop_start)

# print W and V matrices
print(W)
print(V)

# plot errors through iterations
plot(1:(iteration + 1), objective_values, type = "l", 
     lwd = 2, las = 1, 
     xlab = "Iteration", ylab = "Error")

# calculate confusion matrix for train data
y_train <- rowSums(sapply(X = 1:5, FUN = function(c) {y_train[,c] * c}))
y_predicted <- apply(y_predicted, 1, which.max)
confusion_matrix_train <- table(y_predicted, y_train)

# display confusion matrix for train data
print(confusion_matrix_train)

# predict y values using sigmoid function for test data
Z <- sigmoid(cbind(1, x_test) %*% W)
y_predicted_test <- softmax(Z,V)

# calculate confusion matrix for test data
y_test = rowSums(sapply(X = 1:5, FUN = function(c) {y_test[,c] * c}))
y_predicted_test <- apply(y_predicted_test, 1, which.max)
confusion_matrix_test <- table(y_predicted_test, y_test)

# display confusion matrix for train data
print(confusion_matrix_test)
