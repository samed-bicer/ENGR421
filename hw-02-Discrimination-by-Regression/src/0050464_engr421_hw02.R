#############################################################
####                                                     ####
####  Mehmet Samed Bicer - 0050464                       ####
####  ENGR421 Homework-02, Discrimination by Regression  ####
####  Koc University, Istanbul - 21-Oct-19               ####
####                                                     ####
#############################################################

# set working directory
#setwd("C:/Users/MBICER14/Desktop/ENGR421/hw-02")

# global variables for learning parameters
eta <- 0.0001 
epsilon <- 1e-3 
max_iteration <- 500

# to avoid log(0) in case of returning inf, safelog is written as follows
safelog <- function(x) {
  return (log(x + 1e-100))
}

# sigmoid function 
sigmoid <- function(X, W, w0) {
  return (1 / (1 + exp(- sweep(x = (X %*% W), MARGIN = 2, STATS = w0, FUN = "+"))))
}

# define the gradient functions for w and w0
gradient_W <- function(X, y_truth, y_predicted) {
  return(- t(X) %*% ((y_truth - y_predicted) * y_predicted * (1 - y_predicted)) )
}

gradient_w0 <- function(y_truth, y_predicted) {
  return (- colSums((y_truth - y_predicted) * y_predicted * (1 - y_predicted)))
}

# read data from csv files
data_set <- read.csv(file = "hw02_images.csv", header = FALSE)
labels <- read.csv(file = "hw02_labels.csv", header = FALSE)

# split given dataset into train and test sets
x_train <- as.matrix(data_set[1:500,])
x_test <- as.matrix(data_set[501:1000,])

labels_train <- labels[1:500,]
labels_test <- labels[501:1000,]

# number of classes and number of images for train and test sets
K <- max(labels)
N <- nrow(x_train)

# create y_truth and y_test matrix
# these matrixes are expanded into their 5 columns variants
y_truth <- matrix(0, N, K)
y_truth[cbind(1:N, labels_train)] <- 1

y_test <- matrix(0, N, K)
y_test[cbind(1:N, labels_test)] <- 1

# read initial W and w0 parameters
W <- as.matrix(read.csv(file = "initial_w.csv", header = FALSE))
w0 <- as.matrix(read.csv(file = "initial_w0.csv", header = FALSE))

# initialize loop parameters
iteration <- 1
objective_values <- c()

while (1) {
  # predict y values using sigmoid function for train data
  y_predicted <- sigmoid(x_train, W, w0)
  
  # update objective values array
  objective_values <- c(objective_values, sum(0.5 * (y_truth - y_predicted)^2))
  
  # keep old W and w0 values
  W_old <- W
  w0_old <- w0
  
  # learn W and w0 using gradient descent
  W <- W - eta * gradient_W(x_train, y_truth, y_predicted)
  w0 <- w0 - eta * gradient_w0(y_truth, y_predicted)
  
  # break the loop if the desired improvement couldn't be achieved
  if (sqrt(sum((w0 - w0_old)^2) + sum((W - W_old)^2)) < epsilon) {
    break
  }
  
  # condition check for max iteration
  if (iteration == max_iteration) {
    break
  }
  
  # update iteration counter
  iteration <- iteration + 1
  
}

# plot errors through iterations
plot(1:iteration, objective_values, type = "l", 
     lwd = 2, las = 1, 
     xlab = "Iteration", ylab = "Error")

# calculate confusion matrix for train data
y_train <- rowSums(sapply(X = 1:5, FUN = function(c) {y_truth[,c] * c}))
y_predicted <- apply(y_predicted, 1, which.max)
confusion_matrix_train <- table(y_predicted, y_train)

# display confusion matrix for train data
print(confusion_matrix_train)

# predict y values using sigmoid function for test data
y_predicted_test <- sigmoid(x_test, W, w0)

# calculate confusion matrix for test data
y_test = rowSums(sapply(X = 1:5, FUN = function(c) {y_test[,c] * c}))
y_predicted_test <- apply(y_predicted_test, 1, which.max)
confusion_matrix_test <- table(y_predicted_test, y_test)

# display confusion matrix for train data
print(confusion_matrix_test)

