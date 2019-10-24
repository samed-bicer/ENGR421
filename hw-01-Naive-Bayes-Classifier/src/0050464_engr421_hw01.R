########################################################
####                                                ####
####  Mehmet Samed Bicer - 0050464                  ####
####  ENGR421 Homework-01, Naïve Bayes' Classifier  ####
####  Koc University, Istanbul - 14-Oct-19          ####
####                                                ####
########################################################

# set working directory
#setwd("C:/Users/MBICER14/Desktop/ENGR421/hw-01")

# to avoid log(0) in case of returning inf, safelog is written as follows
safelog <- function(x) {
  return (log(x + 1e-100))
}

# read data    
data_set <- read.csv(file = "hw01_images.csv", header = FALSE)
labels <- read.csv(file = "hw01_labels.csv", header = FALSE)

# split given dataset into train and test sets
x_train <- as.matrix(data_set[1:200,])
x_test <- as.matrix(data_set[201:400,])

y_train <- labels[1:200,]
y_test <- labels[201:400,]

# number of classes and images of train and test sets
K <- max(labels)
N <- NROW(x_train)

# compute mean of each column for both genders
means <- as.matrix(sapply(X = 1:K, FUN = function(c) { colMeans(x_train[y_train == c,]) }))

#print(means[,1])
#print(means[,2])

# compute deviation of each column for both genders
# deviations <- as.matrix(sapply(X = 1:K, FUN = function(c) { sqrt(colMeans((x_train[y_train == c,] - means[,c])^2)) })) 
deviations <- as.matrix(sapply(X = 1:K, FUN = function(c) { apply(x_train[y_train == c,], 2, sd) })) 

#print(deviations[,1])
#print(deviations[,2])

# compute prior probabilities of both genders
class_priors <- sapply(X = 1:K, FUN = function(c) { mean(y_train == c) })

#print(class_priors)

# score functions for female and male according to train dataset
score_func_train_female <- female <- sapply(X = 1:N, FUN = function(c) { sum(- 0.5 * log(2 * pi * deviations[,1]^2) - 0.5 * (x_train[c,] - means[,1])^2 / deviations[,1]^2) + log(class_priors[1]) })
score_func_train_male <- sapply(X = 1:N, FUN = function(c) { sum(- 0.5 * log(2 * pi * deviations[,2]^2) - 0.5 * (x_train[c,] - means[,2])^2 / deviations[,2]^2) + log(class_priors[2]) })
score_func_train <- rbind(score_func_train_female, score_func_train_male)

# predictions for train data
prediction_train <- sapply(X = 1:N, FUN = function(c) { match(max(score_func_train[,c]), score_func_train) })
prediction_train <- (prediction_train - 1) %% 2 + 1

# calculate confusion matrix for training data
conf_train_female <- sapply(X = 1:K, FUN = function(c) { sum(!is.na(match(prediction_train[y_train == 1], c))) })
conf_train_male <- sapply(X=1:K, FUN = function(c) { sum(!is.na(match(prediction_train[y_train == 2], c))) })
conf_train <- rbind(conf_train_female, conf_train_male)

# displaying confusion matrix for train data
conf_train

# score functions for female and male according to test dataset
score_func_test_female <- female <- sapply(X = 1:N, FUN = function(c) { sum(- 0.5 * log(2 * pi * deviations[,1]^2) - 0.5 * (x_test[c,] - means[,1])^2 / deviations[,1]^2) + log(class_priors[1]) })
score_func_test_male <- sapply(X = 1:N, FUN = function(c) { sum(- 0.5 * log(2 * pi * deviations[,2]^2) - 0.5 * (x_test[c,] - means[,2])^2 / deviations[,2]^2) + log(class_priors[2]) })
score_func_test <- rbind(score_func_test_female, score_func_test_male)

# predictions for test data
prediction_test <- sapply(X = 1:N, FUN = function(c) {match(max(score_func_test[,c]), score_func_test)})
prediction_test <- (prediction_test - 1) %% 2 + 1

# calculate confusion matrix for test data
conf_test_female <- sapply(X = 1:K, FUN = function(c) { sum(!is.na(match(prediction_test[y_test == 1], c))) })
conf_test_male <- sapply(X = 1:K, FUN = function(c) { sum(!is.na(match(prediction_test[y_test == 2], c))) })
conf_test <- rbind(conf_test_female, conf_test_male)

# displaying confusion matrix for test data
conf_test

