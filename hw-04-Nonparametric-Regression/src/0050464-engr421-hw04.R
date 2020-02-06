################################################
####                                        ####
####  Mehmet Samed Bicer - 0050464          ####
####  ENGR421 Homework-04                   ####
####  Nonparametric Regression              ####
####  Koc University, Istanbul - 12-Nov-19  ####
####                                        ####
################################################

# set working directory
#setwd("C:/Users/MBICER14/Desktop/ENGR421/hw-04")

# global variables for learning parameters
bin_width <- 0.37
origin <- 1.5

# read data from csv files
data_set <- read.csv(file = "hw04_data_set.csv", header = TRUE)

# split given dataset into x and y values
x <- data_set$eruptions
y <- data_set$waiting

# split given dataset into train and test sets
train_set <- data_set[1:150,]
test_set <- data_set[151:272,]

# split train and test dataset into x and y values
x_train <- as.matrix(train_set$eruptions)
y_train <- as.matrix(train_set$waiting)

x_test <- as.matrix(test_set$eruptions)
y_test <- as.matrix(test_set$waiting)

# number of classes and number of images for train and test sets
K <- max(y_train)
N <- nrow(x_train)

# parameters for the plot
point_colors <- c("blue", "red")
colorMatrix <- c(rep(1, 150), rep(2, 122))
minimum_x <- min(data_set$eruptions)
maximum_x <- max(data_set$eruptions)
minimum_y <- min(data_set$waiting)
maximum_y <- max(data_set$waiting)
data_interval <- seq(from = minimum_x, to = maximum_x, by = 0.01)

# borders and p_head calculation
left_borders <- seq(from = origin, to = maximum_x - bin_width, by = bin_width)
right_borders <- seq(from = origin + bin_width, to = maximum_x, by = bin_width)
p_head <- sapply(1:length(left_borders), function(c) {sum(y_train[left_borders[c] < x_train & x_train <= right_borders[c]]) / sum(left_borders[c] < x_train & x_train <= right_borders[c])}) 

plot(x_train, y_train, 
     type = "p", pch = 19, col = "blue",
     xlim = c(origin, maximum_x), ylim = c(minimum_y, maximum_y),
     xlab = "Eruption time (min)", ylab = "Waiting time to next eruption (min)",
     las = 1, main = sprintf("h = %g", bin_width))
points(x_test, y_test, type = "p", pch = 19, col = "red")

for (i in 1:length(left_borders)) {
  lines(c(left_borders[i], right_borders[i]), c(p_head[i], p_head[i]), col = "black")
  if (i < length(left_borders)) {
    lines(c(right_borders[i], right_borders[i]), c(p_head[i], p_head[i + 1]), col = "black")
  }
}

legend("topleft", legend = c("training", "test"),
       col = point_colors, pch = 19)

# RMSE for Regressogram
RSME <- sqrt(sum(unlist(sapply(1:length(y_test), function(i){(y_test[i] - p_head[(as.integer((x_test[i] - origin) / bin_width))]) ^ 2}))) / length(y_test))
sprintf("Regressogram => RMSE is %g when h is %g", RSME, bin_width)


################################################################################################
# Running Mean Smoother
g_head <- sapply(data_interval, function(x) {sum(y_train[(x - 1.5 * bin_width) < x_train & x_train <= (x + 1.5 * bin_width)])/ sum((x - 1.5 * bin_width) < x_train & x_train <= (x + 1.5 * bin_width))}) 
plot(x_train, y_train, type = "p", pch = 19, col = "blue",
     ylim = c(min(y_train), max(y_train)), xlim = c(origin, maximum_x),
     xlab = "Eruption time (min)", ylab = "Waiting time to next eruption (min)", 
     las = 1, main = sprintf("h = %g", bin_width))

points(x_test, y_test, type = "p", pch = 19, col = "red")
lines(data_interval, g_head, type = "l", lwd = 2, col = "black")

legend("topleft", legend = c("training", "test"),
       col = point_colors, pch = 19)

# RMSE for Running Mean Smoother
RSME_rms <- sqrt(sum((y_test - g_head[(x_test - 1.5) * 100] ) ^ 2) / length(y_test))
sprintf("Running Mean Smoother => RMSE is %g when h is %g", RSME_rms, bin_width)

################################################################################################
# Kernel Smoother
g_head <- sapply(data_interval, function(x) {sum(1 / sqrt(2 * pi) * exp(-0.5 * (x - x_train)^2 / bin_width^2) * y_train)/ sum(1 / sqrt(2 * pi) * exp(-0.5 * (x - x_train)^2 / bin_width^2))}) 

plot(x_train, y_train, type = "p", pch = 19, col = "blue",
     ylim = c(min(y_train), max(y_train)), xlim = c(origin, maximum_x),
     xlab = "Eruption time (min)", ylab = "Waiting time to next eruption (min)", 
     las = 1, main = sprintf("h = %g", bin_width))

points(x_test, y_test, type = "p", pch = 19, col = "red")
lines(data_interval, g_head, type = "l", lwd = 2, col = "black")

legend("topleft", legend = c("training", "test"),
       col = point_colors, pch = 19)

# # RMSE for Kernel Smoother
RSME_ks <- sqrt(sum((y_test - g_head[x_test * 100])^2)/length(y_test))
sprintf("Kernel Smoother => RMSE is %g when h is %g", RSME_ks, bin_width)

