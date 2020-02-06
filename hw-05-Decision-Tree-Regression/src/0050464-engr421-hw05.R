################################################
####                                        ####
####  Mehmet Samed Bicer - 0050464          ####
####  ENGR421 Homework-05                   ####
####  Decision Tree Regression              ####
####  Koc University, Istanbul - 2-Dec-19   ####
####                                        ####
################################################

# set working directory
#setwd("C:/Users/MBICER14/Desktop/ENGR421/hw-05")

# read data from csv files
data_set <- read.csv(file = "hw05_data_set.csv", header = TRUE)

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
K_train <- max(y_train)
N_train <- nrow(x_train)

K_test <- max(y_test)
N_test <- nrow(x_test)

# Find the min and max values for X and Y
minimum_x <- min(x)
maximum_x <- max(x)
minimum_y <- min(y)
maximum_y <- max(y)

# generate a decision tree with given P
decisionTreeFunction <- function(P) {
  
  # create necessary data structures
  node_indices <- list()
  is_terminal <- c()
  need_split <- c()

  node_splits <- c()
  node_means <- c()
  
  # put all training instances into the root node
  node_indices <- list(1:N_train)
  is_terminal <- c(FALSE)
  need_split <- c(TRUE)
  
  # learning algorithm
  while (1) {
    # find nodes that need splitting
    split_nodes <- which(need_split)
    
    # check whether we reach all terminal nodes
    if (length(split_nodes) == 0) {
      break
    }
    
    # find best split positions for all nodes
    for (split_node in split_nodes) {
      data_indices <- node_indices[[split_node]]
      need_split[split_node] <- FALSE
      node_mean <- mean(y_train[data_indices])
      
      # check whether the condition is satisfied for terminal node
      if (length(x_train[data_indices]) <= P) {
        is_terminal[split_node] <- TRUE
        node_means[split_node] <- node_mean
      } 
      else {
        is_terminal[split_node] <- FALSE
        unique_values <- sort(unique(x_train[data_indices]))
        split_positions <- (unique_values[- 1] + unique_values[- length(unique_values)]) / 2
        split_scores <- rep(0, length(split_positions))
        
        for (s in 1:length(split_positions)) {
          left_indices <- data_indices[which(x_train[data_indices] <= split_positions[s])]
          right_indices <- data_indices[which(x_train[data_indices] > split_positions[s])]
          total_error <- 0
          
          if (length(left_indices) > 0) {
            mean <- mean(y_train[left_indices])
            total_error <- total_error + sum((y_train[left_indices] - mean) ^ 2)
          }
          
          if (length(right_indices) > 0) {
            mean <- mean(y_train[right_indices])
            total_error <- total_error + sum((y_train[right_indices] - mean) ^ 2)
          }
          
          split_scores[s] <- total_error / (length(left_indices) + length(right_indices))
        }
        
        if (length(unique_values) == 1) {
          is_terminal[split_node] <- TRUE
          node_means[split_node] <- node_mean
          next 
        }
        
        best_split <- split_positions[which.min(split_scores)]
        node_splits[split_node] <- best_split
        
        # create left node using the selected split
        left_indices <- data_indices[which(x_train[data_indices] < best_split)]
        node_indices[[2 * split_node]] <- left_indices
        is_terminal[2 * split_node] <- FALSE
        need_split[2 * split_node] <- TRUE
        
        # create left node using the selected split
        right_indices <- data_indices[which(x_train[data_indices] >= best_split)]
        node_indices[[2 * split_node + 1]] <- right_indices
        is_terminal[2 * split_node + 1] <- FALSE
        need_split[2 * split_node + 1] <- TRUE
      }
    }
  }
  return(list("splits"= node_splits, "means"= node_means, "is_terminal"= is_terminal))
}

# Set the pre-pruning parameter P = 25
P <- 25

decision_tree <- decisionTreeFunction(P)
node_splits <- decision_tree$splits
node_means <- decision_tree$means
is_terminal <- decision_tree$is_terminal

# predict function for test data points
predict <- function(dataPoint, is_terminal, node_splits, node_means){
  index <- 1
  while (1) {
    if (is_terminal[index] == TRUE) {
      return(node_means[index])
    } 
    else {
      if (dataPoint <= node_splits[index]) {
        index <- index * 2
      } 
      else {
        index <- index * 2 + 1
      }
    }
  }
}

point_colors <- c("blue", "red")
z <- c(rep(1, N_train), rep(2, N_test))

plot(x, y, type = "p", pch = 19, col = point_colors[z],
     ylim = c(minimum_y, maximum_y), xlim = c(minimum_x, maximum_x),
     ylab = "Waiting time to next eruption (min)", 
     xlab = "Eruption time (min)", 
     las = 1, main = "P = 25")

legend("topleft", legend = c("training", "test"),
       col = point_colors, pch = 19, cex = 0.75)

# Draw the fit on the figure
left_borders <- seq(from = 0, to = 59.9, by = 0.1)
right_borders <- seq(from = 0.1, to = 60, by = 0.1)

for (b in 1:length(left_borders)) {
  lines(c(left_borders[b], right_borders[b]),
        c(predict(left_borders[b], is_terminal, node_splits, node_means), predict(left_borders[b], is_terminal, node_splits, node_means)),
        col = "black",
        lwd=2, pch=19)
  if (b < length(left_borders)) {
    lines(c(right_borders[b], right_borders[b]),
          c(predict(left_borders[b], is_terminal, node_splits, node_means), predict(right_borders[b], is_terminal, node_splits, node_means)),
          col = "black",
          lwd=2, pch=19)
  }
}


# Calculate and print RMSE for the test data
y_predicted <- rep(0, N_test)

for (i in 1:N_test) {
  y_predicted[i] <- predict(x_test[i], is_terminal, node_splits, node_means)
}

RMSE <- sqrt(sum((y_test - y_predicted) ^ 2) / length(y_test))
sprintf("RMSE is %.4f when P is %s", RMSE, P)

# Learn decision trees for P = 1 to 20
RMSE_per_P <- rep(0, 10)
for (p in seq(from = 5, to = 50, by = 5)) {
  decision_tree <- decisionTreeFunction(p)
  node_splits <- decision_tree$splits
  node_means <- decision_tree$means
  is_terminal <- decision_tree$is_terminal
  y_predicted <- rep(0, N_test)
  for (i in 1:N_test) {
    y_predicted[i] <- predict(x_test[i], is_terminal, node_splits, node_means)
  }
  RMSE_per_P[p / 5] <- sqrt(sum((y_test - y_predicted) ^ 2) / length(y_test))
}

# Plot P vs RMSE graph
plot(seq(from = 5, to = 50, by = 5), RMSE_per_P, type = "b", las = 1, pch = 1, lty = 2,
     ylim = c(6, 8),
     xlab = "Pre???pruning size (P)", ylab = "RMSE")

lines(seq(from = 5, to = 50, by = 5), RMSE_per_P, type="b", col="black", lwd=2, pch=19)