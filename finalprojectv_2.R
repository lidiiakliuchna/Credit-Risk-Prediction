# Install and load necessary libraries
install.packages("caret")  
install.packages("rpart")  
install.packages("randomForest")
install.packages("devtools")
devtools::install_github("gbm-developers/gbm3")
install.packages("rpart.plot")
install.packages("pROC")
install.packages("ROCR")

library(caret)
train_data <- read.csv("creditdefault_train.csv")
test_data <- read.csv("creditdefault_test.csv")

library(readr)
library(dplyr)
library(tidyr)
library(ggplot2)

# Structure of the dataset
str(train_data)

# Summary of the dataset
summary(train_data)

# Visualise the distribution of the target variable
ggplot(train_data, aes(x = factor(Y))) +
  geom_bar() +
  labs(title = "Distribution of Credit Card Default",
       x = "Credit Card Default (Y)",
       y = "Count")


# Handle missing values
train_data <- na.omit(train_data)


# Load the 'rpart' library for decision trees
library(rpart)

# Create a Decision Tree model
model_decision_tree <- rpart(Y ~ ., data = train_data, method = "class")


# Make predictions on the training set
predictions_decision_tree <- predict(model_decision_tree, newdata = train_data, type = "class")

# Evaluate the Decision Tree model
confusion_matrix_decision_tree <- table(predictions_decision_tree, train_data$Y)
print(confusion_matrix_decision_tree)

# Visualise the Decision Tree
rpart.plot(model_decision_tree)

library(randomForest)      

# Create a Bagging model
model_bagging <- randomForest(Y ~ ., data = train_data, ntree = 500, mtry = ncol(train_data) - 1, importance = TRUE)
# Make predictions on the training set
predictions_bagging <- predict(model_bagging, newdata = train_data)
# Evaluate the Bagging model
confusion_matrix_bagging <- table(predictions_bagging, train_data$Y)
print(confusion_matrix_bagging)
plot(model_bagging$forest[[1]])
summary(model_bagging)

# Create a Random Forest model
model_random_forest <- randomForest(Y ~ ., data = train_data, ntree = 500, importance = TRUE)
# Make predictions on the training set
predictions_random_forest <- predict(model_random_forest, newdata = train_data)
# Evaluate the Random Forest model
confusion_matrix_random_forest <- table(predictions_random_forest, train_data$Y)
print(confusion_matrix_random_forest)
plot(model_random_forest$forest[[1]])

library(gbm3)

# Define the response variable
response_var <- "Y"

# Create a Gradient Boosting model
model_gradient_boosting <- gbm(
  formula = as.formula(paste(response_var, "~ .")),
  data = train_data,
  distribution = "bernoulli",
  n.trees = 500,
  interaction.depth = 4,
  shrinkage = 0.01,
  cv.folds = 5,
  verbose = TRUE
)

# Make predictions on the training set
predictions_gradient_boosting <- predict(
  model_gradient_boosting,
  newdata = train_data,
  n.trees = 500,
  type = "response"
)

# Convert probabilities to binary predictions
threshold <- 0.5
binary_predictions <- as.numeric(predictions_gradient_boosting > threshold)

# Evaluate the Gradient Boosting model
confusion_matrix_gradient_boosting <- table(binary_predictions, train_data$Y)
print(confusion_matrix_gradient_boosting)

# Display the summary
summary(model_gradient_boosting)

# Compute classification metrics
compute_metrics <- function(conf_matrix) {
  accuracy <- sum(diag(conf_matrix)) / sum(conf_matrix)
  
 
  if (sum(conf_matrix[, 2]) == 0) {
    precision <- 0
  } else {
    precision <- conf_matrix[2, 2] / sum(conf_matrix[, 2])
  }
  
  if (sum(conf_matrix[2, ]) == 0) {
    recall <- 0
  } else {
    recall <- conf_matrix[2, 2] / sum(conf_matrix[2, ])
  }
  
  # Calculate F1 score
  if (precision + recall == 0) {
    f1_score <- 0
  } else {
    f1_score <- 2 * (precision * recall) / (precision + recall)
  }
  
  return(c(Accuracy = accuracy, Precision = precision, Recall = recall, F1_Score = f1_score))
}

# Compute metrics for each model
metrics_decision_tree <- compute_metrics(confusion_matrix_decision_tree)
metrics_bagging <- compute_metrics(confusion_matrix_bagging)
metrics_random_forest <- compute_metrics(confusion_matrix_random_forest)
metrics_gradient_boosting <- compute_metrics(confusion_matrix_gradient_boosting)

# Combine metrics into a data frame
comparison_metrics <- data.frame(
  Decision_Tree = metrics_decision_tree,
  Bagging = metrics_bagging,
  Random_Forest = metrics_random_forest,
  Gradient_Boosting = metrics_gradient_boosting
)

print(comparison_metrics)


# Extract non-missing accuracy values
non_missing_accuracy <- comparison_metrics["Accuracy", !is.na(comparison_metrics["Decision_Tree"])]

# Best Model Selection
best_model_index <- which.max(non_missing_accuracy)
best_model <- names(non_missing_accuracy)[best_model_index]
best_model_accuracy <- as.numeric(non_missing_accuracy[best_model_index])

cat("Best Model:", best_model, "\n")
cat("Best Model Accuracy:", best_model_accuracy, "\n")


library(pROC)

# Create ROC curves for each model
roc_decision_tree <- roc(test_data$Y, as.numeric(predictions_decision_tree))
roc_bagging <- roc(test_data$Y, as.numeric(predictions_bagging))
roc_random_forest <- roc(test_data$Y, as.numeric(predictions_random_forest))
roc_gradient_boosting <- roc(test_data$Y, as.numeric(predictions_gradient_boosting))

# Plot ROC curves
plot(roc_decision_tree, col = "blue", main = "ROC Curves", col.main = "darkblue")
lines(roc_bagging, col = "red")
lines(roc_random_forest, col = "green")
lines(roc_gradient_boosting, col = "purple")

legend("bottomright", legend = c("Decision Tree", "Bagging", "Random Forest", "Gradient Boosting"),
       col = c("blue", "red", "green", "purple"), lty = 1)

# Calculate AUC
auc_decision_tree <- auc(roc_decision_tree)
auc_bagging <- auc(roc_bagging)
auc_random_forest <- auc(roc_random_forest)
auc_gradient_boosting <- auc(roc_gradient_boosting)

cat("AUC Decision Tree:", auc_decision_tree, "\n")
cat("AUC Bagging:", auc_bagging, "\n")
cat("AUC Random Forest:", auc_random_forest, "\n")
cat("AUC Gradient Boosting:", auc_gradient_boosting, "\n")


