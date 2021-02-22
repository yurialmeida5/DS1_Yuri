###### Load Libraries ############ 

# Clear environment
rm(list = ls())

# Library 

library(tidyverse)
library(datasets)
library(MASS)
library(ISLR)
library(caret)
library(ggplot2)
library(mctest)
library(ppcor)

# Load the necessary data

data <- readRDS(url('http://www.jaredlander.com/data/manhattan_Train.rds')) %>%
  mutate(logTotalValue = log(TotalValue)) %>%
  drop_na()

# Remove ID Column (Shouldn't be used in the Regression models)
data$ID <- NULL

# a) Do a short exploration of data and find possible predictors of the target variable.

# Explore the data 
skim(data)

# 6 factor variable types, 1 Boolean and another 29 numeric variables. 
# Non-missing values under the following variables, this is really good (specially for predictions and regressions).

# Predicted variable: Total Value
# Check variable distribution 

ggplot(data, aes(TotalValue)) +
  geom_histogram(alpha = 0.8) +
  ylab("count") +
  xlab("TotalValue") +
  theme_classic()

# Right Tailed Skewed Distribution with a considerable amount of outliers

ggplot(data, aes(logTotalValue)) +
  geom_histogram(alpha = 0.8) +
  ylab("count") +
  xlab("ln(TotalValue)") +
  theme_classic()

# Way closer to the normal distribution, the log transformation promotes a better fit for the regression.

# Remove all the non-used variables used in the regression 
explanatory_var <- colnames(data)[
  which(!(colnames(data) %in% c("ID","TotalValue", "logTotalValue")))]

# Check the general linear regression results
outcome_name <- "logTotalValue"
formula_str <- paste(outcome_name, "~", paste(explanatory_var, collapse = " + "))

gen_model <- lm(formula_str, data = data)

# General Model Diagnostic 
par(mfrow=c(2,2))
plot(gen_model)

# Residuals vs Fitted / Scale - Location: Variance is more-less constant, but the graphs indicates the presence of outliers
# Normal Q-Q:  tails are observed to be ‘heavier’ (have larger values)

# Multicolinearity Check
ggcorr(data)

omc_res <- omcdiag(gen_model)
# Really low Determinant |X'X| and high Farrar Chi-Squared, high possibility of multicollinearity. 

# b) Create a training and a test set, assigning 30% of observations to the training set.

set.seed(1234)
training_ratio <- 0.3
train_indices <- createDataPartition(
  y = data[["logTotalValue"]],
  times = 1,
  p = training_ratio,
  list = FALSE
) %>% as.vector()
data_train <- data[train_indices, ]
data_test <- data[-train_indices, ]

fit_control <- trainControl(method = "cv", number = 10)

# c) Use a linear regression to predict logTotalValue and use 10-fold cross validation to assess the predictive power.

# OLS

set.seed(1234)
linear_fit <- train(
  logTotalValue ~ . -TotalValue,
  data = data_train,
  method = "lm",
  preProcess = c("center", "scale"),
  trControl = fit_control
)

# d) Use penalized linear models for the same task.
# Make sure to try LASSO, Ridge and Elastic Net models. Does the best model improve on the simple linear model?

# In statistics, there are two critical characteristics of estimators to be considered: the bias and the variance. 
# The bias is the difference between the true population parameter and the expected estimator
# Variance, on the other hand, measures the spread, or uncertainty, in these estimates.

# The OLS estimator has the desired property of being unbiased. 
# However, it can have a huge variance. Specifically, this happens when:
  
# The predictor variables are highly correlated with each other (Multicolinearity)
# There are many predictors. This is reflected in the formula for variance given above: if m approaches n, the variance approaches infinity.


# LASSO Model
tenpowers <- 10^seq(-1, -5, by = -1)

lasso_tune_grid <- expand.grid(
  "alpha" = c(1),
  "lambda" = c(tenpowers, tenpowers / 2) 
)

set.seed(1234)
lasso_fit <- train(
  logTotalValue ~ . -TotalValue,
  data = data_train,
  method = "glmnet",
  preProcess = c("center", "scale"),
  tuneGrid = lasso_tune_grid,
  trControl = trainControl(method = "cv", number = 10)
)

# Lasso regression is a type of linear regression that uses shrinkage (where data values are shrunk towards a central point)  
# Well-suited for models showing high levels of muticollinearity (variable selection/parameter elimination).
# Coefficients are forced to be zero.

# Ridge model
ridge_tune_grid <- expand.grid(
  "alpha" = c(0),
  "lambda" = seq(0.05, 0.5, by = 0.025)
)

set.seed(1234)
ridge_fit <- train(
  logTotalValue ~ . -TotalValue,
  data = data_train,
  method = "glmnet",
  preProcess = c("center", "scale"),
  tuneGrid = ridge_tune_grid,
  trControl = trainControl(method = "cv", number = 10)
)

# Penalize the variables if they are too far from zero, thus enforcing them to be small in a continuous way
# Decreases model complexity while keeping all variables in the model. 
# Assumed that the predictors are standardized and the response is centered.

# Elastic Net Model

enet_tune_grid <- expand.grid(
  "alpha" = seq(0, 1, by = 0.1),
  "lambda" = union(lasso_tune_grid[["lambda"]], ridge_tune_grid[["lambda"]])
)

set.seed(1234)
enet_fit <- train(
  logTotalValue ~ . -TotalValue,
  data = data_train,
  method = "glmnet",
  preProcess = c("center", "scale"),
  tuneGrid = enet_tune_grid,
  trControl = trainControl(method = "cv", number = 10)
)

# Combine the penalties of ridge regression and lasso to get the best of both.

# Evaluate the models

resample_profile <- resamples(
  list("linear" = linear_fit,
       "ridge" = ridge_fit,
       "lasso" = lasso_fit,
       "elastic net" = enet_fit
  )
) 

summary(resample_profile)

# Yes, as expected the all the models lowered the RMSE without reducing much the R2 Squared, 
# confirming some multicollinearities found on the general (linear model).
# Although you may see a slightly better performance under the elastic net model, 
# LASSO and Ridge also showed closer results.

# e) Which of the models you’ve trained is the “simplest one that is still good enough”? 
# What is its effect? 

# "one standard error"  suggest that the tuning parameter associated with the best performance may over fit. 
# They suggest that the simplest model within one standard error of the empirically optimal model is the better choice. 
# This assumes that the models can be easily ordered from simplest to most complex.

# Adding the one Standard Error Selection Function, the results didn't show much difference from the past analysis (letter d).
# Ridge, Lasso, and Elastic Net also lowered the RMSE presented in the linear model, reducing the multicollinearity between variables.
# Although the results are similar, in this case, Ridge and Elastic showed to be the best models.
# with a preference for Ridge for being simpler.

set.seed(1234)
lasso_fit <- train(
  logTotalValue ~ . -TotalValue,
  data = data_train,
  method = "glmnet",
  preProcess = c("center", "scale"),
  tuneGrid = lasso_tune_grid,
  trControl = trainControl(method = "cv", number = 10, selectionFunction = "oneSE")
)

set.seed(1234)
ridge_fit <- train(
  logTotalValue ~ . -TotalValue,
  data = data_train,
  method = "glmnet",
  preProcess = c("center", "scale"),
  tuneGrid = ridge_tune_grid,
  trControl = trainControl(method = "cv", number = 10, selectionFunction = "oneSE")
)

set.seed(1234)
enet_fit <- train(
  logTotalValue ~ . -TotalValue,
  data = data_train,
  method = "glmnet",
  preProcess = c("center", "scale"),
  tuneGrid = enet_tune_grid,
  trControl = trainControl(method = "cv", number = 10, selectionFunction = "oneSE")
)

resample_profile_2 <- resamples(
  list("linear" = linear_fit,
       "ridge" = ridge_fit,
       "lasso" = lasso_fit,
       "elastic net" = enet_fit
  )
) 

summary(resample_profile)
summary(resample_profile_2)

# f) Does PCA improve the fit over the simple linear model?

# PCR can aptly deal with such situations by excluding some of the low-variance principal components in the regression step. 
# In addition, by usually regressing on only a subset of all the principal components, PCR can result in dimension reduction through substantially lowering the effective number of parameters characterizing the underlying model. 
# This can be particularly useful in settings with high-dimensional covariates.

# Center and scale your variables and use pcr to conduct a search for the optimal number of principal components. 

tune_grid <- data.frame(ncomp = 1:118)
set.seed(1234)
pcr_fit <- train(
  logTotalValue ~ . -TotalValue,
  data = data,
  method = "pcr",
  trControl = trainControl(method = "cv", number = 10),
  tuneGrid = tune_grid,
  preProcess = c("center", "scale")
)

set.seed(1234)
linear_fit <- train(
  logTotalValue ~ . -TotalValue,
  data = data_train,
  method = "lm",
  preProcess = c("center", "scale"),
  trControl = trainControl(method = "cv", number = 10)
)

set.seed(1234)
lasso_fit <- train(
  logTotalValue ~ . -TotalValue,
  data = data_train,
  method = "glmnet",
  preProcess = c("center", "scale"),
  tuneGrid = lasso_tune_grid,
  trControl = trainControl(method = "cv", number = 10)
)

set.seed(1234)
ridge_fit <- train(
  logTotalValue ~ . -TotalValue,
  data = data_train,
  method = "glmnet",
  preProcess = c("center", "scale"),
  tuneGrid = ridge_tune_grid,
  trControl = trainControl(method = "cv", number = 10)
)

set.seed(1234)
enet_fit <- train(
  logTotalValue ~ . -TotalValue,
  data = data_train,
  method = "glmnet",
  preProcess = c("center", "scale"),
  tuneGrid = enet_tune_grid,
  trControl = trainControl(method = "cv", number = 10)
)

resample_profile_3 <- resamples(
  list("linear" = linear_fit,
       "ridge" = ridge_fit,
       "lasso" = lasso_fit,
       "elastic net" = enet_fit,
       "pcr" = pcr_fit
  )
) 

summary(resample_profile_3)

# Yes, by excluding some of the low-variance principal components in the regression step,
# the PCR method not only improved the fit (R-Squared), but also reduced the RMSE.


# g) If you apply PCA prior to estimating penalized models via preProcess, does it help to achieve a better fit?

# The preProcess function estimates whatever it requires (in terms of parameters) from a specific data set (e.g. the training set)
# and then applies these transformations to any data set without recomputing the values

set.seed(1234)
linear_fit <- train(
  logTotalValue ~ . -TotalValue,
  data = data_train,
  method = "lm",
  preProcess = c("center", "scale", "pca" , "nzv"),
  trControl = trainControl(method = "cv", number = 10)
)

set.seed(1234)
lasso_fit <- train(
  logTotalValue ~ . -TotalValue,
  data = data_train,
  method = "glmnet",
  preProcess = c("center", "scale", "pca" , "nzv"),
  tuneGrid = lasso_tune_grid,
  trControl = trainControl(method = "cv", number = 10)
)

set.seed(1234)
ridge_fit <- train(
  logTotalValue ~ . -TotalValue,
  data = data_train,
  method = "glmnet",
  preProcess = c("center", "scale", "pca" , "nzv"),
  tuneGrid = ridge_tune_grid,
  trControl = trainControl(method = "cv", number = 10)
)

set.seed(1234)
enet_fit <- train(
  logTotalValue ~ . -TotalValue,
  data = data_train,
  method = "glmnet",
  preProcess = c("center", "scale", "pca", "nzv"),
  tuneGrid = enet_tune_grid,
  trControl = trainControl(method = "cv", number = 10)
)


resample_profile_final <- resamples(
  list("linear" = lm_fit_pca,
       "ridge" = ridge_fit,
       "lasso" = lasso_fit,
       "elastic net" = enet_fit,
       "pcr" = pcr_fit
  )
) 

summary(resample_profile_3)
summary(resample_profile_final)

# Yes, it's possible to actually visualize a reduction on RMSE in all models (Linear, LASSO, Ridge and Elastic).
# Intuitively, it's possible to say that this reduction was already expected. 
# Using "PCA" (retrieving only the main coefficients that explain 95% of the variance) and "nvz" (excluding "near-zero-variance predictors),
# there is already a reduction the number of coefficients to be penalized by the models, lowering the RMSE in all cases.
# However, using the "pcr" method directly, it was obtained the best model with the smallest errors (RMSE) and best fit (higher R-Squared).


# h) Select the best model of those you’ve trained. Evaluate your preferred model on the test set.

predicted_val <- predict(pcr_fit , newdata = data_test , "raw")
data_test$predicted_vals <- predicted_val
pcr_RMSE_test <- RMSE(data_test$predicted_vals, data_test$logTotalValue)

data_test %>%
  ggplot() +
  geom_point(aes(x = predicted_vals, y = logTotalValue)) +
  geom_line( aes( x = logTotalValue , y = logTotalValue ) , color = "red", size = 1.2) +
  labs( x = "Predicted values", y = "Actual values")


# The predictions for the model showed a really great fit. 
# Based on the graph it's possible to check the errors mostly follow regular pattern,
# their deviation from the actual values remain mostly stable during the whole set of values. 
# The RMSE distribution is closer to normal, with some heavy tails due to outliers.





