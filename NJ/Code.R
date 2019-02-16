require(dummies)
require(caret)
require(keras)
require(MASS)
require(statmod)
require(tweedie)
require(dplyr)
require(keras)
require(insuranceData)
require(xgboost)
require(plotly)
require(iml)
install_keras()

data("dataCar")
dat <- dataCar
rm(dataCar)
dat <- dat[, -c(3, 11)]

set.seed(12457)
ind <- sample(1:nrow(dat), round(0.3*nrow(dat)))
train <- dat[-ind, ]
test <- dat[ind, ]
rm(ind)

rmse <- function(data, targetvar, prediction.obj) {
  rss <- (as.numeric(prediction.obj) - as.numeric(data[, c(targetvar)])) ^ 2
  mse <- sum(rss) / nrow(data)
  rmse <- sqrt(mse)
  return(rmse/100)
}

r_sq <- function(prediction.obj, actuals.vec) {
  cor <- cor(as.numeric(prediction.obj), as.numeric(actuals.vec))
  r_sq <- cor ^ 2
  return(r_sq * 100)
}

##======BASELINE GLM=======##
targetvar <- c("claimcst0")
predictors <- c("veh_value", "veh_body", "veh_age", "gender", "area", "agecat")
predictors <- paste(predictors, collapse = "+")
formula <- as.formula(paste(targetvar,"~",predictors, collapse = "+"))

p_values <- seq(1.1, 1.7, by = 0.1)
p_tuning <- tweedie.profile(formula = formula, p.vec = p_values, data = train, do.plot = FALSE, offset = log(exposure))
p <- p_tuning$p.max
rm(p_values)
rm(p_tuning)

glm <- glm(formula = formula, data = train, family = tweedie(var.power = p, link.power = 0), offset = log(exposure))
rm(formula)
rm(predictors)
rm(p)

predictions_glm <- predict(glm, test, type = "response")

rmse_glm <- rmse(data = test, targetvar = targetvar, prediction.obj = predictions_glm)
rmse_glm

r_sq_glm <- r_sq(prediction.obj = predictions_glm, actuals.vec = test[, c(targetvar)])
r_sq_glm

rm(targetvar)

##======AI/XGB DATA PREP==========##
targetvar <- c("claimcst0")
predictors <- c("veh_value", "veh_body", "veh_age", "gender", "area", "agecat", "exposure")

factors <- c()
for (i in 1:length(predictors)) {
  if (class(dat[, predictors[i]]) == "factor") {
    factors[i] <- predictors[i]
  } else {
    i <- i + 1
  }
}

factors <- factors[!factors %in% NA]
rm(i)

dat_nn <- dummy.data.frame(dat, names = factors)
dat_nn <- as.data.frame(lapply(dat_nn, as.numeric))
train_nn <- dummy.data.frame(train, names = factors)
train_nn <- as.data.frame(lapply(train_nn, as.numeric))
test_nn <- dummy.data.frame(test, names = factors)
test_nn <- as.data.frame(lapply(test_nn, as.numeric))
rm(factors)

allvars <- colnames(dat_nn)
predictors_nn <- allvars[!allvars %in% c(targetvar, "numclaims")]
rm(allvars)

max <- apply(dat_nn, 2, max)
min <- apply(dat_nn, 2, min)

train_nn <- as.data.frame(scale(train_nn, center = min, scale = (max - min)))
test_nn <- as.data.frame(scale(test_nn, center = min, scale = (max - min)))

rm(min)
rm(max)

x_train <- as.matrix(train_nn[predictors_nn])
y_train <- as.matrix(train_nn[targetvar])

x_test <- as.matrix(test_nn[predictors_nn])
y_test <- as.matrix(test_nn[targetvar])

rm(predictors_nn)

##========XGBOOST=========##
xgboost <- xgboost(data = x_train, label = as.numeric(y_train), objective = "reg:linear", nrounds = 600, verbose = 1, max.depth = 3, eta = 0.01, early_stopping_rounds = 50)
predictions_xgb <- predict(xgboost, x_test)
predictions_xgb <- (predictions_xgb * (max(dat_nn[targetvar]) - min(dat_nn[targetvar]))) + min(dat_nn[targetvar])
rmse_xgb <- rmse(data = test, targetvar = targetvar, prediction.obj = predictions_xgb)
rmse_xgb
r_sq_xgb <- r_sq(prediction.obj = predictions_xgb, actuals.vec = test[, c(targetvar)])
r_sq_xgb

##=======AI-1============##
mlp1 <- keras_model_sequential()
mlp1 %>%
  layer_dense(units = 300, activation = "tanh", input_shape = ncol(x_train)) %>%
  layer_dense(units = 150, activation = "tanh") %>%
  layer_dropout(0.2) %>%
  layer_dense(units = 1, activation = "tanh")
mlp1 %>%
  compile(loss = "mean_squared_error", optimizer = optimizer_rmsprop(lr = 0.0001, rho = 0.9))
fit <- fit(mlp1, x = x_train, y = y_train, batch_size = 8000, epochs = 3000, verbose = 1, callbacks = callback_early_stopping(monitor = "loss", min_delta = 0.00001, patience = 20, verbose = 1, mode = "min", restore_best_weights = TRUE))
preds <- mlp1 %>% predict(x_test)
predictions_mlp1 <- (preds * (max(dat_nn[targetvar]) - min(dat_nn[targetvar]))) + min(dat_nn[targetvar])
rm(preds)
rmse_mlp1 <- rmse(data = test, targetvar = targetvar, prediction.obj = predictions_mlp1)
rmse_mlp1
r_sq_mlp1 <- r_sq(prediction.obj = predictions_mlp1, actuals.vec = test[, c(targetvar)])
r_sq_mlp1

##=======AI-2============##
mlp2 <- keras_model_sequential()
mlp2 %>%
  layer_dense(units = 250, activation = "tanh", input_shape = ncol(x_train)) %>%
  layer_dense(units = 150, activation = "tanh") %>%
  layer_dropout(0.2) %>%
  layer_dense(100, activation = "tanh") %>%
  layer_dropout(0.1) %>%
  layer_dense(units = 1, activation = "tanh")
mlp2 %>%
  compile(loss = "mean_squared_error", optimizer = optimizer_rmsprop(lr = 0.0001, rho = 0.9))
fit <- fit(mlp2, x = x_train, y = y_train, batch_size = 8000, epochs = 3000, verbose = 1, callbacks = list(callback_early_stopping(monitor = "loss", min_delta = 0.00001, patience = 50, verbose = 1, mode = "min"), callback_tensorboard(log_dir = "~/Desktop/ML Experiment/Tensorboard")))
preds <- mlp2 %>% predict(x_test)
predictions_mlp2 <- (preds * (max(dat_nn[targetvar]) - min(dat_nn[targetvar]))) + min(dat_nn[targetvar])
rm(preds)
rmse_mlp2 <- rmse(data = test, targetvar = targetvar, prediction.obj = predictions_mlp2)
rmse_mlp2
r_sq_mlp2 <- r_sq(prediction.obj = predictions_mlp2, actuals.vec = test[, c(targetvar)])
r_sq_mlp2

##=======AI-3============##
mlp3 <- keras_model_sequential()
mlp3 %>%
  layer_dense(units = 450, activation = "tanh", input_shape = ncol(x_train)) %>%
  layer_dense(units = 400, activation = "tanh") %>%
  layer_dropout(0.4) %>%
  layer_dense(200, activation = "tanh") %>%
  layer_dropout(0.25) %>%
  layer_dense(units = 1, activation = "tanh")
mlp3 %>%
  compile(loss = "mean_squared_error", optimizer = optimizer_rmsprop(lr = 0.0001, rho = 0.9))
fit <- fit(mlp3, x = x_train, y = y_train, batch_size = 8000, epochs = 3000, verbose = 1, callbacks = callback_early_stopping(monitor = "loss", min_delta = 0.00001, patience = 50, verbose = 1, mode = "min"))
preds <- mlp3 %>% predict(x_test)
predictions_mlp3 <- (preds * (max(dat_nn[targetvar]) - min(dat_nn[targetvar]))) + min(dat_nn[targetvar])
rm(preds)
rmse_mlp3 <- rmse(data = test, targetvar = targetvar, prediction.obj = predictions_mlp3)
rmse_mlp3
r_sq_mlp3 <- r_sq(prediction.obj = predictions_mlp3, actuals.vec = test[, c(targetvar)])
r_sq_mlp3

##=======BRUTE-FORCE AGGREGATION===========##
predictions_bfa <- (predictions_mlp2 + predictions_mlp3) / 2
rmse_bfa <- rmse(data = test, targetvar = targetvar, prediction.obj = predictions_bfa)
rmse_bfa
r_sq_bfa <- r_sq(prediction.obj = predictions_bfa, actuals.vec = test[, c(targetvar)])
r_sq_bfa

##======SOFTMAX AGGREGATION=========##
predictions_sma <- ((exp(rmse_mlp2)/sum(exp(rmse_mlp2), exp(rmse_mlp3))) * predictions_mlp2) +
  ((exp(rmse_mlp3)/sum(exp(rmse_mlp2), exp(rmse_mlp3))) * predictions_mlp3)
rmse_sma <- rmse(data = test, targetvar = targetvar, prediction.obj = predictions_sma)
rmse_sma
r_sq_sma <- r_sq(prediction.obj = predictions_sma, actuals.vec = test[, c(targetvar)])
r_sq_sma

##======TEST OUTPUT & ANALYTICS=========##
test$GLM <- predictions_glm
test$XGBOOST <- predictions_xgb
test$MLP1 <- as.numeric(predictions_mlp1)
test$MLP2 <- as.numeric(predictions_mlp2)
test$MLP3 <- as.numeric(predictions_mlp3)
test$BFA <- as.numeric(predictions_bfa)

#Tensorboard
tensorboard("~/Desktop/ML Experiment/Tensorboard")

#Analysis of Predictions
grp_by_veh_body <- test %>%
  group_by(veh_body) %>%
  summarize(avg_rate_bfa = mean(BFA), avg_rate_glm = mean(GLM))

plot_veh_body <- plot_ly(data = grp_by_veh_body, x = ~veh_body, y = ~avg_rate_bfa, type = "bar") %>%
  add_trace(y = ~avg_rate_glm)
plot_veh_body


