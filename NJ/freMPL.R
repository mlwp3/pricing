library(dummies)
library(ROSE)
library(caret)
library(MASS)
library(tweedie)
library(statmod)
require(xgboost)
library(dplyr)
library(plotly)
library(keras)
library(iml)
library(foreach)
library(doParallel)
library(data.table)
library(ranger)
library(vip)
install_keras()

dat <- read.csv("~/Desktop/CAS MLWP/Pricing/freMPL.csv")

matrix(colnames(dat), ncol = 1)

dat <- dat[, c(2, 3, seq(6, 8, by = 1), seq(10, 20, by = 1), 22, 23)]
colnames(dat)[which(colnames(dat) %in% c("Exposure"))] <- c("exposure")

dat <- dat[dat$exposure > 0, ]
dat <- dat[dat$exposure <= 1, ]
dat <- dat[dat$ClaimAmount >= 0, ]

#Band LicAge into Years
dat$LicAge <- cut(dat$LicAge, breaks = c(0, 365, 730, 1095), right = FALSE)
dat %>%
  group_by(LicAge) %>%
  summarise(exposure = sum(exposure)) #Hardly any exposure in the 2+ years category, so dropping this
dat$LicAge <- as.character(dat$LicAge)
dat <- dat[-which(as.character(dat$LicAge) %in% c("[730,1.1e+03)")), ]
dat$LicAge <- ifelse(dat$LicAge == c("[0,365)"), c("0 - 1"), c("1 - 2")) #Converting LicAge from Days to Years
dat$LicAge <- as.factor(dat$LicAge)

#Convert HasKmLimit to a factor
dat$HasKmLimit <- as.factor(dat$HasKmLimit)

#Band Bonus-Malus according to the schema in the data description
dat$BonusMalus <- ifelse(dat$BonusMalus < 100, "Bonus", "Malus")
dat$BonusMalus <- as.factor(dat$BonusMalus)

#Band DrivAge
dat$DrivAge <- cut(dat$DrivAge, breaks = c(18, 25, 35, 45, 55, 65, 110), right = FALSE)
dat %>%
  group_by(DrivAge) %>%
  summarise(exposure = sum(exposure))
dat$DrivAge <- as.character(dat$DrivAge)
dat$DrivAge <- ifelse(dat$DrivAge == c("[65,110)"), c("[65+)"), dat$DrivAge)
dat$DrivAge <- as.factor(dat$DrivAge)

#Look at Vehicle Price - it has too many levels so might be a good idea to condense
veh_price_grp <- dat %>%
  group_by(VehPrice) %>%
  summarise(burning_cost = sum(ClaimAmount) / sum(exposure))
#Above table shows good reason to club vehicle price groups based on burning cost - trying k-means clustering here
set.seed(34)
veh_price_cluster <- kmeans(veh_price_grp[, c(2)], 5, nstart = 20)
veh_price_grp$VehPriceGrp <- as.factor(veh_price_cluster$cluster)
#Map VehPrice to VehPriceGrp in the raw data
dat$VehPriceGrp <- as.factor(unlist(veh_price_grp[match(dat$VehPrice, veh_price_grp$VehPrice), c("VehPriceGrp")]))
dat <- dat[, -which(colnames(dat) == c("VehPrice"))] #Remove VehPrice from the data since we model VehPriceGrp
rm(veh_price_cluster)
rm(veh_price_grp)

#Looking at VehEngine and VehEnergy
veh_engine_grp <- dat %>%
  group_by(VehEngine) %>%
  summarise(exposure = sum(exposure), row_count = n(), claim = sum(ClaimAmount)) #Only 1 row for GPL with no claim and very low exposure, safe to remove
dat <- dat[!dat$VehEngine == c("GPL"), ]
rm(veh_engine_grp)

veh_energy_grp <- dat %>%
  group_by(VehEnergy) %>%
  summarise(exposure = sum(exposure), row_count = n(), claim = sum(ClaimAmount)) #Not too many electric vehicles, keep for now
rm(veh_energy_grp)

#Split into Train and Test
set.seed(1245788)
ind <- sample(1:nrow(dat), round(0.3 * nrow(dat)))
train <- dat[-ind, ]
test <- dat[ind, ]
rm(ind)

#Define RMSE and MAE metrics for Model Evaluation
rmse <- function(data, targetvar, prediction.obj) {
  rss <- (as.numeric(prediction.obj) - as.numeric(data[, c(targetvar)])) ^ 2
  mse <- sum(rss) / nrow(data)
  rmse <- sqrt(mse)
  rmse <- rmse / 100
  return(rmse)
}

mae <- function(data, targetvar, prediction.obj) {
  err <- as.numeric(prediction.obj) - as.numeric(data[, c(targetvar)])
  abs_err <- abs(err)
  mae <- sum(abs_err) / nrow(data)
  mae <- mae / 1000
  return(mae)
}

##========BASELINE GLM==========##
targetvar <- c("ClaimAmount")
predictors <- c("LicAge", "VehAge", "Gender", "MariStat", "VehUsage", "DrivAge", "HasKmLimit", "BonusMalus", "VehBody", "VehPriceGrp", "VehEngine",
                "VehEnergy", "VehMaxSpeed", "VehClass", "Garage")
predictors <- paste(predictors, collapse = "+")
formula <- as.formula(paste(targetvar,"~",predictors,collapse = "+"))

p_values <- seq(1.1, 1.8, by = 0.1)
p_tuning <- tweedie.profile(formula = formula, data = train, p.vec = p_values, do.plot = FALSE, offset = log(exposure))
p <- p_tuning$p.max
rm(p_values)
rm(p_tuning)

glm <- glm(formula = formula, data = train, family = tweedie(var.power = p, link.power = 0), offset = log(exposure))
rm(formula)
rm(p)
rm(predictors)

predictions_glm <- predict(glm, test, type = "response")

rmse_glm <- rmse(data = test, targetvar = targetvar, prediction.obj = predictions_glm)
rmse_glm

rm(targetvar)

##========AI/XGB DATA PREP==========##
targetvar <- c("ClaimAmount")
predictors <- c("LicAge", "VehAge", "Gender", "MariStat", "VehUsage", "DrivAge", "HasKmLimit", "BonusMalus", "VehBody", "VehPriceGrp", "VehEngine",
                "VehEnergy", "VehMaxSpeed", "VehClass", "Garage", "exposure")

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

#Resample Training Data
table(train$ClaimInd) #Without Resampling
t <- c("ClaimInd")
p <- append(predictors, targetvar)
p <- paste(p, collapse = "+")
f <- as.formula(paste(t,"~",p,collapse = "+"))
train_bal <- ovun.sample(formula = f, data = train, method = "both", p = 0.1)$data
table(train_bal$ClaimInd) #After Resampling
rm(p)
rm(f)
train_bal <- train_bal[, -which(colnames(train_bal) == t)]

#One-Hot Encoding
dat_nn <- dummy.data.frame(dat, names = factors)
dat_nn <- as.data.frame(lapply(dat_nn, as.numeric))
train_nn <- dummy.data.frame(train_bal, names = factors)
train_nn <- as.data.frame(lapply(train_nn, as.numeric))
test_nn <- dummy.data.frame(test, names = factors)
test_nn <- as.data.frame(lapply(test_nn, as.numeric))
rm(factors)
rm(t)

allvars <- colnames(train_nn)
predictors_nn <- allvars[!allvars %in% c(targetvar)]
rm(allvars)

dat_nn <- dat_nn[, c(predictors_nn, targetvar)]
train_nn <- train_nn[, c(predictors_nn, targetvar)]
test_nn <- test_nn[, c(predictors_nn, targetvar)]

max <- as.numeric(apply(dat_nn, 2, max))
min <- as.numeric(apply(dat_nn, 2, min))

train_nn <- as.data.frame(scale(train_nn, center = min, scale = (max - min)))
test_nn <- as.data.frame(scale(test_nn, center = min, scale = (max - min)))

rm(max)
rm(min)

x_train <- as.matrix(train_nn[predictors_nn])
y_train <- as.matrix(train_nn[targetvar])

x_test <- as.matrix(test_nn[predictors_nn])
y_test <- as.matrix(test_nn[targetvar])

rm(predictors_nn)

##=========XGBOOST==========##
xgboost <- xgboost(data = x_train[, -which(colnames(x_train) %in% c("exposure"))], label = as.numeric(y_train), objective = "reg:logistic", nrounds = 50000, verbose = 1, 
                   max.depth = 10, eta = 0.1, early_stopping_rounds = 1000, weights = as.numeric(x_train[, c("exposure")]))
predictions_xgb <- predict(xgboost, x_test[, -which(colnames(x_test) %in% c("exposure"))])
predictions_xgb <- (predictions_xgb * (max(dat_nn[targetvar]) - min(dat_nn[targetvar]))) + min(dat_nn[targetvar])
rmse_xgb <- rmse(data = test, targetvar = targetvar, prediction.obj = predictions_xgb)
rmse_xgb

##===========AI-1===========##
mlp1 <- keras_model_sequential()
mlp1 %>%
  layer_dense(units = 10, activation = "tanh", input_shape = ncol(x_train)) %>%
  layer_dense(units = 6, activation = "tanh") %>%
  layer_dense(units = 2, activation = "tanh") %>%
  layer_dense(units = 1, activation = "sigmoid")
mlp1 %>%
  compile(loss = "mean_squared_error", optimizer = optimizer_rmsprop(lr = 0.001, rho = 0.9))
fit <- fit(mlp1, x = x_train, y = y_train, batch_size = 10000, epochs = 20000, verbose = 1, 
           callbacks = list(callback_early_stopping(monitor = "loss", min_delta = 0.00001, patience = 1000, verbose = 1)))
predictions_mlp1 <- mlp1 %>% predict(x_test)
predictions_mlp1 <- (predictions_mlp1 * (max(dat_nn[targetvar]) - min(dat_nn[targetvar]))) + min(dat_nn[targetvar])
rmse_mlp1 <- rmse(data = test, targetvar = targetvar, prediction.obj = predictions_mlp1)
rmse_mlp1

##===========AI-2===========##
mlp2 <- keras_model_sequential()
mlp2 %>%
  layer_dense(units = 5, activation = "tanh", input_shape = ncol(x_train)) %>%
  layer_dense(units = 1, activation = "sigmoid")
mlp2 %>%
  compile(loss = "mean_squared_error", optimizer = optimizer_rmsprop(lr = 0.01, rho = 0.9))
fit <- fit(mlp2, x = x_train, y = y_train, batch_size = 10000, epochs = 20000, verbose = 1, 
           callbacks = list(callback_early_stopping(monitor = "loss", min_delta = 0.00001, patience = 1500, verbose = 1)))
predictions_mlp2 <- mlp2 %>% predict(x_test)
predictions_mlp2 <- (predictions_mlp2 * (max(dat_nn[targetvar]) - min(dat_nn[targetvar]))) + min(dat_nn[targetvar])
rmse_mlp2 <- rmse(data = test, targetvar = targetvar, prediction.obj = predictions_mlp2)
rmse_mlp2

##===========AI-3===========##
mlp3 <- keras_model_sequential()
mlp3 %>%
  layer_dense(units = 20, activation = "tanh", input_shape = ncol(x_train)) %>%
  layer_dense(units = 15, activation = "tanh") %>%
  layer_dense(units = 5, activation = "tanh") %>%
  layer_dense(units = 1, activation = "sigmoid")
mlp3 %>%
  compile(loss = "mean_squared_error", optimizer = optimizer_sgd(lr = 0.01, momentum = 0.9))
fit <- fit(mlp3, x = x_train, y = y_train, batch_size = 10000, epochs = 15000, verbose = 1, 
           callbacks = list(callback_early_stopping(monitor = "loss", min_delta = 0.0001, patience = 1000, verbose = 1)))
predictions_mlp3 <- mlp3 %>% predict(x_test)
predictions_mlp3 <- (predictions_mlp3 * (max(dat_nn[targetvar]) - min(dat_nn[targetvar]))) + min(dat_nn[targetvar])
rmse_mlp3 <- rmse(data = test, targetvar = targetvar, prediction.obj = predictions_mlp3)
rmse_mlp3

##===========AI-4===========##
mlp4 <- keras_model_sequential()
mlp4 %>%
  layer_dense(units = 4, activation = "tanh", input_shape = ncol(x_train)) %>%
  layer_dense(units = 2, activation = "tanh") %>%
  layer_dense(units = 1, activation = "sigmoid")
mlp4 %>%
  compile(loss = "mean_squared_error", optimizer = optimizer_sgd(lr = 0.01, momentum = 0.9))
fit <- fit(mlp4, x = x_train, y = y_train, batch_size = 10000, epochs = 15000, verbose = 1, 
           callbacks = list(callback_early_stopping(monitor = "loss", min_delta = 0.0001, patience = 1000, verbose = 1)))
predictions_mlp4 <- mlp4 %>% predict(x_test)
predictions_mlp4 <- (predictions_mlp4 * (max(dat_nn[targetvar]) - min(dat_nn[targetvar]))) + min(dat_nn[targetvar])
rmse_mlp4 <- rmse(data = test, targetvar = targetvar, prediction.obj = predictions_mlp4)
rmse_mlp4

##=====STACK APPROACH=======##
#Split Training dataset and Process
set.seed(124570)
ind <- sample(1:nrow(train), round(0.5 * nrow(train)))
train_stack <- train[-ind, ]
val_stack <- train[ind, ]
rm(ind)

targetvar <- c("ClaimAmount")
predictors <- c("LicAge", "VehAge", "Gender", "MariStat", "VehUsage", "DrivAge", "HasKmLimit", "BonusMalus", "VehBody", "VehPriceGrp", "VehEngine",
                "VehEnergy", "VehMaxSpeed", "VehClass", "Garage", "exposure")

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

#Resample Training Data
table(train_stack$ClaimInd) #Without Resampling
t <- c("ClaimInd")
p <- append(predictors, targetvar)
p <- paste(p, collapse = "+")
f <- as.formula(paste(t,"~",p,collapse = "+"))
train_stack_bal <- ovun.sample(formula = f, data = train_stack, method = "both", p = 0.125)$data
table(train_stack_bal$ClaimInd) #After Resampling
rm(p)
rm(f)
train_stack_bal <- train_stack_bal[, -which(colnames(train_stack_bal) == t)]

#One-Hot Encoding
dat_nn <- dummy.data.frame(train, names = factors)
dat_nn <- as.data.frame(lapply(dat_nn, as.numeric))
train_nn <- dummy.data.frame(train_stack_bal, names = factors)
train_nn <- as.data.frame(lapply(train_nn, as.numeric))
val_nn <- dummy.data.frame(val_stack, names = factors)
val_nn <- as.data.frame(lapply(val_nn, as.numeric))
rm(factors)
rm(t)

allvars <- colnames(train_nn)
predictors_nn <- allvars[!allvars %in% c(targetvar, "VehPrice")]
rm(allvars)

dat_nn <- dat_nn[, c(predictors_nn, targetvar)]
train_nn <- train_nn[, c(predictors_nn, targetvar)]
val_nn <- val_nn[, c(predictors_nn, targetvar)]

max <- as.numeric(apply(dat_nn, 2, max))
min <- as.numeric(apply(dat_nn, 2, min))

train_nn <- as.data.frame(scale(train_nn, center = min, scale = (max - min)))
val_nn <- as.data.frame(scale(val_nn, center = min, scale = (max - min)))

rm(max)
rm(min)

x_train <- as.matrix(train_nn[predictors_nn])
y_train <- as.matrix(train_nn[targetvar])

x_val <- as.matrix(val_nn[predictors_nn])
y_val <- as.matrix(val_nn[targetvar])

rm(predictors_nn)

#Level One Modelling
mlp1 <- keras_model_sequential()
mlp1 %>%
  layer_dense(units = 10, activation = "tanh", input_shape = ncol(x_train[, -which(colnames(x_train) %in% c("exposure"))])) %>%
  layer_dense(units = 6, activation = "tanh") %>%
  layer_dense(units = 2, activation = "tanh") %>%
  layer_dense(units = 1, activation = "sigmoid")
mlp1 %>%
  compile(loss = "mean_squared_error", optimizer = optimizer_rmsprop(lr = 0.01, rho = 0.9))
fit <- fit(mlp1, x = x_train[, -which(colnames(x_train) %in% c("exposure"))], y = y_train, batch_size = 10000, epochs = 20000, verbose = 1, 
           callbacks = list(callback_early_stopping(monitor = "loss", min_delta = 0.00001, patience = 1500, verbose = 1)))
predictions_mlp1_l1 <- mlp1 %>% predict(x_val[, -which(colnames(x_val) %in% c("exposure"))])
#predictions_mlp1_l1 <- (predictions_mlp1_l1 * (max(dat_nn[targetvar]) - min(dat_nn[targetvar]))) + min(dat_nn[targetvar])

mlp2 <- keras_model_sequential()
mlp2 %>%
  layer_dense(units = 20, activation = "tanh", input_shape = ncol(x_train[, -which(colnames(x_train) %in% c("exposure"))])) %>%
  layer_dense(units = 15, activation = "tanh") %>%
  layer_dense(units = 5, activation = "tanh") %>%
  layer_dense(units = 1, activation = "sigmoid")
mlp2 %>%
  compile(loss = "mean_squared_error", optimizer = optimizer_rmsprop(lr = 0.01, rho = 0.9))
fit <- fit(mlp2, x = x_train[, -which(colnames(x_train) %in% c("exposure"))], y = y_train, batch_size = 10000, epochs = 20000, verbose = 1, 
           callbacks = list(callback_early_stopping(monitor = "loss", min_delta = 0.00001, patience = 1500, verbose = 1)))
predictions_mlp2_l1 <- mlp2 %>% predict(x_val[, -which(colnames(x_val) %in% c("exposure"))])
#predictions_mlp2_l1 <- (predictions_mlp2_l1 * (max(dat_nn[targetvar]) - min(dat_nn[targetvar]))) + min(dat_nn[targetvar])

mlp3 <- keras_model_sequential()
mlp3 %>%
  layer_dense(units = 4, activation = "tanh", input_shape = ncol(x_train[, -which(colnames(x_train) %in% c("exposure"))])) %>%
  layer_dense(units = 2, activation = "tanh") %>%
  layer_dense(units = 1, activation = "sigmoid")
mlp3 %>%
  compile(loss = "mean_squared_error", optimizer = optimizer_rmsprop(lr = 0.01, rho = 0.9))
fit <- fit(mlp3, x = x_train[, -which(colnames(x_train) %in% c("exposure"))], y = y_train, batch_size = 10000, epochs = 20000, verbose = 1, 
           callbacks = list(callback_early_stopping(monitor = "loss", min_delta = 0.00001, patience = 1500, verbose = 1)))
predictions_mlp3_l1 <- mlp3 %>% predict(x_val[, -which(colnames(x_val) %in% c("exposure"))])
#predictions_mlp3_l1 <- (predictions_mlp3_l1 * (max(dat_nn[targetvar]) - min(dat_nn[targetvar]))) + min(dat_nn[targetvar])

mlp4 <- keras_model_sequential()
mlp4 %>%
  layer_dense(units = 5, activation = "tanh", input_shape = ncol(x_train[, -which(colnames(x_train) %in% c("exposure"))])) %>%
  layer_dense(units = 1, activation = "sigmoid")
mlp4 %>%
  compile(loss = "mean_squared_error", optimizer = optimizer_rmsprop(lr = 0.01, rho = 0.9))
fit <- fit(mlp4, x = x_train[, -which(colnames(x_train) %in% c("exposure"))], y = y_train, batch_size = 10000, epochs = 20000, verbose = 1, 
           callbacks = list(callback_early_stopping(monitor = "loss", min_delta = 0.00001, patience = 1500, verbose = 1)))
predictions_mlp4_l1 <- mlp4 %>% predict(x_val[, -which(colnames(x_val) %in% c("exposure"))])
#predictions_mlp4_l1 <- (predictions_mlp4_l1 * (max(dat_nn[targetvar]) - min(dat_nn[targetvar]))) + min(dat_nn[targetvar])

#Pull in Level 1 Predictions and Append to Validation Set for Level 2 - Set weights for each model as appropriate
l1_predictions <- (0.4*predictions_mlp1_l1) + (0.6*predictions_mlp2_l1) + (0*predictions_mlp3_l1) + (0*predictions_mlp4_l1)
val_stack$l1_pred <- as.numeric(l1_predictions)

#Level Two Modelling
xgboost <- xgboost(data = x_val[, -which(colnames(x_val) %in% c("exposure"))], label = as.numeric(l1_predictions), objective = "reg:logistic", nrounds = 35000, verbose = 1, 
                   max.depth = 6, eta = 0.1, early_stopping_rounds = 1000, weights = as.numeric(x_val[, c("exposure")]))
predictions_xgb_l2 <- predict(xgboost, x_test[, -which(colnames(x_test) %in% c("exposure"))])
predictions_l2 <- (predictions_xgb_l2 * (max(dat_nn[targetvar]) - min(dat_nn[targetvar]))) + min(dat_nn[targetvar])

##======TEST OUTPUT & ANALYTICS=========##
test$GLM <- as.numeric(predictions_glm)
test$MLP1 <- as.numeric(predictions_mlp1)
test$MLP2 <- as.numeric(predictions_mlp2)
test$MLP3 <- as.numeric(predictions_mlp3)
test$MLP4 <- as.numeric(predictions_mlp4)
test$XGBOOST <- as.numeric(predictions_xgb)
test$STACK <- as.numeric(predictions_l2)

rmse_stack <- rmse(data = test, targetvar = c("ClaimAmount"), prediction.obj = as.numeric(test$STACK))
rmse_stack

#One-Way Analysis against Burning Cost for Vehicle Body
grp_by_VehBody <- test %>%
  group_by(VehBody) %>%
  summarize(total_rate_glm = sum(GLM), total_rate_xgb = sum(XGBOOST), total_rate_mlp2 = sum(MLP2), exp = sum(exposure), total_rate_mlp1 = sum(MLP1),
            total_rate_stack = sum(STACK), total_clm = sum(ClaimAmount), total_rate_mlp3 = sum(MLP3), total_rate_mlp4 = sum(MLP4))
grp_by_VehBody$avg_rate_glm <- grp_by_VehBody$total_rate_glm / grp_by_VehBody$exp
grp_by_VehBody$avg_rate_xgb <- grp_by_VehBody$total_rate_xgb / grp_by_VehBody$exp
grp_by_VehBody$avg_rate_mlp2 <- grp_by_VehBody$total_rate_mlp2 / grp_by_VehBody$exp
grp_by_VehBody$avg_rate_mlp1 <- grp_by_VehBody$total_rate_mlp1 / grp_by_VehBody$exp
grp_by_VehBody$avg_rate_mlp3 <- grp_by_VehBody$total_rate_mlp3 / grp_by_VehBody$exp
grp_by_VehBody$avg_rate_mlp4 <- grp_by_VehBody$total_rate_mlp4 / grp_by_VehBody$exp
grp_by_VehBody$avg_rate_stack <- grp_by_VehBody$total_rate_stack / grp_by_VehBody$exp
grp_by_VehBody$avg_clm <- grp_by_VehBody$total_clm / grp_by_VehBody$exp

plot_VehBody <- plot_ly(data = grp_by_VehBody, x = ~VehBody, y = ~avg_clm, type = "scatter", mode = "lines", name = "BC") %>%
  add_trace(y = ~avg_rate_xgb, mode = "lines", name = "XGBoost") %>%
  add_trace(y = ~avg_rate_mlp2, mode = "lines", name = "MLP 2") %>%
  add_trace(y = ~avg_rate_mlp1, mode = "lines", name = "MLP 1") %>%
  add_trace(y = ~avg_rate_mlp3, mode = "lines", name = "MLP 3") %>%
  add_trace(y = ~avg_rate_mlp4, mode = "lines", name = "MLP 4") %>%
  add_trace(y = ~avg_rate_stack, mode = "lines", name = "STACK APPROACH") %>%
  add_trace(y = ~avg_rate_glm, mode = "lines", name = "GLM")
plot_VehBody

#One-Way Analysis against Burning Cost for Vehicle Age
grp_by_VehAge <- test %>%
  group_by(VehAge) %>%
  summarize(total_rate_glm = sum(GLM), total_rate_xgb = sum(XGBOOST), total_rate_mlp2 = sum(MLP2), exp = sum(exposure), total_rate_mlp1 = sum(MLP1),
            total_rate_stack = sum(STACK), total_clm = sum(ClaimAmount), total_rate_mlp3 = sum(MLP3), total_rate_mlp4 = sum(MLP4))
grp_by_VehAge$avg_rate_glm <- grp_by_VehAge$total_rate_glm / grp_by_VehAge$exp
grp_by_VehAge$avg_rate_xgb <- grp_by_VehAge$total_rate_xgb / grp_by_VehAge$exp
grp_by_VehAge$avg_rate_mlp2 <- grp_by_VehAge$total_rate_mlp2 / grp_by_VehAge$exp
grp_by_VehAge$avg_rate_mlp1 <- grp_by_VehAge$total_rate_mlp1 / grp_by_VehAge$exp
grp_by_VehAge$avg_rate_mlp3 <- grp_by_VehAge$total_rate_mlp3 / grp_by_VehAge$exp
grp_by_VehAge$avg_rate_mlp4 <- grp_by_VehAge$total_rate_mlp4 / grp_by_VehAge$exp
grp_by_VehAge$avg_rate_stack <- grp_by_VehAge$total_rate_stack / grp_by_VehAge$exp
grp_by_VehAge$avg_clm <- grp_by_VehAge$total_clm / grp_by_VehAge$exp

plot_VehAge <- plot_ly(data = grp_by_VehAge, x = ~VehAge, y = ~avg_clm, type = "scatter", mode = "lines", name = "BC") %>%
  add_trace(y = ~avg_rate_xgb, mode = "lines", name = "XGBoost") %>%
  add_trace(y = ~avg_rate_mlp2, mode = "lines", name = "MLP 2") %>%
  add_trace(y = ~avg_rate_mlp1, mode = "lines", name = "MLP 1") %>%
  add_trace(y = ~avg_rate_mlp3, mode = "lines", name = "MLP 3") %>%
  add_trace(y = ~avg_rate_mlp4, mode = "lines", name = "MLP 4") %>%
  add_trace(y = ~avg_rate_stack, mode = "lines", name = "STACK APPROACH") %>%
  add_trace(y = ~avg_rate_glm, mode = "lines", name = "GLM")
plot_VehAge

#One-Way Analysis against Burning Cost for Gender
grp_by_Gender <- test %>%
  group_by(Gender) %>%
  summarize(total_rate_glm = sum(GLM), total_rate_xgb = sum(XGBOOST), total_rate_mlp2 = sum(MLP2), exp = sum(exposure), total_rate_mlp1 = sum(MLP1),
            total_rate_stack = sum(STACK), total_clm = sum(ClaimAmount), total_rate_mlp3 = sum(MLP3), total_rate_mlp4 = sum(MLP4))
grp_by_Gender$avg_rate_glm <- grp_by_Gender$total_rate_glm / grp_by_Gender$exp
grp_by_Gender$avg_rate_xgb <- grp_by_Gender$total_rate_xgb / grp_by_Gender$exp
grp_by_Gender$avg_rate_mlp2 <- grp_by_Gender$total_rate_mlp2 / grp_by_Gender$exp
grp_by_Gender$avg_rate_mlp1 <- grp_by_Gender$total_rate_mlp1 / grp_by_Gender$exp
grp_by_Gender$avg_rate_mlp3 <- grp_by_Gender$total_rate_mlp3 / grp_by_Gender$exp
grp_by_Gender$avg_rate_mlp4 <- grp_by_Gender$total_rate_mlp4 / grp_by_Gender$exp
grp_by_Gender$avg_rate_stack <- grp_by_Gender$total_rate_stack / grp_by_Gender$exp
grp_by_Gender$avg_clm <- grp_by_Gender$total_clm / grp_by_Gender$exp

plot_Gender <- plot_ly(data = grp_by_Gender, x = ~Gender, y = ~avg_clm, type = "scatter", mode = "lines", name = "BC") %>%
  add_trace(y = ~avg_rate_xgb, mode = "lines", name = "XGBoost") %>%
  add_trace(y = ~avg_rate_mlp2, mode = "lines", name = "MLP 2") %>%
  add_trace(y = ~avg_rate_mlp1, mode = "lines", name = "MLP 1") %>%
  add_trace(y = ~avg_rate_mlp3, mode = "lines", name = "MLP 3") %>%
  add_trace(y = ~avg_rate_mlp4, mode = "lines", name = "MLP 4") %>%
  add_trace(y = ~avg_rate_stack, mode = "lines", name = "STACK APPROACH") %>%
  add_trace(y = ~avg_rate_glm, mode = "lines", name = "GLM")
plot_Gender

#One-Way Analysis against Burning Cost for Vehicle Usage
grp_by_VehUsage <- test %>%
  group_by(VehUsage) %>%
  summarize(total_rate_glm = sum(GLM), total_rate_xgb = sum(XGBOOST), total_rate_mlp2 = sum(MLP2), exp = sum(exposure), total_rate_mlp1 = sum(MLP1),
            total_rate_stack = sum(STACK), total_clm = sum(ClaimAmount), total_rate_mlp3 = sum(MLP3), total_rate_mlp4 = sum(MLP4))
grp_by_VehUsage$avg_rate_glm <- grp_by_VehUsage$total_rate_glm / grp_by_VehUsage$exp
grp_by_VehUsage$avg_rate_xgb <- grp_by_VehUsage$total_rate_xgb / grp_by_VehUsage$exp
grp_by_VehUsage$avg_rate_mlp2 <- grp_by_VehUsage$total_rate_mlp2 / grp_by_VehUsage$exp
grp_by_VehUsage$avg_rate_mlp1 <- grp_by_VehUsage$total_rate_mlp1 / grp_by_VehUsage$exp
grp_by_VehUsage$avg_rate_mlp3 <- grp_by_VehUsage$total_rate_mlp3 / grp_by_VehUsage$exp
grp_by_VehUsage$avg_rate_mlp4 <- grp_by_VehUsage$total_rate_mlp4 / grp_by_VehUsage$exp
grp_by_VehUsage$avg_rate_stack <- grp_by_VehUsage$total_rate_stack / grp_by_VehUsage$exp
grp_by_VehUsage$avg_clm <- grp_by_VehUsage$total_clm / grp_by_VehUsage$exp

plot_VehUsage <- plot_ly(data = grp_by_VehUsage, x = ~VehUsage, y = ~avg_clm, type = "scatter", mode = "lines", name = "BC") %>%
  add_trace(y = ~avg_rate_xgb, mode = "lines", name = "XGBoost") %>%
  add_trace(y = ~avg_rate_mlp2, mode = "lines", name = "MLP 2") %>%
  add_trace(y = ~avg_rate_mlp1, mode = "lines", name = "MLP 1") %>%
  add_trace(y = ~avg_rate_mlp3, mode = "lines", name = "MLP 3") %>%
  add_trace(y = ~avg_rate_mlp4, mode = "lines", name = "MLP 4") %>%
  add_trace(y = ~avg_rate_stack, mode = "lines", name = "STACK APPROACH") %>%
  add_trace(y = ~avg_rate_glm, mode = "lines", name = "GLM")
plot_VehUsage

##=======INTERPRETABILITY==========##
#Initialize Parallel Processing
cl <- makePSOCKcluster(8)
registerDoParallel(cl)

targetvar <- c("STACK")
predictors <- c("LicAge", "VehAge", "Gender", "MariStat", "VehUsage", "DrivAge", "HasKmLimit", "BonusMalus", "VehBody", "VehPriceGrp", "VehEngine",
                "VehEnergy", "VehMaxSpeed", "VehClass", "Garage")
predictors <- paste(predictors, collapse = "+")
formula <- as.formula(paste(targetvar,"~",predictors,collapse = "+"))

#RF Surrogate Model
rf_surrogate <- ranger(formula = formula, data = test, num.trees = 1000, importance = "impurity", case.weights = as.numeric(test$exposure))
test$RF_SURROGATE <- as.numeric(predict(rf_surrogate, test)$predictions)

#Feature Importance using VIP
features <- c("LicAge", "VehAge", "Gender", "MariStat", "VehUsage", "DrivAge", "HasKmLimit", "BonusMalus", "VehBody", "VehPriceGrp", "VehEngine",
              "VehEnergy", "VehMaxSpeed", "VehClass", "Garage")
imp <- vi(rf_surrogate, method = "model", feature_names = features, scale = TRUE)
vip(rf_surrogate, method = "model", feature_names = features, scale = TRUE) #Impurity-based Feature Importance - I DON'T TRUST THIS
#vip(rf_surrogate, method = "pdp", feature_names = features, scale = TRUE) #PD-based Feature Importance - Experimental so wouldn't use this

##Feature Importance and Effects using IML
#Create Predictor Object
feat_spac <- test[, which(colnames(test) %in% features)]
pred <- function(model, newdata) {
  results <- as.numeric(predict(model, as.data.frame(newdata))$predictions)
  return(results)
}
predictor <- Predictor$new(rf_surrogate, data = feat_spac, y = as.numeric(test$RF_SURROGATE), predict.fun = pred)

#MAE-based Feature Importance
ptm <- proc.time()
imp <- FeatureImp$new(predictor = predictor, loss = "mae", parallel = TRUE)
proc.time() - ptm
plot(imp)

#Interaction Effects
build_int_terms <- function(feature_vec) {
  f1 <- c()
  f2 <- c()
  features_considered <- c()
  int_terms <- list()
  na_terms <- c()
  for(i in 1:length(feature_vec)) {
    f1[i] <- feature_vec[i]
    f <- feature_vec[-i]
    for(j in 1:length(f)) {
      if(is.null(features_considered)) {
        f <- f
      } else {
        f <- f[!f %in% features_considered]
      }
      f2[j] <- paste0(f1[i],"*",f[j])
      int_terms <- append(int_terms, f2[j])
    }
    features_considered <- append(features_considered, f1[i])
  }
  for (n in 1:length(int_terms)) {
    int_terms[[n]] <- unlist(strsplit(int_terms[[n]], split = "*", fixed = TRUE))
    if(int_terms[[n]][2] %in% c('NA')) {
      na_terms <- append(na_terms, n)
    } else {
      n <- n + 1
    }
  }
  int_terms <- int_terms[-c(na_terms)]
  return(int_terms)
}
int_terms <- build_int_terms(features)

int <- c()
for (i in 1:length(int_terms)) {
  int[i] <- as.numeric(vint(rf_surrogate, feature_names = int_terms[[i]])[, 2])
  print(paste("Interactions Considered: ", i, " of ", length(int_terms)))
}

for(i in 1:length(int_terms)) {
  int_terms[[i]] <- paste0(int_terms[[i]][1],":",int_terms[[i]][2])
}
int_terms <- unlist(int_terms)

interactions <- data.frame(int_terms, int)
interactions <- interactions %>%
  arrange(desc(int))
interactions$int_terms <- factor(interactions$int_terms, levels = unique(interactions$int_terms)[order(interactions$int, decreasing = FALSE)])
int_plot <- plot_ly(data = interactions, x = ~int, y = ~int_terms, type = "bar", orientation = "h")
int_plot

#Local Interpretation
































































































































































































































































































































































































































































































































































































































































































































































































































































































































