library(ROSE)
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
library(vip)
library(h2o)
library(fastDummies)
library(DALEX)
library(splitstackshape)
library(processx)
library(ranger)
library(caret)
library(magrittr)
library(pracma)
library(tidyr)
install_keras(tensorflow = "gpu")

dat <- read.csv("freMPL.csv")

matrix(colnames(dat), ncol = 1)

dat <- dat[, c(2, 3, seq(6, 20, by = 1), 22, 23)]
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

#Looking at SocioCateg
ord <- sort(table(dat$SocioCateg), decreasing = TRUE)
top_15_socio_categ <- names(ord)[1:15] #Keeping top 15 SocioCategs
dat <- dat[dat$SocioCateg %in% top_15_socio_categ, ]

dat <- dat %>%
  mutate_if(is.factor, as.character)

dat <- dat %>%
  mutate_if(is.character, as.factor)

#Split into Train and Test
set.seed(1245788)
ind <- sample(1:nrow(dat), round(0.2 * nrow(dat)))
train <- dat[-ind, ]
test <- dat[ind, ]
rm(ind)

#Define RMSE and NRMSE for Model Evaluation
rmse <- function(data, targetvar, prediction.obj) {
  rss <- (as.numeric(prediction.obj) - as.numeric(unlist(data[, c(targetvar)]))) ^ 2
  mse <- sum(rss) / nrow(data)
  rmse <- sqrt(mse)
  rmse <- rmse / 100
  return(rmse)
}

NRMSE <- function(pred, obs) {
  RMSE(pred, obs)/(max(obs)-min(obs))
}

gini_value <- function(predicted_loss_cost, exposure) {
  lc_var <- enquo(predicted_loss_cost)
  exp <- enquo(exposure)
  dataset <- tibble(lc_var = !! lc_var, exp = !! exp)
  dataset %>% 
    arrange(lc_var) %>% 
    mutate(cum_exp = cumsum(exp) / sum(exp),
           cum_pred_lc = cumsum(lc_var) / sum(lc_var)) %$% 
    {trapz(cum_exp, cum_pred_lc) %>% add(-1) %>% abs() %>% subtract(.5) %>% multiply_by(2)}
}

text_project <- element_text(size = 18)

theme_project <- theme(
  axis.title = text_project
  , legend.text = text_project
  , axis.text = text_project
  , legend.position = 'bottom'
)

gini_plot <- function(predicted_loss_cost, exposure){
  
  lc_var <- enquo(predicted_loss_cost)
  exp <- enquo(exposure)
  
  dataset <- tibble(lc_var = !! lc_var, exp = !! exp)
  
  dataset %>% 
    arrange(lc_var) %>% 
    mutate(cum_exp = cumsum(exp) / sum(exp),
           cum_pred_lc = cumsum(lc_var) / sum(lc_var)) %>% 
    ggplot() +
    geom_line(aes(x = cum_exp, y = cum_pred_lc)) +
    geom_abline(intercept = 0, slope = 1) +
    xlab("Exposure") +
    ylab("Predicted Loss Cost") + 
    theme_project
  
}

lift_curve_table <- function(predicted_loss_cost, observed_loss_cost, exposure, n) {
  
  dataset <- tibble(
    pred_lc = predicted_loss_cost
    , obs_lc = observed_loss_cost
    , exp = exposure
  )
  
  dataset <- dataset %>% 
    mutate(buckets = ntile(exp, n)) %>% 
    group_by(buckets) %>% 
    summarise(
      Predicted_Risk_Premium = mean(pred_lc, na.rm = TRUE)
      , Observed_Risk_Premium = mean(obs_lc, na.rm = TRUE)
      , Exposure = sum(exp)
    )
  
  max_bucket <- which.max(dataset$Exposure)
  dataset <- dataset %>% 
    mutate(
      base_predicted_rp = dataset[max_bucket, ]$Predicted_Risk_Premium
      , base_observed_rp = dataset[max_bucket, ]$Observed_Risk_Premium
    )
  
  dataset %>%
    mutate(
      Predicted_Risk_Premium = Predicted_Risk_Premium / base_predicted_rp
      , Observed_Risk_Premium = Observed_Risk_Premium / base_observed_rp
      , buckets = as.factor(buckets)
    ) %>% 
    arrange(Predicted_Risk_Premium) %>% 
    mutate(
      buckets = seq_len(nrow(.))
    ) %>% 
    select(-Exposure)
  
}

lift_curve_plot <- function(tbl_in) {
  
  tbl_in %>% 
    tidyr::pivot_longer(c(Predicted_Risk_Premium, Observed_Risk_Premium)) %>% 
    mutate(buckets = as.factor(buckets)) %>% 
    ggplot(aes(x = buckets, y = value, col = name, group = name)) +
    geom_point(size = 3.5) +
    geom_line(size = 4) +
    labs(x = "Bucket", y = "Average Risk Premium") + 
    theme_project
}

double_lift_chart <- function(predicted_loss_cost_mod_1, predicted_loss_cost_mod_2, observed_loss_cost, n) {
  pred_lc_m1 <- enquo(predicted_loss_cost_mod_1)
  pred_lc_m2 <- enquo(predicted_loss_cost_mod_2)
  obs_lc <- enquo(observed_loss_cost)
  dataset <- tibble(pred_lc_m1 = !! pred_lc_m1, pred_lc_m2 = !! pred_lc_m2, obs_lc = !! obs_lc)
  dataset %>%
    mutate(sort_ratio = pred_lc_m1 / pred_lc_m2) %>%
    arrange(sort_ratio) %>% 
    mutate(buckets = ntile(sort_ratio, n)) %>% 
    group_by(buckets) %>% 
    summarise(Model_1_Predicted_Risk_Premium = mean(pred_lc_m1),
              Model_2_Predicted_Risk_Premium = mean(pred_lc_m2),
              Observed_Risk_Premium = mean(obs_lc))%>% 
    tidyr::pivot_longer(c(Model_1_Predicted_Risk_Premium, Model_2_Predicted_Risk_Premium, Observed_Risk_Premium)) %>%
    ggplot() +
    geom_line(aes(x = as.factor(buckets), y = value, col = name, group = name)) +
    geom_point(aes(x = as.factor(buckets), y = value, col = name, group = name)) +
    xlab("Bucket") + ylab("Average Risk Premium")
}

##========BASELINE GLM==========##
targetvar <- c("ClaimAmount")
predictors <- c("LicAge", "VehAge", "Gender", "MariStat", "VehUsage", "DrivAge", "HasKmLimit", "BonusMalus", "VehBody", "VehPriceGrp", "VehEngine",
                "VehEnergy", "VehMaxSpeed", "VehClass", "Garage", "SocioCateg")
predictors <- paste(predictors, collapse = "+")
formula <- as.formula(paste(targetvar,"~",predictors,collapse = "+"))

p_values <- seq(1.2, 1.65, length = 7)
p_tuning <- tweedie.profile(formula = formula, data = train, p.vec = p_values, do.plot = FALSE, 
                            offset = log(exposure), verbose = 2, method = "series")
p <- p_tuning$p.max
rm(p_values)
rm(p_tuning)

glm <- glm(formula = formula, data = train, family = tweedie(var.power = p, link.power = 0), offset = log(exposure))
rm(formula)
rm(predictors)

predictions_glm <- as.numeric(predict(glm, test, type = "response"))

rmse_glm <- rmse(data = test, targetvar = targetvar, prediction.obj = predictions_glm)
rmse_glm

nrmse_glm <- NRMSE(predictions_glm, test$ClaimAmount)
nrmse_glm

gini_glm <- gini_value(predictions_glm, test$exposure)
gini_glm

#lift_curve_glm <- lift_curve_plot(predictions_glm, as.numeric(unlist(test[targetvar])), 10)
#lift_curve_glm + labs(col = "Model") + ggtitle("GLM Lift Curve")

#gini_index_plot_glm <- gini_plot(predictions_glm, as.numeric(unlist(test[, c("exposure")])))
#gini_index_plot_glm + ggtitle("Tweedie GLM Gini Index Plot")

rm(targetvar)

##=====STACK APPROACH=======##
#Split Training dataset and Process
set.seed(12222)
ind <- sample(1:nrow(train), round(0.5 * nrow(train)))
train_stack <- train[-ind, ]
val_stack <- train[ind, ]
rm(ind)

targetvar <- c("ClaimAmount")
predictors <- c("LicAge", "VehAge", "Gender", "MariStat", "VehUsage", "DrivAge", "HasKmLimit", "BonusMalus", "VehBody", "VehPriceGrp", "VehEngine",
                "VehEnergy", "VehMaxSpeed", "VehClass", "Garage", "exposure", "SocioCateg")

factors <- c()
for (i in 1:length(predictors)) {
  if (class(unlist(dat[, predictors[i]])) == "factor") {
    factors[i] <- predictors[i]
  } else {
    i <- i + 1
  }
}

factors <- factors[!factors %in% NA]
rm(i)

#Resample Training Data
table(train_stack$ClaimInd) #Without Resampling
mean(train_stack$ClaimInd == 1)*100 #Percentage of Records with a Claim
t <- c("ClaimInd")
pp <- append(predictors, targetvar)
pp <- paste(pp, collapse = "+")
f <- as.formula(paste(t,"~",pp,collapse = "+"))
train_stack_bal <- ovun.sample(f, train_stack, p = 0.25, method = "both")$data
train_stack_bal <- train_stack
table(train_stack_bal$ClaimInd) #After Resampling
rm(pp)
rm(f)
train_stack_bal <- train_stack_bal[, -which(colnames(train_stack_bal) == t)]

#One-Hot Encoding - dummy_cols() from fastDummies might be a more efficient way
dat_nn <- dummy_cols(dat, select_columns = factors)
train_nn <- dummy_cols(train_stack, select_columns = factors)
val_nn <- dummy_cols(val_stack, select_columns = factors)
test_nn <- dummy_cols(test, select_columns = factors)

allvars <- colnames(dat_nn)
predictors_nn <- allvars[!allvars %in% c(targetvar, factors, t)]
rm(allvars)
rm(t)
rm(factors)

dat_nn <- dat_nn[, c(predictors_nn, targetvar)]
train_nn <- train_nn[, c(predictors_nn, targetvar)]
val_nn <- val_nn[, c(predictors_nn, targetvar)]
test_nn <- test_nn[, c(predictors_nn, targetvar)]

max <- as.numeric(apply(dat_nn, 2, max))
min <- as.numeric(apply(dat_nn, 2, min))

train_nn <- as.data.frame(scale(train_nn, center = min, scale = (max - min)))
val_nn <- as.data.frame(scale(val_nn, center = min, scale = (max - min)))
test_nn <- as.data.frame(scale(test_nn, center = min, scale = (max - min)))

rm(max)
rm(min)

x_train <- as.matrix(train_nn[predictors_nn])
y_train <- as.matrix(train_nn[targetvar])

x_val <- as.matrix(val_nn[predictors_nn])
y_val <- as.matrix(val_nn[targetvar])

x_test <- as.matrix(test_nn[predictors_nn])
y_test <- as.matrix(test_nn[targetvar])

rm(predictors_nn)

#Level One Modelling
mlp1 <- keras_model_sequential()
mlp1 %>%
  layer_dense(units = 100, activation = "tanh", input_shape = ncol(x_train[, -which(colnames(x_train) %in% c("exposure"))])) %>%
  layer_dense(units = 60, activation = "tanh") %>%
  layer_dense(units = 20, activation = "sigmoid") %>%
  layer_dense(units = 1, activation = "sigmoid")
mlp1 <- multi_gpu_model(mlp1, gpus = 2)
mlp1 %>%
  compile(loss = "mean_squared_error", optimizer = optimizer_rmsprop(lr = 0.01, rho = 0.9))
fit <- fit(mlp1, x = x_train[, -which(colnames(x_train) %in% c("exposure"))], y = y_train, batch_size = 2000, epochs = 20000, verbose = 1, 
           callbacks = list(callback_early_stopping(monitor = "loss", min_delta = 0.00001, patience = 1500, verbose = 1)))
predictions_mlp1_l1 <- mlp1 %>% predict(x_val[, -which(colnames(x_val) %in% c("exposure"))])
#predictions_mlp1_l1 <- (predictions_mlp1_l1 * (max(dat_nn[targetvar]) - min(dat_nn[targetvar]))) + min(dat_nn[targetvar])

mlp2 <- keras_model_sequential()
mlp2 %>%
  layer_dense(units = 200, activation = "tanh", input_shape = ncol(x_train[, -which(colnames(x_train) %in% c("exposure"))])) %>%
  layer_dense(units = 150, activation = "tanh") %>%
  layer_dense(units = 50, activation = "sigmoid") %>%
  layer_dense(units = 1, activation = "sigmoid")
mlp2 <- multi_gpu_model(mlp2, gpus = 2)
mlp2 %>%
  compile(loss = "mean_squared_error", optimizer = optimizer_rmsprop(lr = 0.01, rho = 0.9))
fit <- fit(mlp2, x = x_train[, -which(colnames(x_train) %in% c("exposure"))], y = y_train, batch_size = 2000, epochs = 20000, verbose = 1, 
           callbacks = list(callback_early_stopping(monitor = "loss", min_delta = 0.00001, patience = 1500, verbose = 1)))
predictions_mlp2_l1 <- mlp2 %>% predict(x_val[, -which(colnames(x_val) %in% c("exposure"))])
#predictions_mlp2_l1 <- (predictions_mlp2_l1 * (max(dat_nn[targetvar]) - min(dat_nn[targetvar]))) + min(dat_nn[targetvar])

mlp3 <- keras_model_sequential()
mlp3 %>%
  layer_dense(units = 400, activation = "tanh", input_shape = ncol(x_train[, -which(colnames(x_train) %in% c("exposure"))])) %>%
  layer_dense(units = 200, activation = "tanh") %>%
  layer_dense(units = 1, activation = "sigmoid")
mlp3 <- multi_gpu_model(mlp3, gpus = 2)
mlp3 %>%
  compile(loss = "mean_squared_error", optimizer = optimizer_rmsprop(lr = 0.01, rho = 0.9))
fit <- fit(mlp3, x = x_train[, -which(colnames(x_train) %in% c("exposure"))], y = y_train, batch_size = 2000, epochs = 20000, verbose = 1, 
           callbacks = list(callback_early_stopping(monitor = "loss", min_delta = 0.00001, patience = 1500, verbose = 1)))
predictions_mlp3_l1 <- mlp3 %>% predict(x_val[, -which(colnames(x_val) %in% c("exposure"))])
#predictions_mlp3_l1 <- (predictions_mlp3_l1 * (max(dat_nn[targetvar]) - min(dat_nn[targetvar]))) + min(dat_nn[targetvar])

mlp4 <- keras_model_sequential()
mlp4 %>%
  layer_dense(units = 500, activation = "sigmoid", input_shape = ncol(x_train[, -which(colnames(x_train) %in% c("exposure"))])) %>%
  layer_dropout(0.4) %>%
  layer_dense(units = 1, activation = "sigmoid")
mlp4 <- multi_gpu_model(mlp4, gpus = 2)
mlp4 %>%
  compile(loss = "mean_squared_error", optimizer = optimizer_rmsprop(lr = 0.01, rho = 0.9))
fit <- fit(mlp4, x = x_train[, -which(colnames(x_train) %in% c("exposure"))], y = y_train, batch_size = 2000, epochs = 20000, verbose = 1, 
           callbacks = list(callback_early_stopping(monitor = "loss", min_delta = 0.00001, patience = 1500, verbose = 1)))
predictions_mlp4_l1 <- mlp4 %>% predict(x_val[, -which(colnames(x_val) %in% c("exposure"))])
#predictions_mlp4_l1 <- (predictions_mlp4_l1 * (max(dat_nn[targetvar]) - min(dat_nn[targetvar]))) + min(dat_nn[targetvar])

#h2o.init(min_mem_size = "255g")
#train_h2o <- as.h2o(cbind(x_train, y_train))
#colnames(train_h2o) <- colnames(cbind(x_train, y_train))
#train_h2o <- train_h2o[-1, ]
#rf_params <- list(ntrees = c(4000, 6000), max_depth = c(10, 25, 40))
#rf_grid <- h2o.grid("randomForest", x = colnames(train_h2o)[-which(colnames(train_h2o) %in% c(targetvar, "exposure"))], 
#                    y = targetvar, training_frame = train_h2o, seed = 13578, hyper_params = rf_params, grid_id = "rf_grid")
#rf_gridPerf <- h2o.getGrid("rf_grid", sort_by = "rmse", decreasing = FALSE)
#rf <- h2o.getModel(rf_gridPerf@model_ids[[1]])
#rm(train_h2o)
#predictions_rf_l1 <- h2o.predict(rf, as.h2o(x_val[-1, -which(colnames(x_val) %in% c("exposure"))]))
#predictions_rf_l1 <- as.numeric(as.list(predictions_rf_l1))
#rm(rf_params)
#rm(rf_grid)
#rm(rf_gridPerf)
#rm(rf)
#h2o.shutdown(prompt = FALSE)

#Pull in Level 1 Predictions and Append to Validation Set for Level 2 - Set weights for each model based on individual predictive accuracy
rmse_1 <- rmse(val_stack, targetvar, as.numeric((predictions_mlp1_l1 * (max(dat_nn[targetvar]) - min(dat_nn[targetvar]))) + min(dat_nn[targetvar])))
rmse_2 <- rmse(val_stack, targetvar, as.numeric((predictions_mlp2_l1 * (max(dat_nn[targetvar]) - min(dat_nn[targetvar]))) + min(dat_nn[targetvar])))
rmse_3 <- rmse(val_stack, targetvar, as.numeric((predictions_mlp3_l1 * (max(dat_nn[targetvar]) - min(dat_nn[targetvar]))) + min(dat_nn[targetvar])))
rmse_4 <- rmse(val_stack, targetvar, as.numeric((predictions_mlp4_l1 * (max(dat_nn[targetvar]) - min(dat_nn[targetvar]))) + min(dat_nn[targetvar])))
l1_rmse <- c(rmse_1, rmse_2, rmse_3, rmse_4)
soft_sum <- sum(exp(l1_rmse)) 
weights <- c()
for (i in 1:length(l1_rmse)) {
  weights[i] <- 1 - (exp(l1_rmse[i]) / soft_sum)
}
weights <- weights / sum(weights)
rm(i)
l1_predictions <- (weights[1]*predictions_mlp1_l1) + (weights[2]*predictions_mlp2_l1) + (weights[3]*predictions_mlp3_l1) + (weights[4]*predictions_mlp4_l1)
val_stack$l1_pred <- as.numeric(l1_predictions)
#Level Two Modelling
#params <- list(
#  objective = "reg:tweedie", tweedie_variance_parameter = p, eval_metric = "rmse"
#)
#xgb_train <- xgb.DMatrix(cbind(x_val, l1_predictions), label = as.numeric(l1_predictions))
#colnames(xgb_train)[length(colnames(xgb_train))] <- targetvar
#min_exp <- min(x_val[, c("exposure")])
#max_exp <- max(x_val[, c("exposure")])
#offset <- log(x_val[, c("exposure")])
#offset_scaled <- scale(offset, center = min_exp, scale = (max_exp - min_exp))
#rm(min_exp, max_exp, offset)
#setinfo(xgb_train, "base_margin", offset_scaled)
#xgboost <- xgboost(data = xgb_train, params = params, nrounds = 50000, verbose = 1, max.depth = 10, eta = 0.1, 
#                   early_stopping_rounds = 2500)
#predictions_xgb_l2 <- predict(xgboost, xgb.DMatrix(cbind(x_test, y_test), label = as.numeric(y_test)))
#predictions_l2 <- (predictions_xgb_l2 * (max(dat_nn[targetvar]) - min(dat_nn[targetvar]))) + min(dat_nn[targetvar])
params <- list(
  objective = "reg:tweedie", eval_metric = "rmse", tweedie_variance_power = p
)
xgb_train <- xgb.DMatrix(cbind(x_val, l1_predictions), label = as.numeric(l1_predictions))
colnames(xgb_train)[length(colnames(xgb_train))] <- targetvar
setinfo(xgb_train, "base_margin", log(xgb_train[, c("exposure")]))
xgboost <- xgboost(data = xgb_train, params = params, nrounds = 50000, verbose = 1, max.depth = 10, 
                   eta = 0.1, early_stopping_rounds = 2500)
predictions_xgb_l2 <- predict(xgboost, xgb.DMatrix(cbind(x_test, y_test), label = as.numeric(y_test)))
predictions_l2 <- (predictions_xgb_l2 * (max(dat_nn[targetvar]) - min(dat_nn[targetvar]))) + min(dat_nn[targetvar])

##======TEST OUTPUT & ANALYTICS=========##
test$GLM <- as.numeric(predictions_glm)
test$STACK <- as.numeric(predictions_l2)

rmse_stack <- rmse(data = test, targetvar = c("ClaimAmount"), prediction.obj = as.numeric(test$STACK))
rmse_stack

nrmse_stack <- NRMSE(test$STACK, test$ClaimAmount)
nrmse_stack

gini_stack<- gini_value(predictions_l2, test$exposure)
gini_stack

lift_curve_stack <- lift_curve_plot(predictions_l2, as.numeric(unlist(test[targetvar])), 10)
lift_curve_stack + labs(col = "Model") +
  ggtitle("ML Stack Lift Curve")

double_lift_chart_glm_stack <- double_lift_chart(test$GLM, test$STACK, test$ClaimAmount, 10)
double_lift_chart_glm_stack + ggtitle("Double Lift Chart - GLM (Model 1) vs. ML Stack (Model 2)") +
  labs(col = "Model")

gini_index_plot_stack <- gini_plot(predictions_l2, as.numeric(unlist(test[, c("exposure")])))
gini_index_plot_stack + ggtitle("ML Stack Gini Index Plot")

#Comparative Analytics Summary
rmse <- c(rmse_glm, rmse_stack)
rmse <- rmse * 100
nrmse <- c(nrmse_glm, nrmse_stack)
gini_indices <- c(gini_glm, gini_stack)
models <- c("Tweedie GLM", "ML Stack")
comp_stat_summary <- data.frame(models, rmse, nrmse, gini_indices)
colnames(comp_stat_summary) <- c("Model", "RMSE", "NRMSE", "Gini Index")
write.csv(comp_stat_summary, "~/comp_stat_summary.csv")

#One-Way Analysis against Burning Cost for Vehicle Body
grp_by_VehBody <- test %>%
  group_by(VehBody) %>%
  summarize(total_rate_glm = sum(GLM), exp = sum(exposure), total_rate_stack = sum(STACK), total_clm = sum(ClaimAmount))
grp_by_VehBody$avg_rate_glm <- grp_by_VehBody$total_rate_glm / grp_by_VehBody$exp
grp_by_VehBody$avg_rate_stack <- grp_by_VehBody$total_rate_stack / grp_by_VehBody$exp
grp_by_VehBody$avg_clm <- grp_by_VehBody$total_clm / grp_by_VehBody$exp

plot_VehBody <- plot_ly(data = grp_by_VehBody, x = ~VehBody, y = ~avg_clm, type = "scatter", mode = "lines", name = "BC") %>%
  add_trace(y = ~avg_rate_stack, mode = "lines", name = "STACK APPROACH") %>%
  add_trace(y = ~avg_rate_glm, mode = "lines", name = "GLM")
plot_VehBody

#One-Way Analysis against Burning Cost for Vehicle Age
grp_by_Gender <- test %>%
  group_by(Gender) %>%
  summarize(total_rate_glm = sum(GLM), exp = sum(exposure), total_rate_stack = sum(STACK), total_clm = sum(ClaimAmount))
grp_by_Gender$avg_rate_glm <- grp_by_Gender$total_rate_glm / grp_by_Gender$exp
grp_by_Gender$avg_rate_stack <- grp_by_Gender$total_rate_stack / grp_by_Gender$exp
grp_by_Gender$avg_clm <- grp_by_Gender$total_clm / grp_by_Gender$exp

plot_Gender <- plot_ly(data = grp_by_Gender, x = ~Gender, y = ~avg_clm, type = "scatter", mode = "lines", name = "BC") %>%
  add_trace(y = ~avg_rate_stack, mode = "lines", name = "STACK APPROACH") %>%
  add_trace(y = ~avg_rate_glm, mode = "lines", name = "GLM")
plot_Gender

#One-Way Analysis against Burning Cost for Gender
grp_by_Garage <- test %>%
  group_by(Garage) %>%
  summarize(total_rate_glm = sum(GLM), exp = sum(exposure), total_rate_stack = sum(STACK), total_clm = sum(ClaimAmount))
grp_by_Garage$avg_rate_glm <- grp_by_Garage$total_rate_glm / grp_by_Garage$exp
grp_by_Garage$avg_rate_stack <- grp_by_Garage$total_rate_stack / grp_by_Garage$exp
grp_by_Garage$avg_clm <- grp_by_Garage$total_clm / grp_by_Garage$exp

plot_Garage <- plot_ly(data = grp_by_Garage, x = ~Garage, y = ~avg_clm, type = "scatter", mode = "lines", name = "BC") %>%
  add_trace(y = ~avg_rate_stack, mode = "lines", name = "STACK APPROACH") %>%
  add_trace(y = ~avg_rate_glm, mode = "lines", name = "GLM")
plot_Garage

#One-Way Analysis against Burning Cost for License Age
grp_by_LicAge <- test %>%
  group_by(LicAge) %>%
  summarize(total_rate_glm = sum(GLM), exp = sum(exposure), total_rate_stack = sum(STACK), total_clm = sum(ClaimAmount))
grp_by_LicAge$avg_rate_glm <- grp_by_LicAge$total_rate_glm / grp_by_LicAge$exp
grp_by_LicAge$avg_rate_stack <- grp_by_LicAge$total_rate_stack / grp_by_LicAge$exp
grp_by_LicAge$avg_clm <- grp_by_LicAge$total_clm / grp_by_LicAge$exp

plot_LicAge <- plot_ly(data = grp_by_LicAge, x = ~LicAge, y = ~avg_clm, type = "scatter", mode = "lines", name = "BC") %>%
  add_trace(y = ~avg_rate_stack, mode = "lines", name = "STACK APPROACH") %>%
  add_trace(y = ~avg_rate_glm, mode = "lines", name = "GLM")
plot_LicAge

#One-Way Analysis against Burning Cost for Vehicle Age
grp_by_VehAge <- test %>%
  group_by(VehAge) %>%
  summarize(total_rate_glm = sum(GLM), exp = sum(exposure), total_rate_stack = sum(STACK), total_clm = sum(ClaimAmount))
grp_by_VehAge$avg_rate_glm <- grp_by_VehAge$total_rate_glm / grp_by_VehAge$exp
grp_by_VehAge$avg_rate_stack <- grp_by_VehAge$total_rate_stack / grp_by_VehAge$exp
grp_by_VehAge$avg_clm <- grp_by_VehAge$total_clm / grp_by_VehAge$exp

plot_VehAge <- plot_ly(data = grp_by_VehAge, x = ~VehAge, y = ~avg_clm, type = "scatter", mode = "lines", name = "BC") %>%
  add_trace(y = ~avg_rate_stack, mode = "lines", name = "STACK APPROACH") %>%
  add_trace(y = ~avg_rate_glm, mode = "lines", name = "GLM")
plot_VehAge

##=======INTERPRETABILITY==========##
#Trim out Dead Wood first
rm(glm, grp_by_Garage, grp_by_Gender, grp_by_LicAge, grp_by_VehAge, grp_by_VehBody, l1_predictions, params, plot_Garage, plot_Gender, plot_LicAge, plot_VehAge,
   predictions_mlp1_l1, predictions_mlp2_l1, predictions_mlp3_l1, predictions_mlp4_l1, test_nn, tr_control, train_nn, train_stack, train_stack_bal, val_nn, val_stack,
   x_test, x_train, x_val, xgboost, y_test, y_train, y_val, l1_rmse, mlp1, mlp2, mlp3, mlp4, ord, p, predictions_glm, predictions_l2, predictions_rf_l1,
   predictions_xgb_l2, rmse_1, rmse_2, rmse_3, rmse_4, rmse_5, soft_sum, top_15_socio_categ, weights, xgb_train, predictors, 
   plot_VehBody, targetvar, dat_nn, fit)

#Set up Model Formula
targetvar <- c("STACK")
predictors <- c("LicAge", "VehAge", "Gender", "MariStat", "VehUsage", "DrivAge", "HasKmLimit", "BonusMalus", "VehBody", "VehPriceGrp", "VehEngine",
                "VehEnergy", "VehMaxSpeed", "VehClass", "Garage", "SocioCateg")
predictors <- paste(predictors, collapse = "+")
formula <- as.formula(paste(targetvar,"~",predictors,collapse = "+"))

#RF Surrogate Model
rf_surrogate <- ranger(formula, data = test, num.trees = 2000, importance = "permutation", 
                       verbose = TRUE, mtry = round(length(predictors) / 3))
test$RF_SURROGATE <- as.numeric(predict(rf_surrogate, test)$predictions)

#Evaluate Surrogate using R-Squared
r_sq <- function(x, y) {
  r_sq <- cor(x, y)^2
  return(r_sq)
}

r_sq(test$STACK, test$RF_SURROGATE)*100

#Feature Importance using VIP
features <- c("LicAge", "VehAge", "Gender", "MariStat", "VehUsage", "DrivAge", "HasKmLimit", "BonusMalus", "VehBody", "VehPriceGrp", "VehEngine",
              "VehEnergy", "VehMaxSpeed", "VehClass", "Garage", "SocioCateg")
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
cl <- makePSOCKcluster(detectCores() - 4)
registerDoParallel(cl)
ptm <- proc.time()
imp_iml <- FeatureImp$new(predictor = predictor, loss = "mae", compare = "difference", parallel = TRUE)
proc.time() - ptm
stopCluster(cl)
plot(imp_iml)
rm(cl)
rm(ptm)

#Permutation RMSE-Based Feature Importance
explainer_rf_surrogate <- explain(rf_surrogate, data = test[, c(features)], predict_function = pred, y = as.numeric(test$RF_SURROGATE))
ptm <- proc.time()
imp <- variable_importance(explainer_rf_surrogate, loss_function = loss_root_mean_square, n_sample = -1)
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

vint <- function(object, feature_names, progress = "none", parallel = FALSE,
                 paropts = NULL) {
  warning("This function is experimental, use at your own risk!", call. = FALSE)
  all.pairs <- utils::combn(feature_names, m = 2)
  clusterExport(cl=cl, varlist=c("rf_surrogate","test"), envir=environment()) # added
  ints <- plyr::aaply(
    all.pairs, .margins = 2, .progress = progress, .parallel = parallel,
    .paropts = list(.packages="ranger"),  # added
    .fun = function(x) {
      pd <- pdp::partial(object, pred.var = x)
      mean(c(
        stats::sd(tapply(pd$yhat, INDEX = pd[[x[1L]]], FUN = stats::sd)),
        stats::sd(tapply(pd$yhat, INDEX = pd[[x[2L]]], FUN = stats::sd))
      ))
    })
  ints <- data.frame(
    "Variables" = paste0(all.pairs[1L, ], "*", all.pairs[2L, ]),
    "Interaction" = ints
  )
  ints <- ints[order(ints["Interaction"], decreasing = TRUE), ]
  tibble::as_tibble(ints)
}

cl <- makePSOCKcluster(detectCores() - 4)
registerDoParallel(cl)
int <- c()
for (i in 1:length(int_terms)) {
  int[i] <- as.numeric(vint(rf_surrogate, feature_names = int_terms[[i]], parallel = TRUE)[, 2])
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
random_row_from_test <- test[which.max(test$STACK - test$ClaimAmount), ]
lime <- LocalModel$new(predictor, k = length(features), x.interest = random_row_from_test) %>% plot()
lime

#Trying to Enhance existing GLM using above results
targetvar <- c("ClaimAmount")
predictors <- c("VehAge", "VehMaxSpeed", "DrivAge", "VehClass", "Gender", "SocioCateg", "BonusMalus", "VehPriceGrp", "LicAge",
                "BonusMalus:VehPriceGrp", "BonusMalus:VehBody", "DrivAge:VehBody", "Gender:DrivAge")
predictors <- paste(predictors, collapse = "+")
formula <- as.formula(paste(targetvar,"~",predictors,collapse = "+"))

p_values <- seq(1.2, 1.65, length = 7)
p_tuning <- tweedie.profile(formula = formula, data = train, p.vec = p_values, do.plot = FALSE, 
                            offset = log(exposure), verbose = 2, method = "series")
p <- p_tuning$p.max
rm(p_values)
rm(p_tuning)

glm2 <- glm(formula = formula, data = train, family = tweedie(var.power = p, link.power = 0), offset = log(exposure))
rm(formula)
rm(predictors)

predictions_glm2 <- as.numeric(predict(glm2, test, type = "response"))

rmse_glm2 <- rmse(data = test, targetvar = targetvar, prediction.obj = predictions_glm2)
rmse_glm2

nrmse_glm2 <- NRMSE(predictions_glm2, test$ClaimAmount)
nrmse_glm2

summary(glm2)

rm(targetvar)



























































































































































































































































































































































































































































































































































































































































































































































































































































































































