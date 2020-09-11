library(ROSE)
library(MASS)
library(tweedie)
library(statmod)
library(xgboost)
library(dplyr)
library(plotly)
library(keras)
library(iml)
library(foreach)
library(doParallel)
library(data.table)
library(vip)
library(fastDummies)
library(DALEX)
library(splitstackshape)
library(processx)
library(ranger)
library(caret)
library(magrittr)
library(pracma)
library(tidyr)
library(readr)
library(ingredients)
library(flashlight)
library(pryr)
library(gtools)

factor_levels_check <- function(data, factor_variable) {
  n_levels <- length(unique(as.character(unlist(data[, factor_variable]))))
  return(paste(factor_variable, " : ", n_levels))
  rm(n_levels)
}

reorder_factor_levels <- function(data, factor_variable) {
  ord <- data %>%
    group_by(get(factor_variable)) %>%
    summarise(exp = sum(exposure)) %>%
    arrange(desc(exp))
  ord <- as.data.frame(ord)
  colnames(ord) <- c(factor_variable, "exposure")
  ord <- as.character(unlist(ord[, 1]))
  var <- unlist(data[, factor_variable])
  reordered_factor <- factor(var, ord)
  var <- reordered_factor
  data[, factor_variable] <- var
  return(data[, factor_variable])
  rm(ord)
  rm(reordered_factor)
  rm(var)
}

ascii_reorder <- function(data, factor_variable) {
  fac <- unique(as.character(data[[factor_variable]]))
  ord <- mixedsort(fac)
  fac <- factor(as.character(data[[factor_variable]]), ord)
  data[[factor_variable]] <- fac
  return(data[[factor_variable]])
  rm(ord)
  rm(fac)
}

rmse <- function(data, targetvar, prediction.obj) {
  rss <- (as.numeric(prediction.obj) - as.numeric(unlist(data[, c(targetvar)]))) ^ 2
  mse <- sum(rss) / nrow(data)
  rmse <- sqrt(mse)
  rmse <- rmse / 100
  return(rmse)
} #Use utils

NRMSE <- function(data, targetvar, prediction.obj) {
  obs <- as.numeric(data[[targetvar]])
  pred <- as.numeric(prediction.obj)
  r <- rmse(data, targetvar, prediction.obj)
  nrmse <- r / (max(obs) - min(obs))
  return(nrmse)
} #Use utils

agg_rpr <- function(data, targetvar, prediction.obj) {
  tot_obs <- sum(as.numeric(data[[targetvar]]))
  tot_pred <- sum(as.numeric(prediction.obj))
  rpr <- tot_obs / tot_pred
  return(rpr)
}

norm_rp_deviance <- function(data, targetvar, prediction.obj) {
  obs <- as.numeric(data[[targetvar]])
  pred <- as.numeric(prediction.obj)
  mean_obs <- mean(obs)
  mean_pred <- mean(pred)
  rpd <- mean_obs / mean_pred
  rpd <- 1 - abs(1 - rpd)
  return(rpd)
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

text_project <- element_text(size = 12)

theme_project <- theme(
  axis.title = text_project
  , legend.text = text_project
  , legend.position = 'bottom'
  , plot.title = element_text(size = 16, face = "bold")
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
    xlab("Exposure") + ggtitle("Model Gini Index Plot") +
    ylab("Predicted Loss Cost") + theme_project
}

lift_curve <- function(predicted_loss_cost, observed_loss_cost, exposure, n) {
  dataset <- tibble(
    pred_lc = predicted_loss_cost
    , obs_lc = observed_loss_cost
    , exp = exposure
  )
  dataset <- dataset %>%
    arrange(pred_lc) %>%
    mutate(buckets = ntile(pred_lc, n)) %>% 
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
  dataset <- dataset %>%
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
  x <- list(title = "Bucket")
  y <- list(title = "Average Relative Risk Premium")
  lift_curve <- plot_ly(dataset, x = ~buckets) %>%
    add_lines(y = ~Predicted_Risk_Premium, name = "Predicted Risk Premium") %>%
    add_lines(y = ~Observed_Risk_Premium, name = "Observed Risk Premium") %>%
    layout(title = "GLM Sample Decile Curve", xaxis = x, yaxis = y)
  return(lift_curve)
} #Use utils

mean_pseudohuber_loss <- function(data, targetvar, prediction.obj, delta = 1.0) {
  error <- data[[targetvar]] - prediction.obj
  pseudohuber_loss <- (delta ^ 2) * ((sqrt(1 + ((error / delta) ^ 2))) - 1)
  mphe <- sum(pseudohuber_loss) / nrow(data)
  return(mphe)
}

mean_poisson_deviance_loss <- function(data, targetvar, prediction_obj) {
  obs <- as.numeric(data[[targetvar]])
  pred <- as.numeric(prediction_obj)
  pd <- (pred - obs) + (log((obs / pred) ^ obs))
  agg_pd <- 2 * sum(pd)
  mean_pd_loss <- agg_pd / nrow(data)
  return(mean_pd_loss)
}

#extract_levels <- function(data, factor_name) {
  return(levels(data[[factor_name]]))
}

one_way_prep <- function(data, predictors, targetvar, prediction_obj) {
  groups <- list()
  n_predictors <- length(predictors)
  data$predictions <- prediction_obj
  for(i in 1:n_predictors) {
    one_way_table <- data %>%
      group_by(get(predictors[i])) %>%
      summarise(actuals = sum(get(targetvar)) / sum(exposure), expected = sum(predictions) / sum(exposure), exposure = sum(exposure))
    one_way_table$variable <- predictors[i]
    groups[[i]] <- one_way_table
  }
  groups <- do.call(rbind, groups)
  colnames(groups)[1] <- c("level")
  return(groups)
}

mean_tweedie_deviance <- function(data, targetvar, prediction_obj, var_power) {
  obs <- as.numeric(data[[targetvar]])
  pred <- as.numeric(prediction_obj)
  p1 <- 1 - var_power
  p2 <- 2 - var_power
  t1 <- (obs ^ p2) / (p1 * p2)
  t2 <- (obs * (pred ^ p1)) / p1
  t3 <- (pred ^ p2) / p2
  tweedie_deviance <- 2 * (t1 - t2 + t3)
  mean_tweedie_deviance <- sum(tweedie_deviance) / nrow(data)
  return(mean_tweedie_deviance)
}

prep_xgb <- function(data, predictors) {
  data <- data[, predictors]
  data <- dummy_cols(data, predictors)
  predictors <- setdiff(colnames(data), predictors)
  return(as.matrix(data[, predictors]))
}

#get_sample_weights <- function(data, weight_column) {
  tab <- as.data.frame(table(data[[weight_column]]))
  total <- sum(tab$Freq)
  tab <- tab %>%
    mutate(sample_weight = Freq / total) %>%
    select(-Freq)
  tab$Var1 <- as.numeric(as.character(tab$Var1))
  colnames(tab)[1] <- weight_column
  tab$sample_weight <- 1 - tab$sample_weight
  data <- inner_join(data, tab, by = weight_column)
  sample_weights <- data %>%
    select(sample_weight)
  sample_weights <- as.numeric(unlist(sample_weights))
  return(sample_weights)
}

mean_huber_loss <- function(y_true, y_pred, delta = 1.0) {
  K <- k_backend()
  error <- y_true - y_pred
  condition <- k_abs(error) <= delta
  condition <- k_cast(condition, "float32")
  squared_loss <- 0.5 * k_square(error)
  squared_loss <- k_cast(squared_loss, "float32")
  linear_loss <- delta * (k_abs(error)  - (0.5 * delta))
  linear_loss <- k_cast(linear_loss, "float32")
  loss <- (condition * squared_loss) + ((1 - condition) * linear_loss)
  loss <- k_cast(loss, "float32")
  mean_loss <- k_mean(loss)
  return(mean_loss)
}

loss_huber <- function(y_true, y_pred) {
  mean_huber_loss(y_true, y_pred, delta = d)
}

tweedie_deviance_loss <- function(y_true, y_pred, var_power) {
  K <- k_backend()
  p1 <- 1 - var_power
  p2 <- 2 - var_power
  t1 <- (y_true ^ p2) / (p1 * p2)
  t2 <- (y_true * (y_pred ^ p1)) / p1
  t3 <- (y_pred ^ p2) / p2
  tweedie_deviance <- 2 * (t1 - t2 + t3)
  tweedie_deviance <- k_cast(tweedie_deviance, "float32")
  mean_tweedie_deviance <- k_mean(tweedie_deviance)
  return(mean_tweedie_deviance)
}

loss_tweedie <- function(y_true, y_pred) {
  tweedie_deviance_loss(y_true, y_pred, var_power = p)
}

gamma_deviance_loss <- function(y_true, y_pred) {
  K <- k_backend()
  t1 <- k_log((y_pred / y_true))
  t2 <- (y_true / y_pred)
  gamma_deviance <- 2 * (t1 + t2 - 1)
  gamma_deviance <- k_cast(gamma_deviance, "float32")
  mean_gamma_deviance <- k_mean(gamma_deviance)
  return(mean_gamma_deviance)
}

mean_gamma_deviance_loss <- function(data, targetvar, prediction_obj) {
  obs <- as.numeric(data[[targetvar]])
  pred <- as.numeric(prediction_obj)
  gd <- 2 * (log((pred / obs)) + (obs / pred) - 1)
  mean_gd <- sum(gd) / nrow(data)
  return(mean_gd)
}

loss_gamma <- function(y_true, y_pred) {
  gamma_deviance_loss(y_true, y_pred)
}

NMAE <- function(pred, obs) {
  MAE(pred, obs) / (max(obs) - min(obs))
}

harmonic_mean <- function(x) {
  1 / mean(1 / x)
}

geometric_mean <- function(x) {
  prod(x) ^ (1 / length(x))
}

train <- read_csv("train_new_final.csv")
train$dtype <- "train"

test <- read_csv("test_new_final.csv")
test$dtype <- "test"

dat <- rbind(train, test)

colnames(dat)

colnames(dat)[which(colnames(dat) %in% c("ClaimAmount"))] <- c("ClaimAmount")
colnames(dat)[which(colnames(dat) %in% c("ClaimNb"))] <- c("ClaimCount")
colnames(dat)[which(colnames(dat) %in% c("Exposure"))] <- c("exposure")
dat <- dat %>%
  mutate(severity = ifelse(ClaimCount == 0, 0, ClaimAmount / ClaimCount))

dat <- dat[dat$exposure > 0, ]
dat <- dat[dat$ClaimCount >= 0, ]
dat <- dat[dat$ClaimAmount >= 0, ]
dat <- dat[dat$severity >= 0, ]

cat_vars <- dat %>%
  select_if(~class(.) %in% c("character", "factor")) %>%
  colnames()

tab <- list()
for(i in 1:length(cat_vars)) {
  tab[[i]] <- table(dat[[cat_vars[i]]])
}
names(tab) <- cat_vars

tab

matrix(colnames(dat), ncol = 1)
dat <- dat[, c(seq(1, 8, by = 1), seq(10, 16, by = 1))]
dat <- dat[complete.cases(dat), ]

dat$VehPower <- as.factor(dat$VehPower)

dat <- dat %>%
  mutate_if(is.character, as.factor)

factors <- c()
for (i in 1:ncol(dat)) {
  if (class(unlist(dat[, i])) == "factor") {
    factors[i] <- colnames(dat)[i]
  } else {
    i <- i + 1
  }
}
factors <- factors[!factors %in% c(NA, "dtype")]
rm(i)

factor_levels_output <- sapply(factors, factor_levels_check, data = dat)
names(factor_levels_output) <- NULL
factor_levels_output

factors <- factors[factors %in% colnames(dat)]

for (i in 1:length(factors)) {
  dat[, factors[i]] <- reorder_factor_levels(data = dat, factor_variable = factors[i])
}

dat <- dat %>%
  mutate(BonusMalus = BonusMalus / 100, total_offset = exposure * BonusMalus)

rm(factors, i, factor_levels_output, factor_levels_check, reorder_factor_levels, tab, cat_vars)

#Split into Train and Test
train <- dat %>%
  filter(dtype == "train") %>%
  select(-dtype)

test <- dat %>%
  filter(dtype == "test") %>%
  select(-dtype)

##========BASELINE GLM==========##
targetvar <- c("ClaimCount")
exposure <- c("total_offset")
excl_vars <- c("ClaimAmount", "severity", "RecordID", "BonusMalus", "exposure", "dtype")
predictors <- colnames(dat)[!colnames(dat) %in% c(targetvar, exposure, excl_vars)]
predictors <- paste(predictors, collapse = "+")
formula <- as.formula(paste(targetvar,"~",predictors,collapse = "+"))

glm_freq <- glm(formula = formula, data = train, family = poisson(link = "log"), offset = log(exposure), 
                weight = rep(1, nrow(train)))
rm(formula)
rm(predictors)
rm(targetvar)
rm(excl_vars)

summary(glm_freq)
predictions_glm_freq <- as.numeric(predict(glm_freq, test, type = "response"))

targetvar <- c("severity")
weight <- c("ClaimCount")
excl_vars <- c("ClaimAmount", "exposure", "dtype", "BonusMalus", "total_offset", "RecordID")
predictors <- colnames(dat)[!colnames(dat) %in% c(targetvar, weight, excl_vars)]
predictors <- paste(predictors, collapse = "+")
formula <- as.formula(paste(targetvar,"~",predictors,collapse = "+"))

train_sev <- train %>%
  filter(severity > 0)

glm_sev <- glm(formula = formula, data = train_sev, family = Gamma(link = "log"), weight = get(weight))
rm(formula)
rm(predictors)
rm(targetvar)
rm(excl_vars)

summary(glm_sev)
predictions_glm_sev <- as.numeric(predict(glm_sev, test, type = "response"))

rm(train_sev)

targetvar <- c("ClaimAmount")
exposure <- c("total_offset")
excl_vars <- c("ClaimCount", "severity", "dtype", "BonusMalus", "exposure", "RecordID")
predictors <- colnames(dat)[!colnames(dat) %in% c(targetvar, exposure, excl_vars)]
predictors <- paste(predictors, collapse = "+")
formula <- as.formula(paste(targetvar,"~",predictors,collapse = "+"))

p_values <- seq(1.3, 1.6, length = 8)
p_tuning <- tweedie.profile(formula = as.formula(paste0(targetvar,"~",1)), data = train, p.vec = p_values, do.plot = FALSE, 
                            verbose = 2, do.smooth = TRUE, method = "series", fit.glm = FALSE)
p <- p_tuning$p.max
rm(p_values)
rm(p_tuning)

glm_lc <- glm(formula = formula, data = train, family = tweedie(var.power = p, link.power = 0), offset = log(exposure),
              weight = rep(1, nrow(train)))
rm(formula)
rm(predictors)

summary(glm_lc)
predictions_glm_lc <- as.numeric(predict(glm_lc, test, type = "response"))

##=====FREQUENCY DATA PREP FOR ML=======##
approach_used <- 1 #Pick 0 for Single-Model Approach and 1 for Stack Approach

#Split Training dataset and Process
set.seed(797)
ind <- createDataPartition(train$ClaimCount, p = 0.5, list = FALSE, times = 1)
train_stack <- train[-ind, ]
val_stack <- train[ind, ]
rm(ind)
mean(train_stack$ClaimCount); mean(val_stack$ClaimCount); var(train_stack$ClaimCount); var(val_stack$ClaimCount)

targetvar <- c("ClaimCount")
exposure <- c("total_offset")
excl_vars <- c("ClaimAmount", "severity", "dtype", "BonusMalus", "exposure", "RecordID")
predictors <- colnames(dat)[!colnames(dat) %in% c(targetvar, exposure, excl_vars)]

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
train_stack$ClaimInd <- ifelse(train_stack$ClaimAmount == 0, 0, 1)
table(train_stack$ClaimInd) #Without Resampling
mean(train_stack$ClaimInd == 1)*100 #Percentage of Records with a Claim
t <- c("ClaimInd")
pp <- c(predictors, targetvar, exposure)
pp <- paste(pp, collapse = "+")
f <- as.formula(paste(t,"~",pp,collapse = "+"))
set.seed(134567)
train_stack_bal <- ovun.sample(f, train_stack, p = 0.25, method = "both")$data
train_stack_bal <- train_stack
table(train_stack_bal$ClaimInd) #After Resampling
rm(pp)
rm(f)
train_stack_bal <- train_stack_bal[, -which(colnames(train_stack_bal) == t)]

#One-Hot Encoding
dat_nn <- dummy_cols(dat, select_columns = factors)
train_nn <- dummy_cols(train_stack_bal, select_columns = factors)
val_nn <- dummy_cols(val_stack, select_columns = factors)
test_nn <- dummy_cols(test, select_columns = factors)

allvars <- colnames(dat_nn)
predictors_nn <- allvars[!allvars %in% c(targetvar, factors, excl_vars)]
rm(allvars)
rm(t)
rm(factors)
rm(excl_vars)

dat_nn <- dat_nn[, c(predictors_nn, targetvar)]
train_nn <- train_nn[, c(predictors_nn, targetvar)]
val_nn <- val_nn[, c(predictors_nn, targetvar)]
test_nn <- test_nn[, c(predictors_nn, targetvar)]

#Scale Output Variable
#max_target <- max(dat_nn[[targetvar]])
#min_target <- min(dat_nn[[targetvar]])

#train_nn[, targetvar] <- scale(train_nn[, targetvar], center = min_target, scale = (max_target - min_target))
#val_nn[, targetvar] <- scale(val_nn[, targetvar], center = min_target, scale = (max_target - min_target))
#test_nn[, targetvar] <- scale(test_nn[, targetvar], center = min_target, scale = (max_target - min_target))

predictors_nn <- predictors_nn[!predictors_nn %in% c(exposure)]

if(approach_used == 1) {
  x_train <- as.matrix(train_nn[predictors_nn])
  y_train <- as.matrix(train_nn[targetvar])
  o_train <- as.matrix(log(train_nn[[exposure]]))
  x_val <- as.matrix(val_nn[predictors_nn])
  y_val <- as.matrix(val_nn[targetvar])
  o_val <- as.matrix(log(val_nn[[exposure]]))
  x_test <- as.matrix(test_nn[predictors_nn])
  y_test <- as.matrix(test_nn[targetvar])
  o_test <- as.matrix(log(test_nn[[exposure]]))
} else {
  train_nn <- rbind(train_nn, val_nn)
  x_train <- as.matrix(train_nn[predictors_nn])
  y_train <- as.matrix(train_nn[targetvar])
  o_train <- as.matrix(log(train_nn[[exposure]]))
  x_val <- as.matrix(test_nn[predictors_nn])
  y_val <- as.matrix(test_nn[targetvar])
  o_val <- as.matrix(log(test_nn[[exposure]]))
}

#Setup Neural Network Input Parameters
input_data <- layer_input(shape = ncol(x_train), dtype = "float32", name = "x_data")
offset <- layer_input(shape = c(1), dtype = "float32", name = "offset")

##==========FREQUENCY MODELLING==========##
selected_model <- c("mlp3") #Only Applicable if Modelling Approach is set to 0 - Can be one of mlp1, mlp2, mlp3, mlp4 or gbm

if(approach_used == 1) {
  set.seed(14578)
  network <- input_data %>%
    layer_dense(units = 20, activation = "tanh", name = "h2") %>%
    layer_dropout(0.2) %>%
    layer_dense(units = 10, activation = "tanh", name = "h3") %>%
    layer_dropout(0.1) %>%
    layer_dense(units = 1, activation = "linear", name = "network",
                weights = list(array(0, dim = c(10, 1)), 
                               array(log(sum(as.numeric(y_train)) / sum(train_nn[[exposure]])), 
                                     dim = c(1))))
  response <- list(network, offset) %>% layer_add(name = "add") %>%
    layer_dense(1, activation = k_exp, name = "response", trainable = FALSE, 
                weights = list(array(1, dim = c(1, 1)), array(0, dim = c(1))))
  mlp1 <- keras_model(inputs = c(input_data, offset), outputs = c(response))
  mlp1 <- multi_gpu_model(mlp1, gpus = 2)
  mlp1 %>%
    compile(loss = "poisson", optimizer = optimizer_adam())
  fit <- fit(mlp1, list(x_train, o_train), y = y_train, validation_split = 0.1, sample_weights = rep(1, nrow(x_train)),
             batch_size = 0.2 * nrow(x_train), epochs = 10000, verbose = 1, shuffle = TRUE,
             callbacks = list(callback_early_stopping(monitor = "val_loss", min_delta = 1e-06, 
                                                      patience = 500, verbose = 1, 
                                                      restore_best_weights = TRUE)))
  predictions_mlp1_l1 <- mlp1 %>% predict(list(x_val, o_val))
  #predictions_mlp1_l1_unscaled <- as.numeric((predictions_mlp1_l1 * (max_target - min_target)) + min_target)
  
  set.seed(1234777)
  network <- input_data %>%
    layer_dense(units = 20, activation = "tanh", name = "h1") %>%
    layer_dropout(0.25) %>%
    layer_dense(units = 15, activation = "tanh", name = "h2") %>%
    layer_dropout(0.2) %>%
    layer_dense(units = 4, activation = "tanh", name = "h3") %>%
    layer_dense(units = 1, activation = "linear", name = "network", 
                weights = list(array(0, dim = c(4, 1)), 
                               array(log(sum(as.numeric(y_train)) / sum(train_nn[[exposure]])), 
                                     dim = c(1))))
  response <- list(network, offset) %>% layer_add(name = "add") %>%
    layer_dense(1, activation = k_exp, name = "response", trainable = FALSE, 
                weights = list(array(1, dim = c(1, 1)), array(0, dim = c(1))))
  mlp2 <- keras_model(inputs = c(input_data, offset), outputs = c(response))
  mlp2 <- multi_gpu_model(mlp2, gpus = 2)
  mlp2 %>%
    compile(loss = "poisson", optimizer = optimizer_adam())
  fit <- fit(mlp2, list(x_train, o_train), y = y_train, validation_split = 0.1, sample_weights = rep(1, nrow(x_train)),
             batch_size = 0.2 * nrow(x_train), epochs = 10000, verbose = 1, shuffle = TRUE,
             callbacks = list(callback_early_stopping(monitor = "val_loss", min_delta = 1e-06, 
                                                      patience = 500, verbose = 1,
                                                      restore_best_weights = TRUE)))
  predictions_mlp2_l1 <- mlp2 %>% predict(list(x_val, o_val))
  #predictions_mlp2_l1_unscaled <- as.numeric((predictions_mlp2_l1 * (max_target - min_target)) + min_target)
  
  set.seed(130)
  network <- input_data %>%
    layer_dense(units = 20, activation = "tanh", name = "h1") %>%
    layer_dropout(0.25) %>%
    layer_dense(units = 15, activation = "tanh", name = "h2") %>%
    layer_dropout(0.2) %>%
    layer_dense(units = 10, activation = "tanh", name = "h3") %>%
    layer_dropout(0.2) %>%
    layer_dense(units = 5, activation = "tanh", name = "h4") %>%
    layer_dense(units = 1, activation = "linear", name = "network",
                weights = list(array(0, dim = c(5, 1)), 
                               array(log(sum(as.numeric(y_train)) / sum(train_nn[[exposure]])), 
                                     dim = c(1))))
  response <- list(network, offset) %>% layer_add(name = "add") %>%
    layer_dense(1, activation = k_exp, name = "response", trainable = FALSE, 
                weights = list(array(1, dim = c(1, 1)), array(0, dim = c(1))))
  mlp3 <- keras_model(inputs = c(input_data, offset), outputs = c(response))
  mlp3 <- multi_gpu_model(mlp3, gpus = 2)
  mlp3 %>%
    compile(loss = "poisson", optimizer = optimizer_adam())
  fit <- fit(mlp3, list(x_train, o_train), y = y_train, validation_split = 0.1, sample_weights = rep(1, nrow(x_train)),
             batch_size = 0.2 * nrow(x_train), epochs = 10000, verbose = 1, shuffle = TRUE,
             callbacks = list(callback_early_stopping(monitor = "val_loss", min_delta = 1e-06, 
                                                      patience = 500, verbose = 1,
                                                      restore_best_weights = TRUE)))
  predictions_mlp3_l1 <- mlp3 %>% predict(list(x_val, o_val))
  #predictions_mlp3_l1_unscaled <- as.numeric((predictions_mlp3_l1 * (max_target - min_target)) + min_target)
  
  set.seed(8755)
  network <- input_data %>%
    layer_dense(units = 40, activation = "tanh", name = "h1") %>%
    layer_dropout(0.4) %>%
    layer_dense(units = 25, activation = "tanh", name = "h2") %>%
    layer_dropout(0.2) %>%
    layer_dense(units = 10, activation = "tanh", name = "h3") %>%
    layer_dropout(0.2) %>%
    layer_dense(units = 1, activation = "linear", name = "network",
                weights = list(array(0, dim = c(10, 1)), 
                               array(log(sum(as.numeric(y_train)) / sum(train_nn[[exposure]])), 
                                     dim = c(1)))) 
  response <- list(network, offset) %>% layer_add(name = "add") %>%
    layer_dense(1, activation = k_exp, name = "response", trainable = FALSE, 
                weights = list(array(1, dim = c(1, 1)), array(0, dim = c(1))))
  mlp4 <- keras_model(inputs = c(input_data, offset), outputs = c(response))
  mlp4 <- multi_gpu_model(mlp4, gpus = 2)
  mlp4 %>%
    compile(loss = "poisson", optimizer = optimizer_adam())
  fit <- fit(mlp4, list(x_train, o_train), y = y_train, validation_split = 0.1, shuffle = TRUE,
             batch_size = 0.2 * nrow(x_train), epochs = 10000, verbose = 1, sample_weights = rep(1, nrow(x_train)),
             callbacks = list(callback_early_stopping(monitor = "val_loss", min_delta = 1e-06, 
                                                      patience = 500, verbose = 1,
                                                      restore_best_weights = TRUE)))
  predictions_mlp4_l1 <- mlp4 %>% predict(list(x_val, o_val))
  #predictions_mlp4_l1_unscaled <- as.numeric((predictions_mlp4_l1 * (max_target - min_target)) + min_target)
  
  #Pull in Level 1 Predictions and Append to Validation Set for Level 2 - Set weights for each model based on individual predictive accuracy
  rmse_1 <- mean_poisson_deviance_loss(val_stack, targetvar, predictions_mlp1_l1)
  rmse_2 <- mean_poisson_deviance_loss(val_stack, targetvar, predictions_mlp2_l1)
  rmse_3 <- mean_poisson_deviance_loss(val_stack, targetvar, predictions_mlp3_l1)
  rmse_4 <- mean_poisson_deviance_loss(val_stack, targetvar, predictions_mlp4_l1)
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
  xgb_train <- xgb.DMatrix(x_val, label = as.numeric(l1_predictions), info = list(base_margin = as.numeric(o_val)), weight = rep(1, nrow(x_val)))
  xgb_test <- xgb.DMatrix(x_test, label = as.numeric(y_test), info = list(base_margin = as.numeric(o_test)), weight = rep(1, nrow(x_test)))
  set.seed(874747)
  xgboost <- xgboost(objective = "count:poisson", data = xgb_train, nrounds = 100000, verbose = 1, max.depth = 5, 
                          eta = 0.2, early_stopping_rounds = 500)
  predictions_freq <- predict(xgboost, xgb_test)
  #predictions_freq <- (predictions_freq * (max_target - min_target)) + min_target
} else {
  if(selected_model == "mlp1") {
    set.seed(14578)
    network <- input_data %>%
      layer_dense(units = 20, activation = "tanh", name = "h2") %>%
      layer_dropout(0.2) %>%
      layer_dense(units = 10, activation = "tanh", name = "h3") %>%
      layer_dropout(0.1) %>%
      layer_dense(units = 1, activation = "linear", name = "network",
                  weights = list(array(0, dim = c(10, 1)), 
                                 array(log(sum(as.numeric(y_train)) / sum(train_nn[[exposure]])), 
                                       dim = c(1))))
    response <- list(network, offset) %>% layer_add(name = "add") %>%
      layer_dense(1, activation = k_exp, name = "response", trainable = FALSE, 
                  weights = list(array(1, dim = c(1, 1)), array(0, dim = c(1))))
    mlp <- keras_model(inputs = c(input_data, offset), outputs = c(response))
    mlp <- multi_gpu_model(mlp, gpus = 2)
    mlp %>%
      compile(loss = "poisson", optimizer = optimizer_adam())
    fit <- fit(mlp, list(x_train, o_train), y = y_train, validation_split = 0.1, sample_weights = rep(1, nrow(x_train)),
               batch_size = 0.2 * nrow(x_train), epochs = 10000, verbose = 1, shuffle = FALSE,
               callbacks = list(callback_early_stopping(monitor = "val_loss", min_delta = 1e-06, 
                                                        patience = 500, verbose = 1, 
                                                        restore_best_weights = TRUE)))
    predictions_freq <- mlp %>% predict(list(x_val, o_val))
    #predictions_freq <- as.numeric((predictions_freq * (max_target - min_target)) + min_target)
  } else 
    if(selected_model == "mlp2") {
      set.seed(1234777)
      network <- input_data %>%
        layer_dense(units = 20, activation = "tanh", name = "h1") %>%
        layer_dropout(0.25) %>%
        layer_dense(units = 15, activation = "tanh", name = "h2") %>%
        layer_dropout(0.2) %>%
        layer_dense(units = 4, activation = "tanh", name = "h3") %>%
        layer_dense(units = 1, activation = "linear", name = "network", 
                    weights = list(array(0, dim = c(4, 1)), 
                                   array(log(sum(as.numeric(y_train)) / sum(train_nn[[exposure]])), 
                                         dim = c(1))))
      response <- list(network, offset) %>% layer_add(name = "add") %>%
        layer_dense(1, activation = k_exp, name = "response", trainable = FALSE, 
                    weights = list(array(1, dim = c(1, 1)), array(0, dim = c(1))))
      mlp <- keras_model(inputs = c(input_data, offset), outputs = c(response))
      mlp <- multi_gpu_model(mlp, gpus = 2)
      mlp %>%
        compile(loss = "poisson", optimizer = optimizer_adam())
      fit <- fit(mlp, list(x_train, o_train), y = y_train, validation_split = 0.1, sample_weights = rep(1, nrow(x_train)),
                 batch_size = 0.2 * nrow(x_train), epochs = 10000, verbose = 1, shuffle = FALSE,
                 callbacks = list(callback_early_stopping(monitor = "val_loss", min_delta = 1e-06, 
                                                          patience = 500, verbose = 1, 
                                                          restore_best_weights = TRUE)))
      predictions_freq <- mlp %>% predict(list(x_val, o_val))
      #predictions_freq <- as.numeric((predictions_freq * (max_target - min_target)) + min_target)
  } else 
    if(selected_model == "mlp3") {
      network <- input_data %>%
        layer_dense(units = 20, activation = "tanh", name = "h1") %>%
        layer_dropout(0.25) %>%
        layer_dense(units = 15, activation = "tanh", name = "h2") %>%
        layer_dropout(0.2) %>%
        layer_dense(units = 10, activation = "tanh", name = "h3") %>%
        layer_dropout(0.2) %>%
        layer_dense(units = 5, activation = "tanh", name = "h4") %>%
        layer_dense(units = 1, activation = "linear", name = "network",
                    weights = list(array(0, dim = c(5, 1)), 
                                   array(log(sum(as.numeric(y_train)) / sum(train_nn[[exposure]])), 
                                         dim = c(1))))
    response <- list(network, offset) %>% layer_add(name = "add") %>%
      layer_dense(1, activation = k_exp, name = "response", trainable = FALSE, 
                  weights = list(array(1, dim = c(1, 1)), array(0, dim = c(1))))
    mlp <- keras_model(inputs = c(input_data, offset), outputs = c(response))
    mlp <- multi_gpu_model(mlp, gpus = 2)
    mlp %>%
      compile(loss = "poisson", optimizer = optimizer_adam())
    fit <- fit(mlp, list(x_train, o_train), y = y_train, validation_split = 0.1, sample_weights = rep(1, nrow(x_train)),
               batch_size = 0.2 * nrow(x_train), epochs = 10000, verbose = 1, shuffle = FALSE,
               callbacks = list(callback_early_stopping(monitor = "val_loss", min_delta = 1e-06, 
                                                        patience = 500, verbose = 1, 
                                                        restore_best_weights = TRUE)))
    predictions_freq <- mlp %>% predict(list(x_val, o_val))
    #predictions_freq <- as.numeric((predictions_freq * (max_target - min_target)) + min_target)
  } else 
    if (selected_model == "mlp4") {
    network <- input_data %>%
      network <- input_data %>%
        layer_dense(units = 40, activation = "tanh", name = "h1") %>%
        layer_dropout(0.4) %>%
        layer_dense(units = 25, activation = "tanh", name = "h2") %>%
        layer_dropout(0.2) %>%
        layer_dense(units = 10, activation = "tanh", name = "h3") %>%
        layer_dropout(0.2) %>%
        layer_dense(units = 1, activation = "linear", name = "network",
                    weights = list(array(0, dim = c(10, 1)), 
                                   array(log(sum(as.numeric(y_train)) / sum(train_nn[[exposure]])), 
                                         dim = c(1))))   
    response <- list(network, offset) %>% layer_add(name = "add") %>%
      layer_dense(1, activation = k_exp, name = "response", trainable = FALSE, 
                  weights = list(array(1, dim = c(1, 1)), array(0, dim = c(1))))
    mlp <- keras_model(inputs = c(input_data, offset), outputs = c(response))
    mlp <- multi_gpu_model(mlp, gpus = 2)
    mlp %>%
      compile(loss = "poisson", optimizer = optimizer_adam())
    fit <- fit(mlp, list(x_train, o_train), y = y_train, validation_split = 0.1, sample_weights = rep(1, nrow(x_train)),
               batch_size = 0.2 * nrow(x_train), epochs = 10000, verbose = 1, shuffle = FALSE,
               callbacks = list(callback_early_stopping(monitor = "val_loss", min_delta = 1e-06, 
                                                        patience = 500, verbose = 1, 
                                                        restore_best_weights = TRUE)))
    predictions_freq <- mlp %>% predict(list(x_val, o_val))
    #predictions_freq <- as.numeric((predictions_freq * (max_target - min_target)) + min_target)
    } else 
      if (selected_model == "gbm") {
        xgb_train <- xgb.DMatrix(x_train, label = as.numeric(y_train), info = list(base_margin = as.numeric(o_train), weight = rep(1, nrow(x_train))))
        xgb_test <- xgb.DMatrix(x_val, label = as.numeric(y_val), info = list(base_margin = as.numeric(o_val), weight = rep(1, nrow(x_val))))
        set.seed(874747)
        xgboost <- xgboost(objective = "count:poisson", data = xgb_train, nrounds = 100000, verbose = 1, max.depth = 5, 
                           eta = 0.2, early_stopping_rounds = 500)
        predictions_freq <- predict(xgboost, xgb_test)
        #predictions_freq <- (predictions_freq * (max_target - min_target)) + min_target
  }
  else {
    print("Selected Model is not within valid choices.")
  }
}

##=====SEVERITY DATA PREP FOR ML=======##
approach_used <- 0 #Pick 0 for Single-Model Approach and 1 for Stack Approach

train_sev <- train %>%
  filter(severity > 0)
#Split Training dataset and Process
set.seed(598)
ind <- createDataPartition(train_sev$severity, p = 0.5, list = FALSE, times = 1)
train_stack <- train_sev[-ind, ]
val_stack <- train_sev[ind, ]
rm(ind)
mean(train_stack$severity); mean(val_stack$severity); var(train_stack$severity); var(val_stack$severity)

targetvar <- c("severity")
exposure <- c("total_offset")
weight <- c("ClaimCount")
excl_vars <- c("ClaimAmount", "dtype", "BonusMalus", "exposure", "RecordID")
predictors <- colnames(dat)[!colnames(dat) %in% c(targetvar, exposure, excl_vars)]

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

#One-Hot Encoding
dat_nn <- dummy_cols(dat, select_columns = factors)
train_nn <- dummy_cols(train_stack, select_columns = factors)
val_nn <- dummy_cols(val_stack, select_columns = factors)
test_nn <- dummy_cols(test, select_columns = factors)

allvars <- colnames(dat_nn)
predictors_nn <- allvars[!allvars %in% c(targetvar, factors, exposure, excl_vars)]
rm(allvars)
rm(factors)
rm(excl_vars)

dat_nn <- dat_nn[, c(predictors_nn, targetvar)]
train_nn <- train_nn[, c(predictors_nn, targetvar)]
val_nn <- val_nn[, c(predictors_nn, targetvar)]
test_nn <- test_nn[, c(predictors_nn, targetvar)]

#Scale Output Variable
#max_target <- max(train_nn[[targetvar]])
#min_target <- min(train_nn[[targetvar]])

#train_nn[, targetvar] <- scale(train_nn[, targetvar], center = min_target, scale = (max_target - min_target))
#val_nn[, targetvar] <- scale(val_nn[, targetvar], center = min_target, scale = (max_target - min_target))
#test_nn[, targetvar] <- scale(test_nn[, targetvar], center = min_target, scale = (max_target - min_target))

predictors_nn <- predictors_nn[!predictors_nn %in% c(weight)]

d <- 1e-08 #To be specified if Huber Loss is used to optimize NNs for Severity

if(approach_used == 1) {
  x_train <- as.matrix(train_nn[predictors_nn])
  y_train <- as.matrix(train_nn[targetvar])
  x_val <- as.matrix(val_nn[predictors_nn])
  y_val <- as.matrix(val_nn[targetvar])
  x_test <- as.matrix(test_nn[predictors_nn])
  y_test <- as.matrix(test_nn[targetvar])
  #w_train <- get_sample_weights(train_stack, weight)
  #w_val <- get_sample_weights(val_stack, weight)
  #w_test <- get_sample_weights(test, weight)
  w_train <- as.numeric(train_nn[[weight]])
  w_val <- as.numeric(val_nn[[weight]])
  w_test <- as.numeric(test_nn[[weight]])
} else {
  train_nn <- rbind(train_nn, val_nn)
  x_train <- as.matrix(train_nn[predictors_nn])
  y_train <- as.matrix(train_nn[targetvar])
  x_val <- as.matrix(test_nn[predictors_nn])
  y_val <- as.matrix(test_nn[targetvar])
  #w_train <- get_sample_weights(rbind(train_stack, val_stack), weight)
  #w_val <- get_sample_weights(test, weight)
  w_train <- as.numeric(train_nn[[weight]])
  w_val <- as.numeric(test_nn[[weight]])
}

#Setup Neural Network Input Parameters
input_data <- layer_input(shape = ncol(x_train), dtype = "float32", name = "x_data")

##==========SEVERITY MODELLING==========##
selected_model <- c("gbm") #Only Applicable if Modelling Approach is set to 0 - Can be one of mlp1, mlp2, mlp3, mlp4 or gbm

if(approach_used == 1) {
  set.seed(14578)
  network <- input_data %>%
    layer_dense(units = 100, activation = "tanh", name = "h1") %>%
    layer_dense(units = 60, activation = "tanh", name = "h2") %>%
    layer_dense(units = 25, activation = "tanh", name = "h3") %>%
    layer_dense(units = 1, activation = "linear", name = "network")
  mlp1 <- keras_model(inputs = c(input_data), outputs = c(network))
  mlp1 <- multi_gpu_model(mlp1, gpus = 2)
  mlp1 %>%
    compile(loss = "mean_squared_logarithmic_error", optimizer = optimizer_adam())
  fit <- fit(mlp1, list(x_train), y = y_train, validation_split = 0.1, sample_weight = as.numeric(w_train),
             batch_size = 0.05 * nrow(x_train), epochs = 10000, verbose = 1, shuffle = FALSE,
             callbacks = list(callback_early_stopping(monitor = "val_loss", min_delta = 1e-08, 
                                                      patience = 500, verbose = 1, 
                                                      restore_best_weights = TRUE)))
  predictions_mlp1_l1 <- mlp1 %>% predict(list(x_val))
  #predictions_mlp1_l1_unscaled <- as.numeric((predictions_mlp1_l1 * (max_target - min_target)) + min_target)

  set.seed(1234777)
  network <- input_data %>%
    layer_dense(units = 50, activation = "tanh", name = "h1") %>%
    layer_dense(units = 25, activation = "tanh", name = "h2") %>%
    layer_dense(units = 1, activation = "linear", name = "network")
  mlp2 <- keras_model(inputs = c(input_data), outputs = c(network))
  mlp2 <- multi_gpu_model(mlp2, gpus = 2)
  mlp2 %>%
    compile(loss = "mean_squared_logarithmic_error", optimizer = optimizer_adam())
  fit <- fit(mlp2, list(x_train), y = y_train, validation_split = 0.1, shuffle = FALSE,
             batch_size = 0.05 * nrow(x_train), epochs = 10000, verbose = 1, sample_weight = as.numeric(w_train),
             callbacks = list(callback_early_stopping(monitor = "val_loss", min_delta = 1e-08, 
                                                      patience = 500, verbose = 1,
                                                      restore_best_weights = TRUE)))
  predictions_mlp2_l1 <- mlp2 %>% predict(list(x_val))
  #predictions_mlp2_l1_unscaled <- as.numeric((predictions_mlp2_l1 * (max_target - min_target)) + min_target)

  set.seed(130)
  network <- input_data %>%
    layer_dense(units = 200, activation = "tanh", name = "h1") %>%
    layer_dense(units = 150, activation = "tanh", name = "h2") %>%
    layer_dense(units = 100, activation = "tanh", name = "h3") %>%
    layer_dense(units = 40, activation = "tanh", name = "h4") %>%
    layer_dense(units = 1, activation = "linear", name = "network")
  mlp3 <- keras_model(inputs = c(input_data), outputs = c(network))
  mlp3 <- multi_gpu_model(mlp3, gpus = 2)
  mlp3 %>%
    compile(loss = "mean_squared_logarithmic_error", optimizer = optimizer_adam())
  fit <- fit(mlp3, list(x_train), y = y_train, validation_split = 0.1, shuffle = FALSE,
             batch_size = 0.05 * nrow(x_train), epochs = 10000, verbose = 1, sample_weight = as.numeric(w_train),
             callbacks = list(callback_early_stopping(monitor = "val_loss", min_delta = 1e-08, 
                                                      patience = 500, verbose = 1,
                                                      restore_best_weights = TRUE)))
  predictions_mlp3_l1 <- mlp3 %>% predict(list(x_val))
  #predictions_mlp3_l1_unscaled <- as.numeric((predictions_mlp3_l1 * (max_target - min_target)) + min_target)

  set.seed(8755)
  network <- input_data %>%
    layer_dense(units = 50, activation = "tanh", name = "h1") %>%
    layer_dense(units = 40, activation = "tanh", name = "h2") %>%
    layer_dense(units = 15, activation = "tanh", name = "h3") %>%
    layer_dense(units = 1, activation = "linear", name = "network")
  mlp4 <- keras_model(inputs = c(input_data), outputs = c(network))
  mlp4 <- multi_gpu_model(mlp4, gpus = 2)
  mlp4 %>%
    compile(loss = "mean_squared_logarithmic_error", optimizer = optimizer_adam())
  fit <- fit(mlp4, list(x_train), y = y_train, validation_split = 0.1, sample_weight = as.numeric(w_train),
             batch_size = 0.05 * nrow(x_train), epochs = 10000, verbose = 1, shuffle = FALSE,
             callbacks = list(callback_early_stopping(monitor = "val_loss", min_delta = 1e-08, 
                                                      patience = 500, verbose = 1,
                                                      restore_best_weights = TRUE)))
  predictions_mlp4_l1 <- mlp4 %>% predict(list(x_val))
  #predictions_mlp4_l1_unscaled <- as.numeric((predictions_mlp4_l1 * (max_target - min_target)) + min_target)

  #Pull in Level 1 Predictions and Append to Validation Set for Level 2 - Set weights for each model based on individual predictive accuracy
  #rmse_1 <- rmse(val_stack, targetvar, as.numeric(predictions_mlp1_l1_unscaled))
  #rmse_2 <- rmse(val_stack, targetvar, as.numeric(predictions_mlp2_l1_unscaled))
  #rmse_3 <- rmse(val_stack, targetvar, as.numeric(predictions_mlp3_l1_unscaled))
  #rmse_4 <- rmse(val_stack, targetvar, as.numeric(predictions_mlp4_l1_unscaled))
  rmse_1 <- mean_gamma_deviance_loss(val_stack, targetvar, as.numeric(predictions_mlp1_l1))
  rmse_2 <- mean_gamma_deviance_loss(val_stack, targetvar, as.numeric(predictions_mlp2_l1))
  rmse_3 <- mean_gamma_deviance_loss(val_stack, targetvar, as.numeric(predictions_mlp3_l1))
  rmse_4 <- mean_gamma_deviance_loss(val_stack, targetvar, as.numeric(predictions_mlp4_l1))
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
  xgb_train <- xgb.DMatrix(x_val, label = as.numeric(l1_predictions), weight = as.numeric(w_val))
  xgb_test <- xgb.DMatrix(x_test, label = as.numeric(y_test), weight = as.numeric(w_test))
  set.seed(874747)
  xgboost <- xgboost(objective = "reg:squaredlogerror", data = xgb_train, nrounds = 25000, verbose = 1, max.depth = 15, 
                         eta = 0.3, early_stopping_rounds = 1500)
  predictions_sev <- predict(xgboost, xgb_test)
  #predictions_sev <- (predictions_sev  * (max_target - min_target)) + min_target
} else {
  if(selected_model == "mlp1") {
    network <- input_data %>%
      layer_dense(units = 100, activation = "tanh", name = "h1") %>%
      layer_dense(units = 60, activation = "tanh", name = "h2") %>%
      layer_dense(units = 25, activation = "tanh", name = "h3") %>%
      layer_dense(units = 1, activation = "linear", name = "network")
    mlp <- keras_model(inputs = c(input_data), outputs = c(network))
    mlp <- multi_gpu_model(mlp, gpus = 2)
    mlp %>%
      compile(loss = "mean_squared_logarithmic_error", optimizer = optimizer_adam())
    fit <- fit(mlp, list(x_train), y = y_train, validation_split = 0.1, sample_weight = as.numeric(w_train),
               batch_size = 0.05 * nrow(x_train), epochs = 10000, verbose = 1, shuffle = FALSE,
               callbacks = list(callback_early_stopping(monitor = "val_loss", min_delta = 1e-08, 
                                                        patience = 500, verbose = 1, 
                                                        restore_best_weights = TRUE)))
    predictions_sev <- mlp %>% predict(list(x_val))
    #predictions_sev <- as.numeric((predictions_sev * (max_target - min_target)) + min_target)
  } else 
    if(selected_model == "mlp2") {
      set.seed(1234777)
      network <- input_data %>%
        layer_dense(units = 50, activation = "tanh", name = "h1") %>%
        layer_dense(units = 25, activation = "tanh", name = "h2") %>%
        layer_dense(units = 1, activation = "linear", name = "network")
      mlp <- keras_model(inputs = c(input_data), outputs = c(network))
      mlp <- multi_gpu_model(mlp, gpus = 2)
      mlp %>%
        compile(loss = "mean_squared_logarithmic_error", optimizer = optimizer_adam())
      fit <- fit(mlp, list(x_train), y = y_train, validation_split = 0.1, shuffle = FALSE,
                 batch_size = 0.05 * nrow(x_train), epochs = 10000, verbose = 1, sample_weight = as.numeric(w_train),
                 callbacks = list(callback_early_stopping(monitor = "val_loss", min_delta = 1e-08, 
                                                          patience = 500, verbose = 1,
                                                          restore_best_weights = TRUE)))
      predictions_sev <- mlp %>% predict(list(x_val))
      #predictions_sev <- as.numeric((predictions_sev * (max_target - min_target)) + min_target)
    } else 
      if(selected_model == "mlp3") {
        set.seed(130)
        network <- input_data %>%
          layer_dense(units = 200, activation = "tanh", name = "h1") %>%
          layer_dense(units = 150, activation = "tanh", name = "h2") %>%
          layer_dense(units = 100, activation = "tanh", name = "h3") %>%
          layer_dense(units = 40, activation = "tanh", name = "h4") %>%
          layer_dense(units = 1, activation = "linear", name = "network")
        mlp <- keras_model(inputs = c(input_data), outputs = c(network))
        mlp <- multi_gpu_model(mlp3, gpus = 2)
        mlp %>%
          compile(loss = "mean_squared_logarithmic_error", optimizer = optimizer_adam())
        fit <- fit(mlp, list(x_train), y = y_train, validation_split = 0.1, shuffle = FALSE,
                   batch_size = 0.05 * nrow(x_train), epochs = 10000, verbose = 1, sample_weight = as.numeric(w_train),
                   callbacks = list(callback_early_stopping(monitor = "val_loss", min_delta = 1e-08, 
                                                            patience = 500, verbose = 1,
                                                            restore_best_weights = TRUE)))
        predictions_sev <- mlp %>% predict(list(x_val))
        #predictions_sev <- as.numeric((predictions_sev * (max_target - min_target)) + min_target)
      } else 
        if(selected_model == "mlp4") {
    network <- input_data %>%
      layer_dense(units = 50, activation = "tanh", name = "h1") %>%
      layer_dense(units = 40, activation = "tanh", name = "h2") %>%
      layer_dense(units = 15, activation = "tanh", name = "h3") %>%
      layer_dense(units = 1, activation = "linear", name = "network")
    mlp <- keras_model(inputs = c(input_data), outputs = c(network))
    mlp <- multi_gpu_model(mlp4, gpus = 2)
    mlp %>%
      compile(loss = "mean_squared_logarithmic_error", optimizer = optimizer_adam())
    fit <- fit(mlp, list(x_train), y = y_train, validation_split = 0.1, sample_weight = as.numeric(w_train),
               batch_size = 0.05 * nrow(x_train), epochs = 10000, verbose = 1, shuffle = FALSE,
               callbacks = list(callback_early_stopping(monitor = "val_loss", min_delta = 1e-08, 
                                                        patience = 500, verbose = 1,
                                                        restore_best_weights = TRUE)))
    predictions_sev <- mlp %>% predict(list(x_val))
    #predictions_sev <- as.numeric((predictions_sev * (max_target - min_target)) + min_target)
        } else
          if(selected_model == "gbm") {
            xgb_train <- xgb.DMatrix(x_train, label = as.numeric(y_train), weight = as.numeric(w_train))
            xgb_test <- xgb.DMatrix(x_val, label = as.numeric(y_val), weight = as.numeric(w_val))
            set.seed(874747)
            xgboost <- xgboost(objective = "reg:gamma", data = xgb_train, nrounds = 100000, verbose = 1, max.depth = 5, 
                               eta = 0.1, early_stopping_rounds = 1000)
            predictions_sev <- predict(xgboost, xgb_test)
            #predictions_sev <- (predictions_sev  * (max_target - min_target)) + min_target
          }
  else {
    print("Selected Model is not within valid choices.")
  }
}

##=====PURE PREMIUM DATA PREP FOR ML=======##
approach_used <- 1 #Pick 0 for Single-Model Approach and 1 for Stack Approach

#Split Training dataset and Process
set.seed(8718)
ind <- createDataPartition(train$ClaimAmount, p = 0.5, list = FALSE, times = 1)
train_stack <- train[-ind, ]
val_stack <- train[ind, ]
rm(ind)
mean(train_stack$ClaimAmount); mean(val_stack$ClaimAmount); var(train_stack$ClaimAmount); var(val_stack$ClaimAmount)

targetvar <- c("ClaimAmount")
exposure <- c("total_offset")
weight <- c("ClaimCount")
excl_vars <- c("severity", "dtype", "BonusMalus", "exposure", "RecordID")
predictors <- colnames(dat)[!colnames(dat) %in% c(targetvar, exposure, weight, excl_vars)]

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
train_stack$ClaimInd <- ifelse(train_stack$ClaimAmount == 0, 0, 1)
table(train_stack$ClaimInd) #Without Resampling
mean(train_stack$ClaimInd == 1)*100 #Percentage of Records with a Claim
t <- c("ClaimInd")
pp <- c(predictors, targetvar, exposure)
pp <- paste(pp, collapse = "+")
f <- as.formula(paste(t,"~",pp,collapse = "+"))
set.seed(134567)
train_stack_bal <- ovun.sample(f, train_stack, p = 0.25, method = "both")$data
train_stack_bal <- train_stack
table(train_stack_bal$ClaimInd) #After Resampling
rm(pp)
rm(f)
train_stack_bal <- train_stack_bal[, -which(colnames(train_stack_bal) == t)]

#One-Hot Encoding
dat_nn <- dummy_cols(dat, select_columns = factors)
train_nn <- dummy_cols(train_stack_bal, select_columns = factors)
val_nn <- dummy_cols(val_stack, select_columns = factors)
test_nn <- dummy_cols(test, select_columns = factors)

allvars <- colnames(dat_nn)
predictors_nn <- allvars[!allvars %in% c(targetvar, factors, excl_vars)]
rm(allvars)
rm(t)
rm(factors)
rm(excl_vars)

dat_nn <- dat_nn[, c(predictors_nn, targetvar)]
train_nn <- train_nn[, c(predictors_nn, targetvar)]
val_nn <- val_nn[, c(predictors_nn, targetvar)]
test_nn <- test_nn[, c(predictors_nn, targetvar)]

#Scale Output Variable
#max_target <- max(dat_nn[[targetvar]])
#min_target <- min(dat_nn[[targetvar]])

#train_nn[, targetvar] <- scale(train_nn[, targetvar], center = min_target, scale = (max_target - min_target))
#val_nn[, targetvar] <- scale(val_nn[, targetvar], center = min_target, scale = (max_target - min_target))
#test_nn[, targetvar] <- scale(test_nn[, targetvar], center = min_target, scale = (max_target - min_target))

predictors_nn <- predictors_nn[!predictors_nn %in% c(exposure, weight)]

if(approach_used == 1) {
  x_train <- as.matrix(train_nn[predictors_nn])
  y_train <- as.matrix(train_nn[targetvar])
  o_train <- as.matrix(log(train_nn[[exposure]]))
  x_val <- as.matrix(val_nn[predictors_nn])
  y_val <- as.matrix(val_nn[targetvar])
  o_val <- as.matrix(log(val_nn[[exposure]]))
  x_test <- as.matrix(test_nn[predictors_nn])
  y_test <- as.matrix(test_nn[targetvar])
  o_test <- as.matrix(log(test_nn[[exposure]]))
} else {
  train_nn <- rbind(train_nn, val_nn)
  x_train <- as.matrix(train_nn[predictors_nn])
  y_train <- as.matrix(train_nn[targetvar])
  o_train <- as.matrix(log(train_nn[[exposure]]))
  x_val <- as.matrix(test_nn[predictors_nn])
  y_val <- as.matrix(test_nn[targetvar])
  o_val <- as.matrix(log(test_nn[[exposure]]))
}

#Setup Neural Network Input Parameters
input_data <- layer_input(shape = ncol(x_train), dtype = "float32", name = "x_data")
offset <- layer_input(shape = c(1), dtype = "float32", name = "offset")

##==========PURE PREMIUM MODELLING==========##
selected_model <- c("mlp1") #Only Applicable if Modelling Approach is set to 0 - Can be one of mlp1, mlp2, mlp3, mlp4 or gbm

if(approach_used == 1) {
  network <- input_data %>%
    layer_dense(units = 50, activation = "tanh", name = "h1") %>%
    layer_dropout(0.25) %>%
    layer_dense(units = 25, activation = "tanh", name = "h2") %>%
    layer_dropout(0.1) %>%
    layer_dense(units = 1, activation = "linear", name = "network",
                weights = list(array(0, dim = c(25, 1)), 
                               array(log(sum(as.numeric(y_train) * (train_nn[[exposure]] ^ (1 - p))) / sum(train_nn[[exposure]] ^ (2 - p))), 
                                     dim = c(1))))
  response <- list(network, offset) %>% layer_add(name = "add") %>%
    layer_dense(1, activation = k_exp, name = "response", trainable = FALSE, 
                weights = list(array(1, dim = c(1, 1)), array(0, dim = c(1))))
  mlp1 <- keras_model(inputs = c(input_data, offset), outputs = c(response))
  mlp1 <- multi_gpu_model(mlp1, gpus = 2)
  mlp1 %>%
    compile(loss = loss_tweedie, optimizer = optimizer_adam())
  fit <- fit(mlp1, list(x_train, o_train), y = y_train, validation_split = 0.1, sample_weights = rep(1, nrow(x_train)),
             batch_size = 0.2 * nrow(x_train), epochs = 10000, verbose = 1, shuffle = FALSE,
             callbacks = list(callback_early_stopping(monitor = "val_loss", min_delta = 1e-08, 
                                                      patience = 500, verbose = 1, 
                                                      restore_best_weights = TRUE)))
  predictions_mlp1_l1 <- mlp1 %>% predict(list(x_val, o_val))
  #predictions_mlp1_l1_unscaled <- as.numeric((predictions_mlp1_l1 * (max_target - min_target)) + min_target)
  
  set.seed(1234777)
  network <- input_data %>%
    layer_dense(units = 200, activation = "tanh", name = "h1") %>%
    layer_dropout(0.4) %>%
    layer_dense(units = 150, activation = "tanh", name = "h2") %>%
    layer_dropout(0.25) %>%
    layer_dense(units = 40, activation = "tanh", name = "h3") %>%
    layer_dropout(0.2) %>%
    layer_dense(units = 1, activation = "linear", name = "network",
                weights = list(array(0, dim = c(40, 1)), 
                               array(log(sum(as.numeric(y_train) * (train_nn[[exposure]] ^ (1 - p))) / sum(train_nn[[exposure]] ^ (2 - p))), 
                                     dim = c(1))))
  response <- list(network, offset) %>% layer_add(name = "add") %>%
    layer_dense(1, activation = k_exp, name = "response", trainable = FALSE, 
                weights = list(array(1, dim = c(1, 1)), array(0, dim = c(1))))
  mlp2 <- keras_model(inputs = c(input_data, offset), outputs = c(response))
  mlp2 <- multi_gpu_model(mlp2, gpus = 2)
  mlp2 %>%
    compile(loss = loss_tweedie, optimizer = optimizer_adam())
  fit <- fit(mlp2, list(x_train, o_train), y = y_train, validation_split = 0.1,
             batch_size = 0.2 * nrow(x_train), epochs = 10000, verbose = 1, shuffle = FALSE,
             callbacks = list(callback_early_stopping(monitor = "val_loss", min_delta = 1e-08, 
                                                      patience = 500, verbose = 1,
                                                      restore_best_weights = TRUE)))
  predictions_mlp2_l1 <- mlp2 %>% predict(list(x_val, o_val))
  #predictions_mlp2_l1_unscaled <- as.numeric((predictions_mlp2_l1 * (max_target - min_target)) + min_target)

  set.seed(130)
  network <- input_data %>%
    layer_dense(units = 100, activation = "tanh", name = "h2") %>%
    layer_dropout(0.25) %>%
    layer_dense(units = 60, activation = "tanh", name = "h3") %>%
    layer_dropout(0.2) %>%
    layer_dense(units = 1, activation = "linear", name = "network",
                weights = list(array(0, dim = c(60, 1)), 
                               array(log(sum(as.numeric(y_train) * (train_nn[[exposure]] ^ (1 - p))) / sum(train_nn[[exposure]] ^ (2 - p))), 
                                     dim = c(1))))
  response <- list(network, offset) %>% layer_add(name = "add") %>%
    layer_dense(1, activation = k_exp, name = "response", trainable = FALSE, 
                weights = list(array(1, dim = c(1, 1)), array(0, dim = c(1))))
  mlp3 <- keras_model(inputs = c(input_data, offset), outputs = c(response))
  mlp3 <- multi_gpu_model(mlp3, gpus = 2)
  mlp3 %>%
    compile(loss = loss_tweedie, optimizer = optimizer_adam())
  fit <- fit(mlp3, list(x_train, o_train), y = y_train, validation_split = 0.1,
             batch_size = 0.2 * nrow(x_train), epochs = 10000, verbose = 1, shuffle = FALSE,
             callbacks = list(callback_early_stopping(monitor = "val_loss", min_delta = 1e-08, 
                                                      patience = 500, verbose = 1,
                                                      restore_best_weights = TRUE)))
  predictions_mlp3_l1 <- mlp3 %>% predict(list(x_val, o_val))
  #predictions_mlp3_l1_unscaled <- as.numeric((predictions_mlp3_l1 * (max_target - min_target)) + min_target)

  set.seed(8755)
  network <- input_data %>%
    layer_dense(units = 50, activation = "tanh", name = "h2") %>%
    layer_dropout(0.2) %>%
    layer_dense(units = 1, activation = "linear", name = "network",
                weights = list(array(0, dim = c(50, 1)), 
                               array(log(sum(as.numeric(y_train) * (train_nn[[exposure]] ^ (1 - p))) / sum(train_nn[[exposure]] ^ (2 - p))), 
                                     dim = c(1))))
  response <- list(network, offset) %>% layer_add(name = "add") %>%
    layer_dense(1, activation = k_exp, name = "response", trainable = FALSE, 
                weights = list(array(1, dim = c(1, 1)), array(0, dim = c(1))))
  mlp4 <- keras_model(inputs = c(input_data, offset), outputs = c(response))
  mlp4 <- multi_gpu_model(mlp4, gpus = 2)
  mlp4 %>%
    compile(loss = loss_tweedie, optimizer = optimizer_adam())
  fit <- fit(mlp4, list(x_train, o_train), y = y_train, validation_split = 0.1, shuffle = FALSE,
             batch_size = 0.2 * nrow(x_train), epochs = 10000, verbose = 1,
             callbacks = list(callback_early_stopping(monitor = "val_loss", min_delta = 1e-08, 
                                                      patience = 500, verbose = 1,
                                                      restore_best_weights = TRUE)))
  predictions_mlp4_l1 <- mlp4 %>% predict(list(x_val, o_val))
  #predictions_mlp4_l1_unscaled <- as.numeric((predictions_mlp4_l1 * (max_target - min_target)) + min_target)

  #Pull in Level 1 Predictions and Append to Validation Set for Level 2 - Set weights for each model based on individual predictive accuracy
  #rmse_1 <- mean_tweedie_deviance(val_stack, targetvar, predictions_mlp1_l1_unscaled, p)
  #rmse_2 <- mean_tweedie_deviance(val_stack, targetvar, predictions_mlp2_l1_unscaled, p)
  #rmse_3 <- mean_tweedie_deviance(val_stack, targetvar, predictions_mlp3_l1_unscaled, p)
  #rmse_4 <- mean_tweedie_deviance(val_stack, targetvar, predictions_mlp4_l1_unscaled, p)
  rmse_1 <- mean_tweedie_deviance(val_stack, targetvar, predictions_mlp1_l1, p)
  rmse_2 <- mean_tweedie_deviance(val_stack, targetvar, predictions_mlp2_l1, p)
  rmse_3 <- mean_tweedie_deviance(val_stack, targetvar, predictions_mlp3_l1, p)
  rmse_4 <- mean_tweedie_deviance(val_stack, targetvar, predictions_mlp4_l1, p)
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
  xgb_train <- xgb.DMatrix(x_val, label = as.numeric(l1_predictions), info = list(base_margin = as.numeric(o_val)))
  xgb_test <- xgb.DMatrix(x_test, label = as.numeric(y_test), info = list(base_margin = as.numeric(o_test)))
  set.seed(874747)
  xgboost <- xgboost(objective = "reg:tweedie", data = xgb_train, nrounds = 100000, verbose = 1, max.depth = 5, 
                        eta = 0.2, early_stopping_rounds = 500, tweedie_variance_power = p)
  predictions_lc <- predict(xgboost, xgb_test)
  #predictions_lc <- (predictions_lc * (max_target - min_target)) + min_target
} else {
  if(selected_model == "mlp1") {
    set.seed(14578)
    network <- input_data %>%
      layer_dense(units = 50, activation = "tanh", name = "h1") %>%
      layer_dropout(0.4) %>%
      layer_dense(units = 25, activation = "tanh", name = "h2") %>%
      layer_dropout(0.2) %>%
      layer_dense(units = 1, activation = "linear", name = "network",
                  weights = list(array(0, dim = c(25, 1)), 
                                 array(log(sum(as.numeric(y_train) * (train_nn[[exposure]] ^ (1 - p))) / sum(train_nn[[exposure]] ^ (2 - p))), 
                                       dim = c(1))))
    response <- list(network, offset) %>% layer_add(name = "add") %>%
      layer_dense(1, activation = k_exp, name = "response", trainable = FALSE, 
                  weights = list(array(1, dim = c(1, 1)), array(0, dim = c(1))))
    mlp <- keras_model(inputs = c(input_data, offset), outputs = c(response))
    mlp <- multi_gpu_model(mlp, gpus = 2)
    mlp %>%
      compile(loss = loss_tweedie, optimizer = optimizer_adam())
    fit <- fit(mlp, list(x_train, o_train), y = y_train, validation_split = 0.1,
               batch_size = 0.2 * nrow(x_train), epochs = 10000, verbose = 1, shuffle = FALSE,
               callbacks = list(callback_early_stopping(monitor = "val_loss", min_delta = 1e-08, 
                                                        patience = 500, verbose = 1, 
                                                        restore_best_weights = TRUE)))
    predictions_lc <- mlp %>% predict(list(x_val, o_val))
    #predictions_lc <- as.numeric((predictions_lc * (max_target - min_target)) + min_target)
  } else 
    if(selected_model == "mlp2") {
      set.seed(1234777)
      network <- input_data %>%
        layer_dense(units = 200, activation = "tanh", name = "h1") %>%
        layer_dropout(0.4) %>%
        layer_dense(units = 150, activation = "tanh", name = "h2") %>%
        layer_dropout(0.25) %>%
        layer_dense(units = 40, activation = "tanh", name = "h3") %>%
        layer_dropout(0.2) %>%
        layer_dense(units = 1, activation = "linear", name = "network",
                    weights = list(array(0, dim = c(40, 1)), 
                                   array(log(sum(as.numeric(y_train) * (train_nn[[exposure]] ^ (1 - p))) / sum(train_nn[[exposure]] ^ (2 - p))), 
                                         dim = c(1))))
      response <- list(network, offset) %>% layer_add(name = "add") %>%
        layer_dense(1, activation = k_exp, name = "response", trainable = FALSE, 
                    weights = list(array(1, dim = c(1, 1)), array(0, dim = c(1))))
      mlp <- keras_model(inputs = c(input_data, offset), outputs = c(response))
      mlp <- multi_gpu_model(mlp, gpus = 2)
      mlp %>%
        compile(loss = loss_tweedie, optimizer = optimizer_adam())
      fit <- fit(mlp, list(x_train, o_train), y = y_train, validation_split = 0.1,
                 batch_size = 0.2 * nrow(x_train), epochs = 10000, verbose = 1, shuffle = FALSE,
                 callbacks = list(callback_early_stopping(monitor = "val_loss", min_delta = 1e-08, 
                                                          patience = 500, verbose = 1, 
                                                          restore_best_weights = TRUE)))
      predictions_lc <- mlp %>% predict(list(x_val, o_val))
      #predictions_lc <- as.numeric((predictions_lc * (max_target - min_target)) + min_target)
    } else 
      if(selected_model == "mlp3") {
        set.seed(130)
        network <- input_data %>%
          layer_dense(units = 100, activation = "tanh", name = "h2") %>%
          layer_dropout(0.25) %>%
          layer_dense(units = 60, activation = "tanh", name = "h3") %>%
          layer_dropout(0.2) %>%
          layer_dense(units = 1, activation = "linear", name = "network",
                      weights = list(array(0, dim = c(60, 1)), 
                                     array(log(sum(as.numeric(y_train) * (train_nn[[exposure]] ^ (1 - p))) / sum(train_nn[[exposure]] ^ (2 - p))), 
                                           dim = c(1))))
        response <- list(network, offset) %>% layer_add(name = "add") %>%
          layer_dense(1, activation = k_exp, name = "response", trainable = FALSE, 
                      weights = list(array(1, dim = c(1, 1)), array(0, dim = c(1))))
        mlp <- keras_model(inputs = c(input_data, offset), outputs = c(response))
        mlp <- multi_gpu_model(mlp, gpus = 2)
        mlp %>%
          compile(loss = loss_tweedie, optimizer = optimizer_adam())
        fit <- fit(mlp, list(x_train, o_train), y = y_train, validation_split = 0.1,
                   batch_size = 0.2 * nrow(x_train), epochs = 10000, verbose = 1, shuffle = FALSE,
                   callbacks = list(callback_early_stopping(monitor = "val_loss", min_delta = 1e-08, 
                                                            patience = 500, verbose = 1, 
                                                            restore_best_weights = TRUE)))
        predictions_lc <- mlp %>% predict(list(x_val, o_val))
        #predictions_lc <- as.numeric((predictions_lc * (max_target - min_target)) + min_target)
      } else 
        if (selected_model == "mlp4") {
    network <- input_data %>%
      layer_dense(units = 50, activation = "tanh", name = "h2") %>%
      layer_dropout(0.2) %>%
      layer_dense(units = 1, activation = "linear", name = "network",
                  weights = list(array(0, dim = c(50, 1)), 
                                 array(log(sum(as.numeric(y_train) * (train_nn[[exposure]] ^ (1 - p))) / sum(train_nn[[exposure]] ^ (2 - p))), 
                                       dim = c(1))))
    response <- list(network, offset) %>% layer_add(name = "add") %>%
      layer_dense(1, activation = k_exp, name = "response", trainable = FALSE, 
                  weights = list(array(1, dim = c(1, 1)), array(0, dim = c(1))))
    mlp <- keras_model(inputs = c(input_data, offset), outputs = c(response))
    mlp <- multi_gpu_model(mlp, gpus = 2)
    mlp %>%
      compile(loss = loss_tweedie, optimizer = optimizer_adam())
    fit <- fit(mlp, list(x_train, o_train), y = y_train, validation_split = 0.1,
               batch_size = 0.2 * nrow(x_train), epochs = 10000, verbose = 1, shuffle = FALSE,
               callbacks = list(callback_early_stopping(monitor = "val_loss", min_delta = 1e-08, 
                                                        patience = 500, verbose = 1, 
                                                        restore_best_weights = TRUE)))
    predictions_lc <- mlp %>% predict(list(x_val, o_val))
    #predictions_lc <- as.numeric((predictions_lc * (max_target - min_target)) + min_target)
        } else
          if(selected_model == "gbm") {
            xgb_train <- xgb.DMatrix(x_train, label = as.numeric(y_train), info = list(base_margin = as.numeric(o_train)))
            xgb_test <- xgb.DMatrix(x_val, label = as.numeric(y_val), info = list(base_margin = as.numeric(o_val)))
            set.seed(874747)
            xgboost <- xgboost(objective = "reg:tweedie", data = xgb_train, nrounds = 100000, verbose = 1, max.depth = 5, 
                               eta = 0.2, early_stopping_rounds = 500, tweedie_variance_power = p)
            predictions_lc <- predict(xgboost, xgb_test)
            #predictions_lc <- (predictions_lc * (max_target - min_target)) + min_target
          }
  else {
    print("Selected Model is not within valid choices.")
  }
}

##======TEST OUTPUT & MODEL EVALUATION=========##
test$GLM_FREQ <- as.numeric(predictions_glm_freq)
test$GLM_SEV <- as.numeric(predictions_glm_sev)
test$ML_FREQ <- as.numeric(predictions_freq)
test$ML_SEV <- as.numeric(predictions_sev)
test$GLM_LC <- as.numeric(predictions_glm_lc)
test$ML_LC <- as.numeric(predictions_lc)

test <- test %>%
  mutate(GLM = GLM_FREQ * GLM_SEV, ML = ML_FREQ * ML_SEV)

write_csv(test, "test_w_predictions.csv")

targetvar <- c("ClaimAmount")

#ML Freq/Sev Model Metrics
rmse_ML <- rmse(data = test, targetvar = targetvar, prediction.obj = as.numeric(test$ML))
nrmse_ML <- NRMSE(test, targetvar, test$ML)
cor_ML_pearson <- as.numeric(cor(test$ML, as.numeric(test[[targetvar]])))
cor_ML_spearman <- as.numeric(cor(test$ML, as.numeric(test[[targetvar]]), method = "spearman"))
mae_ML <- MAE(test$ML, as.numeric(test[[targetvar]]))
gini_ML <- gini_value(test$ML, test[[exposure]])
agg_rpr_ML <- agg_rpr(test, targetvar, test$ML)
nmae_ML <- NMAE(test$ML, test$ClaimAmount)
norm_rpd_ML <- norm_rp_deviance(test, targetvar, test$ML)

#GLM Freq/Sev Model Metrics
rmse_glm <- rmse(data = test, targetvar = targetvar, prediction.obj = as.numeric(test$GLM))
nrmse_glm <- NRMSE(test, targetvar, test$GLM)
cor_glm_pearson <- as.numeric(cor(test$GLM, as.numeric(test[[targetvar]])))
cor_glm_spearman <- as.numeric(cor(test$GLM, as.numeric(test[[targetvar]]), method = "spearman"))
mae_glm <- MAE(test$GLM, as.numeric(test[[targetvar]]))
gini_glm <- gini_value(test$GLM, test[[exposure]])
agg_rpr_glm <- agg_rpr(test, targetvar, test$GLM)
nmae_glm <- NMAE(test$GLM, test$ClaimAmount)
norm_rpd_glm <- norm_rp_deviance(test, targetvar, test$GLM)

#GLM LC Model Metrics
rmse_glm_lc <- rmse(data = test, targetvar = targetvar, prediction.obj = as.numeric(test$GLM_LC))
nrmse_glm_lc <- NRMSE(test, targetvar, test$GLM_LC)
cor_glm_lc_pearson <- as.numeric(cor(test$GLM_LC, as.numeric(test[[targetvar]])))
cor_glm_lc_spearman <- as.numeric(cor(test$GLM_LC, as.numeric(test[[targetvar]]), method = "spearman"))
mae_glm_lc <- MAE(test$GLM_LC, as.numeric(test[[targetvar]]))
gini_glm_lc <- gini_value(test$GLM_LC, test[[exposure]])
agg_rpr_glm_lc <- agg_rpr(test, targetvar, test$GLM_LC)
nmae_glm_lc <- NMAE(test$GLM_LC, test$ClaimAmount)
norm_rpd_glm_lc <- norm_rp_deviance(test, targetvar, test$GLM_LC)

#ML LC Model Metrics
rmse_ML_lc <- rmse(data = test, targetvar = targetvar, prediction.obj = as.numeric(test$ML_LC))
nrmse_ML_lc <- NRMSE(test, targetvar, test$ML_LC)
cor_ML_lc_pearson <- as.numeric(cor(test$ML_LC, as.numeric(test[[targetvar]])))
cor_ML_lc_spearman <- as.numeric(cor(test$ML_LC, as.numeric(test[[targetvar]]), method = "spearman"))
mae_ML_lc <- MAE(test$ML_LC, as.numeric(test[[targetvar]]))
gini_ML_lc <- gini_value(test$ML_LC, test[[exposure]])
agg_rpr_ML_lc <- agg_rpr(test, targetvar, test$ML_LC)
nmae_ML_lc <- NMAE(test$ML_LC, test$ClaimAmount)
norm_rpd_ML_lc <- norm_rp_deviance(test, targetvar, test$ML_LC)

#Compilation of All Model Metrics
models <- c("GLM Freq/Sev", "ML Freq/Sev", "GLM LC", "ML LC")
rmse_compiled <- c(rmse_glm, rmse_ML, rmse_glm_lc, rmse_ML_lc) * 100
nrmse_compiled <- 1 - (c(nrmse_glm, nrmse_ML, nrmse_glm_lc, nrmse_ML_lc) * 100)
nmae_compiled <- 1 - (c(nmae_glm, nmae_ML, nmae_glm_lc, nmae_ML_lc) * 100)
mae_compiled <- c(mae_glm, mae_ML, mae_glm_lc, mae_ML_lc)
pearson_cor_compiled <- c(cor_glm_pearson, cor_ML_pearson, cor_glm_lc_pearson, cor_ML_lc_pearson)
spearman_cor_compiled <- c(cor_glm_spearman, cor_ML_spearman, cor_glm_lc_spearman, cor_ML_lc_spearman)
gini_index_compiled <- c(gini_glm, gini_ML, gini_glm_lc, gini_ML_lc)
agg_rpr_compiled <- c(agg_rpr_glm, agg_rpr_ML, agg_rpr_glm_lc, agg_rpr_ML_lc)
norm_rpd_compiled <- c(norm_rpd_glm, norm_rpd_ML, norm_rpd_glm_lc, norm_rpd_ML_lc)
model_metrics <- data.frame(models, rmse_compiled, mae_compiled, pearson_cor_compiled, spearman_cor_compiled, gini_index_compiled, agg_rpr_compiled)
colnames(model_metrics) <- c("Model", "RMSE", "MAE", "Pearson (Linear) Correlation", "Spearman Rank Correlation", "Gini Index", "Aggregate Risk Premium Differential")

eval_metrics <- data.frame(gini_index_compiled, nrmse_compiled, nmae_compiled, spearman_cor_compiled, norm_rpd_compiled)
lscore <- apply(eval_metrics, 1, mean)
gscore <- apply(eval_metrics, 1, geometric_mean)
hscore <- apply(eval_metrics, 1, harmonic_mean)
model_metrics$lscore <- lscore
model_metrics$gscore <- gscore
model_metrics$hscore <- hscore
model_metrics[is.na(model_metrics)] <- 0

View(model_metrics)
write_csv(model_metrics, "model_evaluation_stats.csv")

lift_curve(test$GLM, observed_loss_cost = test[[targetvar]], exposure = test[[exposure]], n = 40)
lift_curve(test$ML, observed_loss_cost = test[[targetvar]], exposure = test[[exposure]], n = 40)
lift_curve(test$GLM_LC, observed_loss_cost = test[[targetvar]], exposure = test[[exposure]], n = 40)
lift_curve(test$ML_LC, observed_loss_cost = test[[targetvar]], exposure = test[[exposure]], n = 40)

gini_index_plot_ML <- gini_plot(test$ML, as.numeric(unlist(test[, c(exposure)])))
gini_index_plot_ML + ggtitle("ML Gini Index Plot")

gini_index_plot_GLM <- gini_plot(test$GLM, as.numeric(unlist(test[, c(exposure)])))
gini_index_plot_GLM + ggtitle("GLM Gini Index Plot")

##=======MODEL ANALYTICS==========##
#Trim out Dead Wood first
rm(dat, dat_nn, fit, glm_freq, glm_lc, glm_sev, input_data, network, o_test, o_train, o_val, offset,
   response, test_nn, train, train_nn, train_sev, train_stack, train_stack_bal, val_nn, val_stack, x_test, x_train,
   x_val, y_test, y_train, y_val, cor_glm_lc_pearson, cor_glm_lc_spearman, eval_metrics, norm_rpd_compiled,
   cor_glm_pearson, cor_glm_spearman, cor_ML_lc_pearson, cor_ML_lc_spearman, cor_ML_pearson, norm_rpd_glm,
   cor_ML_spearman, gini_glm, gini_glm_lc, gini_index_compiled, gini_ML, gini_ML_lc, nmae_glm, nmae_ML, nmae_ML_lc,
   l1_rmse, mae_compiled, mae_glm, mae_glm_lc, mae_ML, mae_ML_lc, max_target, min_target, mlp1, norm_rpd_glm_lc,
   models, p, pearson_cor_compiled, predictions_glm_freq, predictions_glm_lc, predictions_glm_sev, norm_rpd_ML,
   predictors_nn, rmse_compiled, rmse_glm, rmse_glm_lc, rmse_ML, rmse_ML_lc, soft_sum, spearman_cor_compiled, targetvar,
   w_test, w_train, w_val, weights, xgb_test, xgb_train, extract_levels, gscore, hscore, lscore, norm_rpd_ML_lc,
   loss_tweedie, tweedie_deviance_loss, exposure, predictions_l2_lc, agg_rpr_glm_lc, nmae_compiled, nmae_glm_lc,
   predictions_l2_sev, weight, gini_index_plot_GLM, agg_rpr_glm, agg_rpr_ML, agg_rpr_compiled, agg_rpr_ML_lc,
   predictions_mlp1_l1, ay, gini_index_plot_stack, grouped_data, one_way_analysis_plot, l1_predictions, model_metrics, 
   predictions_mlp2_l1, predictions_mlp3_l1, predictions_mlp4_l1, text_project, theme_project, xgboost, approach_used,
   d, mlp, mlp2, mlp3, mlp4, predictions_freq, predictions_lc, predictions_mlp1_l1_unscaled, predictions_mlp2_l1_unscaled,
   predictions_mlp3_l1_unscaled, predictions_mlp4_l1_unscaled, predictions_sev, predictions_xgb_l2, rmse_1,
   rmse_2, rmse_3, rmse_4, selected_model, gini_index_plot_ML, nrmse_compiled, nrmse_glm, nrmse_glm_lc, nrmse_ML, nrmse_ML_lc)

#Set up Parameters and Take Stratified Sample from Test
targetvar <- c("ML_LC")
exposure <- c("total_offset")
excl_vars <- c("ML_FREQ", "ML_SEV", "ML", "GLM_FREQ", "GLM_SEV", "GLM_LC", "GLM", "ClaimAmount", "ClaimCount",
               "severity", "year", "exposure", "BonusMalus", "dtype", "RecordID")
predictors <- colnames(test)[!colnames(test) %in% c(targetvar, exposure, excl_vars)]

xgbtrain <- xgb.DMatrix(prep_xgb(test, predictors), label = as.numeric(test[[targetvar]]), 
                        info = list(base_margin = log(test[[exposure]])))

#RF Surrogate Model
set.seed(5999)
rf_surrogate <- xgboost(objective = "reg:gamma", data = xgbtrain, nrounds = 250, verbose = 1, max.depth = 5, 
                        eta = 0.3, early_stopping_rounds = 50)
test$RF_SURROGATE <- predict(rf_surrogate, xgb.DMatrix(prep_xgb(test, predictors),
                                                             info = list(base_margin = log(test$exposure))))

#Evaluate Surrogate using R-Squared
r_sq <- function(x, y) {
  r_sq <- cor(x, y)^2
  return(r_sq)
}

r_sq(test$ML_LC, test$RF_SURROGATE)*100
cor(test$ML_LC, test$RF_SURROGATE, method = "spearman")*100

rm(xgbtrain)

##Setup Data for Model-Agnostic Analytics
feat_spac <- test[, predictors]
pred <- function(model, newdata) {
  newdata <- xgb.DMatrix(prep_xgb(newdata, predictors), info = list(base_margin = log(as.numeric(newdata[[exposure]]))))
  results <- as.numeric(predict(model, newdata))
  return(results)
}

#Permutation RMSE-Based Feature Importance
explainer_rf_surrogate <- explain(rf_surrogate, data = test[, c(predictors)], predict_function = pred, y = as.numeric(test$RF_SURROGATE))
ptm <- proc.time()
imp <- variable_importance(explainer_rf_surrogate, loss_function = loss_root_mean_square, n_sample = nrow(test))
proc.time() - ptm
plot(imp)
imp <- imp[!startsWith(as.character(imp$variable), "_"), ]
imp <- imp %>%
  group_by(variable) %>%
  summarise(dropout_loss = mean(dropout_loss))
min <- min(imp$dropout_loss); max <- max(imp$dropout_loss)
imp$scaled_dropout_loss <- (imp$dropout_loss - min) / (max - min)
colnames(imp)[1] <- c("Variable")

#Shapley Value Based Feature Importance
flash <- flashlight(model = rf_surrogate, data = test[, c(predictors, "RF_SURROGATE")], y = "RF_SURROGATE", label = "RF_SURROGATE", predict_function = pred)
flash <- add_shap(flash, visit_strategy = "importance", n_shap = 20)
imp_shap <- light_importance(flash, type = "shap", m_repititions = 10)$data
imp_shap <- imp_shap %>%
  select(c(variable, value))
colnames(imp_shap)[1] <- c("Variable")
min <- min(imp_shap$value); max <- max(imp_shap$value)
imp_shap$scaled_shap_importance <- (imp_shap$value - min) / (max - min)

imp_combined <- inner_join(imp, imp_shap, by = "Variable")
imp_combined <- imp_combined %>%
  group_by(Variable) %>%
  mutate(composite_variable_importance_score = (0.05 * scaled_shap_importance) + (0.95 * scaled_dropout_loss))
min <- min(imp_combined$composite_variable_importance_score); max <- max(imp_combined$composite_variable_importance_score)
imp_combined$scaled_composite_variable_importance_score <- scale(imp_combined$composite_variable_importance_score, center = min, scale = (max - min))
imp_combined <- imp_combined[, c(1, 3, 5, 7)]

write_csv(imp_combined, "feature_importance.csv")

top_vars <- imp_combined %>%
  arrange(desc(as.numeric(scaled_composite_variable_importance_score))) %>%
  select(Variable) %>%
  head(min(nrow(imp_combined), 6))
top_vars <- as.character(unlist(top_vars))

rm(explainer_rf_surrogate, imp_shap, imp, ptm, max, min, imp_combined, flash)

#Select Top 50 Best and Top 50 Worst Cases for Local Interpretation
test$Record_ID <- 1:nrow(test)

top_50_best_cases <- test %>%
  arrange(desc(get(targetvar))) %>%
  head(50)

top_50_worst_cases <- test %>%
  arrange(get(targetvar)) %>%
  head(50)

top_50_best_id <- top_50_best_cases$Record_ID
top_50_worst_id <- top_50_worst_cases$Record_ID

top_50_best_cases <- top_50_best_cases[, which(colnames(top_50_best_cases) %in% predictors)]
top_50_worst_cases <- top_50_worst_cases[, which(colnames(top_50_worst_cases) %in% predictors)]

#Set up Local Interpretation Outputs for Top 50 Best and Worst Cases and 50 Random Cases - Using Shapley Values
predictor <- Predictor$new(rf_surrogate, data = feat_spac, y = as.numeric(test$RF_SURROGATE), predict.fun = pred)

n <- nrow(top_50_best_cases)
lime_output_best_cases <- list()
for (i in 1:n) {
  shap <- Shapley$new(predictor, x.interest = top_50_best_cases[i, ])$results
  feature_values <- strsplit(shap$feature.value, "=", fixed = TRUE)
  shap$feature_value <- unlist(feature_values)[c(FALSE, TRUE)]
  shap$Record_ID <- top_50_best_id[i]
  shap <- shap[, c("Record_ID", "feature", "feature_value", "phi")]
  lime_output_best_cases[[i]] <- shap
  print(i)
}
lime_output_best_cases <- do.call(rbind, lime_output_best_cases)
lime_output_best_cases <- inner_join(lime_output_best_cases, test[, c("Record_ID", targetvar)], by = "Record_ID")
lime_output_best_cases$Record_ID <- rev(ntile(lime_output_best_cases[[targetvar]], length(unique(lime_output_best_cases[[targetvar]]))))

n <- nrow(top_50_worst_cases)
lime_output_worst_cases <- list()
for (i in 1:n) {
  shap <- Shapley$new(predictor, x.interest = top_50_worst_cases[i, ])$results
  feature_values <- strsplit(shap$feature.value, "=", fixed = TRUE)
  shap$feature_value <- unlist(feature_values)[c(FALSE, TRUE)]
  shap$Record_ID <- top_50_worst_id[i]
  shap <- shap[, c("Record_ID", "feature", "feature_value", "phi")]
  lime_output_worst_cases[[i]] <- shap
  print(i)
}
lime_output_worst_cases <- do.call(rbind, lime_output_worst_cases)
lime_output_worst_cases <- inner_join(lime_output_worst_cases, test[, c("Record_ID", targetvar)], by = "Record_ID")
lime_output_worst_cases$Record_ID <- ntile(lime_output_worst_cases[[targetvar]], length(unique(lime_output_worst_cases[[targetvar]])))

write_csv(lime_output_best_cases, "local_interpretation_best_cases.csv")
write_csv(lime_output_worst_cases, "local_interpretation_worst_cases.csv")

#ALE Profiles
ale <- FeatureEffects$new(predictor, predictors, method = "ale", center.at = 0, grid.size = nrow(test))
ale <- do.call(rbind.data.frame, ale$results)
ale_profile <- ale %>%
  select(c(4, 3, 2))
rownames(ale_profile) <- NULL
colnames(ale_profile) <- c("feature", "level", "value")
ale_profile <- ale_profile %>%
  mutate(value = value + mean(test$RF_SURROGATE))
write_csv(ale_profile, "ale_profile.csv")
rm(ale, ale_profile)

rm(shap, top_50_best_cases, top_50_worst_cases, i, n, top_50_best_id, top_50_worst_id, feature_values, feat_spac, 
   predictor, lime_output_best_cases, lime_output_worst_cases)

#Feature Effect Analysis
flash <- flashlight(model = rf_surrogate, data = test[, c(predictors, "RF_SURROGATE")], y = "RF_SURROGATE", label = "RF_SURROGATE", predict_function = pred)

ice_profile <- list()
for(i in 1:length(predictors)) {
  ice <- light_ice(flash, v = predictors[i], n_max = 0.25*nrow(test))$data
  ice <- ice %>%
    select(-label) %>%
    mutate(variable = colnames(ice)[2])
  colnames(ice)[2] <- c("level")
  ice <- ice[, c(1, 4, 2, 3)]
  ice_profile[[i]] <- ice
  print(i)
}
ice_profile <- do.call("rbind", ice_profile)
write_csv(ice_profile, "ice_profile.csv")
rm(ice, i, ice_profile)

pd_profile <- list()
for(i in 1:length(predictors)) {
  pd <- light_profile(flash, v = predictors[i], type = "partial dependence", pd_n_max = nrow(test))$data
  pd <- pd %>%
    select(-c(label, type)) %>%
    mutate(variable = colnames(pd)[1])
  colnames(pd)[1] <- c("level")
  pd <- pd[, c(4, 1, 2, 3)]
  pd_profile[[i]] <- pd
  print(i)
}
pd_profile <- do.call("rbind", pd_profile)
write_csv(pd_profile, "pd_profile.csv")
rm(pd, i, pd_profile)

#Interaction Effects using Friedman's H-Statistic
test <- test %>%
  arrange(get(targetvar)) %>%
  mutate(bucket = ntile(get(targetvar), 20000))
strat_cols <- c(predictors, "bucket")
test_strat <- stratified(test, "bucket", 0.2)
test_strat <- test_strat %>%
  select(-bucket)

nfolds <- round(nrow(test_strat) / 500)
nreps <- 2
rm(strat_cols)

int_effects_nfolds <- list()
for(m in 1:nreps) {
  groups <- createFolds(test_strat$Record_ID, nfolds)
  interaction_effects <- list()
  for(i in 1:nfolds) {
    dat <- test_strat %>%
      dplyr::filter(Record_ID %in% groups[[i]])
    int <- light_interaction(flash, v = top_vars, pairwise = TRUE, n_max = nrow(dat))$data
    int <- int %>%
      select(-c(label, error))
    int$fold <- i
    interaction_effects[[i]] <- int
    print(paste("Fold ", i, " Completed"))
    i <- i + 1
  }
  interaction_effects <- do.call(rbind.data.frame, interaction_effects)
  interactions_h_stat <- interaction_effects %>%
    dplyr::group_by(variable) %>%
    dplyr::summarize(int_strength = mean(value))
  interactions_h_stat$rep <- m
  int_effects_nfolds[[m]] <- interactions_h_stat
  print(paste("Rep ", m, " Completed"))
  m <- m + 1
}
int_effects_nfolds <- do.call(rbind.data.frame, int_effects_nfolds)
rm(dat)

interactions_h_stat <- int_effects_nfolds %>%
  dplyr::group_by(variable) %>%
  dplyr::summarize(int_strength = sum(int_strength))

min_h <- min(interactions_h_stat$int_strength)
max_h <- max(interactions_h_stat$int_strength)
interactions_h_stat$scaled_h <- scale(interactions_h_stat$int_strength, center = min_h, scale = (max_h - min_h))
colnames(interactions_h_stat) <- c("Interaction_Term", "H_Statistic", "Scaled_Interaction_Strength")

write_csv(interactions_h_stat, "interaction_effects.csv")
rm(flash, min_h, max_h, top_vars, groups, nfolds, nreps, m)
