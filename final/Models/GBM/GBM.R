rm(list = ls())

setwd("/home/marco/Documents/gitreps/pricing/final")

source("./Utils/utils.R")

# Import Data -------------------------------------------------------------

train <- import_data("./Data/train_final.csv") %>% mutate(id = "train")

test <- import_data("./Data/test_final.csv") %>% mutate(id = "test")

comb_data <- bind_rows(train, test)

xgb_model_numb <- xgb.load("./Models/GBM/xgb_numbers")

xgb_model_sev <- xgb.load("./Models/GBM/xgb_severity")


# Add Variables -----------------------------------------------------------

train_data_numb <- create_data_numb(filter(comb_data, id == "train") %>% select(-id))

train_data_sev <- create_data_sev(filter(comb_data, id == "train", sev > 0) %>% select(-id))

test_data_numb <- create_data_numb(filter(comb_data, id == "test") %>% select(-id))
 
test_data_sev <- create_data_sev(filter(comb_data, id == "test") %>% select(-id))

train_numb <- xgb.DMatrix(data = train_data_numb, label = filter(comb_data, id == "train") %>% pull(ClaimInd))

train_sev <- xgb.DMatrix(data = train_data_sev, label = filter(comb_data, id == "train", sev > 0) %>% pull(sev))

test_numb <- xgb.DMatrix(data = test_data_numb, label = filter(comb_data, id == "test") %>% pull(ClaimInd))

test_sev <- xgb.DMatrix(data = test_data_sev, label = filter(comb_data, id == "test") %>% pull(sev))


# Model Section -----------------------------------------------------------

# Numbers

# DON'T RUN
# 
# params <- list(booster = "gbtree", 
#                objective = "count:poisson",
#                eval_metric = "poisson-nloglik",
#                eta = 0.1, 
#                gamma = 1, 
#                max_depth = 6)
# 
# 
# xgbcv <- xgb.cv(params = params, 
#                 data = train_numb, 
#                 nrounds = 1000, 
#                 nfold = 5, 
#                 showsd = TRUE, 
#                 stratified = TRUE, 
#                 print_every_n = 100, 
#                 early_stop_round = 20, 
#                 maximize = FALSE,
#                 prediction = TRUE)
# 
# xgb_model_numb <- xgb.train(data = train_numb,
#                             params = params,
#                             watchlist = list(train = train_numb, test = test_numb),
#                             nrounds = which.min(xgbcv$evaluation_log$train_poisson_nloglik_mean),
#                             verbose = 1,
#                             print_every_n = 10)

numb_pred <- predict(xgb_model_numb, test_numb)

# Severity

# DON'T RUN
# 
# params <- list(booster = "gbtree", 
#                eval_metric = "gamma-nloglik",
#                objective = "reg:gamma",
#                eta = .01, 
#                gamma = 1, 
#                max_depth = 10)
# 
# 
# xgbcv <- xgb.cv(params = params, 
#                 data = train_sev, 
#                 weight = train$ClaimInd,
#                 nrounds = 10000, 
#                 nfold = 5, 
#                 showsd = TRUE, 
#                 stratified = TRUE, 
#                 print_every_n = 100, 
#                 early_stop_round = 20, 
#                 maximize = FALSE,
#                 prediction = TRUE)
# 
# xgb_model_sev <- xgb.train(data = train_sev,
#                            weight = train$ClaimInd,
#                            params = params,
#                            watchlist = list(train = train_sev, test = test_sev),
#                            nrounds = which.min(xgbcv$evaluation_log$train_gamma_nloglik_mean),
#                            verbose = 1,
#                            print_every_n = 100)

sev_pred <- predict(xgb_model_sev, test_sev)

# Prediction

test <- test %>% mutate(observed_lc = ClaimAmount / exposure,
                        predicted_lc = numb_pred * sev_pred / exposure)


# Performance Evaluation --------------------------------------------------

eval_dataset <- test %>% select(exposure, observed_lc, predicted_lc)

eval_dataset %$% NRMSE(predicted_lc, observed_lc)

eval_dataset %$% gini_plot(predicted_lc, exposure)

eval_dataset %$% gini_value(predicted_lc, exposure)
  
eval_dataset %$% lift_curve_plot(predicted_lc, observed_lc, 20) +
                 scale_y_log10()