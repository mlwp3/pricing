rm(list = ls())

set.seed(999)

setwd("/home/marco/Documents/gitrepos/pricing/final")

source("./Utils/utils.R")

# Import Data -------------------------------------------------------------

train <- import_data("./Data/train_new_final.csv") %>% group_data()

test <- import_data("./Data/test_new_final.csv") %>% group_data()

df <- train %>% mutate(id = row_number())

train <- df %>% sample_frac(.70)

validation <- anti_join(df, train, by = "id") %>% select(-id)

train <- train %>% select(-id)

# xgb_model_numb <- xgb.load("./Models/GBM/xgb_numbers")
#  
# xgb_model_sev <- xgb.load("./Models/GBM/xgb_severity")
#
# xgb_model_lc <- xgb.load("./Models/GBM/xgb_loss_cost")


# Create datasets for xgboost ---------------------------------------------

train_numb <- xgb.DMatrix(data = create_data_numb(train), 
                          label = train %>% pull(ClaimNb), 
                          base_margin = train %>% pull(Exposure) %>% log())

train_sev <- xgb.DMatrix(data = create_data_sev(filter(train, Severity > 0)), 
                         label = filter(train, Severity > 0) %>% pull(Severity),
                         weight = filter(train, Severity > 0) %>% pull(ClaimNb))

train_lc <- xgb.DMatrix(data = create_data_numb(train), 
                        label = train %>% pull(Loss_Cost), 
                        base_margin = train %>% pull(Exposure) %>% log())

validation_numb <- xgb.DMatrix(data = create_data_numb(validation), 
                               label = validation %>% pull(ClaimNb), 
                               base_margin = validation %>% pull(Exposure) %>% log())

validation_sev <- xgb.DMatrix(data = create_data_sev(filter(validation, Severity > 0)), 
                              label = filter(validation, Severity > 0) %>% pull(Severity))

validation_lc <- xgb.DMatrix(data = create_data_lc(validation), 
                             label = validation %>% pull(Loss_Cost), 
                             base_margin = validation %>% pull(Exposure) %>% log())

test_numb <- xgb.DMatrix(data = create_data_numb(test), 
                         label = test %>% pull(ClaimNb), 
                         base_margin = test %>% pull(Exposure) %>% log())

test_sev <- xgb.DMatrix(data = create_data_sev(test), 
                        label = test %>% pull(Severity))

test_lc <- xgb.DMatrix(data = create_data_lc(test), 
                       label = test %>% pull(Loss_Cost), 
                       base_margin = test %>% pull(Exposure) %>% log())

# Model Section -----------------------------------------------------------

# Numbers

# DON'T RUN
# 
# params <- list(booster = "gbtree",
#                tree_method = "hist",
#                objective = "count:poisson",
#                eval_metric = "poisson-nloglik",
#                eta = .1,
#                base_score = 1)
# 
# xgbcv_numb <- xgb.cv(params = params,
#                      data = train_numb,
#                      nrounds = 1000,
#                      nfold = 5,
#                      showsd = FALSE,
#                      stratified = TRUE,
#                      print_every_n = 100,
#                      early_stop_round = 20,
#                      maximize = FALSE,
#                      prediction = TRUE)
# 
# xgb_model_numb <- xgb.train(data = train_numb,
#                             params = params,
#                             watchlist = list(train = train_numb, test = validation_numb),
#                             nrounds = which.min(xgbcv_numb$evaluation_log$test_poisson_nloglik_mean),
#                             verbose = 1,
#                             print_every_n = 10)
# 
# xgb.save(xgb_model_numb, "./Models/GBM/xgb_numbers")

importance_matrix_numb <- xgb.importance(colnames(train_numb), model = xgb_model_numb)

xgb.ggplot.importance(importance_matrix_numb, rel_to_first = TRUE, top_n = 10, n_clusters = 3) +
  ggtitle("Feature Importance Frequency") + ggsave("./Output/GBM/feat_imp_freq.png")

numb_pred <- predict(xgb_model_numb, test_numb)

# Severity

# DON'T RUN
# 
# params <- list(booster = "gbtree",
#                tree_method = "hist",
#                eval_metric = "gamma-nloglik",
#                objective = "reg:gamma",
#                eta = .1)
# 
# xgbcv_sev <- xgb.cv(params = params,
#                     data = train_sev,
#                     nrounds = 1000,
#                     nfold = 5,
#                     showsd = FALSE,
#                     stratified = FALSE,
#                     print_every_n = 100,
#                     early_stop_round = 20,
#                     maximize = FALSE,
#                     prediction = TRUE)
# 
# xgb_model_sev <- xgb.train(data = train_sev,
#                            params = params,
#                            watchlist = list(train = train_sev, test = validation_sev),
#                            nrounds = which.min(xgbcv_sev$evaluation_log$test_gamma_nloglik_mean),
#                            verbose = 1,
#                            print_every_n = 100)
# 
# xgb.save(xgb_model_sev, "./Models/GBM/xgb_severity")

importance_matrix_sev <- xgb.importance(colnames(train_sev), model = xgb_model_sev)

xgb.ggplot.importance(importance_matrix_sev, rel_to_first = TRUE, top_n = 10, n_clusters = 3) + 
  ggtitle("Feature Importance Severity") + ggsave("./Output/GBM/feat_imp_sev.png")

sev_pred <- predict(xgb_model_sev, test_sev)

# Loss Cost

# DON'T RUN
# 
# params <- list(booster = "gbtree",
#                tree_method = "gpu_hist",
#                eval_metric = "tweedie-nloglik@1.6",
#                objective = "reg:tweedie",
#                eta = .1,
#                base_score = 1)
# 
# xgbcv_lc <- xgb.cv(params = params,
#                    data = train_lc,
#                    nrounds = 1000,
#                    nfold = 5,
#                    print_every_n = 100,
#                    showsd = FALSE,
#                    early_stop_round = 20,
#                    maximize = FALSE,
#                    prediction = TRUE)
# 
# xgb_model_lc <- xgb.train(data = train_lc,
#                           params = params,
#                           watchlist = list(train = train_lc, test = validation_lc),
#                           nrounds = which.min(xgbcv_lc$evaluation_log$`test_tweedie_nloglik@1.6_mean`),
#                           verbose = 1,
#                           print_every_n = 100)
# 
# xgb.save(xgb_model_lc, "./Models/GBM/xgb_loss_cost")

importance_matrix_lc <- xgb.importance(colnames(train_lc), model = xgb_model_lc)

xgb.ggplot.importance(importance_matrix_lc, rel_to_first = TRUE, top_n = 10, n_clusters = 3) + 
  ggtitle("Feature Importance Loss Cost") + ggsave("./Output/GBM/feat_imp_lc.png")

lc_pred <- predict(xgb_model_lc, test_lc)

# Predictions

test <- test %>% mutate(observed_lc = ClaimAmount / Exposure,
                        predicted_lc = numb_pred * sev_pred / Exposure,
                        predicted_lc_tw = lc_pred)

# Performance Evaluation --------------------------------------------------

eval_dataset <- test %>% select(Exposure, observed_lc, predicted_lc, predicted_lc_tw)

eval_dataset %$% NRMSE(predicted_lc, observed_lc)

eval_dataset %$% NRMSE(predicted_lc_tw, observed_lc)

eval_dataset %$% gini_plot(predicted_lc, Exposure) + ggtitle("Gini index Freq / Sev") + ggsave("./Output/GBM/gini_freq_sev.png")

eval_dataset %$% gini_plot(predicted_lc_tw, Exposure) + ggtitle("Gini index Loss Cost") + ggsave("./Output/GBM/gini_lc.png")

eval_dataset %$% gini_value(predicted_lc, Exposure)

eval_dataset %$% gini_value(predicted_lc_tw, Exposure)

eval_dataset %$% lift_curve_table(predicted_lc, observed_lc, Exposure, 20) %>% 
                 lift_curve_plot() +
                 ggtitle("Lift Curve Freq / Sev") + ggsave("./Output/GBM/lift_curve_freq_sev.png")

eval_dataset %$% lift_curve_table(predicted_lc_tw, observed_lc, Exposure, 20) %>% 
                 lift_curve_plot() +
                 ggtitle("Lift Curve Loss Cost") + ggsave("./Output/GBM/lift_curve_lc.png")

eval_dataset %$% double_lift_chart(predicted_lc, predicted_lc_tw, observed_lc, Exposure, 20, "Freq / Sev", "Loss Cost") + 
                 ggtitle("Double Lift Curve") + ggsave("./Output/GBM/double_lift_curve.png")
