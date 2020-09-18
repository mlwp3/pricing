rm(list = ls())

set.seed(999)

setwd("/home/marco/Documents/gitrepos/pricing/final")

source("./Utils/utils.R")

# Import Data -------------------------------------------------------------

train <- import_data("./Data/train_new_final.csv") %>% group_data()

test <- import_data("./Data/test_new_final.csv") %>% 
        mutate(Severity = ifelse(ClaimNb == 0, 0, ClaimAmount / ClaimNb))

df <- train %>% mutate(id = row_number())

train <- df %>% sample_frac(.70)

validation <- anti_join(df, train, by = "id") %>% select(-id)

train <- train %>% select(-id)

xgb_model_numb <- xgb.load("./Models/GBM/xgb_numbers")

xgb_model_sev <- xgb.load("./Models/GBM/xgb_severity")

xgb_model_losses <- xgb.load("./Models/GBM/xgb_losses")

# Create datasets for xgboost ---------------------------------------------

train_numb <- xgb.DMatrix(data = create_data_numb(train), 
                          label = train %>% pull(ClaimNb), 
                          base_margin = train %>% pull(Exposure) %>% log())

train_sev <- xgb.DMatrix(data = create_data_sev(filter(train, Severity > 0)), 
                         label = filter(train, Severity > 0) %>% pull(Severity),
                         weight = filter(train, Severity > 0) %>% pull(ClaimNb))

train_losses <- xgb.DMatrix(data = create_data_losses(train), 
                        label = train %>% pull(ClaimAmount), 
                        base_margin = train %>% pull(Exposure) %>% log())

validation_numb <- xgb.DMatrix(data = create_data_numb(validation), 
                               label = validation %>% pull(ClaimNb), 
                               base_margin = validation %>% pull(Exposure) %>% log())

validation_sev <- xgb.DMatrix(data = create_data_sev(filter(validation, Severity > 0)), 
                              label = filter(validation, Severity > 0) %>% pull(Severity))

validation_losses <- xgb.DMatrix(data = create_data_losses(validation), 
                             label = validation %>% pull(Loss_Cost), 
                             base_margin = validation %>% pull(Exposure) %>% log())

test_numb <- xgb.DMatrix(data = create_data_numb(test), 
                         label = test %>% pull(ClaimNb), 
                         base_margin = test %>% pull(Exposure) %>% log())

test_sev <- xgb.DMatrix(data = create_data_sev(test), 
                        label = test %>% pull(Severity))

test_losses <- xgb.DMatrix(data = create_data_losses(test), 
                       label = test %>% pull(ClaimAmount), 
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
#                      nrounds = 10000,
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
#                     nrounds = 10000,
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
#                tree_method = "hist",
#                eval_metric = "tweedie-nloglik@1.6",
#                objective = "reg:tweedie",
#                eta = .1,
#                base_score = 1)
# 
# xgbcv_losses <- xgb.cv(params = params,
#                    data = train_losses,
#                    nrounds = 10000,
#                    nfold = 5,
#                    print_every_n = 100,
#                    showsd = FALSE,
#                    early_stop_round = 20,
#                    maximize = FALSE,
#                    prediction = TRUE)
# 
# xgb_model_losses <- xgb.train(data = train_losses,
#                           params = params,
#                           watchlist = list(train = train_losses, test = validation_losses),
#                           nrounds = which.min(xgbcv_losses$evaluation_log$`test_tweedie_nloglik@1.6_mean`),
#                           verbose = 1,
#                           print_every_n = 100)
# 
# xgb.save(xgb_model_losses, "./Models/GBM/xgb_losses")

importance_matrix_losses <- xgb.importance(colnames(train_losses), model = xgb_model_losses)

xgb.ggplot.importance(importance_matrix_losses, rel_to_first = TRUE, top_n = 10, n_clusters = 3) +
  ggtitle("Feature Importance Loss Cost") + ggsave("./Output/GBM/feat_imp_losses.png")

losses_pred <- predict(xgb_model_losses, test_losses)

# Performance Evaluation --------------------------------------------------

test <- test %>% mutate(observed_loss_cost = ClaimAmount / Exposure,
                        predicted_numb = numb_pred,
                        predicted_sev = sev_pred,
                        predicted_loss_cost_freq_sev = numb_pred * sev_pred / Exposure,
                        predicted_loss_cost_tw = losses_pred / Exposure)

eval_dataset <- test %>% select(Exposure, observed_loss_cost, predicted_loss_cost_freq_sev, predicted_loss_cost_tw)

eval_dataset %$% NRMSE(predicted_loss_cost_freq_sev, observed_loss_cost)

eval_dataset %$% NRMSE(predicted_loss_cost_tw, observed_loss_cost)

eval_dataset %$% gini_plot(predicted_loss_cost_freq_sev, Exposure) + ggtitle("Gini index Freq / Sev") + ggsave("./Output/GBM/gini_freq_sev.png")

eval_dataset %$% gini_plot(predicted_loss_cost_tw, Exposure) + ggtitle("Gini index Loss Cost") + ggsave("./Output/GBM/gini_loss_cost.png")

eval_dataset %$% gini_value(predicted_loss_cost_freq_sev, Exposure)

eval_dataset %$% gini_value(predicted_loss_cost_tw, Exposure)

eval_dataset %$% lift_curve_table(predicted_loss_cost_freq_sev, observed_loss_cost, Exposure, 20) %>% 
                 lift_curve_plot() +
                 ggtitle("Lift Curve Freq / Sev") + ggsave("./Output/GBM/lift_curve_freq_sev.png")

eval_dataset %$% lift_curve_table(predicted_loss_cost_tw, observed_loss_cost, Exposure, 20) %>% 
                 lift_curve_plot() +
                 ggtitle("Lift Curve Loss Cost") + ggsave("./Output/GBM/lift_curve_loss_cost.png")

eval_dataset %$% double_lift_chart(predicted_loss_cost_freq_sev, predicted_loss_cost_tw, observed_loss_cost, Exposure, 20, "Freq / Sev", "Loss Cost") + 
                 ggtitle("Double Lift Curve") + ggsave("./Output/GBM/double_lift_curve.png")

# Export Final ------------------------------------------------------------

test %>% select(RecordID,
                Exposure,
                ClaimNb,
                ClaimAmount,
                Severity,
                predicted_numb,
                predicted_sev,
                predicted_loss_cost_freq_sev,
                predicted_loss_cost_tw) %>% 
                mutate(Loss_Cost = ClaimAmount / Exposure) %>% 
                select(RecordID,
                       Exposure,
                       ClaimNb,
                       Severity,
                       Loss_Cost,
                       predicted_numb,
                       predicted_sev,
                       predicted_loss_cost_freq_sev,
                       predicted_loss_cost_tw) %>%
                write_csv("./Output/GBM/dataset_predictions.csv")

