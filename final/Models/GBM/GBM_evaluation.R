rm(list = ls())

set.seed(999)

setwd("/home/marco/Documents/gitrepos/pricing/final")

source("./Utils/perf_eval.R")

test_set <- read_csv("./Output/GBM/dataset_predictions_gbm.csv")

LC_rmse <-  rmse(data = test_set, targetvar = "Loss_Cost", prediction.obj = test_set$predicted_loss_cost)
LC_nrmse <- NRMSE(test_set, "Loss_Cost", test_set$predicted_loss_cost)
LC_cor_pearson <- as.numeric(cor(test_set$predicted_loss_cost, test_set$Loss_Cost))
LC_cor_spearman <- as.numeric(cor(test_set$predicted_loss_cost, test_set$Loss_Cost, method = "spearman"))
LC_mae <- MAE(test_set$predicted_loss_cost, test_set$Loss_Cost)
LC_nmae <- NMAE(test_set$predicted_loss_cost, test_set$Loss_Cost)
LC_gini <- gini_value(test_set$predicted_loss_cost, test_set$Exposure)
LC_agg_rpr <- agg_rpr(test_set, "Loss_Cost", test_set$predicted_loss_cost)
LC_norm_rpd <- norm_rp_deviance(test_set, "Loss_Cost", test_set$predicted_loss_cost)

FS_rmse <-  rmse(data = test_set, targetvar = "Loss_Cost", prediction.obj = test_set$predicted_loss_cost_freq_sev)
FS_nrmse <- NRMSE(test_set, "Loss_Cost", test_set$predicted_loss_cost_freq_sev)
FS_cor_pearson <- as.numeric(cor(test_set$predicted_loss_cost_freq_sev, test_set$Loss_Cost))
FS_cor_spearman <- as.numeric(cor(test_set$predicted_loss_cost_freq_sev, test_set$Loss_Cost, method = "spearman"))
FS_mae <- MAE(test_set$predicted_loss_cost_freq_sev, test_set$Loss_Cost)
FS_nmae <- NMAE(test_set$predicted_loss_cost_freq_sev, test_set$Loss_Cost)
FS_gini <- gini_value(test_set$predicted_loss_cost_freq_sev, test_set$Exposure)
FS_agg_rpr <- agg_rpr(test_set, "Loss_Cost", test_set$predicted_loss_cost_freq_sev)
FS_norm_rpd <- norm_rp_deviance(test_set, "Loss_Cost", test_set$predicted_loss_cost_freq_sev)


models <- c("GBM LC", "GBM Freq/Sev")
rmse_compiled <- c(LC_rmse, FS_rmse) * 100
nrmse_compiled <- 1 - (c(LC_nrmse, FS_nrmse) * 100)
nmae_compiled <- 1 - (c(LC_nmae, FS_nmae) * 100)
mae_compiled <- c(LC_mae, FS_mae)
pearson_cor_compiled <- c(LC_cor_pearson, FS_cor_pearson)
spearman_cor_compiled <- c(LC_cor_spearman, FS_cor_spearman)
gini_index_compiled <- c(LC_gini, FS_gini)
agg_rpr_compiled <- c(LC_agg_rpr, FS_agg_rpr)
norm_rpd_compiled <- c(LC_norm_rpd, FS_norm_rpd)
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

write_csv(model_metrics, "./Output/GBM/GBM_model_evaluation_stats.csv")
