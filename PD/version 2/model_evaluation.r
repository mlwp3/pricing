
source("./perf_eval.R")

setwd("C:/Users/pjdys/Documents/git repos/mlwp3/pricing/PD/version 2")
test_output <- read_csv("FS Output.csv")

FS_rmse      <-  rmse(data = test_output, targetvar = "ClaimAmount", prediction.obj = test_output$predicted_total)
FS_nrmse    <- NRMSE(test_output, "ClaimAmount", test_output$predicted_total)
FS_mae      <- MAE(test_output$predicted_total, test_output$ClaimAmount)
FS_nmae     <- NMAE(test_output$predicted_total, test_output$ClaimAmount)
FS_agg_rpr  <- agg_rpr(test_output, "ClaimAmount", test_output$predicted_total)
FS_norm_rpd <- norm_rp_deviance(test_output, "ClaimAmount", test_output$predicted_total)


FS_gini     <- gini_value(test_output$predicted_total, test_output$ClaimAmount)
FS_cor_pearson <- as.numeric(cor(test_output$predicted_total, test_output$ClaimAmount))
FS_cor_spearman <- as.numeric(cor(test_output$predicted_total, test_output$ClaimAmount, method = "spearman"))

model_metrics <- data.frame('RF F/S', FS_rmse, FS_mae, FS_cor_pearson, FS_cor_spearman, FS_gini, FS_agg_rpr)


