

setwd("C:/Users/pjdys/Documents/git repos/mlwp3/pricing/final/Models/RF")
source("../../Utils/utils.R")
source("./perf_eval.R")
#import data
test_output <- read_csv("FS Test Predictions Output.csv")

#calc metrics
FS_rmse     <- RMSE(test_output$ClaimAmount, test_output$predicted_total)
FS_nrmse    <- NRMSE(test_output, "ClaimAmount", test_output$predicted_total)
FS_mae      <- MAE(test_output$predicted_total, test_output$ClaimAmount)
FS_nmae     <- NMAE(test_output$predicted_total, test_output$ClaimAmount)
FS_agg_rpr  <- agg_rpr(test_output, "ClaimAmount", test_output$predicted_total)
FS_norm_rpd <- norm_rp_deviance(test_output, "ClaimAmount", test_output$predicted_total)
FS_gini     <- gini_value(test_output$predicted_total, test_output$ClaimAmount)
FS_cor_pearson <- as.numeric(cor(test_output$predicted_total, test_output$ClaimAmount))
FS_cor_spearman <- as.numeric(cor(test_output$predicted_total, test_output$ClaimAmount, method = "spearman"))

#output to table
model_metrics <- data.frame('RF F/S', FS_rmse, FS_mae, FS_cor_pearson, FS_cor_spearman, FS_gini, FS_agg_rpr)
colnames(model_metrics) <- c("Model", "RMSE", "MAE", "Pearson (Linear) Correlation", "Spearman Rank Correlation", "Gini Index", "Aggregate Risk Premium Differential")

write_csv(model_metrics, "../../Output/RF/FS_RF_model_evaluation_stats.csv")

# write charts

test_output %$% gini_plot(predicted_total, Exposure) + ggtitle("Gini index Freq/Sev") + ggsave("../../Output/RF/FS_gini.png",  scale = 3, dpi = 300)

test_output %$% lift_curve_table(predicted_total, ClaimAmount, Exposure, 20) %>% 
  lift_curve_plot() + ggtitle("Lift Curve Freq / Sev") + ggsave("../../Output/RF/FS_lift_curve.png",  scale = 3, dpi = 300)
