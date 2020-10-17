library(earth)
library(tweedie)
library(statmod)
library(tidyverse)
library(splines)
library(glmnet)

rm(list=ls())

source("final/Utils/Utils.R")
source("final/Utils/perf_eval.R")

train <- import_data("final/Data/train.csv") %>% mutate(severity=ClaimAmount/ClaimNb)
sev_train <- train %>% filter(ClaimAmount>0) #It's about the same if you choose claim amount > 0 or > 100

test <- import_data("final/Data/test.csv") %>% mutate(severity=ClaimAmount/ClaimNb)

freq_mod <- earth(ClaimNb ~ Area +
                    VehPower +
                    VehBrand +
                    VehGas +
                    Region +
                    DrivAgeBand +
                    DensityBand + 
                    VehAgeBand + offset(log(Exposure)), data = train, degree = 2, glm=list(family=poisson(link="log")))
summary(freq_mod)

# fmod <- glm(ClaimNb ~ (VehBrand=='B12')+
#               (VehGas=='Regular') +
#               (Region=='R24') +
#               (Region=='R53') +
#               DrivAgeBand +
#               (VehBrand=='B12'):(DrivAgeBand=='(65,Inf]'), offset=log(Exposure), data=train, family=poisson(link="log"))
# summary(fmod)

sev_mod <- earth(pmin(severity,5000) ~ Area +
                   VehPower +
                   VehBrand +
                   VehGas +
                   Region +
                   DrivAgeBand +
                   DensityBand + 
                   VehAgeBand, data = sev_train, weights=ClaimNb, degree = 1, glm=list(family=Gamma(link="log")))
summary(sev_mod)

# smod <- glm(pmin(severity,5000) ~
#               (Region=='R93') +
#               (VehBrand=='B12'), weights=ClaimNb, data=sev_train, family=Gamma(link="log"))
# summary(smod)

fs_pred <- predict(freq_mod, newdata = test, type="response")*predict(sev_mod, newdata = test, type="response")

test <- test %>% mutate(observed_loss_cost = ClaimAmount / Exposure,
                        predicted_loss_cost_fs = fs_pred / Exposure)
FS_rmse <- rmse(test,"observed_loss_cost", test$predicted_loss_cost_fs)*100 #73.52
FS_nrmse <- 1-NRMSE(test,"observed_loss_cost", test$predicted_loss_cost_fs)*100 #0.3684164
FS_cor_pearson <- as.numeric(cor(test$predicted_loss_cost_fs,test$observed_loss_cost))
FS_cor_spearman <- as.numeric(cor(test$predicted_loss_cost_fs,test$observed_loss_cost,method="spearman"))
FS_mae <- MAE(test$predicted_loss_cost_fs, test$observed_loss_cost)
FS_nmae <- 1-NMAE(test$predicted_loss_cost_fs, test$observed_loss_cost)*100
FS_gini <- gini_value(test$predicted_loss_cost_fs,test$Exposure) #0.1756933
FS_agg_rpr <- agg_rpr(test, "observed_loss_cost", test$predicted_loss_cost_fs)
FS_norm_rpd <- norm_rp_deviance(test, "observed_loss_cost", test$predicted_loss_cost_fs)

model_metrics <- data.frame("MARS Freq/Sev", FS_rmse, FS_mae, FS_cor_pearson, FS_cor_spearman, FS_gini, FS_agg_rpr)
colnames(model_metrics) <- c("Model", "RMSE", "MAE", "Pearson (Linear) Correlation", "Spearman Rank Correlation", "Gini Index", "Aggregate Risk Premium Differential")
eval_metrics <- data.frame(FS_gini, FS_nrmse, FS_nmae, FS_cor_spearman, FS_norm_rpd)
lscore <- apply(eval_metrics, 1, mean)
gscore <- apply(eval_metrics, 1, geometric_mean)
hscore <- apply(eval_metrics, 1, harmonic_mean)
model_metrics$lscore <- lscore
model_metrics$gscore <- gscore
model_metrics$hscore <- hscore
model_metrics[is.na(model_metrics)] <- 0

write.csv(model_metrics,"final/Output/MARS/MARS_model_evaluation_stats.csv")

#Pure Premium MARS model basically doesn't work at all. Not yet at least?
#It picks up no predictors at all.
#I think that the severities are messing it up?
#####################################################
# p_values <- seq(1.4, 1.6, length = 8)
# p_tuning <- tweedie.profile(ClaimAmount ~ Area +
#                               VehPower +
#                               VehBrand +
#                               VehGas +
#                               Region +
#                               DrivAgeBand +
#                               DensityBand + 
#                               VehAgeBand + offset(log(Exposure)), data = train, p.vec = p_values, do.plot = FALSE, 
#                               verbose = 2, do.smooth = TRUE, method = "series", fit.glm = FALSE)
# p <- p_tuning$p.max #1.555102
# p_tuning$p.max
# rm(p_values)
# 
# p <- 1.555102
# 
# pp_mod <- earth(pmin(ClaimAmount,5000) ~ Area +
#                   VehPower +
#                   VehBrand +
#                   VehGas +
#                   Region +
#                   DrivAgeBand +
#                   DensityBand + 
#                   VehAgeBand + offset(log(Exposure)), data = train, degree = 2, glm=list(family = tweedie(var.power = p, link.power = 0)))
# summary(pp_mod)
# 
# pp_mod2 <- earth(pmin(round(ClaimAmount,0),5000) ~ Area +
#                    VehPower +
#                    VehBrand +
#                    VehGas +
#                    Region +
#                    DrivAgeBand +
#                    DensityBand + 
#                    VehAgeBand + offset(log(Exposure)), data = train, degree = 2, glm=list(family = poisson(link="log")))
# 
# summary(pp_mod2)
# 
# pp_mod <- glm(pmin(round(ClaimAmount,0),5000) ~ Area +
#                   VehPower +
#                   VehBrand +
#                   VehGas +
#                   Region +
#                   DrivAgeBand +
#                   DensityBand + 
#                   VehAgeBand + offset(log(Exposure)), data = train, family = poisson(link="log"))
# summary(pp_mod)
# 
# plot(density(log(sev_train$severity)))
# plot(density(log(sev_train$ClaimAmount)))
# 
# pp_pred <- predict(pp_mod, newdata = test, type="response")
# pp_pred2 <- predict(pp_mod2, newdata=test, type="response")
# 
# rmse(test, "ClaimAmount",pp_pred) #19.94898
# rmse(test, "ClaimAmount",pp_pred2) #19.95066
# NRMSE(test, "ClaimAmount", pp_pred) #.000499707
# NRMSE(test, "ClaimAmount", pp_pred2) #.000499749
# 
