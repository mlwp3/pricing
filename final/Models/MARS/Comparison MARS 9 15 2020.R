library(earth)
library(tweedie)
library(statmod)
library(tidyverse)
library(splines)
library(glmnet)

rm(list=ls())

source("final/Utils/Utils.R")
source("final/Utils/perf_eval.R")

set.seed(42)

train <- import_data("final/Data/train.csv") %>% mutate(severity=ClaimAmount/ClaimNb)
sev_train <- train %>% filter(ClaimAmount>0) #It's about the same if you choose claim amount > 0 or > 100, but we'll stick with >0

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

sev_mod <- earth(severity ~ Area +
                   VehPower +
                   VehBrand +
                   VehGas +
                   Region +
                   DrivAgeBand +
                   DensityBand +
                   VehAgeBand, data = sev_train, weights=ClaimNb, degree = 2, glm=list(family=Gamma(link="log")))
summary(sev_mod)

# sev_mod <- glm(severity ~ (VehBrand=='B11') +
#                (Region=='R21') +
#                and((Area=='C'),(Region=='R21')) +
#                and((VehPower=='7'),(Region=='R21')) +
#                and((Region=='R21'),(DrivAgeBand=='(35,45]')) +
#                and((Region=='R21'),(VehAgeBand=='1')), weights=ClaimNb, data=sev_train, family=Gamma(link="log"))
# summary(sev_mod)

fs_pred <- predict(freq_mod, newdata = test, type="response")*predict(sev_mod, newdata = test, type="response")

FS_rmse <- rmse(test,"ClaimAmount", fs_pred)*100
FS_nrmse <- 1-NRMSE(test,"ClaimAmount", fs_pred)*100
FS_cor_pearson <- as.numeric(cor(fs_pred,test$ClaimAmount))
FS_cor_spearman <- as.numeric(cor(fs_pred,test$ClaimAmount,method="spearman"))
FS_mae <- MAE(fs_pred, test$ClaimAmount)
FS_nmae <- 1-NMAE(fs_pred, test$ClaimAmount)*100
FS_gini <- gini_value(fs_pred,test$Exposure)
FS_agg_rpr <- agg_rpr(test, "ClaimAmount", fs_pred)
FS_norm_rpd <- norm_rp_deviance(test, "ClaimAmount", fs_pred)

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
p <- 1.555102

pp_mod <- earth(ClaimAmount ~ Area +
                  VehPower +
                  VehBrand +
                  VehGas +
                  Region +
                  DrivAgeBand +
                  DensityBand +
                  VehAgeBand + offset(log(Exposure)), data = train, degree = 2, glm=list(family = tweedie(var.power = p, link.power = 0)))
summary(pp_mod)
# 
pp_mod2 <- earth(round(ClaimAmount,0) ~ Area +
                   VehPower +
                   VehBrand +
                   VehGas +
                   Region +
                   DrivAgeBand +
                   DensityBand +
                   VehAgeBand + offset(log(Exposure)), data = train, degree = 2, glm=list(family = poisson(link="log")))

summary(pp_mod2)


pp_pred <- predict(pp_mod, newdata = test, type="response")

pp_rmse <- rmse(test,"ClaimAmount", pp_pred)*100
pp_nrmse <- 1-NRMSE(test,"ClaimAmount", pp_pred)*100
pp_cor_pearson <- as.numeric(cor(pp_pred,test$ClaimAmount))
pp_cor_spearman <- as.numeric(cor(pp_pred,test$ClaimAmount,method="spearman"))
pp_mae <- MAE(pp_pred, test$ClaimAmount)
pp_nmae <- 1-NMAE(pp_pred, test$ClaimAmount)*100
pp_gini <- gini_value(pp_pred,test$Exposure)
pp_agg_rpr <- agg_rpr(test, "ClaimAmount", pp_pred)
pp_norm_rpd <- norm_rp_deviance(test, "ClaimAmount", pp_pred)

rmse_compiled <- c(FS_rmse, pp_rmse)
nrmse_compiled <- c(FS_nrmse, pp_nrmse)
nmae_compiled <- c(FS_nmae, pp_nmae)
mae_compiled <- c(FS_mae, pp_mae)
pearson_cor_compiled <- c(FS_cor_pearson, pp_cor_pearson)
spearman_cor_compiled <- c(FS_cor_spearman, pp_cor_spearman)
gini_index_compiled <- c(FS_gini, pp_gini)
agg_rpr_compiled <- c(FS_agg_rpr, pp_agg_rpr)
norm_rpd_compiled <- c(FS_norm_rpd, pp_norm_rpd)

model_metrics <- data.frame(c("MARS Freq/Sev","MARS PP"), rmse_compiled, mae_compiled, pearson_cor_compiled, spearman_cor_compiled, gini_index_compiled, agg_rpr_compiled)
colnames(model_metrics) <- c("Model", "RMSE", "MAE", "Pearson (Linear) Correlation", "Spearman Rank Correlation", "Gini Index", "Aggregate Risk Premium Differential")
eval_metrics <- data.frame(gini_index_compiled, nrmse_compiled, nmae_compiled, spearman_cor_compiled, norm_rpd_compiled)
lscore <- apply(eval_metrics, 1, mean)
gscore <- apply(eval_metrics, 1, geometric_mean)
hscore <- apply(eval_metrics, 1, harmonic_mean)

model_metrics$lscore <- lscore
model_metrics$gscore <- gscore
model_metrics$hscore <- hscore
model_metrics[is.na(model_metrics)] <- 0

write.csv(model_metrics,"final/Output/MARS/MARS_model_evaluation_stats.csv")

test <- test %>% mutate(MARS_FS = fs_pred, MARS_PP = pp_pred)
write_csv(test, "final/Output/MARS/test_w_predictions.csv")
