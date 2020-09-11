library(earth)
library(tweedie)
library(statmod)
library(tidyverse)
source("final/Utils/utils.R")
train <- import_data("final/Data/train_new_final.csv") %>% mutate(id = "train")
train$severity <- train$ClaimAmount / train$ClaimNb

test <- import_data("final/Data/test_new_final.csv") %>% mutate(id = "test")

freq_mod <- earth(ClaimNb ~ Area +
                    VehPower +
                    VehBrand +
                    VehGas +
                    Region +
                    DrivAgeBand +
                    DensityBand + 
                    VehAgeBand + offset(log(Exposure)), data = train, degree = 3, glm=list(family=poisson(link="log")))
summary(freq_mod)

#The severity model is based on only those claims for which the claim amount is nonzero.
sev_train <- train[which(train$ClaimAmount > 0),]


sev_mod <- earth(severity ~ Area +
                   VehPower +
                   VehBrand +
                   VehGas +
                   Region +
                   DrivAgeBand +
                   DensityBand + 
                   VehAgeBand, data = sev_train, weights=ClaimNb, degree = 3, glm=list(family=Gamma(link="log")))

summary(sev_mod)
summary(sev_train$severity)

# fs_pred <- predict(freq_mod, newdata = test)*predict(sev_mod, newdata = test)
# RMSE(fs_pred,test$ClaimAmount) #2069.684
# NRMSE(fs_pred,test$ClaimAmount) #0.005296801
# gini_value(as.numeric(fs_pred),test$Exposure) #-0.05571802
# 
# gini_plot_fs <- gini_plot(as.numeric(fs_pred),test$Exposure) + 
#   ggtitle("Frequency x Severity MARS Gini Plot")
# 
# tbl_lift_fs <- lift_curve_table(as.numeric(fs_pred), test$ClaimAmount, test$Exposure,10)
# plt_lift_fs <- tbl_lift_fs %>% 
#   lift_curve_plot() +   
#   labs(col = "Model")
# plt_lift_fs

p_values <- seq(1.3, 1.6, length = 8)
p_tuning <- tweedie.profile(ClaimAmount ~ Area +
                              VehPower +
                              VehBrand +
                              VehGas +
                              Region +
                              DrivAgeBand +
                              DensityBand + 
                              VehAgeBand + offset(log(Exposure)), data = train, p.vec = p_values, do.plot = FALSE, 
                              verbose = 2, do.smooth = TRUE, method = "series", fit.glm = FALSE)
p <- p_tuning$p.max
rm(p_values)

pp_mod <- earth(ClaimAmount ~ Area +
                  VehPower +
                  VehBrand +
                  VehGas +
                  Region +
                  DrivAgeBand +
                  DensityBand + 
                  VehAgeBand + offset(log(Exposure)), data = train, degree = 3, glm=list(family = tweedie(var.power = p, link.power = 0)))

summary(pp_mod)

# pp_pred <- predict(pp_mod, newdata = test)
# RMSE(pp_pred,test$ClaimAmount) #2055.142
# NRMSE(pp_pred,test$ClaimAmount) #0.005259585
# gini_value(as.numeric(pp_pred),test$Exposure) #0.9236354
# 
# gini_plot_pp <- gini_plot(as.numeric(pp_pred),test$Exposure) + 
#   ggtitle("Pure Premium MARS Gini Plot")
# 
# tbl_lift_pp <- lift_curve_table(as.numeric(pp_pred),test$ClaimAmount,test$Exposure,10)
# 
# plt_lift_pp <- tbl_lift_pp %>% 
#   lift_curve_plot() + 
#   labs(col = "Model")
# 
# plt_lift_pp
