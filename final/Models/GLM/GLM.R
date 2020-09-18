rm(list = ls())

set.seed(999)

setwd("/home/marco/Documents/gitrepos/pricing/final")

source("./Utils/utils.R")

# Import Data -------------------------------------------------------------

train <- import_data("./Data/train_new_final.csv") %>% 
           mutate(Severity = ifelse(ClaimNb == 0, 0, ClaimAmount / ClaimNb))

test <- import_data("./Data/test_new_final.csv") %>% 
          mutate(Severity = ifelse(ClaimNb == 0, 0, ClaimAmount / ClaimNb))

# train_model <- model.matrix(~ Area +
#                          VehPower +
#                          VehBrand +
#                          VehGas +
#                          Region +
#                          DrivAgeBand +
#                          DensityBand +
#                          VehAgeBand + 
#                          Exposure + 
#                          0, 
#                        data = train) %>% 
#          cbind(ClaimNb = train$ClaimNb,
#                ClaimAmount = train$ClaimAmount,
#                Severity = train$Severity) %>%  
#          as.data.frame()
# 
# train_model$`DensityBand(5e+03,Inf]` <- NULL
# 
# test_model <- model.matrix(~ Area +
#                         VehPower +
#                         VehBrand +
#                         VehGas +
#                         Region +
#                         DrivAgeBand +
#                         DensityBand +
#                         VehAgeBand + 
#                         Exposure +
#                         0, 
#                       data = test) %>%
#         as.data.frame()
# 
# test_model$`DensityBand(5e+03,Inf]` <- NULL

# glm_model_numb <- read_rds("./Models/GLM/glm_numbers.RDS")
# 
# glm_model_sev <- read_rds("./Models/GLM/glm_severity.RDS")
# 
# glm_model_losses <- read_rds("./Models/GLM/glm_losses.RDS")

# Model Section -----------------------------------------------------------

# Numbers

glm_model_numb <- glm(ClaimNb ~
                        Area +
                        VehPower +
                        VehBrand +
                        VehGas +
                        Region +
                        DrivAgeBand +
                        DensityBand +
                        VehAgeBand +
                        offset(log(Exposure)) +
                        0,
                      family = poisson(link = "log"),
                      data = train)

glm_model_numb <- glm(ClaimNb ~
                        Area +
                        VehPower +
                        VehBrand +
                        VehGas +
                        Region +
                        DrivAgeBand +
                        DensityBand +
                        VehAgeBand +
                        offset(log(Exposure)) +
                        0,
                      family = poisson(link = "log"),
                      data = train)

summary(glm_model_numb)

# write_rds(glm_model_numb %>% strip_glm(), "./Models/GLM/glm_numbers.RDS")

numb_pred <- predict(glm_model_numb, newdata = test, type = "response")

size(glm_model_numb)


# Severity

glm_model_sev <- glm(Severity ~
                        Area +
                        VehPower +
                        VehBrand +
                        VehGas +
                        Region +
                        DrivAgeBand +
                        DensityBand +
                        VehAgeBand +
                        0,
                     weights = ClaimNb,
                    family = Gamma(link = "log"),
                    data = train,
                    subset = ClaimNb > 0)

summary(glm_model_sev)

# write_rds(glm_model_sev %>% strip_glm(), "./Models/GLM/glm_severity.RDS")

sev_pred <- predict(glm_model_sev, newdata = test, type = "response")

# Losses
# 
# prof <- tweedie.profile(ClaimAmount ~
#                           Area +
#                           VehPower +
#                           VehBrand +
#                           VehGas +
#                           Region +
#                           DrivAgeBand +
#                           DensityBand +
#                           VehAgeBand +
#                           offset(log(Exposure)) +
#                           0,
#                         data = train)

glm_model_losses <- glm(ClaimAmount ~
                         Area +
                         VehPower +
                         VehBrand +
                         VehGas +
                         Region +
                         DrivAgeBand +
                         DensityBand +
                         VehAgeBand +
                         offset(log(Exposure)) +
                         0,
                     family = tweedie(var.power = 1.55102, link.power = 0),
                     data = train)

summary(glm_model_losses)

# write_rds(glm_model_losses %>% strip_glm(), "./Models/GLM/glm_losses.RDS")

losses_pred <- predict(glm_model_losses, newdata = test, type = "response")

# Performance Evaluation --------------------------------------------------

test <- test %>% mutate(observed_loss_cost = ClaimAmount / Exposure,
                        predicted_numb = numb_pred,
                        predicted_sev = sev_pred,
                        predicted_loss_cost_freq_sev = numb_pred * sev_pred / Exposure,
                        predicted_loss_cost_tw = losses_pred / Exposure)

eval_dataset <- test %>% select(Exposure, observed_loss_cost, predicted_loss_cost_freq_sev, predicted_loss_cost_tw)

eval_dataset %$% NRMSE(predicted_loss_cost_freq_sev, observed_loss_cost)

eval_dataset %$% NRMSE(predicted_loss_cost_tw, observed_loss_cost)

eval_dataset %$% gini_plot(predicted_loss_cost_freq_sev, Exposure) + ggtitle("Gini index Freq / Sev") + ggsave("./Output/GLM/gini_freq_sev.png")

eval_dataset %$% gini_plot(predicted_loss_cost_tw, Exposure) + ggtitle("Gini index Loss Cost") + ggsave("./Output/GLM/gini_loss_cost.png")

eval_dataset %$% gini_value(predicted_loss_cost_freq_sev, Exposure)

eval_dataset %$% gini_value(predicted_loss_cost_tw, Exposure)

eval_dataset %$% lift_curve_table(predicted_loss_cost_freq_sev, observed_loss_cost, Exposure, 20) %>% 
  lift_curve_plot() +
  ggtitle("Lift Curve Freq / Sev") + ggsave("./Output/GLM/lift_curve_freq_sev.png")

eval_dataset %$% lift_curve_table(predicted_loss_cost_tw, observed_loss_cost, Exposure, 20) %>% 
  lift_curve_plot() +
  ggtitle("Lift Curve Loss Cost") + ggsave("./Output/GLM/lift_curve_loss_cost.png")

eval_dataset %$% double_lift_chart(predicted_loss_cost_freq_sev, predicted_loss_cost_tw, observed_loss_cost, Exposure, 20, "Freq / Sev", "Loss Cost") + 
  ggtitle("Double Lift Curve") + ggsave("./Output/GLM/double_lift_curve.png")

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
  write_csv("./Output/GLM/dataset_predictions.csv")

