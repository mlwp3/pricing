library(earth)
library(tidyverse)

source("final/Utils/utils.R")

train <- import_data("final/Data/train_final.csv") %>% mutate(id = "train")

test <- import_data("final/Data/test_final.csv") %>% mutate(id = "test")

train$Freq <- train$ClaimInd / train$exposure
train$PurePrem <- train$ClaimAmount / train$exposure

test$Freq <- test$ClaimInd / test$exposure
test$PurePrem <- test$ClaimAmount / test$exposure

#Remove some variables from the training set.
#This is because the earth package automatically considers every possible variable, and I don't want it
#to try to predict the claim frequency based on the claim indicator, for example. That would be cheating!
freq_train <- train
freq_train$exposure <- NULL
freq_train$ClaimInd <- NULL
freq_train$ClaimAmount <- NULL
freq_train$PurePrem <- NULL
freq_train$X <- NULL

freq_mod <- freq_train %>% 
  select(-id) %>% 
  earth(Freq ~ ., data = ., degree = 3)

#The severity model is based on only those claims for which the claim amount is nonzero.
sev_train <- train[which(train$ClaimAmount > 0),]

sev_train$PurePrem <- NULL
sev_train$X <- NULL
sev_train$Freq <- NULL
sev_train$exposure <- NULL
sev_mod <- sev_train %>% 
  select(-id) %>% 
  earth(ClaimAmount ~ ., data = ., degree = 3)

fs_pred <- predict(freq_mod, newdata = test)*predict(sev_mod, newdata = test) * test$exposure
RMSE(fs_pred,test$ClaimAmount) #2238.138
NRMSE(fs_pred,test$ClaimAmount) #0.01444359
gini_value(as.numeric(fs_pred),test$exposure)

gini_plot_fs <- gini_plot(as.numeric(fs_pred),test$exposure) + 
  ggtitle("Frequency x Severity MARS Gini Plot")

lift_curve_fs <- lift_curve_plot(as.numeric(fs_pred),test$ClaimAmount,test$exposure,10) +  
  labs(col = "Model") + ggtitle("Frequency x Severity MARS Lift Curve")

train$ClaimAmount <- NULL
train$Freq <- NULL
train$ClaimInd <- NULL
train$exposure <- NULL
pp_mod <- train %>% 
  select(-id) %>% 
  earth(PurePrem ~ ., data = ., degree = 3)

pp_pred <- predict(pp_mod, newdata = test)*test$exposure
RMSE(pp_pred,test$ClaimAmount) #2025.51
NRMSE(pp_pred,test$ClaimAmount) #0.01307142
gini_value(as.numeric(pp_pred),test$exposure)

gini_plot_pp <- gini_plot(as.numeric(pp_pred),test$exposure) + 
  ggtitle("Pure Premium MARS Gini Plot")

lift_curve_pp <- lift_curve_plot(as.numeric(pp_pred),test$ClaimAmount,test$exposure,10) + 
  labs(col = "Model") + ggtitle("Pure Premium MARS Lift Curve")
