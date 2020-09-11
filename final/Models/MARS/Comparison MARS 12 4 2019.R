library(earth)
library(tidyverse)

source("final/Utils/utils.R")

train <- import_data("final/Data/train_new_final.csv") %>% mutate(id = "train")

test <- import_data("final/Data/test_new_final.csv") %>% mutate(id = "test")

train$Freq <- train$ClaimNb / train$Exposure
train$PurePrem <- train$ClaimAmount / train$Exposure

test$Freq <- test$ClaimNb / test$Exposure
test$PurePrem <- test$ClaimAmount / test$Exposure

#Remove some variables from the training set.
#This is because the earth package automatically considers every possible variable, and I don't want it
#to try to predict the claim frequency based on the claim indicator, for example. That would be cheating!
freq_train <- train
freq_train$Exposure <- NULL
freq_train$ClaimNb <- NULL
freq_train$ClaimAmount <- NULL
freq_train$PurePrem <- NULL
freq_train$RecordID <- NULL

freq_mod <- freq_train %>% 
  select(-id) %>% 
  earth(Freq ~ ., data = ., degree = 3)


#The severity model is based on only those claims for which the claim amount is nonzero.
sev_train <- train[which(train$ClaimAmount > 0),]

sev_train$PurePrem <- NULL
sev_train$RecordID <- NULL
sev_train$Freq <- NULL
sev_train$Exposure <- NULL
sev_mod <- sev_train %>% 
  select(-id) %>% 
  earth(ClaimAmount ~ ., data = ., degree = 3)

fs_pred <- predict(freq_mod, newdata = test)*predict(sev_mod, newdata = test) * test$Exposure
RMSE(fs_pred,test$ClaimAmount) #2069.684
NRMSE(fs_pred,test$ClaimAmount) #0.005296801
gini_value(as.numeric(fs_pred),test$Exposure) #-0.05571802

gini_plot_fs <- gini_plot(as.numeric(fs_pred),test$Exposure) + 
  ggtitle("Frequency x Severity MARS Gini Plot")

tbl_lift_fs <- lift_curve_table(as.numeric(fs_pred), test$ClaimAmount, test$Exposure,10)
plt_lift_fs <- tbl_lift_fs %>% 
  lift_curve_plot() +   
  labs(col = "Model")
plt_lift_fs

train$ClaimAmount <- NULL
train$Freq <- NULL
train$ClaimInd <- NULL
train$Exposure <- NULL
pp_mod <- train %>% 
  select(-id) %>% 
  earth(PurePrem ~ ., data = ., degree = 3)

pp_pred <- predict(pp_mod, newdata = test)*test$Exposure
RMSE(pp_pred,test$ClaimAmount) #2055.142
NRMSE(pp_pred,test$ClaimAmount) #0.005259585
gini_value(as.numeric(pp_pred),test$Exposure) #0.9236354

gini_plot_pp <- gini_plot(as.numeric(pp_pred),test$Exposure) + 
  ggtitle("Pure Premium MARS Gini Plot")

tbl_lift_pp <- lift_curve_table(as.numeric(pp_pred),test$ClaimAmount,test$Exposure,10)

plt_lift_pp <- tbl_lift_pp %>% 
  lift_curve_plot() + 
  labs(col = "Model")

plt_lift_pp
