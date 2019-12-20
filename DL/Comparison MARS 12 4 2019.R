library(earth)
library(caret)
library(plyr)
library(tidyverse)
library(broom)
library(boot)

rm(list=ls())
NRMSE <- function(pred, obs){
	
	RMSE(pred, obs)/(max(obs)-min(obs))
	
}

train <- read.csv("https://raw.githubusercontent.com/mlwp3/pricing/master/final/Data/train_final.csv")

train$Freq <- train$ClaimInd / train$exposure
train$PurePrem <- train$ClaimAmount / train$exposure

test <- read.csv("https://raw.githubusercontent.com/mlwp3/pricing/master/final/Data/test_final.csv")

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

freq_mod <- earth(Freq ~ ., data=freq_train, degree=3)

#The severity model is based on only those claims for which the claim amount is nonzero.
sev_train <- train[which(train$ClaimAmount>0),]

sev_train$PurePrem <- NULL
sev_train$X <- NULL
sev_train$Freq <- NULL
sev_train$exposure <- NULL
sev_mod <- earth(ClaimAmount ~ ., data=sev_train, degree=3)

fs_pred <- predict(freq_mod, newdata=test)*predict(sev_mod, newdata=test)*test$exposure
RMSE(fs_pred,test$ClaimAmount) #2238.138
NRMSE(fs_pred,test$ClaimAmount) #0.01444359


train$ClaimAmount <- NULL
train$Freq <- NULL
train$ClaimInd <- NULL
train$exposure <- NULL
pp_mod <- earth(PurePrem ~ ., data=train, degree=3)

pp_pred <- predict(pp_mod, newdata=test)*test$exposure
RMSE(pp_pred,test$ClaimAmount) #2025.51
NRMSE(pp_pred,test$ClaimAmount) #0.01307142
