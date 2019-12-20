library(rstan)
library(tidyverse)
library(broom)
library(glmnet)
library(boot)
library(lme4)

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

mod8 <- glm(Freq~DrivAge+LicAge+VehAge+HasKmLimit*BonusMalus,
						data=train, family=quasipoisson(link="log"))

sev_train <- train[which(train$ClaimAmount>0),]

smod2 <- glm(ClaimAmount ~ VehPriceGrp+VehEngine+DrivAge*VehBody+BonusMalus, data=sev_train, family=gaussian(link="log"))

fs_pred <- predict.glm(mod8, newdata=test, type="response")*test$exposure*predict.glm(smod2, newdata=test, type="response")
RMSE(fs_pred,test$ClaimAmount) #2132.452
NRMSE(fs_pred,test$ClaimAmount) #0.01376156
