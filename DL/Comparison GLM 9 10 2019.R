library(rstan)
library(tidyverse)
library(broom)
library(glmnet)
library(boot)
library(lme4)
library(Matrix)
library(readr)
library(dplyr)
library(magrittr)
library(pracma)

rm(list=ls())

gini_plot <- function(predicted_loss_cost, exposure){
	
	lc_var <- enquo(predicted_loss_cost)
	exp <- enquo(exposure)
	
	dataset <- tibble(lc_var = !! lc_var, exp = !! exp)
	
	dataset %>% 
		arrange(lc_var) %>% 
		mutate(cum_exp = cumsum(exp) / sum(exp),
					 cum_pred_lc = cumsum(lc_var) / sum(lc_var)) %>% 
		ggplot()+
		geom_line(aes(x = cum_exp, y = cum_pred_lc))+
		geom_abline(intercept = 0, slope = 1)+
		xlab("Exposure")+
		ylab("Predicted Loss Cost")
	
}

gini_value <- function(predicted_loss_cost, exposure){
	
	lc_var <- enquo(predicted_loss_cost)
	exp <- enquo(exposure)
	
	dataset <- tibble(lc_var = !! lc_var, exp = !! exp)
	
	dataset %>% 
		arrange(lc_var) %>% 
		mutate(cum_exp = cumsum(exp) / sum(exp),
					 cum_pred_lc = cumsum(lc_var) / sum(lc_var)) %$% 
					 {trapz(cum_exp, cum_pred_lc) %>% add(-1) %>% abs() %>% subtract(.5) %>% multiply_by(2)}
}

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
gini_plot(fs_pred,test$exposure)
gini_value(fs_pred,test$exposure)

