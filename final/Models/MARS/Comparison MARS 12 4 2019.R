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
gini_plot(as.numeric(fs_pred),test$exposure)
gini_value(as.numeric(fs_pred),test$exposure)


train$ClaimAmount <- NULL
train$Freq <- NULL
train$ClaimInd <- NULL
train$exposure <- NULL
pp_mod <- earth(PurePrem ~ ., data=train, degree=3)

pp_pred <- predict(pp_mod, newdata=test)*test$exposure
RMSE(pp_pred,test$ClaimAmount) #2025.51
NRMSE(pp_pred,test$ClaimAmount) #0.01307142
gini_plot(as.numeric(pp_pred),test$exposure)
gini_value(as.numeric(pp_pred),test$exposure)
