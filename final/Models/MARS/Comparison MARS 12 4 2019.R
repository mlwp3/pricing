library(earth)
library(caret)
library(plyr)
library(tidyverse)
library(broom)
library(boot)
library(Matrix)
library(readr)
library(dplyr)
library(magrittr)
library(pracma)

rm(list=ls())

NRMSE <- function(pred, obs){
	
	RMSE(pred, obs)/(max(obs)-min(obs))
	
}

gini_value <- function(predicted_loss_cost, exposure) {
	lc_var <- enquo(predicted_loss_cost)
	exp <- enquo(exposure)
	dataset <- tibble(lc_var = !! lc_var, exp = !! exp)
	dataset %>% 
		arrange(lc_var) %>% 
		mutate(cum_exp = cumsum(exp) / sum(exp),
					 cum_pred_lc = cumsum(lc_var) / sum(lc_var)) %$% 
					 {trapz(cum_exp, cum_pred_lc) %>% add(-1) %>% abs() %>% subtract(.5) %>% multiply_by(2)}
}

#Lift Curve
lift_curve_plot <- function(predicted_loss_cost, observed_loss_cost, exposure, n) {
	pred_lc <- enquo(predicted_loss_cost)
	obs_lc <- enquo(observed_loss_cost)
	exp <- enquo(exposure)
	dataset <- tibble(pred_lc = !! pred_lc, obs_lc = !! obs_lc, exp = !! exp)
	dataset <- dataset %>% 
		arrange(exp) %>% 
		mutate(buckets = ntile(exp, n)) %>% 
		group_by(buckets) %>% 
		summarise(Predicted_Risk_Premium = mean(pred_lc),
							Observed_Risk_Premium = mean(obs_lc), 
							Exposure = sum(exp))
	max_bucket <- which.max(dataset$Exposure)
	base_predicted_rp <- dataset[max_bucket, ]$Predicted_Risk_Premium
	base_observed_rp <- dataset[max_bucket, ]$Observed_Risk_Premium
	dataset <- dataset %>%
		select(-Exposure) %>%
		mutate(Predicted_Risk_Premium = Predicted_Risk_Premium / base_predicted_rp, 
					 Observed_Risk_Premium = Observed_Risk_Premium / base_observed_rp)  %>%
		tidyr::pivot_longer(c(Predicted_Risk_Premium, Observed_Risk_Premium)) %>%
		ggplot() +
		geom_line(aes(x = as.factor(buckets), y = value, col = name, group = name)) +
		geom_point(aes(x = as.factor(buckets), y = value, col = name, group = name)) +
		xlab("Bucket") + ylab("Average Risk Premium")
}

#Double Lift Curve
double_lift_chart <- function(predicted_loss_cost_mod_1, predicted_loss_cost_mod_2, observed_loss_cost, exposure, n) {
	pred_lc_m1 <- enquo(predicted_loss_cost_mod_1)
	pred_lc_m2 <- enquo(predicted_loss_cost_mod_2)
	obs_lc <- enquo(observed_loss_cost)
	exp <- enquo(exposure)
	dataset <- tibble(pred_lc_m1 = !! pred_lc_m1, pred_lc_m2 = !! pred_lc_m2, obs_lc = !! obs_lc, exp = !! exp)
	dataset <- dataset %>%
		mutate(sort_ratio = pred_lc_m1 / pred_lc_m2) %>%
		arrange(exp) %>% 
		mutate(buckets = ntile(exp, n)) %>% 
		group_by(buckets) %>% 
		summarise(Model_1_Predicted_Risk_Premium = mean(pred_lc_m1),
							Model_2_Predicted_Risk_Premium = mean(pred_lc_m2),
							Observed_Risk_Premium = mean(obs_lc), 
							Exposure = sum(exp))
	max_bucket <- which.max(dataset$Exposure)
	base_predicted_rp1 <- dataset[max_bucket, ]$Model_1_Predicted_Risk_Premium
	base_predicted_rp2 <- dataset[max_bucket, ]$Model_2_Predicted_Risk_Premium
	base_observed_rp <- dataset[max_bucket, ]$Observed_Risk_Premium 
	dataset <- dataset %>%
		select(-Exposure) %>%
		tidyr::pivot_longer(c(Model_1_Predicted_Risk_Premium, Model_2_Predicted_Risk_Premium, Observed_Risk_Premium)) %>%
		ggplot() +
		geom_line(aes(x = as.factor(buckets), y = value, col = name, group = name)) +
		geom_point(aes(x = as.factor(buckets), y = value, col = name, group = name)) +
		xlab("Bucket") + ylab("Average Risk Premium")
}

#Gini Plot
gini_plot <- function(predicted_loss_cost, exposure) {
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
		xlab("Exposure") +
		ylab("Predicted Loss Cost")
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
gini_plot_fs <- gini_plot(as.numeric(fs_pred),test$exposure)
gini_plot_fs + ggtitle("Frequency x Severity MARS Gini Plot")
gini_value(as.numeric(fs_pred),test$exposure)
lift_curve_fs <- lift_curve_plot(as.numeric(fs_pred),test$ClaimAmount,test$exposure,10)
lift_curve_fs +  labs(col = "Model") + ggtitle("Frequency x Severity MARS Lift Curve")

train$ClaimAmount <- NULL
train$Freq <- NULL
train$ClaimInd <- NULL
train$exposure <- NULL
pp_mod <- earth(PurePrem ~ ., data=train, degree=3)

pp_pred <- predict(pp_mod, newdata=test)*test$exposure
RMSE(pp_pred,test$ClaimAmount) #2025.51
NRMSE(pp_pred,test$ClaimAmount) #0.01307142
gini_plot_pp <- gini_plot(as.numeric(pp_pred),test$exposure)
gini_plot_pp + ggtitle("Pure Premium MARS Gini Plot")
gini_value(as.numeric(pp_pred),test$exposure)
lift_curve_pp <- lift_curve_plot(as.numeric(pp_pred),test$ClaimAmount,test$exposure,10)
lift_curve_pp + labs(col="Model") + ggtitle("Pure Premium MARS Lift Curve")
