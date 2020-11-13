#First, install packages if not present (add any other missing packages)
for(pkg in c("Metrics","dplyr","tidyr","gam","mgcv","mda","ggplot2","magrittr","pracma")){
  if(!pkg %in% installed.packages()){
    install.packages(pkg)
  }
}

#Libraries for evaluation functions
library(Metrics)
library(dplyr)
library(tidyr)
library(ggplot2)
library(magrittr)
library(pracma)

#Libraries for models
library(gam)
library(mgcv)

#Data

fre.train<-read.csv(file.choose(),stringsAsFactors = FALSE) #final/train.csv from github.com/MLWP3/Pricing
fre.test<-read.csv(file.choose(),stringsAsFactors = FALSE) #final/test.csv from github.com/MLWP3/Pricing
fre.train$DrivAge<-as.numeric(as.character(fre.train$DrivAge)) #Just to be sure
fre.test$DrivAge<-as.numeric(as.character(fre.test$DrivAge))

#Evaluation functions

#Normalized RMSE
NRMSE <- function(pred, obs) {
  rmse(pred, obs)/(max(obs)-min(obs))
}

#For aggregated risk premium diff
agg_rpr <- function(data, targetvar, prediction.obj) {
  tot_obs <- sum(as.numeric(data[[targetvar]]))
  tot_pred <- sum(as.numeric(prediction.obj))
  rpr <- tot_obs / tot_pred
  return(rpr)
}

#Gini
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

#Gini plot
gini_plot <- function(predicted_loss_cost, exposure){
  
  lc_var <- enquo(predicted_loss_cost)
  exp <- enquo(exposure)
  
  dataset <- tibble(lc_var = !! lc_var, exp = !! exp)
  
  dataset %>% 
    arrange(lc_var) %>% 
    mutate(pred_losses = lc_var * exp,
           cum_exp = cumsum(exp) / sum(exp),
           cum_pred_lc = cumsum(pred_losses) / sum(pred_losses)) %>% 
    ggplot() +
    geom_line(aes(x = cum_exp, y = cum_pred_lc)) +
    geom_abline(intercept = 0, slope = 1) +
    xlab("Percentage of Exposure") +
    ylab("Percentage of Losses")
}


#Models

predictors<-c("bs(DrivAge)","bs(BonusMalus)","VehPower","Area","VehGas","VehBrand","bs(Density)","Region")

#Pure premium

#MGCV - GAM
fre.GAM.2<-mgcv::gam(as.formula(paste("ClaimAmount~",paste(predictors,collapse="+"),sep="")),data=fre.train)
pred.fre.GAM.2<-predict(fre.GAM.2,newdata=fre.test)

#F/S

#MGCV - GAM
fre.GAM.2.freq<-mgcv::gam(as.formula(paste("ClaimNb~",paste(predictors,collapse="+"),sep="")),data=fre.train)
fre.GAM.2.sev<-mgcv::gam(as.formula(paste("severity~",paste(predictors,collapse="+"),sep="")),data=fre.train, subset = ClaimNb > 0, weights = ClaimNb)
pred.fre.GAM.2.freq<-predict(fre.GAM.2.freq,newdata=fre.test)
pred.fre.GAM.2.sev<-predict(fre.GAM.2.sev,newdata=fre.test)

#Results

#Pure premium

#MGCV - GAM
rmse(pred.fre.GAM.2,fre.test$ClaimAmount)                               #RMSE
mae(predicted = pred.fre.GAM.2, actual = fre.test$ClaimAmount)          #MAE
cor(pred.fre.GAM.2,fre.test$ClaimAmount)                                #Pearson correlation
cor(pred.fre.GAM.2,fre.test$ClaimAmount,method="spearman")              #Spearman correlation
gini_value(pred.fre.GAM.2, fre.test$Exposure)  #Gini index
agg_rpr(fre.test, "ClaimAmount", pred.fre.GAM.2)                        #Aggregated risk prem diff
#NRMSE(pred=pred.fre.GAM.2,obs=fre.test$ClaimAmount)
gini_plot(pred.fre.GAM.2,fre.test$Exposure)
gini_value(pred.fre.GAM.2,fre.test$Exposure)
lift_curve_plot(pred.fre.GAM.2,fre.test$ClaimAmount,fre.test$Exposure,25)

#F/S

#MGCV - GAM
rmse(pred.fre.GAM.2.freq*pred.fre.GAM.2.sev,fre.test$ClaimAmount)                               #RMSE
mae(predicted = pred.fre.GAM.2.freq*pred.fre.GAM.2.sev, actual = fre.test$ClaimAmount)          #MAE
cor(pred.fre.GAM.2.freq*pred.fre.GAM.2.sev,fre.test$ClaimAmount)                                #Pearson correlation
cor(pred.fre.GAM.2.freq*pred.fre.GAM.2.sev,fre.test$ClaimAmount,method="spearman")              #Spearman correlation
gini_value(pred.fre.GAM.2.freq*pred.fre.GAM.2.sev, fre.test$Exposure)  #Gini index
agg_rpr(fre.test, "ClaimAmount", pred.fre.GAM.2.freq*pred.fre.GAM.2.sev)                        #Aggregated risk prem diff
#NRMSE(pred=pred.fre.GAM.2,obs=fre.test$ClaimAmount)
gini_plot(pred.fre.GAM.2.freq*pred.fre.GAM.2.sev,fre.test$Exposure)
gini_value(pred.fre.GAM.2.freq*pred.fre.GAM.2.sev,fre.test$Exposure)
lift_curve_plot(pred.fre.GAM.2.freq*pred.fre.GAM.2.sev,fre.test$ClaimAmount,fre.test$Exposure,25)
