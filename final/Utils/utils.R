
library(tidyverse)
library(xgboost)
library(Matrix)
library(magrittr)
library(caret)
library(pracma)

# Import Data Functions ----------------------------------------------------

import_data <- function(data){
  
  read_csv(data, 
           col_types = cols(Gender = col_character(), 
                            VehAge = col_character(), 
                            HasKmLimit = col_factor())) %>% 
    mutate_if(is.character, as.factor) %>% 
    filter((ClaimInd==0 & ClaimAmount==0) | (ClaimInd>0 & ClaimAmount>0)) %>% 
    mutate(sev = ifelse(ClaimInd == 0, 0, ClaimAmount / ClaimInd))
  
}

create_data_numb <- function(data){
  
  sparse.model.matrix(ClaimInd ~
                        LicAge +
                        VehAge +
                        Gender +
                        MariStat +
                        SocioCateg +
                        VehUsage +
                        DrivAge +
                        HasKmLimit +
                        BonusMalus +
                        VehBody +
                        VehPrice +
                        VehEngine +
                        VehEnergy      +
                        VehMaxSpeed +
                        VehClass +
                        Garage +
                        LicAge_Band +
                        BonusMalus_Cat +
                        DrivAge_Band +
                        VehPriceGrp, data = data)}

create_data_sev <- function(data){
  
  sparse.model.matrix(sev ~
                        LicAge +
                        VehAge +
                        Gender +
                        MariStat +
                        SocioCateg +
                        VehUsage +
                        DrivAge +
                        HasKmLimit +
                        BonusMalus +
                        VehBody +
                        VehPrice +
                        VehEngine +
                        VehEnergy      +
                        VehMaxSpeed +
                        VehClass +
                        Garage +
                        LicAge_Band +
                        BonusMalus_Cat +
                        DrivAge_Band +
                        VehPriceGrp, data = data)}

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

lift_curve_plot <- function(predicted_loss_cost, observed_loss_cost, n){
  
  pred_lc <- enquo(predicted_loss_cost)
  obs_lc <- enquo(observed_loss_cost)
  
  dataset <- tibble(pred_lc = !! pred_lc, obs_lc = !! obs_lc)

  dataset %>% 
    arrange(pred_lc) %>% 
    mutate(buckets = ntile(pred_lc, n)) %>% 
    group_by(buckets) %>% 
    summarise(avg_pred = mean(pred_lc),
              avg_obs = mean(obs_lc))%>% 
    pivot_longer(c(avg_pred,avg_obs)) %>%
    ggplot() +
    geom_line(aes(x = as.factor(buckets), y = value, col = name, group = name))+
    geom_point(aes(x = as.factor(buckets), y = value, col = name, group = name))+
    xlab("Bucket")
}
