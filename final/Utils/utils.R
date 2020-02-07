
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

gini_value <- function(predicted_loss_cost, exposure){
  
  lc_var <- enquo(predicted_loss_cost)
  exp <- enquo(exposure)
  
  dataset <- tibble(lc_var = !!lc_var, exp = !!exp)
  
  dataset %>% 
    arrange(lc_var) %>% 
    mutate(cum_exp = cumsum(exp) / sum(exp),
           cum_pred_lc = cumsum(lc_var) / sum(lc_var)) %$% 
    {trapz(cum_exp, cum_pred_lc) %>% add(-1) %>% abs() %>% subtract(.5) %>% multiply_by(2)}
}

#-------------------
# Plot functions

theme_project <- theme(
  axis.title = element_text(size = 18)
  , legend.position = 'bottom'
  , legend.text = element_text(size = 18)
)

geom_project <- function() {
  
  update_geom_defaults(
    'line', list(size = 4))
  
  update_geom_defaults(
    'point', list(size = 4)
  )
  
}


gini_plot <- function(predicted_loss_cost, exposure){
  
  lc_var <- enquo(predicted_loss_cost)
  exp <- enquo(exposure)
  
  dataset <- tibble(lc_var = !! lc_var, exp = !! exp)
  
  dataset %>% 
    arrange(lc_var) %>% 
    mutate(cum_exp = cumsum(exp) / sum(exp),
           cum_pred_lc = cumsum(lc_var) / sum(lc_var)) %>% 
    ggplot() +
    geom_line(aes(x = cum_exp, y = cum_pred_lc)) +
    geom_abline(intercept = 0, slope = 1) +
    xlab("Exposure") +
    ylab("Predicted Loss Cost") + 
    theme_project
  
}

lift_curve_table <- function(predicted_loss_cost, observed_loss_cost, exposure, n) {
  
  dataset <- tibble(
    pred_lc = predicted_loss_cost
    , obs_lc = observed_loss_cost
    , exp = exposure
  )
  
  dataset <- dataset %>% 
    mutate(buckets = ntile(exp, n)) %>% 
    group_by(buckets) %>% 
    summarise(
      Predicted_Risk_Premium = mean(pred_lc, na.rm = TRUE)
      , Observed_Risk_Premium = mean(obs_lc, na.rm = TRUE)
      , Exposure = sum(exp)
    )
  
  max_bucket <- which.max(dataset$Exposure)
  dataset <- dataset %>% 
    mutate(
      base_predicted_rp = dataset[max_bucket, ]$Predicted_Risk_Premium
      , base_observed_rp = dataset[max_bucket, ]$Observed_Risk_Premium
    )
  
  dataset %>%
    mutate(
      Predicted_Risk_Premium = Predicted_Risk_Premium / base_predicted_rp
      , Observed_Risk_Premium = Observed_Risk_Premium / base_observed_rp
      , buckets = as.factor(buckets)
    ) %>% 
    arrange(Predicted_Risk_Premium) %>% 
    mutate(
      buckets = seq_len(nrow(.))
    ) %>% 
    select(-Exposure)
    
}

lift_curve_plot <- function(tbl_in) {
  
  tbl_in %>% 
    tidyr::pivot_longer(c(Predicted_Risk_Premium, Observed_Risk_Premium)) %>% 
    ggplot(aes(x = buckets, y = value, col = name, group = name)) +
    geom_line(size = 4) +
    geom_point(size = 4) +
    labs(x = "Bucket", y = "Average Risk Premium") + 
    theme_project
}

double_lift_chart <- function(predicted_loss_cost_mod_1, predicted_loss_cost_mod_2, observed_loss_cost, exposure, n) {
  
  pred_lc_m1 <- enquo(predicted_loss_cost_mod_1)
  pred_lc_m2 <- enquo(predicted_loss_cost_mod_2)
  obs_lc <- enquo(observed_loss_cost)
  exp <- enquo(exposure)
  
  dataset <- tibble(
    pred_lc_m1 = !! pred_lc_m1
    , pred_lc_m2 = !! pred_lc_m2
    , obs_lc = !! obs_lc
    , exp = !! exp
  )
  
  dataset <- dataset %>%
    mutate(sort_ratio = pred_lc_m1 / pred_lc_m2) %>%
    arrange(exp) %>% 
    mutate(buckets = ntile(exp, n)) %>% 
    group_by(buckets) %>% 
    summarise(
      Model_1_Predicted_Risk_Premium = mean(pred_lc_m1)
      , Model_2_Predicted_Risk_Premium = mean(pred_lc_m2)
      , Observed_Risk_Premium = mean(obs_lc)
      , Exposure = sum(exp)
    )
  
  max_bucket <- which.max(dataset$Exposure)
  base_predicted_rp1 <- dataset[max_bucket, ]$Model_1_Predicted_Risk_Premium
  base_predicted_rp2 <- dataset[max_bucket, ]$Model_2_Predicted_Risk_Premium
  base_observed_rp <- dataset[max_bucket, ]$Observed_Risk_Premium 
  
  dataset <- dataset %>%
    select(-Exposure) %>%
    mutate(
      Model_1_Predicted_Risk_Premium = Model_1_Predicted_Risk_Premium / base_predicted_rp1
      , Model_2_Predicted_Risk_Premium = Model_2_Predicted_Risk_Premium / base_predicted_rp2
      , Observed_Risk_Premium = Observed_Risk_Premium / base_observed_rp
      , buckets = as.factor(buckets)
    ) %>% 
    tidyr::pivot_longer(c(Model_1_Predicted_Risk_Premium, Model_2_Predicted_Risk_Premium, Observed_Risk_Premium)) %>%
    ggplot(aes(x = buckets, y = value, col = name, group = name)) +
    geom_line() +
    geom_point() +
      labs(x = "Bucket", y = "Average Risk Premium") + 
      theme_project
}