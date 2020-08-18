library(tidyverse)
library(xgboost)
library(Matrix)
library(magrittr)
library(caret)
library(pracma)
library(tweedie)

# Import Data Functions ----------------------------------------------------

import_data <- function(data){
  read_csv(data, col_types = cols(VehPower = col_character(),
                                  Freq = col_skip(),
                                  BonusMalus = col_skip())) %>% 
    mutate_if(is.character, as.factor) %>% 
    filter((ClaimNb==0 & ClaimAmount==0) | (ClaimNb>0 & ClaimAmount>0))
}

group_data <- function(data){
  data %>% 
    group_by(Area,
             VehPower,
             VehBrand,
             VehGas,
             Region,
             DrivAgeBand,
             DensityBand,
             VehAgeBand) %>% 
    summarize(ClaimNb = sum(ClaimNb),
              ClaimAmount = sum(ClaimAmount),
              Exposure = sum(Exposure)) %>% 
    mutate(Severity = ifelse(ClaimNb == 0, 0, ClaimAmount / ClaimNb),
           Loss_Cost = ClaimAmount / Exposure) %>% 
    ungroup()
}

create_data_numb <- function(data){
  
  sparse.model.matrix(ClaimNb ~
                        Area +
                        VehPower +
                        VehBrand +
                        VehGas +
                        Region +
                        DrivAgeBand +
                        DensityBand +
                        VehAgeBand +
                        0,
                      data = data)}

create_data_sev <- function(data){
  
  sparse.model.matrix(Severity ~
                        Area +
                        VehPower +
                        VehBrand +
                        VehGas +
                        Region +
                        DrivAgeBand +
                        DensityBand +
                        VehAgeBand +
                        0, 
                      data = data)}

create_data_lc <- function(data){
  
  sparse.model.matrix(Loss_Cost ~
                        Area +
                        VehPower +
                        VehBrand +
                        VehGas +
                        Region +
                        DrivAgeBand +
                        DensityBand +
                        VehAgeBand +
                        0, 
                      data = data)}

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

text_project <- element_text(size = 18)

theme_project <- theme(
  axis.title = text_project
  , legend.text = text_project
  , axis.text = text_project
  , legend.position = 'bottom'
)

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
    pred_lc = predicted_loss_cost,
    obs_lc = observed_loss_cost,
    exp = exposure
  )

  dataset <- dataset %>%
    mutate(buckets = ntile(exp, n)) %>%
    group_by(buckets) %>%
    summarise(
      Predicted_Risk_Premium = mean(pred_lc, na.rm = TRUE),
      Observed_Risk_Premium = mean(obs_lc, na.rm = TRUE),
      Exposure = sum(exp)
    )

  max_bucket <- which.max(dataset$Exposure)
  dataset <- dataset %>%
    mutate(
      base_predicted_rp = dataset[max_bucket, ]$Predicted_Risk_Premium,
      base_observed_rp = dataset[max_bucket, ]$Observed_Risk_Premium,
    )

  dataset %>%
    mutate(
      Predicted_Risk_Premium = Predicted_Risk_Premium / base_predicted_rp,
      Observed_Risk_Premium = Observed_Risk_Premium / base_observed_rp,
      buckets = as.factor(buckets)
    ) %>%
    arrange(Predicted_Risk_Premium) %>%
    mutate(
      buckets = seq_len(nrow(.))
    )
}

lift_curve_plot <- function(tbl_in) {

  tbl_in %>%
    tidyr::pivot_longer(c(Predicted_Risk_Premium, Observed_Risk_Premium)) %>%
    mutate(buckets = as.factor(buckets)) %>%
    ggplot() +
    geom_bar(aes(x = buckets, y = Exposure * ( max( c(tbl_in$Predicted_Risk_Premium, tbl_in$Observed_Risk_Premium) ) / 
                                                 max(tbl_in$Exposure) )),
             stat="identity", alpha = 0.5, fill="grey")+
    geom_point(aes(x = buckets, y = value, col = name, group = name), size = 3.5) +
    geom_line(aes(x = buckets, y = value, col = name, group = name), size = 4) +
    labs(x = "Bucket", y = "Average Risk Premium") +
    scale_y_continuous(sec.axis = sec_axis(~./( max( c(tbl_in$Predicted_Risk_Premium, tbl_in$Observed_Risk_Premium) ) / 
                                                  max(tbl_in$Exposure) )*1000), 
                                           name = "Exposure (k)") +
    theme_project + 
    labs(color = "")
}

double_lift_chart <- function(predicted_loss_cost_mod_1, predicted_loss_cost_mod_2, observed_loss_cost, exposure, n,
                              name_1, name_2) {
  
  pred_lc_m1 <- enquo(predicted_loss_cost_mod_1)
  pred_lc_m2 <- enquo(predicted_loss_cost_mod_2)
  obs_lc <- enquo(observed_loss_cost)
  exp <- enquo(exposure)
  
  dataset <- tibble(
    pred_lc_m1 = !! pred_lc_m1,
    pred_lc_m2 = !! pred_lc_m2,
    obs_lc = !! obs_lc,
    exp = !! exp
  )
  
  dataset <- dataset %>%
    mutate(sort_ratio = pred_lc_m1 / pred_lc_m2) %>%
    arrange(exp) %>% 
    mutate(buckets = ntile(exp, n)) %>% 
    group_by(buckets) %>% 
    summarise(
      Model_1_Predicted_Risk_Premium = mean(pred_lc_m1),
      Model_2_Predicted_Risk_Premium = mean(pred_lc_m2),
      Observed_Risk_Premium = mean(obs_lc),
      Exposure = sum(exp)
    )
  
  max_bucket <- which.max(dataset$Exposure)
  base_predicted_rp1 <- dataset[max_bucket, ]$Model_1_Predicted_Risk_Premium
  base_predicted_rp2 <- dataset[max_bucket, ]$Model_2_Predicted_Risk_Premium
  base_observed_rp <- dataset[max_bucket, ]$Observed_Risk_Premium 
  
  dataset <- dataset %>%
    mutate(
      Model_1_Predicted_Risk_Premium = Model_1_Predicted_Risk_Premium / base_predicted_rp1,
      Model_2_Predicted_Risk_Premium = Model_2_Predicted_Risk_Premium / base_predicted_rp2,
      Observed_Risk_Premium = Observed_Risk_Premium / base_observed_rp,
      buckets = as.factor(buckets)) 
  
  dataset %>%
    ggplot() +
    geom_bar(aes(x = buckets, y = Exposure * ( max(c(Observed_Risk_Premium, 
                                                     Model_1_Predicted_Risk_Premium, 
                                                     Model_2_Predicted_Risk_Premium)) / 
                                               max(Exposure))),
             stat="identity", alpha = 0.5, fill="grey") +
    geom_line(aes(x = buckets, y = Model_1_Predicted_Risk_Premium, group = 1, col = name_1), size = 4) +
    geom_line(aes(x = buckets, y = Model_2_Predicted_Risk_Premium, group = 1, col = name_2), size = 4) +
    geom_line(aes(x = buckets, y = Observed_Risk_Premium, group = 1, col = "Observed_Risk_Premium"), size = 4) +

    geom_point(aes(x = buckets, y = Model_1_Predicted_Risk_Premium, col = name_1), size = 3.5) +
    geom_point(aes(x = buckets, y = Model_2_Predicted_Risk_Premium, col = name_2), size = 3.5) +
    geom_point(aes(x = buckets, y = Observed_Risk_Premium, col = "Observed_Risk_Premium"), size = 3.5) +

    scale_y_continuous(sec.axis = sec_axis(~./(max(c(dataset$Observed_Risk_Premium, 
                                                     dataset$Model_1_Predicted_Risk_Premium, 
                                                     dataset$Model_2_Predicted_Risk_Premium)) / 
                                                 max(dataset$Exposure)*1000),
                                           name = "Exposure (k)")) +

    labs(x = "Bucket", y = "Average Risk Premium") +
    theme_project + 
    labs(color = "")
}
