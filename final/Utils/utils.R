library(tidyverse)
library(xgboost)
library(Matrix)
library(magrittr)
library(caret)
library(pracma)
library(tweedie)
library(tweedie)
library(statmod)

# Import Data Functions ----------------------------------------------------

import_data <- function(data){
  read_csv(data, col_types = cols(VehPower = col_character(),
                                  BonusMalus = col_skip(),
                                  Density = col_skip(),
                                  severity = col_skip())) %>% 
    mutate_if(is.character, as.factor) %>% 
    filter((ClaimNb==0 & ClaimAmount==0) | (ClaimNb>0 & ClaimAmount>0))
}

group_data <- function(data){
  data %>% 
    # group_by(Area,
    #          VehPower,
    #          VehBrand,
    #          VehGas,
    #          Region,
    #          DrivAgeBand,
    #          DensityBand,
    #          VehAgeBand) %>% 
    # summarize(ClaimNb = sum(ClaimNb),
    #           ClaimAmount = sum(ClaimAmount),
    #           Exposure = sum(Exposure)) %>% 
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

create_data_losses <- function(data){
  
  sparse.model.matrix(ClaimAmount ~
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

gini_value <- function(observed_loss_cost, predicted_loss_cost, exposure){
  
  obs_lc <- enquo(observed_loss_cost)
  pred_lc <- enquo(predicted_loss_cost)
  exp <- enquo(exposure)
  
  dataset <- tibble(obs_lc = !!obs_lc, pred_lc = !!pred_lc, exp = !!exp)
  
  dataset %>% 
    arrange(pred_lc) %>% 
    mutate(losses = obs_lc * exp,
           cum_exp = cumsum(exp) / sum(exp),
           cum_losses = cumsum(losses) / sum(losses)) %$%
  {trapz(cum_exp, cum_losses) %>% add(-1) %>% abs() %>% subtract(.5) %>% multiply_by(2)}
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

gini_plot <- function(observed_loss_cost, predicted_loss_cost, exposure){
  
  obs_lc <- enquo(observed_loss_cost)
  pred_lc <- enquo(predicted_loss_cost)
  exp <- enquo(exposure)
  
  dataset <- tibble(obs_lc = !!obs_lc, pred_lc = !!pred_lc, exp = !!exp)
  
  dataset %>% 
    arrange(pred_lc) %>% 
    mutate(losses = obs_lc * exp,
           cum_exp = cumsum(exp) / sum(exp),
           cum_losses = cumsum(losses) / sum(losses)) %>% 
    ggplot() +
    geom_line(aes(x = cum_exp, y = cum_losses)) +
    geom_abline(intercept = 0, slope = 1) +
    xlab("Percentage of Exposure") +
    ylab("Percentage of Losses") + 
    theme_project
}

lift_curve_table <- function(predicted_loss_cost, observed_loss_cost, exposure, n) {

  dataset <- tibble(
    pred_lc = predicted_loss_cost,
    obs_lc = observed_loss_cost,
    exp = exposure
  )

  dataset <- dataset %>%
    mutate(buckets = cut_interval(cumsum(exp), n = n, labels = 1:n)) %>%
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
  
  scale_factor <-  max(c(tbl_in$Predicted_Risk_Premium, tbl_in$Observed_Risk_Premium)) / max(tbl_in$Exposure)
  
  tbl_in %>%
    tidyr::pivot_longer(c(Predicted_Risk_Premium, Observed_Risk_Premium)) %>%
    mutate(buckets = as.factor(buckets)) %>%
    ggplot() +
    geom_bar(aes(x = buckets, y = .5 * Exposure * scale_factor),
             stat="identity", alpha = 0.5, fill="grey")+
    geom_point(aes(x = buckets, y = value, col = name, group = name), size = 3.5) +
    geom_line(aes(x = buckets, y = value, col = name, group = name), size = 4) +
    labs(x = "Bucket", y = "Average Risk Premium") +
    scale_y_continuous(sec.axis = sec_axis(~./ (scale_factor * 1000),
                                           name = "Exposure (k)")) +
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
    mutate(buckets = cut_interval(cumsum(exp), n = n, labels = 1:n)) %>% 
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

plot_act_metrics <- function(data, var, n = 10, categorical = FALSE){
  
  if (!categorical) {
    table <- data %>%
      mutate(feature = ntile({{var}}, n)) %>%
      group_by(feature) %>%
      summarise(prem = sum(premium),
                exp = sum(exposure),
                losses = sum(losses),
                count = sum(numbers)) %>%
      mutate(loss_cost = losses / exp,
             loss_ratio = losses / prem,
             frequency = count / exp,
             severity = losses / count) %>%
      na.omit()  
  
    } else {
    
    table <- data %>%
      mutate(feature = as_factor({{var}})) %>%
      group_by(feature) %>%
      summarise(prem = sum(premium),
                exp = sum(exposure),
                losses = sum(losses),
                count = sum(numbers)) %>%
      mutate(loss_cost = losses / exp,
             loss_ratio = losses / prem,
             frequency = count / exp,
             severity = losses / count) %>%
      na.omit()
  }
  
  plot_lr <- table %>%
    ggplot()+
    geom_bar(aes(x = as.factor(feature), y = prem * (max(table$loss_ratio) / max(table$prem))), alpha = .5,
             stat="identity", fill="slategrey")+
    geom_line(aes(x = as.factor(feature), y = loss_ratio, group = 1), col = scales::hue_pal()(4)[1])+
    xlab(enquo(var)) +
    scale_y_continuous(sec.axis = sec_axis(~./((max(table$loss_ratio) / max(table$prem))*1000000), name = "Premium (m)"))+
    labs(y = "Loss Ratio")+
    ggtitle("Loss Ratio")
  
  plot_lc <- table %>%
    ggplot()+
    geom_bar(aes(x = as.factor(feature), y = exp * (max(table$loss_cost) / max(table$exp))), alpha = .5,
             stat="identity", fill="slateblue2")+
    geom_line(aes(x = as.factor(feature), y = loss_cost, group = 1), col = scales::hue_pal()(4)[2])+
    xlab(enquo(var))+
    scale_y_continuous(sec.axis = sec_axis(~./((max(table$loss_cost) / max(table$exp))*1000), name = "Exposure (k)"))+
    labs(y = "Loss Cost")+
    ggtitle("Loss Cost")
  
  plot_freq <- table %>%
    
    ggplot()+
    geom_bar(aes(x = as.factor(feature), y = exp * (max(table$frequency) / max(table$exp))), alpha = .5,
             stat="identity", fill="slateblue2")+
    geom_line(aes(x = as.factor(feature), y = frequency, group = 1), col = scales::hue_pal()(4)[3])+
    xlab(enquo(var))+
    scale_y_continuous(sec.axis = sec_axis(~./((max(table$frequency) / max(table$exp))*1000), name = "Exposure (k)"))+
    labs(y = "Frequency")+
    ggtitle("Frequency")

  plot_sev <- table %>%
    ggplot()+
    geom_bar(aes(x = as.factor(feature), y = count * (max(table$severity) / max(table$count))), alpha = .5,
             stat="identity", fill="peachpuff3")+
    geom_line(aes(x = as.factor(feature), y = severity, group = 1), col = scales::hue_pal()(4)[4])+
    xlab(enquo(var))+
    scale_y_continuous(sec.axis = sec_axis(~./((max(table$severity) / max(table$count))*1000), name = "Numbers (k)"))+
    labs(y = "Severity")+
    ggtitle("Severity")

  g <- grid.arrange(plot_lr, plot_lc, plot_freq, plot_sev,
                    ncol = 2, nrow = 2,
                    top = textGrob(rlang::as_name(enquo(var)), gp = gpar(fontsize=15)))
}

eval_act_metrics <- function(data, var, predicted_freq_sev, predicted_loss_cost, n = 10, categorical = FALSE){
  
  data <- data %>%
    mutate(losses_freq_sev = {{predicted_freq_sev}} * exposure,
           losses_loss_cost = {{predicted_loss_cost}} * exposure)
  
  if (!categorical) {

  table <- data %>%
      mutate(feature = ntile({{var}}, n)) %>%
      group_by(feature) %>%
      summarise(exp = sum(exposure),
                obs_losses = sum(losses),
                pred_lc_freq_sev = sum(losses_freq_sev),
                pred_lc_lc = sum(losses_loss_cost)) %>%
      mutate(obs_lc = obs_losses / exp,
             pred_freq_sev = pred_lc_freq_sev / exp,
             pred_lc = pred_lc_lc / exp) %>%
      na.omit()
    
  } else {
    
    table <- data %>%
      mutate(feature = as_factor({{var}})) %>%
      group_by(feature) %>%
      summarise(exp = sum(exposure),
                obs_losses = sum(losses),
                pred_lc_freq_sev = sum(losses_freq_sev),
                pred_lc_lc = sum(losses_loss_cost)) %>%
      mutate(obs_lc = obs_losses / exp,
             pred_freq_sev = pred_lc_freq_sev / exp,
             pred_lc = pred_lc_lc / exp) %>%
      na.omit()
    
  }
  
  scale_factor <- max(c(table$pred_lc, table$pred_freq_sev, table$obs_lc)) / max(table$exp)
  
  table %>%
    select(feature, exp, obs_lc, pred_freq_sev, pred_lc) %>%
    tidyr::pivot_longer(c(obs_lc, pred_freq_sev, pred_lc)) %>%
    ggplot()+
    geom_bar(aes(x = as.factor(feature), y = 1/3 * exp * scale_factor), alpha = .5,
             stat="identity", fill="grey")+
    geom_line(aes(x = as.factor(feature), y = value, group = name, col = name))+
    geom_point(aes(x = as.factor(feature), y = value, col = name))+
    xlab(enquo(var)) +
    scale_y_continuous(sec.axis = sec_axis(~./(scale_factor * 1000), name = "Exposure (k)"))+
    labs(y = "Loss Cost")+
    ggtitle(paste0("Loss Cost Comparison ", rlang::as_name(enquo(var))))+
    labs(color = "")
}

strip_glm = function(model){
  
  model$data <- NULL
  model$y <- NULL
  model$linear.predictors <- NULL
  model$weights <- NULL
  model$fitted.values <- NULL
  model$model <- NULL
  model$prior.weights <- NULL
  model$residuals <- NULL
  model$effects <- NULL
  model$qr$qr <- NULL
  
  return(model)
}