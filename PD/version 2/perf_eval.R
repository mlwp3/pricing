library(tidyverse)
library(xgboost)
library(Matrix)
library(magrittr)
library(caret)
library(pracma)
library(tweedie)
library(tweedie)
library(statmod)

rmse <- function(data, targetvar, prediction.obj) {
  rss <- (as.numeric(prediction.obj) - as.numeric(unlist(data[, c(targetvar)]))) ^ 2
  mse <- sum(rss) / nrow(data)
  rmse <- sqrt(mse)
  rmse <- rmse / 100
  return(rmse)
} #Use utils

NRMSE <- function(data, targetvar, prediction.obj) {
  obs <- as.numeric(data[[targetvar]])
  pred <- as.numeric(prediction.obj)
  r <- rmse(data, targetvar, prediction.obj)
  nrmse <- r / (max(obs) - min(obs))
  return(nrmse)
} #Use utils

agg_rpr <- function(data, targetvar, prediction.obj) {
  tot_obs <- sum(as.numeric(data[[targetvar]]))
  tot_pred <- sum(as.numeric(prediction.obj))
  rpr <- tot_obs / tot_pred
  return(rpr)
}

norm_rp_deviance <- function(data, targetvar, prediction.obj) {
  obs <- as.numeric(data[[targetvar]])
  pred <- as.numeric(prediction.obj)
  mean_obs <- mean(obs)
  mean_pred <- mean(pred)
  rpd <- mean_obs / mean_pred
  rpd <- 1 - abs(1 - rpd)
  return(rpd)
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

NMAE <- function(pred, obs) {
  MAE(pred, obs) / (max(obs) - min(obs))
}

harmonic_mean <- function(x) {
  1 / mean(1 / x)
}

geometric_mean <- function(x) {
  prod(x) ^ (1 / length(x))
}