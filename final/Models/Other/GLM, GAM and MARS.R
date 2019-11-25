#Install in case missing
for(pack in c("caret","statmod","tweedie","mgcv","gam","earth","pracma")){
  if(!(pack %in% installed.packages())){
    install.packages(pack)
  }
}


#Marco's functions for data import and model evaluation:
library(Matrix)
library(caret)
library(tidyverse)
library(readr)
library(dplyr)
library(magrittr)
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

#DATA IMPORT
fre.train<-as.data.frame(import_data("https://raw.githubusercontent.com/mlwp3/pricing/master/final/Data/train_final.csv"))
fre.test<-as.data.frame(import_data("https://raw.githubusercontent.com/mlwp3/pricing/master/final/Data/test_final.csv"))
freMPL<-rbind(fre.train,fre.test)

### MODEL TESTING

#loss cost

#Target variable is ClaimAmount, offset by exposure
targetvar<-"ClaimAmount"
predictors.GLM<-c("LicAge_Band", "VehAge", "Gender", "MariStat", 
                  "VehUsage", "DrivAge_Band", "HasKmLimit", 
                  "BonusMalus_Cat", "VehBody","VehPriceGrp", 
                  "VehEngine", "VehEnergy", "VehMaxSpeed", 
                  "VehClass", "Garage")
#splining some in advance
predictors.GAM<-c("s(LicAge)", "VehAge", "Gender", "MariStat", 
                  "VehUsage", "s(DrivAge)", "HasKmLimit", 
                  "s(BonusMalus)", "VehBody","VehPriceGrp", 
                  "VehEngine", "VehEnergy", "VehMaxSpeed",
                  "VehClass", "Garage")
predictors.MARS<-c("LicAge", "VehAge", "Gender", "MariStat", 
                  "VehUsage", "DrivAge", "HasKmLimit", 
                  "BonusMalus", "VehBody","VehPriceGrp", 
                  "VehEngine", "VehEnergy", "VehMaxSpeed",
                  "VehClass", "Garage")
formula.GLM<-as.formula(paste(targetvar,"~",paste(predictors.GLM,collapse = "+",sep=""),sep=""))
formula.GAM<-as.formula(paste(targetvar,"~",paste(predictors.GAM,collapse = "+",sep=""),sep=""))
formula.MARS<-as.formula(paste(targetvar,"~",paste(predictors.MARS,collapse = "+",sep=""),sep=""))

##GLM
library(statmod)
library(tweedie)
p_values <- seq(1, 2, by = 0.1)
p_tuning <- tweedie.profile(formula = formula.GLM, data = fre.train, p.vec = p_values, do.plot = FALSE, offset = log(exposure))
p <- p_tuning$p.max
fre.GLM <- glm(formula = formula.GLM, data = fre.train, family = tweedie(var.power = p, link.power = 0), offset = log(exposure))
pred.GLM<-predict(fre.GLM, fre.test, type = "response")

#Let's apply AIC here
AICtweedie(fre.GLM)
#Yet to figure out how to automate this. MuMIn could be of use?

#GAM
#Testing out both MGCV and GAM's implementation - GAM seems a lot better interestingly
library(mgcv)
library(gam)
fre.GAM<-mgcv::gam(formula = formula.GAM,data=fre.train, family = Tweedie(p = p, link = power(0)), offset = log(exposure))
fre.GAM2<-gam::gam(formula = formula.GAM,data=fre.train, family = tweedie(var.power = p, link.power=0), offset = log(exposure))
pred.GAM<-predict(fre.GAM,fre.test,type = "response")
pred.GAM2<-predict(fre.GAM2,fre.test,type = "response")

#MARS - was not far behind last time
library(earth)
#Using the GAM formula here too
fre.MARS<-earth(formula=formula.GLM,data=fre.train)
pred.MARS<-predict(fre.MARS,fre.test,type="response")

### PERFORMANCE EVALUATION
library(pracma)

#loss cost

#GLM
NRMSE(pred.GLM,fre.test$ClaimAmount)
gini_plot(pred.GLM,fre.test$exposure)
gini_value(pred.GLM,fre.test$exposure)

#GAM (MGCV)
NRMSE(pred.GAM,fre.test$ClaimAmount)
gini_plot(pred.GAM,fre.test$exposure)
gini_value(pred.GAM,fre.test$exposure)

#GAM (GAM)
NRMSE(pred.GAM2,fre.test$ClaimAmount)
gini_plot(pred.GAM2,fre.test$exposure)
gini_value(pred.GAM2,fre.test$exposure)

#MARS
NRMSE(pred.MARS,fre.test$ClaimAmount)
gini_plot(pred.MARS,fre.test$exposure)
gini_value(as.numeric(pred.MARS),fre.test$exposure)
