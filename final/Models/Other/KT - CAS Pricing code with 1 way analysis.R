#Install in case missing
for(pack in c("caret","statmod","tweedie","mgcv","earth","pracma")){
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

# create_data_numb <- function(data){
#   
#   sparse.model.matrix(ClaimInd ~
#                         LicAge +
#                         VehAge +
#                         Gender +
#                         MariStat +
#                         SocioCateg +
#                         VehUsage +
#                         DrivAge +
#                         HasKmLimit +
#                         BonusMalus +
#                         VehBody +
#                         VehPrice +
#                         VehEngine +
#                         VehEnergy      +
#                         VehMaxSpeed +
#                         VehClass +
#                         Garage +
#                         LicAge_Band +
#                         BonusMalus_Cat +
#                         DrivAge_Band +
#                         VehPriceGrp, data = data)}
# 
# create_data_sev <- function(data){
#   
#   sparse.model.matrix(sev ~
#                         LicAge +
#                         VehAge +
#                         Gender +
#                         MariStat +
#                         SocioCateg +
#                         VehUsage +
#                         DrivAge +
#                         HasKmLimit +
#                         BonusMalus +
#                         VehBody +
#                         VehPrice +
#                         VehEngine +
#                         VehEnergy      +
#                         VehMaxSpeed +
#                         VehClass +
#                         Garage +
#                         LicAge_Band +
#                         BonusMalus_Cat +
#                         DrivAge_Band +
#                         VehPriceGrp, data = data)}
# 
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
# predictors.MARS<-c("LicAge", "VehAge", "Gender", "MariStat", 
#                   "VehUsage", "DrivAge", "HasKmLimit", 
#                   "BonusMalus", "VehBody","VehPriceGrp", 
#                   "VehEngine", "VehEnergy", "VehMaxSpeed",
#                   "VehClass", "Garage")
formula.GLM<-as.formula(paste(targetvar,"~",paste(predictors.GLM,collapse = "+",sep=""),sep=""))
formula.GAM<-as.formula(paste(targetvar,"~",paste(predictors.GAM,collapse = "+",sep=""),sep=""))
formula.GAM.freq<-as.formula(paste("ClaimInd","~",paste(predictors.GAM,collapse = "+",sep=""),sep=""))
# formula.MARS<-as.formula(paste(targetvar,"~",paste(predictors.MARS,collapse = "+",sep=""),sep=""))

##GLM
library(statmod)
library(tweedie)
p_values <- seq(1, 2, by = 0.1)
p_tuning <- tweedie.profile(formula = formula.GLM, data = fre.train, p.vec = p_values, do.plot = FALSE, offset = log(exposure))
p <- p_tuning$p.max
fre.GLM <- glm(formula = formula.GLM, data = fre.train, family = tweedie(var.power = p, link.power = 0), offset = log(exposure))
pred.GLM<-predict(fre.GLM, fre.test, type = "response")

# #Let's apply AIC here
# AICtweedie(fre.GLM)
# #Yet to figure out how to automate this. MuMIn could be of use?

#GAM
#Testing out both MGCV and GAM's implementation - GAM seems a lot better interestingly
library(mgcv)
library(gam)
fre.GAM<-mgcv::gam(formula = formula.GAM,data=fre.train, family = Tweedie(p = p, link = power(0)), offset = log(exposure),optimizer=c("outer","optim"))
fre.GAM2<-gam::gam(formula = formula.GAM,data=fre.train, family = tweedie(var.power = p, link.power=0), offset = log(exposure))
pred.GAM<-predict(fre.GAM,fre.test,type = "response")
pred.GAM2<-predict(fre.GAM2,fre.test,type = "response")

#GAM FREQ SEV
fre.train.sev<-fre.train[fre.train$ClaimInd==1,]
fre.test.sev<-fre.test[fre.test$ClaimInd==1,]
fre.GAM.freq<-mgcv::gam(formula = formula.GAM.freq,data=fre.train, family = Tweedie(p = p, link = power(0)), offset = log(exposure),optimizer=c("outer","optim"))
pred.GAM.freq<-predict(fre.GAM.freq,fre.test,type = "response")
fre.GAM.sev<-mgcv::gam(formula = formula.GAM,data=fre.train, family = Tweedie(p = p, link = power(0)), offset = log(exposure),optimizer=c("outer","optim"))
pred.GAM.sev<-predict(fre.GAM.sev,fre.test,type = "response")

# #MARS - was not far behind last time
# library(earth)
# #Using the GAM formula here too
# fre.MARS<-earth(formula=formula.GLM,data=fre.train)
# pred.MARS<-predict(fre.MARS,fre.test,type="response")

### PERFORMANCE EVALUATION
library(pracma)

#loss cost

#GLM
NRMSE(pred.GLM,fre.test$ClaimAmount) #0.01378
gini_plot(pred.GLM,fre.test$exposure)
gini_value(pred.GLM,fre.test$exposure) #0.3273

#GAM (MGCV)
NRMSE(pred.GAM,fre.test$ClaimAmount)
gini_plot(pred.GAM,fre.test$exposure)
lift_curve_plot(pred.GAM,fre.test$exposure,20)+scale_y_log10()
gini_value(pred.GAM,fre.test$exposure)

#freq:
table(pred.GAM.freq,fre.test$ClaimInd)
gini_plot(pred.GAM.freq,fre.test$exposure)
lift_curve_plot(pred.GAM.freq,fre.test$exposure,20)+scale_y_log10()
gini_value(pred.GAM.freq,fre.test$exposure)

#sev:
NRMSE(pred.GAM.sev,fre.test.sev$ClaimAmount)
gini_plot(pred.GAM.sev,fre.test.sev$exposure)
lift_curve_plot(pred.GAM.sev,fre.test.sev$exposure,20)+scale_y_log10()
gini_value(pred.GAM.sev,fre.test.sev$exposure)

#Ignore this
#GAM (GAM) 
# NRMSE(pred.GAM2,fre.test$ClaimAmount)
# gini_plot(pred.GAM2,fre.test$exposure)
# gini_value(pred.GAM2,fre.test$exposure)

#One way analysis:

#One-Way Analysis against Burning Cost for Vehicle Body
grp_by_VehBody <- fre.test %>%
  group_by(VehBody) %>%
  summarize(total_rate_gam = sum(pred.GAM), exp = sum(exposure), total_clm = sum(ClaimAmount))
grp_by_VehBody$avg_rate_gam <- grp_by_VehBody$total_rate_gam / grp_by_VehBody$exp
grp_by_VehBody$avg_clm <- grp_by_VehBody$total_clm / grp_by_VehBody$exp

plot_VehBody <- plot_ly(data = grp_by_VehBody, x = ~VehBody, y = ~avg_clm, type = "scatter", mode = "lines", name = "BC") %>%
  add_trace(y = ~avg_rate_gam, mode = "lines", name = "GAM")
plot_VehBody

#One-Way Analysis against Burning Cost for Vehicle Age
grp_by_Gender <- fre.test %>%
  group_by(Gender) %>%
  summarize(total_rate_gam = sum(pred.GAM), exp = sum(exposure), total_clm = sum(ClaimAmount))
grp_by_Gender$avg_rate_gam <- grp_by_Gender$total_rate_gam / grp_by_Gender$exp
grp_by_Gender$avg_clm <- grp_by_Gender$total_clm / grp_by_Gender$exp

plot_Gender <- plot_ly(data = grp_by_Gender, x = ~Gender, y = ~avg_clm, type = "scatter", mode = "lines", name = "BC") %>%
  add_trace(y = ~avg_rate_gam, mode = "lines", name = "GAM")
plot_Gender

#One-Way Analysis against Burning Cost for Gender
grp_by_Garage <- fre.test %>%
  group_by(Garage) %>%
  summarize(total_rate_gam = sum(pred.GAM), exp = sum(exposure), total_clm = sum(ClaimAmount))
grp_by_Garage$avg_rate_gam <- grp_by_Garage$total_rate_gam / grp_by_Garage$exp
grp_by_Garage$avg_clm <- grp_by_Garage$total_clm / grp_by_Garage$exp

plot_Garage <- plot_ly(data = grp_by_Garage, x = ~Garage, y = ~avg_clm, type = "scatter", mode = "lines", name = "BC") %>%
  add_trace(y = ~avg_rate_gam, mode = "lines", name = "GAM")
plot_Garage

#One-Way Analysis against Burning Cost for License Age
grp_by_LicAge <- fre.test %>%
  group_by(LicAge_Band) %>%
  summarize(total_rate_gam = sum(pred.GAM), exp = sum(exposure), total_clm = sum(ClaimAmount))
grp_by_LicAge$avg_rate_gam <- grp_by_LicAge$total_rate_gam / grp_by_LicAge$exp
grp_by_LicAge$avg_clm <- grp_by_LicAge$total_clm / grp_by_LicAge$exp

plot_LicAge <- plot_ly(data = grp_by_LicAge, x = ~LicAge_Band, y = ~avg_clm, type = "scatter", mode = "lines", name = "BC") %>%
  add_trace(y = ~avg_rate_gam, mode = "lines", name = "GAM")
plot_LicAge

#One-Way Analysis against Burning Cost for Vehicle Age
grp_by_VehAge <- fre.test %>%
  group_by(VehAge) %>%
  summarize(total_rate_gam = sum(pred.GAM), exp = sum(exposure), total_clm = sum(ClaimAmount))
grp_by_VehAge$avg_rate_gam <- grp_by_VehAge$total_rate_gam / grp_by_VehAge$exp
grp_by_VehAge$avg_clm <- grp_by_VehAge$total_clm / grp_by_VehAge$exp

plot_VehAge <- plot_ly(data = grp_by_VehAge, x = ~VehAge, y = ~avg_clm, type = "scatter", mode = "lines", name = "BC") %>%
  add_trace(y = ~avg_rate_gam, mode = "lines", name = "GAM")
plot_VehAge

