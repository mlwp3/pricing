rm(list=ls())

set.seed(1)

library(tidyverse)
library(ggExtra)
library(gridExtra)
library(caret)
library(GGally)
library(forcats)
library(gam)
library(earth)
library(pracma)

# Functions ---------------------------------------------------------------

NRMSE <- function(pred, obs){
  
  RMSE(pred, obs)/(max(obs)-min(obs))

  }


gini<- function(x){
  
  abs(2*(trapz(pull(x[,1]), pull(x[,2])) - .5))
}

# Load Dataset ------------------------------------------------------------

freMPL <- read_csv("freMPL.csv", col_types = cols(Gender = col_character(), 
                                                  RecordBeg = col_skip(), 
                                                  RecordEnd = col_skip(), 
                                                  VehAge = col_character(), 
                                                  X1 = col_skip(),
                                                  HasKmLimit = col_factor(),
                                                  RiskVar = col_factor()
                                                  )) %>% 
                                                  mutate_if(is.character, as.factor)


# Consistency Checks ------------------------------------------------------

freMPL %>% filter(ClaimInd<0) %>% tally() # Check claims less than zero

freMPL %>% filter(ClaimInd>0) %>% mutate(test = ClaimAmount>0) %>% pull(test) %>% all() # Check whether each claim has an amount

freMPL %>% filter(ClaimInd==0) %>% mutate(test = ClaimAmount==0) %>% pull(test) %>% all() # Check whether each zero claim does not have an amount

freMPL %>% filter(ClaimInd==0 & ClaimAmount<0) %>% tally() # Count salvage and/or subrogations

freMPL <- freMPL %>% filter((ClaimInd==0 & ClaimAmount==0) | (ClaimInd>0 & ClaimAmount>0)) # Filter out salvage and/or subrogations


# Grouping and adding variables -------------------------------------------

n_class <- 10

freMPL <- freMPL %>% mutate(LicAge = as.factor(ntile(LicAge,n_class)),
                  DrivAge = as.factor(ntile(DrivAge,n_class)),
                  BonusMalus = as.factor(ntile(BonusMalus,n_class)))

variables <- freMPL %>% names() %>% setdiff(c("Exposure", "ClaimAmount", "ClaimInd"))

model_df <- freMPL %>% group_by_at(variables) %>% summarise(exposure = sum(Exposure),
                                                numbers = sum(ClaimInd),
                                                amount = sum(ClaimAmount)) %>% 
                                      mutate(sev = ifelse(numbers == 0, 0, amount/numbers)) %>% 
                                      ungroup()


# Train / Test Split ------------------------------------------------------

model_df <- model_df %>% mutate(id = runif(n(),0,1)) %>% mutate_if(is.factor, fct_lump_min, min = 100)

map(model_df %>% keep(is.factor), ~ table(.x))

model_df <- model_df %>% filter(VehEngine != "Other")

train_df <- model_df %>% filter(id <= .8) %>% select(-id) %>% na.omit() %>% 
            filter(numbers < quantile(numbers, .99) & sev < quantile(sev, .99))

test_df <- model_df %>% filter(id > .8) %>% select(-id) %>% na.omit()

# GLM Modeling ------------------------------------------------------------

### Numbers Modeling

glm_numbers_model <- glm(numbers ~ LicAge + 
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
                            VehEnergy +
                            VehMaxSpeed +
                            VehClass +
                            RiskVar +
                            Garage +
                            offset(log(exposure)), data = train_df, family = poisson(link = "log"))

summary(glm_numbers_model)

### Severity Modeling

glm_sev_model <- glm(sev ~ LicAge + 
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
                    VehEnergy +
                    VehMaxSpeed +
                    VehClass +
                    RiskVar +
                    Garage, data = filter(train_df, sev > 0), weights = numbers, family = Gamma(link = "log"))

summary(glm_sev_model)

### Predictions

test_df <- test_df %>% mutate(glm_numbers_pred = predict(glm_numbers_model, newdata = test_df, type = "response"),
                              glm_sev_pred = predict(glm_sev_model, newdata = test_df, type = "response")) %>% 
                      mutate(glm_amount_pred = glm_numbers_pred * glm_sev_pred)


with(test_df,summary(amount))
with(test_df,summary(glm_amount_pred))


sum_table <- test_df %>% select(amount, glm_amount_pred, exposure) %>% 
  mutate(diff_interval = cut_interval(glm_amount_pred,20)) %>% 
  group_by(diff_interval) %>% 
  summarise(glm_pred_mean = weighted.mean(glm_amount_pred,exposure),
            obs_mean = weighted.mean(amount,exposure),
            exposure = sum(exposure))
  
sum_table %>% 
  ggplot()+
  geom_bar(aes(x = diff_interval, y = exposure), stat="identity")+
  geom_line(aes(x = diff_interval, y = glm_pred_mean* ( max(sum_table$exposure)/max(sum_table$glm_pred_mean,sum_table$obs_mean) ), group=1, color = "Predicted"))+
  geom_point(aes(x = diff_interval, y = glm_pred_mean*( max(sum_table$exposure)/max(sum_table$glm_pred_mean,sum_table$obs_mean) ), group=1, color = "Predicted"))+
  geom_line(aes(x = diff_interval, y = obs_mean*( max(sum_table$exposure)/max(sum_table$glm_pred_mean,sum_table$obs_mean) ), group=1, color = "Observed"))+
  geom_point(aes(x = diff_interval, y = obs_mean*( max(sum_table$exposure)/max(sum_table$glm_pred_mean,sum_table$obs_mean) ), group=1, color = "Observed"))+
  scale_y_continuous(sec.axis = sec_axis(~./( max(sum_table$exposure)/max(sum_table$glm_pred_mean,sum_table$obs_mean) ), name = "Weighted Mean"))+
  labs(y = "Exposure",
        x = "Absolute Value",
        colour = "Legend", title = "GLM Results")

ggsave(filename = "GLM_results.png", device = "png", width = 12, height = 6.75, dpi = 320, units = "in")


test_df %>% arrange(glm_amount_pred) %>% 
  mutate(glm_lc_pred = glm_amount_pred / exposure,
         cum_loss = cumsum(glm_lc_pred)/sum(glm_lc_pred),
         cum_exp = cumsum(exposure)/sum(exposure)) %>% 
  select(cum_loss, cum_exp) %>%
  ggplot()+
  geom_line(aes(x = cum_exp, y = cum_loss))+
  geom_abline(slope = 1, intercept = 0)+
  labs(title = "GLM Gini")


glm_gini <- gini(test_df %>% arrange(glm_amount_pred) %>% 
                   mutate(glm_lc_pred = glm_amount_pred / exposure,
                          cum_loss = cumsum(glm_lc_pred)/sum(glm_lc_pred),
                          cum_exp = cumsum(exposure)/sum(exposure)) %>% 
                   select(cum_exp, cum_loss))

ggsave(filename = "GLM_gini.png", device = "png", width = 12, height = 6.75, dpi = 320, units = "in")


glm_nrmse <- with(test_df,NRMSE(glm_amount_pred, amount)) # NRMSE


# GAM Modeling ------------------------------------------------------------

### Numbers Modeling

gam_numbers_model <- gam(numbers ~ LicAge + 
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
                       VehEnergy +
                       VehMaxSpeed +
                       VehClass +
                       RiskVar +
                       Garage+
                       offset(log(exposure)), data = train_df, family = poisson(link = "log"))

summary(gam_numbers_model)

### Severity Modeling

gam_sev_model <- gam(sev ~ LicAge + 
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
                   VehEnergy +
                   VehMaxSpeed +
                   VehClass +
                   RiskVar +
                   Garage, data = filter(train_df, sev > 0), weights = numbers, family = Gamma(link = "log"))

summary(gam_sev_model)

### Predictions

test_df <- test_df %>% mutate(gam_numbers_pred = predict(gam_numbers_model, newdata = test_df, type = "response", offset=exposure),
                              gam_sev_pred = predict(gam_sev_model, newdata = test_df, type = "response")) %>% 
  mutate(gam_amount_pred = gam_numbers_pred * gam_sev_pred)


test_df %>% select(amount, gam_amount_pred) %>% gather() %>% 
  ggplot()+
  geom_histogram(aes(y=..density.., x=value, fill=key))


sum_table <- test_df %>% select(amount, gam_amount_pred, exposure) %>% 
  mutate(diff_interval = cut_interval(gam_amount_pred,20)) %>% 
  group_by(diff_interval) %>% 
  summarise(gam_pred_mean = weighted.mean(gam_amount_pred,exposure),
            obs_mean = weighted.mean(amount,exposure),
            exposure = sum(exposure))

sum_table %>% 
  ggplot()+
  geom_bar(aes(x = diff_interval, y = exposure), stat="identity")+
  geom_line(aes(x = diff_interval, y = gam_pred_mean* ( max(sum_table$exposure)/max(sum_table$gam_pred_mean,sum_table$obs_mean) ), group=1, color = "Predicted"))+
  geom_point(aes(x = diff_interval, y = gam_pred_mean*( max(sum_table$exposure)/max(sum_table$gam_pred_mean,sum_table$obs_mean) ), group=1, color = "Predicted"))+
  geom_line(aes(x = diff_interval, y = obs_mean*( max(sum_table$exposure)/max(sum_table$gam_pred_mean,sum_table$obs_mean) ), group=1, color = "Observed"))+
  geom_point(aes(x = diff_interval, y = obs_mean*( max(sum_table$exposure)/max(sum_table$gam_pred_mean,sum_table$obs_mean) ), group=1, color = "Observed"))+
  scale_y_continuous(sec.axis = sec_axis(~./( max(sum_table$exposure)/max(sum_table$gam_pred_mean,sum_table$obs_mean) ), name = "Weighted Mean"))+
  labs(y = "Exposure",
       x = "Absolute Value",
       colour = "Legend", title = "GAM Results")

ggsave(filename = "GAM_results.png", device = "png",width = 12, height = 6.75, dpi = 320, units = "in")


test_df %>% arrange(gam_amount_pred) %>% 
  mutate(gam_lc_pred = gam_amount_pred / exposure,
         cum_loss = cumsum(gam_lc_pred)/sum(gam_lc_pred),
         cum_exp = cumsum(exposure)/sum(exposure)) %>% 
  select(cum_loss, cum_exp) %>%
  ggplot()+
  geom_line(aes(x = cum_exp, y = cum_loss))+
  geom_abline(slope = 1, intercept = 0)+
  labs(title = "GAM Gini")


gam_gini <- gini(test_df %>% arrange(gam_amount_pred) %>% 
                   mutate(gam_lc_pred = gam_amount_pred / exposure,
                          cum_loss = cumsum(gam_lc_pred)/sum(gam_lc_pred),
                          cum_exp = cumsum(exposure)/sum(exposure)) %>% 
                   select(cum_exp, cum_loss))

ggsave(filename = "GAM_gini.png", device = "png",width = 12, height = 6.75, dpi = 320, units = "in")


gam_nrmse <- NRMSE(test_df$gam_amount_pred, test_df$amount) # NRMSE


# MARS Modeling ------------------------------------------------------------

### Numbers Modeling

mars_numbers_model <- earth(numbers ~ LicAge + 
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
                       VehEnergy +
                       VehMaxSpeed +
                       VehClass +
                       RiskVar +
                       Garage+
                       offset(log(exposure)), data = train_df, glm = list(family = poisson(link = "log")))

summary(mars_numbers_model)

### Severity Modeling

mars_sev_model <- earth(sev ~ LicAge + 
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
                   VehEnergy +
                   VehMaxSpeed +
                   VehClass +
                   RiskVar +
                   Garage, data = filter(train_df, sev > 0), weights = numbers, glm = list(family = Gamma(link = "log")))

summary(mars_sev_model)

### Predictions

test_df <- test_df %>% mutate(mars_numbers_pred = drop(predict(mars_numbers_model, newdata = test_df, type = "response")),
                              mars_sev_pred = drop(predict(mars_sev_model, newdata = test_df, type = "response"))) %>% 
  mutate(mars_amount_pred = mars_numbers_pred * mars_sev_pred)



test_df %>% select(amount, mars_amount_pred) %>% gather() %>% 
  ggplot()+
  geom_histogram(aes(y=..density.., x=value, fill=key))


sum_table <- test_df %>% select(amount, mars_amount_pred, exposure) %>%
  mutate(diff_interval = cut_interval(mars_amount_pred,20)) %>% 
  group_by(diff_interval) %>% 
  summarise(mars_pred_mean = weighted.mean(mars_amount_pred,exposure),
            obs_mean = weighted.mean(amount,exposure),
            exposure = sum(exposure))

sum_table %>% 
  ggplot()+
  geom_bar(aes(x = diff_interval, y = exposure), stat="identity")+
  geom_line(aes(x = diff_interval, y = mars_pred_mean* ( max(sum_table$exposure)/max(sum_table$mars_pred_mean,sum_table$obs_mean) ), group=1, color = "Predicted"))+
  geom_point(aes(x = diff_interval, y = mars_pred_mean*( max(sum_table$exposure)/max(sum_table$mars_pred_mean,sum_table$obs_mean) ), group=1, color = "Predicted"))+
  geom_line(aes(x = diff_interval, y = obs_mean*( max(sum_table$exposure)/max(sum_table$mars_pred_mean,sum_table$obs_mean) ), group=1, color = "Observed"))+
  geom_point(aes(x = diff_interval, y = obs_mean*( max(sum_table$exposure)/max(sum_table$mars_pred_mean,sum_table$obs_mean) ), group=1, color = "Observed"))+
  scale_y_continuous(sec.axis = sec_axis(~./( max(sum_table$exposure)/max(sum_table$mars_pred_mean,sum_table$obs_mean) ), name = "Weighted Mean"))+
  labs(y = "Exposure",
       x = "Absolute Value",
       colour = "Legend", title = "MARS Results")

ggsave(filename = "MARS_results.png", device = "png", width = 12, height = 6.75, dpi = 320, units = "in")

test_df %>% arrange(mars_amount_pred) %>% 
  mutate(mars_lc_pred = mars_amount_pred / exposure,
         cum_loss = cumsum(mars_lc_pred)/sum(mars_lc_pred),
         cum_exp = cumsum(exposure)/sum(exposure)) %>% 
  select(cum_loss, cum_exp) %>%
  ggplot()+
  geom_line(aes(x = cum_exp, y = cum_loss))+
  geom_abline(slope = 1, intercept = 0)+
  labs(title = "MARS Gini")


mars_gini <- gini(test_df %>% arrange(mars_amount_pred) %>% 
                   mutate(gam_lc_pred = gam_amount_pred / exposure,
                          cum_loss = cumsum(gam_lc_pred)/sum(gam_lc_pred),
                          cum_exp = cumsum(exposure)/sum(exposure)) %>% 
                   select(cum_exp, cum_loss))


ggsave(filename = "MARS_Gini.png", device = "png", width = 12, height = 6.75, dpi = 320, units = "in")

mars_nrmse <- NRMSE(test_df$mars_amount_pred, test_df$amount) # NRMSE


