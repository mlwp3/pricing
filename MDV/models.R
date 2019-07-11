rm(list=ls())

set.seed(1)

library(tidyverse)
library(ggExtra)
library(gridExtra)
library(caret)
library(GGally)
library(forcats)

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

numbers_model <- glm(numbers ~ LicAge + 
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

summary(numbers_model)

### Severity Modeling

sev_model <- glm(sev ~ LicAge + 
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

summary(sev_model)

# Predictions -------------------------------------------------------------

test_df <- test_df %>% mutate(numbers_pred = predict(numbers_model, newdata = test_df, type = "response", offset=exposure),
                              sev_pred = predict(sev_model, newdata = test_df, type = "response")) %>% 
                      mutate(amount_pred = numbers_pred * sev_pred)


test_df %>% select(amount, amount_pred) %>% gather() %>% 
   ggplot()+
   geom_histogram(aes(y=..density.., x=value, fill=key))


test_df %>% select(amount, amount_pred, exposure) %>% mutate(abs_diff = abs(amount - amount_pred)) %>%
  mutate(diff_interval = cut_interval(abs_diff,20)) %>% 
  group_by(diff_interval) %>% 
  summarise(pred_mean = weighted.mean(amount_pred,exposure),
            obs_mean = weighted.mean(amount,exposure),
            exposure = sum(exposure)) %>% 
  ggplot()+
  geom_bar(aes(x = diff_interval, y = exposure), stat="identity")


