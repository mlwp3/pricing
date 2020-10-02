library(dplyr)
library(caret)
library(readr)

load("~/Desktop/CAS MLWP/Pricing/freMTPL Data/freMTPL2freq.rda")
freq <- freMTPL2freq
rm(freMTPL2freq)

sev <- load("~/Desktop/CAS MLWP/Pricing/freMTPL Data/freMTPL2sev.rda")
sev <- freMTPL2sev
rm(freMTPL2sev)

colnames(freq)
colnames(sev)

dat <- freq %>%
  left_join(sev, by = "IDpol") %>%
  mutate(ClaimAmount = ifelse(is.na(ClaimAmount), 0, ClaimAmount),
         ClaimNb = as.numeric(ClaimNb))

colnames(dat)

dat <- dat %>%
  dplyr::group_by(IDpol) %>%
  mutate(ClaimAmount = cumsum(ClaimAmount),
         id = row_number(), max_id = max(id)) %>%
  filter(id == max_id) %>%
  select(-c(id, max_id)) %>%
  filter(!(ClaimNb > 0 & ClaimAmount == 0))

dat <- dat %>%
  filter(Exposure > (1/12) & Exposure <= 1 & ClaimAmount < 1000000) %>%
  mutate(RecordID = row_number(), severity = ifelse(ClaimNb == 0, 0, ClaimAmount / ClaimNb),
         DrivAgeBand = as.character(cut(DrivAge, c(17, 25, 35, 45, 55, 65, Inf))),
         DensityBand = as.character(cut(Density, c(0, 50, 100, 200, 400, 1000, 2000, 5000, Inf))),
         VehAgeBand = ifelse(VehAge <= 15, as.character(VehAge), "15+"),
         BonusMalus = BonusMalus / 100,
         VehPower = as.character(VehPower))

dat <- as.data.frame(dat)

dat <- dat %>%
  mutate_if(is.character, as.factor)

set.seed(4223)
ind <- createDataPartition(dat$ClaimAmount, 0.7, times = 1, list = FALSE)
train <- dat[ind, ]
test <- dat[-ind, ]

mean(train$ClaimAmount); mean(test$ClaimAmount); var(train$ClaimAmount); var(test$ClaimAmount)
rm(ind)

readr::write_csv(train, "~/Desktop/CAS MLWP/Pricing/freMTPL Data/train_new_final.csv")
readr::write_csv(test, "~/Desktop/CAS MLWP/Pricing/freMTPL Data/test_new_final.csv")
