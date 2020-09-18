library(CASdatasets)
library(tidyverse)

rm(list = ls())

data("freMTPLfreq")
data("freMTPLsev")
data("freMTPL2freq")
data("freMTPL2sev")

all((freMTPLfreq %>% filter(ClaimNb != 0) %>% pull(PolicyID)) %in% (freMTPLsev %>% pull(PolicyID)))

freMTPL <- freMTPLfreq %>% 
            left_join(freMTPLsev %>% 
                      group_by(PolicyID) %>% 
                       summarise(ClaimAmount = sum(ClaimAmount)), 
                      by = "PolicyID") %>% 
            replace_na(list(ClaimAmount = 0))

all((freMTPL2freq %>% filter(ClaimNb != 0) %>% pull(IDpol)) %in% (freMTPL2sev %>% pull(IDpol))) # There are issues in the data

freMTPL2 <- freMTPL2freq %>% 
  left_join(freMTPL2sev %>% 
              group_by(IDpol) %>% 
              summarise(ClaimAmount = sum(ClaimAmount)),
            by = "IDpol") %>% 
  replace_na(list(ClaimAmount = 0))

freMTPL2 %>% 
  filter(ClaimNb != 0, ClaimAmount == 0)

freMTPL2_fixed <- freMTPL2 %>% 
                    mutate(ClaimNb = ifelse(ClaimAmount == 0, 0, ClaimNb))

freMTPL2_fixed %>% 
  filter(ClaimNb != 0, ClaimAmount == 0)
