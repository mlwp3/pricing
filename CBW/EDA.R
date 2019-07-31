# Load packages -----------------------------------------------------------

library(tidyverse)

# Load data ---------------------------------------------------------------

dataset <- read_csv("../data/freMPL.csv")

# View dataset ------------------------------------------------------------

dataset %>% summary()
dataset %>% select(ClaimAmount) %>% as.matrix() %>% hist
dataset %>% select(ClaimInd) %>% table

# Create features ---------------------------------------------------------


