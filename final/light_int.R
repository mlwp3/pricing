library(flashlight)

rm(list = ls())

set.seed(999)

source("./Utils/utils.R")

# Import Data -------------------------------------------------------------

train <- import_data("./Data/train.csv") %>%
  mutate(Severity = ifelse(ClaimNb == 0, 0, ClaimAmount / ClaimNb))

test <- import_data("./Data/test.csv") %>% 
  mutate(Severity = ifelse(ClaimNb == 0, 0, ClaimAmount / ClaimNb))

df <- train %>% mutate(id = row_number())

train <- df %>% sample_frac(.70)

validation <- anti_join(df, train, by = "id") %>% select(-id)

train <- train %>% select(-id)

xgb_model_numb <- xgb.load("./Models/GBM/xgb_numbers")

xgb_model_sev <- xgb.load("./Models/GBM/xgb_severity")

xgb_model_losses <- xgb.load("./Models/GBM/xgb_losses")

predict_lc_xgb <- function(model, data){
  data_post <- xgb.DMatrix(data = create_data_losses(data), 
                           label = data %>% pull(ClaimAmount), 
                           base_margin = data %>% pull(Exposure) %>% log())
  predict(xgb_model_losses, data_post)
}

# predict_lc_xgb(xgb_model_losses, test)

flash_xgb <- flashlight(
  model = xgb_model_losses, 
  label = "xgb",
  y = "ClaimAmount", 
  data = test, 
  predict_function = predict_lc_xgb)

int <- light_interaction(flash_xgb, grid_size = 30, n_max = 50, seed = 42)

int$data

plot(int)
