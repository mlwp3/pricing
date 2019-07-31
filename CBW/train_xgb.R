# Load packages -----------------------------------------------------------

library(tidyverse)
library(xgboost)

# Load data ---------------------------------------------------------------

dataset <- read_csv("../data/freMPL.csv")

# Make features -----------------------------------------------------------

dataset <- dataset %>% filter(ClaimAmount >= 0)

sev <- dataset$ClaimAmount
dataset <- dataset %>% select(-c(ClaimAmount, ClaimInd, X1, RecordBeg, RecordEnd))

# Make model matrix -------------------------------------------------------

train_id <- sample(1:nrow(dataset), floor(0.8 * nrow(dataset)))
test_id <- (1:nrow(dataset))[-train_id]

model_matrix_data <- model.matrix(c(sev[train_id], sev[test_id]) ~ ., data = rbind(dataset[train_id, ], dataset[test_id, ]))
train_model_matrix <- model_matrix_data[1:length(train_id), ]
test_model_matrix <- model_matrix_data[(length(train_id) + 1):nrow(model_matrix_data), ]

dtrain <- xgb.DMatrix(data = train_model_matrix, label = sev[train_id])
dtest <- xgb.DMatrix(data = test_model_matrix, label = sev[test_id])

# Train model -------------------------------------------------------------

parameterList <- c(nrounds = 1000, eta = 0.01, max_depth = 2, gamma = 10, subsample = 0.5)

current_model <- xgb.train(data =  dtrain, 
                           verbose = TRUE,
                           eval_metric="tweedie-nloglik@1.2",
                           objective = "reg:tweedie", 
                           tweedie_variance_power = 1.5, 
                           nrounds = parameterList[["nrounds"]],
                           max.depth = parameterList[["max_depth"]],
                           gamma = parameterList[["gamma"]], 
                           eta = parameterList[["eta"]],                               
                           subsample = parameterList[["subsample"]])

pred_gamma <- predict(current_model, newdata = dtest)

par(mfrow = c(1, 2))
hist(pred_gamma)
hist(sev[test_id])

mean((pred_gamma - sev[test_id])^2)
