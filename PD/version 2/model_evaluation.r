
source("./perf_eval.R")

setwd("C:/Users/pjdys/Documents/git repos/mlwp3/pricing/PD/version 2")
test_fre <- read_csv("freq_test.csv")
test_sev <- read_csv("freq_test.csv")

F_rmse <-  rmse(data = test_fre, targetvar = "freq", prediction.obj = test_set$freq_predicted)

S_rmse <-  rmse(data = test_set, targetvar = "freq", prediction.obj = test_set$freq_predicted)


