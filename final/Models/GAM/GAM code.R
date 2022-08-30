#First, install packages if not present (add any other missing packages)
for(pkg in c("Metrics","dplyr","tidyr","gam","mgcv","mda","ggplot2","magrittr","pracma")){
  if(!pkg %in% installed.packages()){
    install.packages(pkg)
  }
}

#Load up libraries
library(mgcv)
library(Metrics)
#library()

#source utils
source("https://raw.githubusercontent.com/mlwp3/pricing/master/final/Utils/utils.R")

train<-read.csv("https://github.com/mlwp3/pricing/blob/master/final/Data/train.csv?raw=true",stringsAsFactors = FALSE)
test<-read.csv("https://github.com/mlwp3/pricing/blob/master/final/Data/test.csv?raw=true",stringsAsFactors = FALSE)

#Set up models

fre.gam.freq<-gam(ClaimNb ~
                    Area +
                    VehPower +
                    s(VehAge,k=9) +
                    s(DrivAge,k=9) +
                    VehBrand +
                    VehGas +
                    s(Density,k=9) +
                    Region +
                    BonusMalus,
                  family = poisson(link="log"),
                  offset = log(Exposure),
                  data = train,
                  weights = rep(1,nrow(train))
                )

fre.gam.sev<-gam(severity ~
                    Area +
                    VehPower +
                    s(VehAge,k=9) +
                    s(DrivAge,k=9) +
                    VehBrand +
                    VehGas +
                    BonusMalus +
                    s(Density,k=9) + 
                    Region,
                 data = train[train$severity>0,],
                 family=Gamma(link="log"),
                 weights = ClaimNb
)

fre.gam.lc<-gam(ClaimAmount ~
                  Area +
                  VehPower +
                  VehBrand +
                  VehGas +
                  s(Density) +
                  Region +
                  BonusMalus +
                  s(DrivAge) +
                  s(VehAge),
                offset = log(Exposure),
                data = train
)


#Predict outcomes

pred.gam.freq <- predict(fre.gam.freq, newdata = test, type = "response")
pred.gam.sev <- predict(fre.gam.sev, newdata = test, type = "response")
pred.gam.lc <- predict(fre.gam.lc, newdata = test, type = "response")


test$GAM<-pred.gam.freq*pred.gam.sev
test$GAM_LC<-pred.gam.lc
rmse(data=test,targetvar = "ClaimAmount",prediction.obj = test$GAM)
rmse(data=test,targetvar = "ClaimAmount",prediction.obj = test$GAM_LC)

#evaluate results

#loss cost

#Freq sev
#RMSE
test %$% RMSE(predicted_loss_cost_FS, observed_loss_cost)
#NRMSE
test %$% NRMSE(predicted_loss_cost_FS, observed_loss_cost)
#MAE
test %$% MAE(predicted_loss_cost_FS, observed_loss_cost)
#Pearson correlation
test %$% cor(predicted_loss_cost_FS,observed_loss_cost,method="pearson")
#Spearman correlation
test %$% cor(predicted_loss_cost_FS,observed_loss_cost,method="spearman")
#Gini index
test %$% gini_value(observed_loss_cost,predicted_loss_cost_FS, Exposure)
#Aggregate RP differential
test %$% agg_rpr(observed_loss_cost,predicted_loss_cost_FS)
#Lift Curve

#Double Lift Chart
