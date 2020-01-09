#
if(!"DALEX" %in% installed.packages())
  install.packages("DALEX")

library(DALEX)

#Provide training model, training data and list of variables to view partial dependence plots
pdp_breakdowns<-function(train.model,train.data,var.list){
  for(trainvar in var.list){
    model.resp<-feature_response(explain(train.model,data=train.data),feature=trainvar,"pdp")
    cat(trainvar," breakdown\n\n",
        head(model.resp),"\n\n")
    print(plot(model.resp))
  }
}

#To add model performance + variable_importance using loss RMSE function