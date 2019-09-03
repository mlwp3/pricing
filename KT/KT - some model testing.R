library(magrittr)
library(tweedie)
library(statmod)
library(dplyr)
library(CASdatasets)
library(MASS)

data(freMPL1,freMPL2,freMPL3,freMPL4)
freMPL<-rbind(freMPL1,freMPL2,freMPL3[,c(1:11,13:23)],freMPL4[,c(1:11,13:23)])
dat<-unique(freMPL)

#Using Navarun's prepping code for consistency

dat <- dat[dat$Exposure > 0, ]
dat <- dat[dat$Exposure <= 1, ]
dat <- dat[dat$ClaimAmount >= 0, ]

#Convert HasKmLimit to a factor
dat$HasKmLimit <- as.factor(dat$HasKmLimit)

#Look at Vehicle Price - it has too many levels so might be a good idea to condense
veh_price_grp <- dat %>%
  group_by(VehPrice) %>%
  summarise(burning_cost = sum(ClaimAmount) / sum(Exposure))
#Above table shows good reason to club vehicle price groups based on burning cost - trying k-means clustering here
set.seed(34)
veh_price_cluster <- kmeans(veh_price_grp[, c(2)], 5, nstart = 20)
veh_price_grp$VehPriceGrp <- as.factor(veh_price_cluster$cluster)
#Map VehPrice to VehPriceGrp in the raw data
dat$VehPriceGrp <- as.factor(unlist(veh_price_grp[match(dat$VehPrice, veh_price_grp$VehPrice), c("VehPriceGrp")]))
dat <- dat[, -which(colnames(dat) == c("VehPrice"))] #Remove VehPrice from the data since we model VehPriceGrp
rm(veh_price_cluster)
rm(veh_price_grp)

#Looking at VehEngine and VehEnergy
veh_engine_grp <- dat %>%
  group_by(VehEngine) %>%
  summarise(Exposure = sum(Exposure), row_count = n(), claim = sum(ClaimAmount)) #Only 1 row for GPL with no claim and very low Exposure, safe to remove
dat <- dat[!dat$VehEngine == c("GPL"), ]
rm(veh_engine_grp)

veh_energy_grp <- dat %>%
  group_by(VehEnergy) %>%
  summarise(Exposure = sum(Exposure), row_count = n(), claim = sum(ClaimAmount)) #Not too many electric vehicles, keep for now
rm(veh_energy_grp)

#However let's have a dat2 that doesn't bucket continuous variables
dat2<-dat

#Also let's keep one that includes the top 10 SocioCateg,
# so we can use the SocioCateg variable. Top 10 values cover 98.3% of value
dat2.soc<-dat[dat$SocioCateg %in% names(head(sort(table(dat$SocioCateg),decreasing=TRUE),10)),]

dat$LicAge <- cut(dat$LicAge, breaks = c(0, 365, 730, 1095), right = FALSE)
dat %>%
  group_by(LicAge) %>%
  summarise(Exposure = sum(Exposure)) #Hardly any Exposure in the 2+ years category, so dropping this
dat$LicAge <- as.character(dat$LicAge)
dat <- dat[-which(as.character(dat$LicAge) %in% c("[730,1.1e+03)")), ]
dat$LicAge <- ifelse(dat$LicAge == c("[0,365)"), c("0 - 1"), c("1 - 2")) #Converting LicAge from Days to Years
dat$LicAge <- as.factor(dat$LicAge)

#Band DrivAge
dat$DrivAge <- cut(dat$DrivAge, breaks = c(18, 25, 35, 45, 55, 65, 110), right = FALSE)
dat %>%
  group_by(DrivAge) %>%
  summarise(Exposure = sum(Exposure))
dat$DrivAge <- as.character(dat$DrivAge)
dat$DrivAge <- ifelse(dat$DrivAge == c("[65,110)"), c("[65+)"), dat$DrivAge)
dat$DrivAge <- as.factor(dat$DrivAge)

#Band Bonus-Malus according to the schema in the data description
dat$BonusMalus <- ifelse(dat$BonusMalus < 100, "Bonus", "Malus")
dat$BonusMalus <- as.factor(dat$BonusMalus)

dat.soc<-dat[dat$SocioCateg %in% names(head(sort(table(dat$SocioCateg),decreasing=TRUE),10)),]

#Split into Train and Test
set.seed(1245788)
ind <- sample(1:nrow(dat), round(0.3 * nrow(dat)))
train <- dat[-ind, ]
test <- dat[ind, ]
ind <- sample(1:nrow(dat2), round(0.3 * nrow(dat2)))
train2 <- dat2[-ind,]
test2 <- dat2[ind,]
ind <- sample(1:nrow(dat.soc), round(0.3 * nrow(dat.soc)))
train.soc <- dat.soc[-ind,]
test.soc <- dat.soc[ind,]
ind <- sample(1:nrow(dat2.soc), round(0.3 * nrow(dat2.soc)))
train2.soc <- dat2.soc[-ind,]
test2.soc <- dat2.soc[ind,]
rm(ind)

#Define RMSE and MAE metrics for Model Evaluation
rmse <- function(data, targetvar, prediction.obj) {
  rss <- (as.numeric(prediction.obj) - as.numeric(data[, c(targetvar)])) ^ 2
  mse <- sum(rss) / nrow(data)
  rmse <- sqrt(mse)
  rmse <- rmse / 100
  return(rmse)
}

targetvar <- c("ClaimAmount")
predictors <- c("LicAge", "VehAge", "Gender", "MariStat", "VehUsage", "DrivAge", "HasKmLimit", "BonusMalus", "VehBody", "VehPriceGrp", "VehEngine",
                "VehEnergy", "VehMaxSpeed", "VehClass", "Garage")
predictors.soc <- c("LicAge", "VehAge", "Gender", "MariStat", "VehUsage", "DrivAge", "HasKmLimit", "BonusMalus", "VehBody", "VehPriceGrp", "VehEngine",
                "VehEnergy", "VehMaxSpeed", "VehClass", "Garage","SocioCateg")
predictors.gam <- c("s(LicAge)", "VehAge", "Gender", "MariStat", "VehUsage", "s(DrivAge)", "HasKmLimit", "s(BonusMalus)", "VehBody", "VehPriceGrp", "VehEngine",
                "VehEnergy", "VehMaxSpeed", "VehClass", "Garage")
predictors.gam.soc <- c("s(LicAge)", "VehAge", "Gender", "MariStat", "VehUsage", "s(DrivAge)", "HasKmLimit", "s(BonusMalus)", "VehBody", "VehPriceGrp", "VehEngine",
                "VehEnergy", "VehMaxSpeed", "VehClass", "Garage","SocioCateg")
predictors2 <- paste(predictors, collapse = "+")
predictors2.gam <- paste(predictors.gam, collapse = "+")
predictors2.soc <- paste(predictors.soc, collapse = "+")
predictors2.gam.soc <- paste(predictors.gam.soc, collapse = "+")
formula <- as.formula(paste(targetvar,"~",predictors2,collapse = "+"))
formula.gam <- as.formula(paste(targetvar,"~",predictors2.gam,collapse = "+"))
formula.soc <- as.formula(paste(targetvar,"~",predictors2.soc,collapse = "+"))
formula.gam.soc <- as.formula(paste(targetvar,"~",predictors2.gam.soc,collapse = "+"))
#For consistency's sake let's use a common formula for all models

p_values <- seq(1.1, 1.8, by = 0.1)
p_tuning <- tweedie.profile(formula = formula, data = train, p.vec = p_values, do.plot = FALSE, offset = log(Exposure))
p_tuning2 <- tweedie.profile(formula = formula, data = train2, p.vec = p_values, do.plot = FALSE, offset = log(Exposure))
p_tuning.soc <- tweedie.profile(formula = formula.soc, data = train.soc, p.vec = p_values, do.plot = FALSE, offset = log(Exposure))
p_tuning2.soc <- tweedie.profile(formula = formula.soc, data = train2.soc, p.vec = p_values, do.plot = FALSE, offset = log(Exposure))
p <- p_tuning$p.max
p2 <- p_tuning2$p.max
p.soc<-p_tuning.soc$p.max
p2.soc<-p_tuning2.soc$p.max
#rm(p_values)
#rm(p_tuning)
#rm(p_tuning2)
#rm(p_tuning.soc)
#rm(p_tuning2.soc)

##========TWEEDIE GLM==========##
#Starting out with the same baseline as a yardstick
fre.GLM <- glm(formula = formula, data = train, family = tweedie(var.power = p, link.power = 0), offset = log(Exposure))
fre.GLM2 <- glm(formula = formula, data = train2, family = tweedie(var.power = p2, link.power = 0), offset = log(Exposure))
fre.GLM.soc <- glm(formula = formula.soc, data = train.soc, family = tweedie(var.power = p.soc, link.power = 0), offset = log(Exposure))
fre.GLM2.soc <- glm(formula = formula.soc, data = train2.soc, family = tweedie(var.power = p2.soc, link.power = 0), offset = log(Exposure))

predictions_glm <- predict(fre.GLM, test, type = "response")
predictions_glm2 <- predict(fre.GLM2, test2, type = "response")
predictions_glm.soc <- predict(fre.GLM.soc, test.soc, type = "response")
predictions_glm2.soc <- predict(fre.GLM2.soc, test2.soc, type = "response")

rmse_glm <- rmse(data = test, targetvar = targetvar, prediction.obj = predictions_glm)
rmse_glm2 <- rmse(data = test2, targetvar = targetvar, prediction.obj = predictions_glm2)
rmse_glm.soc <- rmse(data = test.soc, targetvar = targetvar, prediction.obj = predictions_glm.soc)
rmse_glm2.soc <- rmse(data = test2.soc, targetvar = targetvar, prediction.obj = predictions_glm2.soc)
rmse_glm
rmse_glm2
rmse_glm.soc
rmse_glm2.soc
 
#22.32285
#24.41855
#26.76682
#19.84867

#Should we also try zero-inflated negative binomial #using library(pscl)?

##===========GAM===========##
library(gam)
fre.GAM<-gam(formula = formula,data=train, family = tweedie(var.power = p, link.power = 0), offset = log(Exposure))
predictions_gam<-predict(fre.GAM,test,type = "response")
rmse_gam<-rmse(data=test, targetvar = targetvar, prediction.obj = predictions_gam)
fre.GAM.soc<-gam(formula = formula.soc,data=train.soc, family = tweedie(var.power = p.soc, link.power = 0), offset = log(Exposure))
predictions_gam.soc<-predict(fre.GAM.soc,test.soc,type = "response")
rmse_gam.soc<-rmse(data=test.soc, targetvar = targetvar, prediction.obj = predictions_gam.soc)
fre.GAM2<-gam(formula = formula,data=train2, family = tweedie(var.power = p2, link.power = 0), offset = log(Exposure))
predictions_gam2<-predict(fre.GAM2,test2,type = "response")
rmse_gam2<-rmse(data=test2, targetvar = targetvar, prediction.obj = predictions_gam2)
fre.GAM2.smooth<-gam(formula = formula.gam,data=train2, family = tweedie(var.power = p2, link.power = 0), offset = log(Exposure))
predictions_gam2.smooth<-predict(fre.GAM2.smooth,test2,type = "response")
rmse_gam2.smooth<-rmse(data=test2, targetvar = targetvar, prediction.obj = predictions_gam2.smooth)
fre.GAM2.soc<-gam(formula = formula.soc,data=train2.soc, family = tweedie(var.power = p2.soc, link.power = 0), offset = log(Exposure))
predictions_gam2.soc<-predict(fre.GAM2.soc,test2.soc,type = "response")
rmse_gam2.soc<-rmse(data=test2.soc, targetvar = targetvar, prediction.obj = predictions_gam2.soc)
fre.GAM2.soc.smooth<-gam(formula = formula.gam.soc, data=train2.soc, family = tweedie(var.power = p2.soc, link.power = 0), offset = log(Exposure))
predictions_gam2.soc.smooth<-predict(fre.GAM2.soc.smooth,test2.soc,type = "response")
rmse_gam2.soc.smooth<-rmse(data=test2.soc, targetvar = targetvar, prediction.obj = predictions_gam2.soc.smooth)

rmse_gam #22.34885
rmse_gam.soc
rmse_gam2
rmse_gam2.smooth
rmse_gam2.soc
rmse_gam2.soc.smooth #Seems one of the best-performing ones

##===========PCR===========##
library(pls)
fre.PCR<-pcr(formula = formula, data = train)
predictions_pcr<-predict(fre.PCR,test)
rmse_pcr<-rmse(test,targetvar,prediction.obj=predictions_pcr)
fre.PCR.soc<-pcr(formula = formula.soc, data = train.soc)
predictions_pcr.soc<-predict(fre.PCR.soc,test.soc)
rmse_pcr.soc<-rmse(test.soc,targetvar,prediction.obj=predictions_pcr.soc)
fre.PCR2<-pcr(formula = formula, data = train2)
predictions_pcr2<-predict(fre.PCR2,test2)
rmse_pcr2<-rmse(test2,targetvar,prediction.obj=predictions_pcr2)
fre.PCR2.soc<-pcr(formula = formula.soc, data = train2.soc)
predictions_pcr2.soc<-predict(fre.PCR2.soc,test2.soc)
rmse_pcr2.soc<-rmse(test2.soc,targetvar,prediction.obj=predictions_pcr2.soc)

rmse_pcr #168.5764
rmse_pcr.soc
rmse_pcr2
rmse_pcr2.soc

##===========SVR===========##
library(e1071)
fre.SVM<-svm(formula = formula, data = train)
predictions_svm<-predict(fre.SVM,test)
rmse_svm<-rmse(test,targetvar,prediction.obj=predictions_svm)
fre.SVM.soc<-svm(formula = formula.soc, data = train.soc)
predictions_svm.soc<-predict(fre.SVM.soc,test.soc)
rmse_svm.soc<-rmse(test.soc,targetvar,prediction.obj=predictions_svm.soc)
fre.SVM2<-svm(formula = formula, data = train2)
predictions_svm2<-predict(fre.SVM2,test2)
rmse_svm2<-rmse(test2,targetvar,prediction.obj=predictions_svm2)
fre.SVM2.soc<-svm(formula = formula.soc, data = train2.soc)
predictions_svm2.soc<-predict(fre.SVM2.soc,test2.soc)
rmse_svm2.soc<-rmse(test2.soc,targetvar,prediction.obj=predictions_svm2.soc)

rmse_svm #22.52824
rmse_svm.soc
rmse_svm2
rmse_svm2.soc


##===========MARS==========##
library(earth)
fre.MARS<-earth(train[,names(train) %in% predictors],train[,targetvar])
predictions_mars<-predict(fre.MARS,test)
rmse_mars<-rmse(test,targetvar,prediction.obj=predictions_mars)
fre.MARS.soc<-earth(train.soc[,names(train.soc) %in% predictors.soc],train.soc[,targetvar])
predictions_mars.soc<-predict(fre.MARS.soc,test.soc)
rmse_mars.soc<-rmse(test.soc,targetvar,prediction.obj=predictions_mars.soc)
fre.MARS2<-earth(train2[,names(train2) %in% predictors],train2[,targetvar])
predictions_mars2<-predict(fre.MARS2,test2)
rmse_mars2<-rmse(test2,targetvar,prediction.obj=predictions_mars2)
fre.MARS2.soc<-earth(train2.soc[,names(train2.soc) %in% predictors.soc],train2.soc[,targetvar])
predictions_mars2.soc<-predict(fre.MARS2.soc,test2.soc)
rmse_mars2.soc<-rmse(test2.soc,targetvar,prediction.obj=predictions_mars2.soc)

rmse_mars #22.35809
rmse_mars.soc
rmse_mars2
rmse_mars2.soc

##===========M5============##
library(Cubist)
#need to remove commas because M5 doesn't like them
train.m5<-train
test.m5<-test
train.m52<-train2
test.m52<-test2
train.m5.soc<-train.soc
test.m5.soc<-test.soc
train.m52.soc<-train2.soc
test.m52.soc<-test2.soc
train.m5$DrivAge<-gsub(",","",train.m5$DrivAge)
test.m5$DrivAge<-gsub(",","",test.m5$DrivAge)
train.m52$DrivAge<-gsub(",","",train.m52$DrivAge)
test.m52$DrivAge<-gsub(",","",test.m52$DrivAge)
train.m5.soc$DrivAge<-gsub(",","",train.m5.soc$DrivAge)
test.m5.soc$DrivAge<-gsub(",","",test.m5.soc$DrivAge)
train.m52.soc$DrivAge<-gsub(",","",train.m52.soc$DrivAge)
test.m52.soc$DrivAge<-gsub(",","",test.m52.soc$DrivAge)
fre.M5<-cubist(train.m5[,names(train.m5) %in% predictors],train.m5[,targetvar])
predictions_m5<-predict(fre.M5,test.m5)
rmse_m5<-rmse(test.m5,targetvar,prediction.obj=predictions_m5)
fre.M5.soc<-cubist(train.m5.soc[,names(train.m5.soc) %in% predictors.soc],train.m5.soc[,targetvar])
predictions_m5.soc<-predict(fre.M5.soc,test.m5.soc)
rmse_m5.soc<-rmse(test.m5.soc,targetvar,prediction.obj=predictions_m5.soc)
fre.M52<-cubist(train.m52[,names(train.m52) %in% predictors],train.m52[,targetvar])
predictions_m52<-predict(fre.M52,test.m52)
rmse_m52<-rmse(test.m52,targetvar,prediction.obj=predictions_m52)
fre.M52.soc<-cubist(train.m52.soc[,names(train.m52.soc) %in% predictors.soc],train.m52.soc[,targetvar])
predictions_m52.soc<-predict(fre.M52.soc,test.m52.soc)
rmse_m52.soc<-rmse(test.m52.soc,targetvar,prediction.obj=predictions_m52.soc)

rmse_m5 #22.52835
rmse_m5.soc
rmse_m52
rmse_m52.soc

