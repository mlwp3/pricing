#' ---
#' title: FreMPL Exploratory Data Analysis
#' author: Daniel Lupton
#' date: April 7, 2019
#' ---

setwd("C:/Users/danlu/OneDrive/Documents/Work/MLWP/Group 3")

freMPL <- read.csv("freMPL.csv")
str(freMPL)
summary(freMPL$ClaimAmount)
center <- function(x){
  (x - mean(x))/sd(x)
}
freMPL$C_ClaimAmount <- center(freMPL$ClaimAmount)
freMPL_Pos <- freMPL[which(freMPL$ClaimAmount>0),]
freMPL_Pos$L_ClaimAmount <- log(freMPL_Pos$ClaimAmount)

freMPL_Pos$VehAge <- factor(freMPL_Pos$VehAge, levels=levels(freMPL_Pos$VehAge)[c(1,2,4,5,6,7,8,9,3)])
freMPL$VehAge <- factor(freMPL$VehAge, levels=levels(freMPL$VehAge)[c(1,2,4,5,6,7,8,9,3)])

boxplot(L_ClaimAmount ~ VehAge, data=freMPL_Pos)
boxplot(L_ClaimAmount ~ Garage, data=freMPL_Pos)
boxplot(L_ClaimAmount ~ VehBody, data=freMPL_Pos)
boxplot(L_ClaimAmount ~ VehPrice, data=freMPL_Pos)
boxplot(L_ClaimAmount ~ VehEngine, data=freMPL_Pos)
boxplot(L_ClaimAmount ~ VehEnergy, data=freMPL_Pos)
boxplot(L_ClaimAmount ~ VehMaxSpeed, data=freMPL_Pos)
boxplot(L_ClaimAmount ~ VehClass, data=freMPL_Pos)
boxplot(L_ClaimAmount ~ Gender, data=freMPL_Pos)
boxplot(L_ClaimAmount ~ MariStat, data=freMPL_Pos)
boxplot(L_ClaimAmount ~ SocioCateg, data=freMPL_Pos)
boxplot(L_ClaimAmount ~ VehUsage, data=freMPL_Pos)
plot(L_ClaimAmount ~ Exposure, data=freMPL_Pos)
plot(L_ClaimAmount ~ LicAge, data=freMPL_Pos)
plot(C_ClaimAmount ~ LicAge, data=freMPL, ylim=c(-1.5,2.5))
plot(L_ClaimAmount ~ DrivAge, data=freMPL_Pos)
plot(C_ClaimAmount ~ DrivAge, data=freMPL, ylim=c(-1.5,2.5))
plot(L_ClaimAmount ~ BonusMalus, data=freMPL_Pos)
plot(C_ClaimAmount ~ BonusMalus, data=freMPL, ylim=c(-1.5,2.5))
quantile(freMPL$C_ClaimAmount,c(0,.5,.9,.95,.99))

options(digits=2)
aggregate(freMPL$ClaimInd,list(freMPL$VehAge),mean)
aggregate(freMPL$ClaimInd,list(freMPL$Garage),mean)
aggregate(freMPL$ClaimInd,list(freMPL$VehBody),mean)
aggregate(freMPL$ClaimInd,list(freMPL$VehPrice),mean)
aggregate(freMPL$ClaimInd,list(freMPL$VehEngine),mean)
aggregate(freMPL$ClaimInd,list(freMPL$VehEnergy),mean)
aggregate(freMPL$ClaimInd,list(freMPL$VehMaxSpeed),mean)
aggregate(freMPL$ClaimInd,list(freMPL$VehClass),mean)
aggregate(freMPL$ClaimInd,list(freMPL$Gender),mean)
aggregate(freMPL$ClaimInd,list(freMPL$MariStat),mean)
aggregate(freMPL$ClaimInd,list(freMPL$SocioCateg),mean)
aggregate(freMPL$ClaimInd,list(freMPL$VehUsage),mean)
