library("lattice")

MyData = read.csv("freMPL.csv")
Claim = table(MyData$ClaimInd)
Bonus = table(MyData$BonusMalus)
SCP = table(MyData$SocioCateg)



barplot(Claim)
hist(MyData$Exposure)
barplot(Bonus)
barplot(SCP)

histogram(MyData$ClaimInd ~ MyData$Gender)
histogram(MyData$ClaimInd ~ MyData$Gender=="Female")

sum(MyData$ClaimInd[MyData$Gender=="Female"])/length(MyData$ClaimInd[MyData$Gender=="Female"])

sum(MyData$ClaimInd[MyData$Gender=="Male"])/length(MyData$ClaimInd[MyData$Gender=="Male"])


index = MyData$Gender=="Female"
FemaleClaim = MyData$ClaimInd[index]
