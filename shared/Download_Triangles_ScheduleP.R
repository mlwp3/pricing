##########################################################################################
####### LEGEND ##########
# B Private_auto
# C commercial_auto/truck
# D Workers_compensation 
# F2 Medical_malpractice
# H1 Other_liability
# R1 Products_liability
##########################################################################################


rm(list = ls())
library(tidyverse)

##### Define functions

get_triangle <- function(csv_file,grp_code,metric_of_int){
  
  tr<-csv_file %>% filter(GRCODE==grp_code) %>% group_by(AccidentYear,DevelopmentLag) %>% summarise(Amounts=!!as.name(metric_of_int)) %>% 
    spread(DevelopmentLag,Amounts) %>% ungroup() %>% select(-AccidentYear) %>% as.matrix()
  
  rownames(tr)<-as.character(1988:1997)
  colnames(tr)<-as.character(paste0("lag_",0:9))
  names(dimnames(tr)) <- c("ay", "lag")
  
  return(tr)
}

get_prem<-function(csv_file,grp_code){
  
  prem<-file %>% filter(GRCODE==grp_code) %>% group_by(AccidentYear) %>% summarise(Prem=first(EarnedPremDIR_B)) %>% select(-AccidentYear) %>% unlist() %>% as.numeric()
  
  return(prem)
}

get_upper_tri<-function(tr){
  
  tr[lower.tri(tr)[,nrow(tr):1]]<-NA
  return(tr)
  
}

#####



##### Define sources
links<-c("https://www.casact.org/research/reserve_data/ppauto_pos.csv",
          "https://www.casact.org/research/reserve_data/wkcomp_pos.csv",
          "https://www.casact.org/research/reserve_data/comauto_pos.csv",
          "https://www.casact.org/research/reserve_data/medmal_pos.csv",
          "https://www.casact.org/research/reserve_data/prodliab_pos.csv",
          "https://www.casact.org/research/reserve_data/othliab_pos.csv")


names(links)<-c("B","C","D","F2","H1","R1")


tr_types<-c("IncurLoss_","CumPaidLoss_","BulkLoss_")

#####



##### Example

LoB<-"B" ####Select Lob According to Legend

## get sources
file<-read_csv(links[LoB])

grp.code<-unique(file$GRCODE)

metrics<-paste0(tr_types,LoB)


## transform into triangles
Full_Triangle<-get_triangle(csv_file = file, grp_code = grp.code[1], metric_of_int = metrics[1])

Prem<-get_prem(csv_file = file, grp_code = grp.code[1])

Upper_Tri<-get_upper_tri(Full_Triangle)


##plot
matplot(t(Upper_Tri),type="b",main="Rep",ylab="Amounts")

matplot(t(sweep(Upper_Tri,1,Prem,"/")),type="b",main="Rep LR",ylab = "LR")








