rm(list=ls())

library(tidyverse)
library(scales)
library(ggExtra)
library(gridExtra)

freMPL <- read_csv("freMPL.csv", col_types = cols(Gender = col_character(), 
                                                  RecordBeg = col_date(format = "%Y-%m-%d"), 
                                                  RecordEnd = col_date(format = "%Y-%m-%d"), 
                                                  VehAge = col_character(), X1 = col_skip())) %>% mutate_if(is.character, as.factor)



num_var <- freMPL %>% keep(is.numeric) %>% names() %>% setdiff(c("Exposure","ClaimAmount","ClaimInd"))

fac_var <- freMPL %>% keep(is.factor) %>% names()

freMPL <- freMPL %>% mutate_at(num_var, ~as.factor(ntile(.,10)))

vars <- c(num_var, fac_var)






for (j in seq_along(vars)){
    
    assign(paste0("table_",vars[j]),
           
           freMPL %>% group_by(get(vars[j])) %>% 
             summarize(numbers=sum(ClaimInd),amounts=sum(ClaimAmount),exposure=sum(Exposure)) %>% 
             mutate(LC=amounts/exposure, FREQ=numbers/exposure, SEV=ifelse(numbers==0,0,amounts/numbers)) %>% 
             rename(!!vars[j]:=paste0("get(vars[j])")) %>% 
             mutate_at(vars[j],as.factor)           
    )
}


LC <- with(freMPL, sum(ClaimAmount)/sum(Exposure))
FREQ <- with(freMPL, sum(ClaimInd)/sum(Exposure))
SEV <- with(freMPL, sum(ClaimAmount)/sum(ClaimInd))



# KPI ---------------------------------------------------------------------

names_list<-ls(pattern = "\\btable.")

for (j in seq_along(names_list)){
    
    k<-get(names_list[j])
      
    lc<-ggplot(data=k) +
      geom_point(aes(x=get(colnames(k)[1]), y=LC, group=1),stat='summary', fun.y=sum,size=2,col=hue_pal()(4)[2])+
      stat_summary(aes(x=get(colnames(k)[1]), y=LC, group=1),fun.y=sum, geom="line",size=2,col=hue_pal()(4)[2])+
      geom_text(aes(x=get(colnames(k)[1]), y=LC, group=1, label=paste0(round(LC,0))), size=4, nudge_y = .015 )+
      geom_bar(aes(x=get(colnames(k)[1]),y=exposure*( max(k$LC) / max(k$exposure) )),stat="identity",alpha=0.5,fill="slateblue2")+
      scale_y_continuous(sec.axis=sec_axis(~./(max(k$LC) / max(k$exposure)*1), name="Exposure",labels=comma),labels=comma,limits = c(0,2000))+
      geom_hline(yintercept = LC)+
      annotate(geom="text", x=2, y=1500, label=paste0("Overall LC=",round(LC,0)))+
      labs(y = "Loss Cost",x=names(k)[1])+
      ggtitle("Loss Cost")
    
    
    freq<-ggplot(data=k) +
      geom_point(aes(x=get(colnames(k)[1]), y=FREQ, group=1),stat='summary', fun.y=sum,size=2,col=hue_pal()(4)[3])+
      stat_summary(aes(x=get(colnames(k)[1]), y=FREQ, group=1),fun.y=sum, geom="line",size=2,col=hue_pal()(4)[3])+
      geom_text(aes(x=get(colnames(k)[1]), y=FREQ, group=1, label=paste0(round(FREQ*100,1),"%")), size=4, nudge_y = .0015)+
      geom_bar(aes(x=get(colnames(k)[1]),y=exposure*( max(k$FREQ) / max(k$exposure) )),stat="identity",alpha=0.5,fill="slateblue2")+
      scale_y_continuous(sec.axis=sec_axis(~./(max(k$FREQ) / max(k$exposure)), name="Exposure",labels=comma),labels=percent, limits=c(0,1))+
      geom_hline(yintercept = FREQ)+
      annotate(geom="text", x=2, y=.6, label=paste0("Overall Frequency=",round(FREQ*100,1),"%"))+
      labs(y = "Frequency",x=names(k)[1])+
      ggtitle("Frequency")
    
    
    sev<-ggplot(data=k)+
      geom_point(aes(x=get(colnames(k)[1]), y=SEV, group=1),stat='summary', fun.y=sum,size=2,col=hue_pal()(4)[4])+
      stat_summary(aes(x=get(colnames(k)[1]), y=SEV, group=1),fun.y=sum, geom="line",size=2,col=hue_pal()(4)[4])+
      geom_text(aes(x=get(colnames(k)[1]), y=SEV, group=1, label=round(SEV,0)), size=4, nudge_y = 100)+
      geom_bar(aes(x=get(colnames(k)[1]),y=numbers*( max(k$SEV) / max(k$numbers) )),stat="identity",alpha=0.5,fill="peachpuff3")+
      scale_y_continuous(sec.axis=sec_axis(~./(max(k$SEV) / max(k$numbers)), name="Claim Numbers",labels=comma),labels=comma, limits=c(0,4000))+
      geom_hline(yintercept = SEV)+
      annotate(geom="text", x=2, y=3500, label=paste0("Overall Severity=",round(SEV,0)))+
      labs(y = "Severity",x=names(k)[1])+
      ggtitle("Severity")
    
    ggsave(plot=grid.arrange(lc,freq,sev,ncol=2),paste0(names_list[j],".png"), width = 12, height = 6.75, dpi = 320, units = "in", device='png')
    
    
}  
