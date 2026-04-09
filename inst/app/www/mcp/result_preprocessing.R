calculate_diff<-function(res, dt, acq){
  
  metrics<-res[,2:16]
  meths<-res$mth
  
  diffs<-c()
  diff_type<-c()
  
  for (i in 1:length(meths)) {
    for (j in 1:length(meths)) {
      diffs<-rbind(diffs, as.numeric(metrics[j,])-as.numeric(metrics[i,]))
      diff_type<-c(diff_type,paste0(meths[j], '-', meths[i]))
    }
  }
  colnames(diffs)<-colnames(metrics)
  diffs<-as.data.frame(diffs)
  diffs$diff_type<-diff_type
  diffs$dt<-dt
  return(diffs)
}

read_res_all<-function(platforms, cbt, deas, acq, view_num, flag=0){
  root_folder<- 'E:/proteomics/manus4_1/'
  dataset_list_dda<-read.table('D:/data/benchmark/data/dataset_info/DDA_Frag.txt', header = T, sep = '\t')
  dataset_list_dia<-read.table('D:/data/benchmark/data/dataset_info/DIA_DIANN.txt', header = T, sep = '\t')
  
  didx_dda<-c(0, 1, 2, 3, 4, 5, 7, 9, 10, 11)+1
  didx_dia<-c(0, 1, 2, 3, 4, 5, 6)+1
  
  res_all<-c()
  diff_all<-c()
  #flag = 0
  
  for (plt in platforms){
    for (dea in deas){
      #res_more_2views_knn_comp_add
      postfix<-paste0('_', cbt)
      if (cbt=='min'){
        postfix<-''
      }
      folder<-paste0(root_folder, 'res_more_', as.character(view_num), 'views_knn_comp_add',
                     postfix, '/reproduce_0_05_mv/', 'temp', as.character(flag),
                     '_', dea, '_', cbt, '_', plt, '/')
      
      if(acq == 'DDA'){
        for (idx in didx_dda){
          dt<-dataset_list_dda$dataset[idx]
          res_file<-paste0(folder, dt, '__metrics_all.csv')
          res<-read.table(res_file, header = T, sep = ',')
          res$acq<-acq
          res$dt<-dt
          
          idx_rbf<-grep('rbf', res$test_type)
          res<-res[setdiff(c(1:length(res$test_type)), idx_rbf), ]
          
          idx_all<-which(res$test_type == 'all_MLE_all_rescore_add_dlfq_hurdle')
          res<-res[1:idx_all,]
          idx_hurdle<-which(res$test_type=='all_hurdle')
          meths<-paste0(c(rep('V', idx_hurdle-1)), as.character(c(1:(idx_hurdle-1))))
          meths<-c(meths, 'Hurdle')
          meths<-c(meths, c('MCCA_RI','MCCA_RA', 'MCCAE_RI', 'MCCAE_RA'))
          meths<-c(meths, c('MVI_RI','MVI_RA', 'MVIE_RI', 'MVIE_RA'))
          meths<-c(meths, c('MLE_RI','MLE_RA', 'MLEE_RI', 'MLEE_RA'))
          res$mth<-meths
          diffs <- calculate_diff(res, dt, acq)
          res_all<-rbind(res_all, res)
          diff_all<-rbind(diff_all, diffs)
        }
        
      }else if(acq == 'DIA'){
        for (idx in didx_dia){
          dt<-dataset_list_dia$dataset[idx]
          res_file<-paste0(folder, dt, '__metrics_all.csv')
          res<-read.table(res_file, header = T, sep = ',')
          res$acq<-acq
          res$dt<-dt
          idx_rbf<-grep('rbf', res$test_type)
          res<-res[setdiff(c(1:length(res$test_type)), idx_rbf), ]
          
          idx_all<-which(res$test_type == 'all_MLE_all_rescore_add_dlfq_hurdle')
          res<-res[1:idx_all,]
          idx_hurdle<-which(res$test_type=='all_hurdle')
          meths<-paste0(c(rep('V', idx_hurdle-1)), as.character(c(1:(idx_hurdle-1))))
          meths<-c(meths, 'Hurdle')
          meths<-c(meths, c('MCCA_RI','MCCA_RA', 'MCCAE_RI', 'MCCAE_RA'))
          meths<-c(meths, c('MVI_RI','MVI_RA', 'MVIE_RI', 'MVIE_RA'))
          meths<-c(meths, c('MLE_RI','MLE_RA', 'MLEE_RI', 'MLEE_RA'))
          res$mth<-meths
          diffs <- calculate_diff(res, dt, acq)
          res_all<-rbind(res_all, res)
          diff_all<-rbind(diff_all, diffs)
        }
      }
    flag = flag + 1  
    }
  }
 
  return (list(res_all=res_all, diff_all=diff_all))
}

plot_diffs<-function(compare_set, compare_metrics, diffs_all, view_num, acq, cbt){
  diff_comp<-c()
  #idx_col<-match(compare_metrics, colnames(diffs_all))
  statiss<-c()

  for (comp in compare_set) {
    idx<-which(diffs_all$diff_type==comp)
    diff_comp<-rbind(diff_comp, diffs_all[idx,])
    statis<-c()
    for (j in 1:(length(colnames(diffs_all))-1)) {
      statis<-c(statis, paste0(as.character(round(mean(as.numeric(diffs_all[idx,][,j])), 2)),
                '|',as.character(round(median(as.numeric(diffs_all[idx,][,j])), 2)), '|',
                as.character(round(sd(as.numeric(diffs_all[idx,][,j])), 4))))

    }
    statis<-c(statis, comp)
    statiss<-rbind(statiss,statis)
    
  }
  colnames(statiss)<-colnames(diffs_all)
  write.table(as.data.frame(statiss), paste0('E:/proteomics/manus4_1/', as.character(view_num), 'views', '_', acq, '_', cbt, '.csv'), sep = ',', col.names = T, row.names = F)
  diff_cbts<-c()
  for (metric in compare_metrics) {
    diff_cbt<-cbind(diff_comp$diff_type, 
                    diff_comp[,which(colnames(diff_comp)==metric)])
    diff_cbt<-cbind(diff_cbt, rep(metric, length(diff_comp$diff_type)))
    diff_cbts<-rbind(diff_cbts, diff_cbt)
  }
  colnames(diff_cbts)<-c('comparison', 'value', 'metric')
  diff_cbts<-as.data.frame(diff_cbts)
  diff_cbts$value<-as.numeric(diff_cbts$value)
  col=c("#b2df8a","#f078b4","#a45ee3",
        "#fd780f","#96ca1c","#13608a","#ccbcb4", 'black','gray','red',"#a6cee3","#33a02c","#fb9a99","#fdbf6f",
        "#15d08a","#1f11b4","#33a02c","#fb9a99","#c3642c","#f2ca01","#64df8a","#1f78b4")
  
  fig4<-ggboxplot(diff_cbts, "metric", "value",width = 0.7, size=1,
                  color = "comparison", fill = 'comparison', alpha=0.5, palette =col, #title = 'all_test',
                  add = "jitter", add.params = list(alpha=0.1,size=0.7))+
    labs(x = '', y = 'difference')+scale_x_discrete(limits=unique(diff_cbts$metric)) + scale_y_continuous(limits = c(-0.4, 0.5))+theme_classic() +
    #theme(axis.text.x = element_text(size = 10), axis.text.x=element_blank(),axis.text.y = element_text(size = 10))+
    theme(axis.title.y= element_text(size=12, face = 'bold'))+theme(axis.title.x = element_text(size = 10), axis.text.x = element_text(size = 10, face = 'bold'),axis.text.y = element_text(size = 10, face = 'bold'))+
    #theme(legend.title=element_text(size=10),legend.text=element_text(size=10))+
    theme(legend.position = "top")+stat_summary(fun=mean, geom="point", shape=20, aes(group=comparison), position=position_dodge(.8), size=2, color="red", fill="red")+
    theme(plot.title = element_text(size = 12,hjust = 0.5, face = "bold"))+geom_hline(yintercept=0,linewidth=0.5,linetype=2)+theme(text=element_text(family="Times New Roman", size=12))#+coord_flip()
  fig4
  return(fig4)
}

library(ggplot2)
library(reshape2)
library(ggpubr)
library(ggsci)
library("readxl")
library(ggalluvial)
library(cowplot)
library(ComplexHeatmap)
library(gridExtra)
#library(extrafont)
#font_import()
#loadfonts(device="win")
##############################################
view_num=2

platforms<-c('FragPipe', 'Maxquant') #'Spectronaut', 'DIANN'
cbt<-'min'#,'max', 'mean') #, 'hurdle'
deas<-c('limma', 'ROTS')

acq<-'DDA'

res_all_dda<-read_res_all(platforms, cbt, deas, acq, view_num)

platforms<-c('Spectronaut', 'DIANN') #'Spectronaut', 'DIANN'
#cbt<-'min'#,'max', 'mean') #, 'hurdle'
deas<-c('limma', 'ROTS')

acq<-'DIA'
res_all_dia<-read_res_all(platforms, cbt, deas, acq, view_num)

compare_set<-c('Hurdle-V1', 'Hurdle-V2', 'MCCA_RI-V1', 'MCCA_RI-V2', 
               'MCCA_RI-Hurdle', 'MCCA_RA-V1', 'MCCA_RA-V2', 'MCCA_RA-hurdle',
               'MCCAE_RI-V1', 'MCCAE_RI-V2', 
               'MCCAE_RI-Hurdle', 'MCCAE_RA-V1', 'MCCAE_RA-V2', 'MCCAE_RA-hurdle')

compare_metrics<-c('Prec', 'Rec', 'nMcc', 'geomean', 'pauc005')

compare_set<-c('MCCA_RI-V1', 'MCCA_RI-V2', 'MCCA_RI-Hurdle', 'MCCA_RA-V1', 
               'MCCA_RA-V2', 'MCCA_RA-Hurdle', 'MCCA_RI-MCCA_RA')

compare_set<-c('MCCAE_RI-V1', 'MCCAE_RI-V2', 'MCCAE_RI-Hurdle', 'MCCAE_RA-V1', 
               'MCCAE_RA-V2', 'MCCAE_RA-Hurdle', 'MCCAE_RI-MCCAE_RA')

fig_2iew_min_v1_v2_H_dda<-plot_diffs(compare_set, compare_metrics, res_all_dda$diff_all, view_num, 'DDA', cbt)
fig_2iew_min_v1_v2_H_dia<-plot_diffs(compare_set, compare_metrics, res_all_dia$diff_all, view_num, 'DIA', cbt)

view_num=3

platforms<-c('FragPipe', 'Maxquant') #'Spectronaut', 'DIANN'
#cbt<-'min'#,'max', 'mean') #, 'hurdle'
deas<-c('limma', 'ROTS')

acq<-'DDA'

res_all_dda<-read_res_all(platforms, cbt, deas, acq, view_num)

platforms<-c('Spectronaut', 'DIANN') #'Spectronaut', 'DIANN'
#cbt<-'min'#,'max', 'mean') #, 'hurdle'
deas<-c('limma', 'ROTS')

acq<-'DIA'
res_all_dia<-read_res_all(platforms, cbt, deas, acq, view_num)

compare_set<-c('Hurdle-V1', 'Hurdle-V2', 'MCCA_RI-V1', 'MCCA_RI-V2', 
               'MCCA_RI-Hurdle', 'MCCA_RA-V1', 'MCCA_RA-V2', 'MCCA_RA-hurdle',
               'MCCAE_RI-V1', 'MCCAE_RI-V2', 
               'MCCAE_RI-Hurdle', 'MCCAE_RA-V1', 'MCCAE_RA-V2', 'MCCAE_RA-hurdle')

compare_metrics<-c('Prec', 'Rec', 'nMcc', 'geomean', 'pauc005')

compare_set<-c('Hurdle-V1', 'Hurdle-V2', 'Hurdle-V3', 'V1-V2', 'V1-V3', 'V-V3')
compare_set<-c('MCCA_RI-V1', 'MCCA_RI-V2', 'MCCA_RI-V3', 'MCCA_RI-Hurdle', 'MCCA_RA-V1', 
               'MCCA_RA-V2', 'MCCA_RA-V3', 'MCCA_RA-Hurdle', 'MCCA_RI-MCCA_RA')
compare_set<-c('MCCAE_RI-V1', 'MCCAE_RI-V2', 'MCCAE_RI-V3', 'MCCAE_RI-Hurdle', 'MCCAE_RA-V1', 
               'MCCAE_RA-V2', 'MCCAE_RA-V3', 'MCCAE_RA-Hurdle', 'MCCAE_RI-MCCAE_RA')
fig_3iew_min_v1_v2_H_dda<-plot_diffs(compare_set, compare_metrics, res_all_dda$diff_all, view_num, 'DDA', cbt)
fig_3iew_min_v1_v2_H_dia<-plot_diffs(compare_set, compare_metrics, res_all_dia$diff_all, view_num, 'DIA', cbt)

grid.arrange(fig_2iew_min_v1_v2_H_dda, fig_2iew_min_v1_v2_H_dia,
             fig_3iew_min_v1_v2_H_dda, fig_3iew_min_v1_v2_H_dia, nrow=2)



########################
view_num=2
platforms<-c('FragPipe', 'Maxquant')
cbt<-'min'
deas<-c('limma', 'ROTS')
acq<-'DDA'

compare_set<-c('Hurdle-V1', 'Hurdle-V2', 'MCCA_RI-V1', 'MCCA_RI-V2', 
               'MCCA_RI-Hurdle', 'MCCA_RA-V1', 'MCCA_RA-V2', 'MCCA_RA-hurdle',
               'MCCAE_RI-V1', 'MCCAE_RI-V2', 
               'MCCAE_RI-Hurdle', 'MCCAE_RA-V1', 'MCCAE_RA-V2', 'MCCAE_RA-hurdle')

res_all_dda<-read_res_all(platforms, cbt, deas, acq, view_num)
fig_2iew_min_v1_v2_H_dda<-plot_diffs(compare_set, compare_metrics, res_all_dda$diff_all, view_num, acq, cbt)

platforms<-c('Spectronaut', 'DIANN')
acq<-'DIA'
res_all_dia<-read_res_all(platforms, cbt, deas, acq, view_num)
fig_2iew_min_v1_v2_H_dia<-plot_diffs(compare_set, compare_metrics, res_all_dia$diff_all, view_num, acq, cbt)

##########################

view_num=2
platforms<-c('FragPipe')#, 'Maxquant'
cbt<-'min'
deas<-c('limma', 'ROTS')
acq<-'DDA'

compare_set<-c('Hurdle-V1', 'Hurdle-V2', 'MCCA_RI-V1', 'MCCA_RI-V2', 
               'MCCA_RI-Hurdle', 'MCCA_RA-V1', 'MCCA_RA-V2', 'MCCA_RA-Hurdle',
               'MCCAE_RI-V1', 'MCCAE_RI-V2', 
               'MCCAE_RI-Hurdle', 'MCCAE_RA-V1', 'MCCAE_RA-V2', 'MCCAE_RA-Hurdle')
# compare_set<-c('MCCA_RI-V1', 'MCCA_RI-V2', 'MCCA_RI-Hurdle', 'MCCA_RA-V1', 
#                'MCCA_RA-V2', 'MCCA_RA-Hurdle', 'MCCA_RI-MCCA_RA')
res_all_dda<-read_res_all(platforms, cbt, deas, acq, view_num)
fig_2iew_min_v1_v2_H_dda_fg<-plot_diffs(compare_set, compare_metrics, res_all_dda$diff_all, view_num, 'Fg', cbt)

platforms<-c('Maxquant')#, 'Maxquant'
res_all_dda<-read_res_all(platforms, cbt, deas, acq, view_num, flag=2)
fig_2iew_min_v1_v2_H_dda_mq<-plot_diffs(compare_set, compare_metrics, res_all_dda$diff_all, view_num, 'mq', cbt)

view_num=3
platforms<-c('FragPipe')#, 'Maxquant'
cbt<-'min'
deas<-c('limma', 'ROTS')
acq<-'DDA'

compare_set<-c('Hurdle-V1', 'Hurdle-V2', 'Hurdle-V3', 'MCCA_RI-V1', 'MCCA_RI-V2', 'MCCA_RI-V3',
               'MCCA_RI-Hurdle', 'MCCA_RA-V1', 'MCCA_RA-V2', 'MCCA_RA-V3', 'MCCA_RA-Hurdle',
               'MCCAE_RI-V1', 'MCCAE_RI-V2', 'MCCAE_RI-V3',
               'MCCAE_RI-Hurdle', 'MCCAE_RA-V1', 'MCCAE_RA-V2', 'MCCAE_RA-V3', 'MCCAE_RA-Hurdle', 
               'MCCAE_RA-MCCAE_RI', 'MCCAE_RA-MCCA_RI', 'MCCAE_RA-MCCA_RA')
# compare_set<-c('MCCA_RI-V1', 'MCCA_RI-V2', 'MCCA_RI-Hurdle', 'MCCA_RA-V1', 
#                'MCCA_RA-V2', 'MCCA_RA-Hurdle', 'MCCA_RI-MCCA_RA')
# compare_set<-c('MCCA_RI-V1', 'MCCA_RI-V2', 'MCCA_RI-V3', 'MCCA_RI-Hurdle', 'MCCA_RA-V1', 
#                'MCCA_RA-V2', 'MCCA_RA-V3', 'MCCA_RA-Hurdle', 'MCCA_RI-MCCA_RA')
res_all_dda<-read_res_all(platforms, cbt, deas, acq, view_num)
fig_3iew_min_v1_v2_H_dda_fg<-plot_diffs(compare_set, compare_metrics, res_all_dda$diff_all, view_num, 'Fg', cbt)

platforms<-c('Maxquant')#, 'Maxquant'
res_all_dda<-read_res_all(platforms, cbt, deas, acq, view_num, flag=2)
fig_3iew_min_v1_v2_H_dda_mq<-plot_diffs(compare_set, compare_metrics, res_all_dda$diff_all, view_num, 'mq', cbt)

grid.arrange(fig_2iew_min_v1_v2_H_dda_fg, fig_2iew_min_v1_v2_H_dda_mq,
             fig_3iew_min_v1_v2_H_dda_fg, fig_3iew_min_v1_v2_H_dda_mq, nrow=2)

###############

view_num=2
platforms<-c('DIANN')#, 'Maxquant'
cbt<-'min'
deas<-c('limma', 'ROTS')
acq<-'DIA'

compare_set<-c('Hurdle-V1', 'Hurdle-V2', 'MCCA_RI-V1', 'MCCA_RI-V2', 
               'MCCA_RI-Hurdle', 'MCCA_RA-V1', 'MCCA_RA-V2', 'MCCA_RA-Hurdle',
               'MCCAE_RI-V1', 'MCCAE_RI-V2', 
               'MCCAE_RI-Hurdle', 'MCCAE_RA-V1', 'MCCAE_RA-V2', 'MCCAE_RA-Hurdle')
# compare_set<-c('MCCA_RI-V1', 'MCCA_RI-V2', 'MCCA_RI-Hurdle', 'MCCA_RA-V1', 
#                'MCCA_RA-V2', 'MCCA_RA-Hurdle', 'MCCA_RI-MCCA_RA')
res_all_dia<-read_res_all(platforms, cbt, deas, acq, view_num, flag=2)
fig_2iew_min_v1_v2_H_dia_diann<-plot_diffs(compare_set, compare_metrics, res_all_dia$diff_all, view_num, 'diann', cbt)

platforms<-c('Spectronaut')#, 'Maxquant'
res_all_dia<-read_res_all(platforms, cbt, deas, acq, view_num, flag=0)
fig_2iew_min_v1_v2_H_dia_spt<-plot_diffs(compare_set, compare_metrics, res_all_dia$diff_all, view_num, 'spt', cbt)

view_num=3
platforms<-c('DIANN')#, 'Maxquant'
cbt<-'min'
deas<-c('limma', 'ROTS')
acq<-'DIA'

compare_set<-c('Hurdle-V1', 'Hurdle-V2', 'Hurdle-V3', 'MCCA_RI-V1', 'MCCA_RI-V2', 'MCCA_RI-V3',
               'MCCA_RI-Hurdle', 'MCCA_RA-V1', 'MCCA_RA-V2', 'MCCA_RA-V3', 'MCCA_RA-Hurdle',
               'MCCAE_RI-V1', 'MCCAE_RI-V2', 'MCCAE_RI-V3',
               'MCCAE_RI-Hurdle', 'MCCAE_RA-V1', 'MCCAE_RA-V2', 'MCCAE_RA-V3', 'MCCAE_RA-Hurdle', 
               'MCCAE_RA-MCCAE_RI', 'MCCAE_RA-MCCA_RI', 'MCCAE_RA-MCCA_RA')
# compare_set<-c('MCCA_RI-V1', 'MCCA_RI-V2', 'MCCA_RI-Hurdle', 'MCCA_RA-V1', 
#                'MCCA_RA-V2', 'MCCA_RA-Hurdle', 'MCCA_RI-MCCA_RA')
# compare_set<-c('MCCA_RI-V1', 'MCCA_RI-V2', 'MCCA_RI-V3', 'MCCA_RI-Hurdle', 'MCCA_RA-V1', 
#                'MCCA_RA-V2', 'MCCA_RA-V3', 'MCCA_RA-Hurdle', 'MCCA_RI-MCCA_RA')
res_all_dia<-read_res_all(platforms, cbt, deas, acq, view_num, flag=2)
fig_3iew_min_v1_v2_H_dia_diann<-plot_diffs(compare_set, compare_metrics, res_all_dia$diff_all, view_num, 'diann', cbt)

platforms<-c('Spectronaut')#, 'Maxquant'
res_all_dia<-read_res_all(platforms, cbt, deas, acq, view_num, flag=0)
fig_3iew_min_v1_v2_H_dia_spt<-plot_diffs(compare_set, compare_metrics, res_all_dia$diff_all, view_num, 'spt', cbt)

grid.arrange(fig_2iew_min_v1_v2_H_dia_diann, fig_2iew_min_v1_v2_H_dia_spt,
             fig_3iew_min_v1_v2_H_dia_diann, fig_3iew_min_v1_v2_H_dia_spt, nrow=2)
