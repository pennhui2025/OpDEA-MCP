library(ggplot2)
library(ggrepel)
library(reshape2)
library(ggpubr)
library(UpSetR)
library(umap)


gen_fig_sv<-function(melt_dt, flag, temp_fold, qval, logFC_t){
  p<-ggplot(melt_dt, aes(x=logFC, y=log10q,color=stable)) +
    geom_point(alpha=0.5, size=2) +
    theme_bw(base_size = 12) +
    xlab("log2(fold change)") +
    ylab("-Log10(adj.pvalue)") +
    scale_colour_manual(values = c("purple",'gray')) +
    geom_hline(yintercept = -log10(qval), lty = 4) +
    geom_vline(xintercept = c(-logFC_t, logFC_t), lty = 4)+
    labs(title = flag)+
    geom_label_repel(data = melt_dt, aes(label = label),
                     size = 3,box.padding = unit(0.5, "lines"),
                     point.padding = unit(0.8, "lines"),
                     segment.color = "black",
                     show.legend = FALSE, max.overlaps = 10)+
    theme(legend.title=element_text(size=12),legend.text=element_text(size=12), legend.position = "top")+
    guides(color=guide_legend(nrow=3), shape=guide_legend(nrow=3))
  p
  ggsave(paste0(temp_fold, "volcano", flag,".pdf"), width = 10, height = 10, units = "cm")
  return(p)
}

fig_mcp<-function(melt_dt, flag, temp_fold, score_threshold, logFC_t){
  p<-ggplot(melt_dt, aes(x=logFC, y=score,color=stable)) +
    geom_point(alpha=0.5, size=2) +
    theme_bw(base_size = 12) +
    xlab("log2(fold change)") +
    ylab("score") +
    scale_colour_manual(values = c("purple",'gray')) +
    geom_hline(yintercept = score_threshold, lty = 4) +
    geom_vline(xintercept = c(-logFC_t, logFC_t), lty = 4)+
    labs(title = flag)+
    geom_label_repel(data = melt_dt, aes(label = label),
                     size = 3,box.padding = unit(0.5, "lines"),
                     point.padding = unit(0.8, "lines"),
                     segment.color = "black",
                     show.legend = FALSE, max.overlaps = 10)+
    theme(legend.title=element_text(size=12),legend.text=element_text(size=12), legend.position = "top")+
    guides(color=guide_legend(nrow=3), shape=guide_legend(nrow=3))
  p
  ggsave(paste0(temp_fold, "volcano", flag,".pdf"), width = 10, height = 10, units = "cm")
  return(p)
}

gen_figs_mvidia<-function(dea_res, logFC_t, qval, temp_fold, views){

  logFC_t<-as.numeric(logFC_t)
  qval<-as.numeric(qval)
  res<-as.data.frame(dea_res)

  melts<-list()
  ps<-list()
  DEPs<-list()
  for (v in 1:length(views)) {
    if(views[v]=='MVIDIA'){
      mth = 'GCCA'
    }else{
      mth = views[v]
    }
    idx_logfc = which(colnames(dea_res)==paste0('logFC_union',mth))
    idx_score = which(colnames(dea_res)==paste0('score_union',mth))
    melts[[v]] = data.frame(protein=dea_res$Protein, score=as.numeric(dea_res[,idx_score]),
                            logFC=as.numeric(dea_res[,idx_logfc]))
    melts[[v]]$qval = 1-melts[[v]]$score
    melts[[v]]$log10q = -log10(melts[[v]]$qval)
    melts[[v]]$stable<-as.factor(ifelse((melts[[v]]$qval <= qval & abs(melts[[v]]$logFC) >= logFC_t),
                                        'DEP',
                                        'non-DEP'))
    melts[[v]]$label = melts[[v]]$protein
    melts[[v]]$label[which(melts[[v]]$stable=='non-DEP')]=''
    ps[[v]]<-gen_fig_sv(melts[[v]], views[v], temp_fold, qval, logFC_t)
    DEPs<-append(DEPs, list(c(melts[[v]]$protein[which(melts[[v]]$stable=='DEP')])))
  }
  print(names(DEPs))
  names(DEPs)<-views
  color_set<-c('gold',"#b2df8a","#33a02c","#1f78b4",
               "#CD6155",
               '#FF1493',
               '#40E0D0'
  )
  color_use<-color_set[1:length(views)]

  library(UpSetR)
  upset_plot<-UpSetR::upset(fromList(DEPs),
                            sets = views,
                            order.by="freq", matrix.color="black", point.size=3,
                            sets.bar.color=color_use,mb.ratio=c(0.4, 0.6), text.scale = 1.5,
                            number.angles=0,
                            keep.order = T, nintersects=20)
  pdf(paste0(temp_fold, "upset_plot",".pdf"))
  upset_plot
  dev.off()
  #ggsave(paste0(temp_fold, "upset_plot",".pdf"), plot = upset_plot, width = 10, height = 10, units = "cm")
  #zip(zipfile = paste0(temp_fold, "MVIDIA_DEA_res.zip"), files = temp_fold, flags = '-r9Xj')
  return(list(p_mv=ps[length(ps)], p_us=upset_plot, zipfile=temp_fold))
}

gen_figs_mcp<-function(dea_res, logFC_t, qval, temp_fold, views){

  logFC_t<-as.numeric(logFC_t)
  qval<-as.numeric(qval)
  res<-as.data.frame(dea_res)


  melts<-list()
  ps<-list()
  DEPs<-list()
  for (v in 1:length(views)) {
    if(views[v]=='MCP'){
      mth = 'MCP'
      new_names = gsub('MVcontrast', 'MCP',colnames(res))
      idx_logfc = which(new_names==paste0(mth, '_logFC.2'))
      idx_score = which(new_names==paste0(mth, '_score.2'))
      idx_threshold = which(new_names==paste0(mth, '_threshold.2'))
      score_threshold = max(dea_res[,idx_threshold])
      melts[[v]] = data.frame(protein=dea_res$Protein, score=as.numeric(dea_res[,idx_score]),
                                logFC=as.numeric(dea_res[,idx_logfc]))

      melts[[v]]$score = melts[[v]]$score
      melts[[v]]$stable<-as.factor(ifelse((melts[[v]]$score >= score_threshold & abs(melts[[v]]$logFC) >= logFC_t),
                                          'DEP',
                                          'non-DEP'))
      melts[[v]]$label = melts[[v]]$protein
      melts[[v]]$label[which(melts[[v]]$stable=='non-DEP')]=''
      ps[[v]]<-fig_mcp(melts[[v]], views[v], temp_fold, score_threshold, logFC_t)
    }else{
      mth = views[v]
      idx_logfc = which(colnames(dea_res)==paste0(mth, '_logFC'))
      idx_score = which(colnames(dea_res)==paste0(mth, '_score'))
      melts[[v]] = data.frame(protein=dea_res$Protein, score=as.numeric(dea_res[,idx_score]),
                              logFC=as.numeric(dea_res[,idx_logfc]))
      melts[[v]]$qval = 1-melts[[v]]$score
      melts[[v]]$log10q = -log10(melts[[v]]$qval)
      melts[[v]]$stable<-as.factor(ifelse((melts[[v]]$qval <= qval & abs(melts[[v]]$logFC) >= logFC_t),
                                          'DEP',
                                          'non-DEP'))
      melts[[v]]$label = melts[[v]]$protein
      melts[[v]]$label[which(melts[[v]]$stable=='non-DEP')]=''
      ps[[v]]<-gen_fig_sv(melts[[v]], views[v], temp_fold, qval, logFC_t)
    }

    DEPs<-append(DEPs, list(c(melts[[v]]$protein[which(melts[[v]]$stable=='DEP')])))
  }
  print(names(DEPs))
  names(DEPs)<-views
  color_set<-c('gold',"#b2df8a","#33a02c","#1f78b4",
               "#CD6155",
               '#FF1493',
               '#40E0D0'
  )
  color_use<-color_set[1:length(views)]


  upset_plot<-UpSetR::upset(fromList(DEPs),
                sets = views,
                order.by="freq", matrix.color="black", point.size=3,
                sets.bar.color=color_use,mb.ratio=c(0.4, 0.6), text.scale = 1.5,
                number.angles=0,
                keep.order = T, nintersects=20)
  pdf(paste0(temp_fold, "upset_plot",".pdf"))
  upset_plot
  dev.off()
  #ggsave(paste0(temp_fold, "upset_plot",".pdf"), plot = upset_plot, width = 10, height = 10, units = "cm")
  #zip(zipfile = paste0(temp_fold, "MVIDIA_DEA_res.zip"), files = temp_fold, flags = '-r9Xj')
  return(list(p_mv=ps[length(ps)], p_us=upset_plot, zipfile=temp_fold))
}

MVIDIA_DEA<-function(view_files, logFC_mv, qval_mv, norm, imput,
                     platform, acq, view_names, dea, g1, g2,
                     python_path_mv_dea, method){

  #python_path_mv_dea = 'D:/anaconda3/envs/directlfq/python'
  #dlfq_path = system.file("app", "www", "run_dlfq.py", package = "OpDEA")
  #browser()
  python_path_mv_dea = system.file("app", "www", "directlfq/python.exe", package = "OpDEA")
  rsc = strsplit(python_path_mv_dea, '/', fixed = T)
  idx = which(rsc[[1]]=='library')
  rsc_root = paste(rsc[[1]][1:(idx-1)],collapse = '/')
  Rscript = paste0(rsc_root, '/bin/')
  #Rscript = 'D:/R-4.3.1/bin/'
  #R_code = './R/MVIDIA/'
  #R_code = 'E:/proteomics/maus2/submission/submission/final/upload/OpDEA/OpDEA-main/inst/app/www/mvidia/'
  R_code = gsub('MVIDIA.py', '', system.file("app", "www", "mvidia/MVIDIA.py", package = "OpDEA"))
  zip_fold = view_files
  #unzip(view_files$datapath, exdir = zip_fold)
  views = strsplit(view_names, ',', fixed = T)[[1]]
  view_file_str = paste(paste0(view_files, views,'.tsv'), collapse = ',')
  design_str = paste0(view_files, 'design.tsv')
  #browser()
  system(paste0(python_path_mv_dea, ' ', R_code,'MVIDIA_DEA.py ',
                '-I views -q ', as.character(qval_mv),
                ' -l ', as.character(logFC_mv),
                ' -p ', platform,
                ' -v ', view_names,
                ' -n ', norm,
                ' -i ', imput,
                ' -D ',  dea,
                ' -g1 ', g1,
                ' -g2 ', g2,
                ' -R ', Rscript,
                ' -r ', R_code,
                ' -A ', acq,
                ' -m ', method,
                ' -py ', python_path_mv_dea,
                ' -f ', view_file_str,
                ' -d ', design_str,
                ' -s ', zip_fold))

  dea_res = read.table(paste0(view_files, '/res/test/test_union_res_all.csv'), sep = ',', header = T)
  temp_fold=paste0(view_files, '/res/test/')
  if(!dir.exists(temp_fold)){
    dir.create(temp_fold, recursive = TRUE)
  }
  views<-append(views, method)
  out<-gen_figs_mvidia(dea_res, logFC_mv, qval_mv, temp_fold, views)

  return(out)
}

MCP_DEA<-function(view_files, logFC_mv, qval_mv, norm, imput,
                     platform, acq, logTs, dea, g1, g2,
                     python_path_mv_dea, trial, save_fold, view_select){

  # python_path_mv_dea = 'E:/proteomics/mauns4_3/Submission/tool/offline/OpDEA/R/library/OpDEA/app/www/directlfq/python.exe'
  #dlfq_path = system.file("app", "www", "run_dlfq.py", package = "OpDEA")
  #browser()
  python_path_mv_dea = system.file("app", "www", "directlfq/python.exe", package = "OpDEA")
  rsc = strsplit(python_path_mv_dea, '/', fixed = T)
  idx = which(rsc[[1]]=='library')
  rsc_root = paste(rsc[[1]][1:(idx-1)],collapse = '/')
  Rscript = paste0(rsc_root, '/bin/')
  #Rscript = 'D:/R-4.3.1/bin/'
  #R_code = './R/MVIDIA/'
  #R_code = 'E:/proteomics/maus2/submission/submission/final/upload/OpDEA/OpDEA-main/inst/app/www/mvidia/'
  R_code = gsub('MCP.py', '', system.file("app", "www", "mcp/MCP.py", package = "OpDEA"))
  #unzip(view_files$datapath, exdir = zip_fold)
  #views = strsplit(view_names, ',', fixed = T)[[1]]
  #view_file_str = paste(paste0(view_files, views,'.tsv'), collapse = ',')
  #design_str = paste0(view_files, 'design.tsv')
  #browser()
  system(paste0(python_path_mv_dea, ' ', R_code,'MCP.py ',
                '-f ', view_files, ' -q ', as.character(qval_mv),
                ' -l ', as.character(logFC_mv),
                ' -p ', platform,
                ' -o ', logTs,
                ' -n ', norm,
                ' -i ', imput,
                ' -D ',  dea,
                ' -g1 ', g1,
                ' -g2 ', g2,
                ' -R ', Rscript,
                ' -A ', acq,
                ' -t ', as.character(trial),
                ' -S ', view_select,
                ' -s ', save_fold))
  #browser()
  dea_res_path = read.table(paste0(save_fold, 'file_path.txt'), sep = '|', header = F)
  dea_res = read.table(as.character(dea_res_path[2]), sep = ',', header = T)
  temp_fold=paste0(view_files, '/res/test/')
  if(!dir.exists(temp_fold)){
    dir.create(temp_fold, recursive = TRUE)
  }
  views = strsplit(as.character(dea_res_path[1]), ',', fixed = T)[[1]]
  views<-append(views, 'MCP')
  out<-gen_figs_mcp(dea_res, logFC_mv, qval_mv, temp_fold, views)

  return(out)
}

MVIDIA_class<-function(view_files_train, view_files_test,
                       platform, acq, view_names, pos, pro_list, #mr,
                       python_path_mv_dea, method){
  if(length(grep('.csv', pro_list))>0){
    pro_list=paste(read.table(pro_list, sep = ',', header = F)[,1], collapse = ',')
  }
  #python_path_mv_dea = 'D:/anaconda3/envs/directlfq/python'
  python_path_mv_dea = system.file("app", "www", "directlfq/python.exe", package = "OpDEA")
  rsc = strsplit(python_path_mv_dea, '/', fixed = T)
  idx = which(rsc[[1]]=='library')
  rsc_root = paste(rsc[[1]][1:(idx-1)],collapse = '/')
  Rscript = paste0(rsc_root, '/bin/')
  #Rscript = 'D:/R-4.3.1/bin/'
  #R_code = './R/MVIDIA/'
  #R_code = 'E:/proteomics/maus2/submission/submission/final/upload/OpDEA/OpDEA-main/inst/app/www/mvidia/'
  R_code = gsub('MVIDIA.py', '', system.file("app", "www", "mvidia/MVIDIA.py", package = "OpDEA"))
  zip_fold_train = view_files_train
  zip_fold_test = view_files_test
  #unzip(view_files_train$datapath, exdir = zip_fold_train)
  #unzip(view_files_test$datapath, exdir = zip_fold_test)
  views = strsplit(view_names, ',', fixed = T)[[1]]
  view_file_str_tr = paste(paste0(zip_fold_train, views,'.tsv'), collapse = ',')
  view_file_str_te = paste(paste0(zip_fold_test, views,'.tsv'), collapse = ',')
  design_str_tr = paste0(zip_fold_train, 'design.tsv')
  design_str_te = paste0(zip_fold_test, 'design.tsv')
  temp_fold=paste0(zip_fold_test, 'res/')
  if(!dir.exists(temp_fold)){
    dir.create(temp_fold, recursive = TRUE)
  }
  #browser()
  system(paste0(python_path_mv_dea, ' ', R_code,'MVIDIA.py ',
                '-T ', 'classification',
                ' -Itr views -Ite views', #as.character(mr),
                ' -p ', platform,
                ' -v ', view_names,
                ' -R ', Rscript,
                ' -r ', R_code,
                ' -A ', acq,
                ' -mth ', method,
                ' -fs ', pro_list,
                ' -pos ', pos,
                ' -s ', temp_fold,
                ' -py ', python_path_mv_dea,
                ' -ftr ', view_file_str_tr,
                ' -fte ', view_file_str_te,
                ' -dtr ', design_str_tr,
                ' -dte ', design_str_te))

  #browser()
  #zip(zipfile = paste0(temp_fold, "MVIDIA_patient_diagnosis.zip"), files = temp_fold, flags = '-r9Xj')
  return(temp_fold)
}




plot_umap<-function(UP, species, view){
  color_set<-c('gold',"#b2df8a","#33a02c","#1f78b4",
               "#CD6155",
               '#FF1493',
               '#40E0D0',"#15d08a","#f3fa11",
               "#a45ee3")
  if(length(unique(species))>length(color_set)){
    color_set<-c(1:length(unique(species)))
  }else{
    color_set<-color_set[1:length(unique(species))]
  }
  dt = data.frame(UMAP1=UP$layout[,1], UMAP2=UP$layout[,2],
                  Cell=species)
  pup<-ggplot(dt, aes(UMAP1, UMAP2, color=Cell))+
    geom_point(alpha=0.5, size=2) +
    theme_bw(base_size = 12) +
    xlab("UMAP1") +  ylab("UMAP2") +labs(title = view)+
    scale_colour_manual(values = color_set)+
    theme(legend.position = 'right', legend.key.size = unit(1, 'cm'))

  return(list(p=pup, dt=dt))
}

MVIDIA_clustering<-function(view_files, platform, acq, view_names,
                            nclus, mr, ld, c,
                            python_path_mv_dea){
  #python_path_mv_dea = 'D:/anaconda3/envs/directlfq/python'
  python_path_mv_dea = system.file("app", "www", "directlfq/python.exe", package = "OpDEA")
  rsc = strsplit(python_path_mv_dea, '/', fixed = T)
  idx = which(rsc[[1]]=='library')
  rsc_root = paste(rsc[[1]][1:(idx-1)],collapse = '/')
  Rscript = paste0(rsc_root, '/bin/')
  #Rscript = 'D:/R-4.3.1/bin/'
  #R_code = './R/MVIDIA/'
  #R_code = 'E:/proteomics/maus2/submission/submission/final/upload/OpDEA/OpDEA-main/inst/app/www/mvidia/'
  R_code = gsub('MVIDIA.py', '', system.file("app", "www", "mvidia/MVIDIA.py", package = "OpDEA"))
  #browser()
  zip_fold = view_files
  #unzip(view_files$datapath, exdir = zip_fold)
  views = strsplit(view_names, ',', fixed = T)[[1]]
  view_file_str = paste(paste0(zip_fold, views,'.tsv'), collapse = ',')
  design_str = paste0(zip_fold, 'design.tsv')
  #browser()
  #temp_fold = paste0(zip_fold, 'res/')
  if (c==''){
    c="blank"
  }
  system(paste0(python_path_mv_dea, ' ', R_code,'MVIDIA.py ',
                '-T ', 'clustering',
                ' -Itr views -m ', as.character(mr),
                ' -p ', platform,
                ' -v ', view_names,
                ' -R ', Rscript,
                ' -r ', R_code,
                ' -A ', acq,
                ' -d ', ld,
                ' -c ', c,
                ' -s ', zip_fold,
                ' -ncluster ', nclus,
                ' -py ', python_path_mv_dea,
                ' -ftr ', view_file_str,
                ' -dtr ', design_str))

  ps = list()
  save_fold=paste0(zip_fold, 'res/')
  if(!dir.exists(save_fold)){
    dir.create(save_fold, recursive = TRUE)
  }
  for (v in 1:(length(views)+1)) {
    if (v<=length(views)){
      fig_name = paste0("umap_", views[v])
      view_name = views[v]
    df<-read.table(paste0(save_fold, 'v', as.character(v), '_scale.csv'),
                   sep = ',', header = T)

    }else if(v==(length(views)+1)){
      fig_name = 'umap_MVIDIA'
      view_name = 'MVIDIA'
      df<-read.table(paste0(save_fold, 'mvidia_latent.csv'),
                     sep = ',', header = T)
    }
    true_lab<-df[,1]
    pred_lab<-df[,2]
    if (length(unique(true_lab))==1){
      species<-pred_lab
    }else{
      species<-true_lab
    }
    library(umap)
    umap_v<-umap::umap(as.matrix(df[,3:length(df[1,])]), random_state=2024)
    p_v<-plot_umap(umap_v, species, fig_name)
    ps[[v]]<-p_v$p
    ggsave(paste0(save_fold, fig_name,".pdf"), width = 10, height = 10, units = "cm")
  }
  #browser()

  pall<-ggarrange(plotlist=ps, nrow = ceiling(length(ps)/3), ncol=3)
  ggsave(paste0(save_fold, "all_umaps",".pdf"), width = 10, height = 10, units = "cm")
  #zip(zipfile = paste0(save_fold, "MVIDIA_clustering.zip"), files = save_fold, flags = '-r9Xj')
  return(list(pall=pall, zipfile=save_fold))
}

