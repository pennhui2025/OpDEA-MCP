source('E:/proteomics/manus4_1/codes/run_DEA_single.R')

args <- commandArgs(trailingOnly = TRUE)
platform = args[1]
python_path = args[2]
raw = args [3]
evid = args[4]
design = args[5]
view_names = args[6]

views = strsplit(view_names, ',', fixed = T)[[1]]

run_dlfq<-function(raw, evid, platform, python_path){
  #root_fold<-''#'inst/app/www/'
  #dlfq_path = system.file("app", "www", "run_dlfq.py", package = "OpDEA")
  dlfq_path = 'E:/proteomics/manus4_1/codes/run_dlfq.py'
  #python_path = system.file("app", "www", "directlfq/python.exe", package = "OpDEA")
  #system(paste0('./directlfq/python', ' ', root_fold,'www/run_dlfq.py ', platform, ' ', raw, ' ', evid))
  #system(paste0('./directlfq/python', ' ', dlfq_path, ' ', platform, ' ', raw, ' ', evid))
  #system(paste0(python_path, ' ','./R/run_dlfq.py ', platform, ' ', raw, ' ', evid))
  #system(paste0(python_path, ' ',dlfq_path, ' ', platform, ' ', raw, ' ', evid))
  print(paste0(python_path, ' ', dlfq_path, ' ', platform, ' ', raw, ' ', evid))
  system(paste0(python_path, ' ', dlfq_path, ' ', platform, ' ', raw, ' ', evid))
  return(paste0(evid, '.protein_intensities.tsv'))
}

get_organism_mq<-function(protein_ids){
  Organs<-c()
  for (i in 1:length(protein_ids)) {
    if(protein_ids[i]==''){
      maj_organ='NULL'
    }else{
      proteins<-strsplit(protein_ids[i], ';',fixed = T)[[1]]
      orgs=c()
      for (pro in proteins) {
        
        strs<-strsplit(pro,'_', fixed = T)[[1]]
        Organ<-strs[length(strs)]
        if(strs[1]=='rev' | strs[1]=='REV'){
          Organ='decoy'
        }
        if(strs[1]=='con' | strs[1]=='CON'){
          Organ='contam'
        }
        
        orgs=c(orgs, Organ)
      }
      uni_og = unique(orgs)
      if(length(uni_og)>1){
        maj_organ = 'mixed'
      }else{
        maj_organ = uni_og
      }
    }
    Organs<-c(Organs,maj_organ)
  }
  return(Organs)
}

get_organism_frag<-function(protein_ids, indist_pro){
  Organs<-c()
  for (i in 1:length(protein_ids)) {
    if(length(indist_pro[i])>0){
      proteins<-c(protein_ids[i], strsplit(indist_pro[i], ', ',fixed = T)[[1]])
    }else{
      proteins<-c(protein_ids[i])
    }
    orgs=c()
    for (pro in proteins) {
      strs<-strsplit(pro,'_', fixed = T)[[1]]
      Organ<-strs[length(strs)]
      if(strs[1]=='rev' | strs[1]=='REV'){
        Organ='decoy'
      }
      if(strs[1]=='con' | strs[1]=='CON'){
        Organ='contam'
      }
      orgs=c(orgs, Organ)
    }
    uni_og = unique(orgs)
    if(length(uni_og)>1){
      maj_organ = 'mixed'
    }else{
      maj_organ = uni_og
    }
    Organs<-c(Organs,maj_organ)
  }
  return(Organs)
}

get_pg_frag<-function(protein_ids, indist_pro){
  pg<-c()
  for (i in 1:length(protein_ids)) {
    if(length(indist_pro[i])>0){
      proteins<-paste0(protein_ids[i], ';', gsub(', ', ';', indist_pro[i]))
    }else{
      proteins<-c(protein_ids[i])
    }
    pg<-c(pg, proteins)
  }
  return(pg)
}

get_organism_dia<-function(protein_group, protein_name){
  proteins<-c()
  for (i in 1:length(protein_group)) {
    pro = strsplit(protein_group[i], ';', fixed=T)[[1]]
    pro_name = strsplit(protein_name[i], ';', fixed=T)[[1]]
    pro_sp<-''
    for (j in 1:length(pro)) {
      if(j==1){
        if(length(grep('_UPS', pro))==1){
          pro_sp=paste0('sp|',pro[j])
        }else{
          pro_sp=paste0('sp|',pro[j],'|',pro_name[j])}
      }else{
        pro_spj=paste0('sp|',pro[j],'|',pro_name[j])
        pro_sp<-paste0(pro_sp, ';', pro_spj)
      }
      
    }
    proteins<-c(proteins, pro_sp)
  }
  
  organisms<-get_organism_mq(proteins)
  return(list(sp=proteins, orga=organisms))
}

get_expression_matrix_fg_dda<-function(raw, evid, design, exps, python_path){
  platform = 'FragPipe'
  out_exp = ''
  root_tmp = strsplit(evid,'/', fixed = T)[[1]]
  temp_res_dir = gsub(root_tmp[length(root_tmp)], 'res/', evid)
  dir.create(temp_res_dir)
  if(length(grep('dlfq', exps))==1){
    dlfq_path<-run_dlfq(raw, evid, platform, python_path)
    dlfq_table<-read.table(dlfq_path, header = T, sep = '\t', quote = "")
    idx<-setdiff(c(1:length(colnames(dlfq_table))),
                 c(grep('protein',colnames(dlfq_table)),grep('Protein',colnames(dlfq_table))))
    
    organims_all<-get_organism_mq(dlfq_table$protein)
    out_dlfq<-cbind(dlfq_table$protein, organims_all,
                    dlfq_table[, idx])
    colnames(out_dlfq)[1:2]<-c('Protein','Organism')
    out_exp = out_dlfq
    write.table(out_exp, paste0(temp_res_dir, 'dlfq.tsv'), sep = '\t', col.names = T, row.names = F)
    print('matrix dlfq extracted successfully')
  }
  
  if (length(intersect(c('top0', 'maxlfq', 'count'), exps))>=1){
    protein_table<-read.table(raw, header = T, sep = '\t', quote = "")
    inten_idx<-grep('Intensity',colnames(protein_table))
    lfq_idx<-grep('MaxLFQ.Intensity',colnames(protein_table))
    frag_inten_idx<-setdiff(inten_idx, lfq_idx)
    count_idx_a<-grep('.Spectral.Count',colnames(protein_table))
    uni_c_idx<-grep('.Unique.Spectral.Count',colnames(protein_table))
    total_c_idx<-grep('.Total.Spectral.Count',colnames(protein_table))
    cbt_c_idx<-grep('Combined.Spectral.Count',colnames(protein_table))
    count_idx<-setdiff(count_idx_a, c(uni_c_idx, total_c_idx, cbt_c_idx))
    
    organims_all<-get_organism_frag(protein_table$Protein,
                                    protein_table$Indistinguishable.Proteins)
    pg_frag<-get_pg_frag(protein_table$Protein,
                         protein_table$Indistinguishable.Proteins)
    
    out_count_frag<-cbind(pg_frag, organims_all,
                          protein_table[,count_idx])
    out_iten_frag<-cbind(pg_frag, organims_all,
                         protein_table[,frag_inten_idx])
    out_inten_maxlfq<-cbind(pg_frag, organims_all,
                            protein_table[,lfq_idx])
    colnames(out_count_frag)[1:2]<-c('Protein','Organism')
    colnames(out_iten_frag)[1:2]<-c('Protein','Organism')
    colnames(out_inten_maxlfq)[1:2]<-c('Protein','Organism')
    
  }
  write.table(out_count_frag, paste0(temp_res_dir, 'count.tsv'), sep = '\t', col.names = T, row.names = F)
  print('matrix count extracted successfully')
  write.table(out_iten_frag, paste0(temp_res_dir, 'top0.tsv'), sep = '\t', col.names = T, row.names = F)
  print('matrix top0 extracted successfully')
  write.table(out_inten_maxlfq, paste0(temp_res_dir, 'maxlfq.tsv'), sep = '\t', col.names = T, row.names = F)
  print('matrix maxlfq extracted successfully')
}

get_expression_matrix_mq_dda<-function(raw, evid, design, exps, python_path){
  platform = 'Maxquant'
  out_exp = ''
  root_tmp = strsplit(evid,'/', fixed = T)[[1]]
  temp_res_dir = gsub(root_tmp[length(root_tmp)], 'res/', evid)
  dir.create(temp_res_dir)
  
  if(length(grep('dlfq', exps))==1){
    dlfq_path<-run_dlfq(raw, evid, platform, python_path)
    dlfq_table<-read.table(dlfq_path, header = T, sep = '\t', quote = "")
    idx<-setdiff(c(1:length(colnames(dlfq_table))),
                 c(grep('protein',colnames(dlfq_table)),grep('Protein',colnames(dlfq_table))))
    
    organims_all<-get_organism_mq(dlfq_table$protein)
    out_dlfq<-cbind(dlfq_table$protein, organims_all,
                    dlfq_table[, idx])
    colnames(out_dlfq)[1:2]<-c('Protein','Organism')
    out_exp = out_dlfq
    write.table(out_exp, paste0(temp_res_dir, 'dlfq.tsv'), sep = '\t', col.names = T, row.names = F)
    print('matrix dlfq extracted successfully')
  }
  
  if (length(intersect(c('top0', 'top3', 'maxlfq', 'count'), exps))>=1){
    protein_table = read.table(raw, header = T, sep = '\t', quote = "")
    protein_table<-protein_table[which(protein_table$Reverse!="+" & protein_table$Potential.contaminant!="+"),]
    
    #spectral counts
    idx_count<-grep('MS.MS.count.',colnames(protein_table))
    idx_mq_inten<-grep('Intensity.',colnames(protein_table))
    idx_lfq<-grep('LFQ.intensity.',colnames(protein_table))
    idx_top3<-grep('Top3.',colnames(protein_table))
    
    #organs<-get_organism_mq(protein_table$Protein.IDs)
    organims_all<-get_organism_mq(protein_table$Protein.IDs)
    
    out_mq_count<-cbind(protein_table$Protein.IDs,
                        organims_all,
                        protein_table[,idx_count])
    colnames(out_mq_count)[c(1:2)]<-c('Protein', 'Organism')
    
    out_mq_inten<-cbind(protein_table$Protein.IDs,
                        organims_all,
                        protein_table[,idx_mq_inten])
    colnames(out_mq_inten)[c(1:2)]<-c('Protein', 'Organism')
    
    if(length(idx_top3)>0){
      out_mq_top3<-cbind(protein_table$Protein.IDs,
                         organims_all,
                         protein_table[,idx_top3])
      colnames(out_mq_top3)[c(1:2)]<-c('Protein', 'Organism')
      write.table(out_mq_top3, paste0(temp_res_dir, 'top3.tsv'), sep = '\t', col.names = T, row.names = F)
      print('matrix top3 extracted successfully')
    }
    
    out_mq_lfq<-cbind(protein_table$Protein.IDs,
                      organims_all,
                      protein_table[,idx_lfq])
    colnames(out_mq_lfq)[c(1:2)]<-c('Protein', 'Organism')
    
    write.table(out_mq_count, paste0(temp_res_dir, 'count.tsv'), sep = '\t', col.names = T, row.names = F)
    print('matrix count extracted successfully')
    write.table(out_mq_inten, paste0(temp_res_dir, 'top0.tsv'), sep = '\t', col.names = T, row.names = F)
    print('matrix top0 extracted successfully')
    write.table(out_mq_lfq, paste0(temp_res_dir, 'maxlfq.tsv'), sep = '\t', col.names = T, row.names = F)
    print('matrix maxlfq extracted successfully')
  }
}

get_expression_matrix_diann_dia<-function(raw, evid, design, exps, python_path){
  platform = 'DIANN'
  out_exp = ''
  root_tmp = strsplit(evid,'/', fixed = T)[[1]]
  temp_res_dir = gsub(root_tmp[length(root_tmp)], 'res/', evid)
  dir.create(temp_res_dir)
  designs<-read.table(design, header = T, sep = '\t')
  library(iq)
  for (exp in exps) {
    if(exp=='dlfq'){
      dlfq_path<-run_dlfq(raw, evid, platform, python_path)
      dlfq_table<-read.table(dlfq_path, header = T, sep = '\t', quote = "")
      colnames(dlfq_table)<-gsub('X', '', colnames(dlfq_table))
      
      sp_orga_dlfq = get_organism_dia(dlfq_table$Protein.Group, dlfq_table$Protein.Names)
      out_table = data.frame(Protein=sp_orga_dlfq$sp, Organism=sp_orga_dlfq$orga)
      sample_idx<-c()
      sample_name<-c()
      for (hn in 1:length(colnames(dlfq_table))) {
        idx<-grep(colnames(dlfq_table)[hn], designs$file)
        if(length(idx)==1){
          sample_idx<-c(sample_idx, hn)
          sample_name<-c(sample_name, designs$sample[idx])
        }
      }
      colnames(dlfq_table)[sample_idx]<-sample_name
      out_table<-cbind(out_table, dlfq_table[,sample_idx])
      
      out_exp = out_table
      write.table(out_exp, paste0(temp_res_dir, 'dlfq.tsv'), sep = '\t', col.names = T, row.names = F)
      print('matrix dlfq extracted successfully')
    }else if(exp=='maxlfq'){
      
      tab=process_long_format(evid,
                              output_filename = paste0(temp_res_dir, 'DIANN_', 'maxlfq','.tsv'),
                              annotation_col = c("Protein.Names", "Genes"),
                              normalization = "median",
                              filter_double_less = c("Global.Q.Value" = "0.01", "Global.PG.Q.Value" = "0.01"),
                              method='maxlfq', N=NULL)
    }else if(exp=='top1'){
      N=1
      tab=process_long_format(evid,
                              output_filename = paste0(temp_res_dir, 'DIANN_', exp,'.tsv'),
                              annotation_col = c("Protein.Names", "Genes"),
                              normalization = "median",
                              filter_double_less = c("Global.Q.Value" = "0.01", "Global.PG.Q.Value" = "0.01"),
                              method='topN', N=N)
    }else if(exp=='top3'){
      N=3
      tab=process_long_format(evid,
                              output_filename = paste0(temp_res_dir, 'DIANN_', exp,'.tsv'),
                              annotation_col = c("Protein.Names", "Genes"),
                              normalization = "median",
                              filter_double_less = c("Global.Q.Value" = "0.01", "Global.PG.Q.Value" = "0.01"),
                              method='topN', N=N)
    }
    tab<-as.data.frame(tab)
    sp_orga = get_organism_dia(tab$Protein.Group, tab$Protein.Names)
    out_table<-data.frame(Protein=sp_orga$sp, Organism=sp_orga$orga)
    
    out_data <- cbind(out_table, subset(tab, select = -c(Protein.Group, Protein.Names, Genes)))
    
    com<-intersect(designs$file, colnames(out_data))
    idx_col<-match(com, colnames(out_data))
    idx_des<-match(com, designs$file)
    colnames(out_data)[idx_col]<-designs$sample[idx_des]
    out_exp = out_data
    colnames(out_exp)[1:2]<-c('Protein','Organism')
    write.table(out_exp, paste0(temp_res_dir, exp, 'tsv'), sep = '\t', col.names = T, row.names = F)
    print(paste0('matrix ', exp, ' extracted successfully'))
  }
}

get_expression_matrix_spt_dia<-function(raw, evid, design, exp, python_path){
  platform = 'spt'
  out_exp = ''
  root_tmp = strsplit(evid,'/', fixed = T)[[1]]
  temp_res_dir = gsub(root_tmp[length(root_tmp)], 'res/', evid)
  dir.create(temp_res_dir)
  designs<-read.table(design, header = T, sep = '\t')
  
  for (exp in exps) {
    library(iq)
    if(exp=='dlfq'){
      dlfq_path<-run_dlfq(raw, evid, platform, python_path)
      dlfq_table<-read.table(dlfq_path, header = T, sep = '\t', quote = "")
      colnames(dlfq_table)<-gsub('X', '', colnames(dlfq_table))
      
      out_table = data.frame(Protein=dlfq_table$protein, Organism=dlfq_table$`PG.Genes`)
      sample_idx<-c()
      sample_name<-c()
      for (hn in 1:length(colnames(dlfq_table))) {
        idx<-grep(colnames(dlfq_table)[hn], designs$file)
        if(length(idx)==1){
          sample_idx<-c(sample_idx, hn)
          sample_name<-c(sample_name, designs$sample[idx])
        }
      }
      colnames(dlfq_table)[sample_idx]<-sample_name
      out_table<-cbind(out_table, dlfq_table[,sample_idx])
      
      out_exp = out_table
      write.table(out_exp, paste0(temp_res_dir, 'dlfq.tsv'), sep = '\t', col.names = T, row.names = F)
      print('matrix dlfq extracted successfully')
    }else if(exp=='maxlfq'){
      
      tab=process_long_format(evid,
                              output_filename = paste0(temp_res_dir, 'spt_', 'maxlfq','.tsv'),
                              sample_id  = "R.FileName",
                              primary_id = "PG.ProteinGroups",
                              secondary_id = c("EG.Library", "FG.Id", "FG.Charge", "F.FrgIon", "F.Charge", "F.FrgLossType"),
                              intensity_col = "F.PeakArea",
                              annotation_col = c("PG.Genes", "PG.ProteinNames", "PG.FastaFiles"),
                              filter_string_equal = c("F.ExcludedFromQuantification" = "False"),
                              filter_double_less = c("PG.Qvalue" = "0.01", "EG.Qvalue" = "0.01"),
                              log2_intensity_cutoff = 0,
                              normalization = "median",
                              method='maxlfq', N=NULL)
    }else if(exp=='top1'){
      N=1
      tab=process_long_format(evid,
                              output_filename = paste0(temp_res_dir, 'spt_', 'topN','.tsv'),
                              sample_id  = "R.FileName",
                              primary_id = "PG.ProteinGroups",
                              secondary_id = c("EG.Library", "FG.Id", "FG.Charge", "F.FrgIon", "F.Charge", "F.FrgLossType"),
                              intensity_col = "F.PeakArea",
                              annotation_col = c("PG.Genes", "PG.ProteinNames", "PG.FastaFiles"),
                              filter_string_equal = c("F.ExcludedFromQuantification" = "False"),
                              filter_double_less = c("PG.Qvalue" = "0.01", "EG.Qvalue" = "0.01"),
                              log2_intensity_cutoff = 0,
                              normalization = "median",
                              method='topN', N=N)
    }else if(exp=='top3'){
      N=3
      tab=process_long_format(evid,
                              output_filename = paste0(temp_res_dir, 'spt_', 'topN','.tsv'),
                              sample_id  = "R.FileName",
                              primary_id = "PG.ProteinGroups",
                              secondary_id = c("EG.Library", "FG.Id", "FG.Charge", "F.FrgIon", "F.Charge", "F.FrgLossType"),
                              intensity_col = "F.PeakArea",
                              annotation_col = c("PG.Genes", "PG.ProteinNames", "PG.FastaFiles"),
                              filter_string_equal = c("F.ExcludedFromQuantification" = "False"),
                              filter_double_less = c("PG.Qvalue" = "0.01", "EG.Qvalue" = "0.01"),
                              log2_intensity_cutoff = 0,
                              normalization = "median",
                              method='topN', N=N)
    }
    tab<-as.data.frame(tab)
    sp_orga = get_organism_dia(tab$PG.ProteinGroups, tab$PG.ProteinNames)
    out_table<-data.frame(Protein=sp_orga$sp, Organism=sp_orga$orga)
    
    out_data <- cbind(out_table, subset(tab, select = -c(PG.ProteinGroups, PG.ProteinNames, PG.Genes, PG.FastaFiles)))
    
    com<-intersect(designs$file, colnames(out_data))
    idx_col<-match(com, colnames(out_data))
    idx_des<-match(com, designs$file)
    colnames(out_data)[idx_col]<-designs$sample[idx_des]
    out_exp = out_data
    colnames(out_exp)[1:2]<-c('Protein','Organism')
    write.table(out_exp, paste0(temp_res_dir, exp, 'tsv'), sep = '\t', col.names = T, row.names = F)
    print(paste0('matrix ', exp, ' extracted successfully'))
  }
}

if (platform == 'FragPipe'){
    
    get_expression_matrix_fg_dda(raw, evid, design, views, python_path)
    
  }else if(platform == 'Maxquant'){
    
    get_expression_matrix_mq_dda(raw, evid, design, views, python_path)
    
  }else if(platform == 'DIANN'){
    
   get_expression_matrix_diann_dia(raw, evid, design, views, python_path)
    
  }else if(platform == 'Spectronaut'){
    
   get_expression_matrix_spt_dia(raw, evid, design, views, python_path)
    
}
  
