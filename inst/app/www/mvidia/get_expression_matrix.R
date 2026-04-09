args <- commandArgs(trailingOnly = TRUE)
print(args)
raw_file<-args[1]
platform<-args[2]
v1_ty<-args[3]
v2_ty<-args[4]
design<-args[5]
save_v1<-args[6]
save_v2<-args[7]

designs<-read.table(design, sep = '\t', header = T)

protein_table<-read.table(raw_file, header = T, sep = '\t', quote = "")

if(platform == 'FragPipe'){
  
  inten_idx<-grep('.Intensity',colnames(protein_table))
  lfq_idx<-grep('.MaxLFQ.Intensity',colnames(protein_table))
  frag_inten_idx<-setdiff(inten_idx, lfq_idx)
  count_idx_a<-grep('.Spectral.Count',colnames(protein_table))
  uni_c_idx<-grep('.Unique.Spectral.Count',colnames(protein_table))
  total_c_idx<-grep('.Total.Spectral.Count',colnames(protein_table))
  cbt_c_idx<-grep('.Combined.Spectral.Count',colnames(protein_table))
  count_idx<-setdiff(count_idx_a, c(uni_c_idx, total_c_idx, cbt_c_idx))
  
  if(length(count_idx)>0){
    out_count<-cbind(protein_table$Protein,  
                          protein_table[,count_idx])
  }else{
    print('no count data can be found')
  }
  
  if(length(frag_inten_idx)>0){
    out_iten<-cbind(protein_table$Protein, 
                       protein_table[,frag_inten_idx])
  }else{
    print('no top0 data can be found')
  }
  
  if(length(lfq_idx)>0){
    out_inten_maxlfq<-cbind(protein_table$Protein,
                          protein_table[,lfq_idx])
    }else{
      print('no maxlfq data can be found')
    }
  out_iten_top3<-vector()
}else if(platform == 'Maxquant'){
  protein_table<-protein_table[which(protein_table$Reverse!="+" & protein_table$Potential.contaminant!="+"),]
  
  idx_count<-grep('MS.MS.count.',colnames(protein_table))
  idx_mq_inten<-grep('Intensity.',colnames(protein_table))
  idx_lfq<-grep('LFQ.intensity.',colnames(protein_table))
  idx_top3<-grep('Top3.',colnames(protein_table))
  
  if(length(idx_count)>0){
    out_count<-cbind(protein_table$Protein.IDs,  
                     protein_table[,idx_count])
  }else{
    print('no count data can be found')
  }
  
  if(length(idx_mq_inten)>0){
    out_iten<-cbind(protein_table$Protein.IDs, 
                         protein_table[,idx_mq_inten])
  }else{
    print('no top0 data can be found')
  }
  
  if(length(idx_lfq)>0){
    out_inten_maxlfq<-cbind(protein_table$Protein.IDs,
                            protein_table[,idx_lfq])
  }else{
    print('no maxlfq data can be found')
  }
  
  if(length(idx_top3)>0){
    out_iten_top3<-cbind(protein_table$Protein.IDs, 
                    protein_table[,idx_top3])
  }else{
    print('no top3 data can be found')
  }
  
}

out_tables<-list(top0=out_iten, top3=out_iten_top3, count=out_count, maxlfq=out_inten_maxlfq)

get_matrix_dda<-function(platform, out_tables, view_type, save_path, designs){
  if(view_type=='top0'){
    out_table = out_tables$top0
  }else if(view_type == 'top3'){
    out_table = out_tables$top3
  }else if(view_type == 'count'){
    out_table = out_tables$count
  }else if(view_type == 'maxlfq'){
    out_table = out_tables$maxlfq
  }
  
  if (length(out_table[,1]>0)){
    colnames(out_table)[1]<-'Protein'
    if (platform == 'FragPipe'){
      colnames(out_table)<-gsub('.MaxLFQ.Intensity', '', colnames(out_table))
      colnames(out_table)<-gsub('.Intensity', '', colnames(out_table))
      colnames(out_table)<-gsub('.Combined.Spectral.Count', '', colnames(out_table))
      colnames(out_table)<-gsub('.Total.Spectral.Count', '', colnames(out_table))
      colnames(out_table)<-gsub('.Unique.Spectral.Count', '', colnames(out_table))
      colnames(out_table)<-gsub('.Spectral.Count', '', colnames(out_table))
      
    }else if(platform == 'Maxquant'){
      
      colnames(out_table)<-gsub('MS.MS.count.', '', colnames(out_table))
      colnames(out_table)<-gsub('LFQ.intensity.', '', colnames(out_table))
      colnames(out_table)<-gsub('Intensity.', '', colnames(out_table))
      colnames(out_table)<-gsub('Top3.', '', colnames(out_table))
    }
    
    write.table(out_table, save_path, sep = '\t', col.names = T, row.names = F)
  }else{
    print('no quant data found!!!')
  }
}

get_matrix_dda(platform, out_tables, v1_ty, save_v1, designs)
get_matrix_dda(platform, out_tables, v2_ty, save_v2, designs)