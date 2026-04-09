library(iq)
source('./iq-master/R/iq.R')
source('./iq-master/R/iq-fast_MOD.R')
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

get_matrix<-function(raw_file, platform, norm, save_path, view_type, designs){
  if(platform=='DIANN'){
    tab=process_long_format(raw_file,
                            output_filename = save_path,
                            annotation_col = c("Protein.Names", "Genes"),
                            normalization = norm,#"median",
                            filter_double_less = c("Global.Q.Value" = "0.01", "Global.PG.Q.Value" = "0.01"),
                            method=view_type)
    
    tab<-as.data.frame(tab)
    #tab$Protein = sp_orga$sp
    #tab$Organism = sp_orga$orga
    out_table<-data.frame(Protein=tab$Protein.Group)
    
    out_data <- cbind(out_table, subset(tab, select = -c(Protein.Group, Protein.Names, Genes)))
    
    com<-intersect(designs$file, colnames(out_data))
    idx_col<-match(com, colnames(out_data))
    idx_des<-match(com, designs$file)
    colnames(out_data)[idx_col]<-designs$sample[idx_des]
    
    write.table(out_data, save_path, sep = '\t', col.names = T, row.names = F)
  }else if(platform == 'Spectronaut'){
    tab=process_long_format(raw_file,
                            output_filename = save_path,
                            sample_id  = "R.FileName",
                            primary_id = "PG.ProteinGroups",
                            secondary_id = c("EG.Library", "FG.Id", "FG.Charge", "F.FrgIon", "F.Charge", "F.FrgLossType"),
                            intensity_col = "F.PeakArea",
                            annotation_col = c("PG.Genes", "PG.ProteinNames", "PG.FastaFiles"),
                            filter_string_equal = c("F.ExcludedFromQuantification" = "False"),
                            filter_double_less = c("PG.Qvalue" = "0.01", "EG.Qvalue" = "0.01"),
                            log2_intensity_cutoff = 0,
                            normalization = norm,#"median",
                            method=view_type)
    
    tab<-as.data.frame(tab)
    
    out_table<-data.frame(Protein=tab$PG.ProteinGroups)
    
    out_data <- cbind(out_table, subset(tab, select = -c(PG.ProteinGroups, PG.ProteinNames, PG.Genes, PG.FastaFiles)))
    
    com<-intersect(designs$file, colnames(out_data))
    idx_col<-match(com, colnames(out_data))
    idx_des<-match(com, designs$file)
    colnames(out_data)[idx_col]<-designs$sample[idx_des]
    
    write.table(out_data, save_path, sep = '\t', col.names = T, row.names = F)
  }
}

if(v1_ty != 'maxlfq'){
  norm1 = 'none'
}else{
  norm1 = 'median'
}

get_matrix(raw_file, platform, norm1, save_v1, v1_ty, designs)

if(v2_ty != 'maxlfq'){
  norm2 = 'none'
}else{
  norm2 = 'median'
}
get_matrix(raw_file, platform, norm2, save_v2, v2_ty, designs)