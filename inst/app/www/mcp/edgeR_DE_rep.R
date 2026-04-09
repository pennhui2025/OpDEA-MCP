library(edgeR)
library(dplyr)


args <- commandArgs(trailingOnly = TRUE)
print(args)

raw_file_path<-args[1]
platform<-args[2]
inten_type<-args[3]
imput<-args[4]
normal<-args[5]
dataset<-args[6]
save_fold<-args[7]
print_lab<-args[8]
true_organism<-args[9]
DE_organism<-args[10]
true_lgfc<-args[11]
logT<-args[12]
design_file_path<-args[13]
R_fold_path<-args[14]
true_lgfc<-paste0(R_fold_path, true_lgfc)

source(paste0(R_fold_path,'/preprocessing_pro_counts.R'))
source(paste0(R_fold_path,'/utils.R'))
# raw_file_path<-args[1]
# save_path<-args[2]
# design_file_path<-args[3]
# g1<-args[4]
# g2<-args[5]

file_name = raw_file_path 
design_file = design_file_path

designs = read.table(design_file, sep = '\t', header = TRUE)

set.seed(123)

if(imput=='blank'){
  imput=''
}

if(normal=='blank'){
  normal=''
}

if(logT=='T'){
  logT=T
}else if(logT=='F'){
  logT=F
}


prepro_res = preprocessing_raw_count(file_name, designs, platform, inten_type, imput = imput, normal = normal)


all_conts<-function(design_data){
  conds<-design_data$condition
  uni_con<-unique(conds)
  conts<-c()
  groups<-vector()
  for (i in 1:(length(uni_con)-1)) {
    for(j in (i+1):length(uni_con)){
      conts<-c(conts, paste0('condition', uni_con[j], '-', 'condition', uni_con[i]))
      groups<-rbind(groups, c(uni_con[i], uni_con[j]))
    }
  }
  return(list(conts=conts, groups=groups))
}

consts<-all_conts(designs)
#conss<-strsplit(conss, ';', fixed = T)[[1]]

res_all<-vector()
for (i in 1:length(consts$conts)) {
  
    groups<-consts$groups
    g1<-groups[i,][1]
    g2<-groups[i,][2]
    
    sample_names1<-designs$sample_name[grep(g1,designs$condition)]
    sample_names2<-designs$sample_name[grep(g2,designs$condition)]
    idx_g1=match(sample_names1,colnames(prepro_res$normed))
    idx_g2=match(sample_names2,colnames(prepro_res$normed))
    
    counts<-prepro_res$normed[,c(idx_g1, idx_g2)]
    condition = as.factor(designs[c(grep(g1, designs$condition), grep(g2, designs$condition)),]$condition)
    
    counts<-as.data.frame(counts)
    
    counts$na_g1 = apply(counts[,grep(g1,condition)],1,function(x) sum(is.na(x)))
    counts$na_g2 = apply(counts[,grep(g2,condition)],1,function(x) sum(is.na(x)))
    # Filter protein table. DEqMS require minimum two values for each group.
    #filter_idx = which(counts$na_g1<2 & counts$na_g2<2)
    filter_idx = which((length(idx_g1)-counts$na_g1)>=2 & (length(idx_g2)-counts$na_g2)>=2)
    counts.filter = counts[filter_idx,][,1:(length(counts)-2)]
    ## edgeR can't process NA, use 0 instead
    counts.filter<-as.matrix(counts.filter)
    counts.filter[is.na(counts.filter)] = 0
    counts.filter[which(counts.filter<0)] = 0
    counts.filter<-counts.filter+1
    
    idx<-which(apply(counts.filter, 1, function(x) sd(x))!=0)
    counts.filter<-counts.filter[idx,]
    
    y <- DGEList(counts = counts.filter, group = condition)
    y <- calcNormFactors(y)
    design <- model.matrix(~0+condition)
    y <- estimateDisp(y, design)
    fit <- glmQLFit(y, design)
    
    cont <- makeContrasts(contrasts =  consts$conts[i], levels = design)
    
    qlf <- glmQLFTest(fit, contrast=cont)
    res.edgeR <- topTags(qlf, n = Inf)$table
    res.edgeR$contrast <- consts$conts[i]
    res.edgeR <- res.edgeR %>% cbind(protein = rownames(.),.)
    
    res_all<-rbind(res_all, res.edgeR)
  
}
colnames(res_all)[length(res_all[1,])]<-'contrast'
colnames(res_all)[c(5,6)]<-c('pvalue', 'adj.pvalue')
res_all<-res_all[order(res_all$pvalue),]

if(print_lab=='T'){
  true_organisms=strsplit(true_organism,';',fixed = T)[[1]]
  DE_organisms=strsplit(DE_organism,';',fixed = T)[[1]]
  labs<-get_labels(res_all, true_organisms, DE_organisms, true_lgfc, R_fold_path)
  res_all$organism<-labs$Organism
  res_all$label<-labs$DEP
  res_all$TlogFC<-labs$TlogFC
}

file_name <- paste0(dataset, '_edgeR_', platform, '_', inten_type, '_', imput, '_', normal, '.csv')
full_file_path <- paste0(save_fold, file_name)
clean_path <- normalizePath(full_file_path, winslash = "\\", mustWork = FALSE)
long_path <- paste0("\\\\?\\", clean_path)

write.table(res_all, long_path, sep = ',', row.names = TRUE, col.names = TRUE)
