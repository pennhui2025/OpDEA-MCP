
library(MSnbase)
set.seed(123)

args <- commandArgs(trailingOnly = TRUE)
raw_path<-args[1]
norm<-args[2]
imput<-args[3]
save_path<-args[4]
logT<-args[5]

# raw_path<-'./data/raw_view1.tsv'
# norm<-'center.median'
# imput<-'blank'
# save_path<-'./data/norm_view1.tsv'
# logT<-'T'

if(raw_path != '_'){
  raw_data<-read.table(raw_path, sep = '\t', header=T)
  raw_data_pro<-raw_data[,2:length(raw_data[1,])]
  row.names(raw_data_pro)<-raw_data[,1]
  raw_data_pro<-as.matrix(raw_data_pro)
  raw_data_pro[raw_data_pro==0]<-NA
  raw_data_pro[is.nan(raw_data_pro)]<-NA
  raw_data_pro[is.infinite(raw_data_pro)]<-NA
  raw_data_pro<-as.data.frame(raw_data_pro)
  #idx_retain<-which(apply(raw_data_pro,1,function(x) sum(is.na(x)))<length(raw_data_pro[1,])*0.8)
  #raw_data_pro<-raw_data_pro[idx_retain,]
}

norm_datas<-function(intens, norm, logT){
  if(norm=='blank'){
    norm = ''
  }
  if(logT=='T'){
    intens_tran<-log2(as.matrix(intens))}
  else{
    intens_tran<-as.matrix(intens)
  }
  fd <- data.frame(row.names(intens))
  row.names(fd)<-row.names(intens)
  pd <- data.frame(colnames(intens))
  row.names(pd) = colnames(intens)
  intens_imp<-MSnSet(intens_tran, fd, pd)
  
  #### normalization
  if(norm!=''){
    intens_norm<-normalise(intens_imp, norm)
  }else{
    intens_norm<-intens_imp
  }
  
  normed_data<-intens_norm@assayData[["exprs"]]
  return(normed_data)
}

imput_datas<-function(intens_norm, imput){
  if(imput=='blank'){
    imput=''
  }
  #### imputation
  if(imput!=''){
    if(imput %in% c('bpca', 'knn', 'QRILC', 'MLE', 'MinDet', 'MinProb', 'min', 'zero', 'mixed', 'nbavg', 'with', 'none')){
      if(imput=='bpca'){
        intens_tran = exprs(intens_norm)
        idx_bpca<-which(apply(intens_tran,1,function(x) sum(is.na(x)))<length(intens_tran[1,]))
        idx_imp<-idx_bpca
        intens_imp<-intens_tran[idx_bpca,]
        intens_imp<-MSnSet(exprs = intens_imp)
      }else if(imput=='knn'){
        intens_tran = exprs(intens_norm)
        idx_knn<-which(apply(intens_tran,1,function(x) sum(is.na(x)))<length(intens_tran[1,])*0.8)
        intens_imp<-intens_tran[idx_knn,]
        idx_imp<-idx_knn
        intens_imp<-MSnSet(exprs = intens_imp)
      }else{
        intens_imp=intens_norm
        idx_imp<-c(1:length(exprs(intens_norm)[,1]))
      }
      intens_imp<-MSnbase::impute(intens_imp, imput)
      intens_out_mt<-exprs(intens_norm)
      intens_out_mt[idx_imp,]<-exprs(intens_imp)
      intens_imp<-MSnSet(exprs = intens_out_mt)
    }else if(imput=='mice'){
      library(mice)
      intens_tran = exprs(intens_norm)
      intens_imp<-mice(intens_tran)
      completeData <- complete(intens_imp,2)
      completeData<-as.matrix(completeData)
      intens_imp<-MSnSet(exprs = as.matrix(completeData))
    }else if(imput=='missForest'){
      library(missForest)
      intens_tran = exprs(intens_norm)
      intens_imp<-missForest(intens_tran)
      intens_imp<-MSnSet(exprs = intens_imp$ximp)
    }else if(imput=='mi'){
      library(mi)
      intens_tran = exprs(intens_norm)
      intens_imp<-missing_data.frame(intens_tran)
      intens_imp<-mi(intens_imp)
      intens_imp<-MSnSet(exprs = intens_imp)
    }
  }else{
    intens_imp = intens_norm
  }
  imput_data_out<-intens_imp@assayData[["exprs"]]
  return(imput_data_out)
}


if(raw_path != '_'){
  normed_pro<-norm_datas(raw_data_pro, norm, logT)
  #a= normed_pro
  if(imput!='blank'){
    fd <- data.frame(row.names(normed_pro))
    row.names(fd)<-row.names(normed_pro)
    pd <- data.frame(colnames(normed_pro))
    row.names(pd) = colnames(normed_pro)
    intens_normed<-MSnSet(normed_pro, fd, pd)
    normed_pro = imput_datas(intens_normed, imput)
  }
  
  colnames(normed_pro)<-gsub('.MaxLFQ.Intensity', '', colnames(normed_pro))
  colnames(normed_pro)<-gsub('.Intensity', '', colnames(normed_pro))
  colnames(normed_pro)<-gsub('.Combined.Spectral.Count', '', colnames(normed_pro))
  colnames(normed_pro)<-gsub('.Total.Spectral.Count', '', colnames(normed_pro))
  colnames(normed_pro)<-gsub('.Unique.Spectral.Count', '', colnames(normed_pro))
  colnames(normed_pro)<-gsub('.Spectral.Count', '', colnames(normed_pro))
  
  colnames(normed_pro)<-gsub('MS.MS.count.', '', colnames(normed_pro))
  colnames(normed_pro)<-gsub('LFQ.intensity.', '', colnames(normed_pro))
  colnames(normed_pro)<-gsub('Intensity.', '', colnames(normed_pro))
  colnames(normed_pro)<-gsub('Top3.', '', colnames(normed_pro))
  
  write.table(normed_pro, save_path, sep = '\t', col.names = T, row.names = T)
}
