library(MSnbase)
library(GMSimpute)
library(SeqKnn)
library(rrcovNA)
library(NormalyzerDE)
library(limma)
set.seed(123)

args <- commandArgs(trailingOnly = TRUE)
sc_file<-args[1]
norm_sc<-args[2]
imput_sc<-args[3]
save_path_sc<-args[4]
logT<-args[5]

pro_file = '_'
#print(args)
#integ<-as.numeric(args[6]) # integrate pep and pro, 0 true 1 false
#Rscript /data/proteomics/data/2023.1.7/temp0/normalization_MVAE_R.R
#/data/proteomics/data/2023.1.7/temp0/CPTAC_raw_sc.csv
#/data/proteomics/data/2023.1.7/temp0/CPTAC_raw_pro.csv
#blank blank MinProb MinProb /data/proteomics/data/2023.1.7/temp0/CPTAC_normed_sc.csv
#/data/proteomics/data/2023.1.7/temp0/CPTAC_normed_pro.csv

# sc_file<-'/data/proteomics/data/2023.1.7/temp0/CPTAC_raw_sc.csv'
# pro_file<-'/data/proteomics/data/2023.1.7/temp0/CPTAC_raw_pro.csv'
# norm_sc<-'blank'
# norm_pro<-'blank'
# imput_sc<-'bpca'
# imput_pro<-'bpca'
# save_path_sc<-args[7]
# save_path_pro<-args[8]
if (sc_file != '_' ){
  raw_data_sc<-read.table(sc_file, sep = ',', header=FALSE)
  raw_data_sc<-as.matrix(raw_data_sc)
  raw_data_sc[raw_data_sc==0]<-NA
  raw_data_sc[is.nan(raw_data_sc)]<-NA
  raw_data_sc[is.infinite(raw_data_sc)]<-NA
  raw_data_sc<-as.data.frame(raw_data_sc)
}

if(pro_file != '_'){
  raw_data_pro<-read.table(pro_file, sep = ',', header=FALSE)
  raw_data_pro<-as.matrix(raw_data_pro)
  raw_data_pro[raw_data_pro==0]<-NA
  raw_data_pro[is.nan(raw_data_pro)]<-NA
  raw_data_pro[is.infinite(raw_data_pro)]<-NA
  raw_data_pro<-as.data.frame(raw_data_pro)
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
  # if(norm!=''){
  #   intens_norm<-normalise(intens_imp, norm)
  # }else{
  #   intens_norm<-intens_imp
  # }
  
  if(norm!=''){
    if(norm=='Rlr'){
      intens_norm = performGlobalRLRNormalization(intens_imp@assayData[["exprs"]], noLogTransform = T)
      intens_norm<-MSnSet(exprs = intens_norm)
    }else if(norm=='lossf'){
      intens_norm = normalizeCyclicLoess(intens_imp@assayData[["exprs"]], method = "fast")
      intens_norm<-MSnSet(exprs = intens_norm)
    }else if(norm=='MBQN'){
      library(MBQN)
      mtx <- as.matrix(intens_imp@assayData[["exprs"]])
      mtx.trqn <- mbqn(mtx, FUN = mean)
      row.names(mtx.trqn) <- row.names(intens_imp@assayData[["exprs"]])
      colnames(mtx.trqn) <- colnames(intens_imp@assayData[["exprs"]])
      intens_norm <- mtx.trqn
      intens_norm<-MSnSet(exprs = intens_norm)
      #intens_norm = normalizeCyclicLoess(intens_imp@assayData[["exprs"]], method = "fast")
      #intens_norm<-MSnSet(exprs = intens_norm)
    }else if(norm=='TIC'){
      exp_mat = intens_imp@assayData[["exprs"]]
      exp_mat[which(is.na(exp_mat))]=0
      tics = colSums(exp_mat)
      intens_norm=vector()
      for (i in 1:length(tics)) {
        intens_norm = cbind(intens_norm, exp_mat[,i]/tics[i])
      }
      intens_norm = intens_norm * median(tics)
      intens_norm[intens_norm==0]=NA
      colnames(intens_norm)=colnames(exp_mat)
      intens_norm<-MSnSet(exprs = intens_norm)
    }else{
      intens_norm<-normalise(intens_imp, norm)
    }
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
        intens_imp<-intens_tran[idx_bpca,]
        intens_imp<-MSnSet(exprs = intens_imp)
      }else if(imput=='knn'){
        intens_tran = exprs(intens_norm)
        idx_knn<-which(apply(intens_tran,1,function(x) sum(is.na(x)))<length(intens_tran[1,])*0.8)
        intens_imp<-intens_tran[idx_knn,]
        intens_imp<-MSnSet(exprs = intens_imp)
      }else{
        intens_imp=intens_norm
      }
      intens_imp<-MSnbase::impute(intens_imp, imput)
      
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
    }else if(imput=='Impseq' | imput=='Impseqrob'){
      #library(mi)
      intens_tran = exprs(intens_norm)
      #intens_imp<-missing_data.frame(intens_tran)
      mat = matrix(NA, nrow=length(intens_tran[,1]),ncol=length(intens_tran[1,]))
      for (i in 1:length(intens_tran[1,])) {
        mat[,i]=as.numeric(intens_tran[,i])
      }
      if(imput=='Impseq'){
        mat<-impSeq(mat)
      }else if(imput=='Impseqrob'){
        mat<-impSeqRob(mat)$x
      }
      
      intens_imp<-intens_tran
      for (i in 1:length(intens_tran[1,])) {
        intens_imp[,i]=mat[,i]
      }
      intens_imp<-MSnSet(exprs = intens_imp)
    }else if(imput=='GMS'){
      #library(mi)
      intens_tran = exprs(intens_norm)
      #idx_gms<-which(apply(intens_tran,1,function(x) sum(is.na(x)))<length(intens_tran[1,])*0.8)
      #intens_tran<-intens_tran[idx_gms,]
      #intens_imp<-missing_data.frame(intens_tran)
      intens_imp<-GMS.Lasso(as.data.frame(intens_tran),nfolds=3, log.scale=F, TS.Lasso=TRUE)
      intens_imp<-MSnSet(exprs = as.matrix(intens_imp))
    }else if(imput=='GRR'){
      intens_tran = exprs(intens_norm)
      #idx_GRR<-which(apply(intens_tran,1,function(x) sum(is.na(x)))<length(intens_tran[1,]))
      #intens_tran<-intens_tran[idx_GRR,]
      library(DreamAI)
      intens_imp<-impute.RegImpute(data=as.matrix(intens_tran), fillmethod = "row_mean", maxiter_RegImpute = 10,conv_nrmse = 1e-03)
      intens_imp<-MSnSet(exprs = intens_imp)
    }else if(imput=='SeqKNN'){
      #library(mi)
      intens_tran = exprs(intens_norm)
      #intens_imp<-missing_data.frame(intens_tran)
      intens_imp<-SeqKNN(intens_tran, k=10)
      intens_imp<-MSnSet(exprs = intens_imp)
    }
  }else{
    intens_imp = intens_norm
  }
  imput_data_out<-intens_imp@assayData[["exprs"]]
  return(imput_data_out)
}

if(sc_file != '_'){
  normed_sc<-norm_datas(raw_data_sc, norm_sc, logT)
  
  if(imput_sc!='blank'){
    fd <- data.frame(row.names(normed_sc))
    row.names(fd)<-row.names(normed_sc)
    pd <- data.frame(colnames(normed_sc))
    row.names(pd) = colnames(normed_sc)
    sc_normed<-MSnSet(normed_sc, fd, pd)
    normed_sc = imput_datas(sc_normed, imput_sc)
  }
  #normed_sc = 2^normed_sc
  #normed_sc[is.na(normed_sc)]<-0
  #normed_pro[is.na(normed_pro)]<-0
  
  
  write.table(normed_sc, save_path_sc, sep = ',', col.names = FALSE, row.names = FALSE)
}

if(pro_file != '_'){
normed_pro<-norm_datas(raw_data_pro, norm_pro, logT)
#a= normed_pro
if(imput_pro!='blank'){
  fd <- data.frame(row.names(normed_pro))
  row.names(fd)<-row.names(normed_pro)
  pd <- data.frame(colnames(normed_pro))
  row.names(pd) = colnames(normed_pro)
  intens_normed<-MSnSet(normed_pro, fd, pd)
  normed_pro = imput_datas(intens_normed, imput_pro)
}


write.table(normed_pro, save_path_pro, sep = ',', col.names = FALSE, row.names = FALSE)
}