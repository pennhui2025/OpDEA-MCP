root_fold<-'inst/app/www/'
b64 <- base64enc::dataURI(file=paste0(root_fold,"introduction.png"), mime="www/png")
b64_1<- base64enc::dataURI(file=paste0(root_fold,"MVIDIA_intro.png"), mime="www/png")
b64_2<- base64enc::dataURI(file=paste0(root_fold,"mcp_intro.png"), mime="www/png")#base64enc::dataURI(file=paste0(root_fold,"MCP_intro.png"), mime="www/png")

tab1 <- fluidRow(
  shinydashboard::box(width=6, height = 100, checkboxGroupInput("filter1", "expression type of DDA:",
                         c("top0" = "top0",
                           "top3" = "top3",
                           "count" = "count",
                           "MaxLFQ" = 'LFQ',
                           "directLFQ" = "dlfq"), inline = TRUE)),
  shinydashboard::box(width=6, height = 100,checkboxGroupInput("filter2", "normalization:",
                         c("No_normalization" = "None",
                           "center.mean" = "center.mean",
                           "center.median" = "center.median",
                           "max" = "max",
                           "sum" = "sum",
                           "vsn" = "vsn",
                           "quantiles" = "quantiles",
                           "quantiles.robust" = "quantiles.robust",
                           "div.mean" = "div.mean",
                           "div.median" = "div.median",
                           'lossf' = 'lossf',
                           'TIC' = 'TIC',
                           "Rlr" = "Rlr",
                           'MBQN' = 'MBQN'
                         ), inline = TRUE)),
  shinydashboard::box(width=6, height = 100,checkboxGroupInput("filter3", "MVI:",
                         c("MinProb" = "MinProb",
                           "QRILC" = "QRILC",
                           "MinDet" = "MinDet",
                           "missForest" = "missForest",
                           "nbavg" = "nbavg",
                           "zero" = "zero",
                           "bpca" = "bpca",
                           "MLE" = "MLE",
                           "knn" = "knn",
                           "No_MVI" = "None",
                           "min" = "min",
                           "mice" = "mice",
                           "Impseq" = "Impseq",
                           "Impseqrob" = "Impseqrob",
                           "GMS" = "GMS",
                           'SeqKNN' = 'SeqKNN'), inline = TRUE)),
  shinydashboard::box(width=6, height = 100,checkboxGroupInput("filter4", "DEA tool:",
                         c("ANOVA" = "ANOVA",
                           "DEP" = "DEP",
                           "DEqMS" = "DEqMS",
                           'limma' = "limma",
                           "proDA" = "proDA",
                           "ROTS" = "ROTS",
                           "SAM" = "SAM",
                           "ttest" = "ttest",
                           "beta_binomial" = "beta_binomial",
                           "edgeR" = "edgeR",
                           "plgem" = "plgem",
                           'MSstats' = 'MSstats'
                         ), inline = TRUE)),
  shinydashboard::box(width = 6, height = 600,
      "workflow benchmarking",
      br(),
      column(width = 12,DT::dataTableOutput("tabFrag"), style = "height:500px; overflow-x: scroll;")
  ),
  shinydashboard::box(width = 6, height = 600,
      "performance distributions",
      br(),
      plotOutput("boxplot1", height = 500, width = 500)
  )
)

tab2 <- fluidRow(
  shinydashboard::box(width=6, height = 100,checkboxGroupInput("filter5", "expression type of DDA:",
                                                               c("top0" = "top0",
                                                                 "top3" = "top3",
                                                                 "count" = "count",
                                                                 "MaxLFQ" = 'LFQ',
                                                                 "directLFQ" = "dlfq"), inline = TRUE)),
  shinydashboard::box(width=6, height = 100,checkboxGroupInput("filter6", "normalization:",
                                                               c("No_normalization" = "None",
                                                                 "center.mean" = "center.mean",
                                                                 "center.median" = "center.median",
                                                                 "max" = "max",
                                                                 "sum" = "sum",
                                                                 "vsn" = "vsn",
                                                                 "quantiles" = "quantiles",
                                                                 "quantiles.robust" = "quantiles.robust",
                                                                 "div.mean" = "div.mean",
                                                                 "div.median" = "div.median",
                                                                 'lossf' = 'lossf',
                                                                 'TIC' = 'TIC',
                                                                 "Rlr" = "Rlr",
                                                                 'MBQN' = 'MBQN'
                                                               ), inline = TRUE)),
  shinydashboard::box(width=6, height = 100,checkboxGroupInput("filter7", "MVI:",
                                                               c("MinProb" = "MinProb",
                                                                 "QRILC" = "QRILC",
                                                                 "MinDet" = "MinDet",
                                                                 "missForest" = "missForest",
                                                                 "nbavg" = "nbavg",
                                                                 "zero" = "zero",
                                                                 "bpca" = "bpca",
                                                                 "MLE" = "MLE",
                                                                 "knn" = "knn",
                                                                 "No_MVI" = "None",
                                                                 "min" = "min",
                                                                 "mice" = "mice",
                                                                 "Impseq" = "Impseq",
                                                                 "Impseqrob" = "Impseqrob",
                                                                 "GMS" = "GMS",
                                                                 'SeqKNN' = 'SeqKNN'), inline = TRUE)),
  shinydashboard::box(width=6, height = 100,checkboxGroupInput("filter8", "DEA tool:",
                                                               c("ANOVA" = "ANOVA",
                                                                 "DEP" = "DEP",
                                                                 "DEqMS" = "DEqMS",
                                                                 'limma' = "limma",
                                                                 "proDA" = "proDA",
                                                                 "ROTS" = "ROTS",
                                                                 "SAM" = "SAM",
                                                                 "ttest" = "ttest",
                                                                 "beta_binomial" = "beta_binomial",
                                                                 "edgeR" = "edgeR",
                                                                 "plgem" = "plgem",
                                                                 'MSstats' = 'MSstats'
                                                               ), inline = TRUE)),
  shinydashboard::box(width = 6, height = 600,
      "workflow benchmarking",
      br(),
      column(width = 12,DT::dataTableOutput("tabMq"), style = "height:500px; overflow-x: scroll;")
  ),
  shinydashboard::box(width = 6, height = 600,
      "performance distributions",
      br(),
      plotOutput("boxplot2", height = 500, width = 500)
  )
)

tab3 <- fluidRow(
  shinydashboard::box(width=6, height = 100,checkboxGroupInput("filter9", "expression type of DIA:",
                                                               c("top1" = "top1",
                                                                 "top3" = "top3",
                                                                 #"count" = "sc",
                                                                 "MaxLFQ" = 'LFQ',
                                                                 "directLFQ" = "dlfq"), inline = TRUE)),
  shinydashboard::box(width=6, height = 100,checkboxGroupInput("filter10", "normalization:",
                                                               c("No_normalization" = "None",
                                                                 "center.mean" = "center.mean",
                                                                 "center.median" = "center.median",
                                                                 "max" = "max",
                                                                 "sum" = "sum",
                                                                 "vsn" = "vsn",
                                                                 "quantiles" = "quantiles",
                                                                 "quantiles.robust" = "quantiles.robust",
                                                                 "div.mean" = "div.mean",
                                                                 "div.median" = "div.median",
                                                                 'lossf' = 'lossf',
                                                                 'TIC' = 'TIC',
                                                                 "Rlr" = "Rlr",
                                                                 'MBQN' = 'MBQN'
                                                               ), inline = TRUE)),
  shinydashboard::box(width=6, height = 100,checkboxGroupInput("filter11", "MVI:",
                                                               c("MinProb" = "MinProb",
                                                                 "QRILC" = "QRILC",
                                                                 "MinDet" = "MinDet",
                                                                 "missForest" = "missForest",
                                                                 "nbavg" = "nbavg",
                                                                 "zero" = "zero",
                                                                 "bpca" = "bpca",
                                                                 "MLE" = "MLE",
                                                                 "knn" = "knn",
                                                                 "No_MVI" = "None",
                                                                 "min" = "min",
                                                                 "mice" = "mice",
                                                                 "Impseq" = "Impseq",
                                                                 "Impseqrob" = "Impseqrob",
                                                                 "GMS" = "GMS",
                                                                 'SeqKNN' = 'SeqKNN'), inline = TRUE)),
  shinydashboard::box(width=6, height = 100,checkboxGroupInput("filter12", "DEA tool:",
                                                               c("ANOVA" = "ANOVA",
                                                                 "DEP" = "DEP",
                                                                 #"DEqMS" = "DEqMS",
                                                                 'limma' = "limma",
                                                                 "proDA" = "proDA",
                                                                 "ROTS" = "ROTS",
                                                                 "SAM" = "SAM",
                                                                 "ttest" = "ttest",
                                                                 #"beta_binomial" = "beta_binomial",
                                                                 #"edgeR" = "edgeR",
                                                                 #"plgem" = "plgem",
                                                                 'MSstats' = 'MSstats'
                                                               ), inline = TRUE)),
  shinydashboard::box(width = 6, height = 600,
      "workflow benchmarking",
      br(),
      column(width = 12,DT::dataTableOutput("tabDia"), style = "height:500px; overflow-x: scroll;")
  ),
  shinydashboard::box(width = 6, height = 600,
      "performance distributions",
      br(),
      plotOutput("boxplot3", height = 500, width = 500)
  )
)

tab4 <- fluidRow(
  shinydashboard::box(width=6, height = 100,checkboxGroupInput("filter13", "expression type of DIA:",
                                                               c("top1" = "top1",
                                                                 "top3" = "top3",
                                                                 #"count" = "sc",
                                                                 "MaxLFQ" = 'LFQ',
                                                                 "directLFQ" = "dlfq"), inline = TRUE)),
  shinydashboard::box(width=6, height = 100,checkboxGroupInput("filter14", "normalization:",
                                                               c("No_normalization" = "None",
                                                                 "center.mean" = "center.mean",
                                                                 "center.median" = "center.median",
                                                                 "max" = "max",
                                                                 "sum" = "sum",
                                                                 "vsn" = "vsn",
                                                                 "quantiles" = "quantiles",
                                                                 "quantiles.robust" = "quantiles.robust",
                                                                 "div.mean" = "div.mean",
                                                                 "div.median" = "div.median",
                                                                 'lossf' = 'lossf',
                                                                 'TIC' = 'TIC',
                                                                 "Rlr" = "Rlr",
                                                                 'MBQN' = 'MBQN'
                                                               ), inline = TRUE)),
  shinydashboard::box(width=6, height = 100,checkboxGroupInput("filter15", "MVI:",
                                                               c("MinProb" = "MinProb",
                                                                 "QRILC" = "QRILC",
                                                                 "MinDet" = "MinDet",
                                                                 "missForest" = "missForest",
                                                                 "nbavg" = "nbavg",
                                                                 "zero" = "zero",
                                                                 "bpca" = "bpca",
                                                                 "MLE" = "MLE",
                                                                 "knn" = "knn",
                                                                 "No_MVI" = "None",
                                                                 "min" = "min",
                                                                 "mice" = "mice",
                                                                 "Impseq" = "Impseq",
                                                                 "Impseqrob" = "Impseqrob",
                                                                 "GMS" = "GMS",
                                                                 'SeqKNN' = 'SeqKNN'), inline = TRUE)),
  shinydashboard::box(width=6, height = 100,checkboxGroupInput("filter16", "DEA tool:",
                                                               c("ANOVA" = "ANOVA",
                                                                 "DEP" = "DEP",
                                                                 #"DEqMS" = "DEqMS",
                                                                 'limma' = "limma",
                                                                 "proDA" = "proDA",
                                                                 "ROTS" = "ROTS",
                                                                 "SAM" = "SAM",
                                                                 "ttest" = "ttest",
                                                                 #"beta_binomial" = "beta_binomial",
                                                                 #"edgeR" = "edgeR",
                                                                 #"plgem" = "plgem",
                                                                 'MSstats' = 'MSstats'
                                                               ), inline = TRUE)),
  shinydashboard::box(width = 6, height = 600,
                      "workflow benchmarking",
                      br(),
                      column(width = 12,DT::dataTableOutput("tabDia1"), style = "height:500px; overflow-x: scroll;")
  ),
  shinydashboard::box(width = 6, height = 600,
                      "performance distributions",
                      br(),
                      plotOutput("boxplot4", height = 500, width = 500)
  )
)

tab5 <- fluidRow(
  shinydashboard::box(width=6, height = 100,checkboxGroupInput("filter17", "expression type of TMT:",
                                                               c("abd" = "abd",
                                                                 "ratio" = "ratio",
                                                                 "phi" = "phi"
                                                                 #"norm_raw" = "norm_raw",
                                                                 ), inline = TRUE)),
  shinydashboard::box(width=6, height = 100,checkboxGroupInput("filter18", "normalization:",
                                                               c("No_normalization" = "None",
                                                                 "center.mean" = "center.mean",
                                                                 "center.median" = "center.median",
                                                                 "max" = "max",
                                                                 "sum" = "sum",
                                                                 "vsn" = "vsn",
                                                                 "quantiles" = "quantiles",
                                                                 "quantiles.robust" = "quantiles.robust",
                                                                 "div.mean" = "div.mean",
                                                                 "div.median" = "div.median",
                                                                 'lossf' = 'lossf',
                                                                 'TIC' = 'TIC',
                                                                 "Rlr" = "Rlr",
                                                                 'MBQN' = 'MBQN'
                                                               ), inline = TRUE)),
  shinydashboard::box(width=6, height = 100,checkboxGroupInput("filter19", "MVI:",
                                                               c("MinProb" = "MinProb",
                                                                 "QRILC" = "QRILC",
                                                                 "MinDet" = "MinDet",
                                                                 "missForest" = "missForest",
                                                                 "nbavg" = "nbavg",
                                                                 "zero" = "zero",
                                                                 "bpca" = "bpca",
                                                                 "MLE" = "MLE",
                                                                 "knn" = "knn",
                                                                 "No_MVI" = "None",
                                                                 "min" = "min",
                                                                 "mice" = "mice",
                                                                 "Impseq" = "Impseq",
                                                                 "Impseqrob" = "Impseqrob",
                                                                 "GMS" = "GMS",
                                                                 'SeqKNN' = 'SeqKNN'), inline = TRUE)),
  shinydashboard::box(width=6, height = 100,checkboxGroupInput("filter20", "DEA tool:",
                                                               c("ANOVA" = "ANOVA",
                                                                 "DEP" = "DEP",
                                                                 #"DEqMS" = "DEqMS",
                                                                 'limma' = "limma",
                                                                 "proDA" = "proDA",
                                                                 "ROTS" = "ROTS",
                                                                 "SAM" = "SAM",
                                                                 "ttest" = "ttest",
                                                                 #"beta_binomial" = "beta_binomial",
                                                                 #"edgeR" = "edgeR",
                                                                 #"plgem" = "plgem",
                                                                 'MSstats' = 'MSstats'
                                                               ), inline = TRUE)),
  shinydashboard::box(width = 6, height = 600,
                      "workflow benchmarking",
                      br(),
                      column(width = 12,DT::dataTableOutput("tabTMT1"), style = "height:500px; overflow-x: scroll;")
  ),
  shinydashboard::box(width = 6, height = 600,
                      "performance distributions",
                      br(),
                      plotOutput("boxplot5", height = 500, width = 500)
  )
)


tab6 <- fluidRow(
    shinydashboard::box(width=6, height = 100,checkboxGroupInput("filter21", "expression type of TMT:",
                                                               c(
                                                                 "intensity" = "intensity"), inline = TRUE)),
  shinydashboard::box(width=6, height = 100,checkboxGroupInput("filter22", "normalization:",
                                                               c("No_normalization" = "None",
                                                                 "center.mean" = "center.mean",
                                                                 "center.median" = "center.median",
                                                                 "max" = "max",
                                                                 "sum" = "sum",
                                                                 "vsn" = "vsn",
                                                                 "quantiles" = "quantiles",
                                                                 "quantiles.robust" = "quantiles.robust",
                                                                 "div.mean" = "div.mean",
                                                                 "div.median" = "div.median",
                                                                 'lossf' = 'lossf',
                                                                 'TIC' = 'TIC',
                                                                 "Rlr" = "Rlr",
                                                                 'MBQN' = 'MBQN'
                                                               ), inline = TRUE)),
  shinydashboard::box(width=6, height = 100,checkboxGroupInput("filter23", "MVI:",
                                                               c("MinProb" = "MinProb",
                                                                 "QRILC" = "QRILC",
                                                                 "MinDet" = "MinDet",
                                                                 "missForest" = "missForest",
                                                                 "nbavg" = "nbavg",
                                                                 "zero" = "zero",
                                                                 "bpca" = "bpca",
                                                                 "MLE" = "MLE",
                                                                 "knn" = "knn",
                                                                 "No_MVI" = "None",
                                                                 "min" = "min",
                                                                 "mice" = "mice",
                                                                 "Impseq" = "Impseq",
                                                                 "Impseqrob" = "Impseqrob",
                                                                 "GMS" = "GMS",
                                                                 'SeqKNN' = 'SeqKNN'), inline = TRUE)),
  shinydashboard::box(width=6, height = 100,checkboxGroupInput("filter24", "DEA tool:",
                                                               c("ANOVA" = "ANOVA",
                                                                 "DEP" = "DEP",
                                                                 #"DEqMS" = "DEqMS",
                                                                 'limma' = "limma",
                                                                 "proDA" = "proDA",
                                                                 "ROTS" = "ROTS",
                                                                 "SAM" = "SAM",
                                                                 "ttest" = "ttest",
                                                                 #"beta_binomial" = "beta_binomial",
                                                                 #"edgeR" = "edgeR",
                                                                 #"plgem" = "plgem",
                                                                 'MSstats' = 'MSstats'
                                                               ), inline = TRUE)),
  shinydashboard::box(width = 6, height = 600,
                      "workflow benchmarking",
                      br(),
                      column(width = 12,DT::dataTableOutput("tabTMT2"), style = "height:500px; overflow-x: scroll;")
  ),
  shinydashboard::box(width = 6, height = 600,
                      "performance distributions",
                      br(),
                      plotOutput("boxplot6", height = 500, width = 500)
  )
)

tabIntro<-fluidRow(
  shinydashboard::box(
    title = "Abstract",
    background = "green",
    width = 12,
    p("Identification of differentially expressed proteins in a proteomics workflow typically encompasses five key steps: raw data quantification, expression matrix construction, matrix normalization, missing value imputation (MVI), and differential expression analysis. The plethora of options in each step makes it challenging to identify optimal workflows that maximize the identification of differentially expressed proteins. To identify optimal workflows and their common properties, we conduct an extensive study involving 34,576 combinatoric experiments on 24 gold standard spike-in datasets. Applying frequent pattern mining techniques to top-ranked workflows, we uncover high-performing rules that demonstrate optimality has conserved properties. Via machine learning, we confirm optimal workflows are indeed predictable, with average cross-validation F1 scores and Matthew's correlation coefficients surpassing 0.84. We introduce an ensemble inference to integrate results from individual top-performing workflows for expanding differential proteome coverage and resolve inconsistencies. Ensemble inference provides gains in pAUC (up to 4.61%) and G-mean (up to 11.14%) and facilitates effective aggregation of information across varied quantification approaches such as topN, directLFQ, MaxLFQ intensities, and spectral counts. However, further development and evaluation are needed to establish acceptable frameworks for conducting ensemble inference on multiple proteomics workflows."))
)

tabWf_intro<-fluidRow(
  shinydashboard::box(
    title = "Workflow of a DEA process for label-free proteomics data and available tools for each step in the workflow",
    background = "navy",
    width = 12,
    img(src=b64, height = "500px", width = "1000px", align = "center")
    # img(src = "img/introduction.png",
    #     height = "500px", width = "1100px", align = "center")
    #HTML('<div id="stats_header"> <img src="img/introduction.png" height = "500px", width = "1100px", align = "center"/> </div>')
    )

)

tabMCPIntro<-fluidRow(
  shinydashboard::box(
    title = "Abstract",
    background = "light-blue",
    width = 12,
    p("Mass spectrometry (MS)-based proteomic quantification is a multi-dimensional problem. Current approaches focus only on single dimensions (views), which is a limited representation of the underlying proteome. Distinct quantification modes (“views”) encodes complementary biological information, but each view is separately explored and not well-integrated. This has led towards biased single-view studies in comparative proteomic analysis, including the identification of differentially expressed proteins (DEPs), which are important for drug target discovery and disease diagnosis. To exploit mutual multi-view complementarity, we introduce Multi-view Contrastive Proteomics (MCP), a semi-supervised representation learning framework that integrates these heterogeneous quantification views into a shared embedding space through contrastive learning. MCP’s novelty lies in the explicit modeling of cross-view interactions while mitigating label scarcity, technical noise, and class imbalance. By jointly leveraging complementary views rather than treating them independently, MCP maximizes biologically meaningful data for DEP detection. Benchmarked on 18 gold-standard spike-in datasets, MCP consistently predicted more true DEPs, achieving >20% increase in recall while maintaining precision (>90%) in comparison with prevailing statistical and machine learning baselines. MCP also provides interpretable insights into cross-view interactions, revealing deeper structural mechanisms underpinning proteomic data. When applied to clinical proteomics cohorts, MCP unlocks previously inaccessible biological signals: robustly identifying unique DEPs that enhance thyroid nodule stratification. MCP also identified VSIG4, a macrophage immune suppressive regulator, as a novel prognostic biomarker for colorectal cancer that was missed by conventional approaches; in perturbation studies, MCP recovers more validated IFN-γ-responsive proteins and pathways from both bulk and single-cell proteomics, confirming the roles by perturbation modeling and virtual cell generation. Overall, MCP establishes a general computational paradigm for complementarity maximization in multi-view proteomic data analysis, enabling deeper biological understanding and translational impact. MCP is available at http://www.ai4pro.tech:3838."))
)

tabMCP<-fluidRow(
  shinydashboard::box(
    title = "MCP-structure and benchmarking results",
    background = "maroon",
    width = 12,
    img(src=b64_2, height = "800px", width = "1000px", align = "center")
    # img(src = "img/introduction.png",
    #     height = "500px", width = "1100px", align = "center")
    #HTML('<div id="stats_header"> <img src="img/introduction.png" height = "500px", width = "1100px", align = "center"/> </div>')
  )
)

tabMVIDIAIntro<-fluidRow(
  shinydashboard::box(
    title = "Abstract",
    background = "purple",
    width = 12,
    p("Artificial intelligence (AI) models are powerful tools for addressing data challenges such as complexity, sparsity, and noise. Multi-view learning (MVL), which leverages multiple data representations (“views”), holds great potential by enhancing signal quality and improving task performance. However, its application in biomedical research remains largely underexplored. To address this gap, we introduce Multi-View representation to Increase Modality Depth using Integrative AI (M-VIDIA), a novel MVL framework. M-VIDIA integrates diverse views to boost performance across various tasks, including differential expression analysis, patient diagnosis, and cell clustering. Our findings demonstrate that M-VIDIA consistently outperforms other approaches, significantly improving sensitivity and signal quality. This represents a major advancement in data-driven AI, highlighting the crucial role of high-quality data representation in producing reproducible and interpretable results in biomedical research. M-VIDIA is available for use at http://www.ai4pro.tech:3838/."))
)

tabMVIDIA<-fluidRow(
  shinydashboard::box(
    title = "Overall pipeline of M-VIDIA and its performance on DEA",
    background = "olive",
    width = 12,
    img(src=b64_1, height = "800px", width = "1000px", align = "center")
    # img(src = "img/introduction.png",
    #     height = "500px", width = "1100px", align = "center")
    #HTML('<div id="stats_header"> <img src="img/introduction.png" height = "500px", width = "1100px", align = "center"/> </div>')
  )

)

tabLOPOCV<-fluidRow(
  shinydashboard::box(
    title = "Good Generalizability confirmed by Leave-One-Dataset-Out Cross-Validation",
    background = "purple",
    width = 12,
    textOutput("LOPOCV performances"),
    br(),
    column(12,plotOutput("fgcv"))#,
    #column(4,plotOutput("mqcv")),
    #column(4,plotOutput("diacv"))
    )
)

tabcls<-fluidRow(
  shinydashboard::box(
    title = "Workflow performance levels are predictable",
    background = "light-blue",
    width = 12,
    textOutput("10-fold cross-validation tests results of CatBoost classifiers"),
    br(),
    column(6,plotOutput("cvCls")),
    column(6,plotOutput("feaImp"))
  )
)

tabAC<-fluidRow(
  shinydashboard::box(
    title = "Acknowledgement and Citation",
    background = "orange",
    width = 12,
    h3("Acknowledgement"),
    p("This research/project is supported by the National Research Foundation, Singapore under its Industry Alignment Fund – Prepositioning (IAF-PP) Funding Initiative. Any opinions, findings and conclusions or recommendations expressed in this material are those of the author(s) and do not reflect the views of National Research Foundation, Singapore."),
    p("This work was partly supported by the National Innovation Fellow Program of the MOST of China (J.L., Grant No. E327130001)."),
    p("WWBG also acknowledges support from an MOE Tier 1 award (RS08/21)."),
    br(),
    h3('Publication'),
    p("Please cite the following papers:"),
    p("Hui Peng, He Wang, Weijia Kong, Jinyan Li*, Wilson Wen Bin Goh*. (2024). Optimizing differential expression analysis for proteomics data via high-performing rules and ensemble inference. Nat Commun 15, 3922 (2024). https://doi.org/10.1038/s41467-024-47899-w"),
    p("Hui Peng, Hexin Cai, Shifu Luo, Jinyan Li*, Wilson Wen Bin Goh*. (2026). Mutual complementarity maximization in multi-view proteomics enables robust biomarker prediction. "),
    p("Hui Peng, Wilson Wen Bin Goh*. (2024). M-VIDIA: A Multi-View representation to Increase modality Depth using Integrative AI. ")
  )
)

