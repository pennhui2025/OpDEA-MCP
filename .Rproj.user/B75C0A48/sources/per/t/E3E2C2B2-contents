tabpanel4<-tabsetPanel(
  tabPanel("DEA",
           h4("Input files"),
           h4(p("!!!please prepare required files per following instructions.",style = "color:red")),
            shinydashboard::box(width = 12, height = 420,
                                 status = "primary", solidHeader = T,
                                 h4("Please provide view files"),
                                 strong('File preparation instructions:'),
                                 p('Step 1. Extraction of views from raw
                                   quantification outputs.'),
                                column(12,h4('We provide a module helping extract views from your proteomics data and use them as inputs to MVIDIA. see View Extraction page.' ,style = "color:red")),
                                p('example view data:'),
                                column(12,h5("Download from zenodo via the following link"),
                                       a(href = "https://zenodo.org/records/13896333",
                                         "Zenodo: views.zip;")),
                                column(12,hr()),
                                 p('Step 2. Select view files e.g., e.g., dlfq.tsv, maxlfq.tsv, etc. from Step 1.'),
                                #column(12,hr()),
                                 p('Step 3. Preparing a design.tsv file
                                 with at least columns of sample_name, condition,
                                 and replicate name, the sample name should be the
                                   name as the view file column names for
                                   intensities'),
                                #column(12,hr()),
                                 p('Step 4. Put your view data and the design file in the same folder
                                   and paste the folder address to below:'),
                                #(12,hr()),
                                column(12,textInput("view_dt_fold_dea", "view data folder", value = 'views/'))
                                 #fileInput("upload_views_dea", "Upload your views.zip file prepared per above instructions",width='100%')
                                 ),


          h4("Parameters"),
          shinydashboard::box(width = 12, height = 200,
                              status = "primary", solidHeader = T,
          column(3,radioButtons("rb_dea_plat", "Quantification platform:",
                                list('FragPipe'='FragPipe', 'Maxquant'='Maxquant',
                                 'DIA-NN' = 'DIANN', 'Spectronaut'='Spectronaut'), inline = T)),
          column(2,radioButtons("rb_dea_acq", "Acqusition type:",
                                list('DDA'='DDA',
                                     'DIA' = 'DIA'), inline = T)),

          column(4,checkboxGroupInput("rb_dea","view_names:",
                                      c("dlfq" = "dlfq",
                                        "maxlfq" = "maxlfq",
                                        "top0" = "top0",
                                        "top1" = "top1",
                                        "top3" = "top3",
                                        "count" = "count"),inline = T)),
          column(12,hr()),
          column(2,textInput("FC", "FC (fold change)", value = '1.5')),
          column(1,textInput("adjp", "adj.p-value", value = '0.05')),
          column(1,textInput("normalization", "normalization", value = 'blank')),
          column(1,textInput("imputation", "imputation", value = 'blank')),
          column(1,textInput("DEA", "limma or ROTS", value = 'limma')),
          column(1,textInput("g1", "g1 (condition 1)", value = 'A')),
          column(1,textInput("g2", "g2 (condition 2)", value = 'B')),
          column(3,radioButtons("rb_dea_mth","Method:",
                                      c("MVIDIA" = "MVIDIA",
                                        "Concat" = "KNN",
                                        "MLE" = "MLE"), inline = T)),
          ),
          shinydashboard::box(width = 12, height = 600,
                              status = "primary", solidHeader = T,
            column(12,actionButton("runMVIDIA_DEA", "MVIDIA-DEA", class = "btn-success")),
          conditionalPanel(
            condition = "input.runMVIDIA_DEA>0",
            shinydashboard::box(
              title = "Volcano plot of the differential expression analysis (DEA)",
              background = "black",
              width = 12,
              height = 600,
              column(12,p("DEA with MVIDIA, please wait for the results...")),
              #downloadLink("DEA_res_MVIDIA", "DEA_results.zip"),
              column(12,textOutput("DEA_result_MVIDIA")),
              column(6,plotOutput("volc_MVIDIA")),
              column(6,plotOutput("upset_MVIDIA"))
            )
          )
)
)
)


tabpanel5<-tabsetPanel(
  tabPanel("patient diagnosis",
           h4("Input files"),
           h4(p("!!!please prepare required files per following instructions.",style = "color:red")),
           shinydashboard::box(width = 12, height = 620,
                               status = "primary", solidHeader = T,
                               h4("Please provide view files"),
                               strong('File preparation instructions:'),
                               p('Step 1. Extraction of views from raw
                                   quantification outputs for both training and testing sample proteomics data.'),
                               column(12,h4('We provide a module helping extract views from your proteomics data and use them as inputs to MVIDIA. see View Extraction page.' ,style = "color:red")),
                               p('example view data:'),
                               column(6,h5("Download from zenodo via the following link"),
                                      a(href = "https://zenodo.org/records/13896333",
                                        "Zenodo: train.zip;")),
                               column(6,h5("Download from zenodo via the following link"),
                                      a(href = "https://zenodo.org/records/13896333",
                                        "Zenodo: test.zip;")),
                               column(12,hr()),
                               p('Step 2. Select a group of training view files e.g., e.g., dlfq.tsv, maxlfq.tsv, etc. from training views extracted from Step 1.'),
                               #column(12,hr()),
                               p('Step 3. Preparing a design.tsv file for training data,
                                 with at least columns of sample_name, sample_ID,
                                 and class, the sample_name should be the
                                   name as the view file column names for
                                   intensities.'),
                               p('Step 4. Put both the view data and design file for training data in the same folder "/train/"'),
                               p('Step 5. Select the same group of test view files e.g., e.g., dlfq.tsv, maxlfq.tsv, etc. from testing views extracted from Step 1.'),
                               #column(12,hr()),
                               p('Step 6. Preparing a design.tsv file for testing data,
                                 with at least columns of sample_name
                                 and sample_ID, the sample_name should be the
                                   name as the view file column names for
                                   intensities'),
                               p('Step 7. Put both the view data and design file for test data in the same folder "/test/"'),
                               p('Step 8. paste the train and test folder addresses to below:'),
                               #column(12,hr()),
                               column(12,textInput("view_dt_pd_tr", "view data folder for training data", value = 'train/')),
                               column(12,textInput("view_dt_pd_te", "view data folder for test data", value = 'test/'))
                               #fileInput("upload_views_pd_tr", "Upload your train_views.zip file prepared per above instructions",width='100%'),
                               #fileInput("upload_views_pd_te", "Upload your test_views.zip file prepared per above instructions",width='100%')
           ),


           h4("Parameters"),
           shinydashboard::box(width = 12, height = 410,
                               status = "primary", solidHeader = T,
                               column(3,radioButtons("rb_pd_plat", "Quantification platform:",
                                                     list('FragPipe'='FragPipe', 'Maxquant'='Maxquant',
                                                          'DIA-NN' = 'DIANN', 'Spectronaut'='Spectronaut'), inline = T)),
                               column(2,radioButtons("rb_pd_acq", "Acqusition type:",
                                                     list('DDA'='DDA',
                                                          'DIA' = 'DIA'), inline = T)),

                               column(4,checkboxGroupInput("rb_pd","view_names:",
                                                           c("dlfq" = "dlfq",
                                                             "maxlfq" = "maxlfq",
                                                             "top0" = "top0",
                                                             "top1" = "top1",
                                                             "top3" = "top3",
                                                             "count" = "count"),inline = T)),
                               column(12,hr()),
                               column(4,textInput("Signature", "Signature (proteins in quotes and seperated by ',')", value = '')),
                               column(3,radioButtons("rb_pd_mth", "Diagnosis Method:",
                                                     list('MVIDIA'='MVIDIA',
                                                          'Concat' = 'Concat',
                                                          'MLE' = 'MLE',
                                                          'Concat-MVIDIA' = 'Concat-MVIDIA'), inline = T)),
                               #column(2,textInput("mr", "filtering_missing_rate", value = '0.5')),
                               column(2,textInput("Pos", "Positive_class", value = 'M')),
                               column(12,hr()),
                               column(12,actionButton("runMVIDIA_pd", "Patient-diagnosis", class = "btn-success")),
                               conditionalPanel(
                                 condition = "input.runMVIDIA_pd>0",
                                 shinydashboard::box(
                                   background = "black",
                                   width = 12,
                                   height = 100,
                                   column(12,p("Classification of patients with MVIDIA, please wait for the results...")),
                                   column(12,textOutput("PD_result_MVIDIA")),
                                   #downloadLink("PD_res_MVIDIA", "PD_results.zip")

                                 )
                               )

           )


  )
)

tabpanel6<-tabsetPanel(
  tabPanel("Cell Clustering",
           h4("Input files"),
           h4(p("!!!please prepare required files per following instructions.",style = "color:red")),
           shinydashboard::box(width = 12, height = 400,
                               status = "primary", solidHeader = T,
                               h4("Please provide view files"),
                               strong('File preparation instructions:'),
                               p('Step 1. Extraction of views from raw
                                   quantification outputs.'),
                               column(12,h4('We provide a module helping extract views from your proteomics data and use them as inputs to MVIDIA. see View Extraction page.' ,style = "color:red")),
                               p('example view data:'),
                               column(12,h5("Download from zenodo via the following link"),
                                      a(href = "https://zenodo.org/records/13896333",
                                        "Zenodo: sc.zip;")),
                               column(12,hr()),
                               p('Step 2. Select view files e.g., e.g., dlfq.tsv, maxlfq.tsv, etc. from Step 1.
                                 (also can provide a design file with true labels.)'),
                               #column(12,hr()),
                               p('Step 3. Put your view data and the design file in the same folder
                                   and paste the folder address to below:'),
                               #column(12,hr()),
                               column(12,textInput("view_dt_cc", "view data folder", value = 'views/'))
                               #fileInput("upload_views_sc", "Upload your views.zip file prepared per above instructions",width='100%')
           ),


           h4("Parameters"),
           shinydashboard::box(width = 12, height = 400,
                               status = "primary", solidHeader = T,
                               column(3,radioButtons("rb_sc_plat", "Quantification platform:",
                                                     list('FragPipe'='FragPipe', 'Maxquant'='Maxquant',
                                                          'DIA-NN' = 'DIANN', 'Spectronaut'='Spectronaut'), inline = T)),
                               column(2,radioButtons("rb_sc_acq", "Acqusition type:",
                                                     list('DDA'='DDA',
                                                          'DIA' = 'DIA'), inline = T)),

                               column(4,checkboxGroupInput("rb_sc","view_names:",
                                                           c("dlfq" = "dlfq",
                                                             "maxlfq" = "maxlfq",
                                                             "top0" = "top0",
                                                             "top1" = "top1",
                                                             "top3" = "top3",
                                                             "count" = "count"),inline = T)),
                               column(12,hr()),

                               column(2,textInput("mr", "filtering_missing_rate", value = '0.5')),
                               column(2,textInput("nclus", "cluster_number", value = '2')),
                               column(2,textInput("ld", "latent dim", value = '3')),
                               column(2,textInput("c", "regularization ", value = '')),
                               column(12,hr()),
                               column(12,actionButton("runMVIDIA_sc", "Cell-clustering", class = "btn-success")),
                               conditionalPanel(
                                 condition = "input.runMVIDIA_sc>0",
                                 shinydashboard::box(
                                   background = "black",
                                   width = 12,
                                   height = 500,
                                   column(12,p("Single cell clustering with MVIDIA, please wait for the results...")),
                                   #downloadLink("Clustering_res_MVIDIA", "Clustering_results.zip"),
                                   column(12,textOutput("clustering_result_MVIDIA")),
                                   column(12,plotOutput("UMAP_MVIDIA"))

                                 )
                               )

           )

  )
)


tabpanel7<-tabsetPanel(
  tabPanel("view Extraction",
           h3("This page provides the function of extracting view data from raw quantification outputs"),
           p('example raw quantification data:'),
           column(12,h5("Download from zenodo via the following link"),
                  a(href = "https://zenodo.org/records/13902028",
                    "Zenodo: example_raw_quantification.zip;")),
           column(12,hr()),
           h4(p('FragPipe-DDA quantification')),
           shinydashboard::box(width = 12, height = 370,
                               status = "primary", solidHeader = T,
                               h4(p("please provide following file paths.",style = "color:red")),
                               column(4,textInput("fg_cb_pro", "combined_protein.tsv", value = './combined_protein.tsv')),
                               column(4,textInput("fg_cb_ion", "combined_ion.tsv", value = './combined_ion.tsv')),
                               column(4,textInput("fg_cb_dg", "design.tsv", value = './design.tsv')),
                               # column(4,fileInput("fg_cb_pro", "combined_protein.tsv", width = '30%')),
                               # column(4,fileInput("fg_cb_ion", "combined_ion.tsv", width = '30%')),
                               # column(4,fileInput("fg_cb_dg", "design.tsv", width = '30%')),
                               br(),
                               # column(6,p('can download example inputs via this link:')),
                               # column(2,downloadLink("fg_exam_raw", "FragPipe_DDA.zip")),
                               # br(),
                               column(12,p('please choose views you want to extract')),
                               column(12,checkboxGroupInput("rb_ext_fg","view_names (we suggest dlfq+maxlfq+top0):",
                                                           c("dlfq" = "dlfq",
                                                             "maxlfq" = "maxlfq",
                                                             "top0" = "top0",
                                                             "count" = "count"),inline = T)),

                               column(12,actionButton("runext_fg", "Extracting views", class = "btn-success")),
                               hr(),
                               conditionalPanel(
                                 condition = "input.runext_fg>0",
                                 column(12,textOutput("ext_fg"))#,
                                 #column(12,downloadLink("fg_ext_view", "FragPipe_DDA_views.zip"))

                               )

                               #fileInput("upload_views_dea", "Upload your views.zip file prepared per above instructions",width='100%')
           ),
           h4(p('Maxquant-DDA quantification')),
           shinydashboard::box(width = 12, height = 370,
                               status = "primary", solidHeader = T,
                               h4(p("please provide following file paths.",style = "color:red")),
                               column(4,textInput("mq_pro", "proteinGroups.txt", value = './txt/proteinGroups.txt')),
                               column(4,textInput("mq_evid", "evidence.txt", value = './txt/evidence.txt')),
                               column(4,textInput("mq_dg", "design.tsv", value = './design.tsv')),
                               # column(4,fileInput("mq_pro", "proteinGroups.txt", width = '30%')),
                               # column(4,fileInput("mq_evid", "evidence.txt", width = '30%')),
                               # column(4,fileInput("mq_dg", "design.tsv", width = '30%')),
                               br(),
                               # column(6,p('can download example inputs via this link:')),
                               # column(2,downloadLink("mq_exam_raw", "Maxquant_DDA.zip")),
                               # br(),
                               column(12,p('please choose views you want to extract')),
                               column(12,checkboxGroupInput("rb_ext_mq","view_namesview_names (we suggest dlfq+maxlfq+top0):",
                                                            c("dlfq" = "dlfq",
                                                              "maxlfq" = "maxlfq",
                                                              "top0" = "top0",
                                                              "count" = "count"),inline = T)),
                               #column(4,p()),
                               column(12,actionButton("runext_mq", "Extracting views", class = "btn-success")),
                               #column(4,p()),
                               hr(),
                               conditionalPanel(
                                 condition = "input.runext_mq>0",
                                 column(12,textOutput("ext_mq"))#,
                                 #column(12,downloadLink("mq_ext_view", "Maxquant_DDA_views.zip"))
                               )
           ),
           h4(p('DIA-NN-DIA quantification')),
           shinydashboard::box(width = 12, height = 350,
                               status = "primary", solidHeader = T,
                               h4(p("please provide following file paths.",style = "color:red")),
                               column(6,textInput("diann_rep", "report.tsv", value = './report.tsv')),
                               column(6,textInput("diann_dg", "design.tsv", value = './design.tsv')),
                               # column(6,fileInput("diann_rep", "report.tsv", width = '30%')),
                               # column(6,fileInput("diann_dg", "design.tsv", width = '30%')),
                               # h4(p("For Spectronaut, please Use this export schema iq.rs to make a long report, as we use 'iq' R package
                               #      to extract the views. See https://github.com/tvpham/iq.",style = "color:red")),
                               #column(12,textInput("mq_evid", "evidence.txt", value = './txt/evidence.txt')),
                               br(),
                               # column(6,p('can download example inputs via this link:')),
                               # column(2,downloadLink("diann_exam_raw", "DIANN_DIA.zip")),
                               # br(),
                               column(12,p('please choose views you want to extract')),
                               column(12,checkboxGroupInput("rb_ext_dia","view_namesview_names (we suggest dlfq+maxlfq+top3):",
                                                            c("dlfq" = "dlfq",
                                                              "maxlfq" = "maxlfq",
                                                              "top3" = "top3",
                                                              "top1" = "top1"),inline = T)),
                               #column(4,p()),
                               column(12,actionButton("runext_diann", "Extracting views", class = "btn-success")),
                               #column(4,p()),
                               hr(),
                               conditionalPanel(
                                 condition = "input.runext_diann>0",
                                 column(12,textOutput("ext_diann"))#,
                                 #column(12,downloadLink("diann_ext_view", "DIANN_DIA_views.zip"))
                               )
           ),
           h4(p('Spectronaut-DIA quantification')),
           shinydashboard::box(width = 12, height = 350,
                               status = "primary", solidHeader = T,
                               h4(p("please provide following file paths.",style = "color:red")),
                               column(6,textInput("spt_rep", "report.tsv", value = './report.tsv')),
                               column(6,textInput("spt_dg", "design.tsv", value = './design.tsv')),
                               # column(6,fileInput("spt_rep", "report.tsv", width = '30%')),
                               # column(6,fileInput("spt_dg", "design.tsv", width = '30%')),
                               h4(p("For Spectronaut, please Use this export schema iq.rs to make a long report, as we use 'iq' R package
                                    to extract the views. See https://github.com/tvpham/iq.",style = "color:red")),
                               #column(12,textInput("mq_evid", "evidence.txt", value = './txt/evidence.txt')),
                               br(),
                               # column(6,p('can download example inputs via this link:')),
                               # column(2,downloadLink("spt_exam_raw", "spt_DIA.zip")),
                               # br(),
                               column(12,p('please choose views you want to extract')),
                               column(12,checkboxGroupInput("rb_ext_dia_spt","view_namesview_names (we suggest dlfq+maxlfq+top3):",
                                                            c("dlfq" = "dlfq",
                                                              "maxlfq" = "maxlfq",
                                                              "top3" = "top3",
                                                              "top1" = "top1"),inline = T)),
                               #column(4,p()),
                               column(12,actionButton("runext_spt", "Extracting views", class = "btn-success")),
                               #column(4,p()),
                               hr(),
                               conditionalPanel(
                                 condition = "input.runext_spt>0",
                                 column(12,textOutput("ext_spt"))#,
                                 #column(12,downloadLink("spt_ext_view", "spt_DIA_views.zip"))
                               )
           )
  )
)
