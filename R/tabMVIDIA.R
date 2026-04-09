tabpanel4<-tabsetPanel(
  tabPanel("DEA",
           h4("Input files"),
           h4(p("!!!please only provide one group of following required files.",style = "color:red")),
           fluidRow(
             shinydashboard::box(width = 6, height = 320,
                                 status = "primary", solidHeader = T,
                                 h4("Please provide view files"),
                                 strong('File preparation instructions:'),
                                 p('step 1. For views e.g., dlfq, maxlfq and top0,
                                   storing log2 transformed intensities
                                   (or original counts) in a
                                   Tab-separated values(tsv) format file with the
                                   first two columns listing the protein uniport
                                   id and the corresponding organism, file names
                                   are the same as the view name, e.g., dlfq.tsv.'),
                                 p('step 2. Preparing a design file in tsv format
                                 with at least columns of sample_name, condition,
                                 and replicate name, the sample name should be the
                                   name as the view file column names for
                                   intensities, with the name of design.tsv'),
                                 p('step 3. Compressing the view files and design file to
                                   a views.zip file.'),
                                 hr(),
                                 fileInput("upload_views_dea", "Upload your views.zip file prepared per above instructions",width='100%')
                                 ),
             shinydashboard::box(width = 6, height = 320,
                               status = "primary", solidHeader = T,

                               h4("Please provide raw outputs from quantification platforms:"),
                               strong('File preparation instructions:'),
                               p('step 1. For Fragpipe outputs, please provide files
                                 [combined_protein.tsv, combined_ion.tsv, design.tsv];
                                 For Maxquant outputs, please provide files
                                 [proteinGroups.txt, evidence.txt, design.tsv];
                                 For DIA-NN or Spectronaut, please provide files
                                 [report.tsv, design.tsv]'),
                               p('step 2. The design file (design.tsv) should be in tsv format
                                 with at least columns of sample name, condition,
                                 and replicate name, the sample_name should be the
                                   name as the samples named in the quantification
                                 output files'),
                               p('step 3. Compressing the files to
                                   a zip file with their quantification platform
                                 name as the file name, e.g., DIANN.zip, Spectranaut.zip,
                                 FragPipe.zip or Maxquant.zip.'),

                               hr(),
                               fileInput("upload_raw_dea", "Upload your zip file for raw quantification outputs prepared per above instructions",width='100%')
                               )

           ),
          h4("Parameters"),
          shinydashboard::box(width = 12, height = 200,
                              status = "primary", solidHeader = T,
          column(3,radioButtons("rb_dea_plat", "Quantification platform:",
                                list('FragPipe'='F', 'Maxquant'='M',
                                 'DIA-NN' = 'D', 'Spectronaut'='S'), inline = T)),
          column(3,radioButtons("rb_dea_acq", "Acqusition type:",
                                list('DDA'='DDA',
                                     'DIA' = 'DIA'), inline = T)),

          column(3,radioButtons("rb_dea_inp", "input file type:",
                                list('views'='V',
                                     'Raw' = 'R'), inline = T)),
          column(12,hr()),
          column(1,textInput("FC", "FC (fold change)", value = '1.5')),
          column(1,textInput("adjp", "adj.p-value", value = '0.05')),
          column(1,textInput("normalization", "normalization", value = 'blank')),
          column(1,textInput("imputation", "imputation", value = 'blank')),
          column(1,textInput("imputation", "imputation", value = 'blank')),
          column(1,textInput("DEA", "limma or ROTS", value = 'limma')),
          column(1,textInput("g1", "g1 (condition 1)", value = 'A')),
          column(1,textInput("g2", "g2 (condition 2)", value = 'B'))
          ),
          shinydashboard::box(width = 12, height = 600,
                              status = "primary", solidHeader = T,
            column(12,actionButton("runMVIDIA_DEA", "MVIDIA-DEA", class = "btn-success")),
          conditionalPanel(
            condition = "input.runMVIDIA_DEA>0",
            shinydashboard::box(
              title = "Volcano plot of the differential expression analysis (DEA)",
              background = "light-blue",
              width = 12,
              height = 500,
              column(12,p("DEA with MVIDIA, please wait for the results...")),
              column(12,textOutput("DEA_res_MVIDIA")),
              column(12,plotOutput("volc_MVIDIA")),
              downloadLink("DEA_res_MVIDIA", "DEA results.zip")

            )
          )
)
)
)

tabpanel5<-tabsetPanel(
  tabPanel("patient diagnosis",
           h4("Input files"),
           h4(p("!!!please only provide one group of following required files.",style = "color:red")),
           fluidRow(
             shinydashboard::box(width = 6, height = 320,
                                 status = "primary", solidHeader = T,
                                 h4("Please provide view files"),
                                 strong('File preparation instructions:'),
                                 p('step 1. For views e.g., dlfq, maxlfq and top0,
                                   storing log2 transformed intensities
                                   (or original counts) in a
                                   Tab-separated values(tsv) format file with the
                                   first two columns listing the protein uniport
                                   id and the corresponding organism, file names
                                   are the same as the view name, e.g., dlfq.tsv.'),
                                 p('step 2. Preparing a design file in tsv format
                                 with at least columns of sample_name, condition,
                                 and replicate name, the sample name should be the
                                   name as the view file column names for
                                   intensities, with the name of design.tsv'),
                                 p('step 3. Compressing the view files and design file to
                                   a views.zip file.'),
                                 hr(),
                                 fileInput("upload_views_dea", "Upload your views.zip file prepared per above instructions",width='100%')
             ),
             shinydashboard::box(width = 6, height = 320,
                                 status = "primary", solidHeader = T,

                                 h4("Please provide raw outputs from quantification platforms:"),
                                 strong('File preparation instructions:'),
                                 p('step 1. For Fragpipe outputs, please provide files
                                 [combined_protein.tsv, combined_ion.tsv, design.tsv];
                                 For Maxquant outputs, please provide files
                                 [proteinGroups.txt, evidence.txt, design.tsv];
                                 For DIA-NN or Spectronaut, please provide files
                                 [report.tsv, design.tsv]'),
                                 p('step 2. The design file (design.tsv) should be in tsv format
                                 with at least columns of sample name, condition,
                                 and replicate name, the sample_name should be the
                                   name as the samples named in the quantification
                                 output files'),
                                 p('step 3. Compressing the files to
                                   a zip file with their quantification platform
                                 name as the file name, e.g., DIANN.zip, Spectranaut.zip,
                                 FragPipe.zip or Maxquant.zip.'),

                                 hr(),
                                 fileInput("upload_raw_dea", "Upload your zip file for raw quantification outputs prepared per above instructions",width='100%')
             )

           ),
           h4("Parameters"),
           shinydashboard::box(width = 12, height = 200,
                               status = "primary", solidHeader = T,
                               column(3,radioButtons("rb_dea_plat", "Quantification platform:",
                                                     list('FragPipe'='F', 'Maxquant'='M',
                                                          'DIA-NN' = 'D', 'Spectronaut'='S'), inline = T)),
                               column(3,radioButtons("rb_dea_acq", "Acqusition type:",
                                                     list('DDA'='DDA',
                                                          'DIA' = 'DIA'), inline = T)),

                               column(3,radioButtons("rb_dea_inp", "input file type:",
                                                     list('views'='V',
                                                          'Raw' = 'R'), inline = T)),
                               column(12,hr()),
                               column(1,textInput("FC", "FC (fold change)", value = '1.5')),
                               column(1,textInput("adjp", "adj.p-value", value = '0.05')),
                               column(1,textInput("normalization", "normalization", value = 'blank')),
                               column(1,textInput("imputation", "imputation", value = 'blank')),
                               column(1,textInput("imputation", "imputation", value = 'blank')),
                               column(1,textInput("DEA", "limma or ROTS", value = 'limma')),
                               column(1,textInput("g1", "g1 (condition 1)", value = 'A')),
                               column(1,textInput("g2", "g2 (condition 2)", value = 'B'))
           ),
           shinydashboard::box(width = 12, height = 600,
                               status = "primary", solidHeader = T,
                               column(12,actionButton("runMVIDIA_DEA", "MVIDIA-DEA", class = "btn-success")),
                               conditionalPanel(
                                 condition = "input.runMVIDIA_DEA>0",
                                 shinydashboard::box(
                                   title = "Volcano plot of the differential expression analysis (DEA)",
                                   background = "light-blue",
                                   width = 12,
                                   height = 500,
                                   column(12,p("DEA with MVIDIA, please wait for the results...")),
                                   column(12,textOutput("DEA_res_MVIDIA")),
                                   column(12,plotOutput("volc_MVIDIA")),
                                   downloadLink("DEA_res_MVIDIA", "DEA results.zip")

                                 )
                               )
           )


  )
)

tabpanel6<-tabsetPanel(
  tabPanel("Cell Clustering",


  )
)
