root_fold<-'inst/app/www/'
b641 <- base64enc::dataURI(file=paste0(root_fold,"introduction_server.png"), mime="www/png")
b642 <- base64enc::dataURI(file=paste0(root_fold,"tutorial_viewing_benchmarking.png"), mime="www/png")
b643 <- base64enc::dataURI(file=paste0(root_fold,"toolkit_tutorial.png"), mime="www/png")
b644 <- base64enc::dataURI(file=paste0(root_fold,"MVIDIA_DEA.png"), mime="www/png")
b645 <- base64enc::dataURI(file=paste0(root_fold,"MVIDIA_patient_diagnosis.png"), mime="www/png")
b646 <- base64enc::dataURI(file=paste0(root_fold,"MVIDIA_cell_clustering.png"), mime="www/png")
b647 <- base64enc::dataURI(file=paste0(root_fold,"MCP_DEA.png"), mime="www/png")

help_tab0<-fluidRow(
  shinydashboard::box(width = 12, height = 1200,
                      status = "primary", solidHeader = T,
                      strong("Introdcution of the Toolkit:"),
                      hr(),
                      p('This toolkit includes 6 main function panels including Introduction,  Benchmarking, Suggestion & DEA, Multi-view Proteomics, Data and Help.
      In the [Introduction] panel, we first present the abstract of this work, then we show an overview of the proteomics data differential expression analysis workflow and the options available in each workflow steps.
      We later present the abstract of our newly design multi-view learning framework for proteomics data analysis together with the pipeline of the MVIDIA. At last, the Acknowledgement and Citation information are shown.'),
                      hr(),
                      strong('In the [Benchmarking] panel,'),
                      p('users can view our benchmarking results including checking the rank position and performance
      metric values of a workflow tested with our benchmaking datasets.'),
                      hr(),
                      strong('In the [Suggestion & DEA] panel, '),
                      p('we provide the tools for suggesting optimal workflows and conduct differential expression analysis with the
       suggested workflow directly.'),
                      hr(),
                      strong('In the [Multi-view Proteomics] panel, '),
                      p('we provide three functional modules for completing proteomics data analysis tasks such as MCP-DEA, MVIDIA-DEA, patient diagnosis and cell clustering.'),
                      hr(),
                      strong('In the [Data] panel, '),
                      p('the user can get the links where raw proteomics data are available. The raw quantification results of the raw data, the extracted expression matrices and our benchmarking results can be downloaded.'),
                      p('The data used in our case studies in our MCP and MVIDIA paper are provided.'),
                      p('We also provide the link for downloading our offline toolkit with the same function as the webserver.',style = "color:red"),
                      hr(),
                      strong('In the [Help] panel, '),
                      p('We introduce the webserver and show what the users can do with our toolkit'),
                      # img(src = "img/introduction_server.png",
                      #          height = "500px", width = "1100px", align = "center")
                      hr(),
                      img(src=b641, height = "700px", width = "800px", align = "center")
  )
)

help_tab1<-fluidRow(
  shinydashboard::box(width = 12, height = 1200,
                      status = "primary", solidHeader = T,
                      strong("View workflow benchmarking results:"),
                      hr(),
                      strong('Step 1: Choosing an interested setting, e.g., label-free DDA data quantified with FragPipe, click the item of DDA_LFQ-FragPipe.'),
                      p('There are 7852, 7852, 6284, 6284, 4720 and 1568 workflows under settings of LFQ_DDA-FragPipe, LFQ_DDA-Maxquant, LFQ_DIA-DIANN,
      LFQ_DIA-Spectronaut, TMT-FragPipe and TMT-Maxquant.'),
                      hr(),
                      strong('Step 2: Filtering workflows, e.g., check the options in the 4 checkboxes that you are interested in, or by using a key word.'),
                      p('Drag the scroll bar below the table can check the ranking with a specific metric. The ranking of workflows is based-on the average
      ranking based on the five performance metrics (the column of avg_rank_mean).'),
                      hr(),
                      strong('Step 3: Click one row of the table to check the performance distributions testing on different datasets.'),
                      p('12 DDA datasets, 7 DIA datasets and 5 TMT datasets were used to evaluate the performances of workflows, the boxplot shows the performance
      distributions of the 5 metrics.'),
                      p('Only one row is permitted to be select each time, click the same row twice can cancel the selection.',style = "color:red" ),
                      hr(),
                      strong('Step 4: View the details of performance distrubutions of the five metrics in the right bottom figure.'),

                      # img(src = "img/tutorial_viewing_benchmarking.png",
                      #          height = "500px", width = "1100px", align = "center")
                      hr(),
                      img(src=b642, height = "500px", width = "800px", align = "center")
  )
)

help_tab2<-fluidRow(
  shinydashboard::box(width = 12, height = 1800,
                      status = "primary", solidHeader = T,
                      strong("Workflow suggestion and testing:"),
                      hr(),
                      strong('Step 1: Choose data type that the user has.'),
                      p('Lable-free DDA (DDA), label free DIA (DIA) and TMT data are supported. Choose the one correponding to the data you have.'),
                      hr(),
                      strong('Step 2: Choose a quantification platform.'),
                      p('For DDA data, FragPipe and Maxquant are supported. For DIA data, DIA-NN and Spectronaut are
      supported. For TMT data, again, FragPipe and Maxquant are supported. Just choose the one that you used to quantify your proteomics data.'),
                      hr(),
                      strong('Step 3: Choose to suggest single workflow or try the ensemble inference.'),
                      p('If single workflow is preferred. The user should choose whether the have some preferred options in the workflow step. If yes, then check Yes, and select the options in below checkboxes, otherwise select No.'),
                      p('For DDA and DIA data, our server also support to suggest ensemble inference where multiple workflow are integrated by a p-value integration method.
      The ens_multi-quant approach is used defaulty. '),
                      p('The imputation method missForest and MLE are quite time-consuming, if they are suggested, we will replace them with MinProb.',style = "color:red" ),
                      hr(),
                      strong('Step 4: Click the suggest worklfow button or Try ensemble inference button.'),
                      p('Our server will suggest the top 1st workflow after filtering the workflows with user specified option preference. The potential alternative
      options will also be shown according to our option comparsion results'),
                      hr(),
                      strong('Step 5: specify the path to the raw quantification result data, the path to python executable where drectlfq python package has been installed, and specify thresholds and submit the DEA task.'),
                      p('The user should specify the file paths of their raw quantification result file, e.g., the combined_protein.tsv file from FragPipe.
                      the directLFQ python package should be installed, see the tutorial at: https://github.com/MannLabs/directlfq.
                        The designed file showing the sample condition and group information must be uploaded at the same time.
                        The log2FC threshold and adj.pvalue (p-value is ajusted with BH method) threshold should be specified. At last, click DEA button to submit the task.'),
                      hr(),
                      strong('Step 6: Check DEA results.'),
                      p('After submitting the DEA task, the user should wait for a while till the task being completed. A volcano plot will be generated. The result file are stored locally, their file folder
                        will be shown above the volcano plot.'),
                      # img(src = "img/toolkit_tutorial.png",
                      #          height = "1000px", width = "1100px", align = "center")
                      hr(),
                      img(src=b643, height = "700px", width = "800px", align = "center")
  )
)

help_tab3<-fluidRow(
  shinydashboard::box(width = 12, height = 1000,
                      status = "primary", solidHeader = T,
                      h4("Contact:"),
                      strong('OpDEA'),
                      p('Any problem about OpDEA please contact Hui Peng: '),
                      p('Email: penghui@suat-sz.edu.cn'),
                      p('or cdph2009@163.com'),
                      br(),
                      p('Or, can email to the corresponding authors: '),
                      p('Jinyan Li: jinyan.li@siat.ac.cn; '),
                      p('Wilson Wen Bin Goh: wilsongoh@ntu.edu.sg'),
                      hr(),
                      strong('MCP'),
                      p('Any problem about MCP please contact Hui Peng: '),
                      p('Email: penghui@suat-sz.edu.cn'),
                      p('or cdph2009@163.com'),
                      br(),
                      p('Or, can email to the corresponding authors: '),
                      p('Jinyan Li: jinyan.li@siat.ac.cn; '),
                      p('Wilson Wen Bin Goh: wilsongoh@ntu.edu.sg'),
                      hr(),
                      strong('M-VIDIA'),
                      p('Any problem about MVIDIA please contact Hui Peng: '),
                      p('Email: penghui@suat-sz.edu.cn '),
                      p('or cdph2009@163.com'),
                      br(),
                      p('Or, can email to the corresponding author: '),
                      p('Wilson Wen Bin Goh: wilsongoh@ntu.edu.sg')
  )
)

help_tab_mCP_dea<-fluidRow(
  shinydashboard::box(width = 12, height = 1800,
                      status = "primary", solidHeader = T,
                      strong("Applying MCP to DEA tasks:"),
                      hr(),
                      strong('Step 1: Choose the task "MCP-DEA" under Multi-view Proteomics in the left menu.'),
                      hr(),
                      strong('Step 2 Preparing your input data.'),
                      p('We provide a module helping extract views from your proteomics data and use them as inputs to MCP. see View Extraction page.'),
                      #downloadLink("view_preparation4", "view_preparation.R"),
                      hr(),
                      strong('Step 3: Upload view data.'),
                      p('After extracting the view data, Compressing the view files  (e.g., dlfq.tsv, maxlfq.tsv, etc.) and the design file (design.tsv) to a zip file named "views.zip". Then, upload the zip file to the server.'),
                      hr(),
                      strong('Step 4: Setting the parameters.'),
                      p('A.) Please specify the parameters including:'),
                      p('B.) Quantification platform, e.g., FragPipe for DDA data, DIA-NN for DIA data, etc.'),
                      p('C.) proteomics data acqusition mode, DDA or DIA'),
                      p('D.) views used, please upload all the views , e.g., dlfq and maxlfq, etc., and design file'),
                      p('E.) The thesholds for defining differentially expressed proteins (DEPs), including the logFC and the adj.pvalue.'),
                      p('F.) Specifying whehter the data should be normalized and the missing values should be imputed. Avaialbe options please see the pages in Benchmarking.'),
                      p('G.) Specify the group label that you are willing to compare, e.g., A, B.'),
                      hr(),
                      strong('Step 5: After specifying the parameters, please click the MCP-DEA button to submit the task.' ),
                      hr(),
                      strong('Step 6: Check DEA results.'),
                      p('After submitting the DEA task, the user should wait for a while till the task being completed. A volcano plot and the a upset plot showing the DEP overlaps obtained by single view methods and the multi-view method will be generated. The result file can be downloaded through the blue link.'),
                      # img(src = "img/toolkit_tutorial.png",
                      #          height = "1000px", width = "1100px", align = "center")
                      hr(),
                      img(src=b647, height = "700px", width = "800px", align = "center")
  )
)

help_tab_mvidia_dea<-fluidRow(
  shinydashboard::box(width = 12, height = 1800,
                      status = "primary", solidHeader = T,
                      strong("Applying MVIDIA to DEA tasks:"),
                      hr(),
                      strong('Step 1: Choose the task "MVIDIA-DEA" under "Multi-view Proteomics" in the left menu.'),
                      hr(),
                      strong('Step 2 Preparing your input data.'),
                      p('We provide a module helping extract views from your proteomics data and use them as inputs to MVIDIA. see View Extraction page.'),
                      #downloadLink("view_preparation4", "view_preparation.R"),
                      hr(),
                      strong('Step 3: Upload view data.'),
                      p('After extracting the view data, Compressing the view files  (e.g., dlfq.tsv, maxlfq.tsv, etc.) and the design file (design.tsv) to a zip file named "views.zip". Then, upload the zip file to the server.'),
                      hr(),
                      strong('Step 4: Setting the parameters.'),
                      p('A.) Please specify the parameters including:'),
                      p('B.) Quantification platform, e.g., FragPipe for DDA data, DIA-NN for DIA data, etc.'),
                      p('C.) proteomics data acqusition mode, DDA or DIA'),
                      p('D.) views used, 3 views are suggested, but please select at least 2 views, e.g., dlfq and maxlfq, etc.'),
                      p('E.) The thesholds for defining differentially expressed proteins (DEPs), including the logFC and the adj.pvalue.'),
                      p('F.) Specifying whehter the data should be normalized and the missing values should be imputed. Avaialbe options please see the pages in Benchmarking.'),
                      p('G.) Specify the group label that you are willing to compare, e.g., A, B.'),
                      p('H.) Choose the multi-view DEA method, MVIDIA, Concat or MLE.'),
                      hr(),
                      strong('Step 5: After specifying the parameters, please click the MVIDIA-DEA button to submit the task.' ),
                      hr(),
                      strong('Step 6: Check DEA results.'),
                      p('After submitting the DEA task, the user should wait for a while till the task being completed. A volcano plot and the a upset plot showing the DEP overlaps obtained by single view methods and the multi-view method will be generated. The result file can be downloaded through the blue link.'),
                      # img(src = "img/toolkit_tutorial.png",
                      #          height = "1000px", width = "1100px", align = "center")
                      hr(),
                      img(src=b644, height = "700px", width = "800px", align = "center")
  )
)

help_tab_mvidia_pd<-fluidRow(
  shinydashboard::box(width = 12, height = 1800,
                      status = "primary", solidHeader = T,
                      strong("Applying MVIDIA to patient diagnosis (classification) tasks:"),
                      hr(),
                      strong('Step 1: Choose the task "Patient Diagnosis" under "Multi-view Proteomics" in the left menu.'),
                      hr(),
                      strong('Step 2 Preparing your input data.'),
                      p('We provide a module helping extract views from your proteomics data and use them as inputs to MVIDIA. see View Extraction page.'),
                      #downloadLink("view_preparation5", "view_preparation.R"),
                      hr(),
                      strong('Step 3: Upload view data.'),
                      p('After extracting the view data for both training data and test data, Compressing the view files  (e.g., dlfq.tsv, maxlfq.tsv, etc.) and the design file (design.tsv) to two zip files named "train.zip" and "test.zip". Then, upload the zip files to the server.'),
                      hr(),
                      strong('Step 4: Setting the parameters.'),
                      p('Please specify the parameters including:'),
                      p('A.) Quantification platform, e.g., FragPipe for DDA data, DIA-NN for DIA data, etc.'),
                      p('B.) proteomics data acqusition mode, DDA or DIA'),
                      p('C.) views used, 3 views are suggested, but please select at least 2 views, e.g., dlfq and maxlfq, etc.'),
                      p('D.) The signatures used for patient diagnosis, it is a set of proteins, can be DEPs detected by our MVIDIA-DEA.'),
                      p('E.) Choose the multi-view DEA method, MVIDIA, Concat or MLE.'),
                      p('F.) Specify the label of the positive class'),
                      hr(),
                      strong('Step 5: After specifying the parameters, please click the "Patient Diagnosis" button to submit the task.'),
                      hr(),
                      strong('Step 6: Check diagnosis results.'),
                      p('After submitting the patient diagnosis task, the user should wait for a while till the task being completed. A notification will appear showing the running time after the task being completed. The result file can be downloaded through the blue link.'),
                      # img(src = "img/toolkit_tutorial.png",
                      #          height = "1000px", width = "1100px", align = "center")
                      hr(),
                      img(src=b645, height = "700px", width = "800px", align = "center")
  )
)

help_tab_mvidia_cluster<-fluidRow(
  shinydashboard::box(width = 12, height = 1800,
                      status = "primary", solidHeader = T,
                      strong("Applying MVIDIA to cell clustering tasks:"),
                      hr(),
                      strong('Step 1: Choose the task "Cell Clustering" under "Multi-view Proteomics" in the left menu.'),
                      hr(),
                      strong('Step 2 Preparing your input data.'),
                      p('We provide a module helping extract views from your proteomics data and use them as inputs to MVIDIA. see View Extraction page.'),
                      #downloadLink("view_preparation5", "view_preparation.R"),
                      hr(),
                      strong('Step 3: Upload view data.'),
                      p('After extracting the view data, Compressing the view files  (e.g., dlfq.tsv, maxlfq.tsv, etc.) and the design file (design.tsv) to a zip file named "sc.zip". Then, upload the zip file to the server.'),
                      hr(),
                      strong('Step 4: Setting the parameters.'),
                      p('Please specify the parameters including:'),
                      p('A.) Quantification platform, e.g., FragPipe for DDA data, DIA-NN for DIA data, etc.'),
                      p('B.) proteomics data acqusition mode, DDA or DIA'),
                      p('C.) views used, 3 views are suggested, but please select at least 2 views, e.g., dlfq and maxlfq, etc.'),
                      p('D.) Cell filtering threshold, per the missing rate.'),
                      p('E.) Specifying the number of clusters.'),
                      p('F.) Specify the hyperparameters for GCCA used for cell embedding.'),
                      hr(),
                      strong('Step 5: After specifying the parameters, please click the "Cell-Clustering" button to submit the task.'),
                      hr(),
                      strong('Step 6: Check clustering results.'),
                      p('After submitting the cell clustering task, the user should wait for a while till the task being completed. A notification will appear showing the running time after the task being completed. The result file can be downloaded through the blue link.'),
                      # img(src = "img/toolkit_tutorial.png",
                      #          height = "1000px", width = "1100px", align = "center")
                      hr(),
                      img(src=b646, height = "700px", width = "800px", align = "center")
  )
)
