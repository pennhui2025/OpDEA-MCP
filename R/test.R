#browser()
shiny::shinyApp(ui = app_ui, server = app_server)
#shiny::shinyApp(ui = ui1, server = server1)
# library(ggVennDiagram)
# a<-list(a=c(1,2,3),b=c(1,3,5,7), d=c(1,2,7,9,0))
# venn=ggVennDiagram(a)
# plot_upset(venn)

#'sp|A0A075B6S2|KVD29_HUMAN','sp|A0A075B6S5|KV127_HUMAN','sp|A0A0B4J1U7|HV601_HUMAN','sp|A0A0B4J1V0|HV315_HUMAN','sp|A0A0C4DH25|KVD20_HUMAN','sp|A0A0C4DH36|HV338_HUMAN','sp|A1X283|SPD2B_HUMAN


#python /home/ubuntu/shiny_data/code/mvidia/MVIDIA.py -Itr views -ftr /home/ubuntu/shiny_data/test/new/train/dlfq.tsv,/home/ubuntu/shiny_data/test/new/train/maxlfq.tsv,/home/ubuntu/shiny_data/test/new/train/top3.tsv -p DIA-NN -v dlfq,maxlfq,top3 -dtr /home/ubuntu/shiny_data/test/new/train/design.tsv -Ite views -fte /home/ubuntu/shiny_data/test/new/test/dlfq.tsv,/home/ubuntu/shiny_data/test/new/test/maxlfq.tsv,/home/ubuntu/shiny_data/test/new/test/top3.tsv -dte /home/ubuntu/shiny_data/test/new/test/design.tsv -A DIA -py /home/ubuntu/anaconda3/bin/python -R '/usr/bin/' -r '/home/ubuntu/shiny_data/code/mvidia/' -fs 'sp|A0A075B6S2|KVD29_HUMAN','sp|A0A075B6S5|KV127_HUMAN','sp|A0A0B4J1U7|HV601_HUMAN','sp|A0A0B4J1V0|HV315_HUMAN','sp|A0A0C4DH25|KVD20_HUMAN','sp|A0A0C4DH36|HV338_HUMAN','sp|A1X283|SPD2B_HUMAN -pos M

#sp|A0A075B6S2|KVD29_HUMAN,sp|A0A075B6S5|KV127_HUMAN,sp|A0A0B4J1U7|HV601_HUMAN,sp|A0A0B4J1V0|HV315_HUMAN,sp|A0A0C4DH25|KVD20_HUMAN,sp|A0A0C4DH36|HV338_HUMAN,sp|A1X283|SPD2B_HUMAN

#"sp|A0A075B6S2|KVD29_HUMAN,sp|A0A075B6S5|KV127_HUMAN,sp|A0A0B4J1U7|HV601_HUMAN,sp|A0A0B4J1V0|HV315_HUMAN,sp|A0A0C4DH25|KVD20_HUMAN,sp|A0A0C4DH36|HV338_HUMAN,sp|A1X283|SPD2B_HUMAN"
