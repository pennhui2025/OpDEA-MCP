import copy
import numpy as np
from Utils import get_views_from_raw, get_views_from_view_file_names, get_datasets_new, DEA_single_hurdle_new, get_multi_view_data
from Utils import seed_torch
import argparse
import os
import MV_classification as MVCLS
import MV_clustering as MVCLU

seed_torch()


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='run MVIDIA for classification and clustering')
    parser.add_argument('-T', '--task', default='clustering', #required=True,
                        help='MVIDIA task type, can be: "classification" for classification task; "clustering" for clustering task')
    parser.add_argument('-Itr', '--input_type_train', default='views', #required=True,
                        help='Input data type, can be "raw" for raw quantification tables; "views" for view matrices')
    parser.add_argument('-ftr', '--view_files_train', default='dlfq.tsv,maxlfq.tsv,top0.tsv', #required=True,
                        help='Input data file name, '
                             'if input_type = "raw" then the view_files is in the format of: '
                             '"protein quantification output file,quantification evidence file", e.g.,'
                             '"combined_protein.tsv,combined_ion.tsv".'
                             'if input_type = "views", then the view_files is in the format of: '
                             '"path to view1,path to view2,...", e.g., "dlfq.tsv,maxlfq.tsv,top0.tsv"'
                             )
    parser.add_argument('-Ite', '--input_type_test', default='views',
                        help='Input data type, can be "raw" for raw quantification tables; "views" for view matrices')
    parser.add_argument('-fte', '--view_files_test', default='dlfq.tsv,maxlfq.tsv,top0.tsv',
                        help='Input data file name, '
                             'if input_type = "raw" then the view_files is in the format of: '
                             '"protein quantification output file,quantification evidence file", e.g.,'
                             '"combined_protein.tsv,combined_ion.tsv".'
                             'if input_type = "views", then the view_files is in the format of: '
                             '"path to view1,path to view2,...", e.g., "dlfq.tsv,maxlfq.tsv,top0.tsv"'
                        )
    parser.add_argument('-dtr', '--design_train', default='design_file_train.txt', #required=True,
                        help='the path to design file of the experiment for training data, with file, sample_name, condition and replicate terms')
    parser.add_argument('-dte', '--design_test', default='design_file_test.txt', #required=True,
                        help='the path to design file of the experiment for test data, with file, sample_name, condition and replicate terms')
    parser.add_argument('-m', '--miss_rate', default=0.5,
                        help='the missing rate threshold, if a protein has a miss_rate higher than this theshold then drop it')
    parser.add_argument('-p', '--platform', default='FragPipe', #required=True,
                        help='quantification platform, e.g., FragPipe, Maxquant, DIANN, Spectronaut, unknow(blank)')
    parser.add_argument('-v', '--view_names', default="dlfq,maxlfq,top0", #required=True,
                        help='view_names for quantification, e.g., top0, maxlfq, top3, dlfq for DDA LFQ; top1, top3, maxlfq, dlfq DIA LFQ')
    parser.add_argument('-R', '--Rscript', default='Rscript ',
                        help='the path to Rscript.exe for running R packages to conduct differential expression analysis')
    parser.add_argument('-g', '--pg_ty', default='remove',
                        help='the method for process proteingroups, we remove them by default, can be "remove", "conserve" and "none"')
    parser.add_argument('-t', '--trial', default=30,
                        help='trial number for the hyperparameter optimization')
    parser.add_argument('-mth', '--method', default='MVIDIA',
                        help='the method for patient diagnosis [MVIDIA, Concat, MLE, Concat-MVIDIA] or clustering [MVIDIA, Concat]')
    parser.add_argument('-s', '--save', default='./',
                        help='result saving folder')
    parser.add_argument('-r', '--Rcodes', default='./',
                        help='folder storing the R scripts')
    parser.add_argument('-A', '--acq', default='DDA', #required=True,
                        help='result saving folder')
    parser.add_argument('-py', '--python_path', default='./', #required=True,
                        help='the path to python.exe for running python codes to extract dlfq intensity')
    parser.add_argument('-fs', '--feature_set', default='',
                        help='a string of protein ids or a txt file storing the strings of protein ids')
    parser.add_argument('-pos', '--positive_class', default='', #required=True,
                        help='the positive class symbol')
    parser.add_argument('-ncluster', '--number_cluster', default=3, #required=True,
                        help='the single cell data filtering type')
    parser.add_argument('-d', '--dimensions', default='3',  # required=True,
                        help='the latent dimension')
    parser.add_argument('-c', '--regular', default='blank',  # required=True,
                        help='the regularization parameter')


    args = parser.parse_args()

    R_fold = args.Rscript #'D:/R-4.3.1/bin/'#
    R_code = args.Rcodes
    platform = args.platform #'Spectronaut' #
    task = args.task #'clustering' #'classification'#
    view_names = args.view_names #'dlfq,maxlfq,top3'#
    in_ty_train = args.input_type_train #'views'#
    view_files_train = args.view_files_train #'C:\\Users\\hui\\AppData\\Local\\Temp\\Rtmp8YY1Q4/510763921ab707673e866721/sc/dlfq.tsv,C:\\Users\\hui\\AppData\\Local\\Temp\\Rtmp8YY1Q4/510763921ab707673e866721/sc/maxlfq.tsv,C:\\Users\\hui\\AppData\\Local\\Temp\\Rtmp8YY1Q4/510763921ab707673e866721/sc/top3.tsv'#
    in_ty_test = args.input_type_test #'views'#
    view_files_test = args.view_files_test #'C:\\Users\\hui\\AppData\\Local\\Temp\\Rtmp8YY1Q4/ba4cb78f5c61a32674a90667/test/dlfq.tsv,C:\\Users\\hui\\AppData\\Local\\Temp\\Rtmp8YY1Q4/ba4cb78f5c61a32674a90667/test/maxlfq.tsv,C:\\Users\\hui\\AppData\\Local\\Temp\\Rtmp8YY1Q4/ba4cb78f5c61a32674a90667/test/top3.tsv'#
    design_train = args.design_train #'C:\\Users\\hui\\AppData\\Local\\Temp\\Rtmp8YY1Q4/510763921ab707673e866721/sc/design.tsv'#
    design_test = args.design_test #'C:\\Users\\hui\\AppData\\Local\\Temp\\Rtmp8YY1Q4/ba4cb78f5c61a32674a90667/test/design.tsv'#
    pg_ty = args.pg_ty
    trial_num = args.trial
    save_data_root = args.save #'C:\\Users\\hui\\AppData\\Local\\Temp\\Rtmp8YY1Q4/510763921ab707673e866721/sc/'#
    acq = args.acq #'DIA'#
    python_path = args.python_path #'python'#
    method = args.method #'MVIDIA'#

    save_folder = save_data_root + 'res/'
    isExist = os.path.exists(save_folder)
    if not isExist:
        os.makedirs(save_folder)

    view_name_split = np.array(view_names.split(','))
    if acq == 'DDA':
        logTs = ['T'] * len(view_name_split)
        ct_idx = np.where((np.array(view_names) == 'count'))[0]
        if len(ct_idx) > 0:
            deas[ct_idx[0]] = 'edgeR'
            logTs[ct_idx[0]] = 'F'
    elif acq == 'DIA':
        logTs = ['T', 'F', 'F']

    if task == 'classification':


        pro_list = args.feature_set
        # ("'sp|P02765|FETUA_HUMAN', 'sp|P04083|ANXA1_HUMAN', 'sp|O00339|MATN2_HUMAN', 'sp|P58546|MTPN_HUMAN', "
        #  "'sp|O75347|TBCA_HUMAN', 'sp|P04216|THY1_HUMAN', 'sp|P02751|FINC_HUMAN', 'sp|P83731|RL24_HUMAN', "
        #  "'sp|P00568|KAD1_HUMAN', 'sp|P78527|PRKDC_HUMAN', 'sp|P04792|HSPB1_HUMAN', 'sp|P57737|CORO7_HUMAN',"
        #  "'sp|P42224|STAT1_HUMAN', 'sp|P27797|CALR_HUMAN', 'sp|Q9HAT2|SIAE_HUMAN', 'sp|P30086|PEBP1_HUMAN', "
        #  "'sp|O14964|HGS_HUMAN', 'sp|P10909|CLUS_HUMAN', 'sp|P17931|LEG3_HUMAN'")
        pos = args.positive_class #'M'

        pro_list = pro_list.replace('\'','').replace(' ','').split(',')
        train_data = MVCLS.get_multi_view_data(in_ty_train, pg_ty, view_files_train, design_train, view_names, R_fold,
                                               R_code, python_path, acq, platform, save_folder, logTs)

        if acq == 'DDA':
            logTs = ['T'] * len(view_name_split)
            ct_idx = np.where((np.array(view_names) == 'count'))[0]
            if len(ct_idx) > 0:
                deas[ct_idx[0]] = 'edgeR'
                logTs[ct_idx[0]] = 'F'
        elif acq == 'DIA':
            logTs = ['T', 'F', 'F']
        test_data = MVCLS.get_multi_view_data(in_ty_test, pg_ty, view_files_test, design_test, view_names, R_fold,
                                               R_code, python_path, acq, platform, save_folder, logTs)

        MVCLS.MVIDIA_classificaion(pro_list, trial_num, train_data, test_data, pos, method, view_names, save_folder)

    elif task == 'clustering':

        name = 'SC'
        filter_type = 'percent'#args.filter_type
        filter_num = float(args.miss_rate)
        ncluster = int(args.number_cluster)
        ld = int(args.dimensions)
        c = args.regular

        view_data_mv = MVCLU.get_multi_view_data(in_ty_train, pg_ty, view_files_train, design_train, view_names, R_fold,
                                         R_code, python_path, acq, platform, save_folder, logTs)
        out_raw, out_intersect = MVCLU.preprocess_multi_view_new(view_data_mv, filter_num, filter_type, save_folder, name)
        MVCLU.clustering(out_intersect, ncluster, save_folder, name, view_name_split, ld, c)

if __name__ == '__main__':

    main()








