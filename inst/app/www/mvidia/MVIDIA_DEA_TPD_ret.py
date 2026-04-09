import copy
import numpy as np
import pandas as pd
from Utils import DEA_single_hurdle_new, get_multi_view_data
from Utils import seed_torch
import argparse
import os
from MV_regression import get_MVDEA_res, metrics_reference

seed_torch()


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='run MVIDIA for DEA')
    parser.add_argument('-I', '--input_type', default='raw', required=True,
                        help='Input data type, can be "raw" for raw quantification tables; "views" for view matrices')
    parser.add_argument('-f', '--view_files', default='dlfq.tsv,maxlfq.tsv,top0.tsv', required=True,
                        help='Input data file name, '
                             'if input_type = "raw" then the view_files is in the format of: '
                             '"protein quantification output file,quantification evidence file", e.g.,'
                             '"combined_protein.tsv,combined_ion.tsv".'
                             'if input_type = "views", then the view_files is in the format of: '
                             '"path to view1,path to view2,...", e.g., "dlfq.tsv,maxlfq.tsv,top0.tsv"'
                        )
    parser.add_argument('-q', '--qval_threshold', default=0.05,
                        help='adj.p-value threshold for determining differential expression')
    parser.add_argument('-l', '--logfc_threshold', default=np.log2(1.5),
                        help='logFC threshold for determining differential expression')
    parser.add_argument('-p', '--platform', default='FragPipe', required=True,
                        help='quantification platform, e.g., FragPipe, Maxquant, DIANN, Spectronaut, unknow(blank)')
    parser.add_argument('-v', '--view_names', default="dlfq,maxlfq,top0", required=True,
                        help='view_names for quantification, e.g., top0, maxlfq, top3, dlfq for DDA LFQ; top1, top3, maxlfq, dlfq DIA LFQ')
    parser.add_argument('-n', '--normalization', default='blank',
                        help='normalization method, can be center.median, center.mean, quantiles, vsn, ...')
    parser.add_argument('-i', '--imputation', default='blank',
                        help='imputation method, can be MinProb, QRILC, knn, ..., we currently recommend no imputation')
    parser.add_argument('-D', '--DEA', default='limma',
                        help='differential expression analysis tool, e.g., limma, ROTS, more tools will be added in our future version')
    parser.add_argument('-d', '--design', default='design_file.txt', required=True,
                        help='the path to design file of the experiment, with file, sample_name, condition and replicate terms')
    parser.add_argument('-g1', '--group1', default='A', required=True,
                        help='condition for contrast, e.g., A, control, should  be one of the conditions in design file')
    parser.add_argument('-g2', '--group2', default='D', required=True,
                        help='another condition for contrast, e.g., D, test, should  be one of the conditions in design file')
    parser.add_argument('-R', '--Rscript', default='Rscript ',
                        help='the path to Rscript.exe for running R packages to conduct differential expression analysis')
    parser.add_argument('-c', '--combine_type', default='min',
                        help='the method for combining confidence scores from two views')
    parser.add_argument('-a', '--cca', default='GCCA',
                        help='the method for combining confidence scores from two views')
    parser.add_argument('-y', '--pos_ty', default='0_h',
                        help='the method for selecting reliable samples, "0_h" means the consistency between dlfq and hurdle; "0_1_2" means the consistency among all views')
    parser.add_argument('-g', '--pg_ty', default='remove',
                        help='the method for process proteingroups, we remove them by default, can be "remove", "conserve" and "none"')
    parser.add_argument('-t', '--trail', default=30,
                        help='trial number for the hyperparameter optimization')
    parser.add_argument('-s', '--save', default='./',
                        help='result saving folder')
    parser.add_argument('-r', '--Rcodes', default='./',
                        help='folder storing the R scripts')
    parser.add_argument('-A', '--acq', default='DDA', required=True,
                        help='result saving folder')
    parser.add_argument('-py', '--python_path', default='./', required=True,
                        help='the path to python.exe for running python codes to extract dlfq intensity')

    args = parser.parse_args()

    R_fold = args.Rscript
    R_code = args.Rcodes
    platform = args.platform
    design = args.design
    imput = args.imputation
    norm = args.normalization
    dea = args.DEA
    view_names = args.view_names
    view_files = args.view_files
    g1 = args.group1
    g2 = args.group2
    in_ty = args.input_type
    qval_thre = args.qval_threshold
    lfc_thre = args.logfc_threshold
    cbt = args.cbt
    pos_ty = args.pos_ty
    pg_ty = args.pg_ty
    trial_num = args.trail
    save_data_root = args.save
    acq = args.acq
    python_path = args.python_path

    hc_thr = 0.0001
    npos_thr = 20
    seed = 2023
    diff_ty = 'auc'
    cross_imp = 'none'

    view_name_split = view_names.split(',')
    view_str = view_names

    logTs = 'F' * len(view_name_split)
    deas = dea * len(view_name_split)
    norms = norm * len(view_name_split)
    imputs = imput * len(view_name_split)

    save_folder = save_data_root + 'res/'

    (views, view_datas, views_intersect, view_datas_intersect, idx_g1g2s, view_paths_dt, dataset, views_ci,
     view_datas_ci,
     view_paths_ci, idx_g1g2s_ci, out_design, view_logFCs,
     view_confis, view_logFCs_intersect, view_confis_intersect, views_union, view_datas_union,
     view_logFCs_union, view_confis_union) = get_multi_view_data(
        in_ty,
        pg_ty,
        view_files, g1,
        g2, design, view_names,
        R_fold, R_code, python_path, acq, platform,
        save_folder, logTs, cross_imp, seed)

    view_paths = copy.deepcopy(view_paths_dt)
    for v in range(len(view_str.split(','))):
        view_paths.update({view_str.split(',')[v]: view_files.split(',')[v]})
    view_paths.update({'save_folder': save_folder})
    view_paths.update({'Rscript_path': R_fold})
    view_paths.update({'R_fold': R_code})
    view_paths.update({'true_organism': '_'})
    view_paths.update({'DE_organism': '_'})
    view_paths.update({'true_lgfc': '-'})
    view_paths.update({'dataset': 'test'})

    view_paths.update({'design': view_paths_dt['design'], 'logT': view_paths_dt['logT']})

    ex_tys = view_str.split(',')
    logTs = view_paths['logT']

    res_union = DEA_single_hurdle_new(views_union, view_datas_union,
                                      idx_g1g2s, ex_tys,
                                      view_paths, logTs,
                                      g1, g2, norms,
                                      imputs, deas,
                                      save_folder,
                                      view_paths['dataset'], 'union', R_code,
                                      platform)

    print('union')
    MVDEA_union = get_MVDEA_res(views_union, res_union, save_folder, lfc_thre, qval_thre, cbt, view_names,
                                hc_thr, npos_thr, trial_num,
                                model_type, seed, diff_ty, pos_ty)

    seed_torch()
    isExist = os.path.exists(save_folder + res_union[4]['dataset'] + '/')
    if not isExist:
        os.makedirs(save_folder + res_union[4]['dataset'] + '/')

    all_test_proteins, idx = np.unique(MVDEA_union[2], return_index=True)

    metrics_all, methods_all, out_all_res = metrics_reference(view_names, all_test_proteins, MVDEA_union[3][idx],
                                                              MVDEA_union[4],
                                                              MVDEA_union[9], MVDEA_union[7],
                                                              MVDEA_union[5],
                                                              MVDEA_union[10], MVDEA_union[8],
                                                              MVDEA_union[11], MVDEA_union[12],
                                                              MVDEA_union[13],
                                                              MVDEA_union[0], MVDEA_union[1],
                                                              MVDEA_union[14],
                                                              MVDEA_union[15], thre, logfc_thr, MVDEA_union[16],
                                                              'union')

    out_all_res.to_csv(
        save_folder + '/' + res_ori[4]['dataset'] + '/' + res_ori[4]['dataset'] + '_union_res_all.csv',
        sep=',', index=False)



if __name__ == '__main__':

    # main()
    '''
    !!! to reproduce the result, please replace following root_fold with your root for the Supplementary_code/
    !!! also replace the value of "Rscript_path" with your address to "Rscript.exe" in your R installation folder
    '''
    root_fold = 'E:/proteomics/manus4_1/draft/'
    Rscript_path = 'D:/R-4.3.1/bin/'
    python_path = ''

    logfc_thr = np.log2(2)
    hc_thr = 0.0001
    npos_thr = 20
    trial_num = 30
    seed = 2023
    diff_ty = 'auc'
    platform = 'DIANN'

    dea = 'limma'
    norm = 'blank'
    imput = 'blank'
    cbt = 'min'
    thre = 0.01
    acq = 'DIA'
    if acq == 'DIA':
        logT = 'F'  # 'F'
    elif acq == 'DDA':
        logT = 'T'


    cca = 'GCCA'

    g1 = 'B'
    g2 = 'M'
    pos_ty = '0_h'
    pg_ty = 'remove'
    cross_imp = 'none'

    view_names = ['dlfq', 'maxlfq', 'top3']
    raw_file = ''
    evidence_file = ''

    logTs = ['F', 'F', 'F']
    deas = [dea] * len(view_names)
    norms = [norm] * len(view_names)
    imputs = [imput] * len(view_names)
    save_data_root = root_fold + 'Supplementary_codes/Result_reproducing/Patient_diagnosis/'
    R_fold = ''
    save_folder = save_data_root + 'raw_res_3v_ret_0_h/'

    view_str = 'dlfq,maxlfq,top3'

    mat_folder = root_fold + 'Supplementary_codes/Result_reproducing/Patient_diagnosis/data/raw_'
    design_path = root_fold + 'Supplementary_codes/Result_reproducing/Patient_diagnosis/data/raw_avg_design_te.csv'
    design = pd.read_csv(design_path, sep='\t', header=0)
    design_value = design.values
    design_headers = np.array(design.columns.values)
    idx_PID = np.where((design_headers == 'MS_file_name'))[0][0]
    # design_value[:, idx_PID] = 'X' + design['MS_file_name'].values
    design_headers[idx_PID] = 'sample_name'
    design_headers[np.where((design_headers == 'Classificaiton_type'))[0][0]] = 'condition'
    design_path_tr = mat_folder + 'design_path_te.csv'
    pd.DataFrame(design_value, columns=design_headers).to_csv(design_path_tr, sep='\t', index=False)

    view_files = mat_folder + 'avg_dlfq_ret.csv,' + mat_folder + 'avg_maxlfq_ret.csv,' + mat_folder + 'avg_top3_ret.csv'

    (views, view_datas, views_intersect, view_datas_intersect, idx_g1g2s, view_paths_dt, dataset, views_ci,
     view_datas_ci,
     view_paths_ci, idx_g1g2s_ci, out_design, view_logFCs,
     view_confis, view_logFCs_intersect, view_confis_intersect, views_union, view_datas_union,
     view_logFCs_union, view_confis_union) = get_multi_view_data(
        'views',
        pg_ty,
        view_files, g1,
        g2, design_path_tr, view_str,
        Rscript_path, R_fold, python_path, acq, platform,
        save_folder, logTs, cross_imp, seed)

    view_paths = copy.deepcopy(view_paths_dt)
    for v in range(len(view_str.split(','))):
        view_paths.update({view_str.split(',')[v]: view_files.split(',')[v]})
    view_paths.update({'save_folder': save_folder})
    view_paths.update({'Rscript_path': Rscript_path})
    view_paths.update({'R_fold': R_fold})
    view_paths.update({'true_organism': '_'})
    view_paths.update({'DE_organism': '_'})
    view_paths.update({'true_lgfc': '-'})
    view_paths.update({'dataset': 'TPD'})

    view_paths.update({'design': view_paths_dt['design'], 'logT': view_paths_dt['logT']})

    ex_tys = view_str.split(',')
    logTs = view_paths['logT']

    res_union = DEA_single_hurdle_new(views_union, view_datas_union,
                                      idx_g1g2s, ex_tys,
                                      view_paths, logTs,
                                      g1, g2, norms,
                                      imputs, deas,
                                      save_folder,
                                      view_paths['dataset'], 'union', R_fold,
                                      platform)

    if len(view_datas_ci) > 0:
        res_ci = DEA_single_hurdle_new(
            views_ci, view_datas_ci,
            idx_g1g2s_ci, ex_tys,
            view_paths, logTs,
            g1, g2, norms,
            imputs, deas,
            save_folder,
            view_paths['dataset'], 'cca', R_fold,
            platform)

    else:
        res_ci = []

    print('union')
    MVDEA_union = get_MVDEA_res(views_union, res_union, save_folder, logfc_thr, thre, cbt, view_names,
                                hc_thr, npos_thr, trial_num,
                                seed, diff_ty, pos_ty)

    seed_torch()
    isExist = os.path.exists(save_folder + res_union[4]['dataset'] + '/')
    if not isExist:
        os.makedirs(save_folder + res_union[4]['dataset'] + '/')

    all_test_proteins, idx = np.unique(MVDEA_union[2], return_index=True)

    metrics_all, methods_all, out_all_res = metrics_reference(view_names, all_test_proteins, MVDEA_union[3][idx],
                                                              MVDEA_union[4],
                                                              MVDEA_union[9], MVDEA_union[7],
                                                              MVDEA_union[5],
                                                              MVDEA_union[10], MVDEA_union[8],
                                                              MVDEA_union[11], MVDEA_union[12],
                                                              MVDEA_union[13],
                                                              MVDEA_union[0], MVDEA_union[1],
                                                              MVDEA_union[14],
                                                              MVDEA_union[15], thre, logfc_thr, MVDEA_union[16],
                                                              'union')

    out_all_res.to_csv(
        save_folder + '/' + res_union[4]['dataset'] + '/' + res_union[4]['dataset'] + '_union_res_all.csv',
        sep=',', index=False)



