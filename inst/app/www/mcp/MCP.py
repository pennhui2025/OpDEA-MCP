import copy
import numpy as np
import pandas as pd
from Utils import DEA_single_hurdle_new, get_multi_view_data
from Utils import seed_torch
import argparse
import os
from MV_regression_contrast_filter import get_MVcontrast_res
from itertools import combinations
from complementary_feature import get_fea_test
import joblib
from combination_selection import process_data, T_model

seed_torch()

def main():

    parser = argparse.ArgumentParser(description='run MCP for DEA')
    parser.add_argument('-f', '--file_folder', default='data/', #required=True,
                        help='The input file folder, should include all the view files, e.g. [dlfq.tsv, maxlfq.tsv,'
                             'top0.tsv, top3.tsv, count.tsv] for DDA data or [dlfq.tsv, maxlfq.tsv, top3.tsv, top1.tsv] '
                             'for DIA data. There must be a design.tsv file'
                        )
    parser.add_argument('-q', '--qval_threshold', default=0.05,
                        help='adj.p-value threshold for determining differential expression')
    parser.add_argument('-l', '--logfc_threshold', default=np.log2(1.5),
                        help='logFC threshold for determining differential expression')
    parser.add_argument('-p', '--platform', default='DIANN', #required=True,
                        help='quantification platform, e.g., FragPipe, Maxquant, DIANN, Spectronaut, unknow(blank)')
    parser.add_argument('-o', '--logTs', default="[T, F, F, F]", #required=True,
                        help='view_names for quantification, e.g., top0, maxlfq, top3, dlfq for DDA LFQ; top1, top3, maxlfq, dlfq DIA LFQ')
    parser.add_argument('-n', '--normalization', default='blank',
                        help='normalization method, can be center.median, center.mean, quantiles, vsn, ...')
    parser.add_argument('-i', '--imputation', default='blank',
                        help='imputation method, can be MinProb, QRILC, knn, ..., we currently recommend no imputation')
    parser.add_argument('-D', '--DEA', default='limma',
                        help='differential expression analysis tool, e.g., limma, ROTS, more tools will be added in our future version')
    parser.add_argument('-g1', '--group1', default='B', #required=True,
                        help='condition for contrast, e.g., A, control, should  be one of the conditions in design file')
    parser.add_argument('-g2', '--group2', default='E', #required=True,
                        help='another condition for contrast, e.g., D, test, should  be one of the conditions in design file')
    parser.add_argument('-R', '--Rscript', default='D:/software/R-4.3.1/bin/',
                        help='the path to Rscript.exe for running R packages to conduct differential expression analysis')
    parser.add_argument('-y', '--pos_ty', default='0_h',
                        help='the method for selecting reliable samples, "0_h" means the consistency between dlfq and hurdle; "0_1_2" means the consistency among all views')
    parser.add_argument('-g', '--pg_ty', default='remove',
                        help='the method for process proteingroups, we remove them by default, can be "remove", "conserve" and "none"')
    parser.add_argument('-t', '--trial', default=3,
                        help='trial number for the hyperparameter optimization')
    parser.add_argument('-s', '--save', default='res/',
                        help='result saving folder')
    parser.add_argument('-A', '--acq', default='DIA', #required=True,
                        help='acqusition type')
    parser.add_argument('-S', '--view_select', default='F',  # required=True,
                        help='whether conduct view combination selection, if N then use the best view combination per'
                             'our benchmarking results')

    args = parser.parse_args()

    current_path = os.path.dirname(os.path.abspath(__file__)) + '/'
    Rscript_path = args.Rscript # 'D:/software/R-4.3.1/bin/'
    python_path = ''

    logfc_thr = float(args.logfc_threshold) #np.log2(1.5)
    trial_num = int(args.trial) #30
    seed = 2023
    platform = args.platform #'Spectronaut'

    dea = args.DEA   #'limma'
    norm = args.normalization   #'blank'
    imput = args.imputation #'blank'
    thre = float(args.qval_threshold) #0.05
    acq = args.acq #'DIA'

    g1 = args.group1 #'N'
    g2 = args.group2 #'C'
    pos_ty = args.pos_ty #'0_h'
    pg_ty = args.pg_ty #'remove'
    cross_imp = 'none'
    postfix = 'test'

    view_select = args.view_select

    if acq == 'DDA':
        view_names_all = ['dlfq', 'maxlfq', 'top0', 'top3', 'count']
    elif acq == 'DIA':
        view_names_all = ['dlfq', 'maxlfq', 'top3', 'top1']

    design_path = args.file_folder + 'design.tsv'
    mat_folder = args.file_folder
    dt_name = 'case'
    logTs_all = args.logTs.strip('[]').split(',')
    name_postfix = '.tsv'
    R_fold = current_path #os.path.abspath(os.path.join(current_path, "..", "R_code")) + '/'
    save_folder = args.save

    if view_select == 'T':  # conduct view combination selection
        '''
            step 1: extract view_combination feature for selecting optimal view combination
            '''
        get_fea_test(Rscript_path, platform, g1, g2, acq, view_names_all, logTs_all, R_fold, dea, save_folder, mat_folder,
                     design_path, dt_name, name_postfix)

        '''
                step2: predict the performance of each view combination
                '''
        trained_model_path = 'trained_model_scale_T_trial305fold.joblib'
        bst = joblib.load(trained_model_path)

        test_data_raw = pd.read_csv(save_folder + 'complementary_feas_' + dt_name + '.csv', header=0)
        scale = 'T'
        final_test_pd, meta_info_test, label_test, data_test = process_data(test_data_raw, scale, '', '', dt_name)

        meta_info = meta_info_test
        test_data = data_test.values
        ty = '5fold'
        predict_performance = T_model(test_data, bst, meta_info, save_folder, dt_name, ty)

        best_views_str = predict_performance['view_name'].values[np.where((predict_performance['pred_mcc'].values == max(predict_performance['pred_mcc'].values)))[0][0]]
        view_names = best_views_str.split('_')[1:]

    else:

        if (acq == 'DDA') & (platform == 'FragPipe'):
            view_names = ['dlfq', 'count']
        elif (acq == 'DDA') & (platform == 'Maxquant'):
            view_names = ['dlfq', 'top0', 'top3']
        elif (acq == 'DIA') & (platform == 'DIANN'):
            view_names = ['dlfq', 'maxlfq', 'top1']
        elif (acq == 'DIA') & (platform == 'Spectronaut'):
            view_names = ['dlfq', 'maxlfq', 'top3']

    '''
        step3: DEA with the best view combination
        '''
    logTs = [logTs_all[view_names_all.index(i)] for i in view_names]
    deas = [dea] * len(view_names)
    norms = [norm] * len(view_names)
    imputs = [imput] * len(view_names)

    design_path_tr = design_path

    view_str = view_names[0]
    for vn in range(1, len(view_names)):
        view_str += ',' + view_names[vn]

    view_files = mat_folder + view_names[0] + '.tsv'
    for vn in range(1, len(view_names)):
        view_files += ',' + mat_folder + view_names[vn] + '.tsv'

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
    view_paths.update({'dataset': postfix})

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



    file_path = get_MVcontrast_res(views_union, res_union, logfc_thr, thre, ex_tys,
                       trial_num, pos_ty, pg_ty, g1, g2, view_paths, acq, postfix, save_folder)
    with open(save_folder + "file_path.txt", "w", encoding="utf-8") as f:
        f.write(view_str + '|' +file_path + '|' + save_folder + 'file_path.txt')
    print()

if __name__ == '__main__':

    main()
    # '''
    # !!! to reproduce the result, please replace following root_fold with your root for the Supplementary_code/
    # !!! also replace the value of "Rscript_path" with your address to "Rscript.exe" in your R installation folder
    # '''
    # current_path = os.path.dirname(os.path.abspath(__file__)) + '/'
    # Rscript_path = 'D:/software/R-4.3.1/bin/'
    # python_path = ''
    #
    # logfc_thr = np.log2(1.5)
    # hc_thr = 0.0001
    # npos_thr = 5
    # trial_num = 30
    # seed = 2023
    # diff_ty = 'auc'
    # platform = 'Spectronaut'
    #
    # dea = 'limma'
    # norm = 'blank'
    # imput = 'blank'
    # cbt = 'min'
    # thre = 0.05
    # acq = 'DIA'
    # if acq == 'DIA':
    #     logT = 'F'  # 'F'
    # elif acq == 'DDA':
    #     logT = 'T'
    #
    # g1 = 'N'
    # g2 = 'C'
    # pos_ty = '0_h'
    # pg_ty = 'remove'
    # cross_imp = 'none'
    # postfix = 'CRC'
    #
    # view_names_all = ['dlfq', 'maxlfq', 'top3', 'top1']
    # for view_num in range(2, len(view_names_all) + 1):
    #     combs = np.array(list(combinations([x for x in range(len(view_names_all))], view_num)))
    #     for comb_idx in range(len(combs)):
    #         view_idx = combs[comb_idx, :]
    #         view_names = list(np.array(view_names_all)[view_idx])
    #
    #         raw_file = ''
    #         evidence_file = ''
    #
    #         logTs = np.array(['F'] * len(view_names))
    #         idx = np.where((np.array(view_names) == 'dlfq'))[0]
    #         if len(idx) == 1:
    #             logTs[idx[0]] = 'F'
    #         logTs = list(logTs)
    #         deas = [dea] * len(view_names)
    #         norms = [norm] * len(view_names)
    #         imputs = [imput] * len(view_names)
    #
    #         save_data_root = current_path + '/res/'
    #         R_fold = os.path.abspath(os.path.join(current_path, "..", "R_code")) + '/'
    #         root_fold = os.path.abspath(os.path.join(current_path, "..", "")) + "/data/case_study_views/"
    #
    #         save_folder = save_data_root + postfix + '_DEA_res/'
    #
    #         view_str = view_names[0]
    #         for vn in range(1, len(view_names)):
    #             view_str += ',' + view_names[vn]
    #
    #         mat_folder = root_fold + '/CRC_views/'
    #         design_path = root_fold + '/CRC_views/design.tsv'
    #         design = pd.read_csv(design_path, sep='\t', header=0)
    #         design_value = design.values
    #         design_headers = np.array(design.columns.values)
    #
    #         design_path_tr = design_path
    #
    #         view_files = mat_folder + view_names[0] + '.tsv'
    #         for vn in range(1, len(view_names)):
    #             view_files += ',' + mat_folder + view_names[vn] + '.tsv'
    #
    #         (views, view_datas, views_intersect, view_datas_intersect, idx_g1g2s, view_paths_dt, dataset, views_ci,
    #          view_datas_ci,
    #          view_paths_ci, idx_g1g2s_ci, out_design, view_logFCs,
    #          view_confis, view_logFCs_intersect, view_confis_intersect, views_union, view_datas_union,
    #          view_logFCs_union, view_confis_union) = get_multi_view_data(
    #             'views',
    #             pg_ty,
    #             view_files, g1,
    #             g2, design_path_tr, view_str,
    #             Rscript_path, R_fold, python_path, acq, platform,
    #             save_folder, logTs, cross_imp, seed)
    #
    #         view_paths = copy.deepcopy(view_paths_dt)
    #         for v in range(len(view_str.split(','))):
    #             view_paths.update({view_str.split(',')[v]: view_files.split(',')[v]})
    #         view_paths.update({'save_folder': save_folder})
    #         view_paths.update({'Rscript_path': Rscript_path})
    #         view_paths.update({'R_fold': R_fold})
    #         view_paths.update({'true_organism': '_'})
    #         view_paths.update({'DE_organism': '_'})
    #         view_paths.update({'true_lgfc': '-'})
    #         view_paths.update({'dataset': postfix})
    #
    #         view_paths.update({'design': view_paths_dt['design'], 'logT': view_paths_dt['logT']})
    #
    #         ex_tys = view_str.split(',')
    #         logTs = view_paths['logT']
    #
    #         res_union = DEA_single_hurdle_new(views_union, view_datas_union,
    #                                           idx_g1g2s, ex_tys,
    #                                           view_paths, logTs,
    #                                           g1, g2, norms,
    #                                           imputs, deas,
    #                                           save_folder,
    #                                           view_paths['dataset'], 'union', R_fold,
    #                                           platform)
    #
    #         get_MVcontrast_res(views_union, res_union, logfc_thr, thre, ex_tys,
    #                            trial_num, pos_ty, pg_ty, g1, g2, view_paths, acq, postfix, save_folder)



