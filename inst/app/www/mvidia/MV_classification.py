import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import cross_val_score, ShuffleSplit, StratifiedShuffleSplit, StratifiedKFold
from xgboost import XGBClassifier, XGBClassifier
from sklearn.metrics import roc_auc_score
from cca_zoo.linear import GCCA
from sklearn import preprocessing
import optuna
import random
import os
from sklearn.metrics import accuracy_score as acc, recall_score as recall, precision_score as precision, f1_score as f1, matthews_corrcoef as mcc
import matplotlib.pyplot as plt
from sklearn.metrics import RocCurveDisplay, roc_curve
from sklearn.metrics import auc as Auc


def process_proteingroup(view, pg_ty):
    '''

        Parameters
        ----------
        view: the view matrix
        pg_ty: processing protein groups with more than 2 proteins, 'remove': totally remove, 'conserve': retain those with
               proteingroups without protein members having being reported independently, 'none': no action

        Returns: view matrix with protein groups processed based on pg_ty
        -------

        '''
    proteins = view['Protein'].values
    pro_num = np.array([len(pro.replace(',', ';').replace(' ', '').split(';')) for pro in proteins])
    uni_pro = proteins[np.where((pro_num == 1))[0]] # pg with 1 protein
    ab_pro = proteins[np.where((pro_num > 1))[0]] # pg with more than 2 protein

    ab_ty = np.array([1 if len(np.intersect1d(uni_pro, pro.replace(',', ';').replace(' ', '').split(';'))) == 0
                      else 0 for pro in ab_pro])

    if pg_ty == 'remove':
        view_new = view.loc[np.where((pro_num == 1))[0], :]#(np.where((pro_num == 1))[0])
    elif pg_ty == 'conserve':
        retain_idx = np.array(list(np.where((pro_num == 1))[0]) + list(np.where((pro_num > 1))[0][np.where((ab_ty==1))[0]]))
        view_new = view.loc[retain_idx, :]#(retain_idx)
    else:
        view_new = view

    return view_new

def get_multi_view_data(in_ty, pg_ty, file_pth, design, view_names, R_fold, Rcode, python_path, acq, platform,
                        save_folder, logT):

    '''

    Parameters
    ----------
    in_ty: input type, 'raw' for raw quantification outputs, 'views' for extracted matrices for each views
    pg_ty: proteingroups process type, 'remove' for removing proteingroups with more than two proteins, 'conserve' for
           retaining the proteingroups with proteins having not been reported independently, 'none' for no action
    file_pth: input quantification files, if in_ty==raw, file_pth = path to raw + ',' path to evidence;
              for in_ty == 'views', file_pth = view1 + ',' + view2 + ',' + ..., view1, view2, ... should in the same
             order as in view names

    design: experiment design file with sample name, sample ID, condition, replicate information being specified
    view_names: names of the views, e.g., dlfq for directLFQ, maxlfq for MaxLFQ, top0, top1, top3,.... concated with ','
    R_fold: the path to RScript.exe
    Rcode: the path for R scripts used for DEA and preprocessing
    python_path: the path for installing directlfq package
    acq: acqusition mode, e.g., DDA, DIA
    platform: quantification platform, e.g., FragPipe, Maxquant, DIANN, Spectronaut
    save_folder: folder for saving results
    logT: whether conduct log2 transforming on the matrix of each view, a list with the same length as the view number
    cross_imp: whether conduct cross-view imputation with given CCA type, e.g., 'MCCA', or 'none' for no imputation

    Returns: view matrices for following DEA
    -------

    '''

    if in_ty == 'raw': # input raw quantification outputs and the evidence, e.g., combined_protein.tsv and combined_ion.tsv
        files = file_pth.split(',')
        raw_file = files[0]
        evidence_file = files[1]

        R_para = platform + ' ' + python_path + ' ' + raw_file + ' ' + evidence_file + ' ' + design + ' ' + view_names
        print(R_fold + 'Rscript ' + Rcode + 'get_matrix.R ' + R_para)
        os.system(R_fold + 'Rscript ' + Rcode + 'get_matrix.R ' + R_para)

        view_names = view_names.split(',')
        view_paths = {}
        headers = []

        for view in view_names:
            mat = pd.read_csv(save_folder + 'dt/' + view + '.tsv', sep='\t', header=0)
            header = np.array(mat.columns.values.tolist())
            view_paths.update({view: mat})
            headers.append(header)

        if acq == 'DDA': # DDA data extracted matrices are always without log transforming
            logT = ['T'] * len(view_names)
            if 'count' in view_names:
                logT[np.where((np.array(view_names) == 'count'))[0][0]] = 'F'
            view_paths.update({'logT': logT})

        elif acq == 'DIA': # DIA data extracted matrices always having been log transformed except for dlfq (will transform below)
            logT = ['F'] * len(view_names)
            if 'dlfq' in view_names:
                logT[np.where((np.array(view_names) == 'dlfq'))[0][0]] = 'T'

            view_paths.update({'logT': logT})

    elif in_ty == 'views':

        view_files = file_pth.split(',')
        view_names = view_names.split(',')

        view_paths = {}
        headers = []

        for i in range(len(view_names)):
            mat = pd.read_csv(view_files[i], sep='\t', header=0)
            header = np.array(mat.columns.values.tolist())
            view_paths.update({view_names[i]: mat})

            headers.append(header)

        view_paths.update({'logT': logT})

    views = []
    comm_pro = []
    union_pro = []
    for v in range(len(view_names)):
        view_new = process_proteingroup(view_paths[view_names[v]], pg_ty)
        if logT[v] == 'T':
            if 'Organism' in view_new.columns.values.tolist():
                exps = np.array(view_new.values[:, 2:len(view_new.values[0, :])], dtype='float')
                info = np.array(view_new.values[:, 0:2])
            else:
                exps = np.array(view_new.values[:, 1:len(view_new.values[0, :])], dtype='float')
                info = np.array(view_new.values[:, 0:1])
            exps[exps == 0] = np.nan
            exps = np.log2(exps)
            view_new = pd.DataFrame(np.column_stack((info, exps)), columns=view_new.columns.values.tolist())
            logT[v] = 'F'

        ##### make sure the samples are ordered consistently

        view_paths.update({view_names[v]: view_new})
        views.append(view_new)

        if len(comm_pro) == 0:
            comm_pro = view_new['Protein']
        else:
            comm_pro = np.intersect1d(comm_pro, view_new['Protein'])

        if 'Organism' in view_new.columns.values.tolist():
            if len(union_pro) == 0:
                union_pro = view_new.values[:, 0:2]
            else:
                union_pro = np.row_stack((union_pro, view_new.values[:, 0:2]))
        else:
            if len(union_pro) == 0:
                union_pro = view_new.values[:, 0:1]
            else:
                union_pro = np.row_stack((union_pro, view_new.values[:, 0:1]))

    view_paths.update({'design_file_path': design})
    designs = pd.read_csv(design, header=0, sep='\t')

    samples = designs['sample_name'].values

    idx_g1g2s = []
    view_datas = []
    view_logFCs = []
    view_confis = []

    for v in range(len(headers)):
        header = headers[v]
        idx_g1g2 = []
        for sample in samples:
            for i in range(len(header)):
                header_pure = header[i].replace('.Spectral.Count', '').replace('.MaxLFQ.Intensity', '').replace(
                    '.Intensity', '')
                header_pure = header_pure.replace('MS.MS.count.', '').replace('Intensity.', '').replace(
                    'LFQ.intensity.', '').replace('Top3.', '')
                if sample == header_pure:
                    idx_g1g2.append(i)

        ##### make sure the samples are ordered consistently
        view_data = np.array(views[v].values[:, idx_g1g2], dtype='float')

        if 'Organism' in header:
            info = np.array(views[v].values[:, 0:2])
            info_name = list(np.array(header)[0:2])
        else:
            info = np.array(views[v].values[:, 0:1])
            info_name = list(np.array(header)[0:1])
        #idx_g1g2s.append(idx_g1g2)
        view_datas.append(view_data)
        views[v] = pd.DataFrame(np.column_stack((info, view_data)),
                                columns=info_name + list(samples))

        if 'Organism' in header:
            idx_g1g2s.append(np.array([i for i in range(2, len(idx_g1g2) + 2)], dtype='int'))
        else:
            idx_g1g2s.append(np.array([i for i in range(1, len(idx_g1g2) + 1)], dtype='int'))

    views_ci, view_datas_ci, view_paths_ci, idx_g1g2s_ci = [], [], [], []

    view_datas_intersect = []
    views_intersect = []
    view_logFCs_intersect = []
    view_confis_intersect = []

    view_datas_union = []
    views_union = []
    view_logFCs_union = []
    view_confis_union = []
    unique_union_pro, idx = np.unique(union_pro[:, 0], return_index=True)
    union_pro = union_pro[idx, :]

    for i in range(len(views)):
        comm_pros, idx1, idx2 = np.intersect1d(views[i]['Protein'], comm_pro, return_indices=True)
        views_intersect.append(pd.DataFrame(views[i].values[idx1, :], columns=list(views[i].columns.values)))

        view_data = np.array(view_datas[i][idx1, :], dtype='float')

        view_datas_intersect.append(view_data)

        view_data_union = np.zeros((len(union_pro), len(view_datas[i][0, :])))

        comm_pros, idx1, idx2 = np.intersect1d(views[i]['Protein'], union_pro[:, 0], return_indices=True)
        view_data_union[idx2, :] = view_datas[i][idx1, :]
        view_data_union[view_data_union==0]=np.nan
        view_datas_union.append(view_data_union)
        views_union.append(
            pd.DataFrame(np.column_stack((union_pro, view_data_union)), columns=list(views[i].columns.values)))

    out_design = pd.DataFrame(designs.values, columns=designs.columns.values.tolist())

    return (views, view_datas, views_intersect, view_datas_intersect, idx_g1g2s, view_paths, 'dt', views_ci,
            view_datas_ci, view_paths_ci, idx_g1g2s_ci, out_design, view_logFCs,
            view_confis, view_logFCs_intersect, view_confis_intersect, views_union, view_datas_union,
            view_logFCs_union, view_confis_union)


def average_replicate(mat, design):

    uni_pat = np.unique(design['sample_ID'].values)

    idx_avg = []
    avg_intens = np.zeros((len(mat.values[:, 0]), 0))
    # label = []
    for i in range(len(uni_pat)):
        pat = uni_pat[i]

        idx = np.where((design['sample_ID'].values == pat))[0]+2
        idx_avg.append(idx[0]-2)

        mat_pat = mat.iloc[:, idx]

        inten = np.array(mat_pat.values, dtype='float')
        avg_inten = np.nanmean(inten, axis=1)
        avg_intens = np.column_stack((avg_intens, avg_inten))

    avg_mat = pd.DataFrame(np.column_stack((mat.values[:, 0:2], avg_intens)), columns=list(mat.columns.values[0:2]) + list(uni_pat))
    design_avg = design.iloc[idx_avg, :]
    return avg_mat, design_avg

def get_pro_list(matrix):
    pros = matrix['Protein'].values
    pros_id = []
    for i in range(len(pros)):
        pro = pros[i].split(';')
        pro_id = ''
        if len(pro) == 1:
            pro_id = pro[0].split('|')[1]
        elif len(pro) > 1:
            for j in range(len(pro)):
                id = pro[j].split('|')[1]
                if pro_id == '':
                    pro_id = id
                else:
                    pro_id += ';' + id

        pros_id.append(pro_id)
    return pros_id

def get_pro_inten(pro_list, matrix, design, logT='F'):

    pros_id = matrix['Protein'].values
    # pros_id = []
    # for i in range(len(pros)):
    #     pro = pros[i].split(';')
    #     pro_id = ''
    #     if len(pro) == 1:
    #         pro_id = pro[0].split('|')[1]
    #     elif len(pro) > 1:
    #         for j in range(len(pro)):
    #             id = pro[j].split('|')[1]
    #             if pro_id == '':
    #                 pro_id = id
    #             else:
    #                 pro_id += ';' + id
    #
    #     pros_id.append(pro_id)
    #
    # pros_id = np.array(pros_id)

    com, idx1, idx2 = np.intersect1d(np.array(pro_list), pros_id, return_indices= True)

    mat_all = np.array(matrix.values[:, 2:len(matrix.values[0, :])], dtype='float')
    #mat_pro = np.zeros((len(pro_list), len(mat_all[0, :])))
    #mat_pro[idx1, :] = mat_all[idx2, :]
    mat_pro = mat_all[idx2, :]
    out_mat = matrix.iloc[idx2]

    # header = np.array(matrix.columns.values)[2:len(matrix.columns.values)]
    # labels = []
    # idxs = []
    # non_find = []
    # samples = []
    #
    # for i in range(len(header)):
    #
    #     sample = (header[i].replace('C:\\Users\\10352\Desktop\\TPD_Discovery\\', '').replace('.mzML.dia', '')
    #               .replace('.mzML', '').replace('.', '_').replace('-', '_').replace('batch', 'b').
    #               replace('TPD_b', 'TPD_DIA_b').replace('(liu)','')).split('_')
    #     sample_name = ''
    #     for t in range(5):
    #         #print(sample)
    #         if sample_name == '':
    #             sample_name = sample[t]
    #         else:
    #             sample_name = sample_name + '_' + sample[t]
    #     idx = np.where((design['MS_file_name'].values == sample_name))[0]
    #     if len(idx) == 1:
    #         labels.append(design['Classificaiton_type'].values[idx[0]])
    #         idxs.append(idx[0])
    #         samples.append(design['MS_file_name'].values[idx[0]])
    #
    #     else:
    #
    #         non_find.append(header[i])
    # #design_final = design.iloc[idxs]
    # labels = np.array(labels)
    #
    # comm, idx11, idx12 = np.intersect1d(np.array(samples), design['MS_file_name'].values, return_indices=True)
    # mat_pro = mat_pro[:, idx11]
    # design_final = design.iloc[idx12]
    #
    if logT == 'T':
        mat_pro[mat_pro == 0] = np.nan
        mat_pro = np.log2(mat_pro)

        out_mat_inten = np.array(out_mat.values[:, 2:len(out_mat.values[0, :])], dtype='float')
        out_mat_inten[out_mat_inten == 0] = np.nan
        out_mat_inten = np.log2(out_mat_inten)
        out_mat = pd.DataFrame(np.column_stack((out_mat.values[:, 0:2], out_mat_inten)),
                               columns=list(out_mat.columns.values))

    # out_mat = out_mat.iloc[:, list([0, 1]) + list(idx11 + 2)]

    avg_mat, design_avg = average_replicate(out_mat, design)
    if 'condition' in design_avg.keys():
        labels = design_avg['condition'].values
    else:
        labels = np.array([0]*len(design_avg['sample_name'].values))
    return mat_pro, out_mat, labels, design, avg_mat, design_avg

def get_pro_inten_tr(pro_list, matrix, design, logT='F'):

    pros = matrix['Protein'].values
    pros_id = []
    for i in range(len(pros)):
        pro = pros[i].split(';')
        pro_id = ''
        if len(pro) == 1:
            pro_id = pro[0].split('|')[1]
        elif len(pro) > 1:
            for j in range(len(pro)):
                id = pro[j].split('|')[1]
                if pro_id == '':
                    pro_id = id
                else:
                    pro_id += ';' + id

        pros_id.append(pro_id)

    pros_id = np.array(pros_id)

    com, idx1, idx2 = np.intersect1d(np.array(pro_list), pros_id, return_indices= True)

    mat_all = np.array(matrix.values[:, 2:len(matrix.values[0, :])], dtype='float')
    #mat_pro = np.zeros((len(pro_list), len(mat_all[0, :])))
    #mat_pro[idx1, :] = mat_all[idx2, :]
    mat_pro = mat_all[idx2, :]
    out_mat = matrix.iloc[idx2]

    header = np.array(matrix.columns.values)[2:len(matrix.columns.values)]
    labels = []
    idxs = []
    non_find = []
    samples = []

    for i in range(len(header)):

        sample = (header[i].replace('C:\\Users\\10352\Desktop\\TPD_Discovery\\', '').replace('.mzML.dia', '')
                  .replace('.mzML', '').replace('.', '_').replace('-', '_').replace('batch', 'b').
                  replace('TPD_b', 'TPD_DIA_b').replace('(liu)','')).split('_')
        sample_name = ''
        for t in range(5):
            #print(sample)
            if sample_name == '':
                sample_name = sample[t]
            else:
                sample_name = sample_name + '_' + sample[t]
        idx = np.where((design['MS_file_name'].values == sample_name))[0]
        if len(idx) == 1:
            labels.append(design['Classificaiton_type'].values[idx[0]])
            idxs.append(idx[0])
            samples.append(design['MS_file_name'].values[idx[0]])

        else:

            non_find.append(header[i])
    #design_final = design.iloc[idxs]
    labels = np.array(labels)

    comm, idx11, idx12 = np.intersect1d(np.array(samples), design['MS_file_name'].values, return_indices=True)
    mat_pro = mat_pro[:, idx11]
    design_final = design.iloc[idx12]

    if logT == 'T':
        mat_pro[mat_pro == 0] = np.nan
        mat_pro = np.log2(mat_pro)

        out_mat_inten = np.array(out_mat.values[:, 2:len(out_mat.values[0, :])], dtype='float')
        out_mat_inten[out_mat_inten == 0] = np.nan
        out_mat_inten = np.log2(out_mat_inten)
        out_mat = pd.DataFrame(np.column_stack((out_mat.values[:, 0:2], out_mat_inten)),
                               columns=list(out_mat.columns.values))

    out_mat = out_mat.iloc[:, list([0, 1]) + list(idx11 + 2)]

    avg_mat, design_avg = average_replicate(out_mat, design_final)
    return mat_pro, out_mat, labels[idx11], design_final, avg_mat, design_avg

def get_pro_inten_te(pro_list, matrix, design, logT='F'):

    pros = matrix['Protein'].values
    pros_id = []
    for i in range(len(pros)):
        pro = pros[i].split(';')
        pro_id = ''
        if len(pro) == 1:
            pro_id = pro[0].split('|')[1]
        elif len(pro) > 1:
            for j in range(len(pro)):
                id = pro[j].split('|')[1]
                if pro_id == '':
                    pro_id = id
                else:
                    pro_id += ';' + id

        pros_id.append(pro_id)

    pros_id = np.array(pros_id)

    com, idx1, idx2 = np.intersect1d(np.array(pro_list), pros_id, return_indices= True)
    np.setdiff1d(np.array(pro_list), pros_id)
    mat_all = np.array(matrix.values[:, 2:len(matrix.values[0, :])], dtype='float')
    # mat_pro = np.zeros((len(pro_list), len(mat_all[0, :])))
    # mat_pro[idx1, :] = mat_all[idx2, :]
    mat_pro = mat_all[idx2, :]
    out_mat = matrix.iloc[idx2]

    header = np.array(matrix.columns.values)[2:len(matrix.columns.values)]
    labels = []
    idxs = []
    non_find = []
    samples = []

    for i in range(len(header)):

        sample = (header[i].replace('D:\\', '').replace('.mzML.dia', '')
                  .replace('.mzML', '').replace('.', '_').replace('-', '_'))

        idx = np.where((design['MS_file_name'].values == sample))[0]
        if len(idx) == 1:
            labels.append(design['Classificaiton_type'].values[idx[0]])
            idxs.append(idx[0])
            samples.append(design['MS_file_name'].values[idx[0]])

        else:

            non_find.append(header[i])
    #design_final = design.iloc[idxs]
    labels = np.array(labels)

    comm, idx11, idx12 = np.intersect1d(np.array(samples), design['MS_file_name'].values, return_indices=True)
    mat_pro = mat_pro[:, idx11]
    design_final = design.iloc[idx12]


    if logT == 'T':
        mat_pro[mat_pro==0] = np.nan
        mat_pro = np.log2(mat_pro)

        out_mat_inten = np.array(out_mat.values[:, 2:len(out_mat.values[0, :])], dtype='float')
        out_mat_inten[out_mat_inten==0] = np.nan
        out_mat_inten = np.log2(out_mat_inten)
        out_mat = pd.DataFrame(np.column_stack((out_mat.values[:, 0:2], out_mat_inten)), columns=list(out_mat.columns.values))

    out_mat = out_mat.iloc[:, list([0, 1]) + list(idx11 + 2)]

    avg_mat, design_avg = average_replicate(out_mat, design_final)
    return mat_pro, out_mat, labels[idx11], design_final, avg_mat, design_avg

def preprocess_multi_view(pro_list):

    folder = 'D:/software/download/PXD036554/Thyroid/'

    dlfq_tr_pth = folder + 'TPD/TPD_Discovery_LibSearch/dlfq.tsv'
    maxlfq_tr_pth = folder + 'TPD/TPD_Discovery_LibSearch/maxlfq.tsv'
    top3_tr_pth = folder + 'TPD/TPD_Discovery_LibSearch/top3.tsv'

    dlfq_te_pth = folder + 'TPD/TPD_Retrospective_LibSearch/dlfq.tsv'
    maxlfq_te_pth = folder + 'TPD/TPD_Retrospective_LibSearch/maxlfq.tsv'
    top3_te_pth = folder + 'TPD/TPD_Retrospective_LibSearch/top3.tsv'

    design_tr_pth = folder + 'TPD_diann_protMatrix_20230207_lab_dis.csv'
    design_te_pth = folder + 'TPD_diann_protMatrix_20230207_lab_ret.csv'

    maxlfq_tr = pd.read_csv(maxlfq_tr_pth, sep='\t', header=0)
    dlfq_tr = pd.read_csv(dlfq_tr_pth, sep='\t', header=0)
    top3_tr = pd.read_csv(top3_tr_pth, sep='\t', header=0)

    maxlfq_te = pd.read_csv(maxlfq_te_pth, sep='\t', header=0)
    dlfq_te = pd.read_csv(dlfq_te_pth, sep='\t', header=0)
    top3_te = pd.read_csv(top3_te_pth, sep='\t', header=0)

    design_tr = pd.read_csv(design_tr_pth, sep=',', header=0)
    design_te = pd.read_csv(design_te_pth, sep=',', header=0)

    # pros_tr_maxlfq = get_pro_list(maxlfq_tr)
    # pros_tr_dlfq = get_pro_list(dlfq_tr)
    # pros_tr_top3 = get_pro_list(top3_tr)
    #
    # pros_te_maxlfq = get_pro_list(maxlfq_te)
    # pros_te_dlfq = get_pro_list(dlfq_te)
    # pros_te_top3 = get_pro_list(top3_te)

    pro_maxlfq_tr, mat_maxlfq_tr, labels_maxlfq_tr, design_maxlfq_tr, avg_mat_maxlfq_tr, avg_design_maxlfq_tr = get_pro_inten_tr(pro_list, maxlfq_tr, design_tr)
    # avg_mat_maxlfq_tr.to_csv(folder + 'avg_maxlfq_dis.csv')
    # avg_design_maxlfq_tr.to_csv(folder + 'avg_design_dis.csv')
    pro_dlfq_tr, mat_dlfq_tr, labels_dlfq_tr, design_dlfq_tr, avg_mat_dlfq_tr, avg_design_dlfq_tr = get_pro_inten_tr(pro_list, dlfq_tr, design_tr, logT='T')
    # avg_mat_dlfq_tr.to_csv(folder + 'avg_dlfq_dis.csv')
    pro_top3_tr, mat_top3_tr, labels_top3_tr, design_top3_tr, avg_mat_top3_tr, avg_design_top3_tr = get_pro_inten_tr(pro_list, top3_tr, design_tr)
    # avg_mat_top3_tr.to_csv(folder + 'avg_top3_dis.csv')

    pro_maxlfq_te, mat_maxlfq_te, labels_maxlfq_te, design_maxlfq_te, avg_mat_maxlfq_te, avg_design_maxlfq_te = get_pro_inten_te(pro_list, maxlfq_te, design_te)
    # avg_mat_maxlfq_te.to_csv(folder + 'avg_maxlfq_ret.csv')
    # avg_design_maxlfq_te.to_csv(folder + 'avg_design_te.csv')
    pro_dlfq_te, mat_dlfq_te, labels_dlfq_te, design_dlfq_te, avg_mat_dlfq_te, avg_design_dlfq_te = get_pro_inten_te(pro_list, dlfq_te, design_te, logT='T')
    # avg_mat_dlfq_te.to_csv(folder + 'avg_dlfq_ret.csv')
    pro_top3_te, mat_top3_te, labels_top3_te, design_top3_te, avg_mat_top3_te, avg_design_top3_te = get_pro_inten_te(pro_list, top3_te, design_te)
    # avg_mat_top3_te.to_csv(folder + 'avg_top3_ret.csv')

    tr_data = {'dlfq':{'data':avg_mat_dlfq_tr, 'design':avg_design_dlfq_tr},
                'maxlfq':{'data': avg_mat_maxlfq_tr, 'design':avg_design_maxlfq_tr},
                'top3':{'data': avg_mat_top3_tr, 'design':avg_design_top3_tr}}

    te_data = {'dlfq': {'data': avg_mat_dlfq_te, 'design': avg_design_dlfq_te},
                'maxlfq': {'data': avg_mat_maxlfq_te, 'design': avg_design_maxlfq_te},
                'top3': {'data': avg_mat_top3_te, 'design': avg_design_top3_te}}

    return tr_data, te_data

def preprocess_multi_view_new(pro_list, train_data, test_data):

    # folder = 'D:/software/download/PXD036554/Thyroid/'
    #
    # dlfq_tr_pth = folder + 'TPD/TPD_Discovery_LibSearch/dlfq.tsv'
    # maxlfq_tr_pth = folder + 'TPD/TPD_Discovery_LibSearch/maxlfq.tsv'
    # top3_tr_pth = folder + 'TPD/TPD_Discovery_LibSearch/top3.tsv'
    #
    # dlfq_te_pth = folder + 'TPD/TPD_Retrospective_LibSearch/dlfq.tsv'
    # maxlfq_te_pth = folder + 'TPD/TPD_Retrospective_LibSearch/maxlfq.tsv'
    # top3_te_pth = folder + 'TPD/TPD_Retrospective_LibSearch/top3.tsv'
    #
    # design_tr_pth = folder + 'TPD_diann_protMatrix_20230207_lab_dis.csv'
    # design_te_pth = folder + 'TPD_diann_protMatrix_20230207_lab_ret.csv'
    #
    v1_tr = train_data[16][0]
    v2_tr = train_data[16][1]
    v3_tr = train_data[16][2]

    v1_te = test_data[16][0]
    v2_te = test_data[16][1]
    v3_te = test_data[16][2]

    design_tr = train_data[11]
    design_te = test_data[11]

    # pros_tr_maxlfq = get_pro_list(maxlfq_tr)
    # pros_tr_dlfq = get_pro_list(dlfq_tr)
    # pros_tr_top3 = get_pro_list(top3_tr)
    #
    # pros_te_maxlfq = get_pro_list(maxlfq_te)
    # pros_te_dlfq = get_pro_list(dlfq_te)
    # pros_te_top3 = get_pro_list(top3_te)

    pro_maxlfq_tr, mat_maxlfq_tr, labels_maxlfq_tr, design_maxlfq_tr, avg_mat_maxlfq_tr, avg_design_maxlfq_tr = get_pro_inten(pro_list, v2_tr, design_tr)
    # avg_mat_maxlfq_tr.to_csv(folder + 'avg_maxlfq_dis.csv')
    # avg_design_maxlfq_tr.to_csv(folder + 'avg_design_dis.csv')
    pro_dlfq_tr, mat_dlfq_tr, labels_dlfq_tr, design_dlfq_tr, avg_mat_dlfq_tr, avg_design_dlfq_tr = get_pro_inten(pro_list, v1_tr, design_tr)
    # avg_mat_dlfq_tr.to_csv(folder + 'avg_dlfq_dis.csv')
    pro_top3_tr, mat_top3_tr, labels_top3_tr, design_top3_tr, avg_mat_top3_tr, avg_design_top3_tr = get_pro_inten(pro_list, v3_tr, design_tr)
    # avg_mat_top3_tr.to_csv(folder + 'avg_top3_dis.csv')

    pro_maxlfq_te, mat_maxlfq_te, labels_maxlfq_te, design_maxlfq_te, avg_mat_maxlfq_te, avg_design_maxlfq_te = get_pro_inten(pro_list, v2_te, design_te)
    # avg_mat_maxlfq_te.to_csv(folder + 'avg_maxlfq_ret.csv')
    # avg_design_maxlfq_te.to_csv(folder + 'avg_design_te.csv')
    pro_dlfq_te, mat_dlfq_te, labels_dlfq_te, design_dlfq_te, avg_mat_dlfq_te, avg_design_dlfq_te = get_pro_inten(pro_list, v1_te, design_te)
    # avg_mat_dlfq_te.to_csv(folder + 'avg_dlfq_ret.csv')
    pro_top3_te, mat_top3_te, labels_top3_te, design_top3_te, avg_mat_top3_te, avg_design_top3_te = get_pro_inten(pro_list, v3_te, design_te)
    # avg_mat_top3_te.to_csv(folder + 'avg_top3_ret.csv')

    tr_data = {'v1':{'data':avg_mat_dlfq_tr, 'design':avg_design_dlfq_tr},
                'v2':{'data': avg_mat_maxlfq_tr, 'design':avg_design_maxlfq_tr},
                'v3':{'data': avg_mat_top3_tr, 'design':avg_design_top3_tr}}

    te_data = {'v1': {'data': avg_mat_dlfq_te, 'design': avg_design_dlfq_te},
                'v2': {'data': avg_mat_maxlfq_te, 'design': avg_design_maxlfq_te},
                'v3': {'data': avg_mat_top3_te, 'design': avg_design_top3_te}}

    return tr_data, te_data

def xg_test_new(train_fea, test_fea, train_lab, test_lab, mean_fpr, ty='single'):
    if ty == 'mv':
        seed_torch(2024)

        kmcca = GCCA(latent_dimensions=30, random_state=2024)
        lat_out = kmcca.fit_transform(train_fea)
        lats_out = lat_out[0]
        for i in range(1, len(lat_out)):
            lats_out = np.column_stack((lats_out, lat_out[i]))

        lat_test = kmcca.transform(test_fea)
        lats_out_test = lat_test[0]
        for i in range(1, len(lat_test)):
            lats_out_test = np.column_stack((lats_out_test, lat_test[i]))

        train_fea = lats_out
        test_fea = lats_out_test

    #bst = XGBClassifier(objective='binary:logistic', scale_pos_weight=len(np.where((train_lab==0))[0]) / len(np.where((train_lab==1))[0]))
    bst = XGBClassifier(objective='binary:logistic', device='cuda', num_round=100)
    #bst = KNN()
    bst.fit(train_fea, train_lab)
    preds = bst.predict_proba(test_fea)
    pred_lab = bst.predict(test_fea)
    auc = roc_auc_score(test_lab[:, 0], preds[:, 1])


    fpr, tpr, thresholds = roc_curve(test_lab[:, 0], preds[:, 1], pos_label=1)

    interp_tpr = np.interp(mean_fpr, fpr, tpr)
    interp_tpr[0] = 0.0

    Acc = acc(test_lab[:, 0], pred_lab)
    Prec = precision(test_lab[:, 0], pred_lab)
    Rec = recall(test_lab[:, 0], pred_lab)
    F1 = f1(test_lab[:, 0], pred_lab)
    Mcc = mcc(test_lab[:, 0], pred_lab)

    return [auc, Acc, Prec, Rec, F1, Mcc], interp_tpr, auc, preds[:, 1], pred_lab

def xg_test_extend(train_fea, test_fea, train_lab, test_lab, mean_fpr, extend_true_lab, extend_score, extend_lab):

    bst = XGBClassifier(objective='binary:logistic')
    #bst = KNN()
    bst.fit(train_fea, train_lab)
    preds = bst.predict_proba(test_fea)
    pred_lab = bst.predict(test_fea)

    pred_prob = np.array(list(extend_score) + list(preds[:, 1]))
    pred_lab = np.array(list(extend_lab) + list(pred_lab))
    auc = roc_auc_score(np.array(list(extend_true_lab[:, 0]) + list(test_lab[:, 0])), pred_prob)

    fpr, tpr, thresholds = roc_curve(np.array(list(extend_true_lab[:, 0]) + list(test_lab[:, 0])), pred_prob, pos_label=1)

    interp_tpr = np.interp(mean_fpr, fpr, tpr)
    interp_tpr[0] = 0.0

    Acc = acc(np.array(list(extend_true_lab[:, 0]) + list(test_lab[:, 0])), pred_lab)
    Prec = precision(np.array(list(extend_true_lab[:, 0]) + list(test_lab[:, 0])), pred_lab)
    Rec = recall(np.array(list(extend_true_lab[:, 0]) + list(test_lab[:, 0])), pred_lab)
    F1 = f1(np.array(list(extend_true_lab[:, 0]) + list(test_lab[:, 0])), pred_lab)
    Mcc = mcc(np.array(list(extend_true_lab[:, 0]) + list(test_lab[:, 0])), pred_lab)

    return [auc, Acc, Prec, Rec, F1, Mcc], interp_tpr, auc, pred_prob, pred_lab

def xg_test_ens(train_fea, test_fea, train_lab, test_lab, mean_fpr):

    preds = []
    pred_labs = []
    for i in range(len(train_fea)):
        bst = XGBClassifier(objective='binary:logistic', device='cuda', num_round=100)
        #bst = KNN()
        bst.fit(train_fea[i], train_lab)
        pred = bst.predict_proba(test_fea[i])
        pred_lab = bst.predict(test_fea[i])
        preds.append(pred[:, 1])
        pred_labs.append(pred_lab)

    preds = np.mean(np.array(preds),axis=0)
    pred_labs = np.array(pred_labs).T

    pred_lab1 = np.array([1 if preds[j]>0.5 else 0 for j in range(len(preds))])
    pred_lab2 = np.array([1 if sum(pred_labs[j, :])>1 else 0 for j in range(len(pred_labs))])

    auc = roc_auc_score(test_lab[:, 0], preds)
    #mean_fpr = np.linspace(0, 1, 100)
    fpr, tpr, thresholds = roc_curve(test_lab[:, 0], preds, pos_label=1)

    interp_tpr = np.interp(mean_fpr, fpr, tpr)
    interp_tpr[0] = 0.0

    Acc1 = acc(test_lab[:, 0], pred_lab1)
    Prec1 = precision(test_lab[:, 0], pred_lab1)
    Rec1 = recall(test_lab[:, 0], pred_lab1)
    F11 = f1(test_lab[:, 0], pred_lab1)
    Mcc1 = mcc(test_lab[:, 0], pred_lab1)

    Acc2 = acc(test_lab[:, 0], pred_lab2)
    Prec2 = precision(test_lab[:, 0], pred_lab2)
    Rec2 = recall(test_lab[:, 0], pred_lab2)
    F12 = f1(test_lab[:, 0], pred_lab2)
    Mcc2 = mcc(test_lab[:, 0], pred_lab2)

    return [auc, Acc1, Prec1, Rec1, F11, Mcc1, auc, Acc2, Prec2, Rec2, F12, Mcc2], interp_tpr, auc

def get_data_lab(dict, view, pos):

    data_dis_dlfq = np.array(dict[view]['data'].values[:, 2:len(dict[view]['data'].values[0, :])].T, dtype='float')
    lab_dis_dlfq = np.zeros((len(data_dis_dlfq[:, 0]), 1))
    if 'condition' in dict[view]['design'].keys():

        lab_dis_dlfq[np.where((dict[view]['design']['condition'].values == pos))[0]] = 1
        lab_dis_dlfq = lab_dis_dlfq

    return data_dis_dlfq, lab_dis_dlfq

def Fill_nan(data, fill_value):

    data[np.isnan(data)] = fill_value
    return data

def cv_model(train_feas, train_lab, seed, param, model='XGB'):
    seed_torch(seed)
    rs = StratifiedKFold(n_splits=5, shuffle=True,  random_state=2024)
    aucs = []
    Mccs = []
    f1s = []
    accs = []
    bsts = []
    for i, (train_index, test_index) in enumerate(rs.split(train_feas, train_lab)):
        train = train_feas[train_index, :]
        test = train_feas[test_index, :]

        train_y = train_lab[train_index]
        test_y = train_lab[test_index]

        if model == 'XGB':
            bst = XGBClassifier(n_estimators=param['num'], max_depth=param['d'], learning_rate=param['lr'], objective='binary:logistic')
        elif model == 'svm':
            bst = SVC(C=param['C'], gamma=param['g'], probability=True)
        # fit model
        bst.fit(train, train_y)
        # make predictions
        preds = bst.predict_proba(test)
        pred_lab = bst.predict(test)
        Mcc = mcc(test_y[:, 0], pred_lab)
        auc = roc_auc_score(test_y[:, 0], preds[:, 1])
        Mccs.append(Mcc)
        aucs.append(auc)
        f1_ = f1(test_y[:, 0], pred_lab)
        f1s.append(f1_)
        ac = acc(test_y[:, 0], pred_lab)
        accs.append(ac)
        bsts.append(bst)
    return np.mean(aucs), np.mean(Mccs), np.mean(f1s), np.mean(accs)


def cv_model_mv(train_feas, train_lab, test_feas, test_lab, seed, param, model='XGB'):
    seed_torch(seed)

    kmcca = GCCA(latent_dimensions=param['N'], c=param['c'], random_state=2024)

    all_data = [np.row_stack((train_feas[i], test_feas[i])) for i in range(len(train_feas))]

    lat_out = kmcca.fit_transform(all_data)
    lats_out_mvidia_all = np.concatenate(lat_out, axis=1)
    lats_out_mvidia = lats_out_mvidia_all[0:len(train_lab), :]

    rs = StratifiedKFold(n_splits=5, shuffle=True,  random_state=2024)
    aucs = []
    Mccs = []
    f1s = []
    accs = []
    for i, (train_index, test_index) in enumerate(rs.split(train_feas[0], train_lab)):
        train = lats_out_mvidia[train_index, :]
        test = lats_out_mvidia[test_index, :]
        cat_train = np.concatenate(train_feas, axis=1)[train_index, :]
        cat_test = np.concatenate(train_feas, axis=1)[test_index, :]

        train_y = train_lab[train_index]
        test_y = train_lab[test_index]
        if model == 'mv-concat':
            bst = XGBClassifier(n_estimators=param['num'], max_depth=param['d'], learning_rate=param['lr'],
                                objective='binary:logistic')
            bst_concat = XGBClassifier(n_estimators=param['num_cat'], max_depth=param['d_cat'], learning_rate=param['lr_cat'],
                                objective='binary:logistic')
            bst_concat.fit(cat_train, train_y)
            bst.fit(train, train_y)
            # make predictions
            pred_mv = bst.predict_proba(test)
            pred_cat = bst_concat.predict_proba(cat_test)
            preds = (pred_mv + pred_cat)/2 #pred_mv * param['alpha'] + pred_cat * (1-param['alpha'])
            auc = roc_auc_score(test_y[:, 0], preds[:, 1])
            pred_lab = np.array([0] * len(preds))
            pred_lab[np.where((preds[:, 1] >= 0.5))[0]] = 1
            Mcc = mcc(test_y[:, 0], pred_lab)
            Mccs.append(Mcc)
            aucs.append(auc)
            f1_ = f1(test_y[:, 0], pred_lab)
            f1s.append(f1_)
            ac = acc(test_y[:, 0], pred_lab)
            accs.append(ac)
        elif model == 'mv+concat':
            bst = XGBClassifier(n_estimators=param['num'], max_depth=param['d'], learning_rate=param['lr'],
                                objective='binary:logistic')

            bst.fit(np.concatenate([train, cat_train],axis=1), train_y)
            preds = bst.predict_proba(np.concatenate([test, cat_test],axis=1))
            pred_lab = bst.predict(np.concatenate([test, cat_test],axis=1))
            Mcc = mcc(test_y[:, 0], pred_lab)
            auc = roc_auc_score(test_y[:, 0], preds[:, 1])
            Mccs.append(Mcc)
            aucs.append(auc)
            f1_ = f1(test_y[:, 0], pred_lab)
            f1s.append(f1_)
            ac = acc(test_y[:, 0], pred_lab)
            accs.append(ac)
        else:
            if model == 'XGB':
                bst = XGBClassifier(n_estimators=param['num'], max_depth=param['d'], learning_rate=param['lr'],
                                    objective='binary:logistic')

            elif model == 'svm':
                bst = SVC(C=param['C'], gamma=param['g'], probability=True)
            # fit model
            bst.fit(train, train_y)
            # make predictions
            preds = bst.predict_proba(test)

            pred_lab = bst.predict(test)
            Mcc = mcc(test_y[:, 0], pred_lab)
            auc = roc_auc_score(test_y[:, 0], preds[:, 1])
            Mccs.append(Mcc)
            aucs.append(auc)
            f1_ = f1(test_y[:, 0], pred_lab)
            f1s.append(f1_)
            ac = acc(test_y[:, 0], pred_lab)
            accs.append(ac)

    return np.mean(aucs), np.mean(Mccs), np.mean(f1s), np.mean(accs)

def cv_model_semi(train_feas, train_lab, test_feas, test_lab, seed, param, model='XGB'):
    seed_torch(seed)

    rs = StratifiedKFold(n_splits=5, shuffle=True,  random_state=2024)
    aucs = []
    Mccs = []
    f1s = []
    accs = []

    test_pesudo = np.array([[0] * len(test_lab)]).T
    test_data = [np.row_stack((train_feas[j], test_feas[j])) for j in range(len(train_feas))]
    test_label = np.row_stack((train_lab, test_pesudo))

    kmcca = GCCA(latent_dimensions=param['N'], c=param['c'], random_state=2024)
    lat_out = kmcca.fit_transform(test_data)
    lats_out_tr = lat_out[0]
    for i in range(1, len(lat_out)):
        lats_out_tr = np.column_stack((lats_out_tr, lat_out[i]))

    for i, (train_index, test_index) in enumerate(rs.split(lats_out_tr, test_label)):
        train = lats_out_tr[train_index, :]
        test = lats_out_tr[test_index, :]

        train_y = test_label[train_index]
        test_y = test_label[test_index]

        if model == 'XGB':
            bst = XGBClassifier(n_estimators=param['num'], max_depth=param['d'], learning_rate=param['lr'],
                                objective='binary:logistic')
        elif model == 'svm':
            bst = SVC(C=param['C'], gamma=param['g'], probability=True)
        # fit model
        bst.fit(train, train_y)
        # make predictions
        preds = bst.predict_proba(test)

        pred_lab = bst.predict(test)
        Mcc = mcc(test_y[:, 0], pred_lab)
        auc = roc_auc_score(test_y[:, 0], preds[:, 1])
        Mccs.append(Mcc)
        aucs.append(auc)
        f1_ = f1(test_y[:, 0], pred_lab)
        f1s.append(f1_)
        ac = acc(test_y[:, 0], pred_lab)
        accs.append(ac)

    return np.mean(aucs), np.mean(Mccs), np.mean(f1s), np.mean(accs)

def cv_model_mvidia(test_fea_single, train_fea_single, train_feas, train_lab, test_feas, test_lab, seed, param, bsts, model='XGB'):
    seed_torch(seed)

    preds = []
    pred_labs = []

    for i in range(len(train_feas)):

        pred = bsts[i].predict_proba(np.row_stack((train_fea_single[i], test_fea_single[i])))
        pred_lab = bsts[i].predict(np.row_stack((train_fea_single[i], test_fea_single[i])))
        preds.append(pred[:, 1])
        pred_labs.append(pred_lab)

    sum_lab = np.sum(pred_labs, axis=0)
    ind_reliable = np.where(((sum_lab == 0) | (sum_lab == len(train_feas))))[0]

    train_fea_reliable = [np.row_stack((train_feas[j], test_feas[j]))[ind_reliable, :] for j in range(len(train_feas))]
    train_lab_reliable = np.array([pred_labs[0][ind_reliable]]).T
    acc0 = acc(np.row_stack((train_lab, test_lab))[ind_reliable, 0], pred_labs[0][ind_reliable])
    if (model == 'mvidia') | (model == 'mv-concat') | (model == 'mv+concat'):
        kmcca = GCCA(latent_dimensions=param['N'], c=param['c'], random_state=2024)
        # fea_all = [np.row_stack((train_fea_single[i], test_fea_single[i])) for i in range(len(train_fea_single))]
        fea_all = [np.row_stack((train_feas[i], test_feas[i])) for i in range(len(train_feas))]
        lat_out = kmcca.fit_transform(fea_all)
        lats_out_all = np.concatenate(lat_out, axis=1)
        lats_out_tr = lats_out_all[ind_reliable, :]

    elif model == 'concat':
        lats_out_tr = np.concatenate(train_fea_reliable, axis=1)
    elif model == 'ens':
        lats_out_tr = train_fea_reliable

    Mccs = []
    aucs = []
    f1s = []
    accs = []
    rs = StratifiedKFold(n_splits=5, shuffle=True, random_state=2024)
    if model != 'ens':
        for i, (train_index, test_index) in enumerate(rs.split(lats_out_tr, train_lab_reliable)):
            train = lats_out_tr[train_index, :]
            test = lats_out_tr[test_index, :]

            cat_train = np.concatenate(train_fea_reliable, axis=1)[train_index, :]
            cat_test = np.concatenate(train_fea_reliable, axis=1)[test_index, :]

            train_y = train_lab_reliable[train_index]
            test_y = train_lab_reliable[test_index]

            if model == 'mv-concat':
                bst = XGBClassifier(n_estimators=param['num'], max_depth=param['d'], learning_rate=param['lr'],
                                    objective='binary:logistic')
                bst.fit(train, train_y)
                preds_mv = bst.predict_proba(test)

                bst_cat = XGBClassifier(n_estimators=param['num_cat'], max_depth=param['d_cat'], learning_rate=param['lr_cat'],
                                    objective='binary:logistic')
                bst_cat.fit(cat_train, train_y)
                preds_cat = bst_cat.predict_proba(cat_test)

                preds = (preds_mv + preds_cat)/2#preds_mv * param['alpha'] + preds_cat * (1-param['alpha'])
                pred_lab = np.array([0] * len(preds))
                pred_lab[np.where((preds[:, 1] >= 0.5))[0]] = 1
            elif model == 'mv+concat':
                bst = XGBClassifier(n_estimators=param['num'], max_depth=param['d'], learning_rate=param['lr'],
                                    objective='binary:logistic')
                bst.fit(np.concatenate([train, cat_train], axis=1), train_y)
                preds = bst.predict_proba(np.concatenate([test, cat_test], axis=1))
                pred_lab = bst.predict(np.concatenate([test, cat_test], axis=1))

            else:
                bst = XGBClassifier(n_estimators=param['num'], max_depth=param['d'], learning_rate=param['lr'],
                                    objective='binary:logistic')

                bst.fit(train, train_y)
                # make predictions
                preds = bst.predict_proba(test)
                pred_lab = bst.predict(test)
            Mcc = mcc(test_y[:, 0], pred_lab)
            auc = roc_auc_score(test_y[:, 0], preds[:, 1])
            Mccs.append(Mcc)
            aucs.append(auc)
            f1_ = f1(test_y[:, 0], pred_lab)
            f1s.append(f1_)
            ac = acc(test_y[:, 0], pred_lab)
            accs.append(ac)

    else:
        for i, (train_index, test_index) in enumerate(rs.split(lats_out_tr[0], train_lab_reliable)):
            train = [lats_out_tr[v][train_index, :] for v in range(len(lats_out_tr))]
            test = [lats_out_tr[v][test_index, :] for v in range(len(lats_out_tr))]

            train_y = train_lab_reliable[train_index]
            test_y = train_lab_reliable[test_index]

            bst = [XGBClassifier(n_estimators=param['num'+str(v)], max_depth=param['d'+str(v)], learning_rate=param['lr'+str(v)],
                                objective='binary:logistic') for v in range(len(bsts))]

            preds = []
            for v in range(len(bsts)):
                bst[v].fit(train[v], train_y)
                pred = bst[v].predict_proba(test[v])

                preds.append(pred[:, 1])
            preds = np.mean(preds, axis=0)

            pred_lab = np.array([0] * len(preds))
            pred_lab[np.where((preds >= 0.5))[0]] = 1

            Mcc = mcc(test_y[:, 0], pred_lab)
            auc = roc_auc_score(test_y[:, 0], preds)
            Mccs.append(Mcc)
            aucs.append(auc)
            f1_ = f1(test_y[:, 0], pred_lab)
            f1s.append(f1_)
            ac = acc(test_y[:, 0], pred_lab)
            accs.append(ac)

    return np.mean(aucs), np.mean(Mccs), np.mean(f1s), np.mean(accs)


def test_model(train_feas, train_lab, seed, test_feas, test_lab, param, model='XGB'):
    seed_torch(seed)

    if model == 'XGB':
        bst = XGBClassifier(n_estimators=param['num'], max_depth=param['d'], learning_rate=param['lr'],
                            objective='binary:logistic')
    elif model == 'svm':
        bst = SVC(C=2**param['C'], gamma=2**param['g'], probability=True)
    bst.fit(train_feas, train_lab)
    # make predictions
    preds = bst.predict_proba(test_feas)
    if len(np.unique(test_lab))>1:
        auc = roc_auc_score(test_lab[:, 0], preds[:, 1])
        pred_lab = bst.predict(test_feas)

        Acc = acc(test_lab[:, 0], pred_lab)
        Prec = precision(test_lab[:, 0], pred_lab)
        Rec = recall(test_lab[:, 0], pred_lab)
        F1 = f1(test_lab[:, 0], pred_lab)
        Mcc = mcc(test_lab[:, 0], pred_lab)
    else:
        auc, Acc, Prec, Rec, F1, Mcc = 0, 0, 0, 0, 0, 0



    return preds[:, 1], [auc, Acc, Prec, Rec, F1, Mcc], bst


def test_model_mv(train_feas, train_lab, seed, test_feas, test_lab, param, model='XGB'):
    seed_torch(seed)

    kmcca = GCCA(latent_dimensions=param['N'], c=param['c'], random_state=2024)
    all_data = [np.row_stack((train_feas[v], test_feas[v])) for v in range(len(train_feas))]
    lat_out = kmcca.fit_transform(all_data)
    lats_out_mvidia = np.concatenate(lat_out, axis=1)

    lats_out_mvidia_test = lats_out_mvidia[len(train_lab):len(lats_out_mvidia[:, 0]), :]
    lats_out_mvidia = lats_out_mvidia[0:len(train_lab), :]

    if model == 'mv-concat':
        bst = XGBClassifier(n_estimators=param['num'], max_depth=param['d'], learning_rate=param['lr'],
                            objective='binary:logistic')
        bst_concat = XGBClassifier(n_estimators=param['num_cat'], max_depth=param['d_cat'], learning_rate=param['lr_cat'],
                            objective='binary:logistic')

        bst.fit(lats_out_mvidia, train_lab)
        preds_mv = bst.predict_proba(lats_out_mvidia_test)

        bst_concat.fit(np.concatenate(train_feas, axis=1), train_lab)
        preds_cat = bst_concat.predict_proba(np.concatenate(test_feas, axis=1))

        preds = (preds_mv + preds_cat)/2#preds_mv * param['alpha'] + preds_cat * (1 - param['alpha'])
        pred_lab = np.array([0] * len(preds))
        pred_lab[np.where((preds[:, 1] >= 0.5))[0]] = 1

        bst = [bst, bst_concat]
    elif model == 'mv+concat':
        bst = XGBClassifier(n_estimators=param['num'], max_depth=param['d'], learning_rate=param['lr'],
                            objective='binary:logistic')
        bst.fit(np.concatenate([lats_out_mvidia, np.concatenate(train_feas, axis=1)], axis=1), train_lab)
        preds = bst.predict_proba(np.concatenate([lats_out_mvidia_test, np.concatenate(test_feas, axis=1)], axis=1))
        pred_lab = bst.predict(np.concatenate([lats_out_mvidia_test, np.concatenate(test_feas, axis=1)], axis=1))
    else:
        if model == 'XGB':
            bst = XGBClassifier(n_estimators=param['num'], max_depth=param['d'], learning_rate=param['lr'],
                                objective='binary:logistic')
        elif model == 'svm':
            bst = SVC(C=2**param['C'], gamma=2**param['g'], probability=True)
        bst.fit(lats_out_mvidia, train_lab)
        preds = bst.predict_proba(lats_out_mvidia_test)
        pred_lab = bst.predict(lats_out_mvidia_test)

    if len(np.unique(test_lab))>1:
        auc = roc_auc_score(test_lab[:, 0], preds[:, 1])
        Acc = acc(test_lab[:, 0], pred_lab)
        Prec = precision(test_lab[:, 0], pred_lab)
        Rec = recall(test_lab[:, 0], pred_lab)
        F1 = f1(test_lab[:, 0], pred_lab)
        Mcc = mcc(test_lab[:, 0], pred_lab)
    else:
        auc, Acc, Prec, Rec, F1, Mcc = 0, 0, 0, 0, 0, 0


    return preds[:, 1], [auc, Acc, Prec, Rec, F1, Mcc], lats_out_mvidia_test, bst

def test_model_semi(train_feas, train_lab, seed, test_feas, test_lab, param, model='XGB'):
    seed_torch(seed)
    seed_torch(seed)

    rs = StratifiedKFold(n_splits=5, shuffle=True, random_state=2024)
    # aucs = []
    # Mccs = []

    test_pesudo = np.array([[0] * len(test_lab)]).T
    test_data = [np.row_stack((train_feas[j], test_feas[j])) for j in range(len(train_feas))]
    test_label = np.row_stack((train_lab, test_pesudo))

    kmcca = GCCA(latent_dimensions=param['N'], c=param['c'], random_state=2024)
    lat_out = kmcca.fit_transform(test_data)
    lats_out_tr = lat_out[0]
    for i in range(1, len(lat_out)):
        lats_out_tr = np.column_stack((lats_out_tr, lat_out[i]))

    preds = []
    for i, (train_index, test_index) in enumerate(rs.split(lats_out_tr, test_label)):
        train = lats_out_tr[train_index, :]
        test = lats_out_tr[test_index, :]

        train_y = test_label[train_index]
        test_y = test_label[test_index]

        if model == 'XGB':
            bst = XGBClassifier(n_estimators=param['num'], max_depth=param['d'], learning_rate=param['lr'],
                                objective='binary:logistic')
        elif model == 'svm':
            bst = SVC(C=param['C'], gamma=param['g'], probability=True)
        # fit model
        bst.fit(train, train_y)

        # make predictions
        pred = bst.predict_proba(test[len(train_lab):len(test_label), :])
        preds.append(pred)

    preds = np.mean(preds, axis=0)

    if len(np.unique(test_lab))>1:
        auc = roc_auc_score(test_lab[:, 0], preds)

        pred_lab = np.array([0] * len(preds))
        pred_lab[np.where((preds >= 0.5))[0]] = 1

        Acc = acc(test_lab[:, 0], pred_lab)
        Prec = precision(test_lab[:, 0], pred_lab)
        Rec = recall(test_lab[:, 0], pred_lab)
        F1 = f1(test_lab[:, 0], pred_lab)
        Mcc = mcc(test_lab[:, 0], pred_lab)
    else:
        auc, Acc, Prec, Rec, F1, Mcc = 0, 0, 0, 0, 0, 0


    return preds, [auc, Acc, Prec, Rec, F1, Mcc], '', ''

def test_model_mvidia(test_fea_single, train_fea_single, train_feas, train_lab, seed, test_feas, test_lab, bsts, param, model='XGB'):
    seed_torch(seed)

    preds = []
    pred_labs = []

    for i in range(len(bsts)):

        pred = bsts[i].predict_proba(np.row_stack((train_fea_single[i], test_fea_single[i])))
        pred_lab = bsts[i].predict(np.row_stack((train_fea_single[i], test_fea_single[i])))
        preds.append(pred)
        pred_labs.append(pred_lab)

    sum_lab = np.sum(pred_labs, axis=0)
    ind_reliable = np.where(((sum_lab == 0) | (sum_lab == len(bsts))))[0]

    acc0 = acc(np.row_stack((train_lab, test_lab))[ind_reliable, 0], pred_labs[0][ind_reliable])

    train_fea_reliable = [np.row_stack((train_feas[j], test_feas[j]))[ind_reliable, :] for j in range(len(bsts))]
    train_lab_reliable = np.array([pred_labs[0]]).T[ind_reliable]

    if (model == 'mvidia') | (model == 'mv-concat') | (model == 'mv+concat'):

        all_data = [np.row_stack((train_fea_reliable[v], test_feas[v])) for v in range(len(bsts))]

        kmcca = GCCA(latent_dimensions=param['N'], c=param['c'], random_state=2024)

        lat_out = kmcca.fit_transform(all_data)
        lats_out_all = np.concatenate(lat_out, axis=1)

        lats_out_te = lats_out_all[len(train_lab_reliable):len(all_data[0][:, 0]), :]
        lats_out_tr = lats_out_all[0:len(train_lab_reliable), :]

        # all_data = [np.row_stack((train_feas[v], test_feas[v])) for v in range(len(bsts))]
        # kmcca = GCCA(latent_dimensions=param['N'], c=param['c'], random_state=2024)
        #
        # lat_out = kmcca.fit_transform(all_data)
        # lats_out_all = np.concatenate(lat_out, axis=1)
        #
        # lats_out_te = lats_out_all[len(train_lab):len(lats_out_all[:, 0]), :]
        # lats_out_tr = lats_out_all[ind_reliable, :]

    elif model == 'concat':
        lats_out_tr = np.concatenate(train_fea_reliable, axis=1)
        lats_out_te = np.concatenate(test_feas, axis=1)
    elif model == 'ens':
        lats_out_tr = train_fea_reliable
        lats_out_te = test_feas

    if model != 'ens':
        if model == 'mv-concat':
            bst = XGBClassifier(n_estimators=param['num'], max_depth=param['d'], learning_rate=param['lr'],
                                objective='binary:logistic')

            bst.fit(lats_out_tr, train_lab_reliable)
            preds_mv = bst.predict_proba(lats_out_te)

            bst_cat = XGBClassifier(n_estimators=param['num_cat'], max_depth=param['d_cat'], learning_rate=param['lr_cat'],
                                objective='binary:logistic')

            bst_cat.fit(np.concatenate(train_fea_reliable, axis=1), train_lab_reliable)
            preds_cat = bst_cat.predict_proba(np.concatenate(test_feas, axis=1))

            preds = (preds_mv + preds_cat)/2#preds_mv * param['alpha'] + preds_cat * (1 - param['alpha'])
            pred_lab = np.array([0] * len(preds))
            pred_lab[np.where((preds[:, 1] >= 0.5))[0]] = 1

            bst = [bst, bst_cat]

        elif model == 'mv+concat':
            bst = XGBClassifier(n_estimators=param['num'], max_depth=param['d'], learning_rate=param['lr'],
                                objective='binary:logistic')
            bst.fit(np.concatenate([lats_out_tr, np.concatenate(train_fea_reliable, axis=1)], axis=1), train_lab_reliable)
            preds = bst.predict_proba(np.concatenate([lats_out_te, np.concatenate(test_feas, axis=1)], axis=1))
            pred_lab = bst.predict(np.concatenate([lats_out_te, np.concatenate(test_feas, axis=1)], axis=1))

        else:
            bst = XGBClassifier(n_estimators=param['num'], max_depth=param['d'], learning_rate=param['lr'],
                                    objective='binary:logistic')

            bst.fit(lats_out_tr, train_lab_reliable)
            preds = bst.predict_proba(lats_out_te)
            pred_lab = bst.predict(lats_out_te)

        if len(np.unique(test_lab)) > 1:
            auc = roc_auc_score(test_lab[:, 0], preds[:, 1])
            Acc = acc(test_lab[:, 0], pred_lab)
            Prec = precision(test_lab[:, 0], pred_lab)
            Rec = recall(test_lab[:, 0], pred_lab)
            F1 = f1(test_lab[:, 0], pred_lab)
            Mcc = mcc(test_lab[:, 0], pred_lab)
        else:
            auc, Acc, Prec, Rec, F1, Mcc = 0, 0, 0, 0, 0, 0


        preds = preds[:, 1]

    else:

        bst = [XGBClassifier(n_estimators=param['num' + str(v)], max_depth=param['d' + str(v)],
                             learning_rate=param['lr' + str(v)],
                             objective='binary:logistic') for v in range(len(bsts))]

        preds = []
        for v in range(len(bsts)):
            bst[v].fit(lats_out_tr[v], train_lab_reliable)
            pred = bst[v].predict_proba(lats_out_te[v])

            preds.append(pred[:, 1])

        preds = np.mean(preds, axis=0)

        if len(np.unique(test_lab)) > 1:
            auc = roc_auc_score(test_lab[:, 0], preds)

            pred_lab = np.array([0] * len(preds))
            pred_lab[np.where((preds >= 0.5))[0]] = 1

            Acc = acc(test_lab[:, 0], pred_lab)
            Prec = precision(test_lab[:, 0], pred_lab)
            Rec = recall(test_lab[:, 0], pred_lab)
            F1 = f1(test_lab[:, 0], pred_lab)
            Mcc = mcc(test_lab[:, 0], pred_lab)
        else:
            auc, Acc, Prec, Rec, F1, Mcc = 0, 0, 0, 0, 0, 0


    return preds, [auc, Acc, Prec, Rec, F1, Mcc], lats_out_te, bst

def test_model_ens(bsts, params, seed, train_feas, test_feas, test_lab, ty):
    seed_torch(seed)


    preds = []
    for i in range(0,len(train_feas)):

        pred = bsts[i].predict_proba(test_feas[i])
        preds.append(pred[:, 1])

    if 'c' in ty: #fusion ens and cat
        idx = ty.index('c')
        preds.append(bsts[idx].predict_proba(np.concatenate(test_feas, axis=1))[:, 1])

    if 'g' in ty: # fusion ens, cat and gcca

        idx = ty.index('g')

        kmcca = GCCA(latent_dimensions=params[idx]['N'], c=params[idx]['c'], random_state=2024)
        all_data = [np.row_stack((train_feas[v], test_feas[v])) for v in range(len(train_feas))]
        lat_out = kmcca.fit_transform(all_data)
        lats_out_mvidia = np.concatenate(lat_out, axis=1)

        test_lat = lats_out_mvidia[len(train_feas[0][:, 0]):len(lats_out_mvidia[:, 0]), :]

        preds.append(bsts[idx].predict_proba(test_lat)[:, 1])


    preds = np.mean(preds, axis=0)

    if len(np.unique(test_lab)) > 1:
        auc = roc_auc_score(test_lab[:, 0], preds)

        pred_lab = np.array([0] * len(preds))
        pred_lab[np.where((preds >= 0.5))[0]] = 1

        Acc = acc(test_lab[:, 0], pred_lab)
        Prec = precision(test_lab[:, 0], pred_lab)
        Rec = recall(test_lab[:, 0], pred_lab)
        F1 = f1(test_lab[:, 0], pred_lab)
        Mcc = mcc(test_lab[:, 0], pred_lab)
    else:
        auc, Acc, Prec, Rec, F1, Mcc = 0, 0, 0, 0, 0, 0



    return preds, [auc, Acc, Prec, Rec, F1, Mcc]

def test_model_ens_all(preds, test_lab):

    preds = np.mean(preds, axis=0)

    if len(np.unique(test_lab)) > 1:
        auc = roc_auc_score(test_lab[:, 0], preds)

        pred_lab = np.array([0] * len(preds))
        pred_lab[np.where((preds >= 0.5))[0]] = 1

        Acc = acc(test_lab[:, 0], pred_lab)
        Prec = precision(test_lab[:, 0], pred_lab)
        Rec = recall(test_lab[:, 0], pred_lab)
        F1 = f1(test_lab[:, 0], pred_lab)
        Mcc = mcc(test_lab[:, 0], pred_lab)
    else:
        auc, Acc, Prec, Rec, F1, Mcc = 0, 0, 0, 0, 0, 0



    return preds, [auc, Acc, Prec, Rec, F1, Mcc]

def seed_torch(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)


def objective(trail, train_feas, train_lab, test_feas, test_lab, seed, model_type):
    seed_torch(seed)

    if model_type == 'single':
        num = trail.suggest_int('num', 2, 100, step=5) # n_estimators
        d = trail.suggest_int('d', 1, 10, step=1) # max depth
        lr = trail.suggest_float('lr', 0.0001, 1) # max depth
        param = {'num': num, 'd': d, 'lr':lr}
        auc, mcc, f1, acc = cv_model(train_feas, train_lab, seed, param)

    elif (model_type == 'mv') | (model_type == 'mv+concat'):
        max_N = min([len(train_feas[0][0, :]), len(train_feas[0][:, 0])])
        N = trail.suggest_int('N', 2, max_N, step=1) # n_estimators
        num = trail.suggest_int('num', 2, 100, step=5)  # n_estimators
        d = trail.suggest_int('d', 1, 15, step=1)  # max depth
        c = trail.suggest_float('c', 0.0001, 1)  # max depth
        lr = trail.suggest_float('lr', 0.0001, 1)  # max depth
        # alpha = trail.suggest_float('alpha', 0, 1)
        param = {'N': N, 'num': num, 'd': d, 'lr':lr, 'c':c}
        if model_type == 'mv+cat':
            auc, mcc, f1, acc = cv_model_mv(train_feas, train_lab, test_feas, test_lab, seed, param, model='mv+concat')
        else:
            auc, mcc, f1, acc = cv_model_mv(train_feas, train_lab, test_feas, test_lab, seed, param)

    elif model_type == 'mv-concat':
        max_N = min([len(train_feas[0][0, :]), len(train_feas[0][:, 0])])
        N = trail.suggest_int('N', 2, max_N, step=1) # n_estimators
        num = trail.suggest_int('num', 2, 100, step=5)  # n_estimators
        d = trail.suggest_int('d', 1, 10, step=1)  # max depth
        c = trail.suggest_float('c', 0.0001, 1)  # max depth
        lr = trail.suggest_float('lr', 0.0001, 1)  # max depth

        num_cat = trail.suggest_int('num_cat', 2, 100, step=5)  # n_estimators
        d_cat = trail.suggest_int('d_cat', 1, 10, step=1)  # max depth
        lr_cat = trail.suggest_float('lr_cat', 0.0001, 1)  # max depth
        # alpha = trail.suggest_float('alpha', 0.0, 1)  # max depth
        param = {'N': N, 'num': num, 'd': d, 'lr':lr, 'c':c, 'num_cat':num_cat, 'd_cat':d_cat, 'lr_cat':lr_cat}#, 'alpha':alpha}

        auc, mcc, f1, acc = cv_model_mv(train_feas, train_lab, test_feas, test_lab, seed, param, model='mv-concat')

    elif model_type == 'single_svm':
        C = trail.suggest_int('C', -10, 10, step=1) # n_estimators
        g = trail.suggest_int('g', -10, 10, step=1) # max depth
        #lr = trail.suggest_float('lr', 0.0001, 1) # max depth
        param = {'C': 2**C, 'g': 2**g}
        auc, mcc, f1, acc = cv_model(train_feas, train_lab, seed, param, model='svm')

    elif model_type == 'mv_svm':

        max_N = min([len(train_feas[0][0, :]), len(train_feas[0][:, 0])])
        N = trail.suggest_int('N', 2, max_N, step=1)  # n_estimators
        C = trail.suggest_int('C', -10, 10, step=1)  # n_estimators
        g = trail.suggest_int('g', -10, 10, step=1)  # max depth
        param = {'N': N,'C': 2**C, 'g': 2**g}

        auc, mcc, f1, acc = cv_model_mv(train_feas, train_lab, seed, param, model='svm')


    return 1 - auc#acc#f1

def objective_semi(trail, train_feas, train_lab, test_feas, test_lab, seed, model_type):
    seed_torch(seed)

    if model_type == 'single':
        num = trail.suggest_int('num', 2, 100, step=5) # n_estimators
        d = trail.suggest_int('d', 1, 10, step=1)  # max depth
        lr = trail.suggest_float('lr', 0.0001, 1)  # max depth
        param = {'num': num, 'd': d, 'lr': lr}
        auc, mcc, f1, acc = cv_model_semi(train_feas, train_lab, test_feas, test_lab, seed, param)

    elif model_type == 'semi':

        max_N = min([len(train_feas[0][0, :]), len(train_feas[0][:, 0])])
        N = trail.suggest_int('N', 2, max_N, step=1)  # n_estimators
        num = trail.suggest_int('num', 2, 100, step=5) # n_estimators
        d = trail.suggest_int('d', 1, 10, step=1)  # max depth
        lr = trail.suggest_float('lr', 0.0001, 1)  # max depth
        c = trail.suggest_float('c', 0.0001, 1)  # max depth
        param = {'N': N, 'num': num, 'd': d, 'lr': lr, 'c':c}

        auc, mcc, f1, acc = cv_model_semi(train_feas, train_lab, test_feas, test_lab, seed, param)

    return 1 - auc#acc#f1

def objective_mvidia(trail, test_fea_single, train_fea_single, train_feas, train_lab, test_feas, test_lab, seed, model_type, bst):
    seed_torch(seed)

    if (model_type == 'concat'):
        num = trail.suggest_int('num', 2, 100, step=5) # n_estimators
        d = trail.suggest_int('d', 1, 10, step=1)  # max depth
        lr = trail.suggest_float('lr', 0.0001, 1)  # max depth
        param = {'num': num, 'd': d, 'lr': lr}
        auc, mcc, f1, acc = cv_model_mvidia(test_fea_single, train_fea_single, train_feas, train_lab, test_feas, test_lab, seed, param, bst, model_type)

    elif model_type == 'mv-concat':
        max_N = min([len(train_feas[0][0, :]), len(train_feas[0][:, 0])])
        N = trail.suggest_int('N', 2, max_N, step=1) # n_estimators
        num = trail.suggest_int('num', 2, 100, step=5)  # n_estimators
        d = trail.suggest_int('d', 1, 10, step=1)  # max depth
        c = trail.suggest_float('c', 0.0001, 1)  # max depth
        lr = trail.suggest_float('lr', 0.0001, 1)  # max depth

        num_cat = trail.suggest_int('num_cat', 2, 100, step=5)  # n_estimators
        d_cat = trail.suggest_int('d_cat', 1, 10, step=1)  # max depth
        lr_cat = trail.suggest_float('lr_cat', 0.0001, 1)  # max depth
        # alpha = trail.suggest_float('alpha', 0.0, 1)  # max depth
        param = {'N': N, 'num': num, 'd': d, 'lr':lr, 'c':c, 'num_cat':num_cat, 'd_cat':d_cat, 'lr_cat':lr_cat}#, 'alpha':alpha}

        auc, mcc, f1, acc = cv_model_mvidia(test_fea_single, train_fea_single, train_feas, train_lab, test_feas, test_lab, seed,
                                   param, bst, model_type)

    elif (model_type == 'mvidia') | (model_type == 'mv+concat') | (model_type == 'mv+concat+ens'):

        max_N = min([len(train_feas[0][0, :]), len(train_feas[0][:, 0])])
        N = trail.suggest_int('N', 2, max_N, step=1)  # n_estimators
        c = trail.suggest_float('c', 0.0001, 1)
        num = trail.suggest_int('num', 2, 100, step=5) # n_estimators
        d = trail.suggest_int('d', 1, 10, step=1)  # max depth
        lr = trail.suggest_float('lr', 0.0001, 1)  # max depth
        param = {'N': N, 'num': num, 'd': d, 'lr': lr, 'c':c}

        auc, mcc, f1, acc = cv_model_mvidia(test_fea_single, train_fea_single, train_feas, train_lab, test_feas, test_lab, seed, param, bst, model_type)

    elif model_type == 'ens':
        param = {}
        for v in range(len(bst)):
            num = trail.suggest_int('num'+str(v), 2, 100, step=5)  # n_estimators
            d = trail.suggest_int('d'+str(v), 1, 10, step=1)  # max depth
            lr = trail.suggest_float('lr'+str(v), 0.0001, 1)  # max depth
            param.update({'num'+str(v): num, 'd'+str(v): d, 'lr'+str(v): lr})

        auc, mcc, f1, acc = cv_model_mvidia(test_fea_single, train_fea_single, train_feas, train_lab, test_feas, test_lab, seed, param, bst, model_type)

    return 1 - auc#acc#f1

def optuna_optimize(train_feas, train_lab, seed, model_type, test_feas, test_lab, trial_num):
    seed_torch(seed)
    study = optuna.create_study(study_name='test', direction='minimize', sampler=optuna.samplers.TPESampler(seed=seed),
                                pruner=optuna.pruners.HyperbandPruner())

    if model_type != 'semi':
        func = lambda trial: objective(trial, train_feas, train_lab, test_feas, test_lab, seed, model_type)
    else:
        func = lambda trial: objective_semi(trial, train_feas, train_lab, test_feas, test_lab, seed, model_type)
    study.optimize(func, n_trials=trial_num)
    print(study.best_params)
    print(study.best_trial)
    print(study.best_trial.value)
    print(model_type)
    if (model_type == 'single') | (model_type == 'single_svm'):
        if model_type == 'single_svm':
            model = 'svm'
            #study.best_params.update({'C': 2 ** study.best_params['C'], 'g': 2 ** study.best_params['g']})
        else:
            model = 'XGB'

        pred, auc, bst = test_model(train_feas, train_lab, seed, test_feas, test_lab, study.best_params, model)
        lat = '_'
    elif model_type == 'mv-concat':
        model = model_type
        pred, auc, lat, bst = test_model_mv(train_feas, train_lab, seed, test_feas, test_lab, study.best_params, model)
    elif (model_type == 'mv') | (model_type == 'mv_svm') | (model_type == 'semi') | (model_type == 'mv+concat'):
        if model_type == 'mv_svm':
            model = 'svm'
            #study.best_params.update({'C': 2 ** study.best_params['C'], 'g': 2 ** study.best_params['g']})
        else:
            model = 'XGB'
        pred, auc, lat, bst = test_model_mv(train_feas, train_lab, seed, test_feas, test_lab, study.best_params, model)
    elif model_type == 'semi':
        pred, auc, lat, bst = test_model_semi(train_feas, train_lab, seed, test_feas, test_lab, study.best_params, model)
    return pred, auc, study.best_params, lat, bst

def optuna_optimize_mvidia(test_fea_single, train_fea_single, train_feas, train_lab, seed, model_type, test_feas, test_lab, bst, trial_num):
    seed_torch(seed)
    study = optuna.create_study(study_name='test', direction='minimize', sampler=optuna.samplers.TPESampler(seed=seed),
                                pruner=optuna.pruners.HyperbandPruner())

    func = lambda trial: objective_mvidia(trial, test_fea_single, train_fea_single, train_feas, train_lab, test_feas, test_lab, seed, model_type, bst)
    study.optimize(func, n_trials=trial_num)
    print(study.best_params)
    print(study.best_trial)
    print(study.best_trial.value)
    print(model_type)

    pred, auc, lat, bst = test_model_mvidia(test_fea_single, train_fea_single, train_feas, train_lab, seed, test_feas, test_lab, bst, study.best_params, model_type)
    return pred, auc, study.best_params, lat, bst

def TPD_classificaion(pro_list, res_name, trial_num):

    tr_data, te_data = preprocess_multi_view(pro_list)
    #tr_data, te_data = preprocess_multi_view_bc(pro_list, 'limma')
    # fill the nan value
    data_dis_dlfq, lab_dis_dlfq = get_data_lab(tr_data, 'dlfq')
    data_dis_dlfq = Fill_nan(data_dis_dlfq, min(data_dis_dlfq[~np.isnan(data_dis_dlfq)]))
    # data_dis_dlfq = Fill_nan(data_dis_dlfq, 0)
    data_ret_dlfq, lab_ret_dlfq = get_data_lab(te_data, 'dlfq')
    data_ret_dlfq = Fill_nan(data_ret_dlfq, min(data_dis_dlfq[~np.isnan(data_dis_dlfq)]))
    # data_ret_dlfq = Fill_nan(data_ret_dlfq, 0)

    data_dis_maxlfq, lab_dis_maxlfq = get_data_lab(tr_data, 'maxlfq')
    data_dis_maxlfq = Fill_nan(data_dis_maxlfq, min(data_dis_maxlfq[~np.isnan(data_dis_maxlfq)]))
    # data_dis_maxlfq = Fill_nan(data_dis_maxlfq, 0)
    data_ret_maxlfq, lab_ret_maxlfq = get_data_lab(te_data, 'maxlfq')
    data_ret_maxlfq = Fill_nan(data_ret_maxlfq, min(data_dis_maxlfq[~np.isnan(data_dis_maxlfq)]))
    # data_ret_maxlfq = Fill_nan(data_ret_maxlfq, 0)

    data_dis_top3, lab_dis_top3 = get_data_lab(tr_data, 'top3')
    data_dis_top3 = Fill_nan(data_dis_top3, min(data_dis_top3[~np.isnan(data_dis_top3)]))
    # data_dis_top3 = Fill_nan(data_dis_top3, 0)
    data_ret_top3, lab_ret_top3 = get_data_lab(te_data, 'top3')
    data_ret_top3 = Fill_nan(data_ret_top3, min(data_dis_top3[~np.isnan(data_dis_top3)]))
    # data_ret_top3 = Fill_nan(data_ret_top3, 0)

    seed = 2024

    pred1, auc1, best_params1, lat, bst1 = optuna_optimize(data_dis_dlfq, lab_dis_dlfq, seed, 'single', data_ret_dlfq, lab_ret_dlfq, trial_num)

    pred2, auc2, best_params2, lat, bst2 = optuna_optimize(data_dis_maxlfq, lab_dis_maxlfq, seed, 'single', data_ret_maxlfq,
                                                        lab_ret_maxlfq, trial_num)

    pred3, auc3, best_params3, lat, bst3 = optuna_optimize(data_dis_top3, lab_dis_top3, seed, 'single',
                                                        data_ret_top3,
                                                        lab_ret_top3, trial_num)

    train_fea_single = [data_dis_dlfq, data_dis_maxlfq, data_dis_top3]
    test_fea_single = [data_ret_dlfq, data_ret_maxlfq, data_ret_top3]

    # pred6, auc6 = test_model_ens([bst1, bst2, bst3], [best_params1, best_params2, best_params3], seed, train_fea_single, test_fea_single, lab_ret_dlfq, ['s', 's', 's'])
    pred6, auc6 = test_model_ens_all([pred1, pred2, pred3], lab_ret_dlfq)
    data_dis_concat = np.column_stack((data_dis_dlfq, data_dis_maxlfq, data_dis_top3))
    data_ret_concat = np.column_stack((data_ret_dlfq, data_ret_maxlfq, data_ret_top3))

    pred4, auc4, best_params4, lat, bst4 = optuna_optimize(data_dis_concat, lab_dis_top3, seed, 'single',
                                                         data_ret_concat,
                                                         lab_ret_top3, trial_num)

    pred15, auc15 = test_model_ens_all([pred1, pred2, pred3, pred4], lab_ret_dlfq)

    #normalize the data
    normalizer = StandardScaler()
    # normalizer = StandardScaler()
    normalizer.fit(data_dis_dlfq)

    data_dis_dlfq = normalizer.transform(data_dis_dlfq)
    data_ret_dlfq = normalizer.transform(data_ret_dlfq)

    # normalizer = StandardScaler()
    # normalizer.fit(data_dis_maxlfq)
    data_dis_maxlfq = normalizer.transform(data_dis_maxlfq)
    data_ret_maxlfq = normalizer.transform(data_ret_maxlfq)

    # normalizer = StandardScaler()
    # normalizer.fit(data_dis_top3)
    data_dis_top3 = normalizer.transform(data_dis_top3)
    data_ret_top3 = normalizer.transform(data_ret_top3)

    in_dis_exp = [data_dis_dlfq, data_dis_maxlfq, data_dis_top3]
    in_ret_exp = [data_ret_dlfq, data_ret_maxlfq, data_ret_top3]


    pred5, auc5, best_params5, lat, bst5 = optuna_optimize(in_dis_exp, lab_dis_top3, seed, 'mv',
                                                        in_ret_exp,
                                                        lab_ret_top3, trial_num)

    pred16, auc16 = test_model_ens_all([pred1, pred2, pred3, pred4, pred5], lab_ret_dlfq)
    pred17, auc17 = test_model_ens_all([pred1, pred2, pred3, pred5], lab_ret_dlfq)
    pred18, auc18 = test_model_ens_all([pred4, pred5], lab_ret_dlfq)
    pred11, auc11, best_params11, lat, bst11 = optuna_optimize(in_dis_exp, lab_dis_top3, seed, 'mv-concat',
                                                         in_ret_exp,
                                                         lab_ret_top3, trial_num)
    pred13, auc13, best_params13, lat, bst13 = optuna_optimize(in_dis_exp, lab_dis_top3, seed, 'mv+concat',
                                                           in_ret_exp,
                                                           lab_ret_top3, trial_num)

    pred7, auc7, best_params7, lat, bst7 = optuna_optimize(in_dis_exp, lab_dis_top3, seed, 'semi',
                                                         in_ret_exp,
                                                         lab_ret_top3, trial_num)

    pred8, auc8, best_params8, lat, bst8 = optuna_optimize_mvidia(test_fea_single, train_fea_single, in_dis_exp, lab_dis_top3, seed, 'concat',
                                                         in_ret_exp,
                                                         lab_ret_top3, [bst1, bst2, bst3], trial_num) #, bst3
    pred9, auc9, best_params9, lat9, bst9 = optuna_optimize_mvidia(test_fea_single, train_fea_single, in_dis_exp,
                                                                lab_dis_top3, seed, 'mvidia',
                                                                in_ret_exp,
                                                                lab_ret_top3, [bst1, bst2, bst3], trial_num) #, bst3
    pred10, auc10, best_params10, lat, bst10 = optuna_optimize_mvidia(test_fea_single, train_fea_single, in_dis_exp,
                                                                lab_dis_top3, seed, 'ens',
                                                                in_ret_exp,
                                                                lab_ret_top3, [bst1, bst2, bst3], trial_num) #, bst3
    pred11, auc12, best_params11, lat, bst12 = optuna_optimize_mvidia(test_fea_single, train_fea_single, in_dis_exp,
                                                                  lab_dis_top3, seed, 'mv-concat',
                                                                  in_ret_exp,
                                                                  lab_ret_top3, [bst1, bst2, bst3], trial_num) #, bst3
    pred14, auc14, best_params14, lat, bst14 = optuna_optimize_mvidia(test_fea_single, train_fea_single, in_dis_exp,
                                                                  lab_dis_top3, seed, 'mv+concat',
                                                                  in_ret_exp,
                                                                  lab_ret_top3, [bst1, bst2, bst3], trial_num) #, bst3

    test_feas = test_fea_single
    test_feas.append(np.concatenate(test_fea_single, axis=1))
    test_feas.append(lat9)

    pred19, auc19 = test_model_ens_all([pred1, pred2, pred3, pred4, pred9], lab_ret_dlfq)
    pred20, auc20 = test_model_ens_all([pred4, pred9], lab_ret_dlfq)
    pred21, auc21 = test_model_ens_all([pred8, pred9, pred10], lab_ret_dlfq)
    pred22, auc22 = test_model_ens_all([pred8, pred10], lab_ret_dlfq)
    pred23, auc23 = test_model_ens_all([pred9, pred10], lab_ret_dlfq)
    pred24, auc24 = test_model_ens_all([pred1, pred2, pred3, pred9], lab_ret_dlfq)

    # print(auc0)
    res_all = np.array([auc1, auc2, auc3, auc4, auc5, auc6, auc7, auc8, auc9, auc10, auc11, auc12, auc13, auc14, auc15,
                        auc16, auc17, auc18, auc19, auc20, auc21, auc22, auc23, auc24])

    df = pd.DataFrame(res_all, columns=['auc', 'Acc', 'Prec', 'Rec', 'F1', 'Mcc'],
                 index=['dlfq', 'maxlfq', 'top3', 'concat', 'gcca', 'ens','semi', 'mvidia_concat', 'mvidia_gcca',
                        'mvidia_ens', 'gcca-cat', 'mvidia_gcca-cat', 'gcca+cat', 'mvidia_gcca+cat', 'ens_s_c',
                        'ens_s_c_g', 'ens_s_g', 'ens_c_g', 'ens_s_c_mg', 'ens_c_mg', 'ens_mc_mg_ms', 'ens_mc_ms', 'ens_mg_ms', 'ens_s_mg'])
    print(df)
    df.to_csv(res_name +'_XGB_perform_test_mv+cat_correct_ens_all.csv')

def MVIDIA_classificaion(pro_list, trial_num, train_data, test_data, pos, method, view_names, save_fold):
    view_names = view_names.split(',')
    tr_data, te_data = preprocess_multi_view_new(pro_list, train_data, test_data)
    # fill the nan value
    data_dis_dlfq, lab_dis_dlfq = get_data_lab(tr_data, 'v1', pos)
    data_dis_dlfq = Fill_nan(data_dis_dlfq, min(data_dis_dlfq[~np.isnan(data_dis_dlfq)]))
    # data_dis_dlfq = Fill_nan(data_dis_dlfq, 0)
    data_ret_dlfq, lab_ret_dlfq = get_data_lab(te_data, 'v1', pos)
    data_ret_dlfq = Fill_nan(data_ret_dlfq, min(data_dis_dlfq[~np.isnan(data_dis_dlfq)]))
    # data_ret_dlfq = Fill_nan(data_ret_dlfq, 0)

    data_dis_maxlfq, lab_dis_maxlfq = get_data_lab(tr_data, 'v2', pos)
    data_dis_maxlfq = Fill_nan(data_dis_maxlfq, min(data_dis_maxlfq[~np.isnan(data_dis_maxlfq)]))
    # data_dis_maxlfq = Fill_nan(data_dis_maxlfq, 0)
    data_ret_maxlfq, lab_ret_maxlfq = get_data_lab(te_data, 'v2', pos)
    data_ret_maxlfq = Fill_nan(data_ret_maxlfq, min(data_dis_maxlfq[~np.isnan(data_dis_maxlfq)]))
    # data_ret_maxlfq = Fill_nan(data_ret_maxlfq, 0)

    data_dis_top3, lab_dis_top3 = get_data_lab(tr_data, 'v3', pos)
    data_dis_top3 = Fill_nan(data_dis_top3, min(data_dis_top3[~np.isnan(data_dis_top3)]))
    # data_dis_top3 = Fill_nan(data_dis_top3, 0)
    data_ret_top3, lab_ret_top3 = get_data_lab(te_data, 'v3', pos)
    data_ret_top3 = Fill_nan(data_ret_top3, min(data_dis_top3[~np.isnan(data_dis_top3)]))
    # data_ret_top3 = Fill_nan(data_ret_top3, 0)

    seed = 2024

    preds_all = []
    method_names = []
    aucs = []

    pred1, auc1, best_params1, lat, bst1 = optuna_optimize(data_dis_dlfq, lab_dis_dlfq, seed, 'single', data_ret_dlfq, lab_ret_dlfq, trial_num)

    preds_all.append(pred1)
    method_names.append(view_names[0])
    aucs.append(auc1)

    pred2, auc2, best_params2, lat, bst2 = optuna_optimize(data_dis_maxlfq, lab_dis_maxlfq, seed, 'single', data_ret_maxlfq,
                                                        lab_ret_maxlfq, trial_num)
    preds_all.append(pred2)
    method_names.append(view_names[1])
    aucs.append(auc2)

    pred3, auc3, best_params3, lat, bst3 = optuna_optimize(data_dis_top3, lab_dis_top3, seed, 'single',
                                                        data_ret_top3,
                                                        lab_ret_top3, trial_num)

    preds_all.append(pred3)
    method_names.append(view_names[2])
    aucs.append(auc3)

    train_fea_single = [data_dis_dlfq, data_dis_maxlfq, data_dis_top3]
    test_fea_single = [data_ret_dlfq, data_ret_maxlfq, data_ret_top3]

    if method == 'MLE':
        pred6, auc6 = test_model_ens_all([pred1, pred2, pred3], lab_ret_dlfq)
        preds_all.append(pred6)
        method_names.append('MLE')
        aucs.append(auc6)

    if (method == 'Concat') | (method == 'Concat-MVIDIA'):
        data_dis_concat = np.column_stack((data_dis_dlfq, data_dis_maxlfq, data_dis_top3))
        data_ret_concat = np.column_stack((data_ret_dlfq, data_ret_maxlfq, data_ret_top3))

        pred4, auc4, best_params4, lat, bst4 = optuna_optimize(data_dis_concat, lab_dis_top3, seed, 'single',
                                                             data_ret_concat,
                                                             lab_ret_top3, trial_num)
        preds_all.append(pred6)
        method_names.append('MLE')
        aucs.append(auc4)

    if (method == 'MVIDIA') | (method == 'Concat-MVIDIA'):
        #normalize the data
        normalizer = StandardScaler()
        normalizer.fit(data_dis_dlfq)

        data_dis_dlfq = normalizer.transform(data_dis_dlfq)
        data_ret_dlfq = normalizer.transform(data_ret_dlfq)

        data_dis_maxlfq = normalizer.transform(data_dis_maxlfq)
        data_ret_maxlfq = normalizer.transform(data_ret_maxlfq)

        data_dis_top3 = normalizer.transform(data_dis_top3)
        data_ret_top3 = normalizer.transform(data_ret_top3)

        in_dis_exp = [data_dis_dlfq, data_dis_maxlfq, data_dis_top3]
        in_ret_exp = [data_ret_dlfq, data_ret_maxlfq, data_ret_top3]


        pred9, auc9, best_params9, lat9, bst9 = optuna_optimize_mvidia(test_fea_single, train_fea_single, in_dis_exp,
                                                                    lab_dis_top3, seed, 'mvidia',
                                                                    in_ret_exp,
                                                                    lab_ret_top3, [bst1, bst2, bst3], trial_num) #, bst3

        preds_all.append(pred9)
        method_names.append('GCCA')
        aucs.append(auc9)

        pred24, auc24 = test_model_ens_all([pred1, pred2, pred3, pred9], lab_ret_dlfq)
        preds_all.append(pred24)
        method_names.append('MVIDIA')
        aucs.append(auc24)

        if method == 'Concat-MVIDIA':

            pred19, auc19 = test_model_ens_all([pred1, pred2, pred3, pred4, pred9], lab_ret_dlfq)
            preds_all.append(pred19)
            method_names.append('Concat-MVIDIA')
            aucs.append(auc19)

    method_names = ['predict_by_' + method_names[i] for i in range(len(method_names))]
    preds_res = np.array(preds_all)
    test_samples = te_data['v1']['design']['sample_ID'].values

    if 'condition' in te_data['v1']['design'].keys():
        true_label = te_data['v1']['design']['condition'].values
        out_pd = pd.DataFrame(np.column_stack((test_samples, true_label, preds_res.T)), columns=['sample_ID', 'True label'] + method_names)
        metric_all = np.array(aucs)
        df = pd.DataFrame(metric_all, columns=['auc', 'Acc', 'Prec', 'Rec', 'F1', 'Mcc'],
                          index=method_names)
        print(df)
        df.to_csv(save_fold + 'Metrics_for_Methods_PD.csv', index=False)
    else:
        out_pd = pd.DataFrame(np.column_stack((test_samples, preds_res.T)), columns=['sample_ID'] + method_names)
    out_pd.to_csv(save_fold + 'Prediction_by_MVIDIA_Methods.csv', index=False)

    return out_pd

if __name__ == "__main__":
    pro_list = ['P02765', 'P04083', 'O00339', 'P58546', 'O75347', 'P04216', 'P02751', 'P83731', 'P00568', 'P78527', 'P04792', 'P57737', 'P42224', 'P27797', 'Q9HAT2', 'P30086', 'O14964', 'P10909', 'P17931']

    res_name = 'author19'
    tri_num = 30 # trial number in hyperparameter selection
    # TPD_classificaion(pro_list, res_name, tri_num)
