import pandas as pd
import numpy as np
import os
import csv
from sklearn import preprocessing
import random
import copy
from sklearn.cross_decomposition import CCA, PLSCanonical
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import optuna
#import mvlearn
from itertools import combinations


def seed_torch(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

def get_view_from_view_file(view_file, g1, g2, design):

    view_raw = pd.read_csv(view_file, sep='\t', header=0)

    designs = pd.read_csv(design, sep='\t', header=0)

    sample_g1 = designs['sample_name'].values[np.where((designs['condition'].values == g1))[0]]
    sample_g2 = designs['sample_name'].values[np.where((designs['condition'].values == g2))[0]]
    header_v1 = np.array(view_raw.columns.values.tolist())

    protein_v1 = view_raw['Protein'].values

    idx_g1g2_v1 = []

    for sample in list(sample_g1) + list(sample_g2):
        for i in range(len(header_v1)):
            if sample in header_v1[i]:
                idx_g1g2_v1.append(i)

    idx_g1g2_v1 = np.array(np.unique(idx_g1g2_v1), dtype='int')

    view1_data = view_raw.values[:, idx_g1g2_v1]

    idx_r = np.where(((designs['condition'].values == g1) | (designs['condition'].values == g2)))[0]

    save_path_design = view_file.replace(view_file.split('/')[-1], '') + design.split('/')[-1]

    view1_pd = pd.DataFrame(np.column_stack((protein_v1, view1_data)), columns=['Protein'] + list(header_v1[idx_g1g2_v1]))

    pd.DataFrame(designs.values[idx_r, :], columns=designs.columns.values.tolist()).to_csv(save_path_design, sep='\t', index=False)
    return view_raw, view1_pd, save_path_design, idx_g1g2_v1

def get_views_from_view_files(view1_file, view2_file, g1, g2, design):

    view1 = pd.read_csv(view1_file, sep='\t', header=0)
    view2 = pd.read_csv(view2_file, sep='\t', header=0)

    designs = pd.read_csv(design, sep='\t', header=0)

    sample_g1 = designs['sample_name'].values[np.where((designs['condition'].values == g1))[0]]
    sample_g2 = designs['sample_name'].values[np.where((designs['condition'].values == g2))[0]]
    header_v1 = np.array(view1.columns.values.tolist())
    header_v2 = np.array(view2.columns.values.tolist())
    protein_v1 = view1['Protein'].values
    protein_v2 = view2['Protein'].values

    idx_g1g2_v1 = []
    idx_g1g2_v2 = []

    for sample in list(sample_g1) + list(sample_g2):
        for i in range(len(header_v1)):
            if sample in header_v1[i]:
                idx_g1g2_v1.append(i)

        for i in range(len(header_v2)):
            if sample in header_v2[i]:
                idx_g1g2_v2.append(i)

    idx_g1g2_v1 = np.array(np.unique(idx_g1g2_v1), dtype='int')
    idx_g1g2_v2 = np.array(np.unique(idx_g1g2_v2), dtype='int')

    view1_data = view1.values[:, idx_g1g2_v1]
    view2_data = view2.values[:, idx_g1g2_v2]

    idx_r = np.where(((designs['condition'].values == g1) | (designs['condition'].values == g2)))[0]
    file_name_design = design.split('/')
    save_path_design = './data/' + file_name_design[len(file_name_design) - 1]

    view1_pd = pd.DataFrame(np.column_stack((protein_v1, view1_data)), columns=['Protein'] + list(header_v1[idx_g1g2_v1]))
    view2_pd = pd.DataFrame(np.column_stack((protein_v2, view2_data)), columns=['Protein'] + list(header_v2[idx_g1g2_v2]))

    pd.DataFrame(designs.values[idx_r, :], columns=designs.columns.values.tolist()).to_csv(save_path_design, sep='\t', index=False)
    return view1, view2, view1_pd, view2_pd, save_path_design

def get_views_from_raw(platform, raw_file, evidence_file, view_names, g1, g2, design, R_fold, Rcode, python_path, acq):

    view_str = view_names[0]
    for i in range(1, len(view_names)):
        view_str = view_str + ',' + view_names[i]
    R_para = platform + ' ' + python_path + ' ' + raw_file + ' ' + evidence_file + ' ' + design + ' ' + view_str
    print(R_fold + 'Rscript ' + Rcode + 'get_matrix.R ' + R_para)
    os.system(R_fold + 'Rscript ' + Rcode + 'get_matrix.R ' + R_para)

    view_paths = {}
    view_paths.update({'design_file_path':design})

    root_tmp = evidence_file.replace(evidence_file.split('/')[-1], '')
    temp_res_dir = root_tmp + 'res/'

    views = []
    headers = []
    view_datas = []
    idx_g1g2s = []

    comm_pro = []

    for ex_ty in view_names:
        view_paths.update({ex_ty:temp_res_dir + ex_ty + '.tsv'})
        view = pd.read_csv(temp_res_dir + ex_ty + '.tsv', header=0, sep='\t')
        header = np.array(view.columns.values.tolist())

        views.append(view)
        headers.append(header)

        if len(comm_pro) == 0:
            comm_pro = view['Protein']
        else:
            comm_pro = np.intersect1d(comm_pro, view['Protein'])

    for i in range(len(views)):
        comm_pros, idx1, idx2 = np.intersect1d(views[i]['Protein'], comm_pro, return_indices=True)
        views[i] = pd.DataFrame(views[i].values[idx1], columns=views[i].columns.values.tolist())

    designs = pd.read_csv(view_paths['design_file_path'], header=0, sep='\t')

    sample_g1 = designs['sample_name'].values[np.where((designs['condition'].values == g1))[0]]
    sample_g2 = designs['sample_name'].values[np.where((designs['condition'].values == g2))[0]]

    for v in range(len(headers)):
        header = headers[v]
        idx_g1g2 = []
        for sample in list(sample_g1) + list(sample_g2):
            for i in range(len(header)):
                header_pure = header[i].replace('.Spectral.Count', '').replace('.MaxLFQ.Intensity', '').replace(
                    '.Intensity', '')
                header_pure = header_pure.replace('MS.MS.count.', '').replace('Intensity.', '').replace(
                    'LFQ.intensity.', '').replace('Top3.', '')
                if sample == header_pure:
                    idx_g1g2.append(i)

        idx_g1g2 = np.array(np.unique(idx_g1g2), dtype='int')

        idx_g1g2s.append(idx_g1g2)

        view_data = views[v].values[:, idx_g1g2]
        if (view_names[v] == 'dlfq') & (acq == 'DIA'):
            view_data = np.array(view_data, dtype='float')
            view_data[view_data == 0] = np.nan
            view_data = np.log2(view_data)

        view_datas.append(view_data)

    idx_r = np.where(((designs['condition'].values == g1) | (designs['condition'].values == g2)))[0]
    file_name_design = view_paths['design_file_path'].split('/')
    save_path = temp_res_dir + file_name_design[len(file_name_design) - 1]

    pd.DataFrame(designs.values[idx_r, :], columns=designs.columns.values.tolist()).to_csv(save_path, sep='\t',
                                                                                           index=False)

    view_paths['design'] = save_path
    return views, view_datas, idx_g1g2s, view_paths, 'dt'

def get_views_from_view_file_names(view_files, view_names, g1, g2, design, acq):

    view_files = view_files.split(',')
    view_paths = {}
    view_paths.update({'design_file_path': design})

    root_tmp = view_files[0].replace(view_files[0].split('/')[-1], '')
    temp_res_dir = root_tmp + 'res/'

    views = []
    headers = []
    view_datas = []
    idx_g1g2s = []

    comm_pro = []

    for i in range(len(view_names)):
        view_paths.update({view_names[i]: view_files[i]})
        view = pd.read_csv(view_files[i], header=0, sep='\t')
        header = np.array(view.columns.values.tolist())

        views.append(view)
        headers.append(header)

        if len(comm_pro) == 0:
            comm_pro = view['Protein']
        else:
            comm_pro = np.intersect1d(comm_pro, view['Protein'])

    for i in range(len(views)):
        comm_pros, idx1, idx2 = np.intersect1d(views[i]['Protein'], comm_pro, return_indices=True)
        views[i] = pd.DataFrame(views[i].values[idx1], columns=views[i].columns.values.tolist())

    designs = pd.read_csv(view_paths['design_file_path'], header=0, sep='\t')

    sample_g1 = designs['sample_name'].values[np.where((designs['condition'].values == g1))[0]]
    sample_g2 = designs['sample_name'].values[np.where((designs['condition'].values == g2))[0]]

    for v in range(len(headers)):
        header = headers[v]
        idx_g1g2 = []
        for sample in list(sample_g1) + list(sample_g2):
            for i in range(len(header)):
                header_pure = header[i].replace('.Spectral.Count', '').replace('.MaxLFQ.Intensity', '').replace(
                    '.Intensity', '')
                header_pure = header_pure.replace('MS.MS.count.', '').replace('Intensity.', '').replace(
                    'LFQ.intensity.', '').replace('Top3.', '')
                if sample == header_pure:
                    idx_g1g2.append(i)

        idx_g1g2 = np.array(np.unique(idx_g1g2), dtype='int')

        idx_g1g2s.append(idx_g1g2)

        view_data = views[v].values[:, idx_g1g2]
        if (view_names[v] == 'dlfq') & (acq == 'DIA'):
            view_data = np.array(view_data, dtype='float')
            view_data[view_data == 0] = np.nan
            view_data = np.log2(view_data)

        view_datas.append(view_data)

    idx_r = np.where(((designs['condition'].values == g1) | (designs['condition'].values == g2)))[0]
    file_name_design = view_paths['design_file_path'].split('/')
    save_path = temp_res_dir + file_name_design[len(file_name_design) - 1]

    pd.DataFrame(designs.values[idx_r, :], columns=designs.columns.values.tolist()).to_csv(save_path, sep='\t',
                                                                                           index=False)

    view_paths['design'] = save_path
    return views, view_datas, idx_g1g2s, view_paths, 'dt'

def norm_imp_R(view_pd, norm, imput, save_path, R_fold, logT):
    raw_path = './data/raw_' + save_path
    norm_path = './data/norm_' + save_path

    view_pd.to_csv(raw_path, sep='\t', index=False)
    print(R_fold + 'Rscript ' + 'normalization_MV_R.R ' + raw_path + ' ' +
          norm + ' ' + imput + ' ' + norm_path + ' ' + logT)
    os.system(R_fold + 'Rscript ' + 'normalization_MV_R.R ' + raw_path + ' ' +
          norm + ' ' + imput + ' ' + norm_path + ' ' + logT)

    view_norm = pd.read_csv(norm_path, sep='\t', header=0).values

    return view_norm, norm_path

def run_DEA_Rscript(tool, data_path, save_path, R_fold, design_path, g1, g2):
    '''

    :param tool: DEA tool
    :param data_path: the file storing quantification matrix with protein list in first column
    :param save_path: path for saving quantification results
    :return: quantification results
    '''

    if tool == 'hurdle':

        all_res_files = data_path[0]
        for i in range(1, len(data_path)):
            all_res_files = all_res_files + ' ' + data_path[i]

        print(R_fold + 'Rscript ' + 'Hurdle.R ' + all_res_files)
        os.system(R_fold + 'Rscript ' + 'Hurdle.R ' + all_res_files)

        res_file_name = data_path[0]
    else:

        print(R_fold + 'Rscript ' + tool + '_DE_new.R' + ' ' + data_path + ' ' + save_path + ' ' + design_path + ' ' +
              g1 + ' ' + g2)

        os.system(R_fold + 'Rscript ' + tool + '_DE_new.R' + ' ' + data_path + ' ' + save_path + ' ' + design_path + ' ' +
              g1 + ' ' + g2)


        res_file_name = save_path


    if (os.path.exists(res_file_name)):
        res = pd.read_csv(res_file_name, sep=',', header=0)
        idx = np.where(((~np.isnan(np.array(res['pvalue']))) & (
            ~np.isnan(np.array(res['adj.pvalue']))) &
                        (~np.isinf(np.array(res['pvalue']))) & (
                            ~np.isinf(np.array(res['adj.pvalue'])))))[0]

        proteins = np.array(res['protein'])[idx]
        qval = np.array(res['adj.pvalue'])[idx]
        p_value = np.array(res['pvalue'])[idx]

        const = np.array(res['contrast'])[idx]
        logFC = np.array(res['logFC'])[idx]

        score = 1 - np.array(res['adj.pvalue'])[idx]

        return res_file_name, proteins, logFC, p_value, qval, score, const

    else:

        print('no DEA result available !!!')
        return res_file_name, '', '', '', '', '', ''




def processing_type(data, scale):
    if scale >= 0:
        if scale == 0:
            mm = preprocessing.MinMaxScaler(feature_range=(0, 1))
        elif scale == 1:
            mm = preprocessing.MinMaxScaler(feature_range=(-1, 1))
        elif scale == 2:
            mm = preprocessing.StandardScaler()
        data = mm.fit_transform(data)
    elif scale < 0:
        mm = 0
        data = data
    return data, mm

def preprocess_DEA_views(view_pd, design_path, norm, imp, g1, g2, DEA_tool, R_fold, save_name, logT):

    view_norm, norm_path = norm_imp_R(view_pd, norm, imp, save_name, R_fold, logT)

    view_scale, mm = processing_type(view_norm, 0)

    save_path = 'data/' + DEA_tool + '_' + save_name

    res_real_norm = run_DEA_Rscript(DEA_tool, norm_path, save_path, R_fold, design_path, g1, g2)

    return view_scale, res_real_norm, save_path

def CCA_trans_MVSVM(view1_pd,view2_pd, design_path, norm, imp, g1, g2, DEA_tool, R_fold, logT, save_name1, save_name2):
    view_norm_v1, norm1_path = norm_imp_R(view1_pd, norm, imp, save_name1, R_fold, logT)
    view_scale_v1, mm1 = processing_type(view_norm_v1, 0)

    view_norm_v2, norm2_path = norm_imp_R(view2_pd, norm, imp, save_name2, R_fold, logT)
    view_scale_v2, mm2 = processing_type(view_norm_v2, 0)

    view_scale_v1_cca, view_scale_v2_cca = CCA_trans(view_scale_v1, view_scale_v2)

    view_norm_v1_cca = mm1.inverse_transform(view_scale_v1_cca)
    view_norm_v2_cca = mm2.inverse_transform(view_scale_v2_cca)

    out_pd_ori1 = pd.read_csv(norm1_path, sep='\t', header=0, index_col=0)
    out_pd_view1 = pd.DataFrame(np.column_stack((out_pd_ori1.index.values, view_norm_v1_cca)),
                                columns=['Protein'] + out_pd_ori1.columns.values.tolist(),
                                index=None)
    out_pd_view1.to_csv(norm1_path, sep='\t')

    out_pd_ori2 = pd.read_csv(norm2_path, sep='\t', header=0, index_col=0)
    out_pd_view2 = pd.DataFrame(np.column_stack((out_pd_ori2.index.values, view_norm_v2_cca)),
                                columns=['Protein'] + out_pd_ori2.columns.values.tolist(),
                                index=None)
    out_pd_view2.to_csv(norm2_path, sep='\t')

    save_path1 = 'data/' + DEA_tool + '_' + save_name1
    save_path2 = 'data/' + DEA_tool + '_' + save_name2

    res_real_norm1 = run_DEA_Rscript(DEA_tool, norm1_path, save_path1, R_fold, design_path, g1, g2)
    res_real_norm2 = run_DEA_Rscript(DEA_tool, norm2_path, save_path2, R_fold, design_path, g1, g2)

    res_real_hurdle = run_DEA_Rscript('hurdle', ['data/hurdle_res.csv', save_path1, save_path2],
                                      'data/hurdle_res.csv', R_fold, design_path, g1, g2)

    return view_scale_v1_cca, res_real_norm1, view_scale_v2_cca, res_real_norm2, res_real_hurdle


def prepare_for_MVSVM(view1_pd,view2_pd, design_path, norm, imp, g1, g2, DEA_tool, R_fold, logT, cca='T'):

    if cca == 'F':
        view_scale1, res_real_norm1, res_path1 = preprocess_DEA_views(view1_pd, design_path, norm, imp, g1, g2, DEA_tool, R_fold,
                                                           'view1.tsv', logT)

        view_scale2, res_real_norm2, res_path2 = preprocess_DEA_views(view2_pd, design_path, norm, imp, g1, g2, DEA_tool, R_fold,
                                                           'view2.tsv', logT)

        res_real_hurdle = run_DEA_Rscript('hurdle', ['data/hurdle_res.csv', res_path1, res_path2],
                                          'data/hurdle_res.csv', R_fold, design_path, g1, g2)
    elif cca == 'T':

        view_scale1, res_real_norm1, view_scale2, res_real_norm2, res_real_hurdle = CCA_trans_MVSVM(view1_pd,view2_pd,
                                                                                                    design_path, norm,
                                                                                                    imp, g1, g2,
                                                                                                    DEA_tool, R_fold,
                                                                                                    logT, 'view1.tsv',
                                                                                                    'view2.tsv')



    return view_scale1, res_real_norm1, view_scale2, res_real_norm2, res_real_hurdle


def load_data_reproduce(data_idx, platform, g1, g2, ex_ty1, ex_ty2, save_data_root, sf, R_fold, acq):
    maxtrix_folder = 'E:/proteomics/manus4_1/' + acq + '/' + platform + '/'
    Rscript_path = 'D:/R-4.3.1/bin/'

    if platform == 'FragPipe':
        dataset_info_file = 'D:/data/benchmark/data/dataset_info/' + acq + '_' + 'Frag.txt'
    elif platform == 'Maxquant':
        dataset_info_file = 'D:/data/benchmark/data/dataset_info/' + acq + '_' + 'mq.txt'
    elif platform == 'DIANN':
        dataset_info_file = 'D:/data/benchmark/data/dataset_info/' + acq + '_' + 'DIANN.txt'
    elif platform == 'Spectronaut':
        dataset_info_file = 'D:/data/benchmark/data/dataset_info/' + acq + '_' + 'spt.txt'


    save_folder = save_data_root + sf

    isExist = os.path.exists(save_folder)

    if not isExist:
        os.makedirs(save_folder)



    if acq == 'DDA':
        dataset_info = pd.read_csv(dataset_info_file, sep='\t', header=0)
        dataset = dataset_info['dataset'][data_idx]
        true_organism = dataset_info['organism'][data_idx]
        DE_organism = dataset_info['trueDE'][data_idx]
        true_lgfc = dataset_info['true_lgfc'][data_idx]
        all_reported_protein_path = maxtrix_folder + dataset_info['all_protein_file'][
            data_idx]
        design_file_path = maxtrix_folder + dataset + '_' + platform + '_design.tsv'

        view_paths = {'top0': maxtrix_folder + dataset + '_' + platform + '_pro_intensity.tsv',
                      'top3': maxtrix_folder + dataset + '_' + platform + '_top3_pro_intensity.tsv',
                      'maxlfq': maxtrix_folder + dataset + '_' + platform + '_pro_maxlfq.tsv',
                      'dlfq': maxtrix_folder + dataset + '_' + platform + '_dlfq_pro_intensity.tsv',
                      'count': maxtrix_folder + dataset + '_' + platform + '_pro_count.tsv',
                      'design_file_path': design_file_path,
                      'R_fold': R_fold,
                      'save_folder': save_folder,
                      'maxtrix_folder': maxtrix_folder,
                      'Rscript_path': Rscript_path,
                      'dataset_info_path': dataset_info_file,
                      'all_candidate': all_reported_protein_path,
                      'true_lgfc': true_lgfc,
                      'true_organism':true_organism,
                      'DE_organism':DE_organism,
                      'dataset':dataset
                      }

    elif acq == 'DIA':
        dataset_info = pd.read_csv(dataset_info_file, sep='\t', header=0)
        dataset = dataset_info['dataset'][data_idx]
        true_organism = dataset_info['organism'][data_idx]
        DE_organism = dataset_info['trueDE'][data_idx]
        true_lgfc = dataset_info['true_lgfc'][data_idx]
        all_reported_protein_path = maxtrix_folder + dataset_info['all_protein_file'][
            data_idx]
        if platform == 'Spectronaut':
            platform = 'spt'
        design_file_path = maxtrix_folder + dataset + '_' + platform + '_design.tsv'
        view_paths = {'top1': maxtrix_folder + dataset + '_' + platform + '_top1.tsv',
                      'top3': maxtrix_folder + dataset + '_' + platform + '_top3',
                      'maxlfq': maxtrix_folder + dataset + '_' + platform + '_maxlfq.tsv',
                      'dlfq': maxtrix_folder + dataset + '_' + platform + '_dlfq.tsv',
                      'median_polish': maxtrix_folder + dataset + '_' + platform + '_median_polish.tsv',
                      'design_file_path': design_file_path,
                      'R_fold': R_fold,
                      'save_folder': save_folder,
                      'maxtrix_folder': maxtrix_folder,
                      'Rscript_path': Rscript_path,
                      'dataset_info_path': dataset_info_file,
                      'all_candidate': all_reported_protein_path,
                      'true_lgfc': true_lgfc,
                      'true_organism': true_organism,
                      'DE_organism': DE_organism,
                      'dataset': dataset
                      }


    view1 = pd.read_csv(view_paths[ex_ty1], header=0, sep='\t')
    header_v1 = np.array(view1.columns.values.tolist())

    view2 = pd.read_csv(view_paths[ex_ty2], header=0, sep='\t')
    header_v2 = np.array(view2.columns.values.tolist())

    comm_pro, idx1, idx2 = np.intersect1d(view1['Protein'], view2['Protein'], return_indices=True)
    view1 = view1.loc[idx1]
    view2 = view2.loc[idx2]

    designs = pd.read_csv(view_paths['design_file_path'], header=0, sep='\t')

    sample_g1 = designs['sample_name'].values[np.where((designs['condition'].values == g1))[0]]
    sample_g2 = designs['sample_name'].values[np.where((designs['condition'].values == g2))[0]]

    idx_g1g2_v1 = []
    idx_g1g2_v2 = []
    for sample in list(sample_g1) + list(sample_g2):
        for i in range(len(header_v1)):
            if sample in header_v1[i]:
                idx_g1g2_v1.append(i)

        for i in range(len(header_v2)):
            if sample in header_v2[i]:
                idx_g1g2_v2.append(i)

    idx_g1g2_v1 = np.array(np.unique(idx_g1g2_v1), dtype='int')
    idx_g1g2_v2 = np.array(np.unique(idx_g1g2_v2), dtype='int')

    view1_data = view1.values[:, idx_g1g2_v1]
    view2_data = view2.values[:, idx_g1g2_v2]

    if (ex_ty1 == 'dlfq') & (acq == 'DIA'):
        view1_data = np.array(view1_data, dtype='float')
        view1_data[view1_data==0]=np.nan
        view1_data = np.log2(view1_data)

    if (ex_ty2 == 'dlfq') & (acq == 'DIA'):
        view2_data = np.log2(view2_data)

    idx_r = np.where(((designs['condition'].values == g1) | (designs['condition'].values == g2)))[0]
    file_name_design = view_paths['design_file_path'].split('/')
    save_path = save_data_root + sf + file_name_design[len(file_name_design) - 1]

    pd.DataFrame(designs.values[idx_r, :], columns=designs.columns.values.tolist()).to_csv(save_path, sep='\t', index=False)

    # with open(save_path, 'w', newline='') as ws:
    #     writer = csv.writer(ws, delimiter='\t', quoting=csv.QUOTE_MINIMAL)
    #     writer.writerow(np.array(designs.columns.values.tolist()))
    #     writer.writerows(designs[idx_r, :])

    view_paths['design'] = save_path

    return view1, view2, view1_data, view2_data, idx_g1g2_v1, idx_g1g2_v2, view_paths, dataset

def norm_R_reproduce(view1_data, view2_data, norm1, norm2, imput1, imput2, sf, dataset, save_data_root, R_fold, in_file, logT):
    if len(view1_data) > 0:
        save_path_v1 = save_data_root + sf + dataset + '_' + 'raw_view1.csv'
        with open(save_path_v1, 'w', newline='') as ws:
            writer = csv.writer(ws)
            writer.writerows(view1_data)
        save_path_norm_v1 = save_data_root + sf + dataset + '_normed_view1.csv'
    else:
        save_path_v1 = '_'
        save_path_norm_v1 = '_'

    if len(view2_data) > 0:
        save_path_v2 = save_data_root + sf + dataset + '_' + 'raw_view2.csv'
        with open(save_path_v2, 'w', newline='') as ws:
            writer = csv.writer(ws)
            writer.writerows(view2_data)
        save_path_norm_v2 = save_data_root + sf + dataset + '_normed_view2.csv'
    else:
        save_path_v2 = '_'
        save_path_norm_v2 = '_'

    print(in_file['Rscript_path'] + 'Rscript ' + R_fold + 'normalization_MVAE_R.R ' + save_path_v1 + ' ' + save_path_v2
          + ' ' + norm1 + ' ' + norm2 + ' ' + imput1 + ' ' + imput2 + ' ' + save_path_norm_v1 + ' ' + save_path_norm_v2 + ' ' + logT)
    os.system(in_file['Rscript_path'] + 'Rscript ' + R_fold + 'normalization_MVAE_R.R ' + save_path_v1 + ' ' + save_path_v2
          + ' ' + norm1 + ' ' + norm2 + ' ' + imput1 + ' ' + imput2 + ' ' + save_path_norm_v1 + ' ' + save_path_norm_v2 + ' ' + logT)

    if save_path_norm_v1 != '_':
        v1_norm = pd.read_csv(save_path_norm_v1, sep=',', header=None).values
    else:
        v1_norm = []

    if save_path_norm_v2 != '_':
        v2_norm = pd.read_csv(save_path_norm_v2, sep=',', header=None).values
    else:
        v2_norm = []
    return v1_norm, v2_norm

def run_Rscript_new(tool, in_type, inten_ty, imput, normal, in_files, dataset, logT='T', print_lab='T'):
    '''
    :param tool: DE tool
    :param in_type: platform maxquant/fragpipe
    :param inten_ty: ''/MaxLFQ for pipe or ''/'LFQ' for maxquant
    :param imput: imputation method
    :param normal: normalization method
    :param in_files: full file names for input files and save fold
    :return: DE results
    '''
    save_fold = in_files['save_folder'] + 'DEA_res/' + dataset + '/'

    if not os.path.exists(save_fold):
        os.makedirs(save_fold)

    #print_lab = 'T'

    if in_type[0:6] == 'hurdle':
        file_names = in_files['res_files']
        all_res_files = file_names[0]
        for i in range(1, len(file_names)):
            all_res_files = all_res_files + ' ' + file_names[i]

        print(in_files['Rscript_path'] + 'Rscript ' + in_files['R_fold'] + 'Hurdle_rep.R ' + print_lab + ' ' + in_files['true_organism'] +
                  ' ' + in_files['DE_organism'] + ' ' + in_files['true_lgfc'] + ' ' + logT + ' ' + all_res_files)
        os.system(in_files['Rscript_path'] + 'Rscript ' + in_files['R_fold'] + 'Hurdle_rep.R ' + ' ' + print_lab + ' ' + in_files['true_organism'] +
                  ' ' + in_files['DE_organism'] + ' ' + in_files['true_lgfc'] + ' ' + logT + ' ' + all_res_files)

        res_file_name = file_names[0]
    else:

        res_file_name0 = save_fold + dataset + '_' + tool + '_' + in_type + '_' + inten_ty + '_' + imput + '_' + normal + '.csv'
        res_file_name0 = res_file_name0.replace('blank', '')

        #if not os.path.exists(res_file_name0):

        print(in_files['Rscript_path'] + 'Rscript ' + in_files['R_fold'] + tool + '_DE_rep.R' + ' ' + in_files['save_scale'] + ' ' +
                  in_type + ' ' + inten_ty + ' ' +
                  str(imput) + ' ' + str(normal) + ' ' + dataset + ' ' + save_fold + ' ' + print_lab + ' ' + in_files['true_organism'] +
                  ' ' + in_files['DE_organism'] + ' ' + in_files['true_lgfc'] + ' ' + logT + ' ' + in_files['design'])

        os.system(in_files['Rscript_path'] + 'Rscript ' + in_files['R_fold'] + tool + '_DE_rep.R' + ' ' + in_files['save_scale'] + ' ' +
                  in_type + ' ' + inten_ty + ' ' +
                  str(imput) + ' ' + str(normal) + ' ' + dataset + ' ' + save_fold + ' ' + print_lab + ' ' + in_files['true_organism'] +
                  ' ' + in_files['DE_organism'] + ' ' + in_files['true_lgfc'] + ' ' + logT + ' ' + in_files['design'])



        if (inten_ty == 'basic') | (inten_ty == 'blank'):
            inten_ty = ''

        if imput == 'blank':
            imput = ''

        if normal == 'blank':
            normal = ''

        res_file_name = save_fold + dataset + '_' + tool + '_' + in_type + '_' + inten_ty + '_' + imput + '_' + normal + '.csv'

    if (os.path.exists(res_file_name)):
        res = pd.read_csv(res_file_name, sep=',', header=0)
        idx = np.where(((~np.isnan(np.array(res['pvalue']))) & (
            ~np.isnan(np.array(res['adj.pvalue']))) &
                         (~np.isinf(np.array(res['pvalue']))) & (
                            ~np.isinf(np.array(res['adj.pvalue'])))))[0]

        proteins = np.array(res['protein'])[idx]
        qval = np.array(res['adj.pvalue'])[idx]
        p_value = np.array(res['pvalue'])[idx]


        const = np.array(res['contrast'])[idx]
        logFC = np.array(res['logFC'])[idx]

        lgq = np.array([-1 if qval[i] == 0 else -10 * np.log10(qval[i]) for i in range(0, len(qval))], dtype='float')
        max_q = 160
        if max(lgq) == -1:
            lgp = np.array([-1 if p_value[i] == 0 else -10 * np.log10(p_value[i]) for i in range(0, len(p_value))],
                           dtype='float')
            if max(lgp) == -1:
                max_q = 160
            else:
                max_q = max(lgp[np.where((lgp != -1))[0]])
        else:
            max_q = max(lgq[np.where((lgq != -1))[0]])

        lgq[np.where((lgq == -1))[0]] = max_q

        score = 1 - np.array(res['adj.pvalue'])[idx]
        score_lgq = np.array([abs(logFC[i] * lgq[i]) for i in range(len(lgq))], dtype='float')

        if print_lab == 'T':

            real_logFC = np.array(res['TlogFC'])[idx]
            organisms = np.array(res['organism'])[idx]
            label = np.array(res['label'])[idx]
        else:
            real_logFC = []
            organisms = []
            label = []

        return res_file_name, proteins, organisms, score, score_lgq, label, const, logFC, real_logFC, p_value, qval

def CCA_trans(view1_data, view2_data):

    view1_data[np.isnan(view1_data)] = 0
    view2_data[np.isnan(view2_data)] = 0

    num0_v1 = np.array([len(np.where(((view1_data[i, :] == 0) | (np.isnan(view1_data[i, :]))))[0]) for i in
                        range(len(view1_data[:, 0]))])
    num0_v2 = np.array([len(np.where(((view2_data[i, :] == 0) | (np.isnan(view2_data[i, :]))))[0]) for i in
                        range(len(view2_data[:, 0]))])

    all_missing_v1 = np.where((num0_v1 == len(view1_data[0, :])))[0]
    all_missing_v2 = np.where((num0_v2 == len(view2_data[0, :])))[0]

    mirror_v1_idx = np.setdiff1d(all_missing_v1, all_missing_v2)
    mirror_v2_idx = np.setdiff1d(all_missing_v2, all_missing_v1)

    no_missing_v1 = np.where((num0_v1 == 0))[0]
    no_missing_v2 = np.where((num0_v2 == 0))[0]

    train_idx = np.intersect1d(no_missing_v1, no_missing_v2)

    train_view1 = view1_data[train_idx]
    train_view2 = view2_data[train_idx]

    view1_data_cca = copy.deepcopy(view1_data)
    view2_data_cca = copy.deepcopy(view2_data)

    mses = []
    Ks = []

    for K in range(1, len(view1_data[0, :])):
        logo = KFold(n_splits=3, shuffle=True)
        mse = 0
        Ks.append(K)
        for train_index, test_index in logo.split(train_view1[:, 0]):
            train_x = train_view1[train_index]
            train_y = train_view2[train_index]

            test_x = train_view1[test_index]
            test_y = train_view2[test_index]

            cca1 = []
            cca1 = CCA(n_components=K)

            cca2 = []
            cca2 = CCA(n_components=K)

            cca1.fit(train_x, train_y) # predict view2 from view1
            cca2.fit(train_y, train_x) # predict view1 from view2

            mse += mean_squared_error(cca1.predict(test_x), test_y) + mean_squared_error(cca2.predict(test_y), test_x)
        mses.append(mse / 3)

    Kbest = Ks[np.where((np.array(mses) == min(np.array(mses))))[0][0]]

    cca1 = CCA(n_components=Kbest).fit(train_view1, train_view2)
    cca2 = CCA(n_components=Kbest).fit(train_view2, train_view1)

    if (len(mirror_v1_idx)>0):
        view1_data_cca[mirror_v1_idx, :] = cca2.predict(view2_data_cca[mirror_v1_idx, :])

    if (len(mirror_v2_idx) > 0):
        view2_data_cca[mirror_v2_idx, :] = cca1.predict(view1_data_cca[mirror_v2_idx, :])

    view1_data_cca[view1_data_cca==0] = np.nan
    view2_data_cca[view2_data_cca == 0] = np.nan

    return view1_data_cca, view2_data_cca

def cal_logFC(mat, col_nm, g1, g2, design_pth, logT, view_name):
    mat[mat==0] = np.nan
    idx_g1 = []
    idx_g2 = []
    design = pd.read_csv(design_pth, sep='\t')
    for i in range(len(col_nm)):
        col_nm_i = col_nm[i]
        cod_i = []
        for j in range(len(design['sample_name'])):
            if design['sample_name'][j] in col_nm_i:
                cod_i = design['condition'][j]
        if (g1 in cod_i):
            idx_g1.append(i)
        elif (g2 in cod_i):
            idx_g2.append(i)

    idx_g1 = np.array(idx_g1, dtype='int')
    idx_g2 = np.array(idx_g2, dtype='int')

    logFCs = []
    confis = []
    for i in range(len(mat[:, 0])):
        if i==26:
            a = 0
        confi_num = np.where(((np.array(mat[i, :], dtype='float')!=0) & (~np.isnan(np.array(mat[i, :], dtype='float')))))[0]
        confi = len(confi_num)/len(mat[i, :])
        confis.append(confi)
        exp_g1 = np.array(mat[i, idx_g1], dtype='float')
        exp_g2 = np.array(mat[i, idx_g2], dtype='float')

        exp_g1 = exp_g1[np.where((~np.isnan(exp_g1)))[0]]
        exp_g2 = exp_g2[np.where((~np.isnan(exp_g2)))[0]]

        if len(exp_g1)!=0:
            if logT == 'T':
                exp_g1 = exp_g1[np.where((exp_g1!=0))[0]]
                if len(exp_g1)==0:
                    avg_g1 = 0
                else:
                    avg_g1 = np.mean(np.log2(exp_g1))
            else:
                avg_g1 = np.mean(exp_g1)
        else:
            avg_g1 = 0

        if len(exp_g2)!=0:
            if logT == 'T':
                exp_g2 = exp_g2[np.where((exp_g2 != 0))[0]]
                if len(exp_g2)==0:
                    avg_g2 = 0
                else:
                    avg_g2 = np.mean(np.log2(exp_g2))
            else:
                avg_g2 = np.mean(exp_g2)
        else:
            avg_g2 = 0

        if view_name == 'count':
            if (avg_g2==0) & (avg_g1==0):
                logFCs.append(0)
            elif (avg_g1==0) & (avg_g2>0):
                logFCs.append(3)
            else:
                logFCs.append(np.log2(avg_g2/avg_g1))
        else:
            logFCs.append(avg_g2-avg_g1)
    return np.array([logFCs]).T, np.array([confis]).T

def mv_logFC(logFC_v1, logFC_v2):
    mv_logFC = []
    for i in range(len(logFC_v1[:, 0])):
        if(abs(logFC_v1[i, 0])>=abs(logFC_v2[i, 0])):
            mv_logFC.append(logFC_v1[i, 0])
        else:
            mv_logFC.append(logFC_v2[i, 0])

    return np.array([mv_logFC]).T

def get_datasets_reproduce(flag, data_idx, in_type, norm1, norm2, imput1, imput2, g1, g2, acq, ex_ty1,
                         ex_ty2, save_data_root, R_fold, dea1, dea2, logT):

    sf = 'temp' + str(flag) + '/'
    if not os.path.exists(sf):
        os.makedirs(sf)
    view1, view2, view1_data, view2_data, idx_g1g2_v1, idx_g1g2_v2, view_paths, dataset = load_data_reproduce(data_idx,
                                                                                                           in_type,
                                                                                                           g1, g2,
                                                                                                           ex_ty1,
                                                                                                           ex_ty2,
                                                                                                           save_data_root,
                                                                                                           sf, R_fold, acq)

    v1_scale0, mm10 = processing_type(view1_data, 0)
    v2_scale0, mm20 = processing_type(view2_data, 0)

    logFC_v1_norm = cal_logFC(view1_data, np.array(view1.columns.values.tolist())[idx_g1g2_v1], g1, g2,
                                  view_paths['design'], logT)
    logFC_v2_norm = cal_logFC(view2_data, np.array(view2.columns.values.tolist())[idx_g1g2_v2], g1, g2,
                                  view_paths['design'], logT)

    v1_scale0_cca, v2_scale0_cca = CCA_trans(v1_scale0, v2_scale0)
    v1_cca = mm10.inverse_transform(v1_scale0_cca)
    v2_cca = mm20.inverse_transform(v2_scale0_cca)

    logFC_v1_norm_cca = cal_logFC(v1_cca, np.array(view1.columns.values.tolist())[idx_g1g2_v1], g1, g2, view_paths['design'], logT)
    logFC_v2_norm_cca = cal_logFC(v2_cca, np.array(view2.columns.values.tolist())[idx_g1g2_v2], g1, g2, view_paths['design'], logT)

    logFC_norm = mv_logFC(logFC_v1_norm, logFC_v2_norm)
    logFC_norm_cca = mv_logFC(logFC_v1_norm_cca, logFC_v2_norm_cca)

    v1_norm, v2_norm = norm_R_reproduce(view1_data, view2_data, norm1, norm2, imput1, imput2, sf, dataset,
                                    save_data_root, R_fold, view_paths, logT)

    v1_norm_cca, v2_norm_cca = norm_R_reproduce(v1_cca, v2_cca, norm1, norm2, imput1, imput2, sf, dataset,
                                        save_data_root, R_fold, view_paths, logT)

    ## save real normed data
    v1_file_path = view_paths[ex_ty1].split('/')
    normed_real_v1 = copy.deepcopy(view1.values)
    v1_scale, mm1 = processing_type(v1_norm, 0)
    v1_scale_cca, mm3 = processing_type(v1_norm_cca, 0)

    v2_file_path = view_paths[ex_ty2].split('/')
    normed_real_v2 = copy.deepcopy(view2.values)
    v2_scale, mm2 = processing_type(v2_norm, 0)
    v2_scale_cca, mm3 = processing_type(v2_norm_cca, 0)

    save_scale_v1 = save_data_root + sf + 'real_normed_v1_' + v1_file_path[len(v1_file_path) - 1]
    normed_real_v1[:, idx_g1g2_v1] = v1_norm
    pd.DataFrame(normed_real_v1, columns=view1.columns.values.tolist()).to_csv(save_scale_v1, sep='\t', index=False)
    view_paths.update({'save_scale': save_scale_v1})
    # tool, in_type, inten_ty, imput, normal, in_files, dataset, logT='T'
    res_real_norm_v1 = run_Rscript_new(dea1, in_type, ex_ty1, 'blank', 'blank', view_paths, dataset, 'F')

    save_scale_v2 = save_data_root + sf + 'real_normed_v2_' + v1_file_path[len(v2_file_path) - 1]
    normed_real_v2[:, idx_g1g2_v2] = v2_norm
    pd.DataFrame(normed_real_v2, columns=view2.columns.values.tolist()).to_csv(save_scale_v2, sep='\t', index=False)
    view_paths.update({'save_scale': save_scale_v2})

    res_real_norm_v2 = run_Rscript_new(dea2, in_type, ex_ty2, 'blank', 'blank', view_paths, dataset,'F')

    in_files_hurdle = copy.deepcopy(view_paths)
    save_f = save_data_root + sf + 'hurdle_real.csv'
    in_files_hurdle['res_files'] = [save_f, res_real_norm_v1[0], res_real_norm_v2[0]]
    res_hurdle = run_Rscript_new('hurdle', 'hurdle', 'basic', 'blank', 'blank', in_files_hurdle, dataset)

    normed_real_cca_v1 = copy.deepcopy(view1.values)
    normed_real_cca_v2 = copy.deepcopy(view2.values)
    normed_real_cca_v1[:, idx_g1g2_v1] = v1_norm_cca
    normed_real_cca_v2[:, idx_g1g2_v2] = v2_norm_cca

    save_scale_cca_v1 = save_data_root + sf + 'real_normed_cca_v1_' + v1_file_path[len(v1_file_path) - 1]
    normed_real_cca_v1[:, idx_g1g2_v1] = v1_norm_cca
    pd.DataFrame(normed_real_cca_v1, columns=view1.columns.values.tolist()).to_csv(save_scale_cca_v1, sep='\t', index=False)
    view_paths.update({'save_scale': save_scale_cca_v1})

    res_real_norm_cca_v1 = run_Rscript_new(dea1, in_type, ex_ty1, 'blank', 'blank', view_paths, dataset, 'F')

    save_scale_cca_v2 = save_data_root + sf + 'real_normed_cca_v2_' + v1_file_path[len(v2_file_path) - 1]
    normed_real_cca_v2[:, idx_g1g2_v2] = v2_norm_cca
    pd.DataFrame(normed_real_cca_v2, columns=view2.columns.values.tolist()).to_csv(save_scale_cca_v2, sep='\t', index=False)
    view_paths.update({'save_scale': save_scale_cca_v2})

    res_real_norm_cca_v2 = run_Rscript_new(dea2, in_type, ex_ty2, 'blank', 'blank', view_paths, dataset, 'F')

    in_files_hurdle_cca = copy.deepcopy(view_paths)
    save_f = save_data_root + sf + 'hurdle_real_cca.csv'
    in_files_hurdle_cca['res_files'] = [save_f, res_real_norm_cca_v1[0], res_real_norm_cca_v2[0]]
    res_hurdle_cca = run_Rscript_new('hurdle', 'hurdle', 'basic', 'blank', 'blank', in_files_hurdle_cca, dataset)


    return (v1_scale, v2_scale, v1_scale_cca, v2_scale_cca, res_real_norm_v1, res_real_norm_v2, res_real_norm_cca_v1,
             res_real_norm_cca_v2, res_hurdle, res_hurdle_cca, mm1, mm2, view1, view2, view_paths, logFC_norm, logFC_norm_cca)


def load_data_reproduce_mv(data_idx, platform, g1, g2, ex_tys, save_data_root, sf, R_fold, acq):

    maxtrix_folder = 'E:/proteomics/manus4_1/' + acq + '/' + platform + '/'
    Rscript_path = 'D:/R-4.3.1/bin/'

    if platform == 'FragPipe':
        dataset_info_file = 'D:/data/benchmark/data/dataset_info/' + acq + '_' + 'Frag.txt'
    elif platform == 'Maxquant':
        dataset_info_file = 'D:/data/benchmark/data/dataset_info/' + acq + '_' + 'mq.txt'
    elif platform == 'DIANN':
        dataset_info_file = 'D:/data/benchmark/data/dataset_info/' + acq + '_' + 'DIANN.txt'
    elif platform == 'Spectronaut':
        dataset_info_file = 'D:/data/benchmark/data/dataset_info/' + acq + '_' + 'spt.txt'

    save_folder = save_data_root + sf

    isExist = os.path.exists(save_folder)

    if not isExist:
        os.makedirs(save_folder)

    if acq == 'DDA':
        dataset_info = pd.read_csv(dataset_info_file, sep='\t', header=0)
        dataset = dataset_info['dataset'][data_idx]
        true_organism = dataset_info['organism'][data_idx]
        DE_organism = dataset_info['trueDE'][data_idx]
        true_lgfc = dataset_info['true_lgfc'][data_idx]
        all_reported_protein_path = maxtrix_folder + dataset_info['all_protein_file'][
            data_idx]
        design_file_path = maxtrix_folder + dataset + '_' + platform + '_design.tsv'

        view_paths = {'top0': maxtrix_folder + dataset + '_' + platform + '_pro_intensity.tsv',
                      'top3': maxtrix_folder + dataset + '_' + platform + '_top3_pro_intensity.tsv',
                      'maxlfq': maxtrix_folder + dataset + '_' + platform + '_pro_maxlfq.tsv',
                      'dlfq': maxtrix_folder + dataset + '_' + platform + '_dlfq_pro_intensity.tsv',
                      'count': maxtrix_folder + dataset + '_' + platform + '_pro_count.tsv',
                      'design_file_path': design_file_path,
                      'R_fold': R_fold,
                      'save_folder': save_folder,
                      'maxtrix_folder': maxtrix_folder,
                      'Rscript_path': Rscript_path,
                      'dataset_info_path': dataset_info_file,
                      'all_candidate': all_reported_protein_path,
                      'true_lgfc': true_lgfc,
                      'true_organism': true_organism,
                      'DE_organism': DE_organism,
                      'dataset': dataset
                      }

    elif acq == 'DIA':
        dataset_info = pd.read_csv(dataset_info_file, sep='\t', header=0)
        dataset = dataset_info['dataset'][data_idx]
        true_organism = dataset_info['organism'][data_idx]
        DE_organism = dataset_info['trueDE'][data_idx]
        true_lgfc = dataset_info['true_lgfc'][data_idx]
        all_reported_protein_path = maxtrix_folder + dataset_info['all_protein_file'][
            data_idx]
        if platform == 'Spectronaut':
            platform = 'spt'
        design_file_path = maxtrix_folder + dataset + '_' + platform + '_design.tsv'
        view_paths = {'top1': maxtrix_folder + dataset + '_' + platform + '_top1.tsv',
                      'top3': maxtrix_folder + dataset + '_' + platform + '_top3.tsv',
                      'maxlfq': maxtrix_folder + dataset + '_' + platform + '_maxlfq.tsv',
                      'dlfq': maxtrix_folder + dataset + '_' + platform + '_dlfq.tsv',
                      'median_polish': maxtrix_folder + dataset + '_' + platform + '_median_polish.tsv',
                      'design_file_path': design_file_path,
                      'R_fold': R_fold,
                      'save_folder': save_folder,
                      'maxtrix_folder': maxtrix_folder,
                      'Rscript_path': Rscript_path,
                      'dataset_info_path': dataset_info_file,
                      'all_candidate': all_reported_protein_path,
                      'true_lgfc': true_lgfc,
                      'true_organism': true_organism,
                      'DE_organism': DE_organism,
                      'dataset': dataset
                      }

    views = []
    headers = []
    view_datas = []
    idx_g1g2s = []

    comm_pro = []

    for ex_ty in ex_tys:
        view = pd.read_csv(view_paths[ex_ty], header=0, sep='\t')
        header = np.array(view.columns.values.tolist())

        views.append(view)
        headers.append(header)

        if len(comm_pro)==0:
            comm_pro = view['Protein']
        else:
            comm_pro = np.intersect1d(comm_pro, view['Protein'])

    for i in range(len(views)):
        comm_pros, idx1, idx2 = np.intersect1d(views[i]['Protein'], comm_pro, return_indices=True)
        views[i] = pd.DataFrame(views[i].values[idx1], columns=views[i].columns.values.tolist())

    designs = pd.read_csv(view_paths['design_file_path'], header=0, sep='\t')

    sample_g1 = designs['sample_name'].values[np.where((designs['condition'].values == g1))[0]]
    sample_g2 = designs['sample_name'].values[np.where((designs['condition'].values == g2))[0]]

    for v in range(len(headers)):
        header = headers[v]
        idx_g1g2 = []
        for sample in list(sample_g1) + list(sample_g2):
            for i in range(len(header)):
                header_pure = header[i].replace('.Spectral.Count', '').replace('.MaxLFQ.Intensity', '').replace('.Intensity', '')
                header_pure = header_pure.replace('MS.MS.count.', '').replace('Intensity.', '').replace('LFQ.intensity.', '').replace('Top3.', '')
                if sample == header_pure:
                    idx_g1g2.append(i)

        idx_g1g2 = np.array(np.unique(idx_g1g2), dtype='int')

        idx_g1g2s.append(idx_g1g2)

        view_data = views[v].values[:, idx_g1g2]
        if (ex_tys[v] == 'dlfq') & (acq == 'DIA'):
            view_data = np.array(view_data, dtype='float')
            view_data[view_data == 0] = np.nan
            view_data = np.log2(view_data)

        view_datas.append(view_data)


    idx_r = np.where(((designs['condition'].values == g1) | (designs['condition'].values == g2)))[0]
    file_name_design = view_paths['design_file_path'].split('/')
    save_path = save_data_root + sf + file_name_design[len(file_name_design) - 1]

    pd.DataFrame(designs.values[idx_r, :], columns=designs.columns.values.tolist()).to_csv(save_path, sep='\t',
                                                                                           index=False)

    view_paths['design'] = save_path
    return views, view_datas, idx_g1g2s, view_paths, dataset

def mv_logFC_mv(logFC_v_norms):
    mv_logFC = []
    for i in range(len(logFC_v_norms[0][:, 0])):
        logfcs = np.array([logFC_v_norms[j][i, 0] for j in range(len(logFC_v_norms))], dtype='float')
        max_idx = np.where((abs(logfcs)==max(abs(logfcs))))[0]
        mv_logFC.append(logfcs[max_idx[0]])

    return np.array([mv_logFC]).T

def mean_norm(data):
    data[data==0] = np.nan
    rowmean = np.nanmean(data, axis=1)
    norm_data = copy.deepcopy(data)
    for i in range(len(rowmean)):
        if (rowmean[i] != 0) & (~np.isnan(rowmean[i])):
            norm_data[i, :] = (norm_data[i, :]-rowmean[i])/rowmean[i]

    out_data = norm_data
    return out_data, rowmean

def DEA_single_hurdle(views, view_datas, idx_g1g2s, ex_tys, view_paths, logTs, g1, g2, norms, imputs, deas, save_folder,
                      dataset, ci, R_code, platform):
    v_scales = []
    logFC_v_norms = []
    mms = []
    v_norms = []
    res_real_norm_vs = []
    res_real_norm_v_paths = []

    for v in range(len(ex_tys)):

        logFC_v_norm, confi = cal_logFC(view_datas[v], np.array(views[v].columns.values.tolist())[idx_g1g2s[v]], g1, g2,
                                 view_paths['design'], logTs[v], ex_tys[v])
        logFC_v_norms.append(logFC_v_norm)

        if len(view_datas[v]) > 0:
            save_path_v = save_folder + dataset + '_' + 'raw_view' + str(v) + '_' + ci + '.csv'
            with open(save_path_v, 'w', newline='') as ws:
                writer = csv.writer(ws)
                writer.writerows(view_datas[v])
            save_path_norm_v = save_folder + dataset + '_normed_view' + str(v) + '_' + ci + '.csv'
        else:
            save_path_v = '_'
            save_path_norm_v = '_'

        os.system(view_paths[
                      'Rscript_path'] + 'Rscript ' + R_code + 'normalization_MVAE_R1.R ' + save_path_v + ' ' + norms[
                      v] +
                  ' ' + imputs[v] + ' ' + save_path_norm_v + ' ' + logTs[v])

        if save_path_norm_v != '_':
            v_norm = pd.read_csv(save_path_norm_v, sep=',', header=None).values
        else:
            v_norm = []

        v_norms.append(v_norm)

        v_file_path = view_paths[ex_tys[v]].split('/')
        normed_real_v = copy.deepcopy(views[v].values)


        save_scale_v = save_folder + 'real_normed_v' + str(v)  + '_' + ci + '_' + v_file_path[len(v_file_path) - 1]
        normed_real_v[:, idx_g1g2s[v]] = v_norm
        pd.DataFrame(normed_real_v, columns=views[v].columns.values.tolist()).to_csv(save_scale_v, sep='\t',
                                                                                     index=False)
        view_paths.update({'save_scale': save_scale_v})
        #v_norm, rowmean = mean_norm(v_norm)

        v_scale1, mm1 = processing_type(v_norm, 0)
        mms.append(mm1)
        v_scales.append(v_scale1)
        res_real_norm_v = run_Rscript_new(deas[v], platform, ex_tys[v], 'blank', 'blank', view_paths, dataset+ci, 'F')
        res_real_norm_vs.append(res_real_norm_v)
        res_real_norm_v_paths.append(res_real_norm_v[0])

    if ci!='':
        logFC_norm = mv_logFC_mv(logFC_v_norms)
    else:
        logFC_norm = []

    in_files_hurdle = copy.deepcopy(view_paths)
    save_f = save_folder + dataset + '_' + platform + 'hurdle_real' + '_' + ci + '.csv'
    hurdle_path = [save_f]
    hurdle_path.extend(res_real_norm_v_paths)
    in_files_hurdle['res_files'] = hurdle_path
    res_hurdle = run_Rscript_new('hurdle', 'hurdle', 'basic', 'blank', 'blank', in_files_hurdle, dataset+ci)

    return [v_scales, res_real_norm_vs, res_hurdle, mms, view_paths, logFC_norm]

def DEA_single_hurdle_new(views, view_datas, idx_g1g2s, ex_tys, view_paths, logTs, g1, g2, norms, imputs, deas, save_folder,
                      dataset, ci, R_code, platform):

    v_scales = []
    logFC_v_norms = []
    mms = []
    v_norms = []
    res_real_norm_vs = []
    res_real_norm_v_paths = []

    for v in range(len(ex_tys)):

        logFC_v_norm, confi = cal_logFC(view_datas[v], np.array(views[v].columns.values.tolist())[idx_g1g2s[v]], g1, g2,
                                 view_paths['design'], logTs[v], ex_tys[v])
        logFC_v_norms.append(logFC_v_norm)

        if len(view_datas[v]) > 0:
            save_path_v = save_folder + dataset + '_' + 'raw_view' + str(v) + '_' + ci + '.csv'
            with open(save_path_v, 'w', newline='') as ws:
                writer = csv.writer(ws)
                writer.writerows(view_datas[v])
            save_path_norm_v = save_folder + dataset + '_normed_view' + str(v) + '_' + ci + '.csv'
        else:
            save_path_v = '_'
            save_path_norm_v = '_'

        print(view_paths[
                      'Rscript_path'] + 'Rscript ' + R_code + 'normalization_MVAE_R1.R ' + save_path_v + ' ' + norms[
                      v] +
                  ' ' + imputs[v] + ' ' + save_path_norm_v + ' ' + logTs[v])
        os.system(view_paths[
                      'Rscript_path'] + 'Rscript ' + R_code + 'normalization_MVAE_R1.R ' + save_path_v + ' ' + norms[
                      v] +
                  ' ' + imputs[v] + ' ' + save_path_norm_v + ' ' + logTs[v])

        if save_path_norm_v != '_':
            v_norm = pd.read_csv(save_path_norm_v, sep=',', header=None).values
        else:
            v_norm = []

        v_norms.append(v_norm)


        v_file_path = view_paths[ex_tys[v]].replace('\\','/').split('/')
        normed_real_v = copy.deepcopy(views[v].values)
        #v_norm, rowmean = mean_norm(v_norm)

        save_scale_v = save_folder + 'real_normed_v' + str(v)  + '_' + ci + '_' + v_file_path[len(v_file_path) - 1]
        normed_real_v[:, idx_g1g2s[v]] = v_norm
        pd.DataFrame(normed_real_v, columns=views[v].columns.values.tolist()).to_csv(save_scale_v, sep='\t',
                                                                                     index=False)
        view_paths.update({'save_scale': save_scale_v})

        v_scale1, mm1 = processing_type(v_norm, 0)
        mms.append(mm1)
        v_scales.append(v_scale1)

        res_real_norm_v = run_Rscript_new(deas[v], platform, ex_tys[v], 'blank', 'blank', view_paths, dataset+ci, 'F', print_lab='F')
        res_real_norm_vs.append(res_real_norm_v)
        res_real_norm_v_paths.append(res_real_norm_v[0])

    if ci!='':
        logFC_norm = mv_logFC_mv(logFC_v_norms)
    else:
        logFC_norm = []

    in_files_hurdle = copy.deepcopy(view_paths)
    save_f = save_folder + dataset + '_' + platform + 'hurdle_real' + '_' + ci + '.csv'
    hurdle_path = [save_f]
    hurdle_path.extend(res_real_norm_v_paths)
    in_files_hurdle['res_files'] = hurdle_path
    res_hurdle = run_Rscript_new('hurdle', 'hurdle', 'basic', 'blank', 'blank', in_files_hurdle, dataset+ci,'F', print_lab='F')

    return [v_scales, res_real_norm_vs, res_hurdle, mms, view_paths, logFC_norm]

def get_datasets_reproduce_mv(flag, data_idx, in_type, norms, imputs, g1, g2, acq, ex_tys,
                         save_data_root, R_fold, deas, logTs, seed, pg_ty, Rscript_path, cross_imp):
    sf = 'temp' + str(flag) + '/'

    data_fold = 'data/'
    maxtrix_folder = data_fold + acq + '/' + in_type + '/'
    # Rscript_path = 'D:/R-4.3.1/bin/'
    platform = in_type
    if platform == 'FragPipe':
        dataset_info_file = data_fold + acq + '_' + 'Frag.txt'
    elif platform == 'Maxquant':
        dataset_info_file = data_fold + acq + '_' + 'mq.txt'
    elif platform == 'DIANN':
        dataset_info_file = data_fold + acq + '_' + 'DIANN.txt'
    elif platform == 'Spectronaut':
        dataset_info_file = data_fold + acq + '_' + 'spt.txt'

    save_folder = save_data_root + sf

    isExist = os.path.exists(save_folder)

    if not isExist:
        os.makedirs(save_folder)

    if acq == 'DDA':
        dataset_info = pd.read_csv(dataset_info_file, sep='\t', header=0)
        dataset = dataset_info['dataset'][data_idx]
        true_organism = dataset_info['organism'][data_idx]
        DE_organism = dataset_info['trueDE'][data_idx]
        true_lgfc = dataset_info['true_lgfc'][data_idx]
        all_reported_protein_path = maxtrix_folder + dataset_info['all_protein_file'][
            data_idx]
        design_file_path = maxtrix_folder + dataset + '_' + platform + '_design.tsv'

        view_paths = {'top0': maxtrix_folder + dataset + '_' + platform + '_pro_intensity.tsv',
                      'top3': maxtrix_folder + dataset + '_' + platform + '_top3_pro_intensity.tsv',
                      'maxlfq': maxtrix_folder + dataset + '_' + platform + '_pro_maxlfq.tsv',
                      'dlfq': maxtrix_folder + dataset + '_' + platform + '_dlfq_pro_intensity.tsv',
                      'count': maxtrix_folder + dataset + '_' + platform + '_pro_count.tsv',
                      'design_file_path': design_file_path,
                      'R_fold': R_fold,
                      'save_folder': save_folder,
                      'maxtrix_folder': maxtrix_folder,
                      'Rscript_path': Rscript_path,
                      'dataset_info_path': dataset_info_file,
                      'all_candidate': all_reported_protein_path,
                      'true_lgfc': true_lgfc,
                      'true_organism': true_organism,
                      'DE_organism': DE_organism,
                      'dataset': dataset
                      }

    elif acq == 'DIA':
        dataset_info = pd.read_csv(dataset_info_file, sep='\t', header=0)
        dataset = dataset_info['dataset'][data_idx]
        true_organism = dataset_info['organism'][data_idx]
        DE_organism = dataset_info['trueDE'][data_idx]
        true_lgfc = dataset_info['true_lgfc'][data_idx]
        all_reported_protein_path = maxtrix_folder + dataset_info['all_protein_file'][
            data_idx]
        if platform == 'Spectronaut':
            platform = 'spt'
        design_file_path = maxtrix_folder + dataset + '_' + platform + '_design.tsv'
        view_paths = {'top1': maxtrix_folder + dataset + '_' + platform + '_top1.tsv',
                      'top3': maxtrix_folder + dataset + '_' + platform + '_top3.tsv',
                      'maxlfq': maxtrix_folder + dataset + '_' + platform + '_maxlfq.tsv',
                      'dlfq': maxtrix_folder + dataset + '_' + platform + '_dlfq.tsv',
                      'median_polish': maxtrix_folder + dataset + '_' + platform + '_median_polish.tsv',
                      'design_file_path': design_file_path,
                      'R_fold': R_fold,
                      'save_folder': save_folder,
                      'maxtrix_folder': maxtrix_folder,
                      'Rscript_path': Rscript_path,
                      'dataset_info_path': dataset_info_file,
                      'all_candidate': all_reported_protein_path,
                      'true_lgfc': true_lgfc,
                      'true_organism': true_organism,
                      'DE_organism': DE_organism,
                      'dataset': dataset
                      }

    file_path = ''
    view_str = ''
    for v in range(len(ex_tys)):
        if v == 0:
            file_path = view_paths[ex_tys[v]]
            view_str = ex_tys[v]
        else:
            file_path = file_path + ',' + view_paths[ex_tys[v]]
            view_str = view_str + ',' + ex_tys[v]

    python_path = ''

    (views, view_datas, views_intersect, view_datas_intersect, idx_g1g2s, view_paths_dt, dataset, views_ci, view_datas_ci,
     view_paths_ci, idx_g1g2s_ci, out_design, view_logFCs,
        view_confis, view_logFCs_intersect, view_confis_intersect, views_union, view_datas_union,
        view_logFCs_union, view_confis_union) = get_multi_view_data(
        'views',
        pg_ty,
        file_path, g1,
        g2, view_paths['design_file_path'], view_str,
        Rscript_path, R_fold, python_path, acq, in_type,
        save_folder, logTs, cross_imp, seed)

    view_paths.update({'design': view_paths_dt['design'], 'logT': view_paths_dt['logT']})

    res_ori = []
    res_intersect = []
    # res_ori = DEA_single_hurdle(views, view_datas,
    #                                                                                                idx_g1g2s, ex_tys,
    #                                                                                                view_paths, logTs,
    #                                                                                                g1, g2, norms,
    #                                                                                                imputs, deas,
    #                                                                                                save_folder,
    #                                                                                                view_paths['dataset'], '', R_fold,
    #                                                                                                platform)
    # #
    # res_intersect= DEA_single_hurdle(views_intersect, view_datas_intersect,
    #                                                                                         idx_g1g2s, ex_tys,
    #                                                                                         view_paths, logTs,
    #                                                                                         g1, g2, norms,
    #                                                                                         imputs, deas,
    #                                                                                         save_folder,
    #                                                                                         view_paths['dataset'], 'intersect', R_fold,
    #                                                                                         platform)

    res_union = DEA_single_hurdle(views_union, view_datas_union,
                                      idx_g1g2s, ex_tys,
                                      view_paths, logTs,
                                      g1, g2, norms,
                                      imputs, deas,
                                      save_folder,
                                      view_paths['dataset'], 'union', R_fold,
                                      platform)

    if len(view_datas_ci)>0:

        res_ci= DEA_single_hurdle(
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
    return (res_ori, res_intersect, res_ci, views, view_datas, views_intersect, view_datas_intersect,
            views_ci, view_datas_ci, save_folder, view_paths, res_union, views_union, view_datas_union)

def get_datasets_new(views, view_datas, idx_g1g2s, view_paths, platform, norms, imputs, g1, g2, acq, save_data_root,
                     R_fold, R_code, deas, logTs, view_names, dataset):
    sf = save_data_root + 'res/'
    if not os.path.exists(sf):
        os.makedirs(sf)

    v_scales = []
    logFC_v_norms = []
    mms = []
    v_norms = []
    res_real_norm_vs = []
    res_real_norm_v_paths = []

    for v in range(len(view_names)):

        logFC_v_norm = cal_logFC(view_datas[v], np.array(views[v].columns.values.tolist())[idx_g1g2s[v]], g1, g2,
                              view_paths['design'], logTs[v], view_names[v])
        logFC_v_norms.append(logFC_v_norm)

        if len(view_datas[v]) > 0:
            save_path_v = sf + 'dt_' + 'raw_view' + str(v) + '.csv'
            with open(save_path_v, 'w', newline='') as ws:
                writer = csv.writer(ws)
                writer.writerows(view_datas[v])
            save_path_norm_v = sf + 'dt_normed_view' + str(v) + '.csv'
        else:
            save_path_v = '_'
            save_path_norm_v = '_'

        os.system(R_fold + 'Rscript ' + R_code + 'normalization_MVAE_R1.R ' + save_path_v + ' ' + norms[v] +
                  ' ' + imputs[v] + ' ' + save_path_norm_v + ' ' + logTs[v])

        if save_path_norm_v != '_':
            v_norm = pd.read_csv(save_path_norm_v, sep=',', header=None).values
        else:
            v_norm = []

        v_norms.append(v_norm)

        v_file_path = view_paths[view_names[v]].split('/')
        normed_real_v = copy.deepcopy(views[v].values)
        v_scale1, mm1 = processing_type(v_norm, 0)
        mms.append(mm1)
        v_scales.append(v_scale1)
        save_scale_v = sf + 'real_normed_v' + str(v) + '_' + v_file_path[len(v_file_path) - 1]
        normed_real_v[:, idx_g1g2s[v]] = v_norm
        pd.DataFrame(normed_real_v, columns=views[v].columns.values.tolist()).to_csv(save_scale_v, sep='\t', index=False)
        view_paths.update({'save_scale': save_scale_v})

        res_real_norm_v = run_Rscript_new(deas[v], platform, view_names[v], 'blank', 'blank', view_paths, dataset, 'F', print_lab='F')
        res_real_norm_vs.append(res_real_norm_v)
        res_real_norm_v_paths.append(res_real_norm_v[0])

    logFC_norm = mv_logFC_mv(logFC_v_norms)

    in_files_hurdle = copy.deepcopy(view_paths)
    save_f = sf + 'hurdle_real.csv'
    hurdle_path = [save_f]
    hurdle_path.extend(res_real_norm_v_paths)
    in_files_hurdle['res_files'] = hurdle_path
    # res_hurdle = run_DEA_Rscript('hurdle', save_f, view_paths['save_folder'], view_paths['R_code'], view_paths['design'],
    #                     g1, g2)
    res_hurdle = run_Rscript_new('hurdle', 'hurdle', 'basic', 'blank', 'blank', in_files_hurdle, dataset, 'F', print_lab='F')

    return (v_scales, res_real_norm_vs, res_hurdle, mms, view_paths, logFC_norm)

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

def cv_cross_imp(train_pros, seed, train_view_data, cross_imp, param):
    seed_torch(seed)
    logo = KFold(n_splits=3, shuffle=True)

    param.update({'n_components': min(param['n_components'], len(train_view_data[0][0, :]))})
    mses = 0

    if cross_imp == 'MCCA':

        kernel = 'linear'

        kmcca = mvlearn.embed.KMCCA(n_components=param['n_components'], kernel=kernel,

                                    regs=param['regs'],

                                    multiview_output=True)

    elif cross_imp == 'GCCA':
        from cca_zoo.linear import GCCA

        kmcca = GCCA(latent_dimensions=param['n_components'], c=param['c'], random_state=2024)

    for train_index, test_index in logo.split(train_pros):
        tr_views = [view[train_index, :] for view in train_view_data]
        te_views = [view[test_index, :] for view in train_view_data]

        try:
            kmcca.fit(tr_views)

            test_trans = kmcca.transform(te_views)

            mse = 0
            for v in range(len(te_views)):
                mses += mean_squared_error(test_trans[v], te_views[v])

            mses = mses + mse
        except:

            continue  # doing nothing on exception

    mses = mses/3

    return mses

def objective(trial, train_pros, seed, train_view_data, cross_imp):
    seed_torch(seed)

    if cross_imp == 'MCCA':
        n_components = trial.suggest_int('n_components', 1, 10, step=1)
        regs = trial.suggest_float('regs', 0, 1)

        param = {'n_components':n_components, 'regs':regs}

    elif cross_imp == 'GCCA':

        n_components = trial.suggest_int('n_components', 1, 10, step=1)
        c = trial.suggest_float('c', 1e-12, 20)

        param = {'n_components': n_components, 'c': c}

    mse = cv_cross_imp(train_pros, seed, train_view_data, cross_imp, param)

    return mse

def get_training_view_data(views, view_datas, pg_ty, thre_miss=0.8):
    pros = []
    mrs = []

    all_pro = []
    if 'Organism' in views[0].keys():
        all_oga = []

    for v in range(len(views)):
        pro = views[v]['Protein'].values
        mr = np.array([1-(len(np.where((np.isnan(view_datas[v][i, :])))[0])/len(view_datas[v][i, :])) for i in range(len(view_datas[v][:,0]))])
        pros.append(pro)
        mrs.append(mr)

        all_pro.extend(pro)
        if 'Organism' in views[0].keys():
            all_oga.extend(views[v]['Organism'].values)

    uni_pros, idx = np.unique(all_pro, return_index=True)
    if 'Organism' in views[0].keys():
        uni_oga = np.array(all_oga)[idx]

    if pg_ty == 'conserve':
        pro_num = np.array([len(pro.replace(',', ';').replace(' ', '').split(';')) for pro in uni_pros])
        uni_pro = uni_pros[np.where((pro_num == 1))[0]]  # pg with 1 protein
        ab_pro = uni_pros[np.where((pro_num > 1))[0]]  # pg with more than 2 protein

        ab_ty = np.array([1 if len(np.intersect1d(uni_pro, pro.replace(',', ';').replace(' ', '').split(';'))) == 0
                          else 0 for pro in ab_pro])

        retain_idx = np.array(
            list(np.where((pro_num == 1))[0]) + list(np.where((pro_num > 1))[0][np.where((ab_ty == 1))[0]]))

        uni_pros = uni_pros[retain_idx]
        if 'Organism' in views[0].keys():
            uni_oga = uni_oga[retain_idx]

    view_data_aligns = []
    mr_aligns = []

    for v in range(len(views)):

        data_ini = np.zeros((len(uni_pros), len(view_datas[0][0,:])))
        mr_ini = np.zeros((len(uni_pros), 1))

        com, idx1, idx2 = np.intersect1d(uni_pros, pros[v], return_indices=True)

        data_ini[idx1, :] = view_datas[v][idx2, :]
        mr_ini[idx1, 0] = mrs[v][idx2]

        view_data_aligns.append(data_ini)
        mr_aligns.append(mr_ini)

    mr_algins = np.array(mr_aligns).squeeze().T
    idx_ty = np.array([1 if len(np.where((mr_algins[i, :]>=thre_miss))[0])==len(views) else 0 for i in range(len(mr_algins[:, 0])) ])

    for v in range(len(views)):
        view_data_aligns[v][np.isnan(view_data_aligns[v])] = 0

    train_pros = uni_pros[np.where((idx_ty==1))[0]]
    test_pros = uni_pros[np.where((idx_ty==0))[0]]
    test_mrs = mr_algins[np.where((idx_ty==0))[0], :]
    train_view_data = [view_align[np.where((idx_ty==1))[0], :] for view_align in view_data_aligns]
    test_view_data = [view_align[np.where((idx_ty==0))[0], :] for view_align in view_data_aligns]
    if 'Organism' in views[0].keys():
        train_ogas = uni_oga[np.where((idx_ty==1))[0]]
        test_ogas = uni_oga[np.where((idx_ty==0))[0]]
    else:
        train_ogas = []
        test_ogas = []

    return train_pros, test_pros, test_mrs, train_ogas, test_ogas, train_view_data, test_view_data


def optuna_optimize_cross_imput(train_pros, test_pros, test_mrs, train_ogas, test_ogas, train_view_data, test_view_data, cross_imp, seed=2024):
    seed_torch(seed)
    study = optuna.create_study(study_name='ci', direction='minimize', sampler=optuna.samplers.TPESampler(seed=seed),
                                pruner=optuna.pruners.HyperbandPruner())

    func = lambda trial: objective(trial, train_pros, seed, train_view_data, cross_imp)
    study.optimize(func, n_trials=20)
    print(study.best_params)
    print(study.best_trial)
    print(study.best_trial.value)

    views_ci, view_datas_ci, view_paths_ci = output_ci(train_pros, test_pros, test_mrs, train_ogas, test_ogas,
                                                       train_view_data, test_view_data, cross_imp, study.best_params, seed)
    return views_ci, view_datas_ci, view_paths_ci

def cv_ci(views, idx_g1g2s, view_names, train_pros, test_pros, test_mrs, train_ogas, test_ogas, train_view_data,
          test_view_data, seed, view_paths, thre_miss=0.8):

    seed_torch(seed)
    logo = KFold(n_splits=3, shuffle=True)

    mses = []
    Ks = []

    cbs = list(combinations([i for i in range(len(train_view_data))], 2))
    for k in range(2, min(len(train_view_data[0][0, :]), 11)):
        for train_index, test_index in logo.split(train_pros):
            tr_views = [view[train_index, :] for view in train_view_data]
            te_views = [view[test_index, :] for view in train_view_data]

            mse = 0

            for n in range(len(cbs)):
                cca1 = []
                cca1 = CCA(n_components=k)
                cca2 = []
                cca2 = CCA(n_components=k)

                cca1.fit_transform(tr_views[cbs[n][0]], tr_views[cbs[n][1]])
                cca2.fit_transform(tr_views[cbs[n][1]], tr_views[cbs[n][0]])

                mse += mean_squared_error(cca1.predict(te_views[cbs[n][0]]), te_views[cbs[n][1]]) + mean_squared_error(cca2.predict(te_views[cbs[n][1]]), te_views[cbs[n][0]])
        mses.append(mse/3)
        Ks.append(k)

    Kbest = Ks[np.where((np.array(mses) == min(np.array(mses))))[0][0]]

    ccas = {}

    for n in range(len(cbs)):
        cca1 = []
        cca1 = CCA(n_components=Kbest)
        cca2 = []
        cca2 = CCA(n_components=Kbest)

        cca1.fit_transform(train_view_data[cbs[n][0]], train_view_data[cbs[n][1]])
        cca2.fit_transform(train_view_data[cbs[n][1]], train_view_data[cbs[n][0]])

        if str(cbs[n][1]) not in ccas.keys():
            ccas.update({str(cbs[n][1]):[{'in':cbs[n][0], 'cca':cca1}]})
        else:
            cca_list = []
            cca_list = ccas[str(cbs[n][1])]
            cca_list.append({'in': cbs[n][0], 'cca': cca1})
            ccas.update({str(cbs[n][1]): cca_list})

        if str(cbs[n][0]) not in ccas.keys():
            ccas.update({str(cbs[n][0]): [{'in': cbs[n][1], 'cca': cca2}]})
        else:
            cca_list = []
            cca_list = ccas[str(cbs[n][0])]
            cca_list.append({'in': cbs[n][1], 'cca': cca2})
            ccas.update({str(cbs[n][0]): cca_list})

        #ccas.update({str(cbs[n][0])+'_'+str(cbs[n][1]):cca1,str(cbs[n][1])+'_'+str(cbs[n][2]):cca2})

    test_views_ci = []

    views_ci = []
    view_datas_ci = []
    view_paths_ci = copy.deepcopy(view_paths)
    idx_g1g2s_ci = []

    for v in range(len(train_view_data)):
        cca_v = ccas[str(v)]
        cca_test_v = []
        times = np.zeros((len(test_mrs[:, 0]), 1))
        for i in range(len(cca_v)):
            in_i = cca_v[i]['in']

            idx = np.where(((test_mrs[:, v]==0) & (test_mrs[:, in_i]>=thre_miss)))[0]
            cca_test_v_i = copy.deepcopy(test_view_data[v])
            if len(idx) > 0:
                times[idx, 0] = times[idx, 0] + 1
                pro_cca_i = test_pros[idx]
                cca_test_v_i[idx, :] = cca_v[i]['cca'].predict(test_view_data[in_i][idx,:])

                if len(cca_test_v) == 0:
                    cca_test_v = cca_test_v_i
                else:
                    cca_test_v[idx, :] = cca_test_v[idx, :] + cca_test_v_i[idx, :]
            else:
                cca_test_v = test_view_data[v]

        uni_times = np.unique(times[:, 0])
        for t in uni_times:
            if t>0:
                cca_test_v[np.where((times[:, 0]==t))[0], :] = cca_test_v[np.where((times[:, 0]==t))[0], :]/t

        test_views_ci.append(cca_test_v)
        # if t > 0:
        #     test_views_ci.append(cca_test_v/t)
        # else:
        #     test_views_ci.append(test_view_data[v])
        view_data_ci = np.row_stack((train_view_data[v], test_views_ci[v]))
        pro_info = np.array(list(train_pros) + list(test_pros))
        if len(train_ogas)==0:
            view_ci = np.column_stack((pro_info, view_data_ci))
            header = ['Protein'] + views[v].columns.values[idx_g1g2s[v]].tolist()
            idx_g1g2s_ci.append(np.array([i for i in range(len(idx_g1g2s[v]))])+1)
        else:
            orga_info = np.array(list(train_ogas) + list(test_ogas))
            view_ci = np.column_stack((pro_info, orga_info, view_data_ci))
            header = ['Protein', 'Organism'] + views[v].columns.values[idx_g1g2s[v]].tolist()
            idx_g1g2s_ci.append(np.array([i for i in range(len(idx_g1g2s[v]))]) + 2)

        view_ci = pd.DataFrame(view_ci, columns=header)
        view_datas_ci.append(view_data_ci)
        views_ci.append(view_ci)

        view_paths_ci.update({view_names[v]+'_cca':view_data_ci})

    return views_ci, view_datas_ci, view_paths_ci, idx_g1g2s_ci

def get_multi_view_data(in_ty, pg_ty, file_pth, g1, g2, design, view_names, R_fold, Rcode, python_path, acq, platform,
                        save_folder, logT, cross_imp, seed, imput='F'):

    '''

    Parameters
    ----------
    in_ty: input type, 'raw' for raw quantification outputs, 'views' for extracted matrices for each views
    pg_ty: proteingroups process type, 'remove' for removing proteingroups with more than two proteins, 'conserve' for
           retaining the proteingroups with proteins having not been reported independently, 'none' for no action
    file_pth: input quantification files, if in_ty==raw, file_pth = path to raw + ',' path to evidence;
              for in_ty == 'views', file_pth = view1 + ',' + view2 + ',' + ..., view1, view2, ... should in the same
             order as in view names
    g1: condition 1
    g2: condition 2
    design: experiment design file with sample name, condition, replicate information being specified
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

    sample_g1 = designs['sample_name'].values[np.where((designs['condition'].values == g1))[0]]
    sample_g2 = designs['sample_name'].values[np.where((designs['condition'].values == g2))[0]]

    idx_g1g2s = []
    view_datas = []
    view_logFCs = []
    view_confis = []

    for v in range(len(headers)):
        header = headers[v]
        idx_g1g2 = []
        for sample in list(sample_g1) + list(sample_g2):
            for i in range(len(header)):
                header_pure = header[i].replace('.Spectral.Count', '').replace('.MaxLFQ.Intensity', '').replace(
                    '.Intensity', '')
                header_pure = header_pure.replace('MS.MS.count.', '').replace('Intensity.', '').replace(
                    'LFQ.intensity.', '').replace('Top3.', '')
                if sample == header_pure:
                    idx_g1g2.append(i)

        ##### make sure the samples are ordered consistently
        view_data = np.array(views[v].values[:, idx_g1g2], dtype='float')
        if imput == 'T':
            view_data = impute_MinProb(view_data)

        if 'Organism' in header:
            info = np.array(views[v].values[:, 0:2])
            info_name = list(np.array(header)[0:2])
        else:
            info = np.array(views[v].values[:, 0:1])
            info_name = list(np.array(header)[0:1])

        view_datas.append(view_data)
        views[v] = pd.DataFrame(np.column_stack((info, view_data)),
                                columns=info_name + list(sample_g1) + list(sample_g2))

        view_logFC, view_confi = cal_logFC(view_data, header[idx_g1g2], g1, g2, design, logT[v], view_names[v])
        if 'Organism' in header:
            idx_g1g2s.append(np.array([i for i in range(2, len(idx_g1g2)+2)], dtype='int'))
        else:
            idx_g1g2s.append(np.array([i for i in range(1, len(idx_g1g2) + 1)], dtype='int'))
        view_logFCs.append(view_logFC)
        view_confis.append(view_confi)

    if cross_imp != 'none':
        train_pros, test_pros, test_mrs, train_ogas, test_ogas, train_view_data, test_view_data = get_training_view_data(views, view_datas, pg_ty)
        views_ci, view_datas_ci, view_paths_ci, idx_g1g2s_ci = cv_ci(views, idx_g1g2s, view_names, train_pros, test_pros, test_mrs,
                                                                             train_ogas, test_ogas, train_view_data,
                                                                             test_view_data, seed, view_paths)

    else:
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

        view_logFC, view_confi = cal_logFC(view_data, list(sample_g1) + list(sample_g2), g1, g2, design, logT[i],
                                           view_names[i])
        view_logFCs_intersect.append(view_logFC)
        view_confis_intersect.append(view_confi)

        view_data_union = np.zeros((len(union_pro), len(view_datas[i][0, :])))

        comm_pros, idx1, idx2 = np.intersect1d(views[i]['Protein'], union_pro[:, 0], return_indices=True)
        view_data_union[idx2, :] = view_datas[i][idx1, :]
        view_datas_union.append(view_data_union)
        views_union.append(
            pd.DataFrame(np.column_stack((union_pro, view_data_union)), columns=list(views[i].columns.values)))

        view_logFC, view_confi = cal_logFC(view_data_union, list(sample_g1) + list(sample_g2), g1, g2, design, logT[i],
                                           view_names[i])
        view_logFCs_union.append(view_logFC)
        view_confis_union.append(view_confi)

    idx_r = np.where(((designs['condition'].values == g1) | (designs['condition'].values == g2)))[0]
    #if '/' in view_paths['design_file_path']:
    file_name_design = view_paths['design_file_path'].replace('\\','/').split('/')
    #elif '\\' in view_paths['design_file_path']:
    #    file_name_design = view_paths['design_file_path'].split('\\')

    if not os.path.exists(save_folder + 'dt/'):
        os.makedirs(save_folder + 'dt/')

    save_path = save_folder + 'dt/' + file_name_design[len(file_name_design) - 1]
    out_design = pd.DataFrame(designs.values[idx_r, :], columns=designs.columns.values.tolist())
    out_design.to_csv(save_path, sep='\t',
                      index=False)
    view_paths.update({'design': save_path})

    return (views, view_datas, views_intersect, view_datas_intersect, idx_g1g2s, view_paths, 'dt', views_ci,
            view_datas_ci, view_paths_ci, idx_g1g2s_ci, out_design, view_logFCs,
            view_confis, view_logFCs_intersect, view_confis_intersect, views_union, view_datas_union,
            view_logFCs_union, view_confis_union)















