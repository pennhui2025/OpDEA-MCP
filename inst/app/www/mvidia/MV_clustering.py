from sklearn import preprocessing
from cca_zoo.linear import GCCA
import random
import numpy as np
import scipy.sparse as sp
import os
from sklearn import metrics
from sklearn.metrics import silhouette_score,adjusted_rand_score,normalized_mutual_info_score
from sklearn.metrics.cluster import contingency_matrix
import pandas as pd
#from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from MV_classification import get_multi_view_data

def seed_torch(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

def purity_score(y_true, y_pred):
    contingency_matrix1 = contingency_matrix(y_true, y_pred)
    return np.sum(np.amax(contingency_matrix1, axis=0)) / np.sum(contingency_matrix1)

def get_data(df_name, design_name, logT, scale, filter_num, filter_type, save_folder, name):
    df = pd.read_csv(df_name, sep='\t', header=0)
    design = pd.read_csv(design_name, sep='\t', header=0)
    proteins = df['Protein'].values
    cols = df.columns.values
    cols = np.array([cols[i].replace('.Spectral.Count', '').replace('.MaxLFQ.Intensity', '').replace(
                    '.Intensity', '').replace('MS.MS.count.', '').replace('Intensity.', '').replace(
                    'LFQ.intensity.', '').replace('Top3.', '') for i in range(len(cols))])


    sample, idx1, idx2 = np.intersect1d(cols, design['sample_name'], return_indices=True)

    exps = np.array(df.values[:, idx1], dtype='float')

    if filter_type == 'num':
        count_num = np.array([len(np.where((exps[i, :]>0))[0]) for i in range(len(exps[:, 0]))])
        exps = exps[np.where((count_num >= filter_num))[0], :]
        proteins = proteins[np.where((count_num >= filter_num))[0]]

    elif filter_type == 'percent':

        count_num = np.array([len(np.where((exps[i, :] > 0))[0])/len(exps[i, :]) for i in range(len(exps[:, 0]))])
        exps = exps[np.where((count_num >= filter_num))[0], :]
        proteins = proteins[np.where((count_num >= filter_num))[0]]

    if logT == 'T':
        exps[exps == 0] = np.nan
        exps = np.log2(exps)

    if scale == 'T':
        exps = preprocessing.MinMaxScaler().fit_transform(exps)

    exps[np.isnan(exps)] = 0
    exps = exps.T

    cell_type = design['condition'].values[idx2]
    cell_type[np.where((cell_type == 'HELA'))[0]] = 'Hela'
    pd.DataFrame(np.column_stack((cell_type, exps)),  columns=['cell'] + list(proteins)).to_csv(save_folder + name + '.csv', sep=',', index=False)

    return {'raw': df, 'design': design, 'exp': exps, 'cell_type': cell_type, 'sample': sample, 'protein': proteins}

def transform_data(comm, df, scale, name, save_folder):
    com, idx1, idx2 = np.intersect1d(comm, df['protein'], return_indices=True)
    out_pro = df['protein'][idx2]
    out_cell_type = df['cell_type']
    exp = df['exp'][:, idx2]

    if scale == 'T':
        exp[exp == 0] = np.nan
        exp = exp.T
        exp = preprocessing.MinMaxScaler().fit_transform(exp)

        exp[np.isnan(exp)] = 0
        exp = exp.T

    out = {name: {'protein': out_pro, 'cell_type': out_cell_type,
                                      'exp': exp, 'raw': df}}
    # pd.DataFrame(np.column_stack((out_cell_type, exp)),
    #              columns=['cell'] + list(out_pro)).to_csv(
    #     save_folder + name + '_intersect.csv', sep=',', index=False)
    return out

def combine_DDA_DIA(DDA_folder, DIA_folder, filter_num, filter_type, save_folder):

    DDA_dlfq = DDA_folder + 'SC_DDA_dlfq.tsv'
    DDA_maxlfq = DDA_folder + 'SC_DDA_maxlfq.tsv'
    DDA_top0 = DDA_folder + 'SC_DDA_top0.tsv'
    DDA_design = DDA_folder + 'SC_DDA_design.tsv'

    DIA_dlfq = DIA_folder + 'SC_DIA_dlfq.tsv'
    DIA_maxlfq = DIA_folder + 'SC_DIA_maxlfq.tsv'
    DIA_top3 = DIA_folder + 'SC_DIA_top3.tsv'
    DIA_design = DIA_folder + 'SC_DIA_design.tsv'

    dt_dda_dlfq = get_data(DDA_dlfq, DDA_design, 'T', 'F', filter_num, filter_type, save_folder,'DDA_dlfq')
    dt_dda_maxlfq = get_data(DDA_maxlfq, DDA_design, 'T', 'F', filter_num, filter_type, save_folder,'DDA_maxlfq')
    comm_pro_dda = np.intersect1d(dt_dda_dlfq['protein'], dt_dda_maxlfq['protein'])
    dt_dda_topN = get_data(DDA_top0, DDA_design, 'T', 'F', filter_num, filter_type, save_folder,'DDA_topN')
    comm_pro_dda = np.intersect1d(comm_pro_dda, dt_dda_topN['protein'])

    dt_dia_dlfq = get_data(DIA_dlfq, DIA_design, 'T', 'F', filter_num, filter_type, save_folder,'DIA_dlfq')
    dt_dia_maxlfq = get_data(DIA_maxlfq, DIA_design, 'F', 'F', filter_num, filter_type, save_folder,'DIA_maxlfq')
    comm_pro_dia = np.intersect1d(dt_dia_maxlfq['protein'], dt_dia_dlfq['protein'])
    dt_dia_topN = get_data(DIA_top3, DIA_design, 'F', 'F', filter_num, filter_type, save_folder,'DIA_topN')
    comm_pro_dia = np.intersect1d(comm_pro_dia, dt_dia_topN['protein'])

    out_raw = {'dda_dlfq': dt_dda_dlfq, 'dda_maxlfq': dt_dda_maxlfq, 'dda_topN': dt_dda_topN,
               'dia_dlfq': dt_dia_dlfq, 'dia_maxlfq': dt_dia_maxlfq, 'dia_topN': dt_dia_topN}

    out_intersect = {}
    out_intersect.update(transform_data(comm_pro_dda, dt_dda_dlfq, 'F', 'dda_dlfq'))

    out_intersect.update(transform_data(comm_pro_dda, dt_dda_maxlfq, 'F', 'dda_maxlfq'))

    out_intersect.update(transform_data(comm_pro_dda, dt_dda_topN, 'F', 'dda_topN'))

    out_intersect.update(transform_data(comm_pro_dia, dt_dia_dlfq, 'F', 'dia_dlfq'))

    out_intersect.update(transform_data(comm_pro_dia, dt_dia_maxlfq, 'F', 'dia_maxlfq'))

    out_intersect.update(transform_data(comm_pro_dia, dt_dia_topN, 'F', 'dia_topN'))

    comm_dda_dia = np.intersect1d(comm_pro_dda, comm_pro_dia)

    out_intersect_dda_dia = {}
    out_intersect_dda_dia.update(transform_data(comm_dda_dia, dt_dda_dlfq, 'F', 'dda_dlfq'))

    out_intersect_dda_dia.update(transform_data(comm_dda_dia, dt_dda_maxlfq, 'F', 'dda_maxlfq'))

    out_intersect_dda_dia.update(transform_data(comm_dda_dia, dt_dda_topN, 'F', 'dda_topN'))

    out_intersect_dda_dia.update(transform_data(comm_dda_dia, dt_dia_dlfq, 'F', 'dia_dlfq'))

    out_intersect_dda_dia.update(transform_data(comm_dda_dia, dt_dia_maxlfq, 'F', 'dia_maxlfq'))

    out_intersect_dda_dia.update(transform_data(comm_dda_dia, dt_dia_topN, 'F', 'dia_topN'))
    return out_raw, out_intersect, out_intersect_dda_dia

def get_data_from_multi_view(view_data, design, logT, scale, filter_num, filter_type, save_folder, name):
    df = view_data
    design = design
    proteins = df['Protein'].values
    cols = df.columns.values
    cols = np.array([cols[i].replace('.Spectral.Count', '').replace('.MaxLFQ.Intensity', '').replace(
                    '.Intensity', '').replace('MS.MS.count.', '').replace('Intensity.', '').replace(
                    'LFQ.intensity.', '').replace('Top3.', '') for i in range(len(cols))])

    sample, idx1, idx2 = np.intersect1d(cols, design['sample_name'], return_indices=True)

    exps = np.array(df.values[:, idx1], dtype='float')

    if filter_type == 'num':
        count_num = np.array([len(np.where((exps[i, :]>0))[0]) for i in range(len(exps[:, 0]))])
        exps = exps[np.where((count_num >= filter_num))[0], :]
        proteins = proteins[np.where((count_num >= filter_num))[0]]

    elif filter_type == 'percent':

        count_num = np.array([len(np.where((exps[i, :] > 0))[0])/len(exps[i, :]) for i in range(len(exps[:, 0]))])
        exps = exps[np.where((count_num >= filter_num))[0], :]
        proteins = proteins[np.where((count_num >= filter_num))[0]]

    if logT == 'T':
        exps[exps == 0] = np.nan
        exps = np.log2(exps)

    if scale == 'T':
        exps = preprocessing.MinMaxScaler().fit_transform(exps)

    exps[np.isnan(exps)] = 0
    exps = exps.T

    if 'condition' in design.keys():
        cell_type = design['condition'].values[idx2]
        cell_type = np.array([cell.capitalize() for cell in cell_type])
    else:
        cell_type = ['unknown'] * len(design['sample_name'].values)
    #pd.DataFrame(np.column_stack((cell_type, exps)),  columns=['cell'] + list(proteins)).to_csv(save_folder + name + '.csv', sep=',', index=False)

    return {'raw': df, 'design': design, 'exp': exps, 'cell_type': cell_type, 'sample': sample, 'protein': proteins}

def preprocess_multi_view_new(mv_data, filter_num, filter_type, save_folder, name):

    design = mv_data[11]
    logT = 'F'
    scale = 'F'

    n = len(mv_data[16])
    out_raw = {}
    comm_pro = []
    for i in range(n):
        dt_mv = mv_data[16][i]
        dt_v = get_data_from_multi_view(dt_mv, design, logT, scale, filter_num, filter_type, save_folder, name)

        if i == 0:
            comm_pro = dt_v['protein']
        else:
            comm_pro = np.intersect1d(comm_pro, dt_v['protein'])

        out_raw.update({'v'+str(i+1):dt_v})

    out_intersect = {}

    for i in range(n):
        out_intersect.update(transform_data(comm_pro, out_raw['v'+str(i+1)], 'F', 'v'+str(i+1), save_folder))

    return out_raw, out_intersect

def clustering(view_data, ncluster, save_folder, name, view_names, ld, c):

    seed_torch()

    n = len(view_names)
    ys = []

    in_exp = []

    for i in range(n):

        k_means = []
        k_means = KMeans(n_clusters=ncluster, n_init=10, random_state=42)
        y_v = k_means.fit_predict(preprocessing.StandardScaler().fit_transform(view_data['v'+str(i+1)]['exp']))
        ys.append(y_v)
        pd.DataFrame(np.column_stack((view_data['v'+str(i+1)]['cell_type'], y_v, preprocessing.StandardScaler().fit_transform(view_data['v'+str(i+1)]['exp']))),
                     columns=['cell', 'predict_class'] + ['l' + str(j) for j in range(len(view_data['v'+str(i+1)]['exp'][0, :]))]).to_csv(
            save_folder + 'v'+str(i+1) + '_scale.csv', index=False)

        in_exp.append(preprocessing.StandardScaler().fit_transform(view_data['v'+str(i+1)]['exp']))

    if c=='blank':
        kmcca = GCCA(latent_dimensions=int(ld), random_state=2024)
    else:
        kmcca = GCCA(latent_dimensions=int(ld), c=float(c), random_state=2024)

    lat_out = kmcca.fit_transform(in_exp)
    lat_out_mvidia = np.concatenate(lat_out, axis=1)

    k_means = []
    k_means = KMeans(n_clusters=ncluster, n_init=10, random_state=42)
    y_mvidia = k_means.fit_predict(lat_out_mvidia)
    ys.append(y_mvidia)
    pd.DataFrame(np.column_stack(
        (view_data['v1']['cell_type'], y_mvidia, lat_out_mvidia)),
                 columns=['cell', 'predict_class'] + ['l' + str(j) for j in range(len(lat_out_mvidia[0, :]))]).to_csv(
        save_folder + 'mvidia_latent.csv', index=False)

    cat_3v = np.concatenate(in_exp, axis=1)

    k_means = []
    k_means = KMeans(n_clusters=ncluster, n_init=10, random_state=42)
    y_concat = k_means.fit_predict(cat_3v)
    ys.append(y_concat)
    pd.DataFrame(np.column_stack(
        (view_data['v1']['cell_type'], y_concat, cat_3v)),
        columns=['cell', 'predict_class'] + ['l' + str(j) for j in range(len(cat_3v[0, :]))]).to_csv(
        save_folder + 'Concat_scale.csv', index=False)


    method_names = ['cluster_by_' + view_names[i] for i in range(len(view_names))]

    if 'condition' in view_data['v1']['raw']['design'].keys():
        out_lab_cluster = np.column_stack((view_data['v1']['raw']['sample'], view_data['v1']['raw']['cell_type'], np.array(ys).T))
        out_pd = pd.DataFrame(out_lab_cluster,
                              columns=['sample', 'True_label'] + method_names + ['cluster_by_MVIDIA', 'cluster_by_Concat'])
        out_pd.to_csv(save_folder + name + '_clustering_labels.csv', index=False)

        true_cell_type = view_data['v1']['cell_type']
        uni_cell = np.unique(true_cell_type)
        true_cell_type_num = np.zeros((len(true_cell_type), 1))
        for i in range(len(uni_cell)):
            true_cell_type_num[np.where((true_cell_type == uni_cell[i]))[0], 0] = i

        true_cell_type_num = true_cell_type_num[:, 0]
        df_result = pd.DataFrame()
        df_result['in_type'] = list(view_names) + ['MVIDIA', 'Concat']
        df_result['ARI'] = [np.round(adjusted_rand_score(true_cell_type_num, ys[i]), 4) for i in range(len(ys))]

        df_result['ASW'] = [np.round(silhouette_score(np.array([true_cell_type_num]).T, np.array([ys[i]]).T), 4) for i in range(len(ys))]

        df_result['NMI'] = [np.round(normalized_mutual_info_score(true_cell_type_num, ys[i]), 4) for i in range(len(ys))]

        print('cell clustering result:')
        print(df_result)

        df_result.to_csv(save_folder + name + '_clustering_metrics.csv')

    else:

        out_lab_cluster = np.column_stack((view_data['v1']['raw']['sample'], np.array(ys).T))
        out_pd = pd.DataFrame(out_lab_cluster,
                              columns=['sample'] + method_names + ['cluster_by_MVIDIA',
                                                                                 'cluster_by_Concat'])
        out_pd.to_csv(save_folder + name + '_clustering_labels.csv', index=False)
