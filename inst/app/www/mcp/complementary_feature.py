import copy
import numpy as np
import pandas as pd
from itertools import combinations
from scipy.stats import pearsonr
from sklearn.metrics import mutual_info_score as MI
from Utils import get_datasets_reproduce_mv
import os
import itertools
from sklearn.metrics import mutual_info_score
from Utils import DEA_single_hurdle_new, get_multi_view_data
from Utils import seed_torch
from scipy.stats import pearsonr
from sklearn.decomposition import PCA
import numpy as np
from itertools import combinations
from sklearn.metrics.pairwise import rbf_kernel
from scipy.stats import entropy


def expression_correlation_complementarity(X_views, selected_indices, thre):
    """
    protein expression profile correlation(higher value means higher difference and stronger compelmentarity)

    parameters:
        X_views: List of ndarray,for each view shape=[n_samples, n_proteins]
        selected_indices: list/tuple,the index of the selected view,e.g. (0, 1, 3)

    return:
        mean_complementarity: the mean complementarity score of each protein
    """
    k = len(selected_indices)
    n_proteins = X_views[selected_indices[0]].shape[1]
    complementarity_scores = []

    for p in range(n_proteins):
        corr_vals = []
        expr_list = [X_views[i][:, p] for i in selected_indices]

        for i, j in combinations(range(k), 2):
            xi, xj = expr_list[i], expr_list[j]
            valid_mask = ~np.isnan(xi) & ~np.isnan(xj)
            if np.sum(valid_mask) > len(xi)*thre:
                r = pearsonr(xi[valid_mask], xj[valid_mask])[0]
                if ~np.isnan(r):
                    corr_vals.append(1 - abs(r))  # lower correlation means stronger complementarity
        if corr_vals:
            complementarity_scores.append(np.mean(corr_vals))

    return np.mean(complementarity_scores)


def missingness_complementarity(X_views, selected_indices):
    """
    The complementarity of the missingness(higher score means higher missing complementarity)

    parameters:
        X_views: List of ndarray,for each view shape=[n_samples, n_proteins]
        selected_indices: list/tuple,the index of the selected view,e.g. (0, 1, 3)

    return:
        mean_missing_complementarity: the mean missing complementarity score of each protein
    """
    k = len(selected_indices)
    n_proteins = X_views[selected_indices[0]].shape[1]
    complementarity_scores = []

    for p in range(n_proteins):
        expr_list = [X_views[i][:, p] for i in selected_indices]
        miss_vals = []

        for i, j in combinations(range(k), 2):
            xi, xj = expr_list[i], expr_list[j]
            miss_i = np.isnan(xi)
            present_j = ~np.isnan(xj)
            comp_ij = np.mean(miss_i & present_j)

            miss_j = np.isnan(xj)
            present_i = ~np.isnan(xi)
            comp_ji = np.mean(miss_j & present_i)

            miss_vals.append((comp_ij + comp_ji) / 2)

        if miss_vals:
            complementarity_scores.append(np.mean(miss_vals))

    return np.mean(complementarity_scores)

def compute_HSIC(X, Y, sigma=1.0):
    """compute HSIC,evaluate the dependency between X, Y,bigger values mean stronger dependency"""
    X1 = copy.deepcopy(X)
    Y1 = copy.deepcopy(Y)
    X1[np.isnan(X1)] = 0
    Y1[np.isnan(Y1)] = 0
    K = rbf_kernel(X1, gamma=1.0/(2*sigma**2))
    L = rbf_kernel(Y1, gamma=1.0/(2*sigma**2))
    n = K.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    KH = H @ K @ H
    LH = H @ L @ H
    hsic = np.trace(KH @ LH) / (n - 1)**2
    return hsic

def compute_complementarity_HSIC(X_views, max_k=3):
    K = len(X_views)
    scores = []

    for k in range(max_k, max_k + 1):
        for idx_tuple in combinations(range(K), k):
            hsic_vals = []
            for i, j in combinations(idx_tuple, 2):
                hsic_ij = compute_HSIC(X_views[i], X_views[j])
                hsic_vals.append(1 - hsic_ij)  # 越小越互补,取1-HSIC
            score = np.mean(hsic_vals)
            scores.append(score)
    return np.mean(scores)

def compute_reconstruction_complementarity(X_views0, max_k=3, n_components=5):
    X_views = copy.deepcopy(X_views0)
    for x in X_views:
        x[np.isnan(x)] = 0
    K = len(X_views)
    scores = []

    # 单视角的 PCA 重建误差
    single_errors = []
    for X in X_views:
        pca = PCA(n_components=n_components)
        X_reduced = pca.fit_transform(X)
        X_recon = pca.inverse_transform(X_reduced)
        err = np.mean((X - X_recon)**2)
        single_errors.append(err)

    for k in range(max_k, max_k + 1):
        for idx_tuple in combinations(range(K), k):
            combined = np.concatenate([X_views[i] for i in idx_tuple], axis=1)
            pca = PCA(n_components=n_components)
            X_reduced = pca.fit_transform(combined)
            X_recon = pca.inverse_transform(X_reduced)
            combined_error = np.mean((combined - X_recon)**2)

            avg_single = np.mean([single_errors[i] for i in idx_tuple])
            # 互补性 = 组合重建误差 - 平均单视角误差
            score = combined_error - avg_single
            scores.append(score)
    return np.mean(scores)

def compute_mmd(X, Y, sigma=1.0):
    """compute MMD distance：X vs Y distribution difference"""
    X1 = copy.deepcopy(X)
    Y1 = copy.deepcopy(Y)
    X1[np.isnan(X1)] = 0
    Y1[np.isnan(Y1)] = 0
    Kxx = rbf_kernel(X1, gamma=1.0/(2*sigma**2))
    Kyy = rbf_kernel(Y1, gamma=1.0/(2*sigma**2))
    Kxy = rbf_kernel(X1, Y1, gamma=1.0/(2*sigma**2))
    mmd = np.mean(Kxx) + np.mean(Kyy) - 2 * np.mean(Kxy)
    return mmd

def compute_complementarity_MMD(X_views, max_k=3):
    K = len(X_views)
    scores = []

    for k in range(max_k, max_k + 1):
        for idx_tuple in combinations(range(K), k):
            mmd_vals = []
            for i, j in combinations(idx_tuple, 2):
                mmd = compute_mmd(X_views[i], X_views[j])
                mmd_vals.append(mmd)
            score = np.mean(mmd_vals)
            scores.append(score)
    return np.mean(scores)

def _multi_entropy(views0, view_indices, bins=15):
    """multi-view joint-entropy"""
    # data discretization
    views = copy.deepcopy(views0)
    for x in views:
        x[np.isnan(x)] = 0
    hist_data = []
    for idx in view_indices:
        data = views[idx]
        hist = np.histogram(data, bins=bins)[0] + 1e-9
        hist_data.append(hist)

    # compute joint entropy
    joint_hist = np.prod(np.stack(hist_data), axis=0)
    joint_entropy = entropy(joint_hist)

    # compute normalized complementarity score
    max_entropy = entropy(np.ones_like(joint_hist))
    return (max_entropy - joint_entropy) / max_entropy

def mutual_information_complementarity(X_views, selected_indices, n_bins=10):
    """
    compute the mutual information complementarity of protein expression between given paired views(higher stronger complementarity)

    parameters:
        X_views: list of ndarray,for each view, shape = [n_samples, n_proteins]
        selected_indices: list/tuple, the index of the selected view,e.g. (0, 1, 2)
        n_bins: int,binning the continuous expression for computing mi

    return:
        mean_mi_complementarity: the mean mi complementarity for all proteins (higher with stronger complementarity)
    """
    k = len(selected_indices)
    n_proteins = X_views[selected_indices[0]].shape[1]
    complementarity_scores = []

    for p in range(n_proteins):
        mi_vals = []
        expr_list = [X_views[i][:, p] for i in selected_indices]

        for i, j in combinations(range(k), 2):
            xi, xj = expr_list[i], expr_list[j]
            valid_mask = ~np.isnan(xi) & ~np.isnan(xj)
            if np.sum(valid_mask) > len(xi)*0.6:

                x_disc = np.digitize(xi[valid_mask], bins=np.histogram_bin_edges(xi[valid_mask], bins=n_bins))
                y_disc = np.digitize(xj[valid_mask], bins=np.histogram_bin_edges(xj[valid_mask], bins=n_bins))
                mi = mutual_info_score(x_disc, y_disc)
                # complementarity = 1 - normalized mi
                mi_vals.append(1 - mi / np.log(n_bins))  # scale to [0, 1]
            else:
                mi_vals.append(0.5)  # too high missing rate, then use a default value

        if mi_vals:
            complementarity_scores.append(np.mean(mi_vals))

    return np.mean(complementarity_scores)

class MissingComplementarityAnalyzer:
    def __init__(self, views, feature_names=None):
        """
        parameters:
            views: list of DataFrames - views(needs sample alignment)
            feature_names: list - name of the views
        """
        self.views = [v.copy() for v in views]
        self.n_views = len(views)
        self.featur

def cal_missing_rate(view0):
    view = copy.deepcopy(view0)
    view[np.isnan(view)] = 0
    return np.count_nonzero(view==0)/(len(view[0, :])*len(view[:, 0]))

def cal_comp_miss_rate(views0):
    views = copy.deepcopy(views0)
    view0 = views[0]
    view0[np.isnan(view0)] = 0

    for i in range(1, len(views)):
        views[i][np.isnan(views[i])] = 0
        view0 += views[i]

    return np.count_nonzero(view0==0)/(len(view0[0, :])*len(view0[:, 0]))

def get_views_stat_feature(views_list, ex_tys, acq, dea, plt):

    all_view_type = np.unique(['dlfq', 'maxlfq', 'top0', 'top3', 'count'] + ['dlfq', 'maxlfq', 'top3', 'top1'])
    all_platform = ['FragPipe', 'Maxquant', 'DIANN', 'Spectronaut']

    all_dea = ['limma', 'ROTS']
    all_acq = ['DDA', 'DIA']

    fea_view_type = np.array([0] * len(all_view_type))
    idxs = [list(all_view_type).index(ex_tys[i]) for i in range(len(ex_tys))]
    fea_view_type[idxs] = 1

    fea_view_mr = np.array([1] * len(all_view_type), dtype='float')
    for i in range(len(idxs)):
        fea_view_mr[idxs[i]] = cal_missing_rate(views_list[i])

    fea_comb_mr = cal_comp_miss_rate(views_list)

    fea_plt = np.array([0] * len(all_platform))
    fea_plt[all_platform.index(plt)] = 1

    fea_dea = np.array([0] * len(all_dea))
    fea_dea[all_dea.index(dea)] = 1

    fea_acq = np.array([0] * len(all_acq))
    fea_acq[all_acq.index(acq)] = 1
    mr_reduce = min(fea_view_mr) - fea_comb_mr
    fea_name = list(all_view_type) + [i + '_mr' for i in list(all_view_type)] + ['comb_mr'] + all_platform + all_dea + all_acq# + ['mr_reduce']
    fea = np.array(list(fea_view_type) + list(fea_view_mr) + [fea_comb_mr] + list(fea_plt) + list(fea_dea) + list(fea_acq))
    return fea_name, fea, mr_reduce

def get_feas_all(views_union, res_union, ex_tys, platform, dea, data_idx, save_file_name, acq):
    views = []
    views_p_list = []
    views_list = []
    inds = []

    DEAs = []

    DEA_list = []
    for i in range(len(views_union)):
        values = np.array(views_union[i].values[:, 2:len(views_union[i].values[0, :])], dtype='float').T
        values[values == 0] = np.nan

        proteins = views_union[i]['Protein'].values
        view = pd.DataFrame(values,
                            index=views_union[i].columns.values[2:len(views_union[i].columns.values)],
                            columns=proteins)
        views_list.append(values)
        views_p_list.append(values.T)
        inds.append(i)
        views.append(view)

        DEA_protein = res_union[1][i][1]
        DEA_logfc = res_union[1][i][7]
        DEA_qval = res_union[1][i][10]

        DEA_logfc[np.where((DEA_logfc == 0))[0]] = np.nan
        DEA_qval[np.where((DEA_logfc == 0))[0]] = np.nan

        pro, idx1, idx2 = np.intersect1d(proteins, DEA_protein, return_indices=True)

        DEA_pro = np.array([''] * len(proteins), dtype='object')
        DEA_pro[idx1] = np.array(DEA_protein, dtype='object')[idx2]

        DEA_lgfc = np.array([np.nan] * len(proteins))
        DEA_lgfc[idx1] = DEA_logfc[idx2]

        DEA_q = np.array([np.nan] * len(proteins))
        DEA_q[idx1] = DEA_qval[idx2]

        DEA_value = np.column_stack((DEA_lgfc, DEA_q))
        DEA_list.append(DEA_value)
        DEAs.append(pd.DataFrame(DEA_value,
                                 index=proteins,
                                 columns=['lgfc', 'qval']))

    feas = []
    for view_num in [2, 3, 4, 5]:  #
        combs = np.array(list(combinations([x for x in range(len(ex_tys))], view_num)))
        for comb_idx in range(len(combs)):
            view_idx = combs[comb_idx, :]
            views_names = list(np.array(ex_tys)[view_idx])
            view_name_strs = acq
            for y in range(len(views_names)):
                view_name_strs += '_' + views_names[y]

            views_list_cb = [views_list[i] for i in view_idx]
            views_p_list_cb = [views_p_list[i] for i in view_idx]
            DEA_list_cb = [DEA_list[i] for i in view_idx]
            view_idx_cb = [i for i in range(len(view_idx))]

            v_cr_pro = expression_correlation_complementarity(views_list_cb, view_idx_cb, 0.8)

            v_miss = missingness_complementarity(views_list_cb, view_idx_cb)
            v_HSIC = compute_complementarity_HSIC(views_p_list_cb, max_k=len(view_idx_cb))
            v_pca = compute_reconstruction_complementarity(views_p_list_cb, max_k=len(view_idx_cb))
            v_mmd = compute_complementarity_MMD(views_p_list_cb, max_k=len(view_idx_cb))
            v_mi = mutual_information_complementarity(views_list_cb, view_idx_cb, n_bins=3)
            v_entropy = _multi_entropy(views_list_cb, view_idx_cb)

            d_cr = expression_correlation_complementarity(DEA_list_cb, view_idx_cb, 0.2)
            d_miss = missingness_complementarity(DEA_list_cb, view_idx_cb)
            d_HSIC = compute_complementarity_HSIC(DEA_list_cb, max_k=len(view_idx_cb))

            d_mmd = compute_complementarity_MMD(DEA_list_cb, max_k=len(view_idx_cb))
            d_mi = mutual_information_complementarity([DEA_list_cb[s].T for s in range(len(DEA_list_cb))], view_idx_cb,
                                                      n_bins=3)
            d_entropy = _multi_entropy(DEA_list_cb, view_idx_cb)

            fea_compl_v = [view_name_strs, data_idx, acq, platform, dea, len(views_list_cb), v_cr_pro, v_miss,
                           v_HSIC, v_pca, v_mmd, v_mi, v_entropy]  # v_cr_sample,
            fea_compl_d = [d_cr, d_miss, d_HSIC, d_mmd, d_mi, d_entropy]  # d_pca,

            fea_compl_name_v = ['view_name', 'data_idx', 'acq', 'platform', 'dea', 'view_num', 'v_cr_pro', 'v_miss',
                                'v_HSIC', 'v_pca', 'v_mmd', 'v_mi', 'v_entropy']  # 'v_cr_sample',
            fea_compl_name_d = ['d_cr', 'd_miss', 'd_HSIC', 'd_mmd', 'd_mi', 'd_entropy']  # , 'd_pca'
            fea_name, fea, mr_reduce = get_views_stat_feature(views_list_cb, views_names, acq, dea, platform)
            fea_name_final = fea_compl_name_v + fea_compl_name_d + fea_name + ['mr_reduce']
            feas.append(list(fea_compl_v) + list(fea_compl_d) + list(fea) + list([mr_reduce]))
            print(fea_name_final)
    feas = np.array(feas)
    out_fea = pd.DataFrame(feas, columns=fea_name_final)

    if os.path.exists(save_file_name):
        hs = False
    else:
        hs = True
    out_fea.to_csv(save_file_name, mode='a', header=hs)
    return out_fea

def single_run_train(flag_id, platform, data_idx, g1, g2, acq, ex_tys, logTs, deas, R_code, Rscript_path, save_data_root, data_fold):

    flag = str(flag_id) + '_' + deas[0] + '_' + platform + '_view' + str(len(deas))

    if not os.path.exists(save_data_root):
        os.path.mkdir(save_data_root)

    seed = 2024
    norms = ['blank'] * len(ex_tys)
    imputs = ['blank'] * len(ex_tys)
    cross_imp = 'none'
    pg_ty = 'remove'

    (res_ori, res_intersect, res_ci, views, view_datas, views_intersect, view_datas_intersect,
     views_ci, view_datas_ci, save_folder, view_paths, res_union, views_union,
     view_datas_union) = get_datasets_reproduce_mv(flag, data_idx, platform, norms, imputs, g1, g2, acq,
                                                   ex_tys,
                                                   save_data_root, R_code, deas, logTs, seed, pg_ty, Rscript_path,
                                                   cross_imp, data_fold)
    save_file_name = save_data_root + 'complementary_feas_benchmark.csv'
    out_fea = get_feas_all(views_union, res_union, ex_tys, platform, deas[0], data_idx, save_file_name)
    return out_fea

def reproduce_test(acq, views_names, all_res, R_code, Rscript_path, save_data_root, data_fold):

    if acq == 'DIA':

        flag_id = 0
        for platform in ['DIANN', 'Spectronaut']:
            for dea in ['limma', 'ROTS']:
                for cbt in ['min']:
                    for didx in [2, 3, 4, 5, 6, 1, 0]:
                        if didx in [1, 5]:
                            g1 = 'A'
                            g2 = 'C'
                        elif didx == 4:
                            g1 = 'B'
                            g2 = 'D'
                        elif didx in [2, 3]:
                            g1 = 'B'
                            g2 = 'E'
                        else:
                            g1 = 'A'
                            g2 = 'B'
                        deas = [dea] * len(views_names)
                        logTs = ['F'] * len(views_names)
                        dt_idx = np.where((np.array(views_names) == 'dlfq'))[0]
                        if len(dt_idx) > 0:
                            logTs[dt_idx[0]] = 'T'
                        res = single_run_train(flag_id, platform, didx, g1, g2, acq, views_names, logTs, deas, R_code, Rscript_path, save_data_root, data_fold)
                        if len(all_res) == 0:
                            all_res = res
                        else:
                            all_res = pd.concat([all_res, res], ignore_index=True)

                    flag_id = flag_id + 1

    elif acq == 'DDA':

        flag_id = 0

        for platform in ['FragPipe', 'Maxquant']:
            for dea in ['limma', 'ROTS']:
                for cbt in ['min']:
                    for didx in [6, 2, 3, 0, 1, 7, 9, 4, 5, 10, 11]:
                        if didx in [5, 10]:
                            g1 = 'A'
                            g2 = 'C'
                        elif didx == 4:
                            g1 = 'B'
                            g2 = 'D'
                        elif didx in [6, 7, 8]:
                            g1 = 'C'
                            g2 = 'E'
                        else:
                            g1 = 'A'
                            g2 = 'B'

                        deas = [dea] * len(views_names)
                        logTs = ['T'] * len(views_names)
                        ct_idx = np.where((np.array(views_names) == 'count'))[0]

                        if len(ct_idx) > 0:
                            deas[ct_idx[0]] = 'edgeR'
                            logTs[ct_idx[0]] = 'F'
                        res = single_run_train(flag_id, platform, didx, g1, g2, acq, views_names, logTs, deas, R_code, Rscript_path, save_data_root, data_fold)
                        if len(all_res) == 0:
                            all_res = res
                        else:
                            all_res = pd.concat([all_res, res], ignore_index=True)

                    flag_id = flag_id + 1

def get_fea_test(Rscript_path, platform, g1, g2, acq, view_names_all, logTs_all, R_fold, dea, save_folder, mat_folder, design_path, dt_name, name_postfix):
    seed = 2023
    norm = 'blank'
    imput = 'blank'
    python_path = ''

    pg_ty = 'remove'
    cross_imp = 'none'
    view_names = view_names_all

    logTs = logTs_all
    deas = [dea] * len(view_names)
    norms = [norm] * len(view_names)
    imputs = [imput] * len(view_names)

    view_str = view_names[0]
    for vn in range(1, len(view_names)):
        view_str += ',' + view_names[vn]

    design_path_tr = design_path

    view_files = mat_folder + view_names[0] + name_postfix
    for vn in range(1, len(view_names)):
        view_files += ',' + mat_folder + view_names[vn] + name_postfix

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
    view_paths.update({'dataset': dt_name})

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

    save_file_name = save_folder + 'complementary_feas_' + dt_name + '.csv'
    out_fea = get_feas_all(views_union, res_union, ex_tys, platform, deas[0], 0, save_file_name, acq)
    return out_fea

if __name__ == "__main__":
    '''
    The following codes extract view combination features from view combinations of 11 DDA data and 7 DIA data
    '''
    current_path = os.path.dirname(os.path.abspath(__file__))

    ### !!! please replace the following Rscript_path with your R installing fold where there is a "Rscript" tool
    Rscript_path = 'D:/software/R-4.3.1/bin/'

    R_code = os.path.abspath(os.path.join(current_path, "..", "R_code")) + '/'
    save_folder = current_path + '/res/complementary_feas/'
    root_folder = os.path.abspath(os.path.join(current_path, "..", ""))
    data_fold = root_folder + "/data/benchmark_views/"

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    all_res = []
    acq = 'DDA'
    views_names_all = ['dlfq', 'maxlfq', 'top0', 'top3', 'count']
    reproduce_test(acq, views_names_all, all_res, R_code, Rscript_path, save_folder, data_fold)

    acq = 'DIA'
    views_names_all = ['dlfq', 'maxlfq', 'top3', 'top1']
    reproduce_test(acq, views_names_all, all_res, R_code, Rscript_path, save_folder, data_fold)

    '''
    The following codes extract the combination features from case study data view combinations
    TPD_dis and TPD_ret are two datasets of the thyroid nodule 
    CRC is the dataset of the colon cancer blood plasma samples
    U2OS_bulk is the dataset of U2OS cells treated with interferon-γ (IFN-γ)
    U2OS_scp is the single-cell proteomics data of the U2OS cells treated with or without IFN-γ
    '''

    dts = ['TPD_dis', 'TPD_ret', 'CRC', 'U2OS_bulk', 'U2OS_scp']
    for dt in dts:
        if dt == 'TPD_dis':

            platform = 'DIANN'
            dea = 'limma'
            acq = 'DIA'

            g1 = 'M'
            g2 = 'B'

            view_names_all = ['dlfq', 'maxlfq', 'top3', 'top1']
            design_path = root_folder + '/data/case_study_views/TPD_views/raw_avg_design_dis.csv'
            mat_folder = root_folder + '/data/case_study_views/TPD_views/raw_avg_'
            dt_name = 'TPD'
            logTs_all = ['F', 'F', 'F', 'F']
            name_postfix = '_dis.csv'

        elif dt == 'TPD_ret':

            platform = 'DIANN'
            dea = 'limma'
            acq = 'DIA'

            g1 = 'M'
            g2 = 'B'

            view_names_all = ['dlfq', 'maxlfq', 'top3', 'top1']
            design_path = root_folder + '/data/case_study_views/TPD_views/raw_avg_design_te.csv'
            mat_folder = root_folder + '/data/case_study_views/TPD_views/raw_avg_'
            dt_name = 'TPD_ret'
            logTs_all = ['F', 'F', 'F', 'F']
            name_postfix = '_ret.csv'

        elif dt == 'U2OS_scp':

            platform = 'DIANN'

            dea = 'limma'
            acq = 'DIA'

            g1 = 'Control'
            g2 = 'IFNy'

            view_names_all = ['dlfq', 'maxlfq', 'top3', 'top1']
            design_path = root_folder + '/data/case_study_views/U2OS/single_cell/U2OS_scp/design.tsv'
            mat_folder = root_folder + '/data/case_study_views/U2OS/single_cell/U2OS_scp/'
            dt_name = 'U2OS_filtered'
            logTs_all = ['T', 'F', 'F', 'F']
            name_postfix = '.tsv'

        elif dt == 'CRC':

            platform = 'Spectronaut'

            dea = 'limma'
            acq = 'DIA'

            g1 = 'N'
            g2 = 'C'

            view_names_all = ['dlfq', 'maxlfq', 'top3', 'top1']
            design_path = root_folder + '/data/case_study_views/CRC_views/design.tsv'
            mat_folder = root_folder + '/data/case_study_views/CRC_views/'
            dt_name = 'CRC'
            logTs_all = ['T', 'F', 'F', 'F']
            name_postfix = '.tsv'

        elif dt == 'U2OS_bulk':

            platform = 'Maxquant'

            dea = 'limma'
            acq = 'DDA'

            g1 = 'h0'
            g2 = 'h24'

            view_names_all = ['dlfq', 'maxlfq', 'top0', 'top3', 'count']
            design_path = root_folder + '/data/case_study_views/U2OS/bulk/design.tsv'
            mat_folder = root_folder + '/data/case_study_views/U2OS/bulk/'
            dt_name = 'INF' + '_' + g1 + '_' + g2
            logTs_all = ['T', 'T', 'T', 'T', 'F']
            name_postfix = '.tsv'

        get_fea_test(Rscript_path, platform, g1, g2, acq, view_names_all, logTs_all, R_code, dea, save_folder, mat_folder,
                 design_path, dt_name, name_postfix)


