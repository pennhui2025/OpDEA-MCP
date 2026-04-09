import copy
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, ShuffleSplit, StratifiedShuffleSplit, StratifiedKFold, KFold
from xgboost import XGBClassifier, XGBClassifier, XGBRegressor
from itertools import combinations
from xgboost import plot_importance
import random
import os
from scipy.stats import spearmanr
import optuna
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing

def seed_torch(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

def load_data(views_paths, acq, platform):

    views = [pd.read_csv(views_paths[i]) for i in range(len(views_paths))]
    feas_all = []
    fea_name = []
    if acq == 'DDA':
        views_names_all = ['dlfq', 'maxlfq', 'top0', 'top3', 'count']
    elif acq == 'DIA':
        views_names_all = ['dlfq', 'maxlfq', 'top3', 'top1']

    for i in range(2, len(views_names_all)):
        combs = np.array(list(combinations([x for x in range(len(views_names_all))], i)))

        for comb_idx in range(len(combs)):
            view_idx = combs[comb_idx, :]
            views_names = list(np.array(views_names_all)[view_idx])

            inds = [j for j in range(len(views_names))]

            fea_comp, fea_comp_name = get_complementarity(views, inds)
            for dea in ['ROTS', 'limma']:
                fea_stat, fea_stat_name = get_views_stat_feature(views, views_names, acq, dea, platform)
                feas_all.append(fea_comp + fea_stat)

                fea_name = fea_comp_name + fea_comp_name

    pd.DataFrame(np.array(feas_all), columns=fea_name).to_csv('complementary_feas_all.csv')
    return feas_all, fea_name

def objective(trail, data, label, seed, all_dt, train_o, K):
    seed_torch(seed)
    num = trail.suggest_int('num', 10, 500, step=10)  # n_estimators
    d = trail.suggest_int('d', 1, 10, step=1)  # max depth
    lr = trail.suggest_float('lr', 0.0001, 1)  # max depth
    # sub = trail.suggest_float('sub', 0.0001, 1)
    param = {'num': num, 'd': d, 'lr': lr}#, 'sub':sub}
    spr = train_test(data, label, seed, all_dt, train_o, param, K)

    return 1-spr

def train_test(data, label, seed, all_dt, train_o, param, K):
    rs_i = KFold(n_splits=K, shuffle=True, random_state=seed)

    sprs = []
    for j, (train_in, test_in) in enumerate(rs_i.split(train_o)):
        train_i = train_o[train_in]
        val = train_o[test_in]

        train_data_index = [np.where((np.array(all_dt) == train_i[k]))[0] for k in range(len(train_i))]
        val_data_index = [np.where((np.array(all_dt) == val[k]))[0] for k in range(len(val))]

        train_data = data[np.concatenate(train_data_index, dtype='int'), :]
        val_data = data[np.concatenate(val_data_index, dtype='int'), :]

        train_score = label[np.concatenate(train_data_index, dtype='int')]
        val_score = label[np.concatenate(val_data_index, dtype='int')]
        bst = []
        bst = XGBRegressor(n_estimators=param['num'], max_depth=param['d'], learning_rate=param['lr'], #subsample=param['sub'],
                           objective='reg:squarederror', booster='gbtree')

        bst.fit(train_data, train_score)
        pred = bst.predict(val_data)

        spr = spearmanr(val_score, pred).correlation

        sprs.append(spr)

    return np.mean(sprs)

def optuna_optimize(data, label, seed, all_dt, train_o, trial_num, K):
    seed_torch(seed)
    study = optuna.create_study(study_name='test', direction='minimize', sampler=optuna.samplers.TPESampler(seed=seed),
                                pruner=optuna.pruners.HyperbandPruner())

    func = lambda trial: objective(trial, data, label, seed, all_dt, train_o, K)

    study.optimize(func, n_trials=trial_num)

    return study.best_params

def cv_model(data, meta_info, label, cv_ty, trial_num, seed, prefix, root_fold, postfix, met, feature_name):

    all_dt = [meta_info['acq'].values[i] + '_' + str(meta_info['data_idx'].values[i]) for i in
              range(len(meta_info['acq'].values))]
    uni_dt = np.unique(all_dt)

    all_dt_plt = [meta_info['platform'].values[i] + '_' + str(meta_info['data_idx'].values[i]) for i in
              range(len(meta_info['platform'].values))]

    sprs = []
    dts = []

    if cv_ty == '5fold':
        rs_o = KFold(n_splits=5, shuffle=True, random_state=seed)
    elif cv_ty == 'lodocv':#leave one dataset out cv
        rs_o = KFold(n_splits=len(uni_dt), shuffle=True, random_state=seed)

    for i, (train_index, test_index) in enumerate(rs_o.split(uni_dt)):
        train_o = uni_dt[train_index]
        test = uni_dt[test_index]

        if cv_ty == '5fold':
            K = 4
        elif cv_ty == 'lodocv':
            K = len(uni_dt[train_index])
        param = optuna_optimize(data, label, seed, all_dt, train_o, trial_num, K)
        bst = []
        bst = XGBRegressor(n_estimators=param['num'], max_depth=param['d'], learning_rate=param['lr'],
                           objective='reg:squarederror', booster='gbtree')

        train_data_index = [np.where((np.array(all_dt) == train_o[k]))[0] for k in range(len(train_o))]
        train_data = data[np.concatenate(train_data_index, dtype='int'), :]
        train_score = label[np.concatenate(train_data_index, dtype='int')]
        bst.fit(train_data, train_score)

        for x in range(len(test)):
            test_data_index = np.where((np.array(all_dt) == test[x]))[0]
            uni_test_plt_dt = np.unique(np.array(all_dt_plt)[test_data_index])

            for y in range(len(uni_test_plt_dt)):
                test_data_idx_plt = np.where((np.array(all_dt_plt) == uni_test_plt_dt[y]))[0]
                test_data = data[np.array(test_data_idx_plt, dtype='int'), :]
                test_score = label[np.array(test_data_idx_plt, dtype='int')]

                pred = bst.predict(test_data)

                spr = spearmanr(test_score, pred).correlation

                sprs.append(spr)
                dts.append(uni_test_plt_dt[y])

    pd.DataFrame(np.column_stack([dts,sprs]), columns=['dataset', 'Spearman']).to_csv(root_fold + postfix + '_' + met + '_' + prefix + '_nest_cv_all_res_shap3.csv')

    param = optuna_optimize(data, label, seed, all_dt, uni_dt, trial_num, K)
    bst = []
    bst = XGBRegressor(n_estimators=param['num'], max_depth=param['d'], learning_rate=param['lr'],
                       objective='reg:squarederror', booster='gbtree')
    bst.fit(data, label)

    importance = bst.get_booster().get_score(importance_type="gain")
    importance_df = pd.DataFrame({
        "Feature": importance.keys(),
        "Importance": importance.values()
    })
    importance_df = importance_df.sort_values(by="Importance", ascending=False)
    feature_idx = np.array([str(i).replace('f', '') for i in importance_df['Feature'].values], dtype='int')
    importance_df['Feature Name'] = feature_name[feature_idx]
    importance_df.to_csv(root_fold + postfix + '_' + met + '_' + prefix + '_importance_shap3.csv')

    top_n = 10
    plt.figure(figsize=(10, 8))
    sns.barplot(x="Importance", y="Feature", data=importance_df.head(top_n), palette="viridis")
    plt.title(f"Top {top_n} Feature Importance (Gain) - XGBoost", fontsize=16)
    plt.xlabel("Importance Score", fontsize=18)
    plt.ylabel("Feature", fontsize=14)
    plt.yticks(fontsize=12)
    plt.xticks(fontsize=12)
    plt.tight_layout()
    #plt.show()

    explainer = shap.Explainer(bst)
    shap_values = explainer(data)

    shap_mean = np.abs(shap_values.values).mean(axis=0)
    top_indices = np.argsort(shap_mean)[-10:][::-1]  # 从大到小排序

    top_features = [feature_name[i] for i in top_indices]
    data_top = data[:, top_indices]
    shap_values_top = shap_values.values[:, top_indices]

    shap_mean = np.abs(shap_values_top).mean(axis=0)
    shap_df = pd.DataFrame({
        "Feature": top_features,
        "SHAP_Value": shap_mean
    }).sort_values(by="SHAP_Value", ascending=False)

    sns.set(style="whitegrid", font_scale=1.3, rc={
        'axes.edgecolor': 'black',
        'axes.linewidth': 0.8,
        'axes.facecolor': 'white',
        'axes.titlesize': 16,
        'axes.labelsize': 14,
        'legend.frameon': False,
        'legend.loc': 'best',
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'font.family': 'Arial'
    })

    top_n = 10
    plt.figure(figsize=(7, 6))
    ax = sns.barplot(
        x="SHAP_Value",
        y="Feature",
        data=shap_df.head(top_n),
        palette="RdBu_r"
    )
    plt.title("Top 10 Feature Contributions (SHAP)", fontsize=18, weight='bold')
    plt.xlabel("Mean |SHAP value|", fontsize=18)
    plt.ylabel("Feature", fontsize=18)

    sns.despine(trim=True)
    plt.tight_layout()
    # plt.show()

    plt.savefig(root_fold + postfix + '_' + met + '_' + prefix + "_shap_feature_importance_nature_style3.svg")
    plt.savefig(root_fold + postfix + '_' + met + '_' + prefix + "_shap_feature_importance_nature_style3.pdf")
    plt.savefig(root_fold + postfix + '_' + met + '_' + prefix + "_shap_feature_importance_nature_style3.png", dpi=300)
    plt.close()

    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values_top, data_top, feature_names=top_features, plot_type="dot", show=False)

    plt.gca().set_facecolor('white')
    plt.title("SHAP Summary Plot", fontsize=16, weight='bold')
    plt.xlabel("SHAP value (impact on model output)", fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    #plt.show()

    plt.savefig(root_fold + postfix + '_' + met + '_' + prefix + "_shap_summary_plot_nature_style3.svg")
    plt.savefig(root_fold + postfix + '_' + met + '_' + prefix + "_shap_summary_plot_nature_style3.pdf")
    plt.savefig(root_fold + postfix + '_' + met + '_' + prefix + "_shap_summary_plot_nature_style3.png", dpi=300)
    plt.close()

    shap_df = pd.DataFrame(shap_values_top, columns=top_features)
    data_df = pd.DataFrame(data_top, columns=top_features)
    shap_df.to_csv(root_fold + postfix + '_' + met + '_' + prefix + "_shap_values_matrix3.csv", index=False)
    data_df.to_csv(root_fold + postfix + '_' + met + '_' + prefix + "_input_values_matrix3.csv", index=False)

    return bst

def T_model(test_data, bst, meta_info, root_fold, dt, postfix):
    pred = bst.predict(test_data)

    print(pred)
    res = pd.DataFrame(np.array([pred]).T, columns=['pred_mcc'])
    out_pd = pd.concat([meta_info, res], axis=1)
    out_pd.to_csv(root_fold + dt + '_' + postfix + '_pred_performance.csv')
    return out_pd

def process_data(data_raw, scale, dea_perform_path, parent_fold, dataset):

    data = data_raw.iloc[:, 7:].values
    feature_name = data_raw.columns.values

    if 'mr_reduce' not in feature_name:
        mr_reduce = np.array([min(data[i, 19:25]) - data[i, 25] for i in range(len(data[:, 0]))])
        data_raw['mr_reduce'] = mr_reduce

    feature_name = data_raw.columns.values
    method = 'DP_fdr'
    metric = 'MCC'

    final_out_pd = pd.DataFrame(data=None)

    uni_plt = np.unique(data_raw['platform'].values)
    for plt in uni_plt:
        meta_plt = data_raw.iloc[np.where((data_raw['platform'].values == plt))[0], :]
        uni_dt = np.unique(meta_plt['data_idx'].values)

        for t in range(0, len(uni_dt)):
            dt = uni_dt[t]
            meta_plt_dt = meta_plt.iloc[np.where((meta_plt['data_idx'].values == dt))[0], :]
            acqs = meta_plt_dt['acq'].values
            plts = meta_plt_dt['platform'].values
            vnums = meta_plt_dt['view_num'].values
            vnames = meta_plt_dt['view_name'].values
            deas = meta_plt_dt['dea'].values

            if dea_perform_path != '':
                dt_perform = []
                for i in range(len(acqs)):
                    file_name = dea_perform_path + acqs[i] + '_' + plts[i] + '_' + str(vnums[i]) + 'views_' + vnames[
                        i] + '.xlsx'
                    perf_dt = pd.read_excel(file_name, sheet_name='all')
                    if (plt == 'DIANN') | (plt == 'Spectronaut'):

                        dts_dia = pd.read_csv(parent_fold + '/DEA_Benchmarking/data/dataset_info/DIA_DIANN.txt',
                            sep='\t', header=0)
                        dt_name = dts_dia['dataset'].values[dt] + '_' + plts[i] + '_' + deas[i]

                    else:
                        dts_dda = pd.read_csv(
                            parent_fold + '/DEA_Benchmarking/data/dataset_info/DDA_Frag.txt',
                            sep='\t', header=0)
                        dt_name = dts_dda['dataset'].values[dt] + '_' + plts[i] + '_' +  deas[i]

                    if len(np.where(((perf_dt['dataset'].values == dt_name) & (perf_dt['Method'].values == method) &
                                      (perf_dt['Metric'].values == metric)))[0]) > 0:

                        perf_dt_dl_pred_mcc = perf_dt['Performance'].values[np.where(((perf_dt['dataset'].values == dt_name) & (perf_dt['Method'].values == method) &
                                      (perf_dt['Metric'].values == metric)))[0][0]]
                    else:
                        perf_dt_dl_pred_mcc = np.nan

                    dt_perform.append(perf_dt_dl_pred_mcc)

                print(t)

                dt_perform = np.array([dt_perform], dtype='float').T
                dt_perform = preprocessing.MinMaxScaler().fit_transform(dt_perform)
                out_data = np.column_stack((meta_plt_dt.values, dt_perform))
                out_data = out_data[np.where((~np.isnan(dt_perform)))[0], :]
                feature_name = np.array(list(feature_name) + ['dt_perfom'])

            else:
                out_data = meta_plt_dt.values

            meta_info = out_data[:, 0:7]
            feas = out_data[:, 7:]
            fea_names = feature_name[7:]

            bin_fea_idx = [13, 14, 15, 16, 17, 18, 26, 27, 28, 29, 30, 31, 32, 33]
            non_bin_fea_idx = np.array(list(set([i for i in range(len(feas[0, :]))]) - set(bin_fea_idx)), dtype='int')

            if scale == 'T':
                norm_feas = preprocessing.MinMaxScaler().fit_transform(feas)
                feas[:, non_bin_fea_idx] = norm_feas[:, non_bin_fea_idx]

            if dea_perform_path != '':
                out_pd = pd.DataFrame(np.column_stack((meta_info, feas[:, non_bin_fea_idx])),
                                      columns=list(data_raw.columns.values[0:7]) + list(fea_names[non_bin_fea_idx]))

            else:

                out_pd = pd.DataFrame(np.column_stack((meta_info, feas[:, non_bin_fea_idx])), columns=list(data_raw.columns.values[0:7]) + list(fea_names[non_bin_fea_idx]))
            final_out_pd = pd.concat([final_out_pd, out_pd], axis=0)

    meta_info_out = final_out_pd.iloc[:, :7]
    if dea_perform_path != '':
        label_out = final_out_pd.iloc[:, -1].values
        data_out = final_out_pd.iloc[:, 7:-1]
    else:
        label_out = []
        data_out = final_out_pd.iloc[:, 7:]
    final_out_pd.to_csv(dataset + '_feature.csv')
    return final_out_pd, meta_info_out, label_out, data_out


if __name__ == "__main__":

    current_fold = os.path.dirname(__file__) + '/'
    parent_fold = os.path.dirname(os.path.dirname(current_fold))
    train_data_raw = pd.read_csv(parent_fold + '/Comlementary_feature_extraction/res/complementary_feas/complementary_feas_benchmark.csv', header=0)

    dea_perform_path = parent_fold + '/Fig2/res/res_MCP/'
    scale = 'T'
    final_out_pd, meta_info_out, label_out, data_out = process_data(train_data_raw, scale, dea_perform_path, parent_fold,'train')
    postfix = 'test'
    met = 'MCC'

    meta_info = meta_info_out
    label = label_out
    data = data_out.values
    ty = '5fold'
    feature_name = data_out.columns.values
    trial_num = 30
    seed = 2025

    trained_model_path = current_fold + 'trained_model_scale_' + scale + '_trial' + str(trial_num) + ty + '.joblib'
    if not os.path.exists(trained_model_path):

        bst = cv_model(data, meta_info, label, ty, trial_num, seed, '', current_fold, postfix, met, feature_name)
        joblib.dump(bst, trained_model_path)
    else:
        bst = joblib.load(trained_model_path)

    dts = ['U2OS_filtered', 'TPD', 'TPD_ret', 'CRC', 'INF_h0_h24']

    for dt in dts:
        fea_fold = parent_fold + '/Comlementary_feature_extraction/res/complementary_feas/complementary_feas'
        test_data_raw = pd.read_csv(fea_fold + dt + '.csv', header=0)
        final_test_pd, meta_info_test, label_test, data_test = process_data(test_data_raw, scale, '', parent_fold, dt)

        meta_info = meta_info_test
        test_data = data_test.values
        T_model(test_data, bst, meta_info, current_fold, dt, ty)













