import copy
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score as acc, recall_score as recall, precision_score as precision, f1_score as f1, matthews_corrcoef as mcc
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
import random
import os
import optuna
import csv
from sklearn.model_selection import KFold
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import spearmanr as spr, pearsonr as pr
from sklearn.cross_decomposition import CCA, PLSCanonical
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KNeighborsRegressor
from Utils import process_proteingroup

def seed_torch(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

seed_torch()

def obtain_fasta(path):
    filepath = path
    fasta = {}
    with open(filepath) as file_one:
        for line in file_one:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                active_sequence_name = line[1:]
                if active_sequence_name not in fasta:
                    fasta[active_sequence_name] = ''
                continue
            sequence = line
            fasta[active_sequence_name] += sequence

    protein_names = []
    protein_seqs = []
    protein_len = []
    for key in fasta.keys():
        strs = key.split(' ')
        protein_names.append(strs[0])
        protein_len.append(len(fasta[key]))
        protein_seqs.append(fasta[key])

    out = np.column_stack((np.array([protein_names]).T, np.array([protein_len]).T))
    return fasta, out

def label_pro(logFC, qval, logfc_thr, q_thr):

    label = np.zeros((len(qval), 1))
    #label[np.where((qval <= q_thr))[0]] = 1
    label[np.where(((abs(logFC) >= float(logfc_thr)) & (qval <= float(q_thr))))[0]] = 1
    #label[np.where(((logFC <= -logfc_thr) & (qval <= q_thr)))[0]] = -1

    return label

def get_uniport_id(pros):
    uni_id = []
    gene_id = []
    species = []

    for pro in pros:
        #print(pro)
        if ';' in pro:
            pro_s = pro.split(';')
        elif ',' in pro:
            pro_s = pro.split(',')
        else:
            pro_s = [pro]
        if len(pro_s) == 1:
            if ('UPS' in pro_s[0]) | ('ups' in pro_s[0]):
                uni_id.append(pro_s[0].split('|')[0].replace('ups', '').replace('UPS', ''))
                gene_id.append(pro_s[0].split('|')[1].split('_')[0])
                species.append('UPS')
            else:
                if '|' in pro_s[0]:
                    uni_id.append(pro_s[0].split('|')[1])
                    gene_species = pro_s[0].split('|')[2].split('_')
                    gene_id.append(pro_s[0].split('|')[2].split('_')[0])
                    if len(gene_species)>1:
                        species.append(pro_s[0].split('|')[2].split('_')[1])
                    else:
                        species.append('')
                else:
                    uni_id.append(pro_s[0])
                    gene_id.append('')
                    species.append('')

        elif len(pro_s) > 1:
            spes = []
            ids = ''
            ges = ''
            for proi in pro_s:

                if ('UPS' in proi) | ('ups' in proi):
                    prois = proi.split('|')[0].replace('ups', '').replace('UPS', '')
                    genei = proi.split('|')[1].split('_')[0]
                    spei = 'UPS'
                else:
                    if '|' in proi:
                        prois = proi.split('|')[1]
                        genei = proi.split('|')[2].split('_')[0]
                        spei = proi.split('|')[2].split('_')[1]
                    else:
                        prois = proi
                        genei = ''
                        spei = ''

                if ids == '':
                    ids = prois
                    ges = genei
                else:
                    ids = ids + ';' + prois
                    ges = ges + ';' + genei

                spes.append(spei)

            uni_id.append(ids)
            gene_id.append(ges)

            if(len(np.unique(spes))>1):
                species.append('mixed')
            else:
                species.append(np.unique(spes)[0])

    return uni_id, gene_id, species

def get_true_label(pro, true_organism):
    true_label = []

    if('UPS' in true_organism):
        ups_fasta, ups_stats = obtain_fasta('ups1-ups2-sequences.fasta')
        ups_pro = list(ups_stats[:, 0])
        ups_id, ups_gene, ups_spec = get_uniport_id(ups_pro)

    pro_ids, pro_genes, pro_specs = get_uniport_id(pro)

    for i in range(len(pro)):

        if 'UPS' in true_organism:
            if (pro_ids[i] in ups_id) | (pro_specs[i] in true_organism):
                true_label.append(1)
            else:
                true_label.append(0)
        else:
            if pro_specs[i] in true_organism:
                true_label.append(1)
            else:
                true_label.append(0)

    return true_label

def cal_metrics1(score, q_thr, FC, fc_thr, true_lab):
    qval = 1-score
    y_true = true_lab
    y_pred = (qval <= q_thr) & (abs(FC)>=fc_thr)
    Acc = acc(y_true, y_pred)
    Prec = precision(y_true, y_pred)
    Rec = recall(y_true, y_pred)
    F1 = f1(y_true, y_pred)
    F1w = f1(y_true, y_pred, average='weighted')
    Mcc = mcc(y_true, y_pred)
    nMcc = (Mcc + 1) / 2


    if (len(np.unique(true_lab))==1):

        tn, fp, fn, tp = 0, 0, 0, 0
        spec = 0
        geomean = 0
        pauc001 = 0
        pauc005 = 0
        pauc01 = 0
    elif (len(np.unique(true_lab))>1):
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        spec = tn / (tn + fp)
        geomean = np.sqrt(Rec * spec)
        pauc001 = roc_auc_score(true_lab, score, max_fpr=0.01)
        pauc005 = roc_auc_score(true_lab, score, max_fpr=0.05)
        pauc01 = roc_auc_score(true_lab, score, max_fpr=0.1)

    return Acc, Prec, Rec, F1, F1w, Mcc, nMcc, geomean, tn, fp, fn, tp, pauc001, pauc005, pauc01

def objective(trail, feas, scores, FC, seed, model_type, de_thr, diff_ty, lgfc_thr, views_names):
    seed_torch(seed)

    if model_type == 'KNN':
        C = trail.suggest_int('C', 2, 25, step=1)
        e = trail.suggest_int('e', 1, 5, step=1)

        param = {'C':C, 'e':e}

    elif model_type == 'MLE': # multi-ML averaging

        view_num = len(views_names)
        Cs = []
        es = []

        for v in range(view_num):
            C = trail.suggest_int('C' + str(v), 2, 25, step=1)
            e = trail.suggest_int('e' + str(v), 1, 5, step=1)

            Cs.append(C)
            es.append(e)

        param = {'Cs': Cs, 'es': es}

    elif model_type == 'GCCA':  # MCCA multi-view reduction for representation

        n_components = trail.suggest_int('n_components', 1, 10, step=1)
        c = trail.suggest_float('c', 1e-12, 20)
        C = trail.suggest_int('C', 2, 25, step=1)
        e = trail.suggest_int('e', 1, 5, step=1)
        param = {'n_components': n_components, 'c': c, 'C': C, 'e': e}

    pmse, spr, pr, rt, auc = cv_model_mv(feas, scores, FC, seed, model_type, de_thr, param, lgfc_thr, views_names)

    if diff_ty == 'mse':
        return pmse
    elif diff_ty == 'spr':
        return 1-spr
    elif diff_ty == 'pr':
        return 1-pr
    elif diff_ty == 'r2':
        return 1-rt
    elif diff_ty == 'auc':
        return 1 - auc

def optuna_optimize_mv(train_feas, train_scores, train_FC, seed, model_type, test_feas_final, trial_num, a, de_thr, diff_ty, lgfc_thr, views_names):
    seed_torch(seed)
    study = optuna.create_study(study_name='test', direction='minimize', sampler=optuna.samplers.TPESampler(seed=seed),
                                pruner=optuna.pruners.HyperbandPruner())

    func = lambda trial: objective(trial, train_feas, train_scores,  train_FC, seed, model_type, de_thr, diff_ty, lgfc_thr, views_names)
    study.optimize(func, n_trials=trial_num)
    print(study.best_params)
    print(study.best_trial)
    print(study.best_trial.value)
    print(model_type)
    pred_final, pred_train, trains_lat, test_lat = test_model_mv(train_feas, train_scores, train_FC, seed, model_type, test_feas_final, study.best_params, a, de_thr, lgfc_thr, views_names)
    return pred_final, pred_train, study.best_params, trains_lat, test_lat

def pmse(pred_score, true_score):

    pmse = mean_squared_error(pred_score, true_score)

    return pmse

def cv_model(feas, scores, FC, seed, model_type, de_thr, param, lgfc_thr):
    seed_torch(seed)
    lab = np.zeros((len(scores), 1))
    lab[np.where((((1-scores) < de_thr) & (abs(FC)>=lgfc_thr))), 0] = 1
    num_pos = len(np.where((lab[:,0] == 1))[0])
    num_neg = len(np.where((lab[:,0] == 0))[0])

    ratio = num_neg / num_pos

    train_feas = []
    train_scores = []
    N = 0

    if (ratio > 0.5) & (ratio <= 1) : # more positive than negative
        select_pos = np.random.randint(0, num_pos, size=num_neg)

        train_feas_balance = feas[np.array(
            list(np.where((lab[:, 0] == 1))[0][select_pos]) + list(np.where((lab[:, 0] == 0))[0])), :]
        train_score_balance = scores[
            np.array(list(np.where((lab[:, 0] == 1))[0][select_pos]) + list(np.where((lab[:, 0] == 0))[0]))]

        train_feas.append(train_feas_balance)
        train_scores.append(train_score_balance)
        N =1

    elif (ratio > 1) & (ratio < 2):

        select_neg = np.random.randint(0, num_neg, size=num_pos)

        train_feas_balance = feas[np.array(
            list(np.where((lab[:, 0] == 1))[0]) + list(np.where((lab[:, 0] == 0))[0][select_neg])), :]
        train_score_balance = scores[
            np.array(list(np.where((lab[:, 0] == 1))[0]) + list(np.where((lab[:, 0] == 0))[0][select_neg]))]

        train_feas.append(train_feas_balance)
        train_scores.append(train_score_balance)

        N = 1

    elif ratio < 0.5:

        N = int(num_pos / num_neg)
        remain_pos = np.where((lab[:, 0] == 1))[0]
        neg_idx = np.where((lab[:, 0] == 0))[0]

        for i in range(N):
            select_pos = np.random.randint(0, len(remain_pos), size=num_neg)
            train_feas_balance = feas[np.array(list(neg_idx) + list(remain_pos[select_pos])), :]
            train_score_balance = scores[np.array(list(neg_idx) + list(remain_pos[select_pos]))]
            train_feas.append(train_feas_balance)
            train_scores.append(train_score_balance)
            remain_pos = np.setdiff1d(remain_pos, remain_pos[select_pos])

    elif ratio > 2:

        N = int(ratio)

        remain_neg = np.where((lab[:, 0] == 0))[0]
        pos_idx = np.where((lab[:, 0] == 1))[0]

        for i in range(N):
            select_neg = np.random.randint(0, len(remain_neg), size=num_pos)
            train_feas_balance = feas[np.array(list(pos_idx) + list(remain_neg[select_neg])), :]
            train_score_balance = scores[np.array(list(pos_idx) + list(remain_neg[select_neg]))]
            train_feas.append(train_feas_balance)
            train_scores.append(train_score_balance)
            remain_neg = np.setdiff1d(remain_neg, remain_neg[select_neg])

    pmses = 0
    n = 0
    sprs = 0
    prs = 0
    rt = 0

    for i in range(N):#[0]:#
        print(i)
        seed_torch(seed)
        logo = KFold(n_splits=3, shuffle=True)
        for train_index, test_index in logo.split(train_scores[i]):
            train_x = train_feas[i][train_index]
            train_lab = train_scores[i][train_index]

            test_x = train_feas[i][test_index]
            test_lab = train_scores[i][test_index]

            train_X, train_y = train_x, train_lab
            test_X, test_y = test_x, test_lab
            if model_type != 'mv':
                if model_type == 'SVM':
                    #model = SVR(kernel = "linear", C=param['C'])#, epsilon=param['e'])
                    model = SVR(kernel="rbf", C=param['C'], epsilon=param['e'])
                elif model_type == 'RF':
                    model = RandomForestRegressor(n_estimators=param['N'], max_depth=param['md'], random_state=seed)
                elif model_type == 'XG':
                    seed_torch(seed)

                    model = xgb.XGBRegressor(objective='reg:squarederror',
                                            max_depth=param['max_depth'],
                                            eta=param['lr'],
                                            subsample=param['subsample'],
                                            colsample_bytree=param['colsample_bytree'],
                                            n_estimators=param['nume'])




                model.fit(train_X, train_y)
                pred_score = model.predict(test_X)
                pmses += pmse(pred_score, test_y)
                sprs += spr(pred_score, test_y).statistic
                if np.isnan(pr(pred_score, test_y).statistic):
                    prs += 0
                else:
                    prs += pr(pred_score, test_y).statistic
                rt += r2_score(pred_score, test_y)
                n += 1
            elif model_type == 'mv':

                # Multi-view co-training semi-supervised learning
                if param['n1'] > len(train_X[:, 0]):
                    nnei1 = len(train_X[:, 0])
                else:
                    nnei1 = param['n1']

                if param['n2'] > len(train_X[:, 0]):
                    nnei2 = len(train_X[:, 0])
                else:
                    nnei2 = param['n2']

                estimator1 = KNeighborsRegressor(n_neighbors=nnei1, p=param['p1'])
                estimator2 = KNeighborsRegressor(n_neighbors=nnei2, p=param['p2'])
                model = CTRegressor(estimator1, estimator2, random_state=seed)

                # Train a CTClassifier on all the labeled and unlabeled training data
                model.fit([train_X[:,0:int(len(train_X[0, :])/2)], train_X[:,int(len(train_X[0,:])/2):len(train_X[0,:])]], train_y)
                pred_score = model.predict([test_X[:, 0:int(len(test_X[0,:])/2)], test_X[:,int(len(test_X[0,:])/2):len(test_X[0,:])]])#knn.predict([X_test, X_test])
                pmses += pmse(pred_score, test_y)
                sprs += spr(pred_score, test_y).statistic
                if np.isnan(pr(pred_score, test_y).statistic):
                    prs += 0
                else:
                    prs += pr(pred_score, test_y).statistic
                rt += r2_score(pred_score, test_y)
                n += 1

    return pmses/n, sprs/n, prs/n, rt/n

def test_model(feas, scores, FC, seed, model_type, test_feas_final, params, a, de_thr, lgfc_thr):
    seed_torch(seed)
    model = []
    if model_type == 'SVM':
        seed_torch(seed)
        model = SVR(C=params['C'], epsilon=params['e'])
        #model = SVR(kernel="linear", C=params['C'])  # , epsilon=param['e'])
    elif model_type == 'RF':
        seed_torch(seed)
        model = RandomForestRegressor(n_estimators=params['nbr'], max_depth=params['md'], random_state=seed)
    elif model_type == 'XG':
        seed_torch(seed)
        model = xgb.XGBRegressor(objective='reg:squarederror',
                                 max_depth=params['max_depth'],
                                 eta=params['lr'],
                                 subsample=params['subsample'],
                                 colsample_bytree=params['colsample_bytree'],
                                 n_estimators=params['nume'])

    elif model_type == 'mv':
        seed_torch(seed)
        estimator1 = KNeighborsRegressor(n_neighbors=params['n1'], p=params['p1'])
        estimator2 = KNeighborsRegressor(n_neighbors=params['n2'], p=params['p2'])
        model = CTRegressor(estimator1, estimator2, random_state=seed)

    lab = np.zeros((len(scores), 1))
    lab[np.where(((1 - scores) < de_thr, abs(FC)>=lgfc_thr)), 0] = 1
    num_pos = len(np.where((lab[:, 0] == 1))[0])
    num_neg = len(np.where((lab[:, 0] == 0))[0])

    ratio = num_neg / num_pos

    train_feas_all = []
    train_scores_all = []

    N = 0

    if (ratio > 0.5) & (ratio <= 1):  # more positive than negative but wi
        select_pos = np.random.randint(0, num_pos, size=num_neg)

        train_feas_balance = feas[np.array(
            list(np.where((lab[:, 0] == 1))[0][select_pos]) + list(np.where((lab[:, 0] == 0))[0])), :]
        train_score_balance = scores[
            np.array(list(np.where((lab[:, 0] == 1))[0][select_pos]) + list(np.where((lab[:, 0] == 0))[0]))]

        train_feas_all.append(train_feas_balance)
        train_scores_all.append(train_score_balance)
        N = 1

    elif (ratio > 1) & (ratio < 2):

        select_neg = np.random.randint(0, num_neg, size=num_pos)

        train_feas_balance = feas[np.array(
            list(np.where((lab[:, 0] == 1))[0]) + list(np.where((lab[:, 0] == 0))[0][select_neg])), :]
        train_score_balance = scores[
            np.array(list(np.where((lab[:, 0] == 1))[0]) + list(np.where((lab[:, 0] == 0))[0][select_neg]))]

        train_feas_all.append(train_feas_balance)
        train_scores_all.append(train_score_balance)

        N = 1

    elif ratio < 0.5:

        N = int(num_pos / num_neg)
        remain_pos = np.where((lab[:, 0] == 1))[0]
        neg_idx = np.where((lab[:, 0] == 0))[0]

        for i in range(N):
            select_pos = np.random.randint(0, len(remain_pos), size=num_neg)
            train_feas_balance = feas[np.array(list(neg_idx) + list(remain_pos[select_pos])), :]
            train_score_balance = scores[np.array(list(neg_idx) + list(remain_pos[select_pos]))]
            train_feas_all.append(train_feas_balance)
            train_scores_all.append(train_score_balance)
            remain_pos = np.setdiff1d(remain_pos, remain_pos[select_pos])

    elif ratio > 2:

        N = int(ratio)

        remain_neg = np.where((lab[:, 0] == 0))[0]
        pos_idx = np.where((lab[:, 0] == 1))[0]

        for i in range(N):
            select_neg = np.random.randint(0, len(remain_neg), size=num_pos)
            train_feas_balance = feas[np.array(list(pos_idx) + list(remain_neg[select_neg])), :]
            train_score_balance = scores[np.array(list(pos_idx) + list(remain_neg[select_neg]))]
            train_feas_all.append(train_feas_balance)
            train_scores_all.append(train_score_balance)
            remain_neg = np.setdiff1d(remain_neg, remain_neg[select_neg])

    pfs = []
    trs = []
    for i in range(N):#[0]:#
        if model_type != 'mv':
            model.fit(train_feas_all[i], train_scores_all[i])
            pfs.append(model.predict(test_feas_final))
            trs.append(model.predict(feas))
        elif model_type == 'mv':
            model.fit([train_feas_all[i][:, 0:int(len(train_feas_all[i][0, :])/2)],
                       train_feas_all[i][:, int(len(train_feas_all[i][0, :])/2):len(train_feas_all[i][0, :])]]
                      , train_scores_all[i])

            pfs.append(model.predict([test_feas_final[:, 0:int(len(test_feas_final[0, :])/2)],
                                      test_feas_final[:, int(len(test_feas_final[0, :])/2):len(test_feas_final[0, :])]]))
            trs.append(model.predict([feas[:, 0:int(len(feas[0, :])/2)],
                                      feas[:, int(len(feas[0, :])/2):len(feas[0, :])]]))



    pf = np.mean(np.array(pfs).T, axis=1)
    tr = np.mean(np.array(trs).T, axis=1)

    # pf = preprocessing.MinMaxScaler(feature_range=(min(scores), max(scores))).fit_transform(np.array([pf]).T)[:, 0]
    # tr = preprocessing.MinMaxScaler(feature_range=(min(scores), max(scores))).fit_transform(np.array([tr]).T)[:, 0]
    pf[np.where((pf >= 1))[0]] = max(pf[pf<1])
    pf[np.where((pf < 0))[0]] = 0

    tr[np.where((tr >= 1))[0]] = max(tr[tr<1])
    tr[np.where((tr < 0))[0]] = 0

    #pl = preprocessing.MinMaxScaler(feature_range=(0, 1)).fit_transform(np.array([pl]).T)[:, 0]
    #pf = preprocessing.MinMaxScaler(feature_range=(0, 1)).fit_transform(np.array([pf]).T)[:, 0]
    #pred_local
    #pred_local = mm.inverse_transform(np.array([pl]).T)
    #pred_final = mm.inverse_transform(np.array([pf]).T)

    return pf*a, tr*a

def get_training_proteins(proteins_v1, proteins_v2, view_protein, qval_v1, qval_v2, logFC_v1, logFC_v2, fea_all,
                          hc_thr, de_thr, npos_thr, combine_score_ty, hurdle_qval, hurdle_protein):
    seed_torch(2023)

    label_v1 = label_pro(logFC_v1, qval_v1, np.log2(1.5), de_thr)
    label_v2 = label_pro(logFC_v2, qval_v2, np.log2(1.5), de_thr)

    com, idx1, idx2 = np.intersect1d(proteins_v1, proteins_v2, return_indices=True)

    com_pro = com
    com_pro_lab1 = label_v1[idx1, 0]
    com_pro_lab2 = label_v2[idx2, 0]

    com_pro_cons = com_pro[com_pro_lab1 == com_pro_lab2]

    hc_pro_v1 = proteins_v1[np.where((qval_v1 <= hc_thr))[0]] # highly confident DE proteins
    hc_pro_v2 = proteins_v2[np.where((qval_v2 <= hc_thr))[0]]

    train_protein = np.unique(np.array(list(com_pro_cons)))

    train_qval_v1 = np.ones((len(train_protein), 1))
    train_logfc_v1 = np.zeros((len(train_protein), 1))
    com_v1, idx_v1, idx_tr1 = np.intersect1d(proteins_v1, train_protein, return_indices=True)
    train_qval_v1[idx_tr1, 0] = qval_v1[idx_v1]
    train_logfc_v1[idx_tr1, 0] = logFC_v1[idx_v1]

    train_qval_v2 = np.ones((len(train_protein), 1))
    train_logfc_v2 = np.zeros((len(train_protein), 1))
    com_v2, idx_v2, idx_tr2 = np.intersect1d(proteins_v2, train_protein, return_indices=True)
    train_qval_v2[idx_tr2, 0] = qval_v2[idx_v2]
    train_logfc_v2[idx_tr2, 0] = logFC_v2[idx_v2]

    if combine_score_ty == 'min':
        train_qval = np.array([min(train_qval_v1[i, 0], train_qval_v2[i, 0]) for i in range(len(train_qval_v1))])
    elif combine_score_ty == 'mean':
        train_qval = np.array([np.mean([train_qval_v1[i, 0], train_qval_v2[i, 0]]) for i in range(len(train_qval_v1))])
    elif combine_score_ty == 'max':
        train_qval = np.array([max(train_qval_v1[i, 0], train_qval_v2[i, 0]) for i in range(len(train_qval_v1))])
    elif combine_score_ty == 'hurdle':
        train_qval_hurdle = np.ones((len(train_protein), 1))
        com_v2, idx_v2, idx_tr2 = np.intersect1d(hurdle_protein, train_protein, return_indices=True)
        train_qval_hurdle[idx_tr2, 0] = hurdle_qval[idx_v2]
        train_qval = train_qval_hurdle[:, 0]

    if (len(np.where((np.array(train_qval) <= de_thr))[0]) < npos_thr) & (len(np.where((np.array(train_qval) <= de_thr))[0])>=0):
        train_protein = np.unique(np.array(list(com_pro_cons) + list(hc_pro_v1) + list(hc_pro_v2)))

        train_qval_v1 = np.ones((len(train_protein), 1))
        train_logfc_v1 = np.zeros((len(train_protein), 1))
        com_v1, idx_v1, idx_tr1 = np.intersect1d(proteins_v1, train_protein, return_indices=True)
        train_qval_v1[idx_tr1, 0] = qval_v1[idx_v1]
        train_logfc_v1[idx_tr1, 0] = logFC_v1[idx_v1]

        train_qval_v2 = np.ones((len(train_protein), 1))
        train_logfc_v2 = np.zeros((len(train_protein), 1))
        com_v2, idx_v2, idx_tr2 = np.intersect1d(proteins_v2, train_protein, return_indices=True)
        train_qval_v2[idx_tr2, 0] = qval_v2[idx_v2]
        train_logfc_v2[idx_tr2, 0] = logFC_v2[idx_v2]

        train_qval = np.array([min(train_qval_v1[i, 0], train_qval_v2[i, 0]) for i in range(len(train_qval_v1))])

        if len(np.where((np.array(train_qval) <= de_thr))[0]) < npos_thr:
            print('no enough positive training data!!!')

            train_pro = []
            train_fea = []
            train_scores = []
            train_qval_v1 = []
            train_logfc_v1 = []
            train_qval_v2 = []
            train_logfc_v2 = []

            return (train_pro, train_fea, train_scores, train_qval_v1, train_logfc_v1,
            train_qval_v2, train_logfc_v2)

    train_score = 1-train_qval

    com_pro_train, idx11, idx12 = np.intersect1d(train_protein, view_protein, return_indices=True)

    train_pro = com_pro_train
    train_fea = fea_all[idx12, :]
    train_scores = train_score[idx11]

    train_qval_v1 = train_qval_v1[idx11]
    train_logfc_v1 = train_logfc_v1[idx11]

    train_qval_v2 = train_qval_v2[idx11]
    train_logfc_v2 = train_logfc_v2[idx11]

    return (train_pro, train_fea, train_scores, train_qval_v1, train_logfc_v1,
            train_qval_v2, train_logfc_v2)

def get_test_proteins(proteins_v1, proteins_v2, view_protein, score_v1, score_v2, logFC_v1, logFC_v2, fea_all,
                          hurdle_proteins, hurdle_score, train_pro):

    test_protein = np.setdiff1d(view_protein, train_pro)

    com_pro_test, idx11, idx12 = np.intersect1d(test_protein, view_protein, return_indices=True)

    test_pro = com_pro_test
    test_fea = fea_all[idx12, :]

    score_v1_test = np.zeros((len(test_pro), 1))
    score_v2_test = np.zeros((len(test_pro), 1))
    logfc_v1_test = np.zeros((len(test_pro), 1))
    logfc_v2_test = np.zeros((len(test_pro), 1))

    com1, idx111, idx112 = np.intersect1d(test_pro, proteins_v1, return_indices=True)
    com2, idx211, idx212 = np.intersect1d(test_pro, proteins_v2, return_indices=True)

    score_v1_test[idx111, 0] = score_v1[idx112]
    score_v2_test[idx211, 0] = score_v2[idx212]
    logfc_v1_test[idx111, 0] = logFC_v1[idx112]
    logfc_v2_test[idx211, 0] = logFC_v2[idx212]

    test_score_hurdle = np.zeros((len(test_pro), 1))
    com_h, idx_h, idx_l = np.intersect1d(hurdle_proteins, test_pro, return_indices=True)
    idx_h_l = np.array([list(hurdle_proteins).index(com_h[i]) for i in range(len(com_h))], dtype='int')
    test_score_hurdle[idx_l, 0] = hurdle_score[idx_h_l]

    ## simply mirror feature from view A to view B if view B are largely 0

    # all_missing_v1_idx = np.where(((score_v1_test[:, 0] == 0) & (logfc_v1_test[:, 0] == 0)))[0]
    # all_missing_v2_idx = np.where(((score_v2_test[:, 0] == 0) & (logfc_v2_test[:, 0] == 0)))[0]
    #
    # mirror_v1_idx = np.setdiff1d(all_missing_v1_idx, all_missing_v2_idx)
    # mirror_v2_idx = np.setdiff1d(all_missing_v2_idx, all_missing_v1_idx)
    # if (len(mirror_v1_idx)>0):
    #     test_fea[mirror_v1_idx, 0:int(len(test_fea[0, :])/2)] = test_fea[mirror_v1_idx, int(len(test_fea[0, :])/2):len(test_fea[0, :])]
    #
    # if (len(mirror_v2_idx) > 0):
    #     test_fea[mirror_v2_idx, int(len(test_fea[0, :]) / 2):len(test_fea[0, :])] = test_fea[mirror_v1_idx,
    #                                                           0:int(len(test_fea[0, :]) / 2)]

    #score_v1_test

    return (test_pro, test_fea, score_v1_test[:, 0], score_v2_test[:, 0],
            test_score_hurdle[:, 0], logfc_v1_test[:, 0], logfc_v2_test[:, 0])



def MV_rescore(view1, view2, view1_pd, view2_pd, g1, g2, DEA_tool, norm, imp, logfc_thr, de_thr, combine_score_ty,
               hc_thr, npos_thr, trial_num, model_type, seed, diff_ty, design, R_fold, logT, cca):
    '''

    :param view1: raw view1 data
    :param view2: raw view2 data
    :param view1_pd: cleared view1 dataframe, first column is Protein, following are expression values
    :param view2_pd: cleared view2 dataframe, first column is Protein, following are expression values
    :param logfc_thr: logFC threshold, currently not used
    :param de_thr: q-value threshold for determining differentilly expressed or not
    :param combine_score_ty: the method used for combining confidence scores from two single views
    :param hc_thr: the threshold for selecting highly confident proteins, only used when we don't have enough lable consistent samples
    :param npos_thr: theshold for checking whether we have enough positive training data
    :param trial_num: number of trials for hyperparameter optimization
    :param model_type: model used for confidence score regression
    :param seed: fixed random seed
    :param diff_ty: the optimization objective loss type, pearson correlation is used as default
    :return: list of our predicted confidence scores
    '''

    seed_torch()
    view_scale1, res_real_norm1, view_scale2, res_real_norm2, res_real_hurdle = prepare_for_MVSVM(view1_pd, view2_pd,
                                                                                                  design, norm,
                                                                                                  imp, g1, g2, DEA_tool,
                                                                                                  R_fold, logT, cca)
    #proteins, logFC, p_value, qval, score, const
    save_folder = 'res/'
    isExist = os.path.exists(save_folder)

    if not isExist:
        os.makedirs(save_folder)

    view_pro = view1['Protein'].values
    fea_all = np.column_stack((view_scale1, view_scale2))
    fea_all[np.isnan(fea_all)] = 0

    proteins_v1 = res_real_norm1[0]
    logFC_v1 = res_real_norm1[1]
    qval_v1 = res_real_norm1[3]
    score_v1 = res_real_norm1[4]

    proteins_v2 = res_real_norm2[0]
    logFC_v2 = res_real_norm2[1]
    qval_v2 = res_real_norm2[3]
    score_v2 = res_real_norm2[4]

    hurdle_proteins = res_real_hurdle[0]
    hurdle_logFC = res_real_hurdle[1]
    hurdle_qval = res_real_hurdle[3]
    hurdle_score = res_real_hurdle[4]

    count_zeros = [len(np.where((fea_all[i, :] == 0))[0]) for i in range(len(fea_all[:, 0]))]
    idx_retain = np.where((np.array(count_zeros) <= len(fea_all[0, :]) * 0.8))[0]

    view_pro = view_pro[idx_retain]
    fea_all = fea_all[idx_retain, :]


    (train_pro, train_fea, train_scores, train_qval_v1, train_logfc_v1,
            train_qval_v2, train_logfc_v2) = get_training_proteins(proteins_v1, proteins_v2,
                                                                                             view_pro, qval_v1, qval_v2,
                                                                                             logFC_v1, logFC_v2, fea_all,
                                                                                             hc_thr, de_thr, npos_thr,
                                                                   combine_score_ty, hurdle_qval, hurdle_proteins)

    (test_pro, test_fea, score_v1_test, score_v2_test, test_score_hurdle, logfc_v1_test,
     logfc_v2_test) = get_test_proteins(proteins_v1, proteins_v2, view_pro, score_v1, score_v2, logFC_v1, logFC_v2,
                                        fea_all, hurdle_proteins, hurdle_score, train_pro)

    out_tranin = np.array([list(train_pro), list(train_scores)]).T
    pd.DataFrame(out_tranin, columns=['Protein', 'scores']).to_csv(
        save_folder + 'train_pros.csv', sep=',', index=False)

    if len(train_pro) > 0:

        pred_final, pred_train, best_params = optuna_optimize(train_fea, train_scores, seed, model_type,
                                                                    test_fea, trial_num, 1, de_thr, diff_ty)


        all_proteins = list(train_pro) + list(test_pro)
        # all_organism = list(train_orga) + list(test_orga)
        all_score_v1 = list(1-train_qval_v1[:, 0]) + list(score_v1_test)
        all_score_v2 = list(1-train_qval_v2[:, 0]) + list(score_v2_test)
        all_logFC_v1 = list(train_logfc_v1[:, 0]) + list(logfc_v1_test)
        all_logFC_v2 = list(train_logfc_v2[:, 0]) + list(logfc_v2_test)
        all_score_mv = list(train_scores) + list(pred_final)
        all_score_mv_pred_tr = list(pred_train) + list(pred_final)
        # all_label = list(train_true_label) + list(test_true_label)

        all_feas = np.row_stack((train_fea, test_fea))

        com_h_v, idxhl0, idxhl1 = np.intersect1d(np.array(all_proteins), hurdle_proteins, return_indices=True)
        all_score_hurdle = np.zeros((len(all_proteins), 1))
        all_score_hurdle[idxhl0, 0] = hurdle_score[idxhl1]

        out_combined_dea = [all_proteins, all_logFC_v1, all_logFC_v2, list(1-np.array(all_score_v1)),
                            list(1-np.array(all_score_v2)), list(1-np.array(all_score_mv)),
                            list(1-np.array(all_score_mv_pred_tr)),
                            list(1-np.array(all_score_hurdle[:, 0]))]
        pd.DataFrame(np.column_stack((np.array(out_combined_dea).T, all_feas)), columns=['Protein', 'logFC_view1',
                                                                                         'logFC_view2',
                                                                                        'qval_view1', 'qval_view2', 'qval_mv',
                                                                                         'qval_mv_pred_tr', 'qval_hurdle'] +
                                                                                        ['fea' + str(i) for i in
                                                                                         range(len(all_feas[0, :]))]).to_csv(
            save_folder + DEA_tool + '_' + combine_score_ty + '_' + g1 + '_' + g2 + '_' +
            'mv_outputs.csv', sep=',', index=False)


def get_training_proteins_repoduce(proteins_v1, proteins_v2, view_protein, qval_v1, qval_v2, logFC_v1, logFC_v2, fea_all,
                          logFC_all, view_organism, de_organism, hc_thr, de_thr, npos_thr, combine_score_ty, hurdle_qval,
                          hurdle_protein, poscca='F'):
    seed_torch(2023)

    label_v1 = label_pro(logFC_v1, qval_v1, np.log2(1.5), de_thr)
    label_v2 = label_pro(logFC_v2, qval_v2, np.log2(1.5), de_thr)

    com, idx1, idx2 = np.intersect1d(proteins_v1, proteins_v2, return_indices=True)

    com_pro = com
    com_pro_lab1 = label_v1[idx1, 0]
    com_pro_lab2 = label_v2[idx2, 0]

    com_pro_cons = com_pro[com_pro_lab1 == com_pro_lab2]

    hc_pro_v1 = proteins_v1[np.where((qval_v1 <= hc_thr))[0]] # highly confident DE proteins
    hc_pro_v2 = proteins_v2[np.where((qval_v2 <= hc_thr))[0]]

    train_protein = np.unique(np.array(list(com_pro_cons))) #+ list(hc_pro_v1) + list(hc_pro_v2)))

    train_qval_v1 = np.ones((len(train_protein), 1))
    train_logfc_v1 = np.zeros((len(train_protein), 1))
    com_v1, idx_v1, idx_tr1 = np.intersect1d(proteins_v1, train_protein, return_indices=True)
    train_qval_v1[idx_tr1, 0] = qval_v1[idx_v1]
    train_logfc_v1[idx_tr1, 0] = logFC_v1[idx_v1]

    train_qval_v2 = np.ones((len(train_protein), 1))
    train_logfc_v2 = np.zeros((len(train_protein), 1))
    com_v2, idx_v2, idx_tr2 = np.intersect1d(proteins_v2, train_protein, return_indices=True)
    train_qval_v2[idx_tr2, 0] = qval_v2[idx_v2]
    train_logfc_v2[idx_tr2, 0] = logFC_v2[idx_v2]

    if combine_score_ty == 'min':
        train_qval = np.array([min(train_qval_v1[i, 0], train_qval_v2[i, 0]) for i in range(len(train_qval_v1))])
    elif combine_score_ty == 'mean':
        train_qval = np.array([np.mean([train_qval_v1[i, 0], train_qval_v2[i, 0]]) for i in range(len(train_qval_v1))])
    elif combine_score_ty == 'max':
        train_qval = np.array([max(train_qval_v1[i, 0], train_qval_v2[i, 0]) for i in range(len(train_qval_v1))])
    elif combine_score_ty == 'hurdle':
        train_qval_hurdle = np.ones((len(train_protein), 1))
        com_v2, idx_v2, idx_tr2 = np.intersect1d(hurdle_protein, train_protein, return_indices=True)
        train_qval_hurdle[idx_tr2, 0] = hurdle_qval[idx_v2]
        train_qval = train_qval_hurdle[:, 0]

    if (len(np.where((np.array(train_qval) <= de_thr))[0]) < npos_thr) & (len(np.where((np.array(train_qval) <= de_thr))[0])>=0):
        train_protein = np.unique(np.array(list(com_pro_cons) + list(hc_pro_v1) + list(hc_pro_v2)))

        train_qval_v1 = np.ones((len(train_protein), 1))
        train_logfc_v1 = np.zeros((len(train_protein), 1))
        com_v1, idx_v1, idx_tr1 = np.intersect1d(proteins_v1, train_protein, return_indices=True)
        train_qval_v1[idx_tr1, 0] = qval_v1[idx_v1]
        train_logfc_v1[idx_tr1, 0] = logFC_v1[idx_v1]

        train_qval_v2 = np.ones((len(train_protein), 1))
        train_logfc_v2 = np.zeros((len(train_protein), 1))
        com_v2, idx_v2, idx_tr2 = np.intersect1d(proteins_v2, train_protein, return_indices=True)
        train_qval_v2[idx_tr2, 0] = qval_v2[idx_v2]
        train_logfc_v2[idx_tr2, 0] = logFC_v2[idx_v2]

        train_qval = np.array([min(train_qval_v1[i, 0], train_qval_v2[i, 0]) for i in range(len(train_qval_v1))])

        if len(np.where((np.array(train_qval) <= de_thr))[0]) < npos_thr:
            print('no enough positive training data!!!')

            train_pro = []
            train_fea = []
            train_logFC = []
            train_scores = []
            train_orga = []
            train_true_label = []
            train_qval_v1 = []
            train_logfc_v1 = []
            train_qval_v2 = []
            train_logfc_v2 = []
            cca1 = []
            cca2 = []

            return (train_pro, train_fea, train_scores, train_logFC, train_orga, train_true_label, train_qval_v1, train_logfc_v1,
            train_qval_v2, train_logfc_v2, cca1, cca2)

    train_score = 1-train_qval

    com_pro_train, idx11, idx12 = np.intersect1d(train_protein, view_protein, return_indices=True)

    train_pro = com_pro_train
    train_fea = fea_all[idx12, :]
    train_logFC = logFC_all[idx12, :]
    train_scores = train_score[idx11]
    train_orga = view_organism[idx12]
    train_true_label = get_true_label(train_pro, de_organism)

    train_qval_v1 = train_qval_v1[idx11]
    train_logfc_v1 = train_logfc_v1[idx11]

    train_qval_v2 = train_qval_v2[idx11]
    train_logfc_v2 = train_logfc_v2[idx11]

    idx_remain = np.where(((train_qval_v1 < 1) & (train_qval_v2 < 1)))[0]

    train_pro = train_pro[idx_remain]
    train_fea = train_fea[idx_remain, :]
    train_logFC = train_logFC[idx_remain, :]
    train_scores = train_scores[idx_remain]
    train_orga = train_orga[idx_remain]
    train_true_label = np.array(train_true_label)[idx_remain]
    train_qval_v1 = train_qval_v1[idx_remain]
    train_logfc_v1 = train_logfc_v1[idx_remain]
    train_qval_v2 = train_qval_v2[idx_remain]
    train_logfc_v2 = train_logfc_v2[idx_remain]

    if poscca == 'T':
        mses = []
        Ks = []
        for K in range(1, int(len(train_fea[0, :])/2)):
            logo = KFold(n_splits=3, shuffle=True)
            mse=0
            Ks.append(K)
            for train_index, test_index in logo.split(train_pro):
                train_x = train_fea[:, 0:int(len(train_fea[0, :])/2)][train_index, :]
                train_y = train_fea[:, int(len(train_fea[0, :])/2):len(train_fea[0, :])][train_index, :]

                test_x = train_fea[:, 0:int(len(train_fea[0, :]) / 2)][test_index, :]
                test_y = train_fea[:, int(len(train_fea[0, :]) / 2):len(train_fea[0, :])][test_index, :]

                cca1 = []
                cca1 = CCA(n_components=K)

                cca2 = []
                cca2 = CCA(n_components=K)

                cca1.fit(train_x, train_y)
                cca2.fit(train_y, train_x)

                mse += mean_squared_error(cca1.predict(test_x), test_y) + mean_squared_error(cca2.predict(test_y), test_x)
            mses.append(mse/3)
        Kbest = Ks[np.where((np.array(mses) == min(np.array(mses))))[0][0]]
        cca1 = CCA(n_components=Kbest).fit(train_fea[:, 0:int(len(train_fea[0, :]) / 2)],
                                              train_fea[:, int(len(train_fea[0, :]) / 2):len(train_fea[0, :])])
        cca2 = CCA(n_components=Kbest).fit(train_fea[:, int(len(train_fea[0, :]) / 2):len(train_fea[0, :])],
                                              train_fea[:, 0:int(len(train_fea[0, :]) / 2)])

    else:
        cca1 = []
        cca2 = []

    return (train_pro, train_fea, train_logFC, train_scores, train_orga, train_true_label, train_qval_v1, train_logfc_v1,
            train_qval_v2, train_logfc_v2, cca1, cca2)

def get_test_proteins_reproduce(proteins_v1, proteins_v2, view_protein, score_v1, score_v2, logFC_v1, logFC_v2, fea_all,
                          logFC_all, hurdle_proteins, hurdle_score, view_organism, de_organism, train_pro, cca1, cca2):

    test_protein = np.setdiff1d(view_protein, train_pro)

    com_pro_test, idx11, idx12 = np.intersect1d(test_protein, view_protein, return_indices=True)

    test_pro = com_pro_test
    test_fea = fea_all[idx12, :]
    test_logFC = logFC_all[idx12, :]
    test_orga = view_organism[idx12]
    test_true_label = get_true_label(test_pro, de_organism)

    score_v1_test = np.zeros((len(test_pro), 1))
    score_v2_test = np.zeros((len(test_pro), 1))
    logfc_v1_test = np.zeros((len(test_pro), 1))
    logfc_v2_test = np.zeros((len(test_pro), 1))

    com1, idx111, idx112 = np.intersect1d(test_pro, proteins_v1, return_indices=True)
    com2, idx211, idx212 = np.intersect1d(test_pro, proteins_v2, return_indices=True)

    score_v1_test[idx111, 0] = score_v1[idx112]
    score_v2_test[idx211, 0] = score_v2[idx212]
    logfc_v1_test[idx111, 0] = logFC_v1[idx112]
    logfc_v2_test[idx211, 0] = logFC_v2[idx212]

    test_score_hurdle = np.zeros((len(test_pro), 1))
    com_h, idx_h, idx_l = np.intersect1d(hurdle_proteins, test_pro, return_indices=True)
    idx_h_l = np.array([list(hurdle_proteins).index(com_h[i]) for i in range(len(com_h))], dtype='int')
    test_score_hurdle[idx_l, 0] = hurdle_score[idx_h_l]

    if (cca1 != []) & (cca2 != []):
        ## simply mirror feature from view A to view B if view B are largely 0

        all_missing_v1_idx = np.where(((score_v1_test[:, 0] == 0) & (logfc_v1_test[:, 0] == 0)))[0]
        all_missing_v2_idx = np.where(((score_v2_test[:, 0] == 0) & (logfc_v2_test[:, 0] == 0)))[0]

        #all_missing = np.union1d(all_missing_v2_idx, all_missing_v2_idx)
        #all_missing=[]
        mirror_v1_idx = np.setdiff1d(all_missing_v1_idx, all_missing_v2_idx)
        mirror_v2_idx = np.setdiff1d(all_missing_v2_idx, all_missing_v1_idx)

        if (len(mirror_v1_idx)>0):
            test_fea[mirror_v1_idx, 0:int(len(test_fea[0, :])/2)] = cca2.predict(test_fea[mirror_v1_idx, int(len(test_fea[0, :])/2):len(test_fea[0, :])])

        if (len(mirror_v2_idx) > 0):
            test_fea[mirror_v2_idx, int(len(test_fea[0, :]) / 2):len(test_fea[0, :])] = cca1.predict(test_fea[mirror_v2_idx,
                                                                  0:int(len(test_fea[0, :]) / 2)])

    return (test_pro, test_fea, test_logFC, test_orga, np.array(test_true_label), score_v1_test[:, 0], score_v2_test[:, 0],
            test_score_hurdle[:, 0], logfc_v1_test[:, 0], logfc_v2_test[:, 0])

    # return (
    # test_pro[~all_missing], test_fea[~all_missing, :], test_orga[~all_missing], np.array(test_true_label)[~all_missing],
    # score_v1_test[~all_missing, 0], score_v2_test[~all_missing, 0],
    # test_score_hurdle[~all_missing, 0], logfc_v1_test[~all_missing, 0], logfc_v2_test[~all_missing, 0])

def mapping_scores(all_protein, train_protein, test_protein, train_score, test_score):

    mapped_scores = np.zeros((len(all_protein), 1))

    if len(train_protein) > 0:
        comm1, idx11, idx12 = np.intersect1d(all_protein, train_protein, return_indices=True)
        if len(comm1) > 0:
            mapped_scores[idx11, 0] = train_score[idx12]

    if len(test_protein) > 0:
        comm2, idx21, idx22 = np.intersect1d(all_protein, test_protein, return_indices=True)
        if len(comm2) > 0:
            mapped_scores[idx21, 0] = test_score[idx22]

    return mapped_scores[:, 0]

def MV_DEA_all(view1, view2, v1_scale, v2_scale, v1_scale_cca, v2_scale_cca, res_view1,
                                   res_view2, res_hurdle, res_view1_cca, res_view2_cca,
                                   res_hurdle_cca, lgfc_thr, de_thr, cbt, view_paths, v1_ty,
                                   v2_ty, hc_thr, npos_thr, num_tra, model, seed, op_ty, save_folder, logFC_norm, logFC_norm_cca):

    seed_torch()

    isExist = os.path.exists(save_folder)

    if not isExist:
        os.makedirs(save_folder)

    de_organism = view_paths['DE_organism']
    dataset = view_paths['dataset']
    view_pro = view1['Protein'].values
    organism_pro = view1['Organism'].values
    fea_all = np.column_stack((v1_scale, v2_scale))
    fea_all[np.isnan(fea_all)] = 0

    fea_all_cca = np.column_stack((v1_scale_cca, v2_scale_cca))
    fea_all_cca[np.isnan(fea_all_cca)] = 0

    proteins_v1 = res_view1[1]
    logFC_v1 = res_view1[6]
    qval_v1 = res_view1[9]
    score_v1 = res_view1[2]
    T_label_v1 = res_view1[4]

    proteins_v2 = res_view2[1]
    logFC_v2 = res_view2[6]
    qval_v2 = res_view2[9]
    score_v2 = res_view2[2]
    T_label_v2 = res_view2[4]

    hurdle_proteins = res_hurdle[1]
    hurdle_logFC = res_hurdle[6]
    hurdle_qval = res_hurdle[9]
    hurdle_score = res_hurdle[2]
    hurdle_T_label = res_hurdle[4]

    proteins_cca_v1 = res_view1_cca[1]
    logFC_cca_v1 = res_view1_cca[6]
    qval_cca_v1 = res_view1_cca[9]
    score_cca_v1 = res_view1_cca[2]
    T_label_cca_v1 = res_view1_cca[4]

    proteins_cca_v2 = res_view2_cca[1]
    logFC_cca_v2 = res_view2_cca[6]
    qval_cca_v2 = res_view2_cca[9]
    score_cca_v2 = res_view2_cca[2]
    T_label_cca_v2 = res_view2_cca[4]

    hurdle_cca_proteins = res_hurdle_cca[1]
    hurdle_cca_logFC = res_hurdle_cca[6]
    hurdle_cca_qval = res_hurdle_cca[9]
    hurdle_cca_score = res_hurdle_cca[2]
    hurdle_T_label_cca = res_hurdle_cca[4]

    # add logFC as feature
    # all_logfc_v1 = np.zeros((len(view_pro), 1))
    # all_logfc_v2 = np.zeros((len(view_pro), 1))
    #
    # com_pv1, idall1, idv1 = np.intersect1d(view_pro, proteins_v1, return_indices=True)
    # all_logfc_v1[idall1, 0] = logFC_v1[idv1]
    #
    # com_pv2, idall2, idv2 = np.intersect1d(view_pro, proteins_v2, return_indices=True)
    # all_logfc_v2[idall2, 0] = logFC_v2[idv2]
    #
    # norm_logfcs = preprocessing.MinMaxScaler(feature_range=(0, 1)).fit_transform(np.column_stack((all_logfc_v1, all_logfc_v2)))
    # fea_all = np.column_stack((fea_all, norm_logfcs))


    count_zeros = [len(np.where((fea_all[i, :] == 0))[0]) for i in range(len(fea_all[:, 0]))]
    idx_retain = np.where((np.array(count_zeros) <= len(fea_all[0, :]) * 0.8))[0]

    view_pro = view_pro[idx_retain]
    organism_pro = organism_pro[idx_retain]
    fea_all = fea_all[idx_retain, :]
    logFC_all = logFC_norm[idx_retain, :]

    count_zeros_cca = [len(np.where((fea_all_cca[i, :] == 0))[0]) for i in range(len(fea_all_cca[:, 0]))]
    idx_retain_cca = np.where((np.array(count_zeros_cca) <= len(fea_all_cca[0, :]) * 0.8))[0]

    view_pro_cca = view1['Protein'].values[idx_retain_cca]
    organism_pro_cca = view1['Organism'].values[idx_retain_cca]
    fea_all_cca = fea_all_cca[idx_retain_cca, :]
    logFC_all_cca = logFC_norm_cca[idx_retain_cca, :]

    # with post cca
    # (train_pro, train_fea, train_scores, train_orga, train_true_label, train_qval_v1, train_logfc_v1,
    #  train_qval_v2, train_logfc_v2, cca1, cca2)
    (train_pro, train_fea, train_FC, train_scores, train_orga, train_true_label, train_qval_v1, train_logfc_v1,
     train_qval_v2, train_logfc_v2, cca1_pos, cca2_pos) = get_training_proteins_repoduce(proteins_v1, proteins_v2,
                                                                        view_pro, qval_v1, qval_v2,
                                                                        logFC_v1, logFC_v2, fea_all, logFC_all,
                                                                        organism_pro, de_organism,
                                                                        hc_thr, de_thr, npos_thr,
                                                                        cbt, hurdle_qval, hurdle_proteins, poscca='T')

    # pre cca
    (train_pro_cca, train_fea_cca, train_FC_cca, train_scores_cca, train_orga_cca, train_true_label_cca, train_qval_v1_cca,
     train_logfc_v1_cca,
     train_qval_v2_cca, train_logfc_v2_cca, cca1_pre, cca2_pre) = get_training_proteins_repoduce(proteins_cca_v1, proteins_cca_v2,
                                                            view_pro_cca, qval_cca_v1, qval_cca_v2,
                                                            logFC_cca_v1, logFC_cca_v2, fea_all_cca, logFC_all_cca,
                                                            organism_pro_cca, de_organism,
                                                            hc_thr, de_thr, npos_thr,
                                                            cbt, hurdle_cca_qval, hurdle_cca_proteins)


    # no cca
    (test_pro, test_fea, test_FC, test_orga, test_true_label, score_v1_test, score_v2_test, test_score_hurdle, logfc_v1_test,
     logfc_v2_test) = get_test_proteins_reproduce(proteins_v1, proteins_v2, view_pro, score_v1, score_v2, logFC_v1, logFC_v2,
                                        fea_all, logFC_all, hurdle_proteins, hurdle_score, organism_pro, de_organism, train_pro,
                                        [], [])

    # post cca
    (test_pro_poscca, test_fea_poscca, test_FC_poscca, test_orga_poscca, test_true_label_poscca, score_v1_test_poscca, score_v2_test_poscca,
     test_score_hurdle_poscca, logfc_v1_test_poscca,
     logfc_v2_test_poscca) = get_test_proteins_reproduce(proteins_v1, proteins_v2, view_pro, score_v1, score_v2, logFC_v1, logFC_v2,
                                        fea_all, logFC_all, hurdle_proteins, hurdle_score, organism_pro, de_organism, train_pro,
                                            cca1_pos, cca2_pos)

    #ppre cca
    (test_pro_cca, test_fea_cca, test_FC_cca, test_orga_cca, test_true_label_cca, score_v1_test_cca, score_v2_test_cca,
     test_score_hurdle_cca, logfc_v1_test_cca,
     logfc_v2_test_cca) = get_test_proteins_reproduce(proteins_cca_v1, proteins_cca_v2, view_pro_cca, score_cca_v1, score_cca_v2,
                                            logFC_cca_v1, logFC_cca_v2,
                                        fea_all_cca, logFC_all_cca, hurdle_cca_proteins, hurdle_cca_score, organism_pro_cca,
                                            de_organism, train_pro_cca, cca1_pre, cca2_pre)



    if len(train_pro) > 0:
        out_tranin = np.array(
            [list(train_pro), list(train_scores), list(train_FC[:, 0]), list(train_orga), list(train_true_label)]).T
        pd.DataFrame(out_tranin, columns=['Protein', 'organism', 'scores', 'logFC', 'true_label']).to_csv(
            save_folder + dataset + '_' + v1_ty + '_' + v2_ty + '_train_pros.csv', sep=',', index=False)

        out_tranin_cca = np.array(
            [list(train_pro_cca), list(train_scores_cca), list(train_FC_cca[:, 0]), list(train_orga_cca),
             list(train_true_label_cca)]).T
        pd.DataFrame(out_tranin_cca, columns=['Protein', 'organism', 'scores', 'logFC', 'true_label']).to_csv(
            save_folder + dataset + '_' + v1_ty + '_' + v2_ty + '_train_pros_precca.csv', sep=',', index=False)

        #train_feas, train_scores, seed, model_type, test_feas_final, trial_num, a, de_thr, diff_ty
        pred_final, pred_train, best_params = optuna_optimize(train_fea, train_scores, train_FC[:, 0], seed, model,
                                                                    test_fea, num_tra, 1, de_thr, op_ty, lgfc_thr)

        pred_final_poscca, pred_train_poscca, best_params_poscca = optuna_optimize(train_fea, train_scores, train_FC[:, 0], seed, model,
                                                              test_fea_poscca, num_tra, 1, de_thr, op_ty, lgfc_thr)

        pred_final_precca, pred_train_precca, best_params_precca = optuna_optimize(train_fea_cca, train_scores_cca, train_FC_cca[:, 0], seed, model,
                                                              test_fea_cca, num_tra, 1, de_thr, op_ty, lgfc_thr)


        all_proteins = np.array(list(train_pro) + list(test_pro))
        all_true_label = np.array(list(train_true_label) + list(test_true_label))
        all_orga = np.array(list(train_orga) + list(test_orga))
        all_logFC = np.array(list(train_FC) + list(test_FC))
        #all_proteins_poscca = list(train_pro) + list(test_pro_poscca)
        all_proteins_precca = np.array(list(train_pro_cca) + list(test_pro_cca))
        all_true_label_cca = np.array(list(train_true_label_cca) + list(test_true_label_cca))
        all_orga_cca = np.array(list(train_orga_cca) + list(test_orga_cca))
        all_logFC_cca = np.array(list(train_FC_cca) + list(test_FC_cca))

        all_test_proteins, idx = np.unique(np.array(list(all_proteins) + list(all_proteins_precca)), return_index=True)

        all_test_orga = np.array(list(all_orga) + list(all_orga_cca))[idx]
        all_true_label = mapping_scores(all_test_proteins, all_proteins, all_proteins_precca, all_true_label, all_true_label_cca)
        all_test_score_v1 = mapping_scores(all_test_proteins, proteins_v1, [], score_v1, [])
        all_test_score_v2 = mapping_scores(all_test_proteins, proteins_v2, [], score_v2, [])
        all_test_score_mv = mapping_scores(all_test_proteins, train_pro, test_pro, train_scores, pred_final)
        all_test_score_mv_tr = mapping_scores(all_test_proteins, train_pro, test_pro, pred_train, pred_final)
        all_test_score_hurdle = mapping_scores(all_test_proteins, hurdle_proteins, [], hurdle_score, [])
        all_test_score_mv_poscca = mapping_scores(all_test_proteins, train_pro, test_pro_poscca, train_scores, pred_final_poscca)
        all_test_score_mv_tr_poscca = mapping_scores(all_test_proteins, train_pro, test_pro_poscca, pred_train_poscca, pred_final_poscca)

        all_test_score_v1_precca = mapping_scores(all_test_proteins, proteins_cca_v1, [], score_cca_v1, [])
        all_test_score_v2_precca = mapping_scores(all_test_proteins, proteins_cca_v2, [], score_cca_v2, [])
        all_test_score_mv_precca = mapping_scores(all_test_proteins, train_pro_cca, test_pro_cca, train_scores_cca,
                                                     pred_final_precca)
        all_test_score_mv_tr_precca = mapping_scores(all_test_proteins, train_pro_cca, test_pro_cca, pred_train_precca,
                                                  pred_final_precca)
        all_test_score_hurdle_precca = mapping_scores(all_test_proteins, hurdle_cca_proteins, [], hurdle_cca_score, [])

        all_logFC_v1 = mapping_scores(all_test_proteins, proteins_v1, [], logFC_v1, [])
        all_logFC_v2 = mapping_scores(all_test_proteins, proteins_v2, [], logFC_v2, [])
        all_logFC_hurdle = mapping_scores(all_test_proteins, hurdle_proteins, [], hurdle_logFC, [])
        all_logFC_v1_cca = mapping_scores(all_test_proteins, proteins_cca_v1, [], logFC_cca_v1, [])
        all_logFC_v2_cca = mapping_scores(all_test_proteins, proteins_cca_v2, [], logFC_cca_v2, [])
        all_logFC_hurdle_cca = mapping_scores(all_test_proteins, hurdle_cca_proteins, [], hurdle_cca_logFC, [])
        all_test_logFC_mv = mapping_scores(all_test_proteins, train_pro, test_pro, train_FC[:, 0], test_FC[:, 0])
        all_test_logFC_mv_tr = mapping_scores(all_test_proteins, train_pro, test_pro, train_FC[:, 0], test_FC[:, 0])
        all_test_logFC_mv_poscca = mapping_scores(all_test_proteins, train_pro, test_pro_poscca, train_FC[:, 0], test_FC_poscca[:, 0])
        all_test_logFC_mv_tr_poscca = mapping_scores(all_test_proteins, train_pro, test_pro_poscca, train_FC[:, 0], test_FC_poscca[:, 0])
        all_test_logFC_mv_precca = mapping_scores(all_test_proteins, train_pro_cca, test_pro_cca, train_FC_cca[:, 0], test_FC_cca[:, 0])
        all_test_logFC_mv_tr_precca = mapping_scores(all_test_proteins, train_pro_cca, test_pro_cca, train_FC_cca[:, 0], test_FC_cca[:, 0])


        com, idx_a, idx_c = np.intersect1d(all_test_proteins, view_pro_cca, return_indices=True)
        all_feas = np.zeros((len(all_test_proteins), len(train_fea[0, :])))
        all_feas[idx_a, :] = fea_all_cca[idx_c, :]

        out_combined_dea = [all_test_proteins, all_test_orga, all_true_label, all_logFC_v1,
                            list(1 - np.array(all_test_score_v1)),
                            all_logFC_v2, list(1 - np.array(all_test_score_v2)),
                            all_logFC_hurdle, list(1 - np.array(all_test_score_hurdle)),
                            all_logFC_v1_cca, list(1 - np.array(all_test_score_v1_precca)),
                            all_logFC_v2_cca, list(1 - np.array(all_test_score_v2_precca)),
                            all_logFC_hurdle_cca, list(1 - np.array(all_test_score_hurdle_precca)),
                            all_test_logFC_mv, list(1 - np.array(all_test_score_mv)),
                            all_test_logFC_mv_tr, list(1 - np.array(all_test_score_mv_tr)),
                            all_test_logFC_mv_poscca, list(1 - np.array(all_test_score_mv_poscca)),
                            all_test_logFC_mv_tr_poscca, list(1 - np.array(all_test_score_mv_tr_poscca)),
                            all_test_logFC_mv_precca, list(1 - np.array(all_test_score_mv_precca)),
                            all_test_logFC_mv_tr_precca, list(1 - np.array(all_test_score_mv_tr_precca))]

        pd.DataFrame(np.column_stack((np.array(out_combined_dea).T, all_feas)),
                     columns=['Protein', 'Organism', 'DE',
                              'logFC_' + v1_ty, 'qval_' + v1_ty,
                              'logFC_' + v2_ty, 'qval_' + v2_ty,
                              'logFC_hurdle', 'qval_hurdle',
                              'logFC_cca_' + v1_ty, 'qval_cca_' + v1_ty,
                              'logFC_cca_' + v2_ty, 'qval_cca_' + v2_ty,
                              'logFC_hurdle_precca', 'qval_hurdle_precca',
                              'logFC_mv', 'qval_mv',
                              'logFC_mv_pred_tr', 'qval_mv_pred_tr',
                              'logFC_mv_poscca', 'qval_mv_poscca',
                              'logFC_mv_tr_poscca', 'qval_mv_tr_poscca',
                              'logFC_mv_precca', 'qval_mv_precca',
                              'logFC_mv_pred_tr_precca', 'qval_mv_pred_tr_precca'] +
                             ['fea' + str(i) for i in
                              range(len(all_feas[0, :]))]).to_csv(
            save_folder + dataset + '_' + v1_ty + '_' + v2_ty + '_mv_outputs.csv', sep=',', index=False)


        metric_all_v1 = cal_metrics1(all_test_score_v1, de_thr, all_logFC_v1, lgfc_thr, all_true_label)
        metric_all_v2 = cal_metrics1(all_test_score_v2, de_thr, all_logFC_v2, lgfc_thr, all_true_label)
        metric_all_mv = cal_metrics1(all_test_score_mv, de_thr, all_test_logFC_mv, lgfc_thr, all_true_label)
        metric_all_mv_pred_tr = cal_metrics1(all_test_score_mv_tr, de_thr, all_test_logFC_mv_tr, lgfc_thr, all_true_label)
        metric_all_mv_poscca = cal_metrics1(all_test_score_mv_poscca, de_thr, all_test_logFC_mv_poscca, lgfc_thr, all_true_label)
        metric_all_mv_pred_tr_poscca = cal_metrics1(all_test_score_mv_tr_poscca, de_thr, all_test_logFC_mv_tr_poscca, lgfc_thr, all_true_label)
        metric_all_hurdle = cal_metrics1(all_test_score_hurdle, de_thr, all_logFC_hurdle, lgfc_thr, all_true_label)

        metric_all_v1_precca = cal_metrics1(all_test_score_v1_precca, de_thr, all_logFC_v1_cca, lgfc_thr, all_true_label)
        metric_all_v2_precca = cal_metrics1(all_test_score_v2_precca, de_thr, all_logFC_v2_cca, lgfc_thr, all_true_label)
        metric_all_mv_precca = cal_metrics1(all_test_score_mv_precca, de_thr, all_test_logFC_mv_precca, lgfc_thr, all_true_label)
        metric_all_mv_pred_tr_precca = cal_metrics1(all_test_score_mv_tr_precca, de_thr, all_test_logFC_mv_tr_precca, lgfc_thr, all_true_label)
        metric_all_hurdle_precca = cal_metrics1(all_test_score_hurdle_precca, de_thr, all_logFC_hurdle_cca, lgfc_thr, all_true_label)

        ## v1 list
        v1_test_score_mv = mapping_scores(proteins_v1, train_pro, test_pro, train_scores, pred_final)
        v1_test_score_hurdle = mapping_scores(proteins_v1, hurdle_proteins, [], hurdle_score, [])
        v1_test_score_mv_poscca = mapping_scores(proteins_v1, train_pro, test_pro_poscca, train_scores,
                                                  pred_final_poscca)
        v1_test_score_hurdle_precca = mapping_scores(proteins_v1, hurdle_cca_proteins, [], hurdle_cca_score, [])
        v1_test_score_mv_precca = mapping_scores(proteins_v1, train_pro_cca, test_pro_cca, train_scores_cca,
                                                  pred_final_precca)

        v1_test_logFC_mv = mapping_scores(proteins_v1, train_pro, test_pro, train_FC[:, 0], test_FC[:, 0])
        v1_test_logFC_hurdle = mapping_scores(proteins_v1, hurdle_proteins, [], hurdle_logFC, [])
        v1_test_logFC_mv_poscca = mapping_scores(proteins_v1, train_pro, test_pro_poscca, train_FC[:, 0],
                                                 test_FC_poscca[:, 0])
        v1_test_logFC_hurdle_precca = mapping_scores(proteins_v1, hurdle_cca_proteins, [], hurdle_cca_logFC, [])
        v1_test_logFC_mv_precca = mapping_scores(proteins_v1, train_pro_cca, test_pro_cca, train_FC_cca[:, 0],
                                                 test_FC_cca[:, 0])

        ori_metric_v1 = cal_metrics1(score_v1, de_thr, logFC_v1, lgfc_thr, T_label_v1)
        metric_v1_mv = cal_metrics1(v1_test_score_mv, de_thr, v1_test_logFC_mv, lgfc_thr, T_label_v1)
        metric_v1_mv_poscca = cal_metrics1(v1_test_score_mv_poscca, de_thr, v1_test_logFC_mv_poscca, lgfc_thr, T_label_v1)
        metric_v1_hurdle = cal_metrics1(v1_test_score_hurdle, de_thr, v1_test_logFC_hurdle, lgfc_thr, T_label_v1)
        metric_v1_mv_precca = cal_metrics1(v1_test_score_mv_precca, de_thr, v1_test_logFC_mv_precca, lgfc_thr, T_label_v1)
        metric_v1_hurdle_precca = cal_metrics1(v1_test_score_hurdle_precca, de_thr, v1_test_logFC_hurdle_precca, lgfc_thr, T_label_v1)

        ## v2 list
        v2_test_score_mv = mapping_scores(proteins_v2, train_pro, test_pro, train_scores, pred_final)
        v2_test_score_hurdle = mapping_scores(proteins_v2, hurdle_proteins, [], hurdle_score, [])
        v2_test_score_mv_poscca = mapping_scores(proteins_v2, train_pro, test_pro_poscca, train_scores,
                                                 pred_final_poscca)
        v2_test_score_hurdle_precca = mapping_scores(proteins_v2, hurdle_cca_proteins, [], hurdle_cca_score, [])
        v2_test_score_mv_precca = mapping_scores(proteins_v2, train_pro_cca, test_pro_cca, train_scores_cca,
                                                 pred_final_precca)

        v2_test_logFC_mv = mapping_scores(proteins_v2, train_pro, test_pro, train_FC[:, 0], test_FC[:, 0])
        v2_test_logFC_hurdle = mapping_scores(proteins_v2, hurdle_proteins, [], hurdle_logFC, [])
        v2_test_logFC_mv_poscca = mapping_scores(proteins_v2, train_pro, test_pro_poscca, train_FC[:, 0],
                                                 test_FC_poscca[:, 0])
        v2_test_logFC_hurdle_precca = mapping_scores(proteins_v2, hurdle_cca_proteins, [], hurdle_cca_logFC, [])
        v2_test_logFC_mv_precca = mapping_scores(proteins_v2, train_pro_cca, test_pro_cca, train_FC_cca[:, 0],
                                                 test_FC_cca[:, 0])

        ori_metric_v2 = cal_metrics1(score_v2, de_thr, logFC_v2, lgfc_thr, T_label_v2)
        metric_v2_mv = cal_metrics1(v2_test_score_mv, de_thr, v2_test_logFC_mv, lgfc_thr, T_label_v2)
        metric_v2_mv_poscca = cal_metrics1(v2_test_score_mv_poscca, de_thr, v2_test_logFC_mv_poscca, lgfc_thr, T_label_v2)
        metric_v2_hurdle = cal_metrics1(v2_test_score_hurdle, de_thr, v2_test_logFC_hurdle, lgfc_thr, T_label_v2)
        metric_v2_mv_precca = cal_metrics1(v2_test_score_mv_precca, de_thr, v2_test_logFC_mv_precca, lgfc_thr, T_label_v2)
        metric_v2_hurdle_precca = cal_metrics1(v2_test_score_hurdle_precca, de_thr, v2_test_logFC_hurdle_precca, lgfc_thr, T_label_v2)

        ## hurdle list
        hurdle_test_score_mv = mapping_scores(hurdle_proteins, train_pro, test_pro, train_scores, pred_final)
        hurdle_test_score_mv_poscca = mapping_scores(hurdle_proteins, train_pro, test_pro_poscca, train_scores,
                                                 pred_final_poscca)
        hurdle_test_score_hurdle_precca = mapping_scores(hurdle_proteins, hurdle_cca_proteins, [], hurdle_cca_score, [])
        hurdle_test_score_mv_precca = mapping_scores(hurdle_proteins, train_pro_cca, test_pro_cca, train_scores_cca,
                                                 pred_final_precca)

        hurdle_test_logFC_mv = mapping_scores(hurdle_proteins, train_pro, test_pro, train_FC[:, 0], test_FC[:, 0])
        hurdle_test_logFC_mv_poscca = mapping_scores(hurdle_proteins, train_pro, test_pro_poscca, train_FC[:, 0],
                                                     test_FC_poscca[:, 0])
        hurdle_test_logFC_hurdle_precca = mapping_scores(hurdle_proteins, hurdle_cca_proteins, [], hurdle_cca_logFC, [])
        hurdle_test_logFC_mv_precca = mapping_scores(hurdle_proteins, train_pro_cca, test_pro_cca, train_FC_cca[:, 0],
                                                     test_FC_cca[:, 0])

        ori_metric_hurdle = cal_metrics1(hurdle_score, de_thr, hurdle_logFC, lgfc_thr, hurdle_T_label)
        metric_hurdle_mv = cal_metrics1(hurdle_test_score_mv, de_thr, hurdle_test_logFC_mv, lgfc_thr, hurdle_T_label)
        metric_hurdle_mv_poscca = cal_metrics1(hurdle_test_score_mv_poscca, de_thr, hurdle_test_logFC_mv_poscca, lgfc_thr, hurdle_T_label)
        metric_hurdle_mv_precca = cal_metrics1(hurdle_test_score_mv_precca, de_thr, hurdle_test_logFC_mv_precca, lgfc_thr, hurdle_T_label)
        metric_hurdle_hurdle_precca = cal_metrics1(hurdle_test_score_hurdle_precca, de_thr, hurdle_test_logFC_hurdle_precca, lgfc_thr, hurdle_T_label)


        ## v1 list precca
        v1_cca_test_score_mv = mapping_scores(proteins_cca_v1, train_pro, test_pro, train_scores, pred_final)
        v1_cca_test_score_hurdle = mapping_scores(proteins_cca_v1, hurdle_proteins, [], hurdle_score, [])
        v1_cca_test_score_mv_poscca = mapping_scores(proteins_cca_v1, train_pro, test_pro_poscca, train_scores,
                                                 pred_final_poscca)
        v1_cca_test_score_hurdle_precca = mapping_scores(proteins_cca_v1, hurdle_cca_proteins, [], hurdle_cca_score, [])
        v1_cca_test_score_mv_precca = mapping_scores(proteins_cca_v1, train_pro_cca, test_pro_cca, train_scores_cca,
                                                 pred_final_precca)

        v1_cca_test_logFC_mv = mapping_scores(proteins_cca_v1, train_pro, test_pro, train_FC[:, 0], test_FC[:, 0])
        v1_cca_test_logFC_hurdle = mapping_scores(proteins_cca_v1, hurdle_proteins, [], hurdle_logFC, [])
        v1_cca_test_logFC_mv_poscca = mapping_scores(proteins_cca_v1, train_pro, test_pro_poscca, train_FC[:, 0],
                                                     test_FC_poscca[:, 0])
        v1_cca_test_logFC_hurdle_precca = mapping_scores(proteins_cca_v1, hurdle_cca_proteins, [], hurdle_cca_logFC, [])
        v1_cca_test_logFC_mv_precca = mapping_scores(proteins_cca_v1, train_pro_cca, test_pro_cca, train_FC_cca[:, 0],
                                                     test_FC_cca[:, 0])

        ori_metric_v1_cca = cal_metrics1(score_cca_v1, de_thr, logFC_cca_v1, lgfc_thr, T_label_cca_v1)
        metric_v1_cca_mv = cal_metrics1(v1_cca_test_score_mv, de_thr, v1_cca_test_logFC_mv, lgfc_thr, T_label_cca_v1)
        metric_v1_cca_mv_poscca = cal_metrics1(v1_cca_test_score_mv_poscca, de_thr, v1_cca_test_logFC_mv_poscca, lgfc_thr, T_label_cca_v1)
        metric_v1_cca_hurdle = cal_metrics1(v1_cca_test_score_hurdle, de_thr, v1_cca_test_logFC_hurdle, lgfc_thr, T_label_cca_v1)
        metric_v1_cca_mv_precca = cal_metrics1(v1_cca_test_score_mv_precca, de_thr, v1_cca_test_logFC_mv_precca, lgfc_thr, T_label_cca_v1)
        metric_v1_cca_hurdle_precca = cal_metrics1(v1_cca_test_score_hurdle_precca, de_thr, v1_cca_test_logFC_hurdle_precca, lgfc_thr, T_label_cca_v1)

        ## v2 list precca
        v2_cca_test_score_mv = mapping_scores(proteins_cca_v2, train_pro, test_pro, train_scores, pred_final)
        v2_cca_test_score_hurdle = mapping_scores(proteins_cca_v2, hurdle_proteins, [], hurdle_score, [])
        v2_cca_test_score_mv_poscca = mapping_scores(proteins_cca_v2, train_pro, test_pro_poscca, train_scores,
                                                 pred_final_poscca)
        v2_cca_test_score_hurdle_precca = mapping_scores(proteins_cca_v2, hurdle_cca_proteins, [], hurdle_cca_score, [])
        v2_cca_test_score_mv_precca = mapping_scores(proteins_cca_v2, train_pro_cca, test_pro_cca, train_scores_cca,
                                                 pred_final_precca)

        v2_cca_test_logFC_mv = mapping_scores(proteins_cca_v2, train_pro, test_pro, train_FC[:, 0], test_FC[:, 0])
        v2_cca_test_logFC_hurdle = mapping_scores(proteins_cca_v2, hurdle_proteins, [], hurdle_logFC, [])
        v2_cca_test_logFC_mv_poscca = mapping_scores(proteins_cca_v2, train_pro, test_pro_poscca, train_FC[:, 0],
                                                     test_FC_poscca[:, 0])
        v2_cca_test_logFC_hurdle_precca = mapping_scores(proteins_cca_v2, hurdle_cca_proteins, [], hurdle_cca_logFC, [])
        v2_cca_test_logFC_mv_precca = mapping_scores(proteins_cca_v2, train_pro_cca, test_pro_cca, train_FC_cca[:, 0],
                                                     test_FC_cca[:, 0])

        ori_metric_v2_cca = cal_metrics1(score_cca_v2, de_thr, logFC_cca_v2, lgfc_thr, T_label_cca_v2)
        metric_v2_cca_mv = cal_metrics1(v2_cca_test_score_mv, de_thr, v2_cca_test_logFC_mv, lgfc_thr, T_label_cca_v2)
        metric_v2_cca_mv_poscca = cal_metrics1(v2_cca_test_score_mv_poscca, de_thr, v2_cca_test_logFC_mv_poscca, lgfc_thr, T_label_cca_v2)
        metric_v2_cca_hurdle = cal_metrics1(v2_cca_test_score_hurdle, de_thr, v2_cca_test_logFC_hurdle, lgfc_thr, T_label_cca_v2)
        metric_v2_cca_mv_precca = cal_metrics1(v2_cca_test_score_mv_precca, de_thr, v2_cca_test_logFC_mv_precca, lgfc_thr, T_label_cca_v2)
        metric_v2_cca_hurdle_precca = cal_metrics1(v2_cca_test_score_hurdle_precca, de_thr, v2_cca_test_logFC_hurdle_precca, lgfc_thr, T_label_cca_v2)

        ## hurdle list precca
        hurdle_cca_test_score_mv = mapping_scores(hurdle_cca_proteins, train_pro, test_pro, train_scores, pred_final)
        hurdle_cca_test_score_hurdle = mapping_scores(hurdle_cca_proteins, hurdle_proteins, [], hurdle_score, [])
        hurdle_cca_test_score_mv_poscca = mapping_scores(hurdle_cca_proteins, train_pro, test_pro_poscca, train_scores,
                                                     pred_final_poscca)
        #hurdle_cca_test_score_mv_precca = mapping_scores(hurdle_cca_proteins, hurdle_cca_proteins, [], hurdle_cca_score, [])
        hurdle_cca_test_score_mv_precca = mapping_scores(hurdle_cca_proteins, train_pro_cca, test_pro_cca, train_scores_cca,
                                                     pred_final_precca)

        hurdle_cca_test_logFC_mv = mapping_scores(hurdle_cca_proteins, train_pro, test_pro, train_FC[:, 0], test_FC[:, 0])
        hurdle_cca_test_logFC_hurdle = mapping_scores(hurdle_cca_proteins, hurdle_proteins, [], hurdle_logFC, [])
        hurdle_cca_test_logFC_mv_poscca = mapping_scores(hurdle_cca_proteins, train_pro, test_pro_poscca, train_FC[:, 0],
                                                     test_FC_poscca[:, 0])
        #hurdle_cca_test_logFC_hurdle_precca = mapping_scores(hurdle_cca_proteins, hurdle_cca_proteins, [], hurdle_cca_logFC, [])
        hurdle_cca_test_logFC_mv_precca = mapping_scores(hurdle_cca_proteins, train_pro_cca, test_pro_cca, train_FC_cca[:, 0],
                                                     test_FC_cca[:, 0])

        ori_metric_hurdle_cca = cal_metrics1(hurdle_cca_score, de_thr, hurdle_cca_logFC, lgfc_thr, hurdle_T_label_cca)
        metric_hurdle_cca_mv = cal_metrics1(hurdle_cca_test_score_mv, de_thr, hurdle_cca_test_logFC_mv, lgfc_thr, hurdle_T_label_cca)
        metric_hurdle_cca_mv_poscca = cal_metrics1(hurdle_cca_test_score_mv_poscca, de_thr,
                                                   hurdle_cca_test_logFC_mv_poscca, lgfc_thr, hurdle_T_label_cca)
        metric_hurdle_cca_mv_precca = cal_metrics1(hurdle_cca_test_score_mv_precca, de_thr, hurdle_cca_test_logFC_mv_precca, lgfc_thr, hurdle_T_label_cca)
        metric_hurdle_cca_hurdle = cal_metrics1(hurdle_cca_test_score_hurdle, de_thr, hurdle_cca_test_logFC_hurdle, lgfc_thr, hurdle_T_label_cca)

        metrics_out_all = [['v1_all'] + list(metric_all_v1), ['v2_all'] + list(metric_all_v2),
                           ['mv_all'] + list(metric_all_mv), ['mv_pred_tr_all'] + list(metric_all_mv_pred_tr),
                           ['mv_all_poscca'] + list(metric_all_mv_poscca),
                           ['mv_pred_tr_all_poscca'] + list(metric_all_mv_pred_tr_poscca),
                           ['hurdle_all'] + list(metric_all_hurdle),
                           ['v1_all_precca'] + list(metric_all_v1_precca), ['v2_all_precca'] + list(metric_all_v2_precca),
                           ['mv_all_precca'] + list(metric_all_mv_precca),
                           ['mv_pred_tr_all_precca'] + list(metric_all_mv_pred_tr_precca),
                           ['hurdle_all_precca'] + list(metric_all_hurdle_precca),
                           ['ori_v1'] + list(ori_metric_v1),
                           ['hurdle_v1'] + list(metric_v1_hurdle), ['pred_v1'] + list(metric_v1_mv),
                           ['pred_v1_poscca'] + list(metric_v1_mv_poscca),
                           ['hurdle_v1_precca'] + list(metric_v1_hurdle_precca),
                           ['pred_v1_precca'] + list(metric_v1_mv_precca),
                           ['ori_v2'] + list(ori_metric_v2), ['hurdle_v2'] + list(metric_v2_hurdle),
                           ['pred_v2'] + list(metric_v2_mv), ['pred_v2_poscca'] + list(metric_v2_mv_poscca),
                           ['hurdle_v2_precca'] + list(metric_v2_hurdle_precca),
                           ['pred_v2_precca'] + list(metric_v2_mv_precca),
                           ['ori_hurdle'] + list(ori_metric_hurdle),
                           ['pred_hurdle'] + list(metric_hurdle_mv),
                           ['pred_hurdle_poscca'] + list(metric_hurdle_mv_poscca),
                           ['pred_hurdle_precca'] + list(metric_hurdle_mv_precca),
                           ['pred_hurdle_hudle_cca'] + list(metric_hurdle_hurdle_precca),
                           ['ori_v1_cca'] + list(ori_metric_v1_cca),
                           ['hurdle_v1_cca'] + list(metric_v1_cca_hurdle), ['pred_v1_cca'] + list(metric_v1_cca_mv),
                           ['pred_v1_cca_poscca'] + list(metric_v1_cca_mv_poscca),
                           ['hurdle_v1_cca_precca'] + list(metric_v1_cca_hurdle_precca),
                           ['pred_v1_cca_precca'] + list(metric_v1_cca_mv_precca),
                           ['ori_v2_cca'] + list(ori_metric_v2_cca), ['hurdle_v2_cca'] + list(metric_v2_cca_hurdle),
                           ['pred_v2_cca'] + list(metric_v2_cca_mv), ['pred_v2_cca_poscca'] + list(metric_v2_cca_mv_poscca),
                           ['hurdle_v2_cca_precca'] + list(metric_v2_cca_hurdle_precca),
                           ['pred_v2_cca_precca'] + list(metric_v2_cca_mv_precca),
                           ['ori_hurdle_cca'] + list(ori_metric_hurdle_cca),
                           ['pred_hurdle_cca'] + list(metric_hurdle_cca_mv),
                           ['pred_hurdle_cca_poscca'] + list(metric_hurdle_cca_mv_poscca),
                           ['pred_hurdle_cca_precca'] + list(metric_hurdle_cca_mv_precca),
                           ['pred_hurdle_cca_hurdle'] + list(metric_hurdle_cca_hurdle)
                           ]

        pd.DataFrame(np.array(metrics_out_all),
                     columns=['test_type', 'Acc', 'Prec', 'Rec', 'F1', 'F1w', 'Mcc', 'nMcc', 'geomean', 'tn',
                                                     'fp', 'fn', 'tp', 'pauc001',
                                                     'pauc005', 'pauc01']).to_csv(
            save_folder + dataset + '_' + v1_ty + '_' + v2_ty + '_metrics_all.csv',
            sep=',', index=False)

def get_training_proteins_repoduce_mv(proteins_vs, view_protein, qval_vs, logFC_vs, fea_all,
                          logFC_all, view_organism, de_organism, hc_thr, de_thr, npos_thr, combine_score_ty, hurdle_qval,
                          hurdle_protein, hurdle_logFC, views_names, pos_ty):

    seed_torch(2023)
    label_vs = []
    com_pro = []
    for v in range(len(proteins_vs)):
        label_v = label_pro(logFC_vs[v], qval_vs[v], np.log2(1.5), de_thr)

        if v==0:
            com_pro = proteins_vs[v]
        else:
            com_pro = np.intersect1d(com_pro, proteins_vs[v])

        label_vs.append(label_v)

    label_hurdle = label_pro(hurdle_logFC, hurdle_qval, np.log2(1.5), de_thr)

    if pos_ty == '0_h':  # consistent in all view

        com_pro = np.intersect1d(proteins_vs[0], hurdle_protein)

    elif pos_ty == '1_h':  # consistent in maxlfq & hurdle

        com_pro = np.intersect1d(proteins_vs[1], hurdle_protein)

    com_pro_labs = []
    com_pro_pros = []
    com_pro_cons = []
    hc_pros = []

    for v in range(len(proteins_vs)):
        com, idx1, idx2 = np.intersect1d(proteins_vs[v], com_pro, return_indices=True)
        com_pro_lab = label_vs[v][idx1, 0]
        com_pro_labs.append(com_pro_lab)
        com_pro_pro = proteins_vs[v][idx1]
        com_pro_pros.append(com_pro_pro)
        hc_pro = proteins_vs[v][np.where((qval_vs[v] <= hc_thr))[0]]
        hc_pros.append(hc_pro)

    com, idx1, idx2 = np.intersect1d(hurdle_protein, com_pro, return_indices=True)
    com_pro_lab_hurdle = label_hurdle[idx1, 0]
    com_pro_pro_hurdle = hurdle_protein[idx1]

    if pos_ty == '0_1_2':  # consistent in all view

        for i in range(len(com_pro_labs[0])):
            lbs = [com_pro_labs[j][i] for j in range(len(com_pro_labs))]
            if len(np.unique(np.array(lbs)))==1:
                com_pro_cons.append(com_pro_pros[0][i])

    elif pos_ty == '0_h': # consistent in dlfq & hurdle

        com_pro_cons = com_pro_pro_hurdle[com_pro_labs[0] == com_pro_lab_hurdle]

    elif pos_ty == '1_h': # consistent in maxlfq & hurdle

        com_pro_cons = com_pro_pro_hurdle[com_pro_labs[1] == com_pro_lab_hurdle]

    train_protein = np.unique(np.array(list(com_pro_cons)))

    train_qval_vs = []
    train_logfc_vs = []

    if pos_ty == '0_1_2':  # consistent in all view

        for v in range(len(proteins_vs)):
            train_qval_v = np.ones((len(train_protein), 1))
            train_logfc_v = np.zeros((len(train_protein), 1))
            com_v1, idx_v1, idx_tr1 = np.intersect1d(proteins_vs[v], train_protein, return_indices=True)
            train_qval_v[idx_tr1, 0] = qval_vs[v][idx_v1]
            train_logfc_v[idx_tr1, 0] = logFC_vs[v][idx_v1]

            train_qval_vs.append(train_qval_v)
            train_logfc_vs.append(train_logfc_v)
    else:

        if pos_ty == '0_h': # consistent in dlfq & hurdle

            train_qval_v = np.ones((len(train_protein), 1))
            train_logfc_v = np.zeros((len(train_protein), 1))
            com_v1, idx_v1, idx_tr1 = np.intersect1d(proteins_vs[0], train_protein, return_indices=True)
            train_qval_v[idx_tr1, 0] = qval_vs[0][idx_v1]
            train_logfc_v[idx_tr1, 0] = logFC_vs[0][idx_v1]

            train_qval_vs.append(train_qval_v)
            train_logfc_vs.append(train_logfc_v)

        elif pos_ty == '1_h': # consistent in maxlfq & hurdle

            train_qval_v = np.ones((len(train_protein), 1))
            train_logfc_v = np.zeros((len(train_protein), 1))
            com_v1, idx_v1, idx_tr1 = np.intersect1d(proteins_vs[1], train_protein, return_indices=True)
            train_qval_v[idx_tr1, 0] = qval_vs[1][idx_v1]
            train_logfc_v[idx_tr1, 0] = logFC_vs[1][idx_v1]

            train_qval_vs.append(train_qval_v)
            train_logfc_vs.append(train_logfc_v)

        train_qval_v = np.ones((len(train_protein), 1))
        train_logfc_v = np.zeros((len(train_protein), 1))
        com_v1, idx_v1, idx_tr1 = np.intersect1d(hurdle_protein, train_protein, return_indices=True)
        train_qval_v[idx_tr1, 0] = hurdle_qval[idx_v1]
        train_logfc_v[idx_tr1, 0] = hurdle_logFC[idx_v1]

        train_qval_vs.append(train_qval_v)
        train_logfc_vs.append(train_logfc_v)

    if combine_score_ty == 'min':
        train_qval = np.array([min(np.array([train_qval_vs[j][i] for j in range(len(train_qval_vs))]))
                               for i in range(len(train_qval_vs[0]))])
    elif combine_score_ty == 'dlfq':
        idx = np.where((np.array(views_names) == combine_score_ty))[0][0]
        train_qval = np.array([train_qval_vs[idx][i] for i in range(len(train_qval_vs[0]))])
    elif combine_score_ty == 'mean':
        train_qval = np.array([np.mean(np.array([train_qval_vs[j][i] for j in range(len(train_qval_vs))]))
                               for i in range(len(train_qval_vs[0]))])
    elif combine_score_ty == 'max':
        train_qval = np.array([max(np.array([train_qval_vs[j][i] for j in range(len(train_qval_vs))]))
                               for i in range(len(train_qval_vs[0]))])
    elif combine_score_ty == 'hurdle':
        train_qval_hurdle = np.ones((len(train_protein), 1))
        com_v2, idx_v2, idx_tr2 = np.intersect1d(hurdle_protein, train_protein, return_indices=True)
        train_qval_hurdle[idx_tr2, 0] = hurdle_qval[idx_v2]
        train_qval = train_qval_hurdle

    if (len(np.where((np.array(train_qval) <= de_thr))[0]) < npos_thr) & (len(np.where((np.array(train_qval) <= de_thr))[0])>=0):
        train_protein0 = list(com_pro_cons)
        for v in range(len(proteins_vs)):
            train_protein0 = train_protein0 + list(hc_pros[v])

        train_protein = np.unique(np.array(train_protein0))

        train_qval_vs = []
        train_logfc_vs = []

        if pos_ty == '0_1_2':  # consistent in all view

            for v in range(len(proteins_vs)):
                train_qval_v = np.ones((len(train_protein), 1))
                train_logfc_v = np.zeros((len(train_protein), 1))
                com_v1, idx_v1, idx_tr1 = np.intersect1d(proteins_vs[v], train_protein, return_indices=True)
                train_qval_v[idx_tr1, 0] = qval_vs[v][idx_v1]
                train_logfc_v[idx_tr1, 0] = logFC_vs[v][idx_v1]

                train_qval_vs.append(train_qval_v)
                train_logfc_vs.append(train_logfc_v)
        else:

            if pos_ty == '0_h':  # consistent in dlfq & hurdle

                train_qval_v = np.ones((len(train_protein), 1))
                train_logfc_v = np.zeros((len(train_protein), 1))
                com_v1, idx_v1, idx_tr1 = np.intersect1d(proteins_vs[0], train_protein, return_indices=True)
                train_qval_v[idx_tr1, 0] = qval_vs[0][idx_v1]
                train_logfc_v[idx_tr1, 0] = logFC_vs[0][idx_v1]

                train_qval_vs.append(train_qval_v)
                train_logfc_vs.append(train_logfc_v)

            elif pos_ty == '1_h':  # consistent in maxlfq & hurdle

                train_qval_v = np.ones((len(train_protein), 1))
                train_logfc_v = np.zeros((len(train_protein), 1))
                com_v1, idx_v1, idx_tr1 = np.intersect1d(proteins_vs[1], train_protein, return_indices=True)
                train_qval_v[idx_tr1, 0] = qval_vs[1][idx_v1]
                train_logfc_v[idx_tr1, 0] = logFC_vs[1][idx_v1]

                train_qval_vs.append(train_qval_v)
                train_logfc_vs.append(train_logfc_v)

            train_qval_v = np.ones((len(train_protein), 1))
            train_logfc_v = np.zeros((len(train_protein), 1))
            com_v1, idx_v1, idx_tr1 = np.intersect1d(hurdle_protein, train_protein, return_indices=True)
            train_qval_v[idx_tr1, 0] = hurdle_qval[idx_v1]
            train_logfc_v[idx_tr1, 0] = hurdle_logFC[idx_v1]

            train_qval_vs.append(train_qval_v)
            train_logfc_vs.append(train_logfc_v)

        if len(np.where((np.array(train_qval) <= de_thr))[0]) < npos_thr:
            print('no enough positive training data!!!')

            train_pro = []
            train_fea = []
            train_logFC = []
            train_scores = []
            train_orga = []
            train_true_label = []
            train_qval_vs = []
            train_logfc_vs = []

            return (train_pro, train_fea, train_scores, train_logFC, train_orga, train_true_label, train_qval_vs, train_logfc_vs)

    train_score = 1-train_qval

    com_pro_train, idx11, idx12 = np.intersect1d(train_protein, view_protein, return_indices=True)

    train_pro = com_pro_train
    train_fea = fea_all[idx12, :]
    train_logFC = logFC_all[idx12, :]
    train_scores = train_score[idx11]
    train_orga = view_organism[idx12]
    train_true_label = get_true_label(train_pro, de_organism)


    idx_remain = []

    for v in range(len(train_qval_vs)):
        train_qval_vs[v] = train_qval_vs[v][idx11]
        train_logfc_vs[v] = train_logfc_vs[v][idx11]

        if v == 0:
            idx_remain = np.where((train_qval_vs[v] < 1))[0]
        else:
            idx_remain = np.intersect1d(idx_remain, np.where((train_qval_vs[v] < 1))[0])

    train_pro = train_pro[idx_remain]
    train_fea = train_fea[idx_remain, :]
    train_logFC = train_logFC[idx_remain, :]
    train_scores = train_scores[idx_remain]
    train_orga = train_orga[idx_remain]
    train_true_label = np.array(train_true_label)[idx_remain]

    for v in range(len(train_qval_vs)):
        train_qval_vs[v] = train_qval_vs[v][idx_remain]
        train_logfc_vs[v] = train_logfc_vs[v][idx_remain]

    return (train_pro, train_fea, train_logFC, train_scores, train_orga, train_true_label, train_qval_vs, train_logfc_vs)

def get_test_proteins_reproduce_mv(proteins_vs, view_protein, score_vs, logFC_vs, fea_all,
                          logFC_all, hurdle_proteins, hurdle_score, view_organism, de_organism, train_pro):

    test_protein = np.setdiff1d(view_protein, train_pro)

    com_pro_test, idx11, idx12 = np.intersect1d(test_protein, view_protein, return_indices=True)

    test_pro = com_pro_test
    test_fea = fea_all[idx12, :]
    test_logFC = logFC_all[idx12, :]
    test_orga = view_organism[idx12]
    test_true_label = get_true_label(test_pro, de_organism)

    score_v_tests = []
    logfc_v_tests = []

    for v in range(len(proteins_vs)):
        score_v_test = np.zeros((len(test_pro), 1))
        logfc_v_test = np.zeros((len(test_pro), 1))

        com1, idx111, idx112 = np.intersect1d(test_pro, proteins_vs[v], return_indices=True)

        score_v_test[idx111, 0] = score_vs[v][idx112]
        logfc_v_test[idx111, 0] = logFC_vs[v][idx112]

        score_v_tests.append(score_v_test[:, 0])
        logfc_v_tests.append(logfc_v_test[:, 0])

    test_score_hurdle = np.zeros((len(test_pro), 1))
    com_h, idx_h, idx_l = np.intersect1d(hurdle_proteins, test_pro, return_indices=True)
    idx_h_l = np.array([list(hurdle_proteins).index(com_h[i]) for i in range(len(com_h))], dtype='int')
    test_score_hurdle[idx_l, 0] = hurdle_score[idx_h_l]

    return (test_pro, test_fea, test_logFC, test_orga, np.array(test_true_label), score_v_tests,
            test_score_hurdle[:, 0], logfc_v_tests)

def cv_model_mv(feas, scores, FC, seed, model_type, de_thr, param, lgfc_thr, view_names):
    seed_torch(seed)
    lab = np.zeros((len(scores), 1))
    lab[np.where((((1-scores) < de_thr) & (abs(FC)>=lgfc_thr))), 0] = 1
    num_pos = len(np.where((lab[:,0] == 1))[0])
    num_neg = len(np.where((lab[:,0] == 0))[0])

    ratio = num_neg / num_pos

    train_feas = []
    train_scores = []
    train_labs = []
    N = 0

    if (ratio > 0.5) & (ratio <= 1) : # more positive than negative
        select_pos = np.random.randint(0, num_pos, size=num_neg)

        train_feas_balance = feas[np.array(
            list(np.where((lab[:, 0] == 1))[0][select_pos]) + list(np.where((lab[:, 0] == 0))[0])), :]
        train_score_balance = scores[
            np.array(list(np.where((lab[:, 0] == 1))[0][select_pos]) + list(np.where((lab[:, 0] == 0))[0]))]
        train_lab_balance = lab[
            np.array(list(np.where((lab[:, 0] == 1))[0][select_pos]) + list(np.where((lab[:, 0] == 0))[0]))]
        train_feas.append(train_feas_balance)
        train_scores.append(train_score_balance)
        train_labs.append(train_lab_balance)
        N =1

    elif (ratio > 1) & (ratio < 2):

        select_neg = np.random.randint(0, num_neg, size=num_pos)

        train_feas_balance = feas[np.array(
            list(np.where((lab[:, 0] == 1))[0]) + list(np.where((lab[:, 0] == 0))[0][select_neg])), :]
        train_score_balance = scores[
            np.array(list(np.where((lab[:, 0] == 1))[0]) + list(np.where((lab[:, 0] == 0))[0][select_neg]))]

        train_feas.append(train_feas_balance)
        train_scores.append(train_score_balance)
        train_lab_balance = lab[
            np.array(list(np.where((lab[:, 0] == 1))[0]) + list(np.where((lab[:, 0] == 0))[0][select_neg]))]
        train_labs.append(train_lab_balance)
        N = 1

    elif ratio < 0.5:

        N = int(num_pos / num_neg)
        remain_pos = np.where((lab[:, 0] == 1))[0]
        neg_idx = np.where((lab[:, 0] == 0))[0]

        for i in range(N):
            select_pos = np.random.randint(0, len(remain_pos), size=num_neg)
            train_feas_balance = feas[np.array(list(neg_idx) + list(remain_pos[select_pos])), :]
            train_score_balance = scores[np.array(list(neg_idx) + list(remain_pos[select_pos]))]
            train_feas.append(train_feas_balance)
            train_scores.append(train_score_balance)
            train_lab_balance = lab[np.array(list(neg_idx) + list(remain_pos[select_pos]))]
            train_labs.append(train_lab_balance)
            remain_pos = np.setdiff1d(remain_pos, remain_pos[select_pos])

    elif ratio > 2:

        N = int(ratio)

        remain_neg = np.where((lab[:, 0] == 0))[0]
        pos_idx = np.where((lab[:, 0] == 1))[0]

        for i in range(N):
            select_neg = np.random.randint(0, len(remain_neg), size=num_pos)
            train_feas_balance = feas[np.array(list(pos_idx) + list(remain_neg[select_neg])), :]
            train_score_balance = scores[np.array(list(pos_idx) + list(remain_neg[select_neg]))]
            train_feas.append(train_feas_balance)
            train_scores.append(train_score_balance)
            train_lab_balance = lab[np.array(list(pos_idx) + list(remain_neg[select_neg]))]
            train_labs.append(train_lab_balance)
            remain_neg = np.setdiff1d(remain_neg, remain_neg[select_neg])

    pmses = 0
    n = 0
    sprs = 0
    prs = 0
    rt = 0
    aucs = 0

    for i in range(min([N, 30])):#[0]:# at most 30 repeats
        print(i)
        seed_torch(seed)
        logo = KFold(n_splits=3, shuffle=True)
        for train_index, test_index in logo.split(train_scores[i]):
            train_x = train_feas[i][train_index]
            train_lab = train_scores[i][train_index]
            train_label = train_labs[i][train_index]

            test_x = train_feas[i][test_index]
            test_lab = train_scores[i][test_index]
            test_label = train_labs[i][test_index]

            train_X, train_y = train_x, train_lab
            test_X, test_y = test_x, test_lab
            if model_type == 'KNN':

                if param['C'] > len(train_X[:, 0]):
                    nnei = len(train_X[:, 0])
                    param['C'] = nnei
                else:
                    nnei = param['C']
                model = KNeighborsRegressor(n_neighbors=nnei, p=param['e'])
                model.fit(train_X, train_y)
                pred_score = model.predict(test_X)
                pmses += pmse(pred_score, test_y)
                if len(np.unique(test_label))==1:
                    aucs+=0
                else:
                    aucs += roc_auc_score(test_label, pred_score)
                sprs += spr(pred_score, test_y).statistic
                if len(pred_score) <= 2:
                    prs += 0
                else:
                    if np.isnan(pr(pred_score, test_y).statistic):
                        prs += 0
                    else:
                        prs += pr(pred_score, test_y).statistic
                rt += r2_score(pred_score, test_y)
                n += 1

            elif model_type == 'MLE':
                Xs = []
                X_tests = []

                step = int(len(train_X[0, :]) / len(view_names))
                pred_scores = []
                for v in range(len(view_names)):
                    start = step * v
                    end = step * (v+1)

                    X = train_X[:, start:end]
                    X_test = test_X[:, start:end]
                    Xs.append(X)
                    X_tests.append(X_test)

                    #model = SVR(kernel="rbf", C=param['Cs'][v], epsilon=param['es'][v])
                    if param['Cs'][v] > len(train_X[:, 0]):
                        nnei = len(train_X[:, 0])
                        param['Cs'][v] = nnei
                    else:
                        nnei = param['Cs'][v]
                    model = KNeighborsRegressor(n_neighbors=nnei, p=param['es'][v])
                    model.fit(X, train_y)
                    pred_score = model.predict(X_test)

                    if v == 0:
                        pred_scores = pred_score
                    else:
                        pred_scores = pred_scores + pred_score

                pred_score = pred_scores/len(view_names)

                pmses += pmse(pred_score, test_y)
                if len(np.unique(test_label))==1:
                    aucs+=0
                else:
                    aucs += roc_auc_score(test_label, pred_score)
                sprs += spr(pred_score, test_y).statistic
                if len(pred_score) <= 2:
                    prs += 0
                else:
                    if np.isnan(pr(pred_score, test_y).statistic):
                        prs += 0
                    else:
                        prs += pr(pred_score, test_y).statistic
                rt += r2_score(pred_score, test_y)
                n += 1

            elif model_type == 'GCCA':

                Xs = []

                X_tests = []

                step = int(len(train_X[0, :]) / len(view_names))

                for v in range(len(view_names)):
                    start = step * v

                    end = step * (v + 1)

                    X = train_X[:, start:end]

                    X_test = test_X[:, start:end]

                    Xs.append(X)

                    X_tests.append(X_test)

                param.update({'n_components': min(param['n_components'], len(Xs[0][0, :]))})

                from cca_zoo.linear import GCCA

                kmcca = GCCA(latent_dimensions=param['n_components'], c=param['c'], random_state=2024)


                try:

                    kmcca_trans = kmcca.fit_transform(Xs)

                    kmcca_tran = kmcca_trans[0]

                    for v in range(1, len(kmcca_trans)):
                        kmcca_tran = np.hstack((kmcca_tran, kmcca_trans[v]))

                    kmcca_trans = kmcca_tran

                except:

                    continue  # doing nothing on exception

                # model = SVR(kernel="rbf", C=param['C'], epsilon=param['e'])

                if param['C'] > len(train_X[:, 0]):

                    nnei = len(train_X[:, 0])

                    param['C'] = nnei

                else:

                    nnei = param['C']

                model = KNeighborsRegressor(n_neighbors=nnei, p=param['e'])

                model.fit(kmcca_trans, train_y)

                test_trans = kmcca.transform(X_tests)

                if model_type != 'MCCA':

                    test_tran = test_trans[0]

                    for v in range(1, len(test_trans)):
                        test_tran = np.hstack((test_tran, test_trans[v]))

                    test_trans = test_tran

                pred_score = model.predict(test_trans)

                pmses += pmse(pred_score, test_y)

                sprs += spr(pred_score, test_y).statistic

                if len(np.unique(test_label)) == 1:

                    aucs += 0

                else:

                    aucs += roc_auc_score(test_label, pred_score)

                if len(pred_score) <= 2:

                    prs += 0

                else:

                    if np.isnan(pr(pred_score, test_y).statistic):

                        prs += 0

                    else:

                        prs += pr(pred_score, test_y).statistic

                rt += r2_score(pred_score, test_y)

                n += 1

    print(pmses/n, sprs/n, prs/n, rt/n, aucs/n)
    return pmses/n, sprs/n, prs/n, rt/n, aucs/n

def test_model_mv(feas, scores, FC, seed, model_type, test_feas_final, params, a, de_thr, lgfc_thr, view_names):
    seed_torch(seed)
    model = []
    if model_type == 'KNN':
        seed_torch(seed)
        model = KNeighborsRegressor(n_neighbors=params['C'], p=params['e'])

    elif model_type == 'MLE':

        seed_torch(seed)
        model = []

        for v in range(len(view_names)):
            model0 = KNeighborsRegressor(n_neighbors=params['C'+str(v)], p=params['e'+str(v)])
            model.append(model0)

    elif model_type == 'GCCA':

        seed_torch(seed)

        model = KNeighborsRegressor(n_neighbors=params['C'], p=params['e'])

    lab = np.zeros((len(scores), 1))
    lab[np.where((((1 - scores) < de_thr) & (abs(FC)>=lgfc_thr))), 0] = 1
    num_pos = len(np.where((lab[:, 0] == 1))[0])
    num_neg = len(np.where((lab[:, 0] == 0))[0])

    ratio = num_neg / num_pos

    train_feas_all = []
    train_scores_all = []

    N = 0

    if (ratio > 0.5) & (ratio <= 1):  # more positive than negative but wi
        select_pos = np.random.randint(0, num_pos, size=num_neg)

        train_feas_balance = feas[np.array(
            list(np.where((lab[:, 0] == 1))[0][select_pos]) + list(np.where((lab[:, 0] == 0))[0])), :]
        train_score_balance = scores[
            np.array(list(np.where((lab[:, 0] == 1))[0][select_pos]) + list(np.where((lab[:, 0] == 0))[0]))]

        train_feas_all.append(train_feas_balance)
        train_scores_all.append(train_score_balance)
        N = 1

    elif (ratio > 1) & (ratio < 2):

        select_neg = np.random.randint(0, num_neg, size=num_pos)

        train_feas_balance = feas[np.array(
            list(np.where((lab[:, 0] == 1))[0]) + list(np.where((lab[:, 0] == 0))[0][select_neg])), :]
        train_score_balance = scores[
            np.array(list(np.where((lab[:, 0] == 1))[0]) + list(np.where((lab[:, 0] == 0))[0][select_neg]))]

        train_feas_all.append(train_feas_balance)
        train_scores_all.append(train_score_balance)

        N = 1

    elif (ratio < 0.5) & (num_neg>0):

        N = int(num_pos / num_neg)
        remain_pos = np.where((lab[:, 0] == 1))[0]
        neg_idx = np.where((lab[:, 0] == 0))[0]

        for i in range(N):
            select_pos = np.random.randint(0, len(remain_pos), size=num_neg)
            train_feas_balance = feas[np.array(list(neg_idx) + list(remain_pos[select_pos])), :]
            train_score_balance = scores[np.array(list(neg_idx) + list(remain_pos[select_pos]))]
            train_feas_all.append(train_feas_balance)
            train_scores_all.append(train_score_balance)
            remain_pos = np.setdiff1d(remain_pos, remain_pos[select_pos])

    elif ratio > 2:

        N = int(ratio)

        remain_neg = np.where((lab[:, 0] == 0))[0]
        pos_idx = np.where((lab[:, 0] == 1))[0]

        for i in range(N):
            select_neg = np.random.randint(0, len(remain_neg), size=num_pos)
            train_feas_balance = feas[np.array(list(pos_idx) + list(remain_neg[select_neg])), :]
            train_score_balance = scores[np.array(list(pos_idx) + list(remain_neg[select_neg]))]
            train_feas_all.append(train_feas_balance)
            train_scores_all.append(train_score_balance)
            remain_neg = np.setdiff1d(remain_neg, remain_neg[select_neg])

    pfs = []
    trs = []

    for i in range(min(N, 30)):#[0]:#
        if model_type == 'KNN':
            param0 = model.get_params()
            if (param0['n_neighbors']) > len(train_feas_all[i][:, 0]):
                param0.update({'n_neighbors': len(train_feas_all[i][:, 0])})
                model.set_params(**param0)
            model.fit(train_feas_all[i], train_scores_all[i])
            pfs.append(model.predict(test_feas_final))
            trs.append(model.predict(feas))
            trains_trans, test_trans = [], []

        elif model_type == 'MLE':
            pred_score = []
            pred_score_all = []
            step = int(len(train_feas_all[i][0, :]) / len(view_names))
            for v in range(len(view_names)):
                start = step * v
                end = step * (v + 1)

                X = train_feas_all[i][:, start:end]
                X_test = test_feas_final[:, start:end]
                fea_all = feas[:, start:end]

                model0 = model[v]

                param0 = model0.get_params()
                if (param0['n_neighbors'])> len(X[:,0]):
                    param0.update({'n_neighbors':len(X[:,0])})
                    model0.set_params(**param0)
                model0.fit(X, train_scores_all[i])
                if v == 0:
                    pred_score = model0.predict(X_test)
                    pred_score_all = model0.predict(fea_all)
                else:
                    pred_score = pred_score + model0.predict(X_test)
                    pred_score_all = pred_score_all + model0.predict(fea_all)

            pred_score = pred_score/len(view_names)
            pred_score_all = pred_score_all/len(view_names)
            pfs.append(pred_score)
            trs.append(pred_score_all)
            trains_trans, test_trans = [], []

        elif model_type == 'GCCA':

            step = int(len(train_feas_all[i][0, :]) / len(view_names))

            Xs = []

            X_tests = []

            trains = []

            for v in range(len(view_names)):
                start = step * v

                end = step * (v + 1)

                X = train_feas_all[i][:, start:end]

                X_test = test_feas_final[:, start:end]

                train = feas[:, start:end]

                Xs.append(X)

                X_tests.append(X_test)

                trains.append(train)

            kernel = 'linear'

            params.update({'n_components': min(params['n_components'], len(Xs[0][0, :]))})

            if model_type == 'GCCA':

                from cca_zoo.linear import GCCA

                kmcca = GCCA(latent_dimensions=params['n_components'], c=params['c'], random_state=2024)

            kmcca_trans = kmcca.fit_transform(Xs)

            kmcca_tran = kmcca_trans[0]

            for v in range(1, len(kmcca_trans)):
                kmcca_tran = np.hstack((kmcca_tran, kmcca_trans[v]))

            kmcca_trans = kmcca_tran

            param0 = model.get_params()

            if (param0['n_neighbors']) > len(kmcca_trans[:, 0]):
                param0.update({'n_neighbors': len(kmcca_trans[:, 0])})

                model.set_params(**param0)

            model.fit(kmcca_trans, train_scores_all[i])

            test_trans = kmcca.transform(X_tests)

            trains_trans = kmcca.transform(trains)

            if model_type != 'MCCA':

                test_tran = test_trans[0]

                trains_tran = trains_trans[0]

                for v in range(1, len(test_trans)):
                    test_tran = np.hstack((test_tran, test_trans[v]))

                    trains_tran = np.hstack((trains_tran, trains_trans[v]))

                test_trans = test_tran

                trains_trans = trains_tran

            pred_score = model.predict(test_trans)

            pred_score_all = model.predict(trains_trans)

            #pd.DataFrame(trains_trans).to_csv()

            pfs.append(pred_score)

            trs.append(pred_score_all)

    pf = np.mean(np.array(pfs).T, axis=1)
    tr = np.mean(np.array(trs).T, axis=1)

    pf[np.where((pf >= 1))[0]] = max(pf[pf<1])
    pf[np.where((pf < 0))[0]] = 0

    tr[np.where((tr >= 1))[0]] = max(tr[tr<1])
    tr[np.where((tr < 0))[0]] = 0


    return pf*a, tr*a, trains_trans, test_trans

def metrics_reference(views_names, all_test_proteins, all_true_label, proteins_vs, score_vs, logFC_vs, hurdle_proteins,
                      hurdle_score, hurdle_logFC, train_pro, test_pro, train_scores, pred_finals, pred_trains, train_FC,
                      test_FC, de_thr, lgfc_thr, mv_methods, prefix):
    metrics = []
    methods = []

    labels = []
    scores = []
    fcs = []
    out_all = np.column_stack((all_test_proteins,all_true_label))
    out_name = ['Protein', 'True_lab']
    for v in range(len(views_names)):
        all_test_score_v = mapping_scores(all_test_proteins, proteins_vs[v], [], score_vs[v], [])
        all_logFC_v = mapping_scores(all_test_proteins, proteins_vs[v], [], logFC_vs[v], [])

        metric_all_v = cal_metrics1(all_test_score_v, de_thr, all_logFC_v, lgfc_thr, all_true_label)

        metrics.append(metric_all_v)
        methods.append(prefix + '_view' + str(v))

        scores.append(all_test_score_v)
        fcs.append(all_logFC_v)

        lab = np.zeros((len(all_test_score_v), 1))
        lab[np.where(((all_test_score_v >= 1 - de_thr) & (abs(all_logFC_v) >= lgfc_thr)))[0], 0] = 1
        labels.append(lab)

        out_all = np.column_stack((out_all, all_test_score_v, all_logFC_v, lab))
        out_name.extend(['score_'+prefix+views_names[v], 'logFC_'+prefix+views_names[v], 'lab_'+prefix+views_names[v]])

    all_test_score_hurdle = mapping_scores(all_test_proteins, hurdle_proteins, [], hurdle_score, [])
    all_logFC_hurdle = mapping_scores(all_test_proteins, hurdle_proteins, [], hurdle_logFC, [])
    metric_all_hurdle = cal_metrics1(all_test_score_hurdle, de_thr, all_logFC_hurdle, lgfc_thr, all_true_label)
    metrics.append(metric_all_hurdle)
    methods.append(prefix + '_hurdle')

    lab_hurdle = np.zeros((len(all_test_score_hurdle), 1))
    lab_hurdle[np.where(((all_test_score_hurdle >= 1 - de_thr) & (abs(all_logFC_hurdle) >= lgfc_thr)))[0], 0] = 1

    out_all = np.column_stack((out_all, all_test_score_hurdle, all_logFC_hurdle, lab_hurdle))
    out_name.extend(['score_'+prefix+'_hurdle', 'logFC_'+prefix+'_hurdle', 'lab_'+prefix+'_hurdle'])


    if len(pred_finals) > 0:
        for mth in range(len(pred_finals)):
            all_test_score_incons = mapping_scores(all_test_proteins, train_pro, test_pro, train_scores[:, 0], pred_finals[mth])
            all_test_score_all = mapping_scores(all_test_proteins, train_pro, test_pro, pred_trains[mth], pred_finals[mth])
            all_test_logFC_incons = mapping_scores(all_test_proteins, train_pro, test_pro, train_FC[:, 0], test_FC[:, 0])
            all_test_logFC_all = mapping_scores(all_test_proteins, train_pro, test_pro, train_FC[:, 0], test_FC[:, 0])
            metric_all_incons = cal_metrics1(all_test_score_incons, de_thr, all_test_logFC_incons, lgfc_thr, all_true_label)
            metric_all_all = cal_metrics1(all_test_score_all, de_thr, all_test_logFC_all, lgfc_thr,
                                          all_true_label)
            # metrics.append(metric_all_incons)
            # methods.append(prefix + '_' + mv_methods[mth] + '_incons_rescore')
            metrics.append(metric_all_all)
            methods.append(prefix + '_' + mv_methods[mth] + '_all_rescore')

        #     lab_incons = np.zeros((len(all_test_score_incons), 1))
            lab_all = np.zeros((len(all_test_score_all), 1))
        #     lab_incons[
        #         np.where(((all_test_score_incons >= 1 - de_thr) & (abs(all_test_logFC_incons) >= lgfc_thr)))[0], 0] = 1
            lab_all[
                np.where(((all_test_score_all >= 1 - de_thr) & (abs(all_test_logFC_all) >= lgfc_thr)))[0], 0] = 1
        #
        #     out_all = np.column_stack((out_all, all_test_score_incons, all_test_logFC_incons, lab_incons))
        #     out_name.extend(['score_'+prefix+'incons', 'logFC_'+prefix+'incons', 'lab_'+prefix+'incons'])
            out_all = np.column_stack((out_all, all_test_score_all, all_test_logFC_all, lab_all))
            out_name.extend(['score_' + prefix + mv_methods[mth], 'logFC_' + prefix + mv_methods[mth], 'lab_' + prefix + mv_methods[mth]])
        #
        #     idx_incons = np.where(((labels[0] == 1) & (lab_hurdle == 1) & (lab_incons == 0)))[0]
        #     idx_all = np.where(((labels[0] == 1) & (lab_hurdle == 1) & (lab_all == 0)))[0]
        #
        #     all_test_score_incons_add = copy.deepcopy(all_test_score_incons)
        #     all_test_score_all_add = copy.deepcopy(all_test_score_all)
        #     all_test_logFC_incons_add = copy.deepcopy(all_test_logFC_incons)
        #     all_test_logFC_all_add = copy.deepcopy(all_test_logFC_all)
        #
        #     all_test_score_incons_add[idx_incons] = np.array([max(scores[0][i], all_test_score_hurdle[i]) for i in idx_incons])
        #     all_test_logFC_incons_add[idx_incons] = np.array([[fcs[0][i], all_logFC_hurdle[i]][np.where(
        #         (abs(np.array([fcs[0][i], all_logFC_hurdle[i]])) == max([abs(fcs[0][i]), abs(all_logFC_hurdle[i])])))[0][0]]
        #                                                   for
        #                                                   i in idx_incons])
        #
        #     all_test_score_all_add[idx_all] = np.array([max(scores[0][i], all_test_score_hurdle[i]) for i in idx_all])
        #     all_test_logFC_all_add[idx_all] = np.array([[fcs[0][i], all_logFC_hurdle[i]][np.where(
        #         (abs(np.array([fcs[0][i], all_logFC_hurdle[i]])) == max([abs(fcs[0][i]), abs(all_logFC_hurdle[i])])))[0][0]]
        #                                             for
        #                                             i in idx_all])
        #
        #     metric_all_incons_add = cal_metrics1(all_test_score_incons_add, de_thr, all_test_logFC_incons_add, lgfc_thr, all_true_label)
        #     metric_all_all_add = cal_metrics1(all_test_score_all_add, de_thr, all_test_logFC_all_add, lgfc_thr,
        #                                   all_true_label)
        #     metrics.append(metric_all_incons_add)
        #     methods.append(prefix + '_' + mv_methods[mth] + '_incons_rescore_add_dlfq_hurdle')
        #     metrics.append(metric_all_all_add)
        #     methods.append(prefix + '_' + mv_methods[mth] + '_all_rescore_add_dlfq_hurdle')
        #     lab_incons_add = np.zeros((len(all_test_score_incons_add), 1))
        #     lab_incons_add[
        #         np.where(((all_test_score_incons_add >= 1 - de_thr) & (abs(all_test_logFC_incons_add) >= lgfc_thr)))[0], 0] = 1
        #     lab_all_add = np.zeros((len(all_test_score_all_add), 1))
        #     lab_all_add[
        #         np.where(((all_test_score_all_add >= 1 - de_thr) & (abs(all_test_logFC_all_add) >= lgfc_thr)))[
        #             0], 0] = 1
        #     out_all = np.column_stack((out_all, all_test_score_incons_add, all_test_logFC_incons_add, lab_incons_add))
        #     out_name.extend(['score_' + prefix + 'incons_add', 'logFC_' + prefix + 'incons_add', 'lab_' + prefix + 'incons_add'])
        #     out_all = np.column_stack((out_all, all_test_score_all_add, all_test_logFC_all_add, lab_all_add))
        #     out_name.extend(
        #         ['score_' + prefix + 'all_add', 'logFC_' + prefix + 'all_add', 'lab_' + prefix + 'all_add'])
        #
        # for spi in ['0', 'min_all', 'min_v0h', 'v0','h']:
        #     if spi == '0':
        #         spi_score = np.zeros((len(test_pro), 1))
        #     elif spi == 'min_all':
        #         spi_score = np.zeros((len(test_pro), 1))
        #
        #         for i in range(len(test_pro)):
        #             pro = test_pro[i]
        #             score_v = []
        #             for v in range(len(proteins_vs)):
        #                 idx = np.where((proteins_vs[v]==pro))[0]
        #                 if len(idx)>0:
        #                     score_v.append(score_vs[v][idx[0]])
        #
        #             if len(score_v)>0:
        #                 spi_score[i, 0]=max(score_v)
        #
        #     elif spi == 'min_v0h':
        #
        #         spi_score = np.zeros((len(test_pro), 1))
        #
        #         for i in range(len(test_pro)):
        #             pro = test_pro[i]
        #             score_v = []
        #
        #             idx = np.where((proteins_vs[0] == pro))[0]
        #             if len(idx) > 0:
        #                 score_v.append(score_vs[0][idx[0]])
        #
        #             idx = np.where((hurdle_proteins == pro))[0]
        #             if len(idx) > 0:
        #                 score_v.append(hurdle_score[idx[0]])
        #
        #             if len(score_v) > 0:
        #                 spi_score[i, 0] = max(score_v)
        #
        #     elif spi == 'v0':
        #         spi_score = np.zeros((len(test_pro), 1))
        #         com, idx1, idx2 = np.intersect1d(test_pro, proteins_vs[0], return_indices=True)
        #         spi_score[idx1, 0] = score_vs[0][idx2]
        #
        #     elif spi == 'h':
        #
        #         spi_score = np.zeros((len(test_pro), 1))
        #         com, idx1, idx2 = np.intersect1d(test_pro, hurdle_proteins, return_indices=True)
        #         spi_score[idx1, 0] = hurdle_score[idx2]
        #
        #     all_test_score_spi = mapping_scores(all_test_proteins, train_pro, test_pro, train_scores[:, 0], spi_score[:, 0])
        #     all_test_logFC_spi = mapping_scores(all_test_proteins, train_pro, test_pro, train_FC[:, 0], test_FC[:, 0])
        #     lab_spi = np.zeros((len(all_test_score_spi), 1))
        #     lab_spi[
        #         np.where(((all_test_score_spi >= 1 - de_thr) & (abs(all_test_logFC_spi) >= lgfc_thr)))[0], 0] = 1
        #     idx_spi = np.where(((labels[0] == 1) & (lab_hurdle == 1) & (lab_spi == 0)))[0]
        #
        #     all_test_score_spi[idx_spi] = np.array([max(scores[0][i], all_test_score_hurdle[i]) for i in idx_spi])
        #     all_test_logFC_spi[idx_spi] = np.array([[fcs[0][i], all_logFC_hurdle[i]][np.where(
        #         (abs(np.array([fcs[0][i], all_logFC_hurdle[i]])) == max([abs(fcs[0][i]), abs(all_logFC_hurdle[i])])))[0][0]]
        #                                                       for
        #                                                       i in idx_spi])
        #     lab_test_spi = np.zeros((len(all_test_score_spi),1))
        #     lab_test_spi[
        #         np.where(((all_test_score_spi >= 1 - de_thr) & (abs(all_test_logFC_spi) >= lgfc_thr)))[
        #             0], 0] = 1
        #
        #     metric_all_spi = cal_metrics1(all_test_score_spi, de_thr, all_test_logFC_spi, lgfc_thr,
        #                                          all_true_label)
        #     metrics.append(metric_all_spi)
        #     methods.append(prefix + '_spi_' + spi + '_add_dlfq_hurdle')
        #
        #     out_all = np.column_stack((out_all, all_test_score_spi, all_test_logFC_spi, lab_test_spi))
        #     out_name.extend(
        #         ['score_' + prefix + '_spi_' + spi + '_add_dlfq_hurdle', 'logFC_' + prefix + '_spi_' + spi + '_add_dlfq_hurdle', 'lab_' + prefix + '_spi_' + spi + '_add_dlfq_hurdle'])

    out_all_res = pd.DataFrame(out_all, columns=out_name)
    return metrics, methods, out_all_res

def MV_DEA_all_mv(views, v_scales, res_real_norm_vs, res_hurdle,  lgfc_thr, dethr, cbt, view_paths,
                                   views_names, hc_thr, np_thr, num_tra,
                                      model, seed, op_ty, save_folder, logFC_norm):
    seed_torch()

    isExist = os.path.exists(save_folder)

    if not isExist:
        os.makedirs(save_folder)

    de_organism = view_paths['DE_organism']
    dataset = view_paths['dataset']
    view_pro = views[0]['Protein'].values
    organism_pro = views[0]['Organism'].values
    fea_all = v_scales[0]
    for i in range(1, len(views_names)):
        fea_all = np.column_stack((fea_all, v_scales[i]))
    fea_all[np.isnan(fea_all)] = 0

    count_zeros = [len(np.where((fea_all[i, :] == 0))[0]) for i in range(len(fea_all[:, 0]))]
    idx_retain = np.where((np.array(count_zeros) <= len(fea_all[0, :]) * 0.8))[0]

    view_pro = view_pro[idx_retain]
    organism_pro = organism_pro[idx_retain]
    fea_all = fea_all[idx_retain, :]
    logFC_all = logFC_norm[idx_retain, :]

    proteins_vs = []
    logFC_vs = []
    qval_vs = []
    score_vs = []
    T_label_vs = []
    for i in range(0, len(views_names)):
        proteins_vs.append(res_real_norm_vs[i][1])
        logFC_vs.append(res_real_norm_vs[i][6])
        qval_vs.append(res_real_norm_vs[i][9])
        score_vs.append(res_real_norm_vs[i][2])
        T_label_vs.append(res_real_norm_vs[i][4])

    hurdle_proteins = res_hurdle[1]
    hurdle_logFC = res_hurdle[6]
    hurdle_qval = res_hurdle[9]
    hurdle_score = res_hurdle[2]
    hurdle_T_label = res_hurdle[4]

    (train_pro, train_fea, train_FC, train_scores, train_orga, train_true_label, train_qval_vs, train_logfc_vs) = get_training_proteins_repoduce_mv(proteins_vs,
                                                                                         view_pro, qval_vs,
                                                                                         logFC_vs, fea_all,
                                                                                         logFC_all,
                                                                                         organism_pro, de_organism,
                                                                                         hc_thr, dethr, np_thr,
                                                                                         cbt, hurdle_qval,
                                                                                         hurdle_proteins, hurdle_logFC, views_names)

    # no cca
    (test_pro, test_fea, test_FC, test_orga, test_true_label, score_v_tests, test_score_hurdle,
     logfc_v_tests) = get_test_proteins_reproduce_mv(proteins_vs, view_pro, score_vs, logFC_vs,
                                                  fea_all, logFC_all, hurdle_proteins, hurdle_score, organism_pro,
                                                  de_organism, train_pro)


    if len(train_pro) > 0:
        out_tranin = np.array(
            [list(train_pro), list(train_scores[:, 0]), list(train_FC[:, 0]), list(train_orga), list(train_true_label)]).T
        pd.DataFrame(out_tranin, columns=['Protein', 'organism', 'scores', 'logFC', 'true_label']).to_csv(
            save_folder + dataset + '_' + str(len(views_names)) + 'views_train_pros.csv', sep=',', index=False)

        # train_feas, train_scores, seed, model_type, test_feas_final, trial_num, a, de_thr, diff_ty
        pred_finals, pred_trains, best_paramss = [], [], []
        mv_methods = ['MCCA']#['mv'] #'mv','mv',, 'MVMDS', 'MCCA_rbf',  , 'GCCA', 'GRCCA', 'PRCCA', 'SVM', 'MLE'
        for model in mv_methods:
            pred_final, pred_train, best_params = optuna_optimize_mv(train_fea, train_scores[:, 0], train_FC[:, 0],
                                                                     seed, model,
                                                                     test_fea, 30, 1, dethr, op_ty, lgfc_thr,
                                                                     views_names)

            pred_finals.append(pred_final)
            pred_trains.append(pred_train)
            best_paramss.append(best_params)

        all_proteins = np.array(list(train_pro) + list(test_pro))
        all_true_label = np.array(list(train_true_label) + list(test_true_label))
        all_orga = np.array(list(train_orga) + list(test_orga))
        all_logFC = np.array(list(train_FC) + list(test_FC))

        all_test_proteins, idx = np.unique(all_proteins, return_index=True)

        metrics = []
        methods = []

        metrics_all, methods_all = metrics_reference(views_names, all_test_proteins, all_true_label[idx], proteins_vs,
                                                     score_vs, logFC_vs,
                                                     hurdle_proteins,
                                                     hurdle_score, hurdle_logFC, train_pro, test_pro, train_scores,
                                                     pred_finals, pred_trains,
                                                     train_FC,
                                                     test_FC, dethr, lgfc_thr, mv_methods, 'all')

        metrics.extend(metrics_all)
        methods.extend(methods_all)

        for v in range(len(views_names)):
            metrics_v, methods_v = metrics_reference(views_names, proteins_vs[v], T_label_vs[v], proteins_vs,
                                                     score_vs, logFC_vs,
                                                     hurdle_proteins,
                                                     hurdle_score, hurdle_logFC, train_pro, test_pro, train_scores,
                                                     pred_finals, pred_trains,
                                                     train_FC,
                                                     test_FC, dethr, lgfc_thr, mv_methods, 'view' + str(v))

            metrics.extend(metrics_v)
            methods.extend(methods_v)

        metrics_h, methods_h = metrics_reference(views_names, hurdle_proteins, hurdle_T_label, proteins_vs,
                                                 score_vs, logFC_vs,
                                                 hurdle_proteins,
                                                 hurdle_score, hurdle_logFC, train_pro, test_pro, train_scores,
                                                 pred_finals, pred_trains,
                                                 train_FC,
                                                 test_FC, dethr, lgfc_thr, mv_methods, 'hurdle')

        metrics.extend(metrics_h)
        methods.extend(methods_h)

        metrics = np.array(metrics)
        methods = np.array(methods)
        metrics_out_all = np.column_stack((methods, metrics))
        pd.DataFrame(np.array(metrics_out_all),
                     columns=['test_type', 'Acc', 'Prec', 'Rec', 'F1', 'F1w', 'Mcc', 'nMcc', 'geomean', 'tn',
                              'fp', 'fn', 'tp', 'pauc001',
                              'pauc005', 'pauc01']).to_csv(
            save_folder + dataset + '_' + '_metrics_all.csv',
            sep=',', index=False)

def output_DEA_all(views_names, all_test_proteins, proteins_vs, score_vs, logFC_vs, hurdle_proteins,
                      hurdle_score, hurdle_logFC, train_pro, test_pro, train_scores, pred_finals, pred_trains, train_FC,
                      test_FC, de_thr, lgfc_thr, mv_methods, prefix):
    methods = []

    labels = []
    scores = []
    fcs = []

    for v in range(len(views_names)):
        all_test_score_v = mapping_scores(all_test_proteins, proteins_vs[v], [], score_vs[v], [])
        all_logFC_v = mapping_scores(all_test_proteins, proteins_vs[v], [], logFC_vs[v], [])

        methods.append(prefix + '_view' + str(v))

        scores.append(all_test_score_v)
        fcs.append(all_logFC_v)

        lab = np.zeros((len(all_test_score_v), 1))
        lab[np.where(((all_test_score_v >= 1 - de_thr) & (abs(all_logFC_v) >= lgfc_thr)))[0], 0] = 1
        labels.append(lab)

    all_test_score_hurdle = mapping_scores(all_test_proteins, hurdle_proteins, [], hurdle_score, [])
    all_logFC_hurdle = mapping_scores(all_test_proteins, hurdle_proteins, [], hurdle_logFC, [])
    methods.append(prefix + '_hurdle')

    lab_hurdle = np.zeros((len(all_test_score_hurdle), 1))
    lab_hurdle[np.where(((all_test_score_hurdle >= 1 - de_thr) & (abs(all_logFC_hurdle) >= lgfc_thr)))[0], 0] = 1

    labels.append(lab_hurdle)
    scores.append(all_test_score_hurdle)
    fcs.append(all_logFC_hurdle)

    for mth in range(len(pred_finals)):
        all_test_score_incons = mapping_scores(all_test_proteins, train_pro, test_pro, train_scores[:, 0],
                                               pred_finals[mth])
        all_test_score_all = mapping_scores(all_test_proteins, train_pro, test_pro, pred_trains[mth], pred_finals[mth])
        all_test_logFC_incons = mapping_scores(all_test_proteins, train_pro, test_pro, train_FC[:, 0], test_FC[:, 0])
        all_test_logFC_all = mapping_scores(all_test_proteins, train_pro, test_pro, train_FC[:, 0], test_FC[:, 0])

        methods.append(prefix + '_' + mv_methods[mth] + '_incons_rescore')

        methods.append(prefix + '_' + mv_methods[mth] + '_all_rescore')

        lab_incons = np.zeros((len(all_test_score_incons), 1))
        lab_all = np.zeros((len(all_test_score_all), 1))
        lab_incons[
            np.where(((all_test_score_incons >= 1 - de_thr) & (abs(all_test_logFC_incons) >= lgfc_thr)))[0], 0] = 1
        lab_all[
            np.where(((all_test_score_all >= 1 - de_thr) & (abs(all_test_logFC_all) >= lgfc_thr)))[0], 0] = 1

        labels.append(lab_incons)
        scores.append(all_test_score_incons)
        fcs.append(all_test_logFC_incons)

        labels.append(lab_all)
        scores.append(all_test_score_all)
        fcs.append(all_test_logFC_all)

        idx_incons = np.where(((labels[0] == 1) & (lab_hurdle == 1) & (lab_incons == 0)))[0]
        idx_all = np.where(((labels[0] == 1) & (lab_hurdle == 1) & (lab_all == 0)))[0]

        all_test_score_incons_add = copy.deepcopy(all_test_score_incons)
        all_test_score_all_add = copy.deepcopy(all_test_score_all)
        all_test_logFC_incons_add = copy.deepcopy(all_test_logFC_incons)
        all_test_logFC_all_add = copy.deepcopy(all_test_logFC_all)

        all_test_score_incons_add[idx_incons] = np.array(
            [max(scores[0][i], all_test_score_hurdle[i]) for i in idx_incons])
        all_test_logFC_incons_add[idx_incons] = np.array([[fcs[0][i], all_logFC_hurdle[i]][np.where(
            (abs(np.array([fcs[0][i], all_logFC_hurdle[i]])) == max([abs(fcs[0][i]), abs(all_logFC_hurdle[i])])))[0][0]]
                                                          for
                                                          i in idx_incons])

        all_test_score_all_add[idx_all] = np.array([max(scores[0][i], all_test_score_hurdle[i]) for i in idx_all])
        all_test_logFC_all_add[idx_all] = np.array([[fcs[0][i], all_logFC_hurdle[i]][np.where(
            (abs(np.array([fcs[0][i], all_logFC_hurdle[i]])) == max([abs(fcs[0][i]), abs(all_logFC_hurdle[i])])))[0][0]]
                                                    for
                                                    i in idx_all])

        methods.append(prefix + '_' + mv_methods[mth] + '_incons_rescore_add_dlfq_hurdle')
        methods.append(prefix + '_' + mv_methods[mth] + '_all_rescore_add_dlfq_hurdle')

        lab_all_add = np.zeros((len(all_test_score_all_add), 1))
        lab_all_add[
            np.where(((all_test_score_all_add >= 1 - de_thr) & (abs(all_test_logFC_all_add) >= lgfc_thr)))[0], 0] = 1

        labels.append(lab_all_add)
        scores.append(all_test_score_all_add)
        fcs.append(all_test_logFC_all_add)

        lab_all_incons_add = np.zeros((len(all_test_score_incons_add), 1))
        lab_all_incons_add[
            np.where(((all_test_score_incons_add >= 1 - de_thr) & (abs(all_test_logFC_incons_add) >= lgfc_thr)))[0], 0] = 1

        labels.append(lab_all_incons_add)
        scores.append(all_test_score_incons_add)
        fcs.append(all_test_logFC_incons_add)


    return labels, scores, fcs, methods

def MV_DEA_all_mv_new(views, v_scales, res_real_norm_vs, res_hurdle,  lgfc_thr, dethr, cbt, view_paths,
                                   views_names, hc_thr, np_thr, num_tra,
                                      model, seed, op_ty, save_folder, logFC_norm, cca):
    seed_torch()

    isExist = os.path.exists(save_folder)

    if not isExist:
        os.makedirs(save_folder)

    dataset = 'test'
    de_organism = ''
    view_pro = views[0]['Protein'].values
    if 'Organism' not in views[0].keys():
        organism_pro = np.array(['unknow'] * len(view_pro))
    else:
        organism_pro = views[0]['Organism'].values

    fea_all = v_scales[0]
    for i in range(1, len(views_names)):
        fea_all = np.column_stack((fea_all, v_scales[i]))
    fea_all[np.isnan(fea_all)] = 0

    count_zeros = [len(np.where((fea_all[i, :] == 0))[0]) for i in range(len(fea_all[:, 0]))]
    idx_retain = np.where((np.array(count_zeros) <= len(fea_all[0, :]) * 0.8))[0]

    view_pro = view_pro[idx_retain]
    organism_pro = organism_pro[idx_retain]
    fea_all = fea_all[idx_retain, :]
    logFC_all = logFC_norm[idx_retain, :]

    proteins_vs = []
    logFC_vs = []
    qval_vs = []
    score_vs = []

    for i in range(0, len(views_names)):
        proteins_vs.append(res_real_norm_vs[i][1])
        logFC_vs.append(res_real_norm_vs[i][6])
        qval_vs.append(res_real_norm_vs[i][9])
        score_vs.append(res_real_norm_vs[i][2])

    hurdle_proteins = res_hurdle[1]
    hurdle_logFC = res_hurdle[6]
    hurdle_qval = res_hurdle[9]
    hurdle_score = res_hurdle[2]


    (train_pro, train_fea, train_FC, train_scores, train_orga, train_true_label, train_qval_vs, train_logfc_vs) = get_training_proteins_repoduce_mv(proteins_vs,
                                                                                         view_pro, qval_vs,
                                                                                         logFC_vs, fea_all,
                                                                                         logFC_all,
                                                                                         organism_pro, de_organism,
                                                                                         hc_thr, dethr, np_thr,
                                                                                         cbt, hurdle_qval,
                                                                                         hurdle_proteins, views_names)

    # no cca
    (test_pro, test_fea, test_FC, test_orga, test_true_label, score_v_tests, test_score_hurdle,
     logfc_v_tests) = get_test_proteins_reproduce_mv(proteins_vs, view_pro, score_vs, logFC_vs,
                                                  fea_all, logFC_all, hurdle_proteins, hurdle_score, organism_pro,
                                                  de_organism, train_pro)


    if len(train_pro) > 0:
        out_tranin = np.array(
            [list(train_pro), list(train_scores[:, 0]), list(train_FC[:, 0]), list(train_orga), list(train_true_label)]).T
        pd.DataFrame(out_tranin, columns=['Protein', 'scores', 'logFC', 'organism', 'true_label']).to_csv(
            save_folder + dataset + '_' + str(len(views_names)) + 'views_train_pros.csv', sep=',', index=False)

        # train_feas, train_scores, seed, model_type, test_feas_final, trial_num, a, de_thr, diff_ty
        pred_final, pred_train, best_params = optuna_optimize_mv(train_fea, train_scores[:, 0], train_FC[:, 0],
                                                                 seed, cca,
                                                                 test_fea, num_tra, 1, dethr, op_ty, lgfc_thr,
                                                                 views_names)

        all_proteins = np.array(list(train_pro) + list(test_pro))
        all_orga = np.array(list(train_orga) + list(test_orga))
        all_logFC = np.array(list(train_FC) + list(test_FC))

        all_test_proteins, idx = np.unique(all_proteins, return_index=True)

        labels, scores, fcs, methods = output_DEA_all(views_names, all_test_proteins, proteins_vs, score_vs, logFC_vs, hurdle_proteins,
                      hurdle_score, hurdle_logFC, train_pro, test_pro, train_scores, [pred_final], [pred_train], train_FC,
                      test_FC, dethr, lgfc_thr, [cca], 'DEA')

        out_table = np.column_stack((np.array([all_test_proteins]).T, np.array([all_orga[idx]]).T))
        coln = ['Protein', ' Organism']
        for i in range(len(methods)):
            out_table = np.column_stack((out_table, np.array([fcs[i]]).T, 1-np.array([scores[i]]).T, np.array(labels[i])))
            coln.extend([methods[i] + '_' + mth for mth in ['logFC', 'qval', 'label']])

        pd.DataFrame(out_table, columns=[coln]).to_csv(save_folder + 'MCCAKNN_DEA_res_table.csv')

def get_MVDEA_res(view_single, res_single, save_folder, lgfc_thr, dethr, cbt, views_names, hc_thr, np_thr, num_tra,
                                      seed, op_ty, pos_ty, method):
    seed_torch()

    isExist = os.path.exists(save_folder)

    if not isExist:
        os.makedirs(save_folder)

    view_paths = res_single[4]
    de_organism = view_paths['DE_organism']
    dataset = view_paths['dataset']
    view_pro = view_single[0]['Protein'].values
    if 'Organism' in view_single[0].keys():
        organism_pro = view_single[0]['Organism'].values
    else:
        organism_pro = np.array(['unknown'] * len(view_pro))
    fea_all = res_single[0][0]
    for i in range(1, len(views_names)):
        fea_all = np.column_stack((fea_all, res_single[0][i]))
    fea_all[np.isnan(fea_all)] = 0

    count_zeros = [len(np.where((fea_all[i, :] == 0))[0]) for i in range(len(fea_all[:, 0]))]
    idx_retain = np.where((np.array(count_zeros) <= len(fea_all[0, :]) * 0.8))[0]

    view_pro = view_pro[idx_retain]
    #if len(organism_pro)>0:
    organism_pro = organism_pro[idx_retain]
    fea_all = fea_all[idx_retain, :]
    logFC_all = res_single[5][idx_retain, :]

    proteins_vs = []
    logFC_vs = []
    qval_vs = []
    score_vs = []
    T_label_vs = []
    for i in range(0, len(views_names)):
        proteins_vs.append(res_single[1][i][1])
        logFC_vs.append(res_single[1][i][7])
        qval_vs.append(res_single[1][i][10])
        score_vs.append(res_single[1][i][3])
        T_label_vs.append(res_single[1][i][5])

    hurdle_proteins = res_single[2][1]
    hurdle_logFC = res_single[2][7]
    hurdle_qval = res_single[2][10]
    hurdle_score = res_single[2][3]
    hurdle_T_label = res_single[2][5]

    (train_pro, train_fea, train_FC, train_scores, train_orga, train_true_label, train_qval_vs,
     train_logfc_vs) = get_training_proteins_repoduce_mv(proteins_vs,
                                                         view_pro, qval_vs,
                                                         logFC_vs, fea_all,
                                                         logFC_all,
                                                         organism_pro, de_organism,
                                                         hc_thr, dethr, np_thr,
                                                         cbt, hurdle_qval,
                                                         hurdle_proteins, hurdle_logFC, views_names, pos_ty)

    # no cca
    (test_pro, test_fea, test_FC, test_orga, test_true_label, score_v_tests, test_score_hurdle,
     logfc_v_tests) = get_test_proteins_reproduce_mv(proteins_vs, view_pro, score_vs, logFC_vs,
                                                     fea_all, logFC_all, hurdle_proteins, hurdle_score, organism_pro,
                                                     de_organism, train_pro)

    if len(train_pro) > 0:
        out_tranin = np.array(
            [list(train_pro), list(train_scores[:, 0]), list(train_FC[:, 0]), list(train_orga),
             list(train_true_label)]).T
        pd.DataFrame(out_tranin, columns=['Protein', 'organism', 'scores', 'logFC', 'true_label']).to_csv(
            save_folder + dataset + '_' + str(len(views_names)) + 'views_train_pros.csv', sep=',', index=False)

        # train_feas, train_scores, seed, model_type, test_feas_final, trial_num, a, de_thr, diff_ty
        pred_finals, pred_trains, best_paramss = [], [], []
        mv_methods = [method]#['GCCA', 'KNN', 'MLE']
        for model in mv_methods:
            print(dataset)
            pred_final, pred_train, best_params, trains_lat, test_lat = optuna_optimize_mv(train_fea, train_scores[:, 0], train_FC[:, 0],
                                                                     seed, model,
                                                                     test_fea, num_tra, 1, dethr, op_ty, lgfc_thr,
                                                                     views_names)

            pd.DataFrame(np.column_stack((view_pro, organism_pro, fea_all)),
                         columns=['Protein', 'Organism'] + ['v'+str(m) for m in range(len(fea_all[0,:]))]).to_csv(
                save_folder + '/' +
                res_single[4]['dataset'] + 'scaled_expression.csv')

            lats = np.row_stack((trains_lat, test_lat))
            pros = np.array(list(train_pro) + list(test_pro))
            orgas = np.array(list(train_orga) + list(test_orga))

            if len(lats[0,:])>0:
                pd.DataFrame(np.column_stack((pros, orgas, lats)),
                         columns=['Protein', 'Organism'] + ['lat'+str(m) for m in range(len(lats[0,:]))]).to_csv(save_folder + '/' +
                                                                             res_single[4]['dataset'] +
                                                                              model + 'learned_latent.csv')

            pred_finals.append(pred_final)
            pred_trains.append(pred_train)
            best_paramss.append(best_params)

        all_proteins = np.array(list(train_pro) + list(test_pro))
        all_true_label = np.array(list(train_true_label) + list(test_true_label))
        all_test_proteins, idx = np.unique(all_proteins, return_index=True)

        return (
        pred_finals, pred_trains, all_test_proteins, all_true_label[idx], proteins_vs, hurdle_proteins, logFC_all,
        logFC_vs,
        hurdle_logFC, score_vs, hurdle_score, train_pro, test_pro, train_scores, train_FC, test_FC, mv_methods)
    else:
        return []


def MV_DEA_all_reproduce(views_ci, views_union, res_ori, res_ci, res_union,
                         save_folder, lgfc_thr, dethr, cbt, views_names, hc_thr, np_thr, num_tra,
                                      seed, op_ty, pg_ty, view_paths, pos_ty):
    protein_all_can = pd.read_csv(view_paths['all_candidate'], sep='\t', header=0)
    if pg_ty == 'remove':
        protein_all_can = process_proteingroup(protein_all_can, pg_ty)

    protein_all_can_true_lab = np.array(get_true_label(protein_all_can['Protein'].values, view_paths['DE_organism']))

    MVDEA_ci = []
    if len(res_ci) > 0:
        print('ci')
        MVDEA_ci = get_MVDEA_res(views_ci, res_ci, save_folder, lgfc_thr, dethr, cbt, views_names, hc_thr, np_thr, num_tra,
                                          seed, op_ty, pos_ty)

    # print('intersect')
    # MVDEA_intersect = get_MVDEA_res(views_intersect, res_intersect, save_folder, lgfc_thr, dethr, cbt, views_names, hc_thr, np_thr, num_tra,
    #                                   model, seed, op_ty, pos_ty)

    print('union')
    MVDEA_union = get_MVDEA_res(views_union, res_union, save_folder, lgfc_thr, dethr, cbt, views_names,
                                    hc_thr, np_thr, num_tra,
                                    seed, op_ty, pos_ty)

    seed_torch()
    isExist = os.path.exists(save_folder + res_union[4]['dataset'] + '/')
    if not isExist:
        os.makedirs(save_folder+ res_union[4]['dataset'] + '/')

    metrics = []
    methods = []
    all_test_proteins, idx = np.unique(protein_all_can['Protein'].values, return_index=True)
    if (len(MVDEA_ci) > 0):# & (len(MVDEA_intersect) > 0):

        # proteins_vs_ori = []
        # logFC_vs_ori = []
        # qval_vs_ori = []
        # score_vs_ori = []
        # T_label_vs_ori = []
        # for i in range(0, len(views_names)):
        #     proteins_vs_ori.append(res_ori[1][i][1])
        #     logFC_vs_ori.append(res_ori[1][i][7])
        #     qval_vs_ori.append(res_ori[1][i][10])
        #     score_vs_ori.append(res_ori[1][i][3])
        #     T_label_vs_ori.append(res_ori[1][i][5])
        #
        # hurdle_proteins_ori = res_ori[2][1]
        # hurdle_logFC_ori = res_ori[2][7]
        # hurdle_qval_ori = res_ori[2][10]
        # hurdle_score_ori = res_ori[2][3]
        # hurdle_T_label_ori = res_ori[2][5]




        metrics_all, methods_all, out_all_res = metrics_reference(views_names, all_test_proteins, protein_all_can_true_lab[idx], MVDEA_ci[4],
                                                     MVDEA_ci[9], MVDEA_ci[7],
                                                     MVDEA_ci[5],
                                                     MVDEA_ci[10], MVDEA_ci[8], MVDEA_ci[11], MVDEA_ci[12], MVDEA_ci[13],
                                                     MVDEA_ci[0], MVDEA_ci[1],
                                                     MVDEA_ci[14],
                                                     MVDEA_ci[15], dethr, lgfc_thr, MVDEA_ci[16], 'ci')
        out_all_res.to_csv(save_folder + '/' + res_union[4]['dataset'] + '/' + res_union[4]['dataset'] + '_ci_res_all.csv',
            sep=',', index=False)

        metrics.extend(metrics_all)
        methods.extend(methods_all)

        # metrics_all, methods_all, out_all_res = metrics_reference(views_names, all_test_proteins, protein_all_can_true_lab[idx],proteins_vs_ori,
        #                                              score_vs_ori, logFC_vs_ori, hurdle_proteins_ori,
        #                   hurdle_score_ori, hurdle_logFC_ori, [], [], [],
        #                                              [], [],
        #                                              [],
        #                                              [], dethr, lgfc_thr, [], 'ori')
        #
        # out_all_res.to_csv(save_folder + '/' + res_union[4]['dataset'] + '/' + res_union[4]['dataset'] + '_ori_res_all.csv',
        #                    sep=',', index=False)
        #
        # metrics.extend(metrics_all)
        # methods.extend(methods_all)

        # metrics_all, methods_all, out_all_res = metrics_reference(views_names, all_test_proteins, protein_all_can_true_lab[idx], MVDEA_intersect[4],
        #                                              MVDEA_intersect[9], MVDEA_intersect[7],
        #                                              MVDEA_intersect[5],
        #                                              MVDEA_intersect[10], MVDEA_intersect[8], MVDEA_intersect[11], MVDEA_intersect[12], MVDEA_intersect[13],
        #                                              MVDEA_intersect[0], MVDEA_intersect[1],
        #                                              MVDEA_intersect[14],
        #                                              MVDEA_intersect[15], dethr, lgfc_thr, MVDEA_intersect[16], 'intersect')
        #
        # out_all_res.to_csv(save_folder + '/' + res_ori[4]['dataset'] + '/' + res_ori[4]['dataset'] + '_intersect_res_all.csv',
        #                    sep=',', index=False)
        #
        # metrics.extend(metrics_all)
        # methods.extend(methods_all)

    metrics_all, methods_all, out_all_res = metrics_reference(views_names, all_test_proteins,
                                                              protein_all_can_true_lab[idx], MVDEA_union[4],
                                                              MVDEA_union[9], MVDEA_union[7],
                                                              MVDEA_union[5],
                                                              MVDEA_union[10], MVDEA_union[8],
                                                              MVDEA_union[11], MVDEA_union[12],
                                                              MVDEA_union[13],
                                                              MVDEA_union[0], MVDEA_union[1],
                                                              MVDEA_union[14],
                                                              MVDEA_union[15], dethr, lgfc_thr,
                                                              MVDEA_union[16], 'union')

    out_all_res.to_csv(
        save_folder + '/' + res_union[4]['dataset'] + '/' + res_union[4]['dataset'] + '_union_res_all.csv',
        sep=',', index=False)

    metrics.extend(metrics_all)
    methods.extend(methods_all)

    metrics = np.array(metrics)
    methods = np.array(methods)
    metrics_out_all = np.column_stack((methods, metrics))
    pd.DataFrame(np.array(metrics_out_all),
                 columns=['test_type', 'Acc', 'Prec', 'Rec', 'F1', 'F1w', 'Mcc', 'nMcc', 'geomean', 'tn',
                          'fp', 'fn', 'tp', 'pauc001',
                          'pauc005', 'pauc01']).to_csv(
        save_folder + '/' + res_union[4]['dataset']+ '/' + res_union[4]['dataset'] + '_' + '_metrics_all.csv',
        sep=',', index=False)















