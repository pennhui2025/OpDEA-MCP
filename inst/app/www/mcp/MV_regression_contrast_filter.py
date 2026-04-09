import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import os
import pandas as pd
import optuna
import copy
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader
from sklearn import preprocessing
from scipy.stats import norm
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
from scipy.stats import chi2
from sklearn.decomposition import PCA
from statsmodels.stats.multitest import fdrcorrection as fdr
from sklearn.metrics import accuracy_score as acc, recall_score as recall, precision_score as precision, f1_score as f1, matthews_corrcoef as mcc
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
import seaborn as sns
import umap
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score,adjusted_rand_score,normalized_mutual_info_score
import time
import csv
from MV_regression_DP import get_MVDEA_res_DP, metrics_reference, label_pro, get_training_proteins_repoduce_mv
from sklearn.model_selection import KFold, StratifiedKFold
from Utils import get_datasets_reproduce_mv
from itertools import combinations

def seed_torch(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)

def cal_metrics1(score, q_thr, FC, fc_thr, true_lab):
    qval = 1-score
    y_true = true_lab
    y_pred = (qval <= q_thr) & (abs(FC[:,0])>=fc_thr)
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

def cal_metrics2(score, score_thr, FC, fc_thr, true_lab):
    #qval = 1-score
    y_true = true_lab
    y_pred = (score >= score_thr) & (abs(FC[:,0]) >= fc_thr)
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

    return [Acc, Prec, Rec, F1, F1w, Mcc, nMcc, geomean, tn, fp, fn, tp, pauc001, pauc005, pauc01], y_pred

class AE_old(nn.Module):
    def __init__(self, in_dim, h_dim1, lat_dim):
        super(AE_old, self).__init__()

        # encoder
        self.enc_1 = nn.Linear(in_dim, h_dim1)
        self.mid_1 = nn.Linear(h_dim1, lat_dim)

        # decoder
        self.dec_1 = nn.Linear(lat_dim, h_dim1)
        self.out = nn.Linear(h_dim1, in_dim)
        self.sigmod = nn.Sigmoid()
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                torch.nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # encoder
        enc_h1 = F.relu(self.enc_1(x))


        lat = self.mid_1(enc_h1)
        # decoder
        dec_h1 = F.relu(self.dec_1(lat))

        x_rec = self.out(dec_h1)

        return x_rec, lat

class AE(nn.Module):
    def __init__(self, in_dim, lat_dim):
        super(AE, self).__init__()
        # encoder
        self.enc = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            # nn.Dropout(p=0.1),
            nn.ReLU(),
            nn.Linear(1024, lat_dim),
        )
        # decoder
        self.dec = nn.Sequential(
            nn.Linear(lat_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            # nn.Dropout(p=0.1),
            nn.Linear(256, in_dim)
        )

    def forward(self, x):
        # encoder
        lat = self.enc(x)
        # decoder
        x_rec = self.dec(lat)

        return x_rec, lat

class DNN(nn.Module):
    def __init__(self, in_dim, h_dim, lat_dim):
        super(DNN, self).__init__()

        # encoder
        self.enc = nn.Linear(in_dim, h_dim)
        self.mid = nn.Linear(h_dim, lat_dim)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                torch.nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # encoder
        enc_h = F.relu(self.enc(x))

        lat = self.mid(enc_h)

        return lat


class ResidualAttentionBlock(nn.Module):
    def __init__(self, in_dim, att_dim):
        super().__init__()
        self.q = nn.Linear(in_dim, att_dim)
        self.k = nn.Linear(in_dim, att_dim)
        self.v = nn.Linear(in_dim, att_dim)
        self.out = nn.Linear(att_dim, in_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)

        attention = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(k.size(-1))
        attention = self.relu(attention)
        attention = self.softmax(attention)

        res = self.relu(self.out(torch.matmul(attention, v)))
        return self.dropout(res + x)

class SparseAE_enc(nn.Module):
    """带稀疏约束和特征选择的自编码器"""

    def __init__(self, in_dim, h_dim, lat_dim, dropout=0.2):
        super().__init__()
        # 编码器
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, h_dim),
            nn.BatchNorm1d(h_dim),
            nn.GELU(),
            ResidualAttentionBlock(h_dim, h_dim),
            nn.Dropout(dropout),
            nn.Linear(h_dim, lat_dim)
        )

        # 特征选择门控
        self.selector = nn.Sequential(
            nn.Linear(lat_dim, lat_dim),
            nn.Sigmoid())

    def get_combined_weights(self):
        """获取跨层权重组合"""
        W1 = self.encoder[0].weight  # [h, in]
        W2 = self.encoder[-1].weight  # [lat, h]
        return torch.mm(W2, W1)  # [lat, in]

    def forward(self, x):
        lat = self.encoder(x)
        gate = self.selector(lat)  # 特征选择权重

        # 更新特征重要性
        with torch.no_grad():
            combined_weights = self.get_combined_weights().T  # [in, lat]
            gate_weights = gate.abs().mean(dim=0)  # [lat]
            feature_importance = torch.mv(combined_weights.abs(), gate_weights)

        return lat*gate, gate, feature_importance

class SparseAE_dec(nn.Module):
    """带稀疏约束和特征选择的自编码器"""

    def __init__(self, in_dim, h_dim, lat_dim, dropout=0.2):
        super().__init__()

        # 解码器
        self.decoder = nn.Sequential(
            nn.Linear(lat_dim, h_dim),
            nn.BatchNorm1d(h_dim),
            nn.GELU(),
            ResidualAttentionBlock(h_dim, h_dim),
            nn.Dropout(dropout),
            nn.Linear(h_dim, in_dim))

    def forward(self, x):
        x_rec = self.decoder(x)  # 门控潜在表示
        return x_rec

class MultiViewCrossAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, num_views):
        """
        多视角交叉注意力模块（优化版）
        :param embed_dim: 每个视角特征的嵌入维度
        :param num_heads: 注意力头的数量
        :param num_views: 视角的数量
        """
        super(MultiViewCrossAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_views = num_views
        self.head_dim = embed_dim // num_heads
        self.cross_matrix = None
        # 检查 embed_dim 是否可以被 num_heads 整除
        assert self.head_dim * num_heads == embed_dim, "embed_dim 必须能被 num_heads 整除"

        # 线性变换层，用于生成 Q, K, V
        self.query_proj = nn.Linear(embed_dim, embed_dim)
        self.key_proj = nn.Linear(embed_dim, embed_dim)
        self.value_proj = nn.Linear(embed_dim, embed_dim)

        # 输出线性变换
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        # 动态权重生成
        self.weight_fc = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, num_views),
            nn.Softmax(dim=-1)
        )

        # 层归一化
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        # 缩放因子
        self.scale = self.head_dim ** -0.5
    def _record_interaction(self, query_idx, key_idx, attn_weights):
        """记录跨视角交互"""
        interaction = attn_weights.mean().item()
        self.cross_matrix[query_idx, key_idx] += interaction

    def _calculate_head_importance(self, attn_weights):
        """计算注意力头重要性"""
        return torch.norm(attn_weights, p=2, dim=[-2, -1])  # [H]

    def forward(self, views):
        """
        :param views: 多视角特征列表，每个视角的特征形状为 [batch_size, embed_dim]
        :return: 融合后的特征，形状为 [batch_size, embed_dim]
        """
        batch_size = views[0].size(0)

        # 残差连接：保存输入特征
        residual = torch.stack(views, dim=0).mean(dim=0)  # [batch_size, embed_dim]

        # 动态权重计算
        view_weights = self.weight_fc(residual)  # [batch_size, num_views]
        view_weights = view_weights.unsqueeze(-1)  # [batch_size, num_views, 1]

        # 初始化输出
        outputs = []
        device = views[0].device
        self.cross_matrix = torch.zeros(self.num_views, self.num_views, device=device)

        head_weights = []
        cross_matrix = None
        head_importance = None

        # 对每个视角计算交叉注意力
        for i in range(self.num_views):
            # 当前视角作为 Query
            query = self.query_proj(views[i]).view(batch_size, self.num_heads, self.head_dim)  # [batch_size, num_heads, head_dim]

            # 其他视角作为 Key 和 Value
            keys = []
            values = []
            for j in range(self.num_views):
                if j != i:  # 排除自身视角
                    key = self.key_proj(views[j]).view(batch_size, self.num_heads, self.head_dim)  # [batch_size, num_heads, head_dim]
                    value = self.value_proj(views[j]).view(batch_size, self.num_heads, self.head_dim)  # [batch_size, num_heads, head_dim]
                    keys.append(key)
                    values.append(value)

                    # 记录交互强度
                    attn_scores0 = torch.matmul(query, key.transpose(-2, -1)) * (self.head_dim ** -0.5)
                    self._record_interaction(i, j, attn_scores0)
                    head_weights.append(self._calculate_head_importance(attn_scores0))

            # 拼接所有 Key 和 Value
            keys = torch.cat(keys, dim=1)  # [batch_size, (num_views-1)*num_heads, head_dim]
            values = torch.cat(values, dim=1)  # [batch_size, (num_views-1)*num_heads, head_dim]

            # 计算注意力分数
            attn_scores = torch.matmul(query, keys.transpose(-2, -1)) * self.scale  # [batch_size, num_heads, (num_views-1)*num_heads]
            attn_weights = F.softmax(attn_scores, dim=-1)

            # 加权求和
            attn_output = torch.matmul(attn_weights, values)  # [batch_size, num_heads, head_dim]
            attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1)  # [batch_size, embed_dim]

            # 线性变换
            attn_output = self.out_proj(attn_output)
            outputs.append(attn_output)

        # 动态权重融合
        outputs = torch.stack(outputs, dim=1)  # [batch_size, num_views, embed_dim]
        fused_output = (outputs * view_weights).sum(dim=1)  # [batch_size, embed_dim]

        # 残差连接 + 层归一化
        fused_output = self.norm1(fused_output + residual)

        # 最终层归一化
        output = self.norm2(fused_output)

        # 记录视角贡献
        view_contributions = view_weights.mean(dim=0)  # [num_views]
        cross_matrix = self.cross_matrix
        head_importance = torch.mean(torch.stack(head_weights), dim=0)
        return output, view_contributions, cross_matrix, head_importance

class MultilevelSupConLoss(nn.Module):
    """多层次监督对比损失"""

    def __init__(self, temp=0.1, alpha=0.5, input_dim=None):
        super().__init__()
        self.temp = temp
        self.alpha = alpha  # 视角间对比权重
        # 定义投影头
        if input_dim is not None:
            self.proj_head = nn.Sequential(
                nn.Linear(input_dim, input_dim),
                nn.ReLU()
            )
        else:
            self.proj_head = nn.Identity()  # 默认恒等变换

    def forward(self, features, labels):
        """
        features: list of [bs, lat_dim] from each view
        labels: [bs]
        """
        device = features[0].device
        bs = features[0].size(0)

        # 视角内对比
        intra_loss = 0
        for view_feat in features:
            # 投影到对比空间
            z = F.normalize(self.proj_head(view_feat), dim=1)

            # 计算相似度矩阵
            sim = torch.mm(z, z.T) / self.temp

            # 构建标签mask
            labels = labels.view(-1, 1)
            mask = torch.eq(labels, labels.T).float().to(device)
            try:
                logits_mask = torch.ones_like(mask) - torch.eye(bs).to(device)
            except:
                a = 0

            # 计算log_prob
            exp_logits = torch.exp(sim) * logits_mask
            log_prob = sim - torch.log(exp_logits.sum(dim=1, keepdim=True))

            # 计算损失
            mean_log_prob = (log_prob * mask).sum(1) / mask.sum(1)
            intra_loss += -mean_log_prob.mean()

        # 视角间对比
        inter_loss = 0
        for i in range(len(features)):
            for j in range(i + 1, len(features)):
                z_i = F.normalize(self.proj_head(features[i]), dim=1)
                z_j = F.normalize(self.proj_head(features[j]), dim=1)

                sim = torch.mm(z_i, z_j.T) / self.temp
                mask = torch.eq(labels, labels.T).float().to(device)

                exp_logits = torch.exp(sim)
                log_prob = sim - torch.log(exp_logits.sum(dim=1, keepdim=True))

                mean_log_prob = (log_prob * mask).sum(1) / mask.sum(1)
                inter_loss += -mean_log_prob.mean()

        total_loss = (intra_loss + self.alpha * inter_loss) / len(features)
        return total_loss

class MV_cons(nn.Module):
    def __init__(self, in_dim1, lat_dim1, h_dim2, class_num, nview):#, k, temp):
        super(MV_cons, self).__init__()
        self.lat_dim1 = lat_dim1

        self.enc_list = nn.ModuleList([SparseAE_enc(in_dim1, h_dim2, lat_dim1) for i in range(nview)])
        self.dec_list = nn.ModuleList([SparseAE_dec(in_dim1, h_dim2, lat_dim1) for i in range(nview)])

        self.pred_pro = nn.Sequential(
            nn.Linear(lat_dim1, lat_dim1//2),
            nn.BatchNorm1d(lat_dim1//2),
            nn.GELU(),
            ResidualAttentionBlock(lat_dim1//2, lat_dim1//2),
            nn.Dropout(0.3),
            nn.Linear(lat_dim1//2, class_num))
        self.csatt = MultiViewCrossAttention(embed_dim=lat_dim1, num_heads=lat_dim1, num_views=nview)
        # self.gl_attn = ExternalAttention(d_model=lat_dim1, S=s)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, pro_labs):

        recon_x = []
        lat_con_1 = []
        gates = []
        feature_importances = []
        for i, enc in enumerate(self.enc_list):
            lat1, gate, feature_importance = enc(x[i])
            # lat1 = self.gl_attn(lat1)
            lat_con_1.append(lat1)
            gates.append(gate)
            feature_importances.append(feature_importance)

        fused, view_contributions, cross_matrix, head_importance = self.csatt(lat_con_1)

        for i, dec in enumerate(self.dec_list):
            x_rec = dec(fused)
            recon_x.append(x_rec)
            
        pro_logits = self.pred_pro(fused)
        con_loss1 = MultilevelSupConLoss()([lat_con_1[t][~torch.tensor(pro_labs == -1)] for t in range(len(lat_con_1))], pro_labs[~torch.tensor(pro_labs == -1)])

        pro_prob = self.softmax(pro_logits)
        pro_lab = torch.argmax(pro_prob, dim=1)

        return recon_x, fused, pro_lab, pro_logits, pro_prob, con_loss1, feature_importances, view_contributions, cross_matrix, head_importance

def visualize_contributions(view_names, feature_names, feature_importances, view_contributionss, cross_matrixs, head_importances, save_prefix):
    """
    综合可视化分析
    :param top_k: 显示topk重要特征
    """
    feature_importancess = []
    for i in range(len(feature_importances)):
        if i == 0:
            feature_importancess = feature_importances[i]
        else:
            for j in range(len(feature_importances[0])):
                feature_importancess[j] += feature_importances[i][j]

    feature_importancess = [(fea.cpu().numpy()/len(feature_importances))/sum(fea.cpu().numpy()/len(feature_importances)) for fea in feature_importancess]

    view_contributionss = [v.cpu().numpy() for v in view_contributionss]
    cross_matrixs = [v.cpu().numpy() for v in cross_matrixs]
    # 获取数据
    feat_imps = feature_importancess
    view_weights = np.mean(view_contributionss, axis=0)
    cross_mat = np.mean(cross_matrixs, axis=0)


    pd.DataFrame(view_weights.T, columns=view_names).to_csv(save_prefix + '_view_weights.csv')
    pd.DataFrame(cross_mat, columns=view_names).to_csv(save_prefix + '_view_cross_matrix.csv')

    # 1. 视角贡献
    plt.figure(figsize=(8, 4))
    plt.bar(np.array(view_names), view_weights[:, 0])
    plt.title("View Contribution Weights")
    plt.ylabel("Contribution Score")
    plt.savefig(save_prefix + "_View Contribution.svg", format='svg')
    plt.savefig(save_prefix + "_View Contribution.pdf", format='pdf')
    #plt.show()

    # 2. 特征重要性
    for v_idx, (imp, names) in enumerate(zip(feat_imps, feature_names)):
        pd.DataFrame(imp.reshape(1, -1), columns=list(names)).to_csv(save_prefix + np.array(view_names)[v_idx] + '_fea_imp.csv')

        top_k = len(names)
        sorted_idx = np.argsort(-imp)[:top_k]

        plt.figure(figsize=(12, 4))
        plt.barh(range(min(top_k, len(sorted_idx))), imp[sorted_idx][::-1], tick_label=np.array(names)[sorted_idx][::-1])
        plt.title(f"Top {top_k} Features - {np.array(view_names)[v_idx]}")
        plt.xlabel("Importance Score")
        plt.tight_layout()
        plt.savefig(save_prefix + np.array(view_names)[v_idx] + "_feature_importance.svg", format='svg')
        plt.savefig(save_prefix + np.array(view_names)[v_idx] + "_feature_importance.pdf", format='pdf')
        #plt.show()

    # 3. 视角交互
    plt.figure(figsize=(10, 8))
    sns.heatmap(cross_mat,
                annot=True, fmt=".2f",
                xticklabels=np.array(view_names),
                yticklabels=np.array(view_names),
                cmap="Blues")
    plt.title("Cross-View Interaction Matrix")
    plt.xlabel("Key Views")
    plt.ylabel("Query Views")
    plt.savefig(save_prefix + "_Cross-View.svg", format='svg')
    plt.savefig(save_prefix + "_Cross-View.pdf", format='pdf')
    #plt.show()

class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                torch.nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        # modified to handle edge cases when there is no positive pair
        # for an anchor point.
        # Edge case e.g.:-
        # features of shape: [4,1,...]
        # labels:            [0,1,1,2]
        # loss before mean:  [nan, ..., ..., nan]
        mask_pos_pairs = mask.sum(1)
        mask_pos_pairs = torch.where(mask_pos_pairs < 1e-6, 1, mask_pos_pairs)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_pos_pairs

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss

class ExternalAttention(nn.Module):

    def __init__(self, d_model,S=64):
        super().__init__()
        self.mk=nn.Linear(d_model,S,bias=False)
        self.mv=nn.Linear(S,d_model,bias=False)
        self.softmax=nn.Softmax(dim=1)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                torch.nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)

    def forward(self, queries):
        attn=self.mk(queries) #bs,n,S
        attn=self.softmax(attn) #bs,n,S
        attn=attn/torch.sum(attn,dim=1,keepdim=True) #bs,n,S
        out=self.mv(attn) #bs,n,d_model

        return out

class local_global_attention(nn.Module):
    def __init__(self, d_model, step=3, weight=0.5, S=64):
        super().__init__()
        self.init_weights()
        self.step = step
        self.glo_ext_attn = ExternalAttention(d_model=d_model, S=S)
        self.loc_ext_attn = ExternalAttention(d_model=step, S=S)
        self.weight = weight

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                torch.nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)

    def forward(self, queries):
        seq_len = queries.shape[1]
        local_output = torch.zeros_like(queries)
        for i in range(0, seq_len, self.step):
            end_idx = min(i + self.step, seq_len)
            if end_idx-i!=self.step:
                local_chunk = queries[:, (seq_len-self.step):seq_len]
                local_out = self.loc_ext_attn(local_chunk)
                end_idx_i_1 = i
                local_output[:, (seq_len-self.step):end_idx_i_1] = (local_out[:, 0:(end_idx_i_1-(seq_len-self.step))] + local_output[:, (seq_len-self.step):end_idx_i_1])/2
                local_output[:, end_idx_i_1:end_idx] = local_out[:, (end_idx_i_1-(seq_len-self.step)):len(local_out[0, :])]
            else:
                local_chunk = queries[:, i:end_idx]
                local_out = self.loc_ext_attn(local_chunk)
                local_output[:, i:end_idx] = local_out


            # Global Attention
        global_output = self.glo_ext_attn(queries)

        # Combine local and global features (e.g., weighted sum or concatenation)
        combined_output = self.weight * local_output + (1-self.weight) * global_output
        return combined_output

class MixedMissingMechanism(nn.Module):
    def __init__(self, ratios=(0.6, 0.2, 0.2), missing_rate=0.1, temp=0.7):
        """
        :param ratios: (MNAR, MAR, MCAR)的比例
        :param missing_rate: 总体缺失率
        :param temp: Gumbel-Softmax温度参数
        """
        super().__init__()
        self.ratios = torch.tensor(ratios)
        self.missing_rate = missing_rate
        self.temp = temp
        self.alpha = nn.Parameter(torch.randn(3))  # 三种缺失机制的可学习权重

    def forward(self, x):
        bs, dim = x.shape

        # 生成机制选择概率
        mechanism_probs = F.softmax(self.alpha.to(x.device) * self.ratios.to(x.device), dim=-1)

        # 生成机制选择矩阵
        mask_mechanism = F.gumbel_softmax(torch.log(mechanism_probs).expand(bs, -1),
                                          tau=self.temp, hard=True)

        # 生成各机制掩码
        mnar_mask = self._mnar_mask(x)
        mar_mask = self._mar_mask(x)
        mcar_mask = self._mcar_mask(x)

        # 组合最终掩码
        combined_mask = (mask_mechanism[:, 0].unsqueeze(-1) * mnar_mask +
                         mask_mechanism[:, 1].unsqueeze(-1) * mar_mask +
                         mask_mechanism[:, 2].unsqueeze(-1) * mcar_mask)

        return x * combined_mask

    def _mnar_mask(self, x):
        mask_prob = torch.sigmoid((1-x) * 2) * self.missing_rate
        return (torch.rand_like(x) > mask_prob).float()

    def _mar_mask(self, x):
        weights = F.gumbel_softmax(x.abs(), dim=-1, tau=self.temp)
        mask_prob = weights.mean(dim=1, keepdim=True) * self.missing_rate
        return (torch.rand_like(x) > mask_prob).float()

    def _mcar_mask(self, x):
        return (torch.rand_like(x) > self.missing_rate).float()


class ExpressionAugmentor:
    """
    表达矩阵数据增强器，新增缩放功能：
    1. 随机高斯噪声      2. 行列随机变换
    3. 矩阵翻转         4. 随机置零
    5. 随机缩放（新增）   6. 组合增强
    """

    def __init__(self,
                 noise_std=0.1,
                 row_shuffle_prob=0.5,
                 col_shuffle_prob=0.5,
                 flip_prob=0.5,
                 dropout_prob=0.2,
                 scale_range=(0.8, 1.2),  # 新增缩放参数
                 scale_prob=0.5):  # 缩放触发概率
        # 噪声参数
        self.noise_std = noise_std
        # 行列变换
        self.row_shuffle_prob = row_shuffle_prob
        self.col_shuffle_prob = col_shuffle_prob
        # 翻转参数
        self.flip_prob = flip_prob
        # 随机置零
        self.dropout_prob = dropout_prob
        # 缩放参数（新增）
        self.scale_range = scale_range
        self.scale_prob = scale_prob

    def __call__(self, x):
        # 类型检查与转换
        if not isinstance(x, torch.Tensor):
            x = torch.from_numpy(x)

        x_aug = x.clone()

        # === 新增缩放操作 ===
        if torch.rand(1) < self.scale_prob:
            # 生成随机缩放因子
            scale_factor = torch.empty(1).uniform_(*self.scale_range)
            x_aug = x_aug * scale_factor

        # === 原有增强操作 ===
        # 1. 高斯噪声
        if self.noise_std > 0:
            noise = torch.randn_like(x_aug) * self.noise_std
            x_aug += noise

        # 2. 行随机变换
        if torch.rand(1) < self.row_shuffle_prob:
            row_idx = torch.randperm(x_aug.size(0))
            x_aug = x_aug[row_idx]

        # 3. 列随机变换
        if torch.rand(1) < self.col_shuffle_prob:
            col_idx = torch.randperm(x_aug.size(1))
            x_aug = x_aug[:, col_idx]

        # 4. 矩阵翻转
        if torch.rand(1) < self.flip_prob:
            if torch.rand(1) < 0.5:
                x_aug = torch.flip(x_aug, dims=[0])  # 垂直
            else:
                x_aug = torch.flip(x_aug, dims=[1])  # 水平

        # 5. 随机置零
        if self.dropout_prob > 0:
            mask = torch.rand_like(x_aug) > self.dropout_prob
            x_aug = x_aug * mask.float()

        return x_aug


class ExpressionAugmentor:
    """
    表达矩阵数据增强器，新增缩放功能：
    1. 随机高斯噪声      2. 行随机变换
    3. 矩阵翻转         4. 随机置零
    5. 随机缩放（新增）   6. 组合增强
    """

    def __init__(self,
                 noise_std=0.01,
                 row_shuffle_prob=0.6,
                 dropout_prob=0.1
                 ):
        # 噪声参数
        self.noise_std = noise_std
        # 行列变换
        self.row_shuffle_prob = row_shuffle_prob
        # 随机置零
        self.dropout_prob = dropout_prob

    def __call__(self, x):
        # 类型检查与转换
        if not isinstance(x, torch.Tensor):
            x = torch.from_numpy(x)

        x_aug = x.clone()

        # === 原有增强操作 ===
        # 1. 高斯噪声
        if self.noise_std > 0:
            noise = torch.randn_like(x_aug) * self.noise_std
            x_aug += noise

        # 2. 行随机变换
        if torch.rand(1) < self.row_shuffle_prob:
            row_idx = torch.randperm(x_aug.size(1))
            x_aug = x_aug[:, row_idx]

        # 5. 随机置零
        if self.dropout_prob > 0:
            mask = torch.rand_like(x_aug) > self.dropout_prob
            x_aug = x_aug * mask.float()

        return x_aug

class ExpressionAugmentor_scale:

    def __init__(self,
                 scale_range=(0.8, 1.2),  # 新增缩放参数
                 scale_prob=0.4):  # 缩放触发概率

        # 缩放参数（新增）
        self.scale_range = scale_range
        self.scale_prob = scale_prob

    def __call__(self, x):
        # 类型检查与转换
        if not isinstance(x, torch.Tensor):
            x = torch.from_numpy(x)

        x_aug = x.clone()

        # === 新增缩放操作 ===
        if torch.rand(1) < self.scale_prob:
            # 生成随机缩放因子
            scale_factor = torch.empty(1).uniform_(*self.scale_range).to(x.device)
            x_aug = x_aug * scale_factor

        return x_aug
def aug_proteomics(data, pro_lab, device, sample_class):

    class_counts = torch.bincount(torch.tensor(pro_lab).long().to(device)[pro_lab != -1])
    uni_cls = np.unique(pro_lab[pro_lab != -1])
    max_count = class_counts.max()
    min_count = class_counts.min()
    cls = []

    # 初始化增强器（包含缩放参数）
    augmentor = ExpressionAugmentor(
        noise_std=0.01,
        row_shuffle_prob=1,
        dropout_prob=0.05
    )

    augmentor_scale = ExpressionAugmentor_scale(
        scale_range=(0.8, 1.2),  # 缩放范围扩大
        scale_prob=0.4  # 提高缩放概率
    )

    for c in uni_cls:
        if len(np.where((pro_lab == c))[0]) == min_count:
            cls = c
    num_to_generate = max_count - min_count
    gen_labels = torch.full((num_to_generate,), cls).to(device)

    ori_data = [torch.from_numpy(data[i][np.where((pro_lab==c))[0], :]).to(device).float() for i in range(len(data))]
    ori_lab = pro_lab[pro_lab==c]

    gen_index = torch.randint(0, len(ori_lab), (num_to_generate, 1)).squeeze()
    ref_data = [ori_data[i][gen_index].to(device) for i in range(len(ori_data))]

    gen_datas = []
    for i in range(len(data)):
        uni_g = np.unique(sample_class)
        gen_d = []

        for g in range(len(uni_g)):
            idx = np.where((sample_class == g))[0]

            try:
                if len(ref_data[i].shape) == 1:
                    rd = torch.from_numpy(np.array([ref_data[i].cpu()])).to(device).float()
                    gen_data = augmentor(rd[:, idx])
                else:
                    gen_data = augmentor(ref_data[i][:, idx])
            except:
                m = 0
            gen_d.append(gen_data)
        gen_ds = torch.cat(gen_d, dim=1)
        gen_ds_scale = augmentor_scale(gen_ds)

        gen_ds_scale[gen_ds_scale < 0] = torch.tensor(0)
        gen_ds_scale[gen_ds_scale > 1] = torch.tensor(1)
        gen_datas.append(gen_ds_scale)

    return gen_datas, gen_labels.long()

def train(view_data, sample_class, pro_lab, p_score, q_score, param, device, score_type, preds):
    seed_torch()
    nview = len(view_data)
    class_num = len(np.unique(pro_lab[np.where((pro_lab != -1))[0]]))
    in_dim1 = len(view_data[0][0, :])

    lat_dim1 = param['ld_ae']
    h_dim2 = param['h2']
    #s = param['s']

    model = MV_cons(in_dim1, lat_dim1, h_dim2, class_num, nview).to(
        device)

    view_data_tensor = [torch.from_numpy(view_data[i]).to(device).float() for i in range(len(view_data))]
    Train_data = TensorDataset(*view_data_tensor, torch.from_numpy(pro_lab[:, 0]).to(device).long())

    train_loader = DataLoader(Train_data, batch_size=param['bs'], shuffle=True, drop_last=True)
    optimizer = Adam(model.parameters(), lr=param['lr'], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

    for epoch in range(100):
        model.train()
        loss = 0
        for i, data in enumerate(train_loader):
            pro_labs = data[-1]
            del data[-1]

            has_label = (pro_labs != -1)
            x_labeled = [data[ind][has_label] for ind in range(len(data))]
            gen_datas, gen_lab = aug_proteomics([x_labeled[ind].detach().cpu().numpy() for ind in range(len(data))], pro_labs[has_label].detach().cpu().numpy(), device, sample_class)
            data_aug = [torch.cat([data[ind], gen_datas[ind]]) for ind in range(len(data))]
            labs_aug = torch.cat([pro_labs, gen_lab])

            recon_x, lat_con_1, lab_pred, pro_logits, pro_prob, con_loss1, _, _, _, _ = model(data_aug, labs_aug)

            recon_loss = 0
            for j in range(len(view_data)):
                mask1 = torch.tensor(data_aug[j] == 0)
                recon_loss1 = nn.MSELoss(reduction='mean')(recon_x[j][~mask1], data_aug[j][~mask1])
                recon_loss+=recon_loss1

            recon_loss = torch.sum(recon_loss)

            mask2 = torch.tensor(labs_aug.clone().detach() == -1)

            # pro_loss = nn.MSELoss(reduction='mean')(pro_score[~mask2], pro_prob[~mask2])
            pro_loss = nn.CrossEntropyLoss()(pro_logits[~mask2], labs_aug[~mask2]) #weight=class_weights

            if torch.isnan(con_loss1):
                con_loss1 = torch.tensor(10e6).to(device).float()

            loss = recon_loss + pro_loss + con_loss1

            cls_loss = 0

            print('recon_loss='+str(recon_loss) + '|| cls_loss=' + str(cls_loss) +
                  '|| cont_loss_pro='+str(con_loss1) + "|| pro_loss=" + str(pro_loss) + "|| total loss=" + str(loss))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()



    model.eval()

    with torch.no_grad():

        pro_probs = []
        recon_xs = []
        lat_cons = []
        pred_labs = []
        logits = []
        feature_importances, view_contributionss, cross_matrixs, head_importances = [], [], [], []

        step = 500
        import math
        runs = int(math.ceil(len(pro_lab) / step))
        for r in range(runs):
            start = r * step
            end = (r + 1) * step
            if end > len(pro_lab):
                end = len(pro_lab)

            test_view_data_tensor = [view_data_tensor[i][start: end] for i in range(len(view_data_tensor))]
            test_pro_lab = pro_lab[start: end]

            recon_x, lat_con_1, lab_pred, pro_logits, pro_prob, con_loss1, feature_importance, view_contributions, cross_matrix, head_importance = model(test_view_data_tensor,
                                                                                  torch.from_numpy(test_pro_lab[:, 0]).to(device).long())

            feature_importances.append(feature_importance)
            view_contributionss.append(view_contributions)
            cross_matrixs.append(cross_matrix)
            head_importances.append(head_importance)

            if r == 0:
                pro_probs = pro_prob
                recon_xs = recon_x
                lat_cons = lat_con_1
                pred_labs = lab_pred
                logits = pro_logits

            else:
                recon_xs = [torch.cat([recon_xs[i], recon_x[i]], dim=0) for i in range(len(recon_xs))]
                lat_cons = torch.cat([lat_cons, lat_con_1], dim=0)
                pro_probs = torch.cat([pro_probs, pro_prob], dim=0)
                pred_labs = torch.cat([pred_labs, lab_pred], dim=0)
                logits = torch.cat([logits, pro_logits], dim=0)

        recon_loss = 0
        for j in range(len(view_data)):
            mask1 = torch.tensor(view_data[j] == 0)
            recon_loss1 = nn.MSELoss(reduction='mean')(recon_xs[j][~mask1], view_data_tensor[j][~mask1])
            recon_loss += recon_loss1

        recon_loss = torch.sum(recon_loss)


        if score_type == 'p':
            pro_score = torch.from_numpy(p_score).to(device).float()
        elif score_type == 'q':
            pro_score = torch.from_numpy(q_score).to(device).float()

        mask2 = torch.tensor(pro_lab[:, 0] == -1)

        # pro_loss = nn.MSELoss(reduction='mean')(pro_score[~mask2], pro_prob[~mask2])
        pro_loss = nn.CrossEntropyLoss()(logits[~mask2], torch.from_numpy(pro_lab[:, 0]).to(device).long()[~mask2])

        pred_score = 1 - pro_probs.detach().cpu().numpy()[:, 0]
        pred_score[np.isnan(pred_score)] = 0
        # pred_score = pro_prob.detach().cpu().numpy()
        auc1 = roc_auc_score(pro_lab[:, 0][np.where((pro_lab[:, 0] != -1))[0]],
                             pred_score[np.where((pro_lab[:, 0] != -1))[0]])

        mcc1 = mcc(pro_lab[:, 0][np.where((pro_lab[:, 0] != -1))[0]],pred_labs.detach().cpu().numpy()[np.where((pro_lab[:, 0] != -1))[0]])

        loss = recon_loss + pro_loss + con_loss1# + con_loss2 + cls_loss
        print('test recon_loss=' + str(recon_loss) + '|| test auc=' + str(auc1) +
              '|| test cont_loss=' + str(con_loss1) + "|| test pro_loss=" + str(pro_loss) + "|| test total loss=" + str(loss))

    scheduler.step()
    preds['aucs'].append(auc1)
    preds['probs'].append(pred_score)
    preds['labs'].append(pred_labs.detach().cpu().numpy())
    preds['mccs'].append(mcc1)
    return model, loss, pro_loss, recon_xs, lat_cons, pro_probs, mcc1, feature_importances, view_contributionss, cross_matrixs, head_importances

def cv_train(view_data, sample_class, pro_lab, p_score, q_score, param, device, score_type):
    train_idx = np.where((pro_lab != -1))[0]

    train_view_data = [view_data[i][train_idx, :] for i in range(len(view_data))]
    train_pro_lab = pro_lab[train_idx, :]
    seed_torch(2024)
    # logo = StratifiedKFold(n_splits=3, shuffle=True, random_state=2024)
    logo = StratifiedKFold(n_splits=3, shuffle=True, random_state=2024)
    aucs = 0
    n = 1
    for train_index, test_index in logo.split(train_view_data[0], train_pro_lab):
        # tr_view_data = [train_view_data[i][train_index, :] for i in range(len(view_data))]
        # te_view_data = [train_view_data[i][test_index, :] for i in range(len(view_data))]
        # tr_pro_lab, te_pro_lab = train_pro_lab[train_index], train_pro_lab[test_index]
        tr_pro_lab = copy.deepcopy(train_pro_lab)
        tr_pro_lab[test_index, :] = -1

        model, loss, pro_loss, recon_x, lat_con_1, pro_pred, auc = train(train_view_data,
                                                                                                        sample_class,
                                                                                                        tr_pro_lab,
                                                                                                        p_score,
                                                                                                        q_score, param,
                                                                                                        device,
                                                                                                        score_type)
        aucs += auc
        n += 1

    aucs = aucs/n
    return aucs


def objective_para(trial, view_data, sample_class, pro_lab, p_score, q_score, device, score_type, preds):
    seed_torch()
    bs = trial.suggest_int('bs', 256, 1024, step=32)
    lr = trial.suggest_float('lr', 1e-5, 1e-2)
    ld_ae = trial.suggest_int('ld_ae', 2, 30)#100)
    # h1 = trial.suggest_int('h1', 2, 128)
    h2 = trial.suggest_int('h2', 2, 128)
    s = trial.suggest_float('s', 0.1, 0.3, step=0.01)
    param = {'bs': bs, 'lr': lr, 'ld_ae': ld_ae, 'h2': h2, 's': s}
    # param = {'bs': bs, 'lr':lr, 'ld_ae': ld_ae, 'h1':h1, 'h2':h2, 's':s}

    model, loss, pro_loss, recon_x, lat_con_1, pro_pred, auc, _, _, _, _ = train(view_data, sample_class, pro_lab, p_score, q_score, param, device, score_type, preds)
    # auc = cv_train(view_data, sample_class, pro_lab, p_score, q_score, param, device, score_type)
    return 1-auc#loss #pro_loss# #pro_loss + cls_loss#

def optuna_optimize_para(view_data, sample_class, pro_lab, p_score, q_score, device, ntry, score_type, preds):

    study = optuna.create_study(study_name='test', direction='minimize', sampler=optuna.samplers.TPESampler(seed=2023),
                                pruner=optuna.pruners.HyperbandPruner())#, storage=save_sty)

    func = lambda trial: objective_para(trial, view_data, sample_class, pro_lab, p_score, q_score, device, score_type, preds)
    study.optimize(func, n_trials=ntry)

    final_para = copy.deepcopy(study.best_params)
    filename = "best_param_v6034.txt"
    # 追加写入内容到文件
    with open(filename, "a") as file:
        file.write(str(np.array(final_para)) + '\t')
    model, loss, pro_loss, recon_x, lat_con_1, pro_pred, auc, feature_importances, view_contributionss, cross_matrixs, head_importances = train(view_data, sample_class, pro_lab, p_score, q_score, final_para, device, score_type, preds)
    del preds['aucs'][-1]
    del preds['probs'][-1]
    del preds['labs'][-1]
    del preds['mccs'][-1]
    return model, loss, pro_loss, recon_x, lat_con_1, pro_pred, feature_importances, view_contributionss, cross_matrixs, head_importances

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

def cal_logFC(mat, col_nm, g1, g2, design_pth, logT, view_name):
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
        #idx_g1g2s.append(idx_g1g2)
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
    file_name_design = view_paths['design_file_path'].split('/')

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

def get_datasets_reproduce_mv(flag, data_idx, in_type, norms, imputs, g1, g2, acq, ex_tys,
                         save_data_root, R_fold, deas, logTs, seed, pg_ty, Rscript_path, cross_imp):
    sf = 'temp' + str(flag) + '/'

    data_fold = 'data/'
    maxtrix_folder = data_fold + acq + '/' + in_type + '/'

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

        print(in_files['Rscript_path'] + 'Rscript ' + in_files['R_fold'] + 'Hurdle_rep.R ' + ' ' + print_lab + ' ' + '"' + in_files['true_organism'] + '"' +
                  ' ' + '"' + in_files['DE_organism'] + '"' + ' ' + in_files['true_lgfc'] + ' ' + logT + ' ' + ' ' + in_files['R_fold'] + ' ' + all_res_files)
        os.system(in_files['Rscript_path'] + 'Rscript ' + in_files['R_fold'] + 'Hurdle_rep.R ' + ' ' + print_lab + ' ' + '"' + in_files['true_organism'] + '"' +
                  ' ' + '"' + in_files['DE_organism'] + '"' + ' ' + in_files['true_lgfc'] + ' ' + logT + ' ' + in_files['R_fold'] + ' ' + all_res_files)

        res_file_name = file_names[0]
    else:

        res_file_name0 = save_fold + dataset + '_' + tool + '_' + in_type + '_' + inten_ty + '_' + imput + '_' + normal + '.csv'
        res_file_name0 = res_file_name0.replace('blank', '')

        #if not os.path.exists(res_file_name0):

        print(in_files['Rscript_path'] + 'Rscript ' + in_files['R_fold'] + tool + '_DE_rep.R' + ' ' + in_files['save_scale'] + ' ' +
                  in_type + ' ' + inten_ty + ' ' +
                  str(imput) + ' ' + str(normal) + ' ' + dataset + ' ' + save_fold + ' ' + print_lab + ' ' + '"' + str(in_files['true_organism'])+ '"' +
                  ' ' + '"' + str(in_files['DE_organism']) + '"' + ' ' + in_files['true_lgfc'] + ' ' + logT + ' ' + in_files['design'] + ' ' + in_files['R_fold'])

        os.system(in_files['Rscript_path'] + 'Rscript ' + in_files['R_fold'] + tool + '_DE_rep.R' + ' ' + in_files['save_scale'] + ' ' +
                  in_type + ' ' + inten_ty + ' ' +
                  str(imput) + ' ' + str(normal) + ' ' + dataset + ' ' + save_fold + ' ' + print_lab + ' ' + '"' + str(in_files['true_organism'])+ '"' +
                  ' ' + '"' + str(in_files['DE_organism']) + '"' + ' ' + in_files['true_lgfc'] + ' ' + logT + ' ' + in_files['design'] + ' ' + in_files['R_fold'])



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

        v_file_path = view_paths[ex_tys[v]].split('/')
        normed_real_v = copy.deepcopy(views[v].values)

        save_scale_v = save_folder + 'real_normed_v' + str(v)  + '_' + ci + '_' + v_file_path[len(v_file_path) - 1]
        normed_real_v[:, idx_g1g2s[v]] = v_norm
        pd.DataFrame(normed_real_v, columns=views[v].columns.values.tolist()).to_csv(save_scale_v, sep='\t',
                                                                                     index=False)
        view_paths.update({'save_scale': save_scale_v})
        #v_norm, rowmean = mean_norm(v_norm)

        v_scale1, mm1 = processing_type(v_norm, 0)
        #v_scale1, mm1 = processing_type(v_norm, 2)
        mms.append(mm1)
        v_scales.append(v_scale1)
        res_real_norm_v = run_Rscript_new(deas[v], platform, ex_tys[v], 'blank', 'blank', view_paths, dataset+ci, 'F')
        res_real_norm_vs.append(res_real_norm_v)
        res_real_norm_v_paths.append(res_real_norm_v[0])

    if ci != '':
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

def mv_logFC_mv(logFC_v_norms):
    mv_logFC = []
    for i in range(len(logFC_v_norms[0][:, 0])):
        logfcs = np.array([logFC_v_norms[j][i, 0] for j in range(len(logFC_v_norms))], dtype='float')
        max_idx = np.where((abs(logfcs)==max(abs(logfcs))))[0]
        mv_logFC.append(logfcs[max_idx[0]])

    return np.array([mv_logFC]).T

def stat_test(data, lab, t_type):

    p_vals = []
    mean1s = []
    mean2s = []
    diffs = []

    uni_lab = np.unique(lab)
    idx1 = np.where((lab==uni_lab[0]))[0]
    idx2 = np.where((lab==uni_lab[1]))[0]

    dt_g1 = data[:, idx1]
    dt_g2 = data[:, idx2]

    for i in range(len(data[:, 0])):
        idx_vali1 = np.where(((dt_g1[i, :]!=0) & (~np.isnan(dt_g1[i, :]))))[0]
        idx_vali2 = np.where(((dt_g2[i, :] != 0) & (~np.isnan(dt_g2[i, :]))))[0]

        if (len(idx_vali1)>=2) & (len(idx_vali2)>=2):
            mean1 = np.mean(dt_g1[i, idx_vali1])
            mean2 = np.mean(dt_g2[i, idx_vali2])
            mean1s.append(mean1)
            mean2s.append(mean2)
            diffs.append(mean2-mean1)
            if t_type == 'wrs':
                from scipy.stats import ranksums as rks
                p = rks(dt_g1[i, idx_vali1], dt_g2[i, idx_vali2]).pvalue
            elif t_type == 'tt':
                from  scipy.stats import ttest_ind as tt
                p = tt(dt_g1[i, idx_vali1], dt_g2[i, idx_vali2]).pvalue

            p_vals.append(p)
        else:
            mean1s.append(0)
            mean2s.append(0)
            diffs.append(0)
            p_vals.append(-1)

    from statsmodels.stats.multitest import fdrcorrection as fdr
    qval = copy.deepcopy(np.array(p_vals))
    qval[np.where((qval!=-1))] = fdr(np.array(p_vals)[np.where((qval!=-1))[0]])[1]

    return np.array(p_vals), qval, np.array(diffs)

def preprocess_data(views, g2, logfc_thre, q_thre, res_union, view_paths, pos_ty):
    logFC_mv = res_union[5]
    #logFC_score = preprocessing.MinMaxScaler().fit_transform(abs(logFC_mv))
    # confis = np.mean(confis, axis=0)
    # pro_lab = np.zeros((len(logFC_mv[:, 0]), 1))
    # pro_lab[np.where(((confis>=np.fix(len(views)*0.8))))[0]] = 1 #(logFC_mv[:, 0]<=lab_thre) &
    # mms = [preprocessing.MinMaxScaler() for i in range(len(data_list))]
    # mms = [preprocessing.StandardScaler() for i in range(len(data_list))]
    mms = res_union[3]
    data = res_union[0]#[mms[i].fit_transform(data_list[i]) for i in range(len(data_list))]
    design = pd.read_csv(res_union[4]['design'], sep='\t', header=0)
    # cluster_lab = []
    views_names = ['v'+str(i+1) for i in range(len(views))]# ['v1', 'v2', 'v3']
    header = np.array(views[0].columns.values)
    if 'Organism' in header:
        clns = header[2:len(header)]
    else:
        clns = header[1:len(header)]
    condition = []
    for cln in clns:
        cond = design['condition'].values[np.where((design['sample_name'].values == cln))[0][0]]
        condition.append(cond)
    sample_class = np.zeros((len(condition), 1))
    sample_class[np.where((np.array(condition) == g2))[0]] = 1

    data_cat = np.concatenate(data, axis=1)
    data_cat[np.isnan(data_cat)] = -1
    count_miss = [len(np.where((data_cat[i, :] == -1))[0]) for i in range(len(data_cat[:, 0]))]
    idx_retain = np.where((np.array(count_miss) >= int(len(data_cat[0, :]) * 0.8)))[0]

    # count_misss = []
    # idx_retains = []
    # for i in range(len(data)):
    #     dt = copy.deepcopy(data[i])
    #     dt[np.isnan(dt)] = -1
    #
    #     if (len(data[0][0, :]) > 20):
    #         count_miss = [len(np.where((dt[j, :] != -1))[0]) for j in range(len(dt[:, 0]))]
    #         idx_retain = np.where((np.array(count_miss) <= 10))[0]
    #     else:
    #         count_miss = [len(np.where((dt[j, :] == -1))[0]) for j in range(len(dt[:, 0]))]
    #         idx_retain = np.where((np.array(count_miss) >= int(len(dt[0, :]) * 0.8)))[0]
    #     count_misss.append(count_miss)
    #     idx_retains.append(idx_retain)
    #
    # idx_retain = idx_retains[0]
    # for i in range(1, len(idx_retains)):
    #     idx_retain = np.intersect1d(idx_retain, idx_retains[i])

    for i in range(len(data)):
        # for j in idx_retain:
        #     data[i][j, :][np.isnan(data[i][j,:])] = 0

        data[i][np.isnan(data[i])] = 0
            #data[i][~idx_retain][np.isnan(data[i][~idx_retain])] = -1

    proteins_vs = []
    logFC_vs = []
    qval_vs = []
    score_vs = []
    T_label_vs = []
    view_protein = views[0]['Protein'].values
    if 'Organism' in views[0].keys():
        organism_pro = views[0]['Organism'].values
    else:
        organism_pro = np.array(['unknown'] * len(view_protein))

    for i in range(0, len(views_names)):
        proteins_vs.append(res_union[1][i][1])
        logFC_vs.append(res_union[1][i][7])
        qval_vs.append(res_union[1][i][10])
        score_vs.append(res_union[1][i][3])
        T_label_vs.append(res_union[1][i][5])

    hurdle_protein = res_union[2][1]
    hurdle_logFC = res_union[2][7]
    hurdle_qval = res_union[2][10]
    hurdle_score = res_union[2][3]
    hurdle_T_label = res_union[2][5]
    de_organism = view_paths['DE_organism']
    fea_all = np.zeros((len(view_protein), 100))
    logFC_all = res_union[5]
    hc_thr = 0.0001
    npos_thr = 2
    combine_score_ty = 'min'
    pos_ty = pos_ty
    de_thr = q_thre
    lgfc_thr = logfc_thre

    train_pro, train_fea, train_logFC, train_scores, train_orga, train_true_label, train_qval_vs, train_logfc_vs = get_training_proteins_repoduce_mv(
        proteins_vs, view_protein, qval_vs, logFC_vs, fea_all,
        logFC_all, organism_pro, de_organism, hc_thr, de_thr, lgfc_thr, npos_thr, combine_score_ty,
        hurdle_qval,
        hurdle_protein, hurdle_logFC, views_names, pos_ty)

    com, idx1, idx2 = np.intersect1d(view_protein, train_pro, return_indices=True)
    final_lab = np.array([np.nan]*len(view_protein))
    q_score = np.array([np.nan]*len(view_protein))
    q_score[idx1] = train_scores[:, 0][idx2]
    logFC_final = np.array([np.nan]*len(view_protein))
    logFC_final[idx1] = train_logFC[:, 0][idx2]

    for j in range(len(final_lab)):
        if (~np.isnan(logFC_final[j])) & (q_score[j] != -1):
            if (abs(logFC_final[j])>=lgfc_thr) & ((1-q_score[j])<=de_thr):
                final_lab[j] = 1
            elif (abs(logFC_final[j])<lgfc_thr) | ((1-q_score[j])>de_thr):
                final_lab[j] = 0
    pro_lab = np.array([final_lab]).T

    true_lab = np.array(get_true_label(train_pro, de_organism))
    print(cal_metrics1(train_scores[:, 0], de_thr, train_logFC, lgfc_thr, true_lab))

    pro_lab[np.isnan(pro_lab)] = -1
    q_score[np.isnan(q_score)] = -1
    p_score = q_score


    pro_lab[idx_retain, 0] = 0

    # for i in range(len(data)):
    #     data[i][idx_retain, :][data[i][idx_retain, :]==-1] = 0

    #q_score = 1-np.min(qvals, axis=0)

    # p_score[np.where((pro_lab[:, 0] == -1))[0]] = -1
    # q_score[np.where((pro_lab[:, 0] == -1))[0]] = -1

    return data, sample_class, pro_lab, p_score, q_score, mms, logFC_mv, clns, condition

#train_view_data, train_view, g2, logfc_thre, q_thre, res_union
def preprocess_data_old(views, g2, logfc_thre, q_thre, res_union):
    logFC_mv = res_union[5]

    mms = res_union[3]
    data = res_union[0]#[mms[i].fit_transform(data_list[i]) for i in range(len(data_list))]
    design = pd.read_csv(res_union[4]['design'], sep='\t', header=0)
    # cluster_lab = []

    header = np.array(views[0].columns.values)
    if 'Organism' in header:
        clns = header[2:len(header)]
    else:
        clns = header[1:len(header)]
    condition = []
    for cln in clns:
        cond = design['condition'].values[np.where((design['sample_name'].values == cln))[0][0]]
        condition.append(cond)
    sample_class = np.zeros((len(condition), 1))
    sample_class[np.where((np.array(condition) == g2))[0]] = 1
    labs = []
    pvalss = []
    qvals = []

    # for i in range(len(data)):
    for i in range(len(data)):
        data[i][np.isnan(data[i])] = -1

    for i in range(1):
        pro = res_union[1][i][1]
        all_pro = views[0]['Protein'].values
        pvals, qval, diffs = res_union[1][i][9], res_union[1][i][10], res_union[1][i][7]

        pvals_v = np.array([1] * len(all_pro), dtype='float')
        qvals_v = np.array([1] * len(all_pro), dtype='float')

        diff_v = np.array([0] * len(all_pro), dtype='float')

        com, idx1, idx2 = np.intersect1d(pro, all_pro, return_indices=True)
        pvals_v[idx2] = pvals[idx1]
        qvals_v[idx2] = qval[idx1]
        diff_v[idx2] = diffs[idx1]
        # pros = np.array(['-'] * len(all_pro))
        # pros[idx2] = pro[idx1]

        pvalss.append(pvals_v)
        qvals.append(qvals_v)
        lab = np.array([-1] * len(pvals_v))
        # lab[np.where(((qvals_v <= q_thre) & (diff_v>=logfc_thre)))[0]] = 2
        lab[np.where(((qvals_v <= q_thre) & (abs(diff_v) >= logfc_thre)))[0]] = 1
        # lab[np.where(((qvals_v <= q_thre) & (diff_v <= -logfc_thre)))[0]] = 1
        lab[np.where(((qvals_v > q_thre) | (abs(diff_v)<logfc_thre)))[0]] = 0
        lab[np.where(((qvals_v == -1)))[0]] = -1
        labs.append(lab)


        hurdle_proteins = res_union[2][1]
        hurdle_logFC = res_union[2][7]
        hurdle_pval = res_union[2][9]
        hurdle_qval = res_union[2][10]

        pvals_v = np.array([1] * len(all_pro), dtype='float')
        qvals_v = np.array([1] * len(all_pro), dtype='float')
        diff_v = np.array([0] * len(all_pro), dtype='float')

        com, idx1, idx2 = np.intersect1d(hurdle_proteins, all_pro, return_indices=True)
        pvals_v[idx2] = hurdle_pval[idx1]
        qvals_v[idx2] = hurdle_qval[idx1]
        diff_v[idx2] = hurdle_logFC[idx1]

        pvalss.append(pvals_v)
        qvals.append(qvals_v)
        lab = np.array([-1] * len(pvals_v))
        # lab[np.where(((qvals_v <= q_thre) & (diff_v >= logfc_thre)))[0]] = 2
        # lab[np.where(((qvals_v <= q_thre) & (diff_v <= -logfc_thre)))[0]] = 1
        lab[np.where(((qvals_v <= q_thre) & (abs(diff_v) >= logfc_thre)))[0]] = 1
        lab[np.where(((qvals_v > q_thre) | (abs(diff_v) < logfc_thre)))[0]] = 0
        lab[np.where(((qvals_v == -1)))[0]] = -1
        labs.append(lab)

        #min(data[i][~np.isnan(data[i])])
        #clus_lab = KMeans(n_clusters=3).fit_predict(PCA(n_components=5).fit_transform(data[i][np.where((pro_lab==1))[0]]))
        #logFC_mv_clus = logFC_mv[np.where((pro_lab==1))[0]]
        #logFCs[i][np.where((pro_lab==1))[0]]
        # assign clus_lab to be 1 if it has been labeled as 1 according to logFC
        #pro = views[0]['Protein'].values[np.where((pro_lab==1))[0]]
        #mat = views[0].values[np.where((pro_lab==1))[0], :]
        #new_clus_lab = copy.deepcopy(clus_lab)



    labs = np.array(labs).T
    sums = np.sum(labs, axis=1)
    count_0 = np.array([1 if len(np.where((labs[i, :] == 0))[0])==len(labs[0, :]) else 0 for i in range(len(labs[:, 0]))])
    count_1 = np.array([1 if len(np.where((labs[i, :] == 1))[0]) == len(labs[0, :]) else 0 for i in range(len(labs[:, 0]))])
    # count_2 = np.array([1 if len(np.where((labs[i, :] == 2))[0]) == len(labs[0, :]) else 0 for i in range(len(labs[:, 0]))])
    final_lab = np.array([-1]*len(labs[:, 0]))
    #final_lab[np.where((sums>=np.fix(len(data)*0.8)))[0]] = 1
    final_lab[np.where((count_1 == 1))[0]] = 1
    # final_lab[np.where((count_2 == 1))[0]] = 2
    final_lab[np.where((count_0 == 1))[0]] = 0
    pro_lab = np.array([final_lab]).T

    p_score = 1-np.min(pvalss, axis=0)
    q_score = 1-np.min(qvals, axis=0)

    p_score[np.where((pro_lab[:, 0] == -1))[0]] = -1
    q_score[np.where((pro_lab[:, 0] == -1))[0]] = -1

    return data, sample_class, pro_lab, p_score, q_score, mms, logFC_mv, clns, condition


def impute_MinProb(dataSet_mvs, q=0.01, tune_sigma=1):
    nSamples = dataSet_mvs.shape[1]
    nFeatures = dataSet_mvs.shape[0]

    dataSet_imputed = dataSet_mvs.copy()

    # Select the minimum values sample-wise (corresponding to the q-th quantile)
    min_samples = np.nanquantile(dataSet_imputed, q, axis=0)

    # Estimate protein-wise standard deviation using only proteins containing
    # more than 50% non-NAs
    count_NAs = np.sum(~np.isnan(dataSet_mvs), axis=1) / nSamples
    dataSet_filtered = dataSet_mvs[count_NAs > 0.5, :]

    protSD = np.nanstd(dataSet_filtered, axis=1)
    sd_temp = np.nanmedian(protSD) * tune_sigma

    print(sd_temp)

    for i in range(nSamples):
        dataSet_to_impute_temp = np.random.normal(loc=min_samples[i], scale=sd_temp, size=nFeatures)
        na_indices = np.isnan(dataSet_mvs[:, i])
        dataSet_imputed[na_indices, i] = dataSet_to_impute_temp[na_indices]

    return dataSet_imputed

def compute_outlier_scores(embeddings, pros, n_clusters = 2):
    kmeans = KMeans(n_clusters=n_clusters, random_state=2024)
    kmeans.fit(embeddings)
    y_pred = kmeans.predict(embeddings)
    cluster_centers = kmeans.cluster_centers_
    uni_lab = np.unique(y_pred)
    lens = [len(np.where((y_pred==uni_lab[i]))[0]) for i in range(len(uni_lab))]
    # Compute distances to nearest cluster center
    distances = pairwise_distances(embeddings, np.array([cluster_centers[np.where((np.array(lens)==min(lens)))[0][0], :]]))  # Euclidean distances
    min_distances = np.min(distances, axis=1)
    # Degrees of freedom (adjust based on your data characteristics)
    df = embeddings.shape[1]  # Number of features, for chi-squared distribution

    # Compute p-values based on chi-squared distribution
    p_values = chi2.sf(min_distances ** 2, df)

    return p_values

def fdrsToQvals(fdrs):
  qvals = [0] * len(fdrs)
  qvals[len(fdrs)-1] = fdrs[-1]
  for i in range(len(fdrs)-2, -1, -1):
    qvals[i] = min(qvals[i+1], fdrs[i])
  return qvals

def fdr_2qvalue(score, label):
    ## input score is higher better
    ## fdr = (1+D)/(T+1)
    ##  call fdrsToQvals(fdrs)
    fdrs = [(len(np.where((label[np.where((score>=score[i]))[0]]==-1))[0])+1)/(len(np.where((score>=score[i]))[0])-len(np.where((label[np.where((score>=score[i]))[0]]==-1))[0])+1) for i in range(len(score))]
    fdrs = np.array(fdrs)
    idx = [i for i in range(len(score))]
    idx = np.array([idx]).T
    fdrs = np.array([fdrs]).T
    score = np.array([score]).T
    ifcb = np.column_stack((score, fdrs, idx))
    ifcb = ifcb[np.argsort(-ifcb[:, 0]), :]
    q_value = fdrsToQvals(ifcb[:, 1])
    ifcb[:, 1] = q_value
    ifcb = ifcb[np.argsort(ifcb[:, 2]), :]
    report_qvalue = ifcb[:, 1]
    return report_qvalue

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
                    gene_id.append(pro_s[0].split('|')[2].split('_')[0])
                    if len(pro_s[0].split('|')[2].split('_')) > 1:
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

def cal_qval(views, rec, prob, pred_lab, sample_names, g1, g2, mv_logFC, design, thre, qthre, view_paths):
    logFCs = []
    confis = []
    for v in range(len(rec)):
        view_logFC, view_confi = cal_logFC(rec[v], sample_names, g1, g2, design, 'F', '')
        logFCs.append(view_logFC)
        confis.append(view_confi)

   # avg_mv = np.mean(np.array(logFCs).squeeze().T, axis=1)
    avg_mv = mv_logFC
    TD = np.array([-1]*len(avg_mv))
    TD[np.where(((abs(avg_mv)>=thre) & (prob>0.5)))[0]] = 1

    qval = fdr_2qvalue(prob, TD)
    qval= fdr(np.array(qval))[1]

    protein_true_lab = np.array(get_true_label(views[0]['Protein'].values, view_paths['DE_organism']))
    # q_threshold = 0.05
    score_threshold = qthre
    metrics_prob, prob_lab = cal_metrics2(prob, score_threshold, avg_mv, thre, protein_true_lab)
    output = pd.DataFrame(np.column_stack((views[0]['Protein'].values, prob, mv_logFC, avg_mv, pred_lab, qval, prob_lab, np.array([score_threshold]*len(prob_lab)))),
                          columns=['Protein', 'Probability', 'ori_logFC', 'rec_logFC', 'label_pred', 'qvalue', 'prob_lab', 'threshold'])
    #metrics_qval = cal_metrics1(1-qval, 0.05, avg_mv, thre, protein_true_lab)
    print('probability metrics')
    print(metrics_prob)
    print('qvalue metrics')
    #print(metrics_qval)
    return output, metrics_prob, protein_true_lab

def clustering_metrics(true_lab, pre_lab):
    ARI = np.round(adjusted_rand_score(true_lab, pre_lab), 4)
    ASW = np.round(silhouette_score(np.array([true_lab]).T, np.array([pre_lab]).T), 4)
    NMI = np.round(normalized_mutual_info_score(true_lab, pre_lab), 4)
    return [ARI, ASW, NMI]

def clustering(lat_pro, label_pro, lat_sample, label_sample, view_paths, true_pro_lab, true_sample_label, show='true'):
    # Initialize UMAP
    umap_model = umap.umap_.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)

    # Fit and transform the data
    umap_embedding_pro = umap_model.fit_transform(lat_pro)
    df_pro = pd.DataFrame(lat_pro, columns= ['lat' + str(i) for i in range(len(lat_pro[0, :]))])
    df_pro['label'] = label_pro
    # Add the UMAP results to the DataFrame
    df_pro['UMAP1'] = umap_embedding_pro[:, 0]
    df_pro['UMAP2'] = umap_embedding_pro[:, 1]

    plt.figure(figsize=(10, 7))
    sns.scatterplot(
        x='UMAP1', y='UMAP2',
        hue='label',
        palette=sns.color_palette('hsv', len(np.unique(label_pro))),
        data=df_pro,
        legend='full'
    )
    plt.title('UMAP projection of the proteins')
    plt.xlabel('UMAP1')
    plt.ylabel('UMAP2')
    output_filename = view_paths['save_folder'] + view_paths['dataset']+ '_umap_protein.png'
    plt.savefig(output_filename, format='png', dpi=300, bbox_inches='tight')
    if show == 'true':
        plt.show()

    # Initialize UMAP
    umap_model = umap.umap_.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
    k_means = []
    k_means = KMeans(n_clusters=len(np.unique(label_pro)))
    y_pro = k_means.fit_predict(lat_pro)
    clus_perform_pro = clustering_metrics(true_pro_lab, y_pro)
    # Fit and transform the data
    if len(lat_sample)>0:
        umap_embedding_sample = umap_model.fit_transform(lat_sample)
        df_sample = pd.DataFrame(lat_sample, columns= ['lat' + str(i) for i in range(len(lat_sample[0, :]))])
        df_sample['label'] = label_sample
        # Add the UMAP results to the DataFrame
        df_sample['UMAP1'] = umap_embedding_sample[:, 0]
        df_sample['UMAP2'] = umap_embedding_sample[:, 1]

        plt.figure(figsize=(10, 7))
        sns.scatterplot(
            x='UMAP1', y='UMAP2',
            hue='label',
            palette=sns.color_palette('hsv', len(np.unique(label_sample))),
            data=df_sample,
            legend='full'
        )
        plt.title('UMAP projection of the sample')
        plt.xlabel('UMAP1')
        plt.ylabel('UMAP2')
        output_filename = view_paths['save_folder'] + view_paths['dataset']+ '_umap_sample.png'
        plt.savefig(output_filename, format='png', dpi=300, bbox_inches='tight')
        if show == 'true':
            plt.show()
        k_means = []
        k_means = KMeans(n_clusters=len(np.unique(label_sample)))
        y_sample = k_means.fit_predict(lat_sample)
        clus_perform_sample = clustering_metrics(true_sample_label[:, 0], y_sample)
        print('cluster performance sample:')
        print(clus_perform_sample)
    else:
        df_sample = []
        clus_perform_sample = []
    print('cluster performance protein:')
    print(clus_perform_pro)

    return df_pro, df_sample, clus_perform_pro, clus_perform_sample

def preprocess_data_tt(data_list, views, g2, logFCs, design, lab_thre):
    logFC_mv = logFCs
    #logFC_score = preprocessing.MinMaxScaler().fit_transform(abs(logFC_mv))
    # confis = np.mean(confis, axis=0)
    # pro_lab = np.zeros((len(logFC_mv[:, 0]), 1))
    # pro_lab[np.where(((confis>=np.fix(len(views)*0.8))))[0]] = 1 #(logFC_mv[:, 0]<=lab_thre) &
    mms = [preprocessing.MinMaxScaler() for i in range(len(data_list))]
    #mms = [preprocessing.StandardScaler() for i in range(len(data_list))]
    data = [mms[i].fit_transform(data_list[i]) for i in range(len(data_list))]
    cluster_lab = []

    header = np.array(views[0].columns.values)
    if 'Organism' in header:
        clns = header[2:len(header)]
    else:
        clns = header[1:len(header)]
    condition = []
    for cln in clns:
        cond = design['condition'].values[np.where((design['sample_name'].values == cln))[0][0]]
        condition.append(cond)
    sample_class = np.zeros((len(condition), 1))
    sample_class[np.where((np.array(condition) == g2))[0]] = 1
    labs = []
    pvalss = []
    qvals = []
    for i in range(len(data)):
        pvals, qval, diffs = stat_test(data_list[i], sample_class, 'tt')
        pvalss.append(pvals)
        qvals.append(qval)
        lab = np.array([-1] * len(pvals))
        # lab[np.where(((qval <= 0.05) & (diffs >= lab_thre)))[0]] = 2
        # lab[np.where(((qval <= 0.05) & (diffs <= -lab_thre)))[0]] = 1
        lab[np.where(((qval<=0.05) & (abs(diffs)>=lab_thre)))[0]] = 1
        lab[np.where(((qval > 0.05) | (abs(diffs)<lab_thre)))[0]] = 0
        lab[np.where(((qval == -1)))[0]] = -1
        labs.append(lab)
        data[i][np.isnan(data[i])] = -1

    labs = np.array(labs).T
    sums = np.sum(labs, axis=1)
    count_0 = np.array([1 if len(np.where((labs[i, :]==0))[0])==len(data) else 0 for i in range(len(pvals))])
    # count_2 = np.array(
    #     [1 if len(np.where((labs[i, :] == 2))[0]) == len(labs[0, :]) else 0 for i in range(len(labs[:, 0]))])
    count_1 = np.array(
        [1 if len(np.where((labs[i, :] == 1))[0]) == len(labs[0, :]) else 0 for i in range(len(labs[:, 0]))])
    final_lab = np.array([-1]*len(pvals))
    #final_lab[np.where((sums>=np.fix(len(data)*0.8)))[0]] = 1
    final_lab[np.where((count_1 == 1))[0]] = 1
    # final_lab[np.where((count_2 == 1))[0]] = 2
    final_lab[np.where((count_0 == 1))[0]] = 0
    pro_lab = np.array([final_lab]).T

    p_score = 1-np.min(pvalss, axis=0)
    q_score = 1-np.min(qvals, axis=0)

    p_score[np.where((pro_lab[:, 0] == -1))[0]] = -1
    q_score[np.where((pro_lab[:, 0] == -1))[0]] = -1

    return data, sample_class, pro_lab, p_score, q_score, mms, logFC_mv, clns, condition

def _fdr_de_prediction(posterior_probas: pd.Series, fdr: float = 0.05) -> pd.Series:
    """Compute posterior expected FDR and tag features as DE."""
    if not posterior_probas.ndim == 1:
        raise ValueError("posterior_probas should be 1-dimensional")
    original_index = posterior_probas.index
    sorted_pgs = posterior_probas.sort_values(ascending=False)
    cumulative_fdr = (1.0 - sorted_pgs).cumsum() / (1.0 + np.arange(len(sorted_pgs)))
    d = (cumulative_fdr <= fdr).sum()
    is_pred_de = pd.Series(np.zeros_like(cumulative_fdr).astype(bool), index=sorted_pgs.index)
    is_pred_de.iloc[:d] = True
    is_pred_de = is_pred_de.loc[original_index]
    return is_pred_de


def single_run(flag_id, platform, data_idx, g1, g2, acq, ex_tys, logTs, deas):
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    cuda = torch.cuda.is_available()
    print("use cuda: {}".format(cuda))
    device = torch.device('cuda' if cuda else 'cpu')
    #  flag_id = 0

    ts = time.time()
    flag = str(flag_id) + '_' + deas[0] + '_' + platform + '_view' + str(len(deas))

    R_code = os.path.dirname(__file__).replace('Result_reproducing\\DEA', '') + '/'
    Rscript_path = '/usr/lib/R/bin/'
    # python_path = 'D:/anaconda3/envs/directlfq/python.exe'
    seed = 2024
    logfc_thre = np.log2(1.5)
    pre_label_thre = np.log2(1.5)
    q_thre = 0.05
    seed_torch()
    score_type = 'q'
    norms = ['blank'] * len(ex_tys)
    imputs = ['blank'] * len(ex_tys)
    # deas = ['limma'] * len(ex_tys)
    cross_imp = 'none'  # 'CCA'
    hc_thr = 0.0001
    np_thr = 5
    num_tra = 30
    op_ty = 'auc'
    pg_ty = 'remove'
    pos_ty = '0_h'  # '0_h'  # 0_1_2

    view_name_strs = acq
    for y in range(len(ex_tys)):
        view_name_strs += '_' + ex_tys[y]

    save_data_root = os.path.dirname(__file__).replace('Result_reproducing\\DEA',
                                                       '')  # 'D:/Roffline/MVIDIA/MVIDIA_results_reproducing/Figures/Figure1/data/'
    save_data_root = save_data_root + 'Figures/Figure1/data/' + str(view_num) + 'views_test6034' + str(
        q_thre) + '_' + op_ty + '_' + norm[0] + '_' + imput[0] + '_' + pg_ty + '_' + pos_ty + '/all_exp1/' + view_name_strs + '/'

    filename = "best_param_v6034.txt"
    with open(filename, "a") as file:
        file.write('\n' + flag + ',' + str(data_idx) + ';')
    (res_ori, res_intersect, res_ci, views, view_datas, views_intersect, view_datas_intersect,
     views_ci, view_datas_ci, save_folder, view_paths, res_union, views_union,
     view_datas_union) = get_datasets_reproduce_mv(flag, data_idx, platform, norms, imputs, g1, g2, acq,
                                                   ex_tys,
                                                   save_data_root, R_code, deas, logTs, seed, pg_ty, Rscript_path,
                                                   cross_imp)

    train_view = views_union
    data, sample_class, pro_lab, p_score, q_score, mms, logFC_mv, sample_names, sample_condition = preprocess_data(
        train_view,
        g2, logfc_thre, q_thre, res_union, view_paths, pos_ty)

    preds = {'aucs': [], 'probs': [], 'labs': [], 'mccs': [],}
    t1 = time.time()
    model, loss, pro_loss, recon_x, lat_con_1, pro_pred, feature_importances, view_contributionss, cross_matrixs, head_importances  = optuna_optimize_para(data,
                                                                                                    sample_class,
                                                                                                    pro_lab,
                                                                                                    p_score,
                                                                                                    q_score, device,
                                                                                                    num_tra,
                                                                                                    score_type,
                                                                                                    preds)

    t2 = time.time()
    preds.update({'aucs': np.array([preds['aucs']]).T})
    preds.update({'probs': np.array(preds['probs']).T})
    preds.update({'labs': np.array(preds['labs']).T})

    from sklearn.metrics import roc_curve, precision_recall_curve
    prob_pro = 1 - pro_pred.detach().cpu().numpy()[:, 0]
    pre, rec, thresholds = precision_recall_curve(pro_lab[np.where((pro_lab != -1))[0]],
                                                  prob_pro[np.where((pro_lab != -1))[0]])
    f1_scores = 2 * (pre * rec) / (pre + rec + 1e-7)
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx]
    de_idx = _fdr_de_prediction(pd.Series(prob_pro))
    recons = [mms[i].inverse_transform(recon_x[i].detach().cpu().numpy()) for i in range(len(mms))]
    output, met1, protein_true_lab = cal_qval(train_view, recons, prob_pro, prob_pro, sample_names, g1, g2,
                                              logFC_mv,
                                              view_paths['design_file_path'], logfc_thre, optimal_threshold, view_paths)
    output.sort_values('Probability', ascending=False)
    lab_vote = np.array([0] * len(preds['labs'][:, 0]))
    lab_vote[np.where((np.sum(preds['labs'], axis=1) > len(preds['labs'][0, :]) / 2))[0]] = 1

    prob_vote = np.array([np.mean(preds['probs'][i, :][np.where((preds['labs'][i, :] == lab_vote[i]))[0]]) for i in
                          range(len(lab_vote))])
    pre, rec, thresholds = precision_recall_curve(pro_lab[np.where((pro_lab != -1))[0]],
                                                  prob_vote[np.where((pro_lab != -1))[0]])
    f1_scores = 2 * (pre * rec) / (pre + rec + 1e-7)
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx]
    output, met2, protein_true_lab = cal_qval(train_view, recons, prob_vote, prob_vote, sample_names, g1, g2,
                                              logFC_mv,
                                              view_paths['design_file_path'], logfc_thre,
                                              optimal_threshold,  view_paths)
    output.sort_values('Probability', ascending=False)
    prob_pro_correct = copy.deepcopy(prob_pro)
    prob_pro_correct[(~de_idx) & (prob_pro_correct > 0.5)] = 0.499
    pre, rec, thresholds = precision_recall_curve(pro_lab[np.where((pro_lab != -1))[0]],
                                                  prob_pro_correct[np.where((pro_lab != -1))[0]])
    f1_scores = 2 * (pre * rec) / (pre + rec + 1e-7)
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx]
    output, met3, protein_true_lab = cal_qval(train_view, recons, prob_pro_correct,
                                              prob_pro_correct, sample_names, g1, g2, logFC_mv,
                                              view_paths['design_file_path'], logfc_thre,
                                              optimal_threshold, view_paths)
    output.sort_values('Probability', ascending=False)
    print(model)
    # p_vals = compute_outlier_scores(lat_con_1, pros)
    print('running time of train:' + str(t2 - t1))

    protein_all_can = pd.read_csv(view_paths['all_candidate'], sep='\t', header=0)
    if pg_ty == 'remove':
        protein_all_can = process_proteingroup(protein_all_can, pg_ty)

    protein_all_can_true_lab = np.array(
        get_true_label(protein_all_can['Protein'].values, view_paths['DE_organism']))
    MV_dp = get_MVDEA_res_DP(views_union, res_union, save_folder, logfc_thre, q_thre, 'min', ex_tys, hc_thr,
                             np_thr, num_tra,
                             seed, op_ty, pos_ty, lat_con_1.detach().cpu().numpy(),
                             [recon_x[i].detach().cpu().numpy() for i in range(len(recon_x))])

    seed_torch()
    isExist = os.path.exists(save_folder + res_union[4]['dataset'] + '/')
    all_test_proteins, idx = np.unique(protein_all_can['Protein'].values, return_index=True)
    metrics = []
    methods = []
    if not isExist:
        os.makedirs(save_folder + res_union[4]['dataset'] + '/')

    metrics_all, methods_all, out_all_res = metrics_reference(ex_tys, all_test_proteins,
                                                              protein_all_can_true_lab[idx], MV_dp[4],
                                                              MV_dp[9], MV_dp[7],
                                                              MV_dp[5],
                                                              MV_dp[10], MV_dp[8], MV_dp[11], MV_dp[12],
                                                              MV_dp[13],
                                                              MV_dp[0], MV_dp[1],
                                                              MV_dp[14],
                                                              MV_dp[15], q_thre, logfc_thre, MV_dp[16], 'DP')
    out_all_res.to_csv(
        save_folder + '/' + res_union[4]['dataset'] + '/' + res_union[4]['dataset'] + '_DP_res_all.csv',
        sep=',', index=False)
    metrics.extend(metrics_all)
    metrics.extend([met1, met2, met3])
    methods.extend(methods_all)
    methods.extend(['DP_direct', 'DP_vote', 'DP_fdr'])

    metrics = np.array(metrics)
    methods = np.array(methods)
    metrics_out_all = np.column_stack((methods, metrics))
    out_pd = pd.DataFrame(np.array(metrics_out_all),
                          columns=['test_type', 'Acc', 'Prec', 'Rec', 'F1', 'F1w', 'Mcc', 'nMcc', 'geomean', 'tn',
                                   'fp', 'fn', 'tp', 'pauc001',
                                   'pauc005', 'pauc01'])
    print(out_pd.values)
    out_pd.to_csv(
        save_folder + '/' + res_union[4]['dataset'] + '/' + res_union[4]['dataset'] + '_' + '_metrics_all_s.csv',
        sep=',', index=False)

    a = 0
    te = time.time()
    print('total time= ' + str(te - ts))
    filename = "best_param_v6034.txt"
    with open(filename, "a") as file:
        file.write('\trun_time=' + str(t2 - t1) + ';')

    view_names = ex_tys
    feature_names = [
        [f"F{v}_{i}" for i in range(len(data[0][0, :]))]
        for v in range(len(data))
    ]
    visualize_contributions(view_names, feature_names, feature_importances, view_contributionss, cross_matrixs,
                            head_importances,
                            save_folder + '/' + res_union[4]['dataset'] + '/' + res_union[4]['dataset'])

def reproduce_test(acq, norm, imput, views_names):

    if acq == 'DIA':

        flag_id = 0
        for platform in ['DIANN', 'Spectronaut']:#['DIANN', 'Spectronaut']:  # , 'Spectronaut''DIANN' ,'DIANN', 'Spectronaut','Spectronaut',
            for dea in ['limma', 'ROTS']:#['limma', 'ROTS']:  # 'limma','limma', 'ROTS',
                for cbt in ['min']:  # , 'max', 'mean', 'max', 'mean'
                    flag = str(flag_id) + '_' + dea + '_' + cbt + '_' + platform
                    for didx in [2, 3, 4, 5, 6, 1, 0]:#[2, 3, 4, 5, 6, 1, 0]:  # 1,0, 1, 2, 0, 1, 2, 3, 4,0, 1,2, 3, 4, 5, [0]:#2, 3, 4, 5, 6, 1,
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
                        norms = norm * len(views_names)
                        imputs = imput * len(views_names)
                        logTs = ['F'] * len(views_names)
                        dt_idx = np.where((np.array(views_names) == 'dlfq'))[0]
                        if len(dt_idx) > 0:
                            logTs[dt_idx[0]] = 'T'
                        #logTs = ['T'] + ['F'] * (len(views_names)-1)#, 'F', 'F'] # * len(views_names)
                        try:
                            single_run(flag_id, platform, didx, g1, g2, acq, views_names, logTs, deas)
                        except:
                            continue

                    flag_id = flag_id + 1

    elif acq == 'DDA':

        flag_id = 0
        seed_torch()

        for platform in ['FragPipe', 'Maxquant']:  # ,, 'Maxquant', 'Maxquant',['Maxquant']:#'FragPipe','FragPipe',
            for dea in ['limma', 'ROTS']:  # 'limma',['ROTS']:#
                for cbt in ['min']:  # , 'hurdle', 'max', 'mean''min'
                    flag = str(flag_id) + '_' + dea + '_' + cbt + '_' + platform
                    for didx in [6, 2, 3, 0, 1, 7, 9, 4, 5, 10, 11]:  # 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11   0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11[4, 2, 5, 10, 11]:#2, 3, 0, 6, 1, 7, 9, 4,
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
                        try:
                            single_run(flag_id, platform, didx, g1, g2, acq, views_names, logTs, deas)
                        except:
                            continue

                    flag_id = flag_id + 1

def reports(ex_tys, all_proteins, all_true_label, res_union, res_pred, res_vote, res_fdr, lgfc_thre, q_thre, save_fold):

    out_all = np.column_stack((all_proteins, all_true_label))
    out_name = ['Protein', 'True_lab']

    for i in range(len(ex_tys)):
        protein_v = res_union[1][i][1]
        logFC_v = res_union[1][i][7]
        qval_v = res_union[1][i][10]
        score_v = res_union[1][i][3]

        com, idx1, idx2 = np.intersect1d(all_proteins, protein_v, return_indices=True)
        logfc = np.array([0] * len(all_proteins), dtype='float')
        qval = np.array([1] * len(all_proteins), dtype='float')
        score = np.array([0] * len(all_proteins), dtype='float')

        logfc[idx1] = logFC_v[idx2]
        qval[idx1] = qval_v[idx2]
        score[idx1] = score_v[idx2]

        label = np.array([0] * len(all_proteins))

        label[np.where(((qval <= q_thre) & (abs(logfc) >= lgfc_thre)))[0]] = 1

        out_all = np.column_stack((out_all, logfc, qval, score, label))

        out_name.extend([ex_tys[i] + '_' + s for s in ['logFC', 'qval', 'score', 'label']])

    hurdle_protein = res_union[2][1]
    hurdle_logFC = res_union[2][7]
    hurdle_qval = res_union[2][10]
    hurdle_score = res_union[2][3]

    com, idx1, idx2 = np.intersect1d(all_proteins, hurdle_protein, return_indices=True)
    logfc = np.array([0] * len(all_proteins), dtype='float')
    qval = np.array([1] * len(all_proteins), dtype='float')
    score = np.array([0] * len(all_proteins), dtype='float')

    logfc[idx1] = hurdle_logFC[idx2]
    qval[idx1] = hurdle_qval[idx2]
    score[idx1] = hurdle_score[idx2]

    label = np.array([0] * len(all_proteins))

    label[np.where(((qval <= q_thre) & (abs(logfc) >= lgfc_thre)))[0]] = 1

    out_all = np.column_stack((out_all, logfc, qval, score, label))

    out_name.extend(['hurdle_' + s for s in ['logFC', 'qval', 'score', 'label']])

    for res in [res_pred, res_vote, res_fdr]:
        protein_p = res['Protein'].values
        logFC_p = res['ori_logFC'].values
        score_p = res['Probability'].values
        label_p = res['prob_lab'].values
        threshold = res['threshold'].values

        com, idx1, idx2 = np.intersect1d(all_proteins, protein_p, return_indices=True)
        logfc = np.array([0] * len(all_proteins), dtype='float')
        thres = np.array([0] * len(all_proteins), dtype='float')
        score = np.array([0] * len(all_proteins), dtype='float')
        thres[idx1] = threshold[idx2]
        logfc[idx1] = logFC_p[idx2]
        [1 for i in range(len(label_p)) if label_p[i]]
        # qval[idx1] = hurdle_qval[idx2]
        score[idx1] = score_p[idx2]

        label = np.array([0] * len(all_proteins))

        #label[np.where(((qval <= q_thre) & (abs(logfc) >= lgfc_thre)))[0]] = 1
        label[idx1] = label_p[idx2]

        out_all = np.column_stack((out_all, logfc, score, threshold, label))
        out_name.extend(['MVcontrast_' + s for s in ['logFC', 'score', 'threshold', 'label']])

    pd.DataFrame(out_all, columns=out_name).to_csv(save_fold + 'DEA_res.csv')
    return save_fold + 'DEA_res.csv'


def get_MVcontrast_res(views_union, res_union, logfc_thre, q_thre, ex_tys,
                                num_tra, pos_ty, pg_ty, g1, g2, view_paths, acq, postfix, save_folder):

    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    cuda = torch.cuda.is_available()
    print("use cuda: {}".format(cuda))
    device = torch.device('cuda' if cuda else 'cpu')

    seed_torch()
    score_type = 'q'

    view_name_strs = acq
    for y in range(len(ex_tys)):
        view_name_strs += '_' + ex_tys[y]

    save_data_root = save_folder + view_name_strs + '/'
    save_folder = save_data_root
    train_view = views_union
    data, sample_class, pro_lab, p_score, q_score, mms, logFC_mv, sample_names, sample_condition = preprocess_data(
        train_view,
        g2, logfc_thre, q_thre, res_union, view_paths, pos_ty)

    preds = {'aucs': [], 'probs': [], 'labs': [], 'mccs': []}
    t1 = time.time()
    model, loss, pro_loss, recon_x, lat_con_1, pro_pred, feature_importances, view_contributionss, cross_matrixs, head_importances = optuna_optimize_para(
        data,
        sample_class,
        pro_lab,
        p_score,
        q_score, device,
        num_tra,
        score_type,
        preds)

    t2 = time.time()
    preds.update({'aucs': np.array([preds['aucs']]).T})
    preds.update({'probs': np.array(preds['probs']).T})
    preds.update({'labs': np.array(preds['labs']).T})
    preds.update({'mccs': np.array([preds['mccs']]).T})

    from sklearn.metrics import roc_curve, precision_recall_curve
    prob_pro = 1 - pro_pred.detach().cpu().numpy()[:, 0]
    pre, rec, thresholds = precision_recall_curve(pro_lab[np.where((pro_lab != -1))[0]],
                                                  prob_pro[np.where((pro_lab != -1))[0]])
    f1_scores = 2 * (pre * rec) / (pre + rec + 1e-7)
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx]
    de_idx = _fdr_de_prediction(pd.Series(prob_pro))
    recons = [mms[i].inverse_transform(recon_x[i].detach().cpu().numpy()) for i in range(len(mms))]
    latents = lat_con_1.detach().cpu().numpy()
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    pd.DataFrame(latents, index=list(train_view[0]['Protein'].values)).to_csv(save_data_root + 'fused_latent.csv')
    for s in range(len(train_view)):
        train_view[s].to_csv(save_data_root + 'view' + str(s) + '.csv')
    output_pred, met1, protein_true_lab = cal_qval(train_view, recons, prob_pro, prob_pro, sample_names, g1, g2,
                                              logFC_mv,
                                              view_paths['design_file_path'], logfc_thre, optimal_threshold, view_paths)
    output_pred.sort_values('Probability', ascending=False)
    lab_vote = np.array([0] * len(preds['labs'][:, 0]))
    lab_vote[np.where((np.sum(preds['labs'], axis=1) > len(preds['labs'][0, :]) / 2))[0]] = 1

    prob_vote = np.array([np.mean(preds['probs'][i, :][np.where((preds['labs'][i, :] == lab_vote[i]))[0]]) for i in
                          range(len(lab_vote))])
    pre, rec, thresholds = precision_recall_curve(pro_lab[np.where((pro_lab != -1))[0]],
                                                  prob_vote[np.where((pro_lab != -1))[0]])
    f1_scores = 2 * (pre * rec) / (pre + rec + 1e-7)
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx]
    output_vote, met2, protein_true_lab = cal_qval(train_view, recons, prob_vote, prob_vote, sample_names, g1, g2,
                                              logFC_mv,
                                              view_paths['design_file_path'], logfc_thre,
                                              optimal_threshold, view_paths)
    output_vote.sort_values('Probability', ascending=False)
    prob_pro_correct = copy.deepcopy(prob_pro)
    prob_pro_correct[(~de_idx) & (prob_pro_correct > 0.5)] = 0.499
    pre, rec, thresholds = precision_recall_curve(pro_lab[np.where((pro_lab != -1))[0]],
                                                  prob_pro_correct[np.where((pro_lab != -1))[0]])
    f1_scores = 2 * (pre * rec) / (pre + rec + 1e-7)
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx]
    output_fdr, met3, protein_true_lab = cal_qval(train_view, recons, prob_pro_correct,
                                              prob_pro_correct, sample_names, g1, g2, logFC_mv,
                                              view_paths['design_file_path'], logfc_thre,
                                              optimal_threshold, view_paths)
    output_fdr.sort_values('Probability', ascending=False)
    print(model)

    print('running time of train:' + str(t2 - t1))

    protein_all_can = pd.DataFrame(np.array(np.unique(np.concatenate([views_union[i]['Protein'].values for i in range(len(views_union))]))), columns=['Protein'])
    if pg_ty == 'remove':
        protein_all_can = process_proteingroup(protein_all_can, pg_ty)

    protein_all_can_true_lab = np.array(
        get_true_label(protein_all_can['Protein'].values, view_paths['DE_organism']))

    seed_torch()

    isExist = os.path.exists(save_folder + res_union[4]['dataset'] + '/')
    all_test_proteins, idx = np.unique(protein_all_can['Protein'].values, return_index=True)

    if not isExist:
        os.makedirs(save_folder + res_union[4]['dataset'] + '/')

    file_path = reports(ex_tys, all_test_proteins, protein_all_can_true_lab[idx], res_union, output_pred, output_vote, output_fdr, logfc_thre, q_thre, save_folder)
    return file_path





# if __name__ == '__main__':
#
#     # os.environ["CUDA_VISIBLE_DEVICES"] = '0'
#     # cuda = torch.cuda.is_available()
#     # print("use cuda: {}".format(cuda))
#     # device = torch.device('cuda' if cuda else 'cpu')
#     for view_num in [4]:
#         acq = 'DDA'
#         norm = ['blank']
#         imput = ['blank']
#         for acq in ['DDA']:  # , 'DIA']:'DDA',
#             if acq == 'DDA':
#                 views_names_all = ['dlfq', 'maxlfq', 'top0', 'top3', 'count']
#                 combs = np.array(list(combinations([x for x in range(len(views_names_all))], view_num)))
#                 for comb_idx in [0, 1]:#range(len(combs)):#[6, 7, 8, 9]:[5, 6, 7]:#
#                     view_idx = combs[comb_idx, :]
#                     views_names = list(np.array(views_names_all)[view_idx])
#                     #if len(np.intersect1d(np.array(views_names_all)[view_idx], ['maxlfq','top3','count']))==view_num:
#
#
#                         # views_names = list(np.array(views_names_all)[0:view_num])
#                     reproduce_test(acq, norm, imput, views_names)
#
#             elif acq == 'DIA':
#                 views_names_all = ['dlfq', 'maxlfq', 'top3', 'top1']
#
#                 combs = np.array(list(combinations([x for x in range(len(views_names_all))], view_num)))
#                 for comb_idx in range(len(combs)):
#                     view_idx = combs[comb_idx, :]
#                     views_names = list(np.array(views_names_all)[view_idx])
#                     # if len(np.intersect1d(np.array(views_names_all)[view_idx], ['top3','top1'])) == view_num:
#
#                         # views_names = list(np.array(views_names_all)[0:view_num])
#                     reproduce_test(acq, norm, imput, views_names)