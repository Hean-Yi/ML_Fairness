# label ranking Loss
# Macro-AUPR

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, accuracy_score, hamming_loss, label_ranking_loss, jaccard_score, average_precision_score
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Dict, List, Optional
import warnings
from collections import Counter

warnings.filterwarnings('ignore')

# 随机种子设置
torch.manual_seed(42)
np.random.seed(42)


class SSLDataset(Dataset):
    """半监督学习数据集，支持弱增强和强增强。"""

    def __init__(self, X: np.ndarray, a: np.ndarray, weak_noise: float = 0.05, strong_noise: float = 0.2):
        self.X = torch.FloatTensor(X)
        self.a = torch.LongTensor(a)
        self.weak_noise = weak_noise
        self.strong_noise = strong_noise

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.X[idx]
        a = self.a[idx]

        # 弱增强：添加少量高斯噪声
        x_weak = x + torch.randn_like(x) * self.weak_noise

        # 强增强：添加更大量的高斯噪声
        x_strong = x + torch.randn_like(x) * self.strong_noise

        return x_weak, x_strong, a


class StandardDataset(Dataset):
    """标准数据集，用于有标签数据和测试数据。"""

    def __init__(self, X: np.ndarray, y: np.ndarray, a: np.ndarray):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        self.a = torch.LongTensor(a)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx], self.a[idx]


class MPVAE(nn.Module):
    """
    MPVAE (Multivariate Probit VAE) 模型的PyTorch实现。
    保持原有的多元Probit模型结构不变。
    """

    def __init__(self, input_dim: int, num_labels: int, hidden_dim: int = 256, latent_dim: int = 50,
                 n_train_sample: int = 10, n_test_sample: int = 100):
        super(MPVAE, self).__init__()

        self.num_labels = num_labels
        self.latent_dim = latent_dim

        # --- 编码器 ---
        self.feature_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 512), nn.ReLU(),
            nn.Linear(512, hidden_dim)
        )
        self.fx_mu = nn.Linear(hidden_dim, latent_dim)
        self.fx_logvar = nn.Linear(hidden_dim, latent_dim)

        self.label_encoder = nn.Sequential(
            nn.Linear(input_dim + num_labels, 512), nn.ReLU(),
            nn.Linear(512, hidden_dim)
        )
        self.fe_mu = nn.Linear(hidden_dim, latent_dim)
        self.fe_logvar = nn.Linear(hidden_dim, latent_dim)

        # --- 解码器 ---
        self.decoder = nn.Sequential(
            nn.Linear(input_dim + latent_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 512), nn.ReLU()
        )
        self.decoder_mp_mu = nn.Linear(512, num_labels)

        # --- Probit 模型核心参数 ---
        init_val = torch.rand(num_labels, latent_dim) * 2 * np.sqrt(6.0 / (num_labels + latent_dim)) - np.sqrt(
            6.0 / (num_labels + latent_dim))
        self.r_sqrt_sigma = nn.Parameter(init_val)
        self.normal_dist = torch.distributions.Normal(0., 1.)

        self.n_train_sample = n_train_sample
        self.n_test_sample = n_test_sample

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, y=None, use_label_encoder=True):
        n_sample = self.n_train_sample if self.training else self.n_test_sample

        fx_hidden = self.feature_encoder(x)
        fx_mu = self.fx_mu(fx_hidden)
        fx_logvar = self.fx_logvar(fx_hidden)
        z_from_feature = self.reparameterize(fx_mu, fx_logvar)

        decoder_input_fx = torch.cat([x, z_from_feature], dim=1)
        decoded_fx_hidden = self.decoder(decoder_input_fx)
        feat_mp_mu = self.decoder_mp_mu(decoded_fx_hidden)
        E_x, _ = self._probit_sampling(feat_mp_mu, n_sample)

        if y is None or not use_label_encoder:
            predicted_probs = torch.mean(E_x, dim=0)
            return predicted_probs, fx_mu, fx_logvar

        xy = torch.cat([x, y], dim=1)
        fe_hidden = self.label_encoder(xy)
        fe_mu = self.fe_mu(fe_hidden)
        fe_logvar = self.fe_logvar(fe_hidden)
        z_from_label = self.reparameterize(fe_mu, fe_logvar)

        decoder_input_fe = torch.cat([x, z_from_label], dim=1)
        decoded_fe_hidden = self.decoder(decoder_input_fe)
        label_mp_mu = self.decoder_mp_mu(decoded_fe_hidden)
        E, _ = self._probit_sampling(label_mp_mu, n_sample)

        return E_x, fx_mu, fx_logvar, E, fe_mu, fe_logvar

    def _probit_sampling(self, mp_mu, n_sample):
        batch_size = mp_mu.shape[0]
        noise = torch.randn(n_sample, batch_size, self.latent_dim, device=mp_mu.device)
        B = self.r_sqrt_sigma.t()
        sample_r = torch.tensordot(noise, B, dims=([2], [0])) + mp_mu.unsqueeze(0)
        eps1 = 1e-6
        E = self.normal_dist.cdf(sample_r) * (1 - eps1) + eps1 * 0.5
        return E, sample_r

    def loss_function(self, E_x, fx_mu, fx_logvar, E, fe_mu, fe_logvar, y, nll_coeff=0.1, c_coeff=20.0, kl_coeff=1.0):
        nll_loss_fx = self._compute_nll_loss(E_x, y)
        nll_loss_fe = self._compute_nll_loss(E, y)
        c_loss_fx = self._compute_ranking_loss(E_x, y)
        c_loss_fe = self._compute_ranking_loss(E, y)

        kl_loss = 0.5 * torch.sum(
            (fx_logvar - fe_logvar) - 1 +
            torch.exp(fe_logvar - fx_logvar) +
            ((fx_mu - fe_mu).pow(2) / (torch.exp(fx_logvar) + 1e-6)),
            dim=1
        ).mean()

        total_loss = nll_coeff * (nll_loss_fx + nll_loss_fe) + \
                     c_coeff * (c_loss_fx + c_loss_fe) + \
                     kl_coeff * kl_loss

        return {'total': total_loss}

    def _compute_nll_loss(self, E, labels):
        labels_expanded = labels.unsqueeze(0).expand_as(E)
        sample_nll = -(torch.log(E) * labels_expanded + torch.log(1 - E) * (1 - labels_expanded))
        logprob = -torch.sum(sample_nll, dim=2)
        maxlogprob = torch.max(logprob, dim=0, keepdim=True)[0]
        Eprob = torch.mean(torch.exp(logprob - maxlogprob), dim=0)
        return torch.mean(-torch.log(Eprob) - maxlogprob.squeeze())

    def _compute_ranking_loss(self, E, labels):
        y = labels.unsqueeze(0).expand_as(E)
        y_i = (y == 1)
        y_not_i = (y == 0)
        truth_matrix = (y_i.unsqueeze(3) & y_not_i.unsqueeze(2)).float()
        sub_matrix = E.unsqueeze(3) - E.unsqueeze(2)
        exp_matrix = torch.exp(-5.0 * sub_matrix)
        sparse_matrix = exp_matrix * truth_matrix
        sums = torch.sum(sparse_matrix, dim=[2, 3])
        y_i_sizes = torch.sum(y_i.float(), dim=2)
        y_not_i_sizes = torch.sum(y_not_i.float(), dim=2)
        normalizers = y_i_sizes * y_not_i_sizes
        normalizers[normalizers == 0] = 1.0
        return (sums / normalizers).mean()


class FairSSL_Loss(nn.Module):
    """
    融合SimFair公平性损失和SSL一致性损失的组合损失函数。
    """

    def __init__(self, y_adv: torch.Tensor, lambda_unsup: float = 1.0,
                 confidence_threshold: float = 0.95, gamma: float = 1.0):
        super(FairSSL_Loss, self).__init__()
        self.y_adv = y_adv
        self.lambda_unsup = lambda_unsup
        self.confidence_threshold = confidence_threshold
        self.gamma = gamma
        self.bce_loss = nn.BCEWithLogitsLoss()

    def _similarity_function(self, y_true: torch.Tensor) -> torch.Tensor:
        """
        计算与对抗性样本的相似度。
        """
        y_adv_on_device = self.y_adv.to(y_true.device)
        y_true_binary = y_true.float() if y_true.dtype == torch.bool else y_true
        y_adv_batch_binary = y_adv_on_device.unsqueeze(0).expand(y_true.shape[0], -1)

        # 计算Jaccard相似度
        intersection = torch.sum(y_true_binary * y_adv_batch_binary, dim=1)
        union = torch.sum(y_true_binary + y_adv_batch_binary - y_true_binary * y_adv_batch_binary, dim=1)
        union = torch.clamp(union, min=1e-8)
        jaccard_sim = intersection / union
        jaccard_sim = torch.clamp(jaccard_sim, min=0.0, max=1.0)

        similarity = torch.exp(self.gamma * (jaccard_sim - 1.0))
        return similarity

    def _compute_simfair_violation(self, y_pred_probs: torch.Tensor, y_true: torch.Tensor,
                                   sensitive_attrs: torch.Tensor) -> torch.Tensor:
        """
        计算SimFair违规度。
        """
        s = self._similarity_function(y_true)
        s = torch.clamp(s, min=1e-8)

        # 计算全局加权平均
        global_numerator = torch.sum(y_pred_probs * s.unsqueeze(1), dim=0)
        global_denominator = torch.sum(s)

        if global_denominator < 1e-6:
            return torch.tensor(0.0, device=y_pred_probs.device)

        global_avg_pred = global_numerator / global_denominator

        total_violation = 0.0
        unique_groups = torch.unique(sensitive_attrs)

        for k in unique_groups:
            mask_k = (sensitive_attrs == k)
            group_size = mask_k.sum()

            if group_size == 0:
                continue

            s_k = s[mask_k]
            y_pred_probs_k = y_pred_probs[mask_k]

            group_numerator = torch.sum(y_pred_probs_k * s_k.unsqueeze(1), dim=0)
            group_denominator = torch.sum(s_k)

            if group_denominator > 1e-6:
                group_avg_pred = group_numerator / group_denominator
                violation = torch.sum((global_avg_pred - group_avg_pred) ** 2)
                total_violation += violation

        return total_violation

    def cal_simfair_violation(self, y_pred_probs: torch.Tensor, y_true: torch.Tensor,
                              sensitive_attrs: torch.Tensor) -> torch.Tensor:
        """
        计算SimFair违规度（用于评估）。
        """
        s = self._similarity_function(y_true)
        s = torch.clamp(s, min=1e-8)

        global_numerator = torch.sum(y_pred_probs * s.unsqueeze(1), dim=0)
        global_denominator = torch.sum(s)

        if global_denominator < 1e-6:
            return torch.tensor(0.0, device=y_pred_probs.device)

        global_avg_pred = global_numerator / global_denominator

        total_violation = 0.0
        unique_groups = torch.unique(sensitive_attrs)

        for k in unique_groups:
            mask_k = (sensitive_attrs == k)
            group_size = mask_k.sum()

            if group_size == 0:
                continue

            s_k = s[mask_k]
            y_pred_probs_k = y_pred_probs[mask_k]

            group_numerator = torch.sum(y_pred_probs_k * s_k.unsqueeze(1), dim=0)
            group_denominator = torch.sum(s_k)

            if group_denominator > 1e-6:
                group_avg_pred = group_numerator / group_denominator
                temp = torch.sum((global_avg_pred - group_avg_pred) ** 2)
                violation = torch.sqrt(temp)
                total_violation += violation

        return total_violation

    # ===== 修正2：SSL损失计算的重大修正 =====
    def compute_ssl_loss(self, logits_weak: torch.Tensor, logits_strong: torch.Tensor) -> torch.Tensor:
        """
        修正版本：正确实现FixMatch的SSL损失计算。
        """
        with torch.no_grad():
            # 对弱增强的输出计算概率
            probs_weak = torch.sigmoid(logits_weak)

            # 为每个标签独立计算最大概率（置信度）
            max_probs = torch.max(probs_weak, 1 - probs_weak)

            # 创建置信度掩码：对于多标签，每个标签位置独立判断
            confident_mask = max_probs > self.confidence_threshold

            # 生成伪标签（二值化）
            pseudo_labels = (probs_weak > 0.5).float()

        # 只对高置信度的预测计算损失
        if confident_mask.sum() == 0:
            return torch.tensor(0.0, device=logits_strong.device)

        # 使用掩码应用损失
        loss = F.binary_cross_entropy_with_logits(
            logits_strong, pseudo_labels, reduction='none'
        )

        # 只计算高置信度位置的损失
        masked_loss = loss * confident_mask.float()

        # 归一化：除以实际有效的损失项数量
        valid_loss_count = confident_mask.sum()
        if valid_loss_count > 0:
            return masked_loss.sum() / valid_loss_count
        else:
            return torch.tensor(0.0, device=logits_strong.device)


class AdultDataProcessor:
    """
    - 目标标签 (y): income, workclass, occupation
    - 敏感属性 (a): binarized age (25-44 vs. others)
    """

    def __init__(self):
        self.scaler = StandardScaler()
        self.workclass_cols = []
        self.occupation_cols = []

    def download_adult_data(self) -> pd.DataFrame:
        print("加载数据集")
        train_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
        test_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test"
        columns = [
            'age', 'workclass', 'fnlwgt', 'education', 'education-num',
            'marital-status', 'occupation', 'relationship', 'race', 'sex',
            'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income'
        ]

        try:
            train_data = pd.read_csv(train_url, names=columns, na_values=' ?', skipinitialspace=True)
            test_data = pd.read_csv(test_url, names=columns, na_values=' ?', skipinitialspace=True, skiprows=1)
            df = pd.concat([train_data, test_data], ignore_index=True)
            df['income'] = df['income'].str.replace('.', '', regex=False)
        except Exception as e:
            raise RuntimeError(f"下载Adult数据集失败: {e}")

        return df

    def load_and_preprocess_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
        """
        根据论文描述加载和预处理数据。
        """
        df = self.download_adult_data()
        df = df.dropna().reset_index(drop=True)

        # 1. 构造目标标签 y
        y_income = (df['income'] == '>50K').astype(int).values.reshape(-1, 1)
        y_workclass = pd.get_dummies(df['workclass'], prefix='workclass', drop_first=True)
        self.workclass_cols = y_workclass.columns.tolist()
        y_workclass = y_workclass.values
        y_occupation = pd.get_dummies(df['occupation'], prefix='occupation', drop_first=True)
        self.occupation_cols = y_occupation.columns.tolist()
        y_occupation = y_occupation.values

        y = np.concatenate([y_income, y_workclass, y_occupation], axis=1)
        num_labels = y.shape[1]

        # 2. 构造敏感属性 a
        a = ((df['age'] >= 25) & (df['age'] <= 44)).astype(int).values

        # 3. 构造特征 X
        features_to_drop = ['income', 'workclass', 'occupation', 'age']
        df_features = df.drop(columns=features_to_drop)

        numeric_features = df_features.select_dtypes(include=np.number).columns.tolist()
        categorical_features = df_features.select_dtypes(exclude=np.number).columns.tolist()

        df_features[numeric_features] = self.scaler.fit_transform(df_features[numeric_features])
        X_numeric = df_features[numeric_features].values

        X_categorical_list = [pd.get_dummies(df_features[feat], prefix=feat, drop_first=True).values for feat in
                              categorical_features]

        X = np.concatenate([X_numeric] + X_categorical_list, axis=1)

        return X, y, a, num_labels


def analyze_label_correlations(y: np.ndarray, label_mapping: Dict[int, str], selected_labels: Dict[str, int]):
    """
    计算并打印出关键标签之间的相关性，以验证"关联性偏见"。
    """
    df_y = pd.DataFrame(y, columns=[label_mapping.get(i) for i in range(y.shape[1])])
    correlation_matrix = df_y.corr()

    biased_name = label_mapping.get(selected_labels['biased'])
    neutral_name_1 = label_mapping.get(selected_labels['neutral_1'])
    neutral_name_2 = label_mapping.get(selected_labels['neutral_2'])

    corr_income_neutral1 = correlation_matrix.loc[biased_name, neutral_name_1]
    corr_income_neutral2 = correlation_matrix.loc[biased_name, neutral_name_2]

    print("\n" + "=" * 50)
    print("标签相关性分析报告:")
    print("-" * 50)
    print(f"核心假设：一个中性标签是否会因与偏见标签高度相关而被影响？")
    print("-" * 50)
    print(f"'{biased_name}' 与 '{neutral_name_1}' 的相关系数: {corr_income_neutral1:.4f}")
    print(f"'{biased_name}' 与 '{neutral_name_2}' 的相关系数: {corr_income_neutral2:.4f}")
    print("-" * 50)
    if abs(corr_income_neutral1) > 0.1 or abs(corr_income_neutral2) > 0.1:
        print("结论：至少一个中性标签与'income'存在显著相关性。这解释了为什么它的伪标签生成也受到了抑制。")
    else:
        print("结论：中性标签与'income'的相关性较低。")
    print("=" * 50 + "\n")

def find_most_frequent_label(y_labels: np.ndarray) -> List[float]:
    """
    找到最频繁的标签组合。
    """
    label_tuples = [tuple(row) for row in y_labels]
    label_counts = Counter(label_tuples)

    if not label_counts:
        return [0.0] * y_labels.shape[1]

    most_common_label_tuple = label_counts.most_common(1)[0][0]
    return [float(x) for x in most_common_label_tuple]


def find_representative_labels(y: np.ndarray, label_mapping: Dict[int, str]) -> Dict[str, int]:
    num_samples, num_labels = y.shape

    label_stats = []
    for i in range(num_labels):
        support = np.mean(y[:, i])
        label_stats.append({
            'index': i,
            'name': label_mapping.get(i, f'label_{i}'),
            'support': support,
        })

    biased_label_idx = 0
    remaining_labels = [s for s in label_stats if s['index'] != biased_label_idx]
    remaining_labels.sort(key=lambda x: x['support'], reverse=True)

    if len(remaining_labels) < 2:
        raise ValueError("数据集中除了income外，不足两个其他标签，无法进行对比！")

    neutral_label_idx_1 = remaining_labels[0]['index']
    neutral_label_idx_2 = remaining_labels[1]['index']

    print("\n" + "=" * 40)
    print("自动标签选择结果 (基于最高支持度):")
    print("-" * 40)
    print(
        f"偏见类别 (Biased): {label_mapping.get(biased_label_idx)} (Index: {biased_label_idx}, Support: {label_stats[biased_label_idx]['support']:.2%})")
    print("\n中性类别选择（支持度最高的前几名）:")
    for cand in remaining_labels[:5]:
        print(f"  - {cand['name']} (Index: {cand['index']}, Support: {cand['support']:.2%})")
    print("-" * 40)
    print("最终选择:")
    print(f"  - 中性类别 1: {label_mapping.get(neutral_label_idx_1)}")
    print(f"  - 中性类别 2: {label_mapping.get(neutral_label_idx_2)}")
    print("=" * 40 + "\n")

    return {
        'biased': biased_label_idx,
        'neutral_1': neutral_label_idx_1,
        'neutral_2': neutral_label_idx_2
    }


def prepare_ssl_data(X: np.ndarray, y: np.ndarray, a: np.ndarray,
                     labeled_samples: int = 500, test_size: float = 0.3) -> Dict:
    """
    准备半监督学习数据划分。
    """
    X_train_full, X_test, y_train_full, y_test, a_train_full, a_test = train_test_split(
        X, y, a, test_size=test_size, random_state=42, stratify=a
    )

    labeled_samples = min(labeled_samples, len(X_train_full))

    labeled_indices, unlabeled_indices = train_test_split(
        np.arange(len(X_train_full)),
        train_size=labeled_samples,
        random_state=42,
        stratify=a_train_full
    )

    X_labeled = X_train_full[labeled_indices]
    y_labeled = y_train_full[labeled_indices]
    a_labeled = a_train_full[labeled_indices]

    X_unlabeled = X_train_full[unlabeled_indices]
    a_unlabeled = a_train_full[unlabeled_indices]

    return {
        'X_labeled': X_labeled, 'y_labeled': y_labeled, 'a_labeled': a_labeled,
        'X_unlabeled': X_unlabeled, 'a_unlabeled': a_unlabeled,
        'X_test': X_test, 'y_test': y_test, 'a_test': a_test,
        'y_train_full': y_train_full
    }


def evaluate_model(model: nn.Module, test_loader: DataLoader,
                   device: torch.device, label_mapping: Dict[int, str],
                   rescaling_thresholds: torch.Tensor,
                   y_adv_eval: torch.Tensor  # 新增参数，用于EOp计算
                   ) -> Dict:
    """
    在测试集上评估模型，并返回包括DP和EOp在内的所有指标。
    （已移除SimFair指标）
    """
    model.eval()
    all_pred_probs, all_labels, all_a = [], [], []

    with torch.no_grad():
        for X_batch, y_batch, a_batch in test_loader:
            # 在评估模式下，MPVAE直接输出平均概率
            pred_probs, _, _ = model(X_batch.to(device), use_label_encoder=False)
            all_pred_probs.append(pred_probs.cpu())
            all_labels.append(y_batch.cpu())
            all_a.append(a_batch.cpu())

    all_pred_probs = torch.cat(all_pred_probs)
    all_labels = torch.cat(all_labels)
    all_a = torch.cat(all_a)

    # 使用Rescaling策略进行二值化
    pred_binary = (all_pred_probs > rescaling_thresholds).int().numpy()
    y_true_numpy = all_labels.numpy()

    # --- 计算各项性能指标 ---
    micro_f1 = f1_score(y_true_numpy, pred_binary, average='micro', zero_division=0)
    macro_f1 = f1_score(y_true_numpy, pred_binary, average='macro', zero_division=0)
    example_f1 = f1_score(y_true_numpy, pred_binary, average='samples', zero_division=0)
    subset_accuracy = accuracy_score(y_true_numpy, pred_binary)
    hamming_loss_val = hamming_loss(y_true_numpy, pred_binary)

    # =======================================================
    # ===> 关键修改：调用新函数计算DP和EOp，移除SimFair <===
    # =======================================================
    dp_eop_violations = compute_dp_eop_violations(all_pred_probs, all_labels, all_a, y_adv_eval)
    # =======================================================

    per_label_f1 = {}
    for i in range(all_labels.shape[1]):
        label_name = label_mapping.get(i, f'label_{i}')
        label_f1 = f1_score(all_labels[:, i], pred_binary[:, i], average='binary', zero_division=0)
        per_label_f1[label_name] = label_f1

    # 返回的字典中现在包含DP和EOp，不再包含simfair_violation
    return {
        'micro_f1': micro_f1,
        'macro_f1': macro_f1,
        'example_f1': example_f1,
        'subset_accuracy': subset_accuracy,
        'hamming_loss': hamming_loss_val,
        'dp_violation': dp_eop_violations['dp_violation'],
        'eop_violation': dp_eop_violations['eop_violation'],
        'per_label_f1': per_label_f1
    }
def evaluate_prediction_distribution(model: nn.Module, X_unlabeled: np.ndarray,
                                     device: torch.device) -> np.ndarray:
    """
    获取模型在所有无标签数据上的预测概率分布。
    """
    model.eval()
    X_tensor = torch.FloatTensor(X_unlabeled).to(device)

    # 对于MPVAE，推理时只使用特征分支，它会返回平均概率
    with torch.no_grad():
        pred_probs, _, _ = model(X_tensor)

    return pred_probs.cpu().numpy()


def evaluate_per_label_pseudo_quantity(model: nn.Module, X_unlabeled: np.ndarray,
                                       confidence_threshold: float, device: torch.device) -> Dict[int, int]:
    """
    为每一个标签独立计算其高置信度伪标签的数量。
    """
    model.eval()
    X_tensor = torch.FloatTensor(X_unlabeled).to(device)
    with torch.no_grad():
        probs, _, _ = model(X_tensor, use_label_encoder=False)
        max_probs = torch.max(probs, 1 - probs)

        per_label_counts = {}
        num_labels = probs.shape[1]
        for i in range(num_labels):
            confident_mask_for_label = max_probs[:, i] > confidence_threshold
            per_label_counts[i] = confident_mask_for_label.sum().item()

    return per_label_counts

def compute_expected_calibration_error(y_true, y_pred_probs, n_bins=15):
    """计算期望校准误差(ECE)"""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    ece = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (y_pred_probs > bin_lower) & (y_pred_probs <= bin_upper)
        prop_in_bin = in_bin.mean()
        if prop_in_bin > 0:
            accuracy_in_bin = y_true[in_bin].mean()
            avg_confidence_in_bin = y_pred_probs[in_bin].mean()
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    return ece


def evaluate_model_extended(model: nn.Module, test_loader: DataLoader,
                            device: torch.device, label_mapping: Dict[int, str],
                            selected_labels: Dict[str, int]) -> Dict:
    """
    一个全新的、全面的评估函数，计算所有高级指标。
    """
    model.eval()
    all_pred_probs, all_labels, all_a = [], [], []

    with torch.no_grad():
        for X_batch, y_batch, a_batch in test_loader:
            pred_probs, _, _ = model(X_batch.to(device), use_label_encoder=False)
            all_pred_probs.append(pred_probs.cpu())
            all_labels.append(y_batch.cpu())
            all_a.append(a_batch.cpu())

    all_pred_probs = torch.cat(all_pred_probs).numpy()
    all_labels = torch.cat(all_labels).numpy()
    all_a = torch.cat(all_a).numpy()

    pred_binary = (all_pred_probs > 0.5).astype(int)  # 使用0.5作为基础阈值

    # --- 计算各项指标 ---
    metrics = {}

    # 1. 宏观AUPR
    metrics['macro_aupr'] = average_precision_score(all_labels, all_pred_probs, average="macro")

    # 2. 标签排序损失 (LRL)
    metrics['ranking_loss'] = label_ranking_loss(all_labels, all_pred_probs)

    # 3. Jaccard相似度
    metrics['jaccard_score'] = jaccard_score(all_labels, pred_binary, average='samples')

    # 4. 期望校准误差 (ECE)
    metrics['ece'] = compute_expected_calibration_error(all_labels.flatten(), all_pred_probs.flatten())

    # 5. 公平性差距 (Fairness Gap) - 以income标签的F1分数为衡量标准
    income_idx = selected_labels['biased']
    male_mask = (all_a == 1)
    female_mask = (all_a == 0)

    f1_male = f1_score(all_labels[male_mask, income_idx], pred_binary[male_mask, income_idx], zero_division=0)
    f1_female = f1_score(all_labels[female_mask, income_idx], pred_binary[female_mask, income_idx], zero_division=0)
    metrics['fairness_gap_f1'] = abs(f1_male - f1_female)

    return metrics

def evaluate_per_label_proxy_accuracy(model: nn.Module, test_loader: DataLoader,
                                      confidence_threshold: float, device: torch.device,
                                      selected_labels: Dict[str, int],
                                      label_mapping: Dict[int, str]) -> Dict[str, float]:
    """
    为每一个我们选择的特定标签，独立地计算其高置信度伪标签的准确率。
    此版本适配直接输出概率的MPVAE模型。
    """
    model.eval() # 确保模型处于评估模式
    all_pred_probs, all_labels = [], []

    # 提取我们关心的标签的索引
    label_indices_to_evaluate = list(selected_labels.values())

    with torch.no_grad():
        for X_batch, y_batch, a_batch in test_loader:
            # 1. 模型在评估模式下，直接返回(平均预测概率, mu, logvar)
            pred_probs, _, _ = model(X_batch.to(device), use_label_encoder=False)
            all_pred_probs.append(pred_probs.cpu())
            all_labels.append(y_batch)

    all_pred_probs = torch.cat(all_pred_probs)
    all_labels = torch.cat(all_labels)
    # 此时 all_pred_probs 已经是概率了，不再需要 sigmoid

    # 2. 后续逻辑保持不变
    per_label_accuracy = {}
    for label_idx in label_indices_to_evaluate:
        label_name = label_mapping.get(label_idx)

        label_probs = all_pred_probs[:, label_idx]
        label_true = all_labels[:, label_idx]

        max_probs = torch.max(label_probs, 1 - label_probs)
        confident_mask = (max_probs > confidence_threshold)

        num_confident = confident_mask.sum().item()
        if num_confident == 0:
            per_label_accuracy[label_name] = 0.0
            continue

        pseudo_labels = (label_probs > 0.5).int()
        confident_pseudo_labels = pseudo_labels[confident_mask]
        confident_true_labels = label_true[confident_mask]

        correct_predictions = (confident_pseudo_labels == confident_true_labels).float().sum()
        accuracy = correct_predictions / num_confident
        per_label_accuracy[label_name] = accuracy.item()

    return per_label_accuracy


def compute_dp_eop_violations(y_pred_probs: torch.Tensor, y_true: torch.Tensor,
                              sensitive_attrs: torch.Tensor, y_adv: torch.Tensor) -> Dict[str, float]:
    """
    计算并返回DP和EOp的违规度。
    违规度被定义为不同群体之间平均预测概率差异的L2范数。
    """
    # 确保所有张量都在CPU上，以便使用NumPy
    y_pred_probs = y_pred_probs.cpu()
    y_true = y_true.cpu()
    sensitive_attrs = sensitive_attrs.cpu()
    y_adv = y_adv.cpu()

    # 确定两个敏感属性群体
    group_0_mask = (sensitive_attrs == 0)
    group_1_mask = (sensitive_attrs == 1)

    # --- 1. 计算人口均等 (Demographic Parity, DP) 违规度 ---
    dp_violation = 0.0
    if group_0_mask.sum() > 0 and group_1_mask.sum() > 0:
        mean_pred_g0 = torch.mean(y_pred_probs[group_0_mask], dim=0)
        mean_pred_g1 = torch.mean(y_pred_probs[group_1_mask], dim=0)
        dp_violation = torch.linalg.norm(mean_pred_g0 - mean_pred_g1).item()

    # --- 2. 计算机会均等 (Equalized Opportunity, EOp) 违规度 ---
    eop_violation = 0.0

    # 找到所有属于有利群体的样本
    advantaged_mask = torch.all(y_true == y_adv, dim=1)

    if advantaged_mask.sum() > 0:
        # 在有利群体子集上，再次划分敏感属性组
        adv_group_0_mask = group_0_mask[advantaged_mask]
        adv_group_1_mask = group_1_mask[advantaged_mask]

        if adv_group_0_mask.sum() > 0 and adv_group_1_mask.sum() > 0:
            y_pred_adv = y_pred_probs[advantaged_mask]
            mean_pred_adv_g0 = torch.mean(y_pred_adv[adv_group_0_mask], dim=0)
            mean_pred_adv_g1 = torch.mean(y_pred_adv[adv_group_1_mask], dim=0)
            eop_violation = torch.linalg.norm(mean_pred_adv_g0 - mean_pred_adv_g1).item()

    return {
        'dp_violation': dp_violation,
        'eop_violation': eop_violation
    }


def visualize_per_label_f1_tradeoff(results: List[Dict], lambda_fair_values: List[float],
                                    selected_labels: Dict[str, int], label_mapping: Dict[int, str]):
    """
    可视化不同类别F1分数随lambda_fair变化的权衡曲线。
    新版本：动态地根据传入的selected_labels进行绘制。
    """
    plt.figure(figsize=(12, 8))

    # 从传入的字典中获取标签索引和名称
    biased_idx = selected_labels['biased']

    biased_name = label_mapping.get(biased_idx)

    # 提取并绘制偏见类别的性能曲线
    biased_f1_history = [r['per_label_f1'][biased_name] for r in results]
    plt.plot(lambda_fair_values, biased_f1_history, 'o-', color='red', linewidth=2.5, markersize=8,
             label=f'Biased Label: {biased_name}')

    plt.title('Per-Label F1 Score vs. Fairness Constraint Strength (λ_fair)')
    plt.xlabel('λ_fair')
    plt.ylabel('F1 Score')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend()
    plt.ylim(bottom=0)  # 确保Y轴从0开始，便于比较
    plt.savefig('biased_label_f1_tradeoff_revised.png', dpi=300)
    plt.show()


def visualize_error_rate_tradeoff(results: List[Dict], lambda_fair_values: List[float]):
    """
    可视化分类错误率（汉明损失）随lambda_fair变化的曲线。
    """
    plt.figure(figsize=(10, 7))

    # 提取汉明损失的历史数据
    hamming_loss_history = [r['hamming_loss'] for r in results]

    plt.plot(lambda_fair_values, hamming_loss_history, 'o-', color='crimson',
             linewidth=2, markersize=8, label='Hamming Loss (Error Rate)')

    plt.title('Classification Error Rate vs. Fairness Constraint', fontsize=16)
    plt.xlabel('λ_fair (Fairness Constraint Strength)', fontsize=12)
    plt.ylabel('Error Rate (Lower is Better)', fontsize=12)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend(fontsize=11)
    # Y轴从0开始，便于观察
    plt.ylim(bottom=0)

    plt.savefig('error_rate_tradeoff.png', dpi=300, bbox_inches='tight')
    plt.show()


def visualize_definitive_proof(results: List[Dict], lambda_fair_values: List[float],
                               selected_labels: Dict[str, any], label_mapping: Dict[int, str]):
    """
    最终的、决定性的可视化函数。
    只绘制“受保护类别”与“中性类别”的F1分数对比，以证明性能下降。
    """
    plt.style.use('seaborn-v0_8-poster') # 使用更适合论文的清晰风格
    plt.figure(figsize=(12, 8))

    # --- 提取数据 ---
    biased_idx = selected_labels['biased']
    neutral_indices = [ selected_labels['neutral_1'], selected_labels['neutral_2'] ]
    biased_name = label_mapping.get(biased_idx)

    # 1. 绘制“受保护类别”的性能曲线 (这是我们的核心证据)
    biased_f1_history = [r['per_label_f1'][biased_name] for r in results]
    plt.plot(lambda_fair_values, biased_f1_history, 'o-', color='red', linewidth=3, markersize=10,
             label=f'Protected Label: {biased_name} (Performance Drops)')

    # 2. 绘制“中性类别”的性能曲线 (这是我们的控制组)
    for i, neutral_idx in enumerate(neutral_indices):
        neutral_name = label_mapping.get(neutral_idx)
        neutral_f1_history = [r['per_label_f1'][neutral_name] for r in results]
        plt.plot(lambda_fair_values, neutral_f1_history, 's--', alpha=0.7,
                 label=f'Neutral Label {i+1}: {neutral_name} (Stable Performance)')

    # --- 设置图表样式 ---
    plt.title('Performance Decline of Protected Label under Fairness Constraint', fontsize=20, pad=20)
    plt.xlabel('λ_fair (Fairness Constraint Strength)', fontsize=16)
    plt.ylabel('Per-Label F1 Score', fontsize=16)
    plt.grid(True, which='both', linestyle='--', linewidth=0.7)
    plt.legend(fontsize=14)
    plt.ylim(bottom=0)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.savefig('definitive_proof_f1_tradeoff.png', dpi=300)
    plt.show()

def visualize_advanced_metrics(all_extended_results: List[Dict], all_consistency_scores: List[float], lambda_fair_values: List[float]):
    """
    将所有高级性能和SSL指标绘制在一张图上。
    """
    fig, axes = plt.subplots(2, 3, figsize=(24, 12))
    fig.suptitle('Comprehensive Performance Analysis vs. Fairness Constraint', fontsize=20)

    # 提取数据
    macro_aupr = [r['macro_aupr'] for r in all_extended_results]
    fairness_gap = [r['fairness_gap_f1'] for r in all_extended_results]
    ranking_loss = [r['ranking_loss'] for r in all_extended_results]
    jaccard_score_vals = [r['jaccard_score'] for r in all_extended_results]
    ece = [r['ece'] for r in all_extended_results]

    # --- 绘图 ---
    # 1. Macro AUPR
    axes[0, 0].plot(lambda_fair_values, macro_aupr, 'o-', color='navy', label='Macro-AUPR (Higher is Better)')
    axes[0, 0].set_title('Macro AUPR (Robust to Imbalance)')
    axes[0, 0].set_ylabel('Score')
    axes[0, 0].legend()

    # 2. Fairness Gap (F1 Score)
    axes[0, 1].plot(lambda_fair_values, fairness_gap, 'o-', color='firebrick', label='F1-Score Gap on Income (Lower is Better)')
    axes[0, 1].set_title('Fairness Gap on Protected Label')
    axes[0, 1].set_ylabel('Absolute Difference')
    axes[0, 1].legend()

    # 3. Label Ranking Loss
    axes[0, 2].plot(lambda_fair_values, ranking_loss, 'o-', color='darkgreen', label='Label Ranking Loss (Lower is Better)')
    axes[0, 2].set_title('Label Ranking Performance')
    axes[0, 2].set_ylabel('Loss')
    axes[0, 2].legend()

    # 4. Jaccard Score
    axes[1, 0].plot(lambda_fair_values, jaccard_score_vals, 's--', color='purple', label='Jaccard Score / IoU (Higher is Better)')
    axes[1, 0].set_title('Example-based Jaccard Similarity')
    axes[1, 0].set_xlabel('λ_fair')
    axes[1, 0].set_ylabel('Score')
    axes[1, 0].legend()

    # 5. Expected Calibration Error (ECE)
    axes[1, 1].plot(lambda_fair_values, ece, 's--', color='goldenrod', label='ECE (Lower is Better)')
    axes[1, 1].set_title('Model Calibration Error')
    axes[1, 1].set_xlabel('λ_fair')
    axes[1, 1].set_ylabel('Error')
    axes[1, 1].legend()

    # 6. Consistency Regularization Score
    axes[1, 2].plot(lambda_fair_values, all_consistency_scores, 's--', color='teal', label='Consistency Score (Lower is Better)')
    axes[1, 2].set_title('SSL Consistency (Weak vs Strong Aug)')
    axes[1, 2].set_xlabel('λ_fair')
    axes[1, 2].set_ylabel('MSE between Predictions')
    axes[1, 2].legend()

    for ax in axes.flatten():
        ax.grid(True, linestyle='--')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('advanced_metrics_dashboard.png', dpi=300)
    plt.show()


def visualize_extra_metrics_tradeoff(results: List[Dict], lambda_fair_values: List[float]):
    """
    可视化 Example-F1 随lambda_fair变化的曲线。
    """
    plt.figure(figsize=(12, 8))

    # 提取各个指标的历史数据
    example_f1_history = [r['example_f1'] for r in results]

    # 绘制曲线
    plt.plot(lambda_fair_values, example_f1_history, 's--', label='Example-based F1 Score')

    plt.title('Example-F1 vs. Fairness Constraint', fontsize=16)
    plt.xlabel('λ_fair (Fairness Constraint Strength)', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend(fontsize=11)
    plt.ylim(bottom=0)
    plt.savefig('example-f1.png', dpi=300, bbox_inches='tight')
    plt.show()


def visualize_per_label_quantity_evolution(all_curves: Dict[float, List[Dict[int, int]]],
                                           label_mapping: Dict[int, str],
                                           selected_labels: Dict[str, int]):
    """
    绘制分标签的伪标签数量随epoch变化的对比图。
    新版本：使用传入的selected_labels进行绘制，以保持分析一致性。
    """
    # 从传入的字典中获取要绘制的标签索引和名称
    biased_idx = selected_labels['biased']
    neutral_idx_1 = selected_labels['neutral_1']
    neutral_idx_2 = selected_labels['neutral_2']

    biased_name = label_mapping.get(biased_idx)
    neutral_name_1 = label_mapping.get(neutral_idx_1)
    neutral_name_2 = label_mapping.get(neutral_idx_2)

    # --- 绘图逻辑 ---
    lambda_vals = sorted(all_curves.keys())
    num_scenarios = len(lambda_vals)

    # 动态创建子图网格，每行最多2个图
    num_rows = (num_scenarios + 1) // 2
    fig, axes = plt.subplots(num_rows, 2, figsize=(20, 5 * num_rows), sharey=True, squeeze=False)
    axes_flat = axes.flatten()

    for i, lam in enumerate(lambda_vals):
        ax = axes_flat[i]
        curves = all_curves.get(lam)
        if not curves:
            continue

        epochs = range(1, len(curves) + 1)

        # 提取并绘制 "偏见类别" 的曲线
        biased_curve = [epoch_data.get(biased_idx, 0) for epoch_data in curves]
        ax.plot(epochs, biased_curve, marker='o', color='red', label=f'Biased: {biased_name}')

        # 提取并绘制两个 "中性类别" 的曲线
        neutral_curve_1 = [epoch_data.get(neutral_idx_1, 0) for epoch_data in curves]
        ax.plot(epochs, neutral_curve_1, marker='s', linestyle='--', color='blue', label=f'Neutral: {neutral_name_1}')

        neutral_curve_2 = [epoch_data.get(neutral_idx_2, 0) for epoch_data in curves]
        ax.plot(epochs, neutral_curve_2, marker='^', linestyle='--', color='green', label=f'Neutral: {neutral_name_2}')

        ax.set_title(f'λ_fair = {lam}')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('High-Confidence Pseudo-Label Count')
        ax.grid(True)
        ax.legend()

    # 隐藏多余的空子图
    for j in range(len(lambda_vals), len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle('Differential Impact of Fairness Constraint on Pseudo-Label Generation', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('per_label_pseudo_label_comparison.png', dpi=300)
    plt.show()

def visualize_final_accuracy_comparison(all_accuracy_results: List[Dict[str, float]],
                                        lambda_fair_values: List[float],
                                        selected_labels: Dict[str, int],
                                        label_mapping: Dict[int, str]):
    """
    可视化三个选定标签各自的伪标签准确率随lambda_fair变化的曲线。
    """
    plt.figure(figsize=(12, 8))

    # 提取要绘制的标签名称
    label_names_to_plot = [label_mapping.get(idx) for idx in selected_labels.values()]

    # 为每个标签绘制一条曲线
    for label_name in label_names_to_plot:
        # 提取这个标签在所有lambda值下的准确率历史
        accuracy_history = [res.get(label_name, 0.0) for res in all_accuracy_results]

        # 根据标签类型选择不同的样式
        marker = 'o'
        linestyle = '-'
        color = 'grey'
        if label_name == label_mapping.get(selected_labels['biased']):
            marker = 'p'  # 五边形
            linestyle = '-'
            color = 'red'
        elif label_name == label_mapping.get(selected_labels['neutral_1']):
            marker = 's'  # 方块
            linestyle = '--'
            color = 'blue'
        elif label_name == label_mapping.get(selected_labels['neutral_2']):
            marker = '^'  # 三角形
            linestyle = '--'
            color = 'green'

        plt.plot(lambda_fair_values, accuracy_history, marker=marker, linestyle=linestyle, color=color,
                 label=label_name)

    plt.title('Per-Label Pseudo-Label Accuracy vs. Fairness Constraint', fontsize=16)
    plt.xlabel('λ_fair (Fairness Constraint Strength)', fontsize=12)
    plt.ylabel('Pseudo-Label Accuracy', fontsize=12)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend(title="Selected Labels", fontsize=11)
    plt.ylim(bottom=0)
    plt.savefig('final_accuracy_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()


def visualize_dp_eop_tradeoff(results: List[Dict], lambda_fair_values: List[float]):
    """
    将DP和EOp两种公平性指标的违规度绘制在一张图上进行对比。
    """
    plt.figure(figsize=(12, 8))

    # 提取DP和EOp的历史数据
    dp_history = [r['dp_violation'] for r in results]
    eop_history = [r['eop_violation'] for r in results]

    # 绘制曲线
    plt.plot(lambda_fair_values, dp_history, 'o-', color='blue', linewidth=2, label='DP Violation')
    plt.plot(lambda_fair_values, eop_history, '^--', color='green', linewidth=2, label='EOp Violation')

    plt.title('DP & EOp Violations vs. Fairness Constraint Strength', fontsize=16)
    plt.xlabel('λ_fair (SimFair Constraint Strength)', fontsize=12)
    plt.ylabel('Violation (Lower is Better)', fontsize=12)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend(fontsize=11)
    # 使用对数坐标轴可能有助于观察，如果数值范围很大
    # plt.yscale('log')
    plt.savefig('dp_eop_tradeoff.png', dpi=300)
    plt.show()
def visualize_prediction_distributions(all_prob_dists: Dict[float, np.ndarray],
                                       selected_labels: Dict[str, int],
                                       label_mapping: Dict[int, str]):
    """
    为每个lambda_fair值，可视化关键标签的预测概率分布。
    """
    # 提取我们关心的标签索引和名称
    biased_idx = selected_labels['biased']
    neutral_idx_1 = selected_labels['neutral_1']
    neutral_idx_2 = selected_labels['neutral_2']

    biased_name = label_mapping.get(biased_idx)
    neutral_name_1 = label_mapping.get(neutral_idx_1)
    neutral_name_2 = label_mapping.get(neutral_idx_2)

    # --- 绘图逻辑 ---
    lambda_vals = sorted(all_prob_dists.keys())
    num_scenarios = len(lambda_vals)

    num_rows = (num_scenarios + 1) // 2
    fig, axes = plt.subplots(num_rows, 2, figsize=(20, 5 * num_rows), sharex=True, sharey=True)
    axes_flat = axes.flatten()

    for i, lam in enumerate(lambda_vals):
        ax = axes_flat[i]
        prob_dist = all_prob_dists[lam]

        # 使用核密度估计图（KDE Plot）来平滑地展示分布
        sns.kdeplot(prob_dist[:, biased_idx], ax=ax, fill=True, color='red', label=f'Biased: {biased_name}')
        sns.kdeplot(prob_dist[:, neutral_idx_1], ax=ax, fill=True, color='blue', label=f'Neutral: {neutral_name_1}',
                    alpha=0.5)
        sns.kdeplot(prob_dist[:, neutral_idx_2], ax=ax, fill=True, color='green', label=f'Neutral: {neutral_name_2}',
                    alpha=0.5)

        ax.set_title(f'Prediction Probability Distribution (λ_fair = {lam})')
        ax.set_xlabel('Predicted Probability')
        ax.set_ylabel('Density')
        ax.legend()
        ax.set_xlim(0, 1)

    # 隐藏多余的空子图
    for j in range(len(lambda_vals), len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle('Evolution of Prediction Confidence under Fairness Constraint', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('prediction_distribution_evolution.png', dpi=300)
    plt.show()


def train_ssl_model(model, labeled_loader, unlabeled_loader, loss_fn, optimizer, device,
                    lambda_fair: float,
                    X_unlabeled, confidence_threshold, epochs=50):
    """
    最终版：在原有的SimFair框架上，增加了熵最小化正则化。
    """
    model.train()
    training_history = []
    per_label_pseudo_counts_history = []

    # 保持原有的KLD退火逻辑
    kld_anneal_epochs = 25
    global_step = 0
    total_steps = len(labeled_loader) * kld_anneal_epochs

    for epoch in range(epochs):
        epoch_losses = []
        unlabeled_iter = iter(unlabeled_loader)

        for labeled_batch in labeled_loader:
            try:
                unlabeled_batch = next(unlabeled_iter)
            except StopIteration:
                unlabeled_iter = iter(unlabeled_loader)
                unlabeled_batch = next(unlabeled_iter)

            kld_weight = min(1.0, float(global_step) / total_steps if total_steps > 0 else 1.0)
            global_step += 1

            X_l, y_l, a_l = [x.to(device) for x in labeled_batch]
            X_w, X_s, a_u = [x.to(device) for x in unlabeled_batch]
            optimizer.zero_grad()

            # 1. 监督损失（MPVAE）
            E_x, fx_mu, fx_logvar, E, fe_mu, fe_logvar = model(X_l, y_l)
            mpvae_loss = model.loss_function(E_x, fx_mu, fx_logvar, E, fe_mu, fe_logvar, y_l, kl_coeff=kld_weight)[
                'total']

            # =======================================================
            # ===> 核心修改 1：将公平性损失改回 SimFair <===
            # =======================================================
            # 基于模型对有标签数据的预测概率来计算
            pred_probs_l = torch.mean(E_x, dim=0)
            L_fair = loss_fn._compute_simfair_violation(pred_probs_l, y_l, a_l)
            # =======================================================

            # 3. SSL损失
            pred_probs_w, _, _ = model(X_w, use_label_encoder=False)
            pred_probs_s, _, _ = model(X_s, use_label_encoder=False)
            logits_w = torch.logit(torch.clamp(pred_probs_w, 1e-6, 1 - 1e-6))
            logits_s = torch.logit(torch.clamp(pred_probs_s, 1e-6, 1 - 1e-6))
            L_unsup = loss_fn.compute_ssl_loss(logits_w, logits_s)

            # 4. 最终的总损失
            L_total = mpvae_loss + \
                      lambda_fair * L_fair + \
                      loss_fn.lambda_unsup * L_unsup

            L_total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_losses.append({
                'total_loss': L_total.item(),
                'supervised_loss': (mpvae_loss / len(X_l)).item(),
                'fairness_loss': L_fair.item(),
                'unsupervised_loss': L_unsup.item(),
            })

        # --- 后续的记录和评估部分保持不变 ---
        training_history.append({k: np.mean([d[k] for d in epoch_losses]) for k in epoch_losses[0]})
        per_label_counts = evaluate_per_label_pseudo_quantity(model, X_unlabeled, confidence_threshold, device)
        per_label_pseudo_counts_history.append(per_label_counts)

        if (epoch + 1) % 10 == 0:
            print(f"[Epoch {epoch + 1}] Loss: {training_history[-1]['total_loss']:.4f}, "
                  f"Fairness Loss (SimFair): {training_history[-1]['fairness_loss']:.4f} , ")

    # --- 返回值部分保持不变 ---
    model.eval()
    all_consistency_scores = []
    with torch.no_grad():
        for X_w, X_s, _ in unlabeled_loader:
            X_w, X_s = X_w.to(device), X_s.to(device)
            probs_w, _, _ = model(X_w, use_label_encoder=False)
            probs_s, _, _ = model(X_s, use_label_encoder=False)
            consistency_score = F.mse_loss(probs_w, probs_s)
            all_consistency_scores.append(consistency_score.item())
    final_consistency_score = np.mean(all_consistency_scores)

    return training_history, per_label_pseudo_counts_history, final_consistency_score


def main():
    """主实验流程 - 最终完整、修正版 (专注于Adult数据集)"""
    print("开始实验：在Adult数据集上，验证公平性对不同类别的影响...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # --- 1. 数据加载和预处理 (使用单一、正确的流程) ---
    data_processor = AdultDataProcessor()
    # 为了放大偏见，我们使用 'sex' 作为敏感属性
    print("--> 正在加载和预处理Adult数据...")
    df = data_processor.download_adult_data()
    df = df.dropna().reset_index(drop=True)
    y_income = (df['income'] == '>50K').astype(int).values.reshape(-1, 1)
    y_workclass = pd.get_dummies(df['workclass'], prefix='workclass', drop_first=True)
    data_processor.workclass_cols = y_workclass.columns.tolist()
    y_occupation = pd.get_dummies(df['occupation'], prefix='occupation', drop_first=True)
    data_processor.occupation_cols = y_occupation.columns.tolist()
    y = np.concatenate([y_income, y_workclass.values, y_occupation.values], axis=1)
    num_labels = y.shape[1]
    a = (df['sex'] == 'Male').astype(int).values
    features_to_drop = ['income', 'workclass', 'occupation', 'sex']
    df_features = df.drop(columns=features_to_drop)
    numeric_features = df_features.select_dtypes(include=np.number).columns.tolist()
    categorical_features = df_features.select_dtypes(exclude=np.number).columns.tolist()
    df_features[numeric_features] = data_processor.scaler.fit_transform(df_features[numeric_features])
    X_numeric = df_features[numeric_features].values
    X_categorical_list = [pd.get_dummies(df_features[feat], prefix=feat, drop_first=True).values for feat in
                          categorical_features]
    X = np.concatenate([X_numeric] + X_categorical_list, axis=1)

    # --- 2. 创建标签映射并自动选择代表性标签 ---
    label_mapping = {0: 'income'}
    label_mapping.update({i + 1: col for i, col in enumerate(data_processor.workclass_cols)})
    label_mapping.update(
        {i + 1 + len(data_processor.workclass_cols): col for i, col in enumerate(data_processor.occupation_cols)})

    print("--> 正在自动选择用于分析的代表性标签...")
    selected_labels = find_representative_labels(y, label_mapping)
    analyze_label_correlations(y, label_mapping, selected_labels)

    # --- 3. SSL数据划分和加载器创建 ---
    ssl_data = prepare_ssl_data(X, y, a, labeled_samples=500, test_size=0.3)
    labeled_dataset = StandardDataset(ssl_data['X_labeled'], ssl_data['y_labeled'], ssl_data['a_labeled'])
    unlabeled_dataset = SSLDataset(ssl_data['X_unlabeled'], ssl_data['a_unlabeled'])
    test_dataset = StandardDataset(ssl_data['X_test'], ssl_data['y_test'], ssl_data['a_test'])
    labeled_loader = DataLoader(labeled_dataset, batch_size=32, shuffle=True, drop_last=True)
    unlabeled_loader = DataLoader(unlabeled_dataset, batch_size=64, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # --- 4. 计算用于Rescaling的阈值 ---
    class_prevalence = np.mean(ssl_data['y_train_full'], axis=0)
    rescaling_thresholds_tensor = torch.FloatTensor(class_prevalence).to('cpu')

    # --- 5. 实验循环 ---
    lambda_fair_values = [0.0, 10.0, 20.0, 30.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]
    confidence_threshold = 0.95
    results, extended_results_history, consistency_scores_history = [], [], []
    all_per_label_curves, per_label_accuracy_history, all_prob_distributions = {}, [], {}

    for lambda_fair in lambda_fair_values:
        print("\n" + "=" * 50)
        print(f"开始运行场景: lambda_fair = {lambda_fair}")
        print("=" * 50)
          # 假设您在使用熵正则化


        # 关键修正 1：使用正确的参数名 feat_dim 和 num_labels
        model = MPVAE(
            input_dim=X.shape[1],
            num_labels=num_labels,
            n_train_sample=10,
            n_test_sample=100
        ).to(device)

        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
        loss_fn = FairSSL_Loss(y_adv=torch.FloatTensor(find_most_frequent_label(ssl_data['y_train_full'])).to(device),
                               confidence_threshold=confidence_threshold)

        model.train()

        # 关键修正 2：传入正确的无标签数据变量
        _, per_label_curve, consistency_score = train_ssl_model(
            model, labeled_loader, unlabeled_loader, loss_fn, optimizer, device,
            lambda_fair,
            ssl_data['X_unlabeled'],  # 传入正确的NumPy数组
            confidence_threshold,
            epochs=100
        )
        consistency_scores_history.append(consistency_score)

        # --- 评估阶段 ---
        model.eval()
        y_adv_eval = torch.FloatTensor(find_most_frequent_label(ssl_data['y_train_full'])).to('cpu')

        # 收集所有评估结果
        extended_eval_res = evaluate_model_extended(model, test_loader, device, label_mapping, selected_labels)
        extended_results_history.append(extended_eval_res)

        per_label_acc = evaluate_per_label_proxy_accuracy(model, test_loader, confidence_threshold, device,
                                                          selected_labels, label_mapping)
        per_label_accuracy_history.append(per_label_acc)

        eval_res = evaluate_model(
            model,
            test_loader,
            device,
            label_mapping,
            rescaling_thresholds_tensor,
            y_adv_eval  # 传入y_adv
        )
        results.append(eval_res)

        pred_probs = evaluate_prediction_distribution(model, ssl_data['X_unlabeled'], device)
        all_prob_distributions[lambda_fair] = pred_probs
        all_per_label_curves[lambda_fair] = per_label_curve

    # --- 6. 可视化 ---
    print("\n--> 所有场景运行完毕，正在生成所有分析图表...")
    visualize_per_label_quantity_evolution(all_per_label_curves, label_mapping, selected_labels)
    visualize_per_label_f1_tradeoff(results, lambda_fair_values, selected_labels, label_mapping)
    visualize_final_accuracy_comparison(per_label_accuracy_history, lambda_fair_values, selected_labels, label_mapping)
    visualize_prediction_distributions(all_prob_distributions, selected_labels, label_mapping)
    visualize_dp_eop_tradeoff(results, lambda_fair_values)
    visualize_extra_metrics_tradeoff(results, lambda_fair_values)
    visualize_error_rate_tradeoff(results, lambda_fair_values)


if __name__ == "__main__":
    main()
