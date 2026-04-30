"""
风电场数据分析 — 公共模块
============================
提取所有脚本共享的函数、常量和配置，消除代码重复。
包含：数据预处理、DBSCAN去噪、归一化、相关性分析、
      K-means聚类、模型训练与评估、结果可视化。
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score, mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
import os
import json
import warnings
from typing import Optional, List, Dict, Any, Tuple

warnings.filterwarnings('ignore')

# ============================================================
# 中文字体配置（统一设置，避免每个脚本重复）
# ============================================================
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# ============================================================
# 全局常量
# ============================================================
NUMERIC_COLS = ['WINDSPEED', 'WINDDIRECTION', 'TEMPERATURE', 'HUMIDITY', 'PRESSURE', 'WINDPOWER']
CORRELATION_THRESHOLD = 0.2
SILHOUETTE_SAMPLE_SIZE = 5000
DBSCAN_SAMPLE_SIZE = 5000
DBSCAN_MIN_SAMPLES = 10
DBSCAN_N_NEIGHBORS = 10
DBSCAN_EPS_PERCENTILE = 95
RANDOM_SEED = 42

# 输出目录映射
OUTPUT_DIRS = {
    'task1': 'RW1',
    'task2': 'RW2',
    'task3': 'RW3',
    'task4': 'RW4(SOLO)',
}


# ============================================================
# 工具函数
# ============================================================

def ensure_dir(filepath):
    """确保输出文件的父目录存在。若 filepath 无父目录（纯文件名），则跳过创建。"""
    dirname = os.path.dirname(filepath)
    if dirname:
        os.makedirs(dirname, exist_ok=True)


def save_selected_features(features, filepath='selected_features.json'):
    """将筛选后的特征列表保存为 JSON 文件，供下游脚本读取。"""
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(list(features), f, ensure_ascii=False, indent=2)
    print(f"已保存特征列表: {filepath}")


def load_selected_features(filepath='selected_features.json'):
    """从 JSON 文件读取特征列表。"""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


# ============================================================
# 数据加载与预处理
# ============================================================

def load_and_preprocess(data_path='DATE.csv'):
    """
    加载原始数据、解析时间、排序、填充缺失值、剔除物理异常值。

    返回: (df_raw, df_clean)
        - df_raw:  排序后的原始 DataFrame（含 DATATIME）
        - df_clean: 剔除负值后的 DataFrame
    """
    print("=" * 60)
    print("数据加载与预处理")
    print("=" * 60)

    df = pd.read_csv(data_path)
    print(f"数据集形状: {df.shape[0]} 行 × {df.shape[1]} 列")

    # 缺失值统计
    missing = df.isnull().sum()
    if missing.sum() > 0:
        print(f"缺失值统计:\n{missing[missing > 0].to_string()}")
        # 前向填充
        for col in df.columns:
            if col == 'DATATIME':
                continue
            df[col] = df[col].ffill()
        print("短时缺失已使用前向填充法处理。")
    else:
        print("当前数据无缺失值，无需填充。")

    # 时间解析与排序
    df['DATATIME'] = pd.to_datetime(df['DATATIME'])
    df = df.sort_values('DATATIME').reset_index(drop=True)
    print(f"时间范围: {df['DATATIME'].min()} ~ {df['DATATIME'].max()}")
    print(f"采样频率: 平均间隔约 {pd.Series(df['DATATIME']).diff().dt.total_seconds().mean()/60:.1f} 分钟")

    # 物理异常值剔除（负值）
    print(f"\n物理异常值剔除前数据量: {len(df)}")
    neg_power = (df['WINDPOWER'] < 0).sum()
    neg_wind = (df['WINDSPEED'] < 0).sum()
    print(f"功率负值数量: {neg_power}, 风速负值数量: {neg_wind}")

    df_clean = df[df['WINDPOWER'] >= 0].reset_index(drop=True)
    df_clean = df_clean[df_clean['WINDSPEED'] >= 0].reset_index(drop=True)
    print(f"物理剔除后数据量: {len(df_clean)}")

    return df, df_clean


# ============================================================
# DBSCAN 聚类去噪
# ============================================================

def dbscan_denoise(df_clean, save_plot_path=None):
    """
    基于风速-功率二维空间进行 DBSCAN 聚类去噪。

    参数:
        df_clean: 已剔除物理异常值的 DataFrame
        save_plot_path: 可选，去噪对比图的保存路径

    返回: df_denoised — 剔除噪声点后的 DataFrame
    """
    print("\n" + "=" * 60)
    print("DBSCAN 聚类去噪")
    print("=" * 60)

    X = df_clean[['WINDSPEED', 'WINDPOWER']].values
    scaler_std = StandardScaler()
    X_scaled = scaler_std.fit_transform(X)

    # 采样估计 epsilon
    sample_size = min(DBSCAN_SAMPLE_SIZE, len(X_scaled))
    rng = np.random.RandomState(RANDOM_SEED)
    idx_sample = rng.choice(len(X_scaled), sample_size, replace=False)
    X_sample = X_scaled[idx_sample]

    neigh = NearestNeighbors(n_neighbors=DBSCAN_N_NEIGHBORS)
    neigh.fit(X_sample)
    distances, _ = neigh.kneighbors(X_sample)
    k_dist = np.sort(distances[:, -1])
    epsilon = np.percentile(k_dist, DBSCAN_EPS_PERCENTILE)

    print(f"DBSCAN 参数: epsilon={epsilon:.3f}, min_samples={DBSCAN_MIN_SAMPLES}")

    db = DBSCAN(eps=epsilon, min_samples=DBSCAN_MIN_SAMPLES)
    labels = db.fit_predict(X_scaled)

    n_noise = (labels == -1).sum()
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    print(f"检测到的聚类数: {n_clusters}")
    print(f"噪声点数量: {n_noise} (占比 {n_noise/len(df_clean)*100:.2f}%)")

    # 去噪前后对比图
    if save_plot_path:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        df_temp = df_clean.copy()
        df_temp['dbscan_label'] = labels
        noise_mask = df_temp['dbscan_label'] == -1

        axes[0].scatter(df_temp['WINDSPEED'], df_temp['WINDPOWER'],
                        s=1, alpha=0.5, c='steelblue', label='Normal')
        if noise_mask.any():
            axes[0].scatter(df_temp.loc[noise_mask, 'WINDSPEED'],
                            df_temp.loc[noise_mask, 'WINDPOWER'],
                            s=5, alpha=0.8, c='red', label=f'Outliers ({n_noise})')
        axes[0].set_xlabel('Wind Speed (m/s)')
        axes[0].set_ylabel('Active Power (kW)')
        axes[0].set_title('Before: With Outliers')
        axes[0].legend(markerscale=5)
        axes[0].grid(True, alpha=0.3)

        df_denoised = df_clean[labels != -1].copy().reset_index(drop=True)
        axes[1].scatter(df_denoised['WINDSPEED'], df_denoised['WINDPOWER'],
                        s=1, alpha=0.5, c='forestgreen')
        axes[1].set_xlabel('Wind Speed (m/s)')
        axes[1].set_ylabel('Active Power (kW)')
        axes[1].set_title('After: Outliers Removed')
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        ensure_dir(save_plot_path)
        plt.savefig(save_plot_path, dpi=200)
        plt.close()
        print(f"已保存去噪对比图: {save_plot_path}")

    df_denoised = df_clean[labels != -1].copy().reset_index(drop=True)
    print(f"DBSCAN 去噪后数据量: {len(df_denoised)}")
    return df_denoised


# ============================================================
# Min-Max 归一化
# ============================================================

def minmax_normalize(df, numeric_cols=None):
    """
    对数值列执行 Min-Max 归一化到 [0, 1]。

    返回: (df_normalized, scaler)
    """
    if numeric_cols is None:
        numeric_cols = NUMERIC_COLS

    print("\n" + "=" * 60)
    print("Min-Max 归一化")
    print("=" * 60)

    scaler = MinMaxScaler()
    df_normalized = df.copy()
    df_normalized[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    print("归一化参数 (各特征的 min, max):")
    for i, col in enumerate(numeric_cols):
        print(f"  {col}:  min={scaler.data_min_[i]:.4f}, max={scaler.data_max_[i]:.4f}")

    print(f"\n归一化后的数据范围:")
    print(df_normalized[numeric_cols].describe().loc[['min', 'max']].to_string())

    return df_normalized, scaler


# ============================================================
# 相关性分析与特征筛选
# ============================================================

def compute_correlation(df, numeric_cols=None, threshold=None):
    """
    计算皮尔逊相关系数，筛选与 WINDPOWER 显著相关的特征。

    返回: (correlation_matrix, power_corr_sorted, selected_features)
    """
    if numeric_cols is None:
        numeric_cols = NUMERIC_COLS
    if threshold is None:
        threshold = CORRELATION_THRESHOLD

    print("\n" + "=" * 60)
    print("相关性分析与特征筛选")
    print("=" * 60)

    corr_matrix = df[numeric_cols].corr(method='pearson')
    power_corr = corr_matrix['WINDPOWER'].drop('WINDPOWER').abs()
    power_corr_sorted = power_corr.sort_values(ascending=False)

    print("\n各特征与功率的相关系数绝对值（降序）:")
    for feature, corr in power_corr_sorted.items():
        print(f"  {feature}: {corr:.4f}")

    selected_features = power_corr[power_corr >= threshold].index.tolist()
    print(f"\n设定阈值 ≥ {threshold}")
    print(f"筛选后的关键特征: {selected_features}")
    print(f"筛选后的特征数: {len(selected_features)}")

    return corr_matrix, power_corr_sorted, selected_features


# ============================================================
# K-means 聚类分析
# ============================================================

def kmeans_cluster(X, k_range=None, sample_size=None, random_state=None,
                   save_k_plot_path=None):
    """
    使用肘部法 + 轮廓系数法（采样优化）确定最优 K 并执行 K-means 聚类。

    参数:
        X:           聚类数据 (numpy array)
        k_range:     K 值搜索范围，默认 range(2, 11)
        sample_size: 轮廓系数采样大小，默认 SILHOUETTE_SAMPLE_SIZE
        random_state: 随机种子
        save_k_plot_path: 可选，K 值选择图的保存路径

    返回: {
        'labels':          聚类标签,
        'optimal_k':       最优 K 值,
        'kmeans_model':    最终 KMeans 模型,
        'sse':             SSE 列表,
        'silhouette_scores': 轮廓系数列表,
        'k_range':         K 值范围,
    }
    """
    if k_range is None:
        k_range = range(2, 11)
    if sample_size is None:
        sample_size = SILHOUETTE_SAMPLE_SIZE
    if random_state is None:
        random_state = RANDOM_SEED

    print("\n" + "=" * 60)
    print("K-means 聚类分析")
    print("=" * 60)
    print(f"K 值搜索范围: {list(k_range)}")
    print(f"聚类数据形状: {X.shape}")

    sse = []
    silhouette_scores = []

    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        kmeans.fit(X)
        sse.append(kmeans.inertia_)

        # 采样计算轮廓系数（O(n²) → O(sample²)）
        n_samples = min(sample_size, len(X))
        rng = np.random.RandomState(random_state)
        idx_sample = rng.choice(len(X), n_samples, replace=False)
        sil = silhouette_score(X[idx_sample], kmeans.labels_[idx_sample])
        silhouette_scores.append(sil)
        print(f"  K={k}: SSE={sse[-1]:.2f}, 轮廓系数={sil:.4f}")

    # 找到最优 K
    optimal_k = k_range[np.argmax(silhouette_scores)]
    print(f"\n基于轮廓系数的最优 K 值: {optimal_k}")
    print(f"对应的轮廓系数: {max(silhouette_scores):.4f}")

    # 执行最终聚类
    kmeans_final = KMeans(n_clusters=optimal_k, random_state=random_state, n_init=10)
    labels = kmeans_final.fit_predict(X)

    # 绘制 K 值选择图
    if save_k_plot_path:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        ax1.plot(list(k_range), sse, 'bo-')
        ax1.set_xlabel('聚类数 K')
        ax1.set_ylabel('SSE (误差平方和)')
        ax1.set_title('肘部法确定最优K值')
        ax1.grid(True, alpha=0.3)

        ax2.plot(list(k_range), silhouette_scores, 'ro-')
        ax2.set_xlabel('聚类数 K')
        ax2.set_ylabel('轮廓系数')
        ax2.set_title('轮廓系数法确定最优K值')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        ensure_dir(save_k_plot_path)
        plt.savefig(save_k_plot_path, dpi=200, bbox_inches='tight')
        plt.close()
        print(f"已保存 K 值选择图: {save_k_plot_path}")

    print(f"\n聚类结果统计:")
    for i in range(optimal_k):
        count = sum(labels == i)
        percentage = count / len(labels) * 100
        print(f"  类别 {i}: {count} 个样本 ({percentage:.2f}%)")

    return {
        'labels': labels,
        'optimal_k': optimal_k,
        'kmeans_model': kmeans_final,
        'sse': sse,
        'silhouette_scores': silhouette_scores,
        'k_range': list(k_range),
    }


# ============================================================
# 聚类结果保存
# ============================================================

def save_data_with_clusters(df, labels, output_path):
    """将聚类标签添加到 DataFrame 并保存为 CSV。"""
    df_out = df.copy()
    df_out['cluster'] = labels
    ensure_dir(output_path)
    df_out.to_csv(output_path, index=False)
    print(f"已保存含聚类标签的数据集: {output_path} ({len(df_out)} 行 × {len(df_out.columns)} 列)")
    return df_out
