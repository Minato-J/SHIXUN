"""
================================================================================
风电场数据分析 — 公共模块（详细注释版）
================================================================================

【模块功能概述】
本模块是整个风电场数据分析项目的核心公共库，提取了所有脚本共享的
函数、常量和配置，避免代码重复，提高可维护性。

【包含内容】
  1. 中文字体配置      —— 保证 matplotlib 图表中文正常显示
  2. 全局常量定义      —— 统一管理所有魔术数字和超参数
  3. 工具函数          —— ensure_dir（创建目录）、特征列表存取
  4. 数据加载与预处理  —— load_and_preprocess：解析CSV、填充缺失、剔除异常
  5. DBSCAN 聚类去噪   —— dbscan_denoise：基于密度的噪声剔除
  6. Min-Max 归一化    —— minmax_normalize：将特征缩放到 [0,1]
  7. 相关性分析        —— compute_correlation：皮尔逊相关系数+特征筛选
  8. K-means 聚类      —— kmeans_cluster：肘部法+轮廓系数法确定最优K值
  9. 聚类结果保存      —— save_data_with_clusters：保存带标签的数据集

【被引用关系】
  - analysis.py       → 引用大部分函数（任务一+任务二）
  - Task3.py           → 引用 kmeans_cluster、load_selected_features 等
  - Task4.py           → 引用 kmeans_cluster、load_selected_features 等
  - wind_analysis.py   → 引用 compute_correlation、kmeans_cluster 等
  - wind_analysis_v2.py→ 引用大部分函数（面向对象版）
"""

# ===========================================================================
# 第一部分：导入依赖库
# ===========================================================================

import pandas as pd               # 数据处理：DataFrame 操作、CSV 读写
import numpy as np                # 数值计算：数组运算、随机数生成
import matplotlib.pyplot as plt   # 数据可视化：绘制图表
from sklearn.cluster import DBSCAN, KMeans          # 聚类算法：DBSCAN（去噪）、KMeans（工况划分）
from sklearn.preprocessing import MinMaxScaler, StandardScaler  # 数据缩放：归一化、标准化
from sklearn.neighbors import NearestNeighbors       # 最近邻搜索：用于 DBSCAN 的 epsilon 参数估计
from sklearn.metrics import silhouette_score, mean_absolute_error, mean_squared_error, r2_score
# ↑ 评估指标：轮廓系数（聚类质量）、MAE/RMSE/R²（回归性能）
from sklearn.model_selection import train_test_split # 数据划分：训练集/测试集分割
from sklearn.svm import SVR                          # 支持向量回归模型
from sklearn.neural_network import MLPRegressor      # BP 神经网络（多层感知机回归器）
from sklearn.linear_model import LinearRegression    # 线性回归模型
import os                         # 操作系统接口：路径处理和目录创建
import json                       # JSON 序列化：特征列表的保存和加载
import warnings                   # 警告控制：忽略非关键警告信息
from typing import Optional, List, Dict, Any, Tuple  # 类型提示：提高代码可读性

warnings.filterwarnings('ignore')  # 全局忽略警告信息，保持输出整洁


# ===========================================================================
# 第二部分：全局配置
# ===========================================================================

# --- 中文字体配置 ---
# matplotlib 默认不支持中文，需要指定支持中文的字体。
# 按优先级依次尝试 SimHei（黑体）、Microsoft YaHei（微软雅黑）、Arial Unicode MS
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号 '-' 显示为方块的问题

# --- 全局常量定义 ---
# 将所有魔术数字集中管理，方便统一调整和避免硬编码

# 数值型特征列名（六维特征：风速、风向、温度、湿度、气压、功率）
NUMERIC_COLS = ['WINDSPEED', 'WINDDIRECTION', 'TEMPERATURE', 'HUMIDITY', 'PRESSURE', 'WINDPOWER']

# 相关性阈值：特征与 WINDPOWER 的皮尔逊相关系数绝对值 >= 此值才被保留
CORRELATION_THRESHOLD = 0.2

# 轮廓系数采样大小：轮廓系数计算复杂度为 O(n²)，对大数据集采样以加速
SILHOUETTE_SAMPLE_SIZE = 5000

# DBSCAN 采样大小：用于 epsilon 参数估计的数据点数量
DBSCAN_SAMPLE_SIZE = 5000

# DBSCAN 最小样本数：一个聚类至少包含的点数
DBSCAN_MIN_SAMPLES = 10

# DBSCAN k-距离图的邻居数：用于估计 epsilon 参数
DBSCAN_N_NEIGHBORS = 10

# DBSCAN epsilon 百分位数：取 k-距离的第95百分位作为 epsilon
DBSCAN_EPS_PERCENTILE = 95

# 随机种子：确保结果可复现
RANDOM_SEED = 42

# 输出目录映射：将任务名映射到对应的输出文件夹
OUTPUT_DIRS = {
    'task1': 'RW1',        # 任务1输出：波形图、散点图等
    'task2': 'RW2',        # 任务2输出：散点矩阵、热力图、玫瑰图
    'task3': 'RW3',        # 任务3输出：K-means 聚类可视化
    'task4': 'RW4(SOLO)',  # 任务4输出：预测模型对比图
}


# ===========================================================================
# 第三部分：工具函数
# ===========================================================================

def ensure_dir(filepath):
    """
    确保输出文件的父目录存在。如果父目录不存在，则递归创建。

    【为什么需要这个函数】
    在保存图表文件时，如果目标路径的父目录（如 RW1/、RW2/）不存在，
    plt.savefig() 会报错。调用此函数可预先创建所需目录。

    【参数说明】
    filepath : str
        目标文件的完整路径（如 'RW1/scatter_plot.png'）

    【特殊情况】
    如果 filepath 是纯文件名（不含路径分隔符），则跳过创建（os.path.dirname 返回空字符串）。
    """
    dirname = os.path.dirname(filepath)  # 提取父目录路径
    if dirname:                          # 如果有父目录
        os.makedirs(dirname, exist_ok=True)  # 递归创建（exist_ok=True 避免重复创建报错）


def save_selected_features(features, filepath='selected_features.json'):
    """
    将筛选后的特征列表保存为 JSON 文件，供下游脚本（Task3、Task4）读取。
    这样可确保 Task2→Task3→Task4 使用完全一致的特征集合。

    【参数说明】
    features : list
        特征名称列表，如 ['WINDSPEED', 'HUMIDITY', 'PRESSURE']
    filepath : str
        输出 JSON 文件路径，默认为项目根目录下的 'selected_features.json'
    """
    with open(filepath, 'w', encoding='utf-8') as f:
        # ensure_ascii=False：保留中文字符，不转义为 Unicode
        # indent=2：格式化输出，便于人工阅读
        json.dump(list(features), f, ensure_ascii=False, indent=2)
    print(f"已保存特征列表: {filepath}")


def load_selected_features(filepath='selected_features.json'):
    """
    从 JSON 文件读取特征列表。这是 save_selected_features 的逆操作。

    【参数说明】
    filepath : str
        JSON 文件路径

    【返回值】
    list —— 特征名称列表

    【异常处理】
    如果文件不存在，json.load 会抛出 FileNotFoundError，由调用方捕获处理。
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


# ===========================================================================
# 第四部分：数据加载与预处理
# ===========================================================================

def load_and_preprocess(data_path='DATE.csv'):
    """
    加载原始 CSV 数据文件，执行完整的数据预处理流水线：
      1. 读取 CSV 文件
      2. 缺失值检测与前向填充
      3. 时间列解析与按时间排序
      4. 物理异常值剔除（如负风速、负功率）

    【处理逻辑详解】
    - 缺失值处理：使用 ffill()（前向填充），即用前一个有效值填充当前缺失值。
      这适用于时间序列数据，因为相邻时刻的值通常相近。
    - 物理异常剔除：风速和功率不可能为负数，将负值记录视为传感器故障/传输错误移除。

    【参数说明】
    data_path : str
        原始数据 CSV 文件路径，默认为 'DATE.csv'

    【返回值】
    (df_raw, df_clean) : tuple
        - df_raw  : 排序后的原始 DataFrame（含 DATATIME 时间列）
        - df_clean: 剔除负值后、缺失值已填充的 DataFrame
    """
    print("=" * 60)
    print("数据加载与预处理")
    print("=" * 60)

    # --- 步骤1：读取 CSV 文件 ---
    df = pd.read_csv(data_path)
    print(f"数据集形状: {df.shape[0]} 行 × {df.shape[1]} 列")

    # --- 步骤2：缺失值检测与填充 ---
    # 统计每列的缺失值数量
    missing = df.isnull().sum()
    if missing.sum() > 0:
        # 如果存在缺失值，打印缺失统计
        print(f"缺失值统计:\n{missing[missing > 0].to_string()}")
        # 遍历所有列，对数值列执行前向填充
        # （DATATIME 列跳过，因为它不是数值数据）
        for col in df.columns:
            if col == 'DATATIME':
                continue
            df[col] = df[col].ffill()  # ffill = forward fill，前向填充
        print("短时缺失已使用前向填充法处理。")
    else:
        print("当前数据无缺失值，无需填充。")

    # --- 步骤3：时间列解析与排序 ---
    # 将 DATATIME 字符串列转换为 pandas 的 datetime 类型
    df['DATATIME'] = pd.to_datetime(df['DATATIME'])
    # 按时间升序排列，确保数据的时间连续性
    df = df.sort_values('DATATIME').reset_index(drop=True)
    print(f"时间范围: {df['DATATIME'].min()} ~ {df['DATATIME'].max()}")
    # 计算平均采样间隔（分钟）
    avg_interval = pd.Series(df['DATATIME']).diff().dt.total_seconds().mean() / 60
    print(f"采样频率: 平均间隔约 {avg_interval:.1f} 分钟")

    # --- 步骤4：物理异常值剔除 ---
    # 风速和功率在物理上不可能为负数，将这些异常记录移除
    print(f"\n物理异常值剔除前数据量: {len(df)}")
    neg_power = (df['WINDPOWER'] < 0).sum()  # 功率为负的记录数
    neg_wind = (df['WINDSPEED'] < 0).sum()    # 风速为负的记录数
    print(f"功率负值数量: {neg_power}, 风速负值数量: {neg_wind}")

    # 先剔除功率为负的行
    df_clean = df[df['WINDPOWER'] >= 0].reset_index(drop=True)
    # 再剔除风速为负的行
    df_clean = df_clean[df_clean['WINDSPEED'] >= 0].reset_index(drop=True)
    print(f"物理剔除后数据量: {len(df_clean)}")

    return df, df_clean


# ===========================================================================
# 第五部分：DBSCAN 聚类去噪
# ===========================================================================

def dbscan_denoise(df_clean, save_plot_path=None):
    """
    基于风速-功率二维空间进行 DBSCAN（Density-Based Spatial Clustering
    of Applications with Noise）聚类去噪。

    【算法原理】
    DBSCAN 是一种基于密度的聚类算法，它将高密度区域标记为聚类，
    低密度区域的点标记为噪声。在风速-功率散点图中，正常数据点
    通常聚集在典型的功率曲线上，而异常点则散布在远离主流的位置。

    【epsilon 参数自动估计】
    使用 k-距离图方法：计算每个点到其第 k 近邻的距离，取第95百分位
    作为 epsilon。这样可自适应不同数据集的密度分布。

    【参数说明】
    df_clean : DataFrame
        已经过物理异常剔除的清洗数据集
    save_plot_path : str or None
        可选，去噪前后对比图的保存路径。若为 None 则不绘图。

    【返回值】
    df_denoised : DataFrame
        剔除 DBSCAN 标记的噪声点后的数据集
    """
    print("\n" + "=" * 60)
    print("DBSCAN 聚类去噪")
    print("=" * 60)

    # --- 步骤1：提取特征并标准化 ---
    # DBSCAN 对尺度敏感，需要先标准化（均值为0，标准差为1）
    X = df_clean[['WINDSPEED', 'WINDPOWER']].values  # 提取风速和功率两列
    scaler_std = StandardScaler()                      # 创建标准化器
    X_scaled = scaler_std.fit_transform(X)             # 拟合并转换数据

    # --- 步骤2：采样估计 epsilon 参数 ---
    # 为加速计算，从完整数据集中随机采样
    sample_size = min(DBSCAN_SAMPLE_SIZE, len(X_scaled))
    rng = np.random.RandomState(RANDOM_SEED)  # 固定随机种子，保证可复现
    idx_sample = rng.choice(len(X_scaled), sample_size, replace=False)  # 不重复采样
    X_sample = X_scaled[idx_sample]  # 提取采样数据

    # 使用 NearestNeighbors 计算 k-距离
    neigh = NearestNeighbors(n_neighbors=DBSCAN_N_NEIGHBORS)
    neigh.fit(X_sample)
    distances, _ = neigh.kneighbors(X_sample)  # distances[i, j] = 第i个点到其第j近邻的距离
    k_dist = np.sort(distances[:, -1])          # 取第k近邻距离（最后一列）并排序
    epsilon = np.percentile(k_dist, DBSCAN_EPS_PERCENTILE)  # 取第95百分位作为 epsilon

    print(f"DBSCAN 参数: epsilon={epsilon:.3f}, min_samples={DBSCAN_MIN_SAMPLES}")

    # --- 步骤3：执行 DBSCAN 聚类 ---
    db = DBSCAN(eps=epsilon, min_samples=DBSCAN_MIN_SAMPLES)
    labels = db.fit_predict(X_scaled)  # labels=-1 表示噪声点

    # --- 步骤4：统计聚类结果 ---
    n_noise = (labels == -1).sum()                               # 噪声点数量
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)   # 有效聚类数
    print(f"检测到的聚类数: {n_clusters}")
    print(f"噪声点数量: {n_noise} (占比 {n_noise/len(df_clean)*100:.2f}%)")

    # --- 步骤5：绘制去噪前后对比图 ---
    if save_plot_path:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))  # 1行2列子图

        # 创建临时 DataFrame 便于按标签筛选
        df_temp = df_clean.copy()
        df_temp['dbscan_label'] = labels       # 添加聚类标签列
        noise_mask = df_temp['dbscan_label'] == -1  # 噪声点布尔掩码

        # 左图：去噪前（红色标记噪声点）
        axes[0].scatter(df_temp['WINDSPEED'], df_temp['WINDPOWER'],
                        s=1, alpha=0.5, c='steelblue', label='Normal')
        if noise_mask.any():  # 如果有噪声点，用红色标记
            axes[0].scatter(df_temp.loc[noise_mask, 'WINDSPEED'],
                            df_temp.loc[noise_mask, 'WINDPOWER'],
                            s=5, alpha=0.8, c='red', label=f'Outliers ({n_noise})')
        axes[0].set_xlabel('Wind Speed (m/s)')
        axes[0].set_ylabel('Active Power (kW)')
        axes[0].set_title('Before: With Outliers')
        axes[0].legend(markerscale=5)
        axes[0].grid(True, alpha=0.3)

        # 右图：去噪后（仅保留正常点，绿色显示）
        df_denoised_temp = df_clean[labels != -1].copy().reset_index(drop=True)
        axes[1].scatter(df_denoised_temp['WINDSPEED'], df_denoised_temp['WINDPOWER'],
                        s=1, alpha=0.5, c='forestgreen')
        axes[1].set_xlabel('Wind Speed (m/s)')
        axes[1].set_ylabel('Active Power (kW)')
        axes[1].set_title('After: Outliers Removed')
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        ensure_dir(save_plot_path)  # 确保输出目录存在
        plt.savefig(save_plot_path, dpi=200)  # dpi=200 保证图表清晰度
        plt.close()
        print(f"已保存去噪对比图: {save_plot_path}")

    # --- 步骤6：返回去噪后的数据 ---
    df_denoised = df_clean[labels != -1].copy().reset_index(drop=True)
    print(f"DBSCAN 去噪后数据量: {len(df_denoised)}")
    return df_denoised


# ===========================================================================
# 第六部分：Min-Max 归一化
# ===========================================================================

def minmax_normalize(df, numeric_cols=None):
    """
    对数值列执行 Min-Max 归一化，将所有特征缩放到 [0, 1] 区间。

    【为什么需要归一化】
    不同特征的量纲差异很大（如风速 0~25 m/s，气压 ~1000 hPa），
    直接使用会导致某些特征在聚类/建模中主导距离计算。
    Min-Max 归一化将所有特征映射到统一的 [0,1] 范围。

    【归一化公式】
    X_norm = (X - X_min) / (X_max - X_min)

    【参数说明】
    df : DataFrame
        待归一化的数据集
    numeric_cols : list or None
        需要归一化的数值列名列表。若为 None，使用全局常量 NUMERIC_COLS

    【返回值】
    (df_normalized, scaler) : tuple
        - df_normalized : 归一化后的 DataFrame
        - scaler        : MinMaxScaler 对象（保存了 min/max 参数，可用于逆变换）
    """
    if numeric_cols is None:
        numeric_cols = NUMERIC_COLS  # 默认对全部六维特征进行归一化

    print("\n" + "=" * 60)
    print("Min-Max 归一化")
    print("=" * 60)

    # 创建 MinMaxScaler 并执行归一化
    scaler = MinMaxScaler()
    df_normalized = df.copy()  # 复制一份，避免修改原始数据
    # 仅对指定列进行归一化，其他列（如 DATATIME、cluster）保持不变
    df_normalized[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    # 打印归一化参数（每列的最小值和最大值）
    print("归一化参数 (各特征的 min, max):")
    for i, col in enumerate(numeric_cols):
        # scaler.data_min_ 和 scaler.data_max_ 存储了每列的原始最小/最大值
        print(f"  {col}:  min={scaler.data_min_[i]:.4f}, max={scaler.data_max_[i]:.4f}")

    # 验证归一化结果：所有值应在 [0, 1] 范围内
    print(f"\n归一化后的数据范围:")
    print(df_normalized[numeric_cols].describe().loc[['min', 'max']].to_string())

    return df_normalized, scaler


# ===========================================================================
# 第七部分：相关性分析与特征筛选
# ===========================================================================

def compute_correlation(df, numeric_cols=None, threshold=None):
    """
    计算皮尔逊相关系数矩阵，筛选与 WINDPOWER（目标变量）显著相关的特征。

    【皮尔逊相关系数】
    取值范围 [-1, 1]，绝对值越大表示线性相关性越强。
    - 正值：正相关（一个增大，另一个也增大）
    - 负值：负相关（一个增大，另一个减小）
    - 接近0：无明显线性关系

    【特征筛选策略】
    计算所有特征与 WINDPOWER 的相关系数绝对值，保留 >= threshold 的特征。
    这可以减少后续聚类的维度，去除冗余信息。

    【参数说明】
    df : DataFrame
        输入数据集（通常为去噪后的数据）
    numeric_cols : list or None
        参与计算的特征列名。默认使用 NUMERIC_COLS
    threshold : float or None
        相关系数阈值。默认使用 CORRELATION_THRESHOLD (0.2)

    【返回值】
    (correlation_matrix, power_corr_sorted, selected_features) : tuple
        - correlation_matrix  : 完整的相关系数矩阵 (DataFrame)
        - power_corr_sorted   : 各特征与 WINDPOWER 的相关系数（降序排列）
        - selected_features   : 筛选后保留的特征名列表
    """
    if numeric_cols is None:
        numeric_cols = NUMERIC_COLS
    if threshold is None:
        threshold = CORRELATION_THRESHOLD

    print("\n" + "=" * 60)
    print("相关性分析与特征筛选")
    print("=" * 60)

    # --- 步骤1：计算皮尔逊相关系数矩阵 ---
    # method='pearson' 表示使用皮尔逊相关系数
    corr_matrix = df[numeric_cols].corr(method='pearson')

    # --- 步骤2：提取各特征与 WINDPOWER 的相关系数 ---
    # 取出 'WINDPOWER' 列，再移除 WINDPOWER 自身（自身相关系数为1）
    power_corr = corr_matrix['WINDPOWER'].drop('WINDPOWER').abs()  # 取绝对值
    power_corr_sorted = power_corr.sort_values(ascending=False)    # 降序排列

    # 打印排序结果
    print("\n各特征与功率的相关系数绝对值（降序）:")
    for feature, corr in power_corr_sorted.items():
        print(f"  {feature}: {corr:.4f}")

    # --- 步骤3：基于阈值筛选特征 ---
    # 保留相关系数绝对值 >= threshold 的特征
    selected_features = power_corr[power_corr >= threshold].index.tolist()
    print(f"\n设定阈值 ≥ {threshold}")
    print(f"筛选后的关键特征: {selected_features}")
    print(f"筛选后的特征数: {len(selected_features)}")

    return corr_matrix, power_corr_sorted, selected_features


# ===========================================================================
# 第八部分：K-means 聚类分析
# ===========================================================================

def kmeans_cluster(X, k_range=None, sample_size=None, random_state=None,
                   save_k_plot_path=None):
    """
    使用肘部法（Elbow Method）和轮廓系数法（Silhouette Score）相结合，
    确定最优 K 值并执行 K-means 聚类。

    【K-means 算法原理】
    K-means 是一种基于距离的划分聚类算法：
      1. 随机初始化 K 个聚类中心
      2. 将每个样本分配到最近的中心
      3. 重新计算每个聚类的中心（均值）
      4. 重复步骤2-3直到收敛

    【肘部法（Elbow Method）】
    计算不同 K 值下的 SSE（误差平方和，即各点到其聚类中心的距离平方和）。
    SSE 随 K 增大而减小，在最优 K 处下降速度会明显减缓，形成"肘部"。

    【轮廓系数法（Silhouette Score）】
    取值范围 [-1, 1]，衡量聚类结果的紧密度和分离度：
    - 接近 1：样本与自身聚类匹配良好，与其他聚类分离良好
    - 接近 0：样本处于两个聚类的边界
    - 接近 -1：样本可能被分配到错误的聚类

    【采样优化】
    轮廓系数计算复杂度为 O(n²)，对大数据集进行随机采样以加速计算。

    【参数说明】
    X : numpy.ndarray
        聚类输入数据，形状为 (n_samples, n_features)
    k_range : range or None
        K 值搜索范围，默认 range(2, 11)（即尝试 K=2 到 K=10）
    sample_size : int or None
        轮廓系数计算的采样大小，默认 SILHOUETTE_SAMPLE_SIZE (5000)
    random_state : int or None
        随机种子，默认 RANDOM_SEED (42)
    save_k_plot_path : str or None
        可选，K 值选择图（肘部法+轮廓系数法）的保存路径

    【返回值】
    dict —— 包含以下键：
        - 'labels'           : 每个样本的聚类标签 (numpy array)
        - 'optimal_k'        : 最优 K 值（使得轮廓系数最大）
        - 'kmeans_model'     : 训练好的 KMeans 模型对象
        - 'sse'              : 各 K 值对应的 SSE 列表
        - 'silhouette_scores': 各 K 值对应的轮廓系数列表
        - 'k_range'          : K 值搜索范围列表
    """
    # --- 参数默认值处理 ---
    if k_range is None:
        k_range = range(2, 11)  # 默认搜索 K=2,3,...,10
    if sample_size is None:
        sample_size = SILHOUETTE_SAMPLE_SIZE  # 默认采样5000个点
    if random_state is None:
        random_state = RANDOM_SEED  # 固定随机种子

    print("\n" + "=" * 60)
    print("K-means 聚类分析")
    print("=" * 60)
    print(f"K 值搜索范围: {list(k_range)}")
    print(f"聚类数据形状: {X.shape}")

    # --- 步骤1：遍历候选 K 值，计算 SSE 和轮廓系数 ---
    sse = []               # 存储各 K 值的 SSE（肘部法指标）
    silhouette_scores = []  # 存储各 K 值的轮廓系数

    for k in k_range:
        # 创建 KMeans 对象并训练
        # n_init=10：使用10次不同的随机初始化，选择最优结果
        kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        kmeans.fit(X)

        # 记录 SSE（kmeans.inertia_ 即聚类内误差平方和）
        sse.append(kmeans.inertia_)

        # --- 采样计算轮廓系数（降低 O(n²) 到 O(sample²)） ---
        n_samples = min(sample_size, len(X))  # 不能超过实际数据量
        rng = np.random.RandomState(random_state)
        idx_sample = rng.choice(len(X), n_samples, replace=False)  # 随机不重复采样
        # 仅对采样数据计算轮廓系数
        sil = silhouette_score(X[idx_sample], kmeans.labels_[idx_sample])
        silhouette_scores.append(sil)

        print(f"  K={k}: SSE={sse[-1]:.2f}, 轮廓系数={sil:.4f}")

    # --- 步骤2：确定最优 K 值 ---
    # 选择轮廓系数最大的 K 值（轮廓系数越大，聚类质量越好）
    optimal_k = k_range[np.argmax(silhouette_scores)]
    print(f"\n基于轮廓系数的最优 K 值: {optimal_k}")
    print(f"对应的轮廓系数: {max(silhouette_scores):.4f}")

    # --- 步骤3：使用最优 K 值执行最终聚类 ---
    kmeans_final = KMeans(n_clusters=optimal_k, random_state=random_state, n_init=10)
    labels = kmeans_final.fit_predict(X)  # fit_predict 返回每个样本的聚类标签

    # --- 步骤4：绘制 K 值选择图（肘部法 + 轮廓系数法并排显示） ---
    if save_k_plot_path:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        # 左图：肘部法 — SSE 随 K 值变化的曲线
        ax1.plot(list(k_range), sse, 'bo-')  # 蓝色圆点连线
        ax1.set_xlabel('聚类数 K')
        ax1.set_ylabel('SSE (误差平方和)')
        ax1.set_title('肘部法确定最优K值')
        ax1.grid(True, alpha=0.3)

        # 右图：轮廓系数法 — 轮廓系数随 K 值变化的曲线
        ax2.plot(list(k_range), silhouette_scores, 'ro-')  # 红色圆点连线
        ax2.set_xlabel('聚类数 K')
        ax2.set_ylabel('轮廓系数')
        ax2.set_title('轮廓系数法确定最优K值')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        ensure_dir(save_k_plot_path)
        plt.savefig(save_k_plot_path, dpi=200, bbox_inches='tight')
        plt.close()
        print(f"已保存 K 值选择图: {save_k_plot_path}")

    # --- 步骤5：打印各聚类的样本分布 ---
    print(f"\n聚类结果统计:")
    for i in range(optimal_k):
        count = sum(labels == i)                # 该聚类样本数
        percentage = count / len(labels) * 100  # 该聚类占比
        print(f"  类别 {i}: {count} 个样本 ({percentage:.2f}%)")

    # 返回结构化结果字典
    return {
        'labels': labels,
        'optimal_k': optimal_k,
        'kmeans_model': kmeans_final,
        'sse': sse,
        'silhouette_scores': silhouette_scores,
        'k_range': list(k_range),
    }


# ===========================================================================
# 第九部分：聚类结果保存
# ===========================================================================

def save_data_with_clusters(df, labels, output_path):
    """
    将聚类标签作为新列添加到 DataFrame 中，并保存为 CSV 文件。

    【用途】
    将 Task3 的聚类结果持久化，供 Task4 读取使用。
    这样可以避免 Task4 重复执行聚类，确保两次分析使用相同的聚类结果。

    【参数说明】
    df : DataFrame
        原始数据（通常为归一化后的数据）
    labels : array-like
        聚类标签数组，长度必须与 df 的行数一致
    output_path : str
        输出 CSV 文件路径

    【返回值】
    df_out : DataFrame
        包含 'cluster' 新列的数据框
    """
    df_out = df.copy()           # 复制数据框，不修改原始数据
    df_out['cluster'] = labels   # 添加聚类标签列
    ensure_dir(output_path)      # 确保输出目录存在
    df_out.to_csv(output_path, index=False)  # index=False：不保存行号
    print(f"已保存含聚类标签的数据集: {output_path} ({len(df_out)} 行 × {len(df_out.columns)} 列)")
    return df_out
