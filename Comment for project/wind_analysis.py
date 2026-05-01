"""
================================================================================
风电场数据分析完整项目（详细注释版）
================================================================================

【脚本功能概述】
本脚本是四任务整合脚本（函数式风格），单次运行完成从预处理数据到风电功率
预测的全部分析流程。与独立脚本（analysis.py、Task3.py、Task4.py）不同，
本文件将各任务内联编写，无需分步执行。

【任务覆盖】
  - 任务二：可视化与相关性分析（统计特征表、散点矩阵、风向玫瑰图、热力图）
  - 任务三：K-means 聚类分析（肘部法+轮廓系数选K值、PCA降维、聚类散点图）
  - 任务四：风电功率预测（SVR / BP神经网络 / 线性回归 三模型对比）

【依赖说明】
  - 需要已生成 data_normalized.csv（由 analysis.py 的任务一输出）
  - 若首次运行，需先执行 analysis.py 完成数据预处理
  - 读取公共模块 common.py 提供的工具函数

【输出文件说明】
  - RW2/scatter_matrix.png        : 六维特征散点图矩阵（含直方图对角线）
  - RW2/wind_rose.png             : 风向频率玫瑰图（24分区极坐标柱状图）
  - RW2/correlation_heatmap.png   : 皮尔逊相关系数热力图（6×6矩阵）
  - RW3/kmeans_k_selection.png    : K值选择图（肘部法SSE + 轮廓系数曲线）
  - RW3/kmeans_clusters.png       : PCA二维主成分聚类可视化
  - RW3/speed_power_clusters.png  : 风速-功率按聚类着色散点图
  - RW4(SOLO)/model_comparison.png: 三模型 MAE/RMSE/R² 柱状对比图
  - RW4(SOLO)/prediction_*.png    : 各聚类各模型预测值 vs 真实值曲线

【与 wind_analysis_v2.py 的区别】
  - 本文件：函数式风格，固定输出目录（RW2/RW3/RW4(SOLO)）
  - v2 版本：面向对象风格 + 时间戳输出目录（task_outputs/TaskName/{timestamp}）
"""

# ===========================================================================
# 第一部分：导入依赖库
# ===========================================================================

import pandas as pd               # 数据处理：DataFrame、CSV 读写、散点矩阵绘制
import numpy as np                # 数值计算：数组运算、三角函数（玫瑰图角度计算）
import matplotlib.pyplot as plt   # 数据可视化：各类图表的绘制引擎
import seaborn as sns             # 统计可视化：热力图（比 matplotlib 更美观）
from sklearn.model_selection import train_test_split  # 数据集划分：训练集/测试集
from sklearn.svm import SVR                # 支持向量回归（RBF核）
from sklearn.neural_network import MLPRegressor  # BP 神经网络（多层感知机）
from sklearn.linear_model import LinearRegression  # 线性回归（最小二乘法）
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# 评估指标导入：
#   mean_absolute_error : MAE，预测值与真实值绝对差均值（越小越好）
#   mean_squared_error  : MSE，预测误差平方的均值（取 sqrt 得 RMSE）
#   r2_score            : R²，模型对数据方差解释程度（越接近1越好）
from sklearn.decomposition import PCA  # 主成分分析：高维数据降维到2D用于可视化
import warnings                     # 警告控制
warnings.filterwarnings('ignore')   # 忽略收敛/弃用警告，保持控制台整洁

# --- 从公共模块导入共享函数和常量 ---
# ensure_dir           : 递归创建输出目录，确保文件保存路径存在
# compute_correlation  : 计算皮尔逊相关系数矩阵 + 按阈值筛选特征
# kmeans_cluster       : K-means 聚类（含肘部法SSE + 轮廓系数自动选K）
# NUMERIC_COLS         : 六维数值特征列名常量列表
# CORRELATION_THRESHOLD: 皮尔逊相关性筛选阈值常量（默认 0.2）
from common import (
    ensure_dir, compute_correlation, kmeans_cluster,
    NUMERIC_COLS, CORRELATION_THRESHOLD,
)


# ===========================================================================
# 第二部分：加载预处理数据
# ===========================================================================
# 注意：本脚本假设 data_normalized.csv 已由 analysis.py（任务一）生成
# 该文件包含经缺失填充→异常剔除→DBSCAN去噪→Min-Max归一化后的数据

print("=" * 60)
print("加载预处理后的数据")
print("=" * 60)

df = pd.read_csv('data_normalized.csv')
print(f"数据集形状: {df.shape[0]} 行 × {df.shape[1]} 列")
print(f"列名: {list(df.columns)}")

# DATATIME 列读取时默认为字符串，需要显式转换为 datetime 类型
# 这对于后续按时间排序或时间序列可视化（如有需要）非常重要
df['DATATIME'] = pd.to_datetime(df['DATATIME'])


# ===========================================================================
# 第三部分：任务二 — 可视化与相关性分析
# ===========================================================================

print("\n" + "=" * 60)
print("任务2：可视化与相关性分析")
print("=" * 60)

# ============================================================
# 步骤2.1：统计特征表
# ============================================================
# 对六个数值特征计算描述性统计量：
#   mean（均值）、std（标准差）、min（最小值）、25%/50%/75%（四分位数）、
#   max（最大值）、range（极差 = max - min）
print("\n1. 统计特征表:")
print("-" * 40)
numeric_cols = ['WINDSPEED', 'WINDDIRECTION', 'TEMPERATURE', 'HUMIDITY', 'PRESSURE', 'WINDPOWER']
stats = df[numeric_cols].describe().T   # .T 转置：让特征名成为行索引
stats['range'] = stats['max'] - stats['min']  # 补充极差列
stats_table = stats[['mean', 'std', 'min', '25%', '50%', '75%', 'max', 'range']]
print(stats_table.round(4))  # 保留4位小数便于阅读

# ============================================================
# 步骤2.2：六维特征散点图矩阵
# ============================================================
# 原理：生成 6×6 矩阵图，对角线为各特征的直方图（分布可视化），
#       非对角线为两两特征的散点图（相关性可视化）
print("\n2. 绘制散点图矩阵...")
fig = plt.figure(figsize=(15, 15))
axes = pd.plotting.scatter_matrix(
    df[numeric_cols],
    figsize=(15, 15),
    diagonal='hist',   # 对角线绘制直方图（histogram）
    alpha=0.6,         # 散点透明度，便于观察密度
    s=10               # 散点大小（点数多时用小点避免重叠）
)
plt.suptitle('六维特征散点图矩阵', fontsize=16)
plt.tight_layout()
ensure_dir('RW2/scatter_matrix.png')
plt.savefig('RW2/scatter_matrix.png', dpi=200, bbox_inches='tight')
plt.close()
print("已保存: RW2/scatter_matrix.png")

# ============================================================
# 步骤2.3：风向玫瑰图
# ============================================================
# 原理：利用极坐标柱状图展示风向频率分布
#       将归一化的风向值（0~1）映射到 0°~360°，按 15° 区间（24 bins）统计频率

print("\n3. 绘制风向玫瑰图...")


def wind_rose_plot(df, bins=24):
    """
    绘制风向频率玫瑰图（极坐标柱状图）

    参数:
        df   : 包含 WINDDIRECTION 列的 DataFrame（归一化值，0~1）
        bins : 方向区间数（默认 24，对应每 15° 一个区间）

    返回:
        fig, ax : matplotlib 图形和极坐标轴对象
    """
    # 将归一化风向值（0~1）映射到角度（0°~360°）
    wind_dir_deg = df['WINDDIRECTION'] * 360
    wind_dir_rad = np.radians(wind_dir_deg)  # 转换为弧度制

    # 划分角度区间（0 到 2π 均分 bins 份）
    angle_bins = np.linspace(0, 2 * np.pi, bins + 1)
    hist, _ = np.histogram(wind_dir_rad, bins=angle_bins)  # 统计各区间频数
    centers = (angle_bins[:-1] + angle_bins[1:]) / 2       # 各区间的中心角度

    # 创建极坐标子图
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    width = 2 * np.pi / bins               # 每个柱子的宽度
    bars = ax.bar(centers, hist, width=width, alpha=0.7, edgecolor='black')

    # 设置极坐标参数
    ax.set_theta_zero_location('N')        # 0° 指向正北方向
    ax.set_theta_direction(-1)             # 顺时针方向为正（模仿气象学惯例）
    ax.set_title('风向频率玫瑰图', pad=20, fontsize=16)
    ax.set_rlabel_position(0)              # 径向标签位置

    plt.tight_layout()
    return fig, ax


fig, ax = wind_rose_plot(df, bins=24)  # 24 bins = 每 15° 一个区间
ensure_dir('RW2/wind_rose.png')
plt.savefig('RW2/wind_rose.png', dpi=200, bbox_inches='tight')
plt.close()
print("已保存: RW2/wind_rose.png")

# ============================================================
# 步骤2.4：皮尔逊相关性分析
# ============================================================
print("\n4. 相关性分析...")

# 计算 6×6 皮尔逊相关系数矩阵
# 公式：r = Σ[(xᵢ - x̄)(yᵢ - ȳ)] / √[Σ(xᵢ - x̄)² · Σ(yᵢ - ȳ)²]
# r ∈ [-1, 1]：接近1正相关，接近-1负相关，接近0不相关
correlation_matrix = df[numeric_cols].corr()

# --- 绘制皮尔逊相关系数热力图 ---
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix,
            annot=True,           # 在每个格子内标注数值
            cmap='coolwarm',      # 冷色→暖色渐变（蓝=负相关，红=正相关）
            center=0,             # 色彩映射中心为0
            square=True,          # 正方形格子
            fmt='.3f',            # 数值保留3位小数
            cbar_kws={'shrink': 0.8})  # 颜色条缩放
plt.title('特征相关性热力图')
plt.tight_layout()
ensure_dir('RW2/correlation_heatmap.png')
plt.savefig('RW2/correlation_heatmap.png', dpi=200, bbox_inches='tight')
plt.close()
print("已保存: RW2/correlation_heatmap.png")

# --- 提取与风电功率(WINDPOWER)的相关系数 ---
# 取绝对值并按降序排列，找出与功率关系最密切的特征
power_corr = correlation_matrix['WINDPOWER'].drop('WINDPOWER').abs().sort_values(ascending=False)
print("\n各特征与功率的相关系数（按绝对值降序排列）:")
for feature, corr in power_corr.items():
    print(f"  {feature}: {corr:.4f}")

# --- 特征筛选 ---
# 保留与功率相关系数绝对值 >= 阈值的特征，用于后续聚类和预测建模
threshold = 0.2   # 经验阈值：|r| ≥ 0.2 表示至少存在弱相关
selected_features = power_corr[power_corr >= threshold].index.tolist()
print(f"\n相关系数绝对值 >= {threshold} 的特征（用于后续建模）:")
for feature in selected_features:
    print(f"  {feature}: {power_corr[feature]:.4f}")


# ===========================================================================
# 第四部分：任务三 — K-means 聚类分析
# ===========================================================================
# 详细原理参见 Comment for project/Task3.py

print("\n" + "=" * 60)
print("任务3：K-means聚类分析")
print("=" * 60)

# ============================================================
# 步骤3.1：准备聚类特征
# ============================================================
# 聚类使用的特征 = 筛选特征 + WINDPOWER（目标变量也参与聚类，反映运行工况）
# dict.fromkeys() 去重技巧：保持首次出现的顺序，同时去除可能的重复
cluster_features = list(dict.fromkeys(selected_features + ['WINDPOWER']))
X_cluster = df[cluster_features].values
print(f"聚类使用的特征: {cluster_features}")
print(f"聚类数据形状: {X_cluster.shape}")

# ============================================================
# 步骤3.2：执行 K-means 聚类（调用公共模块）
# ============================================================
# kmeans_cluster 内部流程：
#   1) 对 K=2~8 分别聚类，计算 SSE（肘部法）
#   2) 采样优化后计算轮廓系数（Silhouette Score）
#   3) 综合 SSE 拐点和轮廓系数最大值确定最优 K
#   4) 保存 K 值选择图为 RW3/kmeans_k_selection.png
cluster_result = kmeans_cluster(
    X_cluster,
    save_k_plot_path='RW3/kmeans_k_selection.png'
)

# 提取聚类结果
cluster_labels = cluster_result['labels']      # 每个样本的聚类标签（0/1/...）
optimal_k_silhouette = cluster_result['optimal_k']  # 最优聚类数 K

# 将聚类标签添加到 DataFrame（新增 cluster 列）
df['cluster'] = cluster_labels

# ============================================================
# 步骤3.3：PCA 降维可视化
# ============================================================
# 原理：主成分分析（PCA）将多维特征投影到方差最大的两个方向上（主成分），
#       以便在二维平面上观察聚类效果
print("4. 聚类结果可视化...")

pca = PCA(n_components=2)   # 降至2维
X_pca = pca.fit_transform(X_cluster)  # 拟合 + 转换

plt.figure(figsize=(12, 8))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1],
                      c=cluster_labels,          # 按聚类标签着色
                      cmap='viridis',            # 使用 viridis 颜色映射
                      alpha=0.6, s=20)           # 半透明，点大小20
plt.xlabel(f'第一主成分 (解释方差比: {pca.explained_variance_ratio_[0]:.3f})')
plt.ylabel(f'第二主成分 (解释方差比: {pca.explained_variance_ratio_[1]:.3f})')
plt.title(f'K-means聚类结果可视化 (K={optimal_k_silhouette})')
plt.colorbar(scatter)  # 颜色条显示标签映射
plt.grid(True, alpha=0.3)
ensure_dir('RW3/kmeans_clusters.png')
plt.savefig('RW3/kmeans_clusters.png', dpi=200, bbox_inches='tight')
plt.close()
print("已保存: RW3/kmeans_clusters.png")

# ============================================================
# 步骤3.4：风速-功率聚类散点图
# ============================================================
# 在原始风速-功率空间上展示聚类结果，直观判断聚类是否合理
plt.figure(figsize=(10, 6))
scatter = plt.scatter(df['WINDSPEED'], df['WINDPOWER'],
                      c=cluster_labels, cmap='viridis', alpha=0.6, s=10)
plt.xlabel('风速 (归一化)')
plt.ylabel('功率 (归一化)')
plt.title(f'风速-功率散点图 (按聚类着色, K={optimal_k_silhouette})')
plt.colorbar(scatter)
plt.grid(True, alpha=0.3)
ensure_dir('RW3/speed_power_clusters.png')
plt.savefig('RW3/speed_power_clusters.png', dpi=200, bbox_inches='tight')
plt.close()
print("已保存: RW3/speed_power_clusters.png")

# --- 输出各类别的样本统计 ---
print(f"\n聚类结果统计:")
for i in range(optimal_k_silhouette):
    count = sum(cluster_labels == i)
    percentage = count / len(cluster_labels) * 100
    print(f"  类别 {i}: {count} 个样本 ({percentage:.2f}%)")


# ===========================================================================
# 第五部分：任务四 — 风电功率预测
# ===========================================================================
# 详细原理参见 Comment for project/Task4.py

print("\n" + "=" * 60)
print("任务4：风电功率预测模型")
print("=" * 60)

# ============================================================
# 步骤4.1：按聚类分割数据
# ============================================================
print(f"使用聚类结果分割数据...")
X = df[selected_features].values   # 特征矩阵（仅筛选后的特征，不含 WINDPOWER）
y = df['WINDPOWER'].values         # 目标向量（风电功率）
# cluster_labels 已在任务三中获取

# 按聚类标签将数据划分为独立的子集
clusters_data = {}
for cluster_id in range(optimal_k_silhouette):
    mask = cluster_labels == cluster_id       # 布尔掩码：筛选属于当前聚类的样本
    clusters_data[cluster_id] = {
        'X': X[mask],                         # 当前聚类的特征子集
        'y': y[mask],                         # 当前聚类的目标子集
        'indices': np.where(mask)[0]          # 在原始数据中的索引位置
    }
    print(f"  类别 {cluster_id}: {len(clusters_data[cluster_id]['X'])} 个样本")

# ============================================================
# 步骤4.2：对每个聚类分别训练三种预测模型
# ============================================================
# 模型说明（详见 Task4.py 注释）：
#   SVR    — 支持向量回归（RBF核，C=1.0, epsilon=0.1）
#   BP     — BP神经网络（2×32隐藏层，ReLU激活，Adam优化器）
#   Linear — 线性回归（最小二乘法，性能基准）

# 初始化结果存储
results = {}

# 遍历每个聚类进行建模
for cluster_id in range(optimal_k_silhouette):
    print(f"\n--- 处理聚类 {cluster_id} ---")
    X_cluster = clusters_data[cluster_id]['X']
    y_cluster = clusters_data[cluster_id]['y']

    # 安全检查：样本数过少无法划分训练/测试集
    if len(X_cluster) < 2:
        print(f"  聚类 {cluster_id} 样本数太少，跳过建模")
        continue

    # 8:2 比例随机划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X_cluster, y_cluster, test_size=0.2, random_state=42
    )

    print(f"  训练集大小: {X_train.shape[0]}, 测试集大小: {X_test.shape[0]}")

    results[cluster_id] = {}

    # --- 模型1：支持向量回归 (SVR) ---
    print(f"  1. 训练SVR模型...")
    svr_model = SVR(kernel='rbf', C=1.0, epsilon=0.1)
    svr_model.fit(X_train, y_train)
    y_pred_svr = svr_model.predict(X_test)

    svr_mae = mean_absolute_error(y_test, y_pred_svr)
    svr_rmse = np.sqrt(mean_squared_error(y_test, y_pred_svr))
    svr_r2 = r2_score(y_test, y_pred_svr)

    results[cluster_id]['SVR'] = {
        'model': svr_model, 'predictions': y_pred_svr, 'actual': y_test,
        'mae': svr_mae, 'rmse': svr_rmse, 'r2': svr_r2
    }
    print(f"    MAE: {svr_mae:.4f}, RMSE: {svr_rmse:.4f}, R2: {svr_r2:.4f}")

    # --- 模型2：BP 神经网络 ---
    print(f"  2. 训练BP神经网络模型...")
    bp_model = MLPRegressor(
        hidden_layer_sizes=(32, 32), activation='relu', solver='adam',
        learning_rate_init=0.001, max_iter=500, random_state=42
    )
    bp_model.fit(X_train, y_train)
    y_pred_bp = bp_model.predict(X_test)

    bp_mae = mean_absolute_error(y_test, y_pred_bp)
    bp_rmse = np.sqrt(mean_squared_error(y_test, y_pred_bp))
    bp_r2 = r2_score(y_test, y_pred_bp)

    results[cluster_id]['BP'] = {
        'model': bp_model, 'predictions': y_pred_bp, 'actual': y_test,
        'mae': bp_mae, 'rmse': bp_rmse, 'r2': bp_r2
    }
    print(f"    MAE: {bp_mae:.4f}, RMSE: {bp_rmse:.4f}, R2: {bp_r2:.4f}")

    # --- 模型3：线性回归 ---
    print(f"  3. 训练线性回归模型...")
    linear_model = LinearRegression()
    linear_model.fit(X_train, y_train)
    y_pred_linear = linear_model.predict(X_test)

    linear_mae = mean_absolute_error(y_test, y_pred_linear)
    linear_rmse = np.sqrt(mean_squared_error(y_test, y_pred_linear))
    linear_r2 = r2_score(y_test, y_pred_linear)

    results[cluster_id]['Linear'] = {
        'model': linear_model, 'predictions': y_pred_linear, 'actual': y_test,
        'mae': linear_mae, 'rmse': linear_rmse, 'r2': linear_r2
    }
    print(f"    MAE: {linear_mae:.4f}, RMSE: {linear_rmse:.4f}, R2: {linear_r2:.4f}")


# ============================================================
# 步骤4.3：模型评估与对比
# ============================================================
print(f"\n4. 模型评估与对比")
print("-" * 60)

# --- 扁平化结果，生成汇总 DataFrame ---
summary_results = []
for cluster_id in results:
    for model_name in results[cluster_id]:
        if results[cluster_id][model_name] is not None:
            summary_results.append({
                'Cluster': cluster_id,
                'Model': model_name,
                'MAE': results[cluster_id][model_name]['mae'],
                'RMSE': results[cluster_id][model_name]['rmse'],
                'R2': results[cluster_id][model_name]['r2']
            })

if summary_results:
    summary_df = pd.DataFrame(summary_results)
    print(summary_df.round(4))

    # --- 绘制三指标柱状对比图（1行3列） ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for i, metric in enumerate(['MAE', 'RMSE', 'R2']):
        # 数据透视：行=聚类，列=模型，值=评估指标
        pivot_data = summary_df.pivot(index='Cluster', columns='Model', values=metric)
        if pivot_data is not None and not pivot_data.empty:
            pivot_data.plot(kind='bar', ax=axes[i])
            axes[i].set_title(f'{metric} 对比')
            axes[i].set_ylabel(metric)
            axes[i].legend()
            axes[i].tick_params(axis='x', rotation=45)
            axes[i].grid(True, alpha=0.3)

    plt.tight_layout()
    ensure_dir('RW4(SOLO)/model_comparison.png')
    plt.savefig('RW4(SOLO)/model_comparison.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("已保存: RW4(SOLO)/model_comparison.png")

    # --- 绘制预测值 vs 真实值曲线图 ---
    # 每个聚类 × 每个模型 生成一张独立曲线图
    for cluster_id in results:
        for model_name in results[cluster_id]:
            if results[cluster_id][model_name] is not None:
                actual = results[cluster_id][model_name]['actual']
                pred = results[cluster_id][model_name]['predictions']

                out_path = f'RW4(SOLO)/prediction_{cluster_id}_{model_name}.png'
                ensure_dir(out_path)

                plt.figure(figsize=(10, 6))
                plt.plot(range(len(actual)), actual, label='真实值', alpha=0.7)
                plt.plot(range(len(pred)), pred, label='预测值', alpha=0.7)
                plt.xlabel('样本索引')
                plt.ylabel('功率 (归一化)')
                plt.title(f'聚类 {cluster_id} - {model_name} 模型预测效果')
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.savefig(out_path, dpi=200, bbox_inches='tight')
                plt.close()
                print(f"已保存: {out_path}")


# ===========================================================================
# 第六部分：汇总输出
# ===========================================================================
print("\n" + "=" * 60)
print("所有任务完成！汇总")
print("=" * 60)
print("生成的文件:")
print("  1. scatter_matrix.png        — 六维特征散点图矩阵")
print("  2. wind_rose.png             — 风向玫瑰图")
print("  3. correlation_heatmap.png   — 相关性热力图")
print("  4. kmeans_k_selection.png    — K-means K值选择图")
print("  5. kmeans_clusters.png       — K-means聚类结果")
print("  6. speed_power_clusters.png  — 风速-功率聚类图")
print("  7. model_comparison.png      — 模型性能对比图")
print("  8. prediction_*.png          — 各模型预测效果图")
print("=" * 60)
