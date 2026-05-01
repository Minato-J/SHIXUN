"""
风电场数据分析完整项目
================================
四任务整合脚本 — 单次运行完成所有分析流程。

依赖: 需要先有 DATE.csv 原始数据（首次运行需 analysis.py 生成 data_normalized.csv）
输出: 各任务图表分别保存到 RW2/、RW3/、RW4/ 目录
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

# 导入公共模块
from common import (
    ensure_dir, compute_correlation, kmeans_cluster,
    NUMERIC_COLS, CORRELATION_THRESHOLD,
)


# ============================================================
# 加载预处理后的数据
# ============================================================
print("=" * 60)
print("加载预处理后的数据")
print("=" * 60)

df = pd.read_csv('data_normalized.csv')
print(f"数据集形状: {df.shape[0]} 行 × {df.shape[1]} 列")
print(f"列名: {list(df.columns)}")

# 将DATATIME转为datetime格式
df['DATATIME'] = pd.to_datetime(df['DATATIME'])

# ============================================================
# 任务2：可视化与相关性分析
# ============================================================
print("\n" + "=" * 60)
print("任务2：可视化与相关性分析")
print("=" * 60)

# 1. 计算统计特征
print("\n1. 统计特征表:")
print("-" * 40)
numeric_cols = ['WINDSPEED', 'WINDDIRECTION', 'TEMPERATURE', 'HUMIDITY', 'PRESSURE', 'WINDPOWER']
stats = df[numeric_cols].describe().T
stats['range'] = stats['max'] - stats['min']
stats_table = stats[['mean', 'std', 'min', '25%', '50%', '75%', 'max', 'range']]
print(stats_table.round(4))

# 2. 绘制散点图矩阵
print("\n2. 绘制散点图矩阵...")
fig = plt.figure(figsize=(15, 15))
axes = pd.plotting.scatter_matrix(
    df[numeric_cols],
    figsize=(15, 15),
    diagonal='hist',
    alpha=0.6,
    s=10
)
plt.suptitle('六维特征散点图矩阵', fontsize=16)
plt.tight_layout()
ensure_dir('RW2/scatter_matrix.png')
plt.savefig('RW2/scatter_matrix.png', dpi=200, bbox_inches='tight')
plt.close()
print("已保存: RW2/scatter_matrix.png")

# 3. 风向玫瑰图
print("\n3. 绘制风向玫瑰图...")


def wind_rose_plot(df, bins=24):
    """绘制风向玫瑰图，按15度为一个区间"""
    wind_dir_deg = df['WINDDIRECTION'] * 360
    wind_dir_rad = np.radians(wind_dir_deg)
    angle_bins = np.linspace(0, 2 * np.pi, bins + 1)
    hist, _ = np.histogram(wind_dir_rad, bins=angle_bins)
    centers = (angle_bins[:-1] + angle_bins[1:]) / 2

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    width = 2 * np.pi / bins
    bars = ax.bar(centers, hist, width=width, alpha=0.7, edgecolor='black')
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    ax.set_title('风向频率玫瑰图', pad=20, fontsize=16)
    ax.set_rlabel_position(0)
    plt.tight_layout()
    return fig, ax


fig, ax = wind_rose_plot(df, bins=24)
ensure_dir('RW2/wind_rose.png')
plt.savefig('RW2/wind_rose.png', dpi=200, bbox_inches='tight')
plt.close()
print("已保存: RW2/wind_rose.png")

# 4. 相关性分析
print("\n4. 相关性分析...")
correlation_matrix = df[numeric_cols].corr()

# 绘制相关性热力图
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
            square=True, fmt='.3f', cbar_kws={'shrink': 0.8})
plt.title('特征相关性热力图')
plt.tight_layout()
ensure_dir('RW2/correlation_heatmap.png')
plt.savefig('RW2/correlation_heatmap.png', dpi=200, bbox_inches='tight')
plt.close()
print("已保存: RW2/correlation_heatmap.png")

# 提取与功率的相关系数
power_corr = correlation_matrix['WINDPOWER'].drop('WINDPOWER').abs().sort_values(ascending=False)
print("\n各特征与功率的相关系数（按绝对值降序排列）:")
for feature, corr in power_corr.items():
    print(f"  {feature}: {corr:.4f}")

# 特征筛选（相关系数绝对值 >= 0.2）
threshold = 0.2
selected_features = power_corr[power_corr >= threshold].index.tolist()
print(f"\n相关系数绝对值 >= {threshold} 的特征（用于后续建模）:")
for feature in selected_features:
    print(f"  {feature}: {power_corr[feature]:.4f}")

# ============================================================
# 任务3：K-means聚类分析
# ============================================================
print("\n" + "=" * 60)
print("任务3：K-means聚类分析")
print("=" * 60)

# 1. 特征选择：使用筛选出的特征 + WINDPOWER（与 task3 保持一致）
cluster_features = list(dict.fromkeys(selected_features + ['WINDPOWER']))
X_cluster = df[cluster_features].values
print(f"聚类使用的特征: {cluster_features}")
print(f"聚类数据形状: {X_cluster.shape}")

# 2. 使用公共模块执行 K-means（含采样优化的轮廓系数计算）
cluster_result = kmeans_cluster(
    X_cluster,
    save_k_plot_path='RW3/kmeans_k_selection.png'
)

cluster_labels = cluster_result['labels']
optimal_k_silhouette = cluster_result['optimal_k']

# 将聚类结果添加到数据框
df['cluster'] = cluster_labels

# 4. 结果可视化
print("4. 聚类结果可视化...")

# 使用PCA降维到2D进行可视化
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_cluster)

plt.figure(figsize=(12, 8))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_labels, cmap='viridis', alpha=0.6, s=20)
plt.xlabel(f'第一主成分 (解释方差比: {pca.explained_variance_ratio_[0]:.3f})')
plt.ylabel(f'第二主成分 (解释方差比: {pca.explained_variance_ratio_[1]:.3f})')
plt.title(f'K-means聚类结果可视化 (K={optimal_k_silhouette})')
plt.colorbar(scatter)
plt.grid(True, alpha=0.3)
ensure_dir('RW3/kmeans_clusters.png')
plt.savefig('RW3/kmeans_clusters.png', dpi=200, bbox_inches='tight')
plt.close()
print("已保存: RW3/kmeans_clusters.png")

# 风速-功率散点图，按聚类着色
plt.figure(figsize=(10, 6))
scatter = plt.scatter(df['WINDSPEED'], df['WINDPOWER'], c=cluster_labels, cmap='viridis', alpha=0.6, s=10)
plt.xlabel('风速 (归一化)')
plt.ylabel('功率 (归一化)')
plt.title(f'风速-功率散点图 (按聚类着色, K={optimal_k_silhouette})')
plt.colorbar(scatter)
plt.grid(True, alpha=0.3)
ensure_dir('RW3/speed_power_clusters.png')
plt.savefig('RW3/speed_power_clusters.png', dpi=200, bbox_inches='tight')
plt.close()
print("已保存: RW3/speed_power_clusters.png")

# 输出聚类统计信息
print(f"\n聚类结果统计:")
for i in range(optimal_k_silhouette):
    count = sum(cluster_labels == i)
    percentage = count / len(cluster_labels) * 100
    print(f"  类别 {i}: {count} 个样本 ({percentage:.2f}%)")

# ============================================================
# 任务4：风电功率预测
# ============================================================
print("\n" + "=" * 60)
print("任务4：风电功率预测模型")
print("=" * 60)

# 准备数据：根据聚类结果分割数据
print(f"使用聚类结果分割数据...")
X = df[selected_features].values
y = df['WINDPOWER'].values

# 按聚类结果分割数据
clusters_data = {}
for cluster_id in range(optimal_k_silhouette):
    mask = cluster_labels == cluster_id
    clusters_data[cluster_id] = {
        'X': X[mask],
        'y': y[mask],
        'indices': np.where(mask)[0]
    }
    print(f"  类别 {cluster_id}: {len(clusters_data[cluster_id]['X'])} 个样本")

# 初始化结果存储
results = {}

# 遍历每个聚类进行建模
for cluster_id in range(optimal_k_silhouette):
    print(f"\n--- 处理聚类 {cluster_id} ---")
    X_cluster = clusters_data[cluster_id]['X']
    y_cluster = clusters_data[cluster_id]['y']

    if len(X_cluster) < 2:
        print(f"  聚类 {cluster_id} 样本数太少，跳过建模")
        continue

    X_train, X_test, y_train, y_test = train_test_split(
        X_cluster, y_cluster, test_size=0.2, random_state=42
    )

    print(f"  训练集大小: {X_train.shape[0]}, 测试集大小: {X_test.shape[0]}")

    results[cluster_id] = {}

    # 1. SVR
    print(f"  1. 训练SVR模型...")
    svr_model = SVR(kernel='rbf', C=1.0, epsilon=0.1)
    svr_model.fit(X_train, y_train)
    y_pred_svr = svr_model.predict(X_test)

    svr_mae = mean_absolute_error(y_test, y_pred_svr)
    svr_rmse = np.sqrt(mean_squared_error(y_test, y_pred_svr))
    svr_r2 = r2_score(y_test, y_pred_svr)

    results[cluster_id]['SVR'] = {
        'model': svr_model,
        'predictions': y_pred_svr,
        'actual': y_test,
        'mae': svr_mae,
        'rmse': svr_rmse,
        'r2': svr_r2
    }
    print(f"    MAE: {svr_mae:.4f}, RMSE: {svr_rmse:.4f}, R2: {svr_r2:.4f}")

    # 2. BP神经网络
    print(f"  2. 训练BP神经网络模型...")
    bp_model = MLPRegressor(
        hidden_layer_sizes=(32, 32),
        activation='relu',
        solver='adam',
        learning_rate_init=0.001,
        max_iter=500,
        random_state=42
    )
    bp_model.fit(X_train, y_train)
    y_pred_bp = bp_model.predict(X_test)

    bp_mae = mean_absolute_error(y_test, y_pred_bp)
    bp_rmse = np.sqrt(mean_squared_error(y_test, y_pred_bp))
    bp_r2 = r2_score(y_test, y_pred_bp)

    results[cluster_id]['BP'] = {
        'model': bp_model,
        'predictions': y_pred_bp,
        'actual': y_test,
        'mae': bp_mae,
        'rmse': bp_rmse,
        'r2': bp_r2
    }
    print(f"    MAE: {bp_mae:.4f}, RMSE: {bp_rmse:.4f}, R2: {bp_r2:.4f}")

    # 3. 线性回归模型
    print(f"  3. 训练线性回归模型...")
    linear_model = LinearRegression()
    linear_model.fit(X_train, y_train)
    y_pred_linear = linear_model.predict(X_test)

    linear_mae = mean_absolute_error(y_test, y_pred_linear)
    linear_rmse = np.sqrt(mean_squared_error(y_test, y_pred_linear))
    linear_r2 = r2_score(y_test, y_pred_linear)

    results[cluster_id]['Linear'] = {
        'model': linear_model,
        'predictions': y_pred_linear,
        'actual': y_test,
        'mae': linear_mae,
        'rmse': linear_rmse,
        'r2': linear_r2
    }
    print(f"    MAE: {linear_mae:.4f}, RMSE: {linear_rmse:.4f}, R2: {linear_r2:.4f}")

# 4. 模型评估与对比
print(f"\n4. 模型评估与对比")
print("-" * 60)

# 创建汇总结果表格
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

    # 绘制模型性能对比图
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for i, metric in enumerate(['MAE', 'RMSE', 'R2']):
        pivot_data = summary_df.pivot(index='Cluster', columns='Model', values=metric)
        if pivot_data is not None and not pivot_data.empty:
            pivot_data.plot(kind='bar', ax=axes[i])
            axes[i].set_title(f'{metric} 对比')
            axes[i].set_ylabel(metric)
            axes[i].legend()
            axes[i].tick_params(axis='x', rotation=45)
            axes[i].grid(True, alpha=0.3)

    plt.tight_layout()
    ensure_dir('RW4/model_comparison.png')
    plt.savefig('RW4/model_comparison.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("已保存: RW4/model_comparison.png")

    # 绘制预测值vs真实值曲线图
    for cluster_id in results:
        for model_name in results[cluster_id]:
            if results[cluster_id][model_name] is not None:
                actual = results[cluster_id][model_name]['actual']
                pred = results[cluster_id][model_name]['predictions']

                out_path = f'RW4/prediction_{cluster_id}_{model_name}.png'
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

# ============================================================
# 汇总输出
# ============================================================
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
