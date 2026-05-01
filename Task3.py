"""
风电场数据 K-means 聚类分析
================================
任务目标：使用 K-means 聚类算法对风电数据进行工况划分，识别不同运行模式，分析各类别特征差异。
采用方法：肘部法与轮廓系数法（采样优化）相结合确定最佳聚类数。

依赖: 需先运行 analysis.py 生成 data_normalized.csv 和 selected_features.json
输出: data_with_clusters.csv（归一化数据 + cluster 列，供 Task4 使用）
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

# 导入公共模块
from common import (
    ensure_dir, kmeans_cluster, save_data_with_clusters,
    load_selected_features, NUMERIC_COLS,
)


print("=" * 60)
print("任务3：风电数据 K-means 聚类分析")
print("=" * 60)

# 加载预处理后的数据
df = pd.read_csv('data_normalized.csv')
print(f"数据集形状: {df.shape[0]} 行 × {df.shape[1]} 列")

# 1. 特征选择：从 analysis.py 输出的 selected_features.json 读取，确保与 Task4 一致
try:
    selected_features = load_selected_features()
    print(f"从 selected_features.json 读取筛选特征: {selected_features}")
except FileNotFoundError:
    print("警告: 未找到 selected_features.json，将基于当前数据重新计算特征筛选。")
    from common import compute_correlation
    _, _, selected_features = compute_correlation(df)

# 聚类特征 = 筛选后的特征 + WINDPOWER（用于工况划分），去重保持顺序
cluster_features = list(dict.fromkeys(selected_features + ['WINDPOWER']))
print(f"聚类使用的特征: {cluster_features}")

# 准备聚类数据
X_cluster = df[cluster_features].values
print(f"聚类数据形状: {X_cluster.shape}")

# 2. 执行 K-means 聚类（使用公共模块，含采样优化的轮廓系数计算）
cluster_result = kmeans_cluster(
    X_cluster,
    save_k_plot_path='RW3/task3_kmeans_k_selection.png'
)

cluster_labels = cluster_result['labels']
optimal_k = cluster_result['optimal_k']

# 3. 轮廓系数法验证
print(f"\n3. 轮廓系数法验证...")
print(f"各K值对应的轮廓系数:")
for k, sil in zip(cluster_result['k_range'], cluster_result['silhouette_scores']):
    print(f"  K={k}: {sil:.4f}")

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
plt.title(f'K-means聚类结果可视化 (K={optimal_k})')
plt.colorbar(scatter)
plt.grid(True, alpha=0.3)
ensure_dir('RW3/task3_kmeans_clusters.png')
plt.savefig('RW3/task3_kmeans_clusters.png', dpi=200, bbox_inches='tight')
plt.close()
print("已保存: task3_kmeans_clusters.png")

# 风速-功率散点图，不同簇用不同颜色标记
plt.figure(figsize=(10, 6))
scatter = plt.scatter(df['WINDSPEED'], df['WINDPOWER'], c=cluster_labels, cmap='viridis', alpha=0.6, s=10)
plt.xlabel('风速 (归一化)')
plt.ylabel('功率 (归一化)')
plt.title(f'风速-功率散点图 (按聚类着色, K={optimal_k})')
plt.colorbar(scatter)
plt.grid(True, alpha=0.3)
ensure_dir('RW3/task3_speed_power_clusters.png')
plt.savefig('RW3/task3_speed_power_clusters.png', dpi=200, bbox_inches='tight')
plt.close()
print("已保存: task3_speed_power_clusters.png")

# 输出聚类统计信息
print(f"\n5. 聚类结果统计:")
total_samples = len(cluster_labels)
for i in range(optimal_k):
    count = sum(cluster_labels == i)
    percentage = count / total_samples * 100
    print(f"  类别 {i}: {count} 个样本 ({percentage:.2f}%)")

# 分析各类别特征差异
print(f"\n6. 各类别特征差异分析:")
for i in range(optimal_k):
    cluster_data = df[df['cluster'] == i][cluster_features]
    print(f"\n  类别 {i} 特征统计:")
    print(f"    样本数: {len(cluster_data)}")
    for feature in cluster_features:
        mean_val = cluster_data[feature].mean()
        print(f"    {feature} 平均值: {mean_val:.4f}")

# 评估聚类质量
overall_silhouette = cluster_result['silhouette_scores'][cluster_result['k_range'].index(optimal_k)]
print(f"\n7. 聚类质量评估:")
print(f"  整体轮廓系数: {overall_silhouette:.4f}")

# 保存带有 cluster 标签的数据集供 Task4 使用（写入独立文件，不覆盖 data_normalized.csv）
save_data_with_clusters(df, cluster_labels, 'data_with_clusters.csv')

print("\n" + "=" * 60)
print("任务3完成！")
print("=" * 60)
print("生成的文件:")
print("  1. task3_kmeans_k_selection.png   — K值选择图（肘部法和轮廓系数法）")
print("  2. task3_kmeans_clusters.png      — K-means聚类结果PCA可视化")
print("  3. task3_speed_power_clusters.png — 风速-功率聚类散点图")
print("  4. data_with_clusters.csv         — 含 cluster 列的数据集（供 Task4 使用）")
print("=" * 60)
