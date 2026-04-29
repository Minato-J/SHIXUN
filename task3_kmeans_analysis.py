"""
风电场数据 K-means 聚类分析
================================
任务目标：使用 K-means 聚类算法对风电数据进行工况划分，识别不同运行模式，分析各类别特征差异。
采用方法：K-means 聚类算法，肘部法与轮廓系数法相结合确定最佳聚类数。
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

print("=" * 60)
print("任务3：风电数据 K-means 聚类分析")
print("=" * 60)

# 加载预处理后的数据
df = pd.read_csv('data_normalized.csv')
print(f"数据集形状: {df.shape[0]} 行 × {df.shape[1]} 列")

# 1. 特征选择：基于任务2筛选的特征
numeric_cols = ['WINDSPEED', 'WINDDIRECTION', 'TEMPERATURE', 'HUMIDITY', 'PRESSURE', 'WINDPOWER']
correlation_matrix = df[numeric_cols].corr()
power_corr = correlation_matrix['WINDPOWER'].drop('WINDPOWER').abs().sort_values(ascending=False)

# 特征筛选（相关系数绝对值 >= 0.2）
threshold = 0.2
selected_features = power_corr[power_corr >= threshold].index.tolist()
print(f"基于相关性分析筛选的特征: {selected_features}")

# 结合风速和功率进行聚类（加入 WINDPOWER 用于工况划分）
cluster_features = selected_features + ['WINDPOWER']
print(f"聚类使用的特征（结合风速和功率）: {cluster_features}")

# 准备聚类数据
X_cluster = df[cluster_features].values
print(f"聚类数据形状: {X_cluster.shape}")

# 2. 肘部法确定K值
print("\n2. 使用肘部法确定最优K值...")

k_range = range(2, 11)
sse = []  # Sum of Squared Errors
silhouette_scores = []

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_cluster)
    sse.append(kmeans.inertia_)
    
    # 轮廓系数计算使用采样数据（O(n²)复杂度，采样加速）
    sample_size = min(5000, len(X_cluster))
    rng = np.random.RandomState(42)
    idx_sample = rng.choice(len(X_cluster), sample_size, replace=False)
    silhouette_avg = silhouette_score(X_cluster[idx_sample], kmeans.labels_[idx_sample])
    silhouette_scores.append(silhouette_avg)
    print(f"  K={k}: SSE={sse[-1]:.2f}, 轮廓系数={silhouette_avg:.4f}")

# 绘制肘部法和轮廓系数图
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

# 肘部法图
ax1.plot(k_range, sse, 'bo-')
ax1.set_xlabel('聚类数 K')
ax1.set_ylabel('SSE (误差平方和)')
ax1.set_title('肘部法确定最优K值')
ax1.grid(True, alpha=0.3)

# 轮廓系数图
ax2.plot(k_range, silhouette_scores, 'ro-')
ax2.set_xlabel('聚类数 K')
ax2.set_ylabel('轮廓系数')
ax2.set_title('轮廓系数法确定最优K值')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('task3_kmeans_k_selection.png', dpi=200, bbox_inches='tight')
plt.close()
print("已保存: task3_kmeans_k_selection.png")

# 找到最优K值（基于轮廓系数）
optimal_k = k_range[np.argmax(silhouette_scores)]
print(f"\n基于轮廓系数的最优K值: {optimal_k}")
print(f"对应的轮廓系数: {max(silhouette_scores):.4f}")

# 3. 轮廓系数法验证
print(f"\n3. 轮廓系数法验证...")
print(f"各K值对应的轮廓系数:")
for k_idx, k in enumerate(k_range):
    print(f"  K={k}: {silhouette_scores[k_idx]:.4f}")

# 4. 执行聚类
print(f"\n4. 使用最优K值({optimal_k})执行K-means聚类...")
kmeans_final = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
cluster_labels = kmeans_final.fit_predict(X_cluster)

# 将聚类结果添加到数据框
df['cluster'] = cluster_labels

# 5. 结果可视化
print("5. 聚类结果可视化...")

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
plt.savefig('task3_kmeans_clusters.png', dpi=200, bbox_inches='tight')
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
plt.savefig('task3_speed_power_clusters.png', dpi=200, bbox_inches='tight')
plt.close()
print("已保存: task3_speed_power_clusters.png")

# 输出聚类统计信息
print(f"\n6. 聚类结果统计:")
total_samples = len(cluster_labels)
for i in range(optimal_k):
    count = sum(cluster_labels == i)
    percentage = count / total_samples * 100
    print(f"  类别 {i}: {count} 个样本 ({percentage:.2f}%)")

# 分析各类别特征差异
print(f"\n7. 各类别特征差异分析:")
for i in range(optimal_k):
    cluster_data = df[df['cluster'] == i][cluster_features]
    print(f"\n  类别 {i} 特征统计:")
    print(f"    样本数: {len(cluster_data)}")
    for feature in cluster_features:
        mean_val = cluster_data[feature].mean()
        print(f"    {feature} 平均值: {mean_val:.4f}")

# 评估聚类质量（采样计算）
sample_size_final = min(5000, len(X_cluster))
rng = np.random.RandomState(42)
idx_sample_final = rng.choice(len(X_cluster), sample_size_final, replace=False)
overall_silhouette = silhouette_score(X_cluster[idx_sample_final], cluster_labels[idx_sample_final])
print(f"\n8. 聚类质量评估（采样 {sample_size_final} 条）:")
print(f"  整体轮廓系数: {overall_silhouette:.4f}")

print("\n" + "=" * 60)
print("任务3完成！")
print("=" * 60)
print("生成的文件:")
print("  1. task3_kmeans_k_selection.png   — K值选择图（肘部法和轮廓系数法）")
print("  2. task3_kmeans_clusters.png      — K-means聚类结果PCA可视化")
print("  3. task3_speed_power_clusters.png — 风速-功率聚类散点图")
print("=" * 60)