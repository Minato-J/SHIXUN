"""
================================================================================
风电场数据 K-means 聚类分析（详细注释版）
================================================================================

【脚本功能概述】
本脚本实现任务三：使用 K-means 聚类算法对风电数据进行工况划分。
通过将数据分成不同的运行模式（聚类），为后续 Task4 的分组预测建模奠定基础。

【核心流程】
  1. 加载归一化数据（data_normalized.csv）
  2. 加载筛选后的特征列表（selected_features.json），确保与 Task2 一致
  3. 使用肘部法 + 轮廓系数法确定最优 K 值
  4. 执行 K-means 聚类
  5. PCA 降维可视化 + 风速-功率聚类散点图
  6. 分析各类别特征差异
  7. 保存含聚类标签的数据集（data_with_clusters.csv）

【依赖关系】
  - 上游：需先运行 analysis.py 生成 data_normalized.csv 和 selected_features.json
  - 下游：生成 data_with_clusters.csv 供 Task4.py 使用

【输出文件】
  - RW3/task3_kmeans_k_selection.png   : K值选择图（肘部法 + 轮廓系数法）
  - RW3/task3_kmeans_clusters.png      : PCA降维后的聚类可视化
  - RW3/task3_speed_power_clusters.png : 风速-功率散点图（按聚类着色）
  - data_with_clusters.csv             : 含 cluster 列的数据集
"""

# ===========================================================================
# 第一部分：导入依赖库
# ===========================================================================

import pandas as pd               # 数据处理
import numpy as np                # 数值计算
import matplotlib.pyplot as plt   # 数据可视化
from sklearn.decomposition import PCA  # 主成分分析：高维数据降维到2D进行可视化
import warnings
warnings.filterwarnings('ignore')

# 从公共模块导入共享函数
# ensure_dir            : 确保目录存在再保存文件
# kmeans_cluster        : K-means 聚类分析（含K值选择）
# save_data_with_clusters: 保存含聚类标签的数据集
# load_selected_features : 从 JSON 读取特征列表
# NUMERIC_COLS          : 六维特征列名常量
from common import (
    ensure_dir, kmeans_cluster, save_data_with_clusters,
    load_selected_features, NUMERIC_COLS,
)


# ===========================================================================
# 第二部分：主程序
# ===========================================================================

print("=" * 60)
print("任务3：风电数据 K-means 聚类分析")
print("=" * 60)

# ============================================================
# 步骤1：加载预处理后的归一化数据
# ============================================================
# 读取 analysis.py 生成的归一化数据集
# 该数据已经过缺失值填充、物理异常剔除、DBSCAN去噪、Min-Max归一化
df = pd.read_csv('data_normalized.csv')
print(f"数据集形状: {df.shape[0]} 行 × {df.shape[1]} 列")

# ============================================================
# 步骤2：特征选择
# ============================================================
# 从 selected_features.json 读取 Task2 筛选后的特征列表
# 如果文件不存在（如首次运行），则回退到基于当前数据重新计算
try:
    selected_features = load_selected_features()  # 读取 JSON 文件
    print(f"从 selected_features.json 读取筛选特征: {selected_features}")
except FileNotFoundError:
    # 容错处理：JSON 文件不存在时自动重新计算特征筛选
    print("警告: 未找到 selected_features.json，将基于当前数据重新计算特征筛选。")
    from common import compute_correlation
    _, _, selected_features = compute_correlation(df)

# --- 构建聚类特征列表 ---
# 聚类特征 = 筛选后的特征 + WINDPOWER（功率是工况划分的关键变量）
# dict.fromkeys() 技巧：保持原有顺序的同时去重
cluster_features = list(dict.fromkeys(selected_features + ['WINDPOWER']))
print(f"聚类使用的特征: {cluster_features}")

# --- 准备聚类输入数据 ---
# 从 DataFrame 中提取聚类特征列，转为 numpy 数组
X_cluster = df[cluster_features].values
print(f"聚类数据形状: {X_cluster.shape}")

# ============================================================
# 步骤3：执行 K-means 聚类
# ============================================================
# 调用公共模块的 kmeans_cluster 函数：
#   - 自动搜索 K=2~10 的最优值
#   - 同时使用肘部法（SSE）和轮廓系数法评估
#   - 保存 K 值选择图为 'RW3/task3_kmeans_k_selection.png'
cluster_result = kmeans_cluster(
    X_cluster,
    save_k_plot_path='RW3/task3_kmeans_k_selection.png'
)

# 从结果字典中提取聚类标签和最优 K 值
cluster_labels = cluster_result['labels']     # 每个样本的聚类编号（0,1,...,K-1）
optimal_k = cluster_result['optimal_k']       # 最优聚类数

# ============================================================
# 步骤4：轮廓系数法验证
# ============================================================
# 打印各 K 值对应的轮廓系数，便于人工验证最优 K 选择的合理性
print(f"\n3. 轮廓系数法验证...")
print(f"各K值对应的轮廓系数:")
for k, sil in zip(cluster_result['k_range'], cluster_result['silhouette_scores']):
    print(f"  K={k}: {sil:.4f}")

# --- 将聚类标签添加到 DataFrame ---
df['cluster'] = cluster_labels

# ============================================================
# 步骤5：聚类结果可视化
# ============================================================
print("4. 聚类结果可视化...")

# --- 5.1 PCA 降维到2D进行可视化 ---
# 高维数据无法直接可视化，使用 PCA 将特征空间压缩到2维
# PCA（主成分分析）：找到数据方差最大的方向作为新坐标轴
pca = PCA(n_components=2)              # 创建 PCA 对象，目标维度=2
X_pca = pca.fit_transform(X_cluster)   # 拟合并转换数据

# 绘制 PCA 降维后的散点图，颜色按聚类标签区分
plt.figure(figsize=(12, 8))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1],
                      c=cluster_labels,           # 颜色按聚类标签
                      cmap='viridis',             # 颜色映射方案（从黄到紫）
                      alpha=0.6,                  # 半透明
                      s=20)                       # 点大小
# X轴标签包含第一主成分的解释方差比（该主成分保留了多少原始信息）
plt.xlabel(f'第一主成分 (解释方差比: {pca.explained_variance_ratio_[0]:.3f})')
plt.ylabel(f'第二主成分 (解释方差比: {pca.explained_variance_ratio_[1]:.3f})')
plt.title(f'K-means聚类结果可视化 (K={optimal_k})')
plt.colorbar(scatter)  # 添加颜色条图例
plt.grid(True, alpha=0.3)
ensure_dir('RW3/task3_kmeans_clusters.png')
plt.savefig('RW3/task3_kmeans_clusters.png', dpi=200, bbox_inches='tight')
plt.close()
print("已保存: task3_kmeans_clusters.png")

# --- 5.2 风速-功率散点图（按聚类着色） ---
# 在原始特征空间中观察聚类效果
# X轴=归一化风速，Y轴=归一化功率，颜色=聚类标签
plt.figure(figsize=(10, 6))
scatter = plt.scatter(df['WINDSPEED'], df['WINDPOWER'],
                      c=cluster_labels,
                      cmap='viridis',
                      alpha=0.6,
                      s=10)  # 稍小的点
plt.xlabel('风速 (归一化)')
plt.ylabel('功率 (归一化)')
plt.title(f'风速-功率散点图 (按聚类着色, K={optimal_k})')
plt.colorbar(scatter)
plt.grid(True, alpha=0.3)
ensure_dir('RW3/task3_speed_power_clusters.png')
plt.savefig('RW3/task3_speed_power_clusters.png', dpi=200, bbox_inches='tight')
plt.close()
print("已保存: task3_speed_power_clusters.png")

# ============================================================
# 步骤6：聚类结果统计
# ============================================================
# 统计每个聚类的样本数量和占比
print(f"\n5. 聚类结果统计:")
total_samples = len(cluster_labels)  # 总样本数
for i in range(optimal_k):
    count = sum(cluster_labels == i)                # 该聚类样本数
    percentage = count / total_samples * 100         # 该聚类占比
    print(f"  类别 {i}: {count} 个样本 ({percentage:.2f}%)")

# ============================================================
# 步骤7：各类别特征差异分析
# ============================================================
# 对每个聚类，计算各特征的平均值，分析不同运行模式的特征差异
print(f"\n6. 各类别特征差异分析:")
for i in range(optimal_k):
    # 筛选属于当前聚类的数据
    cluster_data = df[df['cluster'] == i][cluster_features]
    print(f"\n  类别 {i} 特征统计:")
    print(f"    样本数: {len(cluster_data)}")
    for feature in cluster_features:
        mean_val = cluster_data[feature].mean()
        # 打印每个特征的平均值（归一化后值在 [0,1] 之间）
        print(f"    {feature} 平均值: {mean_val:.4f}")

# ============================================================
# 步骤8：聚类质量评估
# ============================================================
# 轮廓系数范围 [-1, 1]，越接近1表示聚类质量越好
overall_silhouette = cluster_result['silhouette_scores'][
    cluster_result['k_range'].index(optimal_k)  # 找到最优K对应的轮廓系数
]
print(f"\n7. 聚类质量评估:")
print(f"  整体轮廓系数: {overall_silhouette:.4f}")

# ============================================================
# 步骤9：保存含聚类标签的数据集
# ============================================================
# 将 cluster 列添加到数据中并保存，供 Task4 使用
# 注意：写入独立文件 data_with_clusters.csv，不覆盖 data_normalized.csv
save_data_with_clusters(df, cluster_labels, 'data_with_clusters.csv')

# ============================================================
# 任务完成汇总
# ============================================================
print("\n" + "=" * 60)
print("任务3完成！")
print("=" * 60)
print("生成的文件:")
print("  1. task3_kmeans_k_selection.png   — K值选择图（肘部法和轮廓系数法）")
print("  2. task3_kmeans_clusters.png      — K-means聚类结果PCA可视化")
print("  3. task3_speed_power_clusters.png — 风速-功率聚类散点图")
print("  4. data_with_clusters.csv         — 含 cluster 列的数据集（供 Task4 使用）")
print("=" * 60)
