"""
风电场数据处理与分析
====================
任务一：数据预处理（缺失值填充、物理异常剔除、DBSCAN去噪、Min-Max归一化）
任务二：可视化与相关性分析（散点矩阵、风向玫瑰图、热力图、特征筛选）

输出:
  - data_normalized.csv       归一化数据集（供 Task3 使用）
  - selected_features.json    筛选后的特征列表（供 Task3/Task4 保持一致）
  - statistical_features.csv  六维特征统计表
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings('ignore')

# 导入公共模块（中文字体配置、ensure_dir、预处理函数等）
from common import (
    ensure_dir, load_and_preprocess, dbscan_denoise, minmax_normalize,
    compute_correlation, save_selected_features,
    NUMERIC_COLS, CORRELATION_THRESHOLD,
)


# ============================================================
# 第1步：数据加载与基本检查（使用公共模块）
# ============================================================
df_raw, df_clean = load_and_preprocess('DATE.csv')

# 补充列名及数据类型信息
print(f"\n列名及数据类型:")
print(df_raw.dtypes.to_string())

# ============================================================
# 第2步：描述性统计（原始数据）
# ============================================================
print("\n" + "=" * 60)
print("第2步：描述性统计")
print("=" * 60)

stats = df_raw[['WINDSPEED', 'WINDPOWER']].describe().T
stats['range'] = stats['max'] - stats['min']
print("\nWINDSPEED 和 WINDPOWER 的描述性统计:")
print(stats[['mean', 'std', 'min', '25%', '50%', '75%', 'max', 'range']].to_string())

# ============================================================
# 第3步：绘制各属性波形图
# ============================================================
print("\n========== 正在绘制各属性波形图... ==========")

fig, axes = plt.subplots(5, 1, figsize=(14, 12), sharex=True)
features = ['WINDSPEED', 'WINDPOWER', 'TEMPERATURE', 'HUMIDITY', 'PRESSURE']
titles  = ['风速 (m/s)', '有功功率 (kW)', '温度 (°C)', '湿度 (%)', '气压 (hPa)']
colors  = ['steelblue', 'firebrick', 'forestgreen', 'orange', 'purple']

for ax, col, title, color in zip(axes, features, titles, colors):
    ax.plot(df_raw['DATATIME'], df_raw[col], color=color, linewidth=0.3, alpha=0.7)
    ax.set_ylabel(title)
    ax.grid(True, alpha=0.3)
    if col == 'WINDPOWER':
        ax.set_ylim(bottom=0)

axes[-1].set_xlabel('DateTime')
plt.tight_layout()
ensure_dir('RW1/waveforms.png')
plt.savefig('RW1/waveforms.png', dpi=200)
plt.close()
print("已保存: waveforms.png")

# ============================================================
# 第4步：双Y轴时间序列图
# ============================================================
print("========== 正在绘制双Y轴时间序列图... ==========")

fig, ax1 = plt.subplots(figsize=(14, 5))
ax1.plot(df_raw['DATATIME'], df_raw['WINDSPEED'], color='steelblue', linewidth=0.3, alpha=0.7, label='Wind Speed (m/s)')
ax1.set_xlabel('DateTime')
ax1.set_ylabel('Wind Speed (m/s)', color='steelblue')
ax1.tick_params(axis='y', labelcolor='steelblue')
ax1.grid(True, alpha=0.3)

ax2 = ax1.twinx()
ax2.plot(df_raw['DATATIME'], df_raw['WINDPOWER'], color='firebrick', linewidth=0.3, alpha=0.7, label='Active Power (kW)')
ax2.set_ylabel('Active Power (kW)', color='firebrick')
ax2.tick_params(axis='y', labelcolor='firebrick')
plt.title('Wind Speed vs Active Power Over Time')
plt.tight_layout()
ensure_dir('RW1/timeseries_plot.png')
plt.savefig('RW1/timeseries_plot.png', dpi=200)
plt.close()
print("已保存: timeseries_plot.png")

# ============================================================
# 第5步：散点图
# ============================================================
print("========== 正在绘制散点图... ==========")

fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(df_raw['WINDSPEED'], df_raw['WINDPOWER'], s=1, alpha=0.5, c='steelblue')
ax.set_xlabel('Wind Speed (m/s)')
ax.set_ylabel('Active Power (kW)')
ax.set_title('Scatter Plot: Wind Speed vs Active Power')
ax.grid(True, alpha=0.3)
plt.tight_layout()
ensure_dir('RW1/scatter_plot.png')
plt.savefig('RW1/scatter_plot.png', dpi=200)
plt.close()
print("已保存: scatter_plot.png")

# ============================================================
# ============== 任务一：数据预处理 ================
# ============================================================
print("\n" + "=" * 60)
print("任务一：数据预处理")
print("=" * 60)

# -------- DBSCAN 聚类去噪（使用公共模块） --------
df_denoised = dbscan_denoise(df_clean, save_plot_path='RW1/scatter_denoised.png')

# -------- Min-Max 归一化（使用公共模块） --------
df_normalized, _ = minmax_normalize(df_denoised)

# 保存归一化数据集（不含 cluster 列，供 Task3 独立读取）
df_normalized.to_csv('data_normalized.csv', index=False)
print("\n已保存: data_normalized.csv")

# 归一化后散点图
fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(df_normalized['WINDSPEED'], df_normalized['WINDPOWER'], s=1, alpha=0.5, c='steelblue')
ax.set_xlabel('Wind Speed (normalized)')
ax.set_ylabel('Active Power (normalized)')
ax.set_title('Scatter Plot (After Normalization)')
ax.grid(True, alpha=0.3)
plt.tight_layout()
ensure_dir('RW1/scatter_normalized.png')
plt.savefig('RW1/scatter_normalized.png', dpi=200)
plt.close()
print("已保存: scatter_normalized.png")

# ============================================================
# 汇总输出
# ============================================================
print("\n" + "=" * 60)
print("任务一完成！汇总")
print("=" * 60)
print(f"原始数据量: {len(df_raw)} 行")
print(f"物理异常剔除后: {len(df_clean)} 行")
print(f"DBSCAN 去噪后: {len(df_denoised)} 行")
print(f"最终归一化数据: {len(df_normalized)} 行 × {len(df_normalized.columns)} 列")
print(f"\n生成的文件:")
print(f"  1. waveforms.png")
print(f"  2. timeseries_plot.png")
print(f"  3. scatter_plot.png")
print(f"  4. scatter_denoised.png")
print(f"  5. scatter_normalized.png")
print(f"  6. data_normalized.csv")

# ============================================================
# ============== 任务二：可视化与相关性分析 ================
# ============================================================
print("\n" + "=" * 60)
print("任务二：可视化与相关性分析")
print("=" * 60)

# -------- 2.1 统计特征表 --------
print("\n--------- 2.1 统计特征表 ---------")

feature_cols = ['WINDSPEED', 'WINDDIRECTION', 'TEMPERATURE', 'HUMIDITY', 'PRESSURE', 'WINDPOWER']
stat_table = df_denoised[feature_cols].describe().T
stat_table['range'] = stat_table['max'] - stat_table['min']
stat_table = stat_table[['mean', 'std', 'min', '25%', '50%', '75%', 'max', 'range']]
stat_table.columns = ['均值', '标准差', '最小值', 'Q1(25%)', 'Q2(中位数)', 'Q3(75%)', '最大值', '极差']
print("\n六维特征统计特征表:")
print(stat_table.to_string())
stat_table.to_csv('statistical_features.csv', float_format='%.4f')
print("\n已保存: statistical_features.csv")

# -------- 2.2 六维特征散点图矩阵 --------
print("\n--------- 2.2 六维特征散点图矩阵 ---------")

feature_names = ['WINDSPEED', 'WINDDIRECTION', 'TEMPERATURE', 'HUMIDITY', 'PRESSURE', 'WINDPOWER']
label_names = ['风速', '风向', '温度', '湿度', '压强', '功率']
n = len(feature_names)

fig, axes = plt.subplots(n, n, figsize=(16, 16))

for i in range(n):
    for j in range(n):
        ax = axes[i, j]
        if i == j:
            # 对角线：频率分布直方图
            ax.hist(df_denoised[feature_names[i]], bins=50, color='steelblue', edgecolor='white', alpha=0.7)
            ax.set_xlabel('')
            ax.set_ylabel('')
        else:
            # 非对角线：散点图
            ax.scatter(df_denoised[feature_names[j]], df_denoised[feature_names[i]], 
                       s=0.5, alpha=0.3, c='steelblue')
        if i == n - 1:
            ax.set_xlabel(label_names[j], fontsize=8)
        if j == 0:
            ax.set_ylabel(label_names[i], fontsize=8)
        ax.tick_params(axis='both', labelsize=6)

plt.suptitle('六维特征散点图矩阵', fontsize=14)
plt.tight_layout()
ensure_dir('RW2/scatter_matrix.png')
plt.savefig('RW2/scatter_matrix.png', dpi=150)
plt.close()
print("已保存: scatter_matrix.png")

# -------- 2.3 风向玫瑰图 --------
print("\n--------- 2.3 风向玫瑰图 ---------")

# 按15度分箱
wind_dir = df_denoised['WINDDIRECTION'].values
bins = np.arange(0, 361, 15)
labels_bins = np.arange(0, 360, 15)
digits = np.digitize(wind_dir, bins) - 1
counts = np.bincount(digits[digits >= 0], minlength=len(labels_bins))

# 转换成极坐标柱状图
angles = np.radians(np.arange(0, 360, 15))
width = np.radians(15)

fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={'projection': 'polar'})
bars = ax.bar(angles, counts, width=width, bottom=0.0, color='steelblue', edgecolor='white', alpha=0.7)
ax.set_theta_direction(-1)
ax.set_theta_offset(np.pi / 2)
ax.set_title('风向玫瑰图 (15° 分箱)', fontsize=14, pad=20)
plt.tight_layout()
ensure_dir('RW2/wind_rose.png')
plt.savefig('RW2/wind_rose.png', dpi=150)
plt.close()
print("已保存: wind_rose.png")

# -------- 2.4 相关性分析与特征筛选 --------
print("\n--------- 2.4 相关性分析与特征筛选 ---------")

# 使用公共模块计算相关性（在去噪后的数据上进行）
corr_matrix, power_corr_sorted, selected_features = compute_correlation(df_denoised)

# 打印相关系数矩阵
print("\n皮尔逊相关系数矩阵:")
print(corr_matrix.to_string())

# 绘制热力图（使用 imshow，无需额外依赖 seaborn）
fig, ax = plt.subplots(figsize=(8, 7))
im = ax.imshow(corr_matrix.values, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')

ax.set_xticks(range(len(feature_cols)))
ax.set_yticks(range(len(feature_cols)))
ax.set_xticklabels(label_names, fontsize=10)
ax.set_yticklabels(label_names, fontsize=10)

# 添加数值标签
for i in range(len(feature_cols)):
    for j in range(len(feature_cols)):
        val = corr_matrix.values[i, j]
        color = 'white' if abs(val) > 0.5 else 'black'
        ax.text(j, i, f'{val:.2f}', ha='center', va='center', fontsize=8, color=color)

plt.title('皮尔逊相关系数热力图', fontsize=14)
plt.colorbar(im, shrink=0.8)
plt.tight_layout()
ensure_dir('RW2/correlation_heatmap.png')
plt.savefig('RW2/correlation_heatmap.png', dpi=150)
plt.close()
print("已保存: correlation_heatmap.png")

# 保存筛选后的特征列表供下游脚本（Task3/Task4）使用，确保特征一致性
save_selected_features(selected_features)

print("\n" + "=" * 60)
print("任务二完成！")
print("=" * 60)
print(f"生成的文件:")
print(f"  1. statistical_features.csv   — 六维特征统计表")
print(f"  2. scatter_matrix.png         — 六维特征散点图矩阵")
print(f"  3. wind_rose.png              — 风向玫瑰图 (15° 分箱)")
print(f"  4. correlation_heatmap.png    — 皮尔逊相关系数热力图")
print(f"  5. selected_features.json     — 筛选后的特征列表")
print("=" * 60)
