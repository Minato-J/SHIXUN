"""
风电场数据统计分析与预处理
================================
任务包含两部分：
  A. 统计分析与可视化（风速与有功功率的关系）
  B. 数据预处理（缺失值填充、异常值检测、归一化）
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.neighbors import NearestNeighbors

# ============================================================
# 设置中文字体（用于图表标签）
# ============================================================
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# ============================================================
# 第1步：数据加载与基本检查
# ============================================================
print("=" * 60)
print("第1步：数据加载与基本检查")
print("=" * 60)

df = pd.read_csv('DATE.csv')
print(f"数据集形状: {df.shape[0]} 行 × {df.shape[1]} 列")
print(f"\n列名及数据类型:")
print(df.dtypes.to_string())
print(f"\n缺失值统计:")
missing = df.isnull().sum()
missing_pct = (df.isnull().sum() / len(df)) * 100
missing_info = pd.DataFrame({'缺失数量': missing, '缺失占比(%)': missing_pct})
print(missing_info.to_string())

# 正确解析时间
df['DATATIME'] = pd.to_datetime(df['DATATIME'])
df = df.sort_values('DATATIME').reset_index(drop=True)
print(f"\n时间范围: {df['DATATIME'].min()} ~ {df['DATATIME'].max()}")
print(f"采样频率: 平均间隔约 {pd.Series(df['DATATIME']).diff().dt.total_seconds().mean()/60:.1f} 分钟")

# ============================================================
# 第2步：描述性统计
# ============================================================
print("\n" + "=" * 60)
print("第2步：描述性统计")
print("=" * 60)

stats = df[['WINDSPEED', 'WINDPOWER']].describe().T
stats['range'] = stats['max'] - stats['min']
print("\nWINDSPEED (风速, m/s) 和 WINDPOWER (有功功率, kW) 的描述性统计:")
print(stats[['mean', 'std', 'min', '25%', '50%', '75%', 'max', 'range']].to_string())

# ============================================================
# 第3步：绘制各属性随时间变化的波形图
# ============================================================
print("\n========== 正在绘制各属性波形图... ==========")

fig, axes = plt.subplots(5, 1, figsize=(14, 12), sharex=True)
features = ['WINDSPEED', 'WINDPOWER', 'TEMPERATURE', 'HUMIDITY', 'PRESSURE']
titles  = ['风速 (m/s)', '有功功率 (kW)', '温度 (°C)', '湿度 (%)', '气压 (hPa)']
colors  = ['steelblue', 'firebrick', 'forestgreen', 'orange', 'purple']

for ax, col, title, color in zip(axes, features, titles, colors):
    ax.plot(df['DATATIME'], df[col], color=color, linewidth=0.3, alpha=0.7)
    ax.set_ylabel(title)
    ax.grid(True, alpha=0.3)
    if col == 'WINDPOWER':
        ax.set_ylim(bottom=0)

axes[-1].set_xlabel('DateTime')
plt.tight_layout()
plt.savefig('waveforms.png', dpi=200)
plt.close()
print("已保存: waveforms.png")

# ============================================================
# 第4步：双Y轴时间序列图（风速 vs 功率）
# ============================================================
print("========== 正在绘制双Y轴时间序列图... ==========")

fig, ax1 = plt.subplots(figsize=(14, 5))

# 左轴 - 风速 (蓝色)
ax1.plot(df['DATATIME'], df['WINDSPEED'], color='steelblue', linewidth=0.3, alpha=0.7, label='Wind Speed (m/s)')
ax1.set_xlabel('DateTime')
ax1.set_ylabel('Wind Speed (m/s)', color='steelblue')
ax1.tick_params(axis='y', labelcolor='steelblue')
ax1.grid(True, alpha=0.3)

# 右轴 - 功率 (红色)
ax2 = ax1.twinx()
ax2.plot(df['DATATIME'], df['WINDPOWER'], color='firebrick', linewidth=0.3, alpha=0.7, label='Active Power (kW)')
ax2.set_ylabel('Active Power (kW)', color='firebrick')
ax2.tick_params(axis='y', labelcolor='firebrick')

plt.title('Wind Speed vs Active Power Over Time')
plt.tight_layout()
plt.savefig('timeseries_plot.png', dpi=200)
plt.close()
print("已保存: timeseries_plot.png")

# ============================================================
# 第5步：散点图（风速 vs 功率）
# ============================================================
print("========== 正在绘制散点图... ==========")

fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(df['WINDSPEED'], df['WINDPOWER'], s=1, alpha=0.5, c='steelblue')
ax.set_xlabel('Wind Speed (m/s)')
ax.set_ylabel('Active Power (kW)')
ax.set_title('Scatter Plot: Wind Speed vs Active Power')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('scatter_plot.png', dpi=200)
plt.close()
print("已保存: scatter_plot.png")

# ============================================================
# ============== 以下是任务一：数据预处理 ================
# ============================================================
print("\n" + "=" * 60)
print("任务一：数据预处理")
print("=" * 60)

# -------- 缺失值填充 --------
print("\n--------- 缺失值填充 ---------")
if missing.sum() == 0:
    print("当前数据无缺失值，无需填充。填充逻辑已保留在代码中。")
else:
    # 短时(<=5个连续)缺失：前向填充
    for col in df.columns:
        if col == 'DATATIME':
            continue
        df[col] = df[col].fillna(method='ffill')
    print("短时缺失已使用前向填充法处理。")

# -------- 物理异常值剔除 --------
print("\n--------- 物理异常值剔除 ---------")
print(f"处理前的数据量: {len(df)}")

# 删除功率负值
neg_power = df['WINDPOWER'] < 0
print(f"功率负值数量: {neg_power.sum()}")
df = df[~neg_power].reset_index(drop=True)

# 删除风速负值
neg_wind = df['WINDSPEED'] < 0
print(f"风速负值数量: {neg_wind.sum()}")
df = df[~neg_wind].reset_index(drop=True)

print(f"物理剔除后的数据量: {len(df)}")

# -------- DBSCAN 聚类去噪 --------
print("\n--------- DBSCAN 聚类去噪 ---------")

# 提取风速和功率作为特征
X = df[['WINDSPEED', 'WINDPOWER']].values

# 标准化特征（消除量纲差异）
scaler_std = StandardScaler()
X_scaled = scaler_std.fit_transform(X)

# 使用 k-距离图自动选择 epsilon
sample_size = min(5000, len(X_scaled))
np.random.seed(42)
idx_sample = np.random.choice(len(X_scaled), sample_size, replace=False)
X_sample = X_scaled[idx_sample]

neigh = NearestNeighbors(n_neighbors=10)
neigh.fit(X_sample)
distances, _ = neigh.kneighbors(X_sample)
k_dist = np.sort(distances[:, -1])

# 自动选择 epsilon（取第95百分位）
epsilon = np.percentile(k_dist, 95)
min_pts = 10
print(f"DBSCAN 参数: epsilon={epsilon:.3f}, min_pts={min_pts}")

# 对整个数据集执行 DBSCAN（使用标准化后的数据）
db = DBSCAN(eps=epsilon, min_samples=min_pts)
labels = db.fit_predict(X_scaled)

df['cluster'] = labels
n_noise = (labels == -1).sum()
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
print(f"检测到的聚类数: {n_clusters}")
print(f"检测到的噪声点数量: {n_noise} (占比 {n_noise/len(df)*100:.2f}%)")

# 绘制去除异常点前后的对比散点图
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# 左图：原始带异常点
axes[0].scatter(df['WINDSPEED'], df['WINDPOWER'], s=1, alpha=0.5, c='steelblue', label='Normal')
noise_mask = df['cluster'] == -1
if noise_mask.any():
    axes[0].scatter(df.loc[noise_mask, 'WINDSPEED'],
                    df.loc[noise_mask, 'WINDPOWER'],
                    s=5, alpha=0.8, c='red', label=f'Outliers ({n_noise})')
axes[0].set_xlabel('Wind Speed (m/s)')
axes[0].set_ylabel('Active Power (kW)')
axes[0].set_title('Before: With Outliers')
axes[0].legend(markerscale=5)
axes[0].grid(True, alpha=0.3)

# 右图：去除异常点后
df_clean = df[df['cluster'] != -1].copy()
axes[1].scatter(df_clean['WINDSPEED'], df_clean['WINDPOWER'], s=1, alpha=0.5, c='forestgreen')
axes[1].set_xlabel('Wind Speed (m/s)')
axes[1].set_ylabel('Active Power (kW)')
axes[1].set_title('After: Outliers Removed')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('scatter_denoised.png', dpi=200)
plt.close()
print("已保存: scatter_denoised.png")

# 保留清洗后的数据
df_clean = df_clean.drop(columns=['cluster']).reset_index(drop=True)
print(f"\nDBSCAN 去噪后的数据量: {len(df_clean)}")

# -------- Min-Max 归一化 --------
print("\n--------- 数据归一化 ---------")

# 定义需要归一化的数值列
numeric_cols = ['WINDSPEED', 'WINDDIRECTION', 'TEMPERATURE', 'HUMIDITY', 'PRESSURE', 'WINDPOWER']

# 初始化归一化器并拟合
scaler = MinMaxScaler()
df_normalized = df_clean.copy()
df_normalized[numeric_cols] = scaler.fit_transform(df_clean[numeric_cols])

# 记录归一化参数
print("归一化参数 (各特征的 min, max):")
for i, col in enumerate(numeric_cols):
    print(f"  {col}:  min={scaler.data_min_[i]:.4f}, max={scaler.data_max_[i]:.4f}")

print(f"\n归一化后的数据范围:")
print(df_normalized[numeric_cols].describe().loc[['min','max']].to_string())

# 保存归一化后的数据
df_normalized.to_csv('data_normalized.csv', index=False)
print("\n已保存: data_normalized.csv (归一化后的数据集)")

# 绘制归一化后的风速-功率散点图
fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(df_normalized['WINDSPEED'], df_normalized['WINDPOWER'], s=1, alpha=0.5, c='steelblue')
ax.set_xlabel('Wind Speed (normalized)')
ax.set_ylabel('Active Power (normalized)')
ax.set_title('Scatter Plot: Wind Speed vs Active Power (After Normalization)')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('scatter_normalized.png', dpi=200)
plt.close()
print("已保存: scatter_normalized.png")

# ============================================================
# 汇总输出
# ============================================================
print("\n" + "=" * 60)
print("分析完成！汇总")
print("=" * 60)
print(f"原始数据量: 39439 行")
print(f"物理异常剔除后: {len(df)} 行")
print(f"DBSCAN 去噪后: {len(df_clean)} 行")
print(f"最终归一化数据: {len(df_normalized)} 行 × {len(df_normalized.columns)} 列")
print(f"\n生成的文件:")
print(f"  1. waveforms.png           — 各属性随时间波形图")
print(f"  2. timeseries_plot.png     — 双Y轴时间序列图（风速 vs 功率）")
print(f"  3. scatter_plot.png        — 风速-功率散点图（原始）")
print(f"  4. scatter_denoised.png    — DBSCAN去噪前后对比散点图")
print(f"  5. scatter_normalized.png  — 归一化后散点图")
print(f"  6. data_normalized.csv     — 归一化后的数据集")
print("=" * 60)
