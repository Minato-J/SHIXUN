"""
风电场数据统计分析与预处理
================================
任务一：数据预处理（已完成）
任务二：可视化与相关性分析
任务三：K-means 聚类分析
任务四：风电功率预测（SVR / BP / LSTM）
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score, mean_absolute_error, mean_squared_error, r2_score
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 设置中文字体
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
    ax.plot(df['DATATIME'], df[col], color=color, linewidth=0.3, alpha=0.7)
    ax.set_ylabel(title)
    ax.grid(True, alpha=0.3)
    if col == 'WINDPOWER':
        ax.set_ylim(bottom=0)

axes[-1].set_xlabel('DateTime')
plt.tight_layout()
plt.savefig('RW1/waveforms.png', dpi=200)
plt.close()
print("已保存: waveforms.png")

# ============================================================
# 第4步：双Y轴时间序列图
# ============================================================
print("========== 正在绘制双Y轴时间序列图... ==========")

fig, ax1 = plt.subplots(figsize=(14, 5))
ax1.plot(df['DATATIME'], df['WINDSPEED'], color='steelblue', linewidth=0.3, alpha=0.7, label='Wind Speed (m/s)')
ax1.set_xlabel('DateTime')
ax1.set_ylabel('Wind Speed (m/s)', color='steelblue')
ax1.tick_params(axis='y', labelcolor='steelblue')
ax1.grid(True, alpha=0.3)

ax2 = ax1.twinx()
ax2.plot(df['DATATIME'], df['WINDPOWER'], color='firebrick', linewidth=0.3, alpha=0.7, label='Active Power (kW)')
ax2.set_ylabel('Active Power (kW)', color='firebrick')
ax2.tick_params(axis='y', labelcolor='firebrick')
plt.title('Wind Speed vs Active Power Over Time')
plt.tight_layout()
plt.savefig('RW1/timeseries_plot.png', dpi=200)
plt.close()
print("已保存: timeseries_plot.png")

# ============================================================
# 第5步：散点图
# ============================================================
print("========== 正在绘制散点图... ==========")

fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(df['WINDSPEED'], df['WINDPOWER'], s=1, alpha=0.5, c='steelblue')
ax.set_xlabel('Wind Speed (m/s)')
ax.set_ylabel('Active Power (kW)')
ax.set_title('Scatter Plot: Wind Speed vs Active Power')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('RW1/scatter_plot.png', dpi=200)
plt.close()
print("已保存: scatter_plot.png")

# ============================================================
# ============== 任务一：数据预处理 ================
# ============================================================
print("\n" + "=" * 60)
print("任务一：数据预处理")
print("=" * 60)

# -------- 缺失值填充 --------
print("\n--------- 缺失值填充 ---------")
if missing.sum() == 0:
    print("当前数据无缺失值，无需填充。")
else:
    for col in df.columns:
        if col == 'DATATIME':
            continue
        df[col] = df[col].fillna(method='ffill')
    print("短时缺失已使用前向填充法处理。")

# -------- 物理异常值剔除 --------
print("\n--------- 物理异常值剔除 ---------")
print(f"处理前的数据量: {len(df)}")

neg_power = df['WINDPOWER'] < 0
print(f"功率负值数量: {neg_power.sum()}")
df = df[~neg_power].reset_index(drop=True)

neg_wind = df['WINDSPEED'] < 0
print(f"风速负值数量: {neg_wind.sum()}")
df = df[~neg_wind].reset_index(drop=True)
print(f"物理剔除后的数据量: {len(df)}")

# -------- DBSCAN 聚类去噪 --------
print("\n--------- DBSCAN 聚类去噪 ---------")

X = df[['WINDSPEED', 'WINDPOWER']].values
scaler_std = StandardScaler()
X_scaled = scaler_std.fit_transform(X)

sample_size = min(5000, len(X_scaled))
np.random.seed(42)
idx_sample = np.random.choice(len(X_scaled), sample_size, replace=False)
X_sample = X_scaled[idx_sample]

neigh = NearestNeighbors(n_neighbors=10)
neigh.fit(X_sample)
distances, _ = neigh.kneighbors(X_sample)
k_dist = np.sort(distances[:, -1])
epsilon = np.percentile(k_dist, 95)
min_pts = 10
print(f"DBSCAN 参数: epsilon={epsilon:.3f}, min_pts={min_pts}")

db = DBSCAN(eps=epsilon, min_samples=min_pts)
labels = db.fit_predict(X_scaled)

df['cluster'] = labels
n_noise = (labels == -1).sum()
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
print(f"检测到的聚类数: {n_clusters}")
print(f"检测到的噪声点数量: {n_noise} (占比 {n_noise/len(df)*100:.2f}%)")

# 去噪前后对比图
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
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

df_clean = df[df['cluster'] != -1].copy()
axes[1].scatter(df_clean['WINDSPEED'], df_clean['WINDPOWER'], s=1, alpha=0.5, c='forestgreen')
axes[1].set_xlabel('Wind Speed (m/s)')
axes[1].set_ylabel('Active Power (kW)')
axes[1].set_title('After: Outliers Removed')
axes[1].grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('RW1/scatter_denoised.png', dpi=200)
plt.close()
print("已保存: scatter_denoised.png")

df_clean = df_clean.drop(columns=['cluster']).reset_index(drop=True)
print(f"\nDBSCAN 去噪后的数据量: {len(df_clean)}")

# -------- Min-Max 归一化 --------
print("\n--------- 数据归一化 ---------")

numeric_cols = ['WINDSPEED', 'WINDDIRECTION', 'TEMPERATURE', 'HUMIDITY', 'PRESSURE', 'WINDPOWER']

scaler = MinMaxScaler()
df_normalized = df_clean.copy()
df_normalized[numeric_cols] = scaler.fit_transform(df_clean[numeric_cols])

print("归一化参数 (各特征的 min, max):")
for i, col in enumerate(numeric_cols):
    print(f"  {col}:  min={scaler.data_min_[i]:.4f}, max={scaler.data_max_[i]:.4f}")

print(f"\n归一化后的数据范围:")
print(df_normalized[numeric_cols].describe().loc[['min','max']].to_string())

df_normalized.to_csv('data_normalized.csv', index=False)
print("\n已保存: data_normalized.csv")

fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(df_normalized['WINDSPEED'], df_normalized['WINDPOWER'], s=1, alpha=0.5, c='steelblue')
ax.set_xlabel('Wind Speed (normalized)')
ax.set_ylabel('Active Power (normalized)')
ax.set_title('Scatter Plot (After Normalization)')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('RW1/scatter_normalized.png', dpi=200)
plt.close()
print("已保存: scatter_normalized.png")

# ============================================================
# 汇总输出
# ============================================================
print("\n" + "=" * 60)
print("任务一完成！汇总")
print("=" * 60)
print(f"原始数据量: 39439 行")
print(f"物理异常剔除后: {len(df)} 行")
print(f"DBSCAN 去噪后: {len(df_clean)} 行")
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
stat_table = df_clean[feature_cols].describe().T
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
            ax.hist(df_clean[feature_names[i]], bins=50, color='steelblue', edgecolor='white', alpha=0.7)
            ax.set_xlabel('')
            ax.set_ylabel('')
        else:
            # 非对角线：散点图
            ax.scatter(df_clean[feature_names[j]], df_clean[feature_names[i]], 
                       s=0.5, alpha=0.3, c='steelblue')
        if i == n - 1:
            ax.set_xlabel(label_names[j], fontsize=8)
        if j == 0:
            ax.set_ylabel(label_names[i], fontsize=8)
        ax.tick_params(axis='both', labelsize=6)

plt.suptitle('六维特征散点图矩阵', fontsize=14)
plt.tight_layout()
plt.savefig('scatter_matrix.png', dpi=150)
plt.close()
print("已保存: scatter_matrix.png")

# -------- 2.3 风向玫瑰图 --------
print("\n--------- 2.3 风向玫瑰图 ---------")

# 按15度分箱
wind_dir = df_clean['WINDDIRECTION'].values
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
plt.savefig('wind_rose.png', dpi=150)
plt.close()
print("已保存: wind_rose.png")

# -------- 2.4 相关性分析与特征筛选 --------
print("\n--------- 2.4 相关性分析与特征筛选 ---------")

corr_matrix = df_clean[feature_cols].corr(method='pearson')
print("\n皮尔逊相关系数矩阵:")
print(corr_matrix.to_string())

# 绘制热力图
fig, ax = plt.subplots(figsize=(8, 7))
im = ax.imshow(corr_matrix.values, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')

ax.set_xticks(range(n))
ax.set_yticks(range(n))
ax.set_xticklabels(label_names, fontsize=10)
ax.set_yticklabels(label_names, fontsize=10)

# 添加数值标签
for i in range(n):
    for j in range(n):
        val = corr_matrix.values[i, j]
        color = 'white' if abs(val) > 0.5 else 'black'
        ax.text(j, i, f'{val:.2f}', ha='center', va='center', fontsize=8, color=color)

plt.title('皮尔逊相关系数热力图', fontsize=14)
plt.colorbar(im, shrink=0.8)
plt.tight_layout()
plt.savefig('correlation_heatmap.png', dpi=150)
plt.close()
print("已保存: correlation_heatmap.png")

# 特征筛选：提取各特征与功率的相关系数
power_corr = corr_matrix['WINDPOWER'].drop('WINDPOWER').abs()
power_corr_sorted = power_corr.sort_values(ascending=False)
print("\n各特征与功率的相关系数绝对值（降序）:")
print(power_corr_sorted.to_string())

# 设定阈值 >= 0.2 筛选关键特征
threshold = 0.2
selected_features = power_corr[power_corr >= threshold].index.tolist()
print(f"\n设定阈值 ≥ {threshold}")
print(f"筛选后的关键特征: {selected_features}")
print(f"筛选后的特征数: {len(selected_features)}")


