"""
================================================================================
风电场数据处理与分析（详细注释版）
================================================================================

【脚本功能概述】
本脚本是项目的核心分析脚本，整合了任务一和任务二：
  任务一：数据预处理
    - 缺失值填充（前向填充法）
    - 物理异常值剔除（负风速、负功率）
    - DBSCAN 密度聚类去噪（自动估计 epsilon 参数）
    - Min-Max 归一化（缩放到 [0,1] 区间）
  任务二：可视化与相关性分析
    - 各属性波形图（5个子图：风速/功率/温度/湿度/气压）
    - 双Y轴时间序列图（风速+功率）
    - 散点图（风速 vs 功率）
    - 六维特征散点图矩阵（含直方图对角线）
    - 风向玫瑰图（15°分箱极坐标柱状图）
    - 皮尔逊相关系数热力图
    - 统计特征表（均值/标准差/分位数/极差）

【输出文件说明】
  - data_normalized.csv       : 归一化后的数据集（供 Task3 读取使用）
  - selected_features.json    : 筛选后的特征列表（供 Task3/Task4 保持一致）
  - statistical_features.csv  : 六维特征的描述性统计表
  - RW1/waveforms.png         : 五属性波形图
  - RW1/timeseries_plot.png   : 风速-功率双Y轴时间序列图
  - RW1/scatter_plot.png      : 风速-功率原始散点图
  - RW1/scatter_denoised.png  : DBSCAN 去噪前后对比图
  - RW1/scatter_normalized.png: 归一化后散点图
  - RW2/scatter_matrix.png    : 六维特征散点图矩阵
  - RW2/wind_rose.png         : 风向频率玫瑰图
  - RW2/correlation_heatmap.png: 皮尔逊相关系数热力图

【运行前提】
  需要项目根目录下存在 DATE.csv 原始数据文件。
"""

# ===========================================================================
# 第一部分：导入依赖库
# ===========================================================================

import pandas as pd               # 数据处理：DataFrame、CSV读写
import numpy as np                # 数值计算：数组运算、数学函数
import matplotlib.pyplot as plt   # 数据可视化：绘制各类图表
import os                         # 操作系统接口：路径处理
import warnings                   # 警告控制
warnings.filterwarnings('ignore') # 忽略所有警告，保持输出整洁

# --- 从公共模块导入共享函数和常量 ---
# ensure_dir           : 确保输出文件的父目录存在
# load_and_preprocess  : 加载原始数据并执行预处理（缺失值填充、物理异常剔除）
# dbscan_denoise       : DBSCAN 密度聚类去噪
# minmax_normalize     : Min-Max 归一化
# compute_correlation  : 皮尔逊相关性分析 + 特征筛选
# save_selected_features: 将筛选后的特征列表保存为 JSON
# NUMERIC_COLS         : 六维数值特征列名常量
# CORRELATION_THRESHOLD: 相关性筛选阈值常量
from common import (
    ensure_dir, load_and_preprocess, dbscan_denoise, minmax_normalize,
    compute_correlation, save_selected_features,
    NUMERIC_COLS, CORRELATION_THRESHOLD,
)


# ===========================================================================
# 第二部分：任务一 - 数据预处理
# ===========================================================================

# ============================================================
# 第1步：数据加载与基本检查
# ============================================================
# 调用公共模块的 load_and_preprocess 函数完成：
#   1) CSV 文件读取
#   2) 缺失值检测与前向填充（用前一个有效值填充）
#   3) 时间列解析与按时间排序
#   4) 物理异常值剔除（负风速/负功率记录）
# 返回值：
#   - df_raw  : 原始含异常值的 DataFrame
#   - df_clean: 剔除物理异常后的清洗 DataFrame
df_raw, df_clean = load_and_preprocess('DATE.csv')

# 补充打印每列的数据类型信息（如 float64、object 等）
print(f"\n列名及数据类型:")
print(df_raw.dtypes.to_string())

# ============================================================
# 第2步：描述性统计（基于原始数据）
# ============================================================
# 对风速(WINDSPEED)和功率(WINDPOWER)进行描述性统计
# describe() 生成：count、mean、std、min、25%、50%、75%、max
# 额外计算极差(range) = max - min
print("\n" + "=" * 60)
print("第2步：描述性统计")
print("=" * 60)

stats = df_raw[['WINDSPEED', 'WINDPOWER']].describe().T  # .T 转置，让特征成为行
stats['range'] = stats['max'] - stats['min']               # 添加极差列
print("\nWINDSPEED 和 WINDPOWER 的描述性统计:")
# 按指定顺序输出：均值、标准差、最小值、Q1、中位数、Q3、最大值、极差
print(stats[['mean', 'std', 'min', '25%', '50%', '75%', 'max', 'range']].to_string())

# ============================================================
# 第3步：绘制各属性波形图
# ============================================================
# 目的：直观展示五个气象/功率属性随时间的变化趋势
# 布局：5行1列垂直堆叠，共享X轴（时间轴）
print("\n========== 正在绘制各属性波形图... ==========")

# 创建子图：5行1列，总尺寸 14×12 英寸，sharex=True 共享X轴
fig, axes = plt.subplots(5, 1, figsize=(14, 12), sharex=True)

# 定义需要绘制的五个特征及其对应的中文标题和颜色
features = ['WINDSPEED', 'WINDPOWER', 'TEMPERATURE', 'HUMIDITY', 'PRESSURE']
titles   = ['风速 (m/s)', '有功功率 (kW)', '温度 (°C)', '湿度 (%)', '气压 (hPa)']
colors   = ['steelblue', 'firebrick', 'forestgreen', 'orange', 'purple']

# 遍历每个子图，绘制时间序列曲线
for ax, col, title, color in zip(axes, features, titles, colors):
    # 绘制折线图：X轴为时间，Y轴为特征值
    # linewidth=0.3 — 细线避免视觉遮挡（数据量大时）
    # alpha=0.7    — 半透明，重叠区域仍可见
    ax.plot(df_raw['DATATIME'], df_raw[col], color=color, linewidth=0.3, alpha=0.7)
    ax.set_ylabel(title)             # 设置Y轴标签
    ax.grid(True, alpha=0.3)         # 添加网格线，半透明
    if col == 'WINDPOWER':
        ax.set_ylim(bottom=0)        # 功率子图从0开始（功率不为负）

axes[-1].set_xlabel('DateTime')      # 仅最底部子图显示X轴标签
plt.tight_layout()                    # 自动调整子图间距
ensure_dir('RW1/waveforms.png')       # 确保输出目录存在
plt.savefig('RW1/waveforms.png', dpi=200)  # 保存为高分辨率 PNG
plt.close()                           # 关闭图形释放内存
print("已保存: waveforms.png")

# ============================================================
# 第4步：双Y轴时间序列图（风速 + 功率）
# ============================================================
# 目的：在同一图中对比风速和功率的时间变化模式
# 技术：使用 twinx() 创建共享X轴的双Y轴图
print("========== 正在绘制双Y轴时间序列图... ==========")

fig, ax1 = plt.subplots(figsize=(14, 5))

# 左Y轴：风速（蓝色）
ax1.plot(df_raw['DATATIME'], df_raw['WINDSPEED'],
         color='steelblue', linewidth=0.3, alpha=0.7, label='Wind Speed (m/s)')
ax1.set_xlabel('DateTime')
ax1.set_ylabel('Wind Speed (m/s)', color='steelblue')  # Y轴标签颜色与曲线一致
ax1.tick_params(axis='y', labelcolor='steelblue')       # 刻度标签颜色与曲线一致
ax1.grid(True, alpha=0.3)

# 右Y轴：功率（红色），通过 twinx() 创建
ax2 = ax1.twinx()  # 创建共享X轴的新Y轴
ax2.plot(df_raw['DATATIME'], df_raw['WINDPOWER'],
         color='firebrick', linewidth=0.3, alpha=0.7, label='Active Power (kW)')
ax2.set_ylabel('Active Power (kW)', color='firebrick')
ax2.tick_params(axis='y', labelcolor='firebrick')

plt.title('Wind Speed vs Active Power Over Time')
plt.tight_layout()
ensure_dir('RW1/timeseries_plot.png')
plt.savefig('RW1/timeseries_plot.png', dpi=200)
plt.close()
print("已保存: timeseries_plot.png")

# ============================================================
# 第5步：散点图（风速 vs 功率）
# ============================================================
# 目的：观察风速与功率之间的非线性关系（功率曲线）
# 散点图中的点大小 s=1，适用于大数据集
print("========== 正在绘制散点图... ==========")

fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(df_raw['WINDSPEED'], df_raw['WINDPOWER'],
           s=1,              # 点大小设为1（数据量大时避免重叠）
           alpha=0.5,        # 半透明，密集区域颜色更深
           c='steelblue')    # 统一蓝色
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
# ============== 第六步：任务一核心处理 ================
# ============================================================
print("\n" + "=" * 60)
print("任务一：数据预处理")
print("=" * 60)

# --- DBSCAN 聚类去噪 ---
# 在风速-功率二维空间进行密度聚类，自动识别并剔除噪声点
# save_plot_path 参数指定去噪对比图的保存位置
df_denoised = dbscan_denoise(df_clean, save_plot_path='RW1/scatter_denoised.png')

# --- Min-Max 归一化 ---
# 将所有数值特征缩放到 [0, 1] 区间，消除量纲影响
# 返回值：(归一化后的DataFrame, MinMaxScaler对象)
df_normalized, _ = minmax_normalize(df_denoised)

# --- 保存归一化数据集 ---
# 供 Task3 独立读取使用，不包含 cluster 列（聚类标签由 Task3 添加）
df_normalized.to_csv('data_normalized.csv', index=False)
print("\n已保存: data_normalized.csv")

# --- 归一化后散点图（验证归一化效果） ---
fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(df_normalized['WINDSPEED'], df_normalized['WINDPOWER'],
           s=1, alpha=0.5, c='steelblue')
ax.set_xlabel('Wind Speed (normalized)')  # 注意轴标签标注了 normalized
ax.set_ylabel('Active Power (normalized)')
ax.set_title('Scatter Plot (After Normalization)')
ax.grid(True, alpha=0.3)
plt.tight_layout()
ensure_dir('RW1/scatter_normalized.png')
plt.savefig('RW1/scatter_normalized.png', dpi=200)
plt.close()
print("已保存: scatter_normalized.png")

# --- 任务一完成汇总 ---
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


# ===========================================================================
# 第三部分：任务二 - 可视化与相关性分析
# ===========================================================================

print("\n" + "=" * 60)
print("任务二：可视化与相关性分析")
print("=" * 60)

# ============================================================
# 2.1 统计特征表
# ============================================================
# 对六维特征（风速、风向、温度、湿度、气压、功率）计算描述性统计
print("\n--------- 2.1 统计特征表 ---------")

# 定义六维特征列名
feature_cols = ['WINDSPEED', 'WINDDIRECTION', 'TEMPERATURE', 'HUMIDITY', 'PRESSURE', 'WINDPOWER']

# describe() 生成 count/mean/std/min/25%/50%/75%/max，转置后每行是一个特征
stat_table = df_denoised[feature_cols].describe().T
stat_table['range'] = stat_table['max'] - stat_table['min']  # 计算极差
# 选取需要展示的列
stat_table = stat_table[['mean', 'std', 'min', '25%', '50%', '75%', 'max', 'range']]
# 将列名改为中文，便于阅读
stat_table.columns = ['均值', '标准差', '最小值', 'Q1(25%)', 'Q2(中位数)', 'Q3(75%)', '最大值', '极差']
print("\n六维特征统计特征表:")
print(stat_table.to_string())
# 保存为 CSV 文件，float_format='%.4f' 控制小数位数
stat_table.to_csv('statistical_features.csv', float_format='%.4f')
print("\n已保存: statistical_features.csv")

# ============================================================
# 2.2 六维特征散点图矩阵
# ============================================================
# 目的：在一个图中展示所有特征两两之间的散点图关系，
#       以及对角线上的频率分布直方图
# 矩阵大小：6×6 = 36个子图
print("\n--------- 2.2 六维特征散点图矩阵 ---------")

# 特征列名（英文）和对应的中文显示标签
feature_names = ['WINDSPEED', 'WINDDIRECTION', 'TEMPERATURE', 'HUMIDITY', 'PRESSURE', 'WINDPOWER']
label_names   = ['风速', '风向', '温度', '湿度', '压强', '功率']
n = len(feature_names)  # 特征数量 = 6

# 创建 n×n 的子图矩阵
fig, axes = plt.subplots(n, n, figsize=(16, 16))

# 遍历所有子图位置 (i, j)
for i in range(n):      # 行索引
    for j in range(n):  # 列索引
        ax = axes[i, j]

        if i == j:
            # === 对角线：绘制频率分布直方图 ===
            # bins=50：50个柱子，edgecolor='white'：白边框，alpha=0.7：半透明
            ax.hist(df_denoised[feature_names[i]], bins=50,
                    color='steelblue', edgecolor='white', alpha=0.7)
            # 对角线不显示轴标签（避免拥挤）
            ax.set_xlabel('')
            ax.set_ylabel('')
        else:
            # === 非对角线：绘制散点图 ===
            # X轴 = 第j个特征，Y轴 = 第i个特征
            # s=0.5：极小点，alpha=0.3：高透明（大数据集避免过度绘制）
            ax.scatter(df_denoised[feature_names[j]], df_denoised[feature_names[i]],
                       s=0.5, alpha=0.3, c='steelblue')

        # --- 设置轴标签 ---
        # 仅最底行显示X轴标签
        if i == n - 1:
            ax.set_xlabel(label_names[j], fontsize=8)
        # 仅最左列显示Y轴标签
        if j == 0:
            ax.set_ylabel(label_names[i], fontsize=8)
        # 设置刻度标签字体大小
        ax.tick_params(axis='both', labelsize=6)

plt.suptitle('六维特征散点图矩阵', fontsize=14)
plt.tight_layout()
ensure_dir('RW2/scatter_matrix.png')
plt.savefig('RW2/scatter_matrix.png', dpi=150)
plt.close()
print("已保存: scatter_matrix.png")

# ============================================================
# 2.3 风向玫瑰图（Wind Rose）
# ============================================================
# 目的：展示风向的频率分布
# 技术：将风向角度按15度分箱，统计各方向出现频率，
#       用极坐标柱状图可视化
print("\n--------- 2.3 风向玫瑰图 ---------")

# --- 步骤1：按15度分箱统计频率 ---
wind_dir = df_denoised['WINDDIRECTION'].values  # 提取风向列（0-360度）

# 创建分箱边界：0, 15, 30, ..., 360（共25个边界，24个区间）
bins = np.arange(0, 361, 15)
labels_bins = np.arange(0, 360, 15)  # 区间标签：0, 15, 30, ..., 345

# 将每个风向值分配到对应的分箱
# np.digitize 返回每个值所属的区间索引（1-based），-1转为0-based
digits = np.digitize(wind_dir, bins) - 1
# 统计每个分箱中的样本数
counts = np.bincount(digits[digits >= 0], minlength=len(labels_bins))

# --- 步骤2：转换为极坐标 ---
# 角度从度数转为弧度
angles = np.radians(np.arange(0, 360, 15))
width = np.radians(15)  # 每个柱子的宽度 = 15度

# --- 步骤3：绘制极坐标柱状图 ---
fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={'projection': 'polar'})

# 在极坐标中绘制柱状图
bars = ax.bar(angles, counts, width=width, bottom=0.0,
              color='steelblue', edgecolor='white', alpha=0.7)

# --- 极坐标方向设置 ---
# set_theta_direction(-1): 顺时针方向（默认逆时针，-1反转）
# set_theta_offset(np.pi/2): 0度指向正上方（北方），而非右侧
ax.set_theta_direction(-1)
ax.set_theta_offset(np.pi / 2)

ax.set_title('风向玫瑰图 (15° 分箱)', fontsize=14, pad=20)
plt.tight_layout()
ensure_dir('RW2/wind_rose.png')
plt.savefig('RW2/wind_rose.png', dpi=150)
plt.close()
print("已保存: wind_rose.png")

# ============================================================
# 2.4 相关性分析与特征筛选
# ============================================================
# 目的：
#   1) 计算六维特征之间的皮尔逊相关系数矩阵
#   2) 筛选与 WINDPOWER 显著相关的特征（用于下游建模）
#   3) 绘制热力图直观展示相关性
#   4) 保存筛选结果供 Task3/Task4 使用
print("\n--------- 2.4 相关性分析与特征筛选 ---------")

# 调用公共模块计算相关性矩阵和筛选特征
corr_matrix, power_corr_sorted, selected_features = compute_correlation(df_denoised)

# 打印完整的相关系数矩阵
print("\n皮尔逊相关系数矩阵:")
print(corr_matrix.to_string())

# --- 绘制相关性热力图 ---
# 使用 imshow() 替代 seaborn.heatmap()，避免额外依赖
fig, ax = plt.subplots(figsize=(8, 7))

# imshow 绘制矩阵图像
# cmap='RdBu_r'：红蓝配色方案（红色=正相关，蓝色=负相关，_r 表示反转）
# vmin=-1, vmax=1：相关系数范围
im = ax.imshow(corr_matrix.values, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')

# 设置坐标轴刻度和标签
ax.set_xticks(range(len(feature_cols)))
ax.set_yticks(range(len(feature_cols)))
ax.set_xticklabels(label_names, fontsize=10)
ax.set_yticklabels(label_names, fontsize=10)

# --- 在每个格子中添加数值标签 ---
for i in range(len(feature_cols)):
    for j in range(len(feature_cols)):
        val = corr_matrix.values[i, j]
        # 根据相关系数绝对值选择文字颜色：强相关用白色，弱相关用黑色
        color = 'white' if abs(val) > 0.5 else 'black'
        ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                fontsize=8, color=color)

plt.title('皮尔逊相关系数热力图', fontsize=14)
plt.colorbar(im, shrink=0.8)  # 添加颜色条（shrink 缩小为80%避免过高）
plt.tight_layout()
ensure_dir('RW2/correlation_heatmap.png')
plt.savefig('RW2/correlation_heatmap.png', dpi=150)
plt.close()
print("已保存: correlation_heatmap.png")

# --- 保存筛选后的特征列表 ---
# 写入 selected_features.json，确保下游 Task3/Task4 使用完全一致的特征
save_selected_features(selected_features)

# --- 任务二完成汇总 ---
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
