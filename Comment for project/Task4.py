"""
================================================================================
风电场风电功率预测（详细注释版）
================================================================================

【脚本功能概述】
本脚本实现任务四：风电功率预测建模。
  - 基于任务三的聚类结果（data_with_clusters.csv），对每一类数据分别训练预测模型
  - 对比三种回归模型的预测效果：支持向量回归(SVR)、BP神经网络、线性回归(Linear)
  - 评估指标：MAE（平均绝对误差）、RMSE（均方根误差）、R²（决定系数）

【算法原理】
  ┌─ SVR（支持向量回归）
  │   核函数：RBF（径向基函数），将数据映射到高维空间进行线性回归
  │   核心参数：C=1.0（正则化系数），epsilon=0.1（不敏感区间宽度）
  │   优点：对小样本数据鲁棒性强，适合非线性关系建模
  │
  ├─ BP神经网络（MLPRegressor）
  │   结构：2个隐藏层，每层32个神经元，ReLU激活函数
  │   优化器：Adam（自适应矩估计），学习率=0.001，最大迭代=500
  │   优点：强大的非线性拟合能力，能捕捉复杂的风速-功率关系
  │
  └─ 线性回归（LinearRegression）
      模型：y = w₁x₁ + w₂x₂ + ... + wₙxₙ + b
      优点：模型简单、可解释性强、计算速度快，用作性能基准对比

【依赖说明】
  - 需要先运行 Task3.py 生成 data_with_clusters.csv（含 cluster 列）
  - 若 data_with_clusters.csv 缺失，自动回退到 data_normalized.csv 并重新聚类
  - 读取 selected_features.json 保持特征筛选一致性

【输出文件说明】
  - RW4(SOLO)/task4_model_comparison.png  : 三模型 MAE/RMSE/R² 柱状对比图
  - RW4(SOLO)/task4_prediction_{簇}_{模型}.png : 各聚类各模型的预测值 vs 真实值曲线
"""

# ===========================================================================
# 第一部分：导入依赖库
# ===========================================================================

import pandas as pd               # 数据处理：DataFrame、CSV 读写
import numpy as np                # 数值计算：数组运算、数学函数（sqrt、mean）
import matplotlib.pyplot as plt   # 数据可视化：绘制预测效果图、柱状对比图
from sklearn.model_selection import train_test_split  # 数据集划分：训练集/测试集 8:2
from sklearn.svm import SVR                # 支持向量回归模型
from sklearn.neural_network import MLPRegressor  # BP 神经网络（多层感知机回归器）
from sklearn.linear_model import LinearRegression  # 线性回归模型
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# 评估指标导入：
#   mean_absolute_error : MAE，预测值与真实值绝对差的均值
#   mean_squared_error  : MSE，预测值与真实值平方差的均值（取 sqrt 得 RMSE）
#   r2_score            : R²，决定系数，衡量模型对数据方差的解释程度（越接近1越好）
import warnings                     # 警告控制
warnings.filterwarnings('ignore')   # 忽略收敛警告等非关键信息，保持输出整洁

# --- 从公共模块导入共享函数 ---
# ensure_dir            : 确保输出文件的父目录存在（递归创建）
# kmeans_cluster        : K-means 聚类（含肘部法+轮廓系数法的 K 值选择）
# load_selected_features: 从 selected_features.json 读取特征筛选结果
from common import (
    ensure_dir, kmeans_cluster, load_selected_features,
)


# ===========================================================================
# 第二部分：主程序入口
# ===========================================================================

print("=" * 60)
print("任务4：风电场风电功率预测")
print("=" * 60)

# ============================================================
# 第1步：数据加载
# ============================================================
# 优先加载含聚类标签的数据集 data_with_clusters.csv
# 若文件不存在或缺少 cluster 列，则回退到 data_normalized.csv 并重新聚类
try:
    df = pd.read_csv('data_with_clusters.csv')
    print(f"加载 data_with_clusters.csv: {df.shape[0]} 行 × {df.shape[1]} 列")
    if 'cluster' not in df.columns:
        raise ValueError("data_with_clusters.csv 缺少 cluster 列")
except FileNotFoundError:
    print("未找到 data_with_clusters.csv，回退到 data_normalized.csv 重新聚类...")
    df = pd.read_csv('data_normalized.csv')
    print(f"加载 data_normalized.csv: {df.shape[0]} 行 × {df.shape[1]} 列")

# ============================================================
# 第2步：获取聚类标签
# ============================================================
# 优先使用已有 cluster 列；否则使用与 Task3 相同的特征和逻辑重新聚类

if 'cluster' in df.columns:
    print("检测到已有 cluster 列，直接使用。")
    cluster_labels = df['cluster'].values           # 转为 numpy 数组便于索引
    n_clusters = len(np.unique(cluster_labels))     # 统计聚类数量
    print(f"聚类数: {n_clusters}")
else:
    print("未检测到 cluster 列，使用与 task3 一致的逻辑重新聚类...")

    # --- 加载统一特征列表（来自 Task2 的皮尔逊相关性筛选结果） ---
    try:
        selected_features = load_selected_features()
        print(f"从 selected_features.json 读取筛选特征: {selected_features}")
    except FileNotFoundError:
        # selected_features.json 不存在时，基于当前数据重新计算
        from common import compute_correlation
        _, _, selected_features = compute_correlation(df)

    # 聚类特征 = 筛选特征 + WINDPOWER（目标变量也参与聚类，反映运行工况）
    # dict.fromkeys() 技巧：去重同时保持顺序
    cluster_features = list(dict.fromkeys(selected_features + ['WINDPOWER']))
    X_cluster = df[cluster_features].values

    # 调用公共模块执行 K-means 聚类（含肘部法+轮廓系数自动选 K）
    cluster_result = kmeans_cluster(X_cluster)
    cluster_labels = cluster_result['labels']
    n_clusters = cluster_result['optimal_k']

# 将聚类标签写回 DataFrame
df['cluster'] = cluster_labels

# --- 打印各类别的样本分布 ---
print(f"\n聚类结果统计:")
for i in range(n_clusters):
    count = sum(cluster_labels == i)
    percentage = count / len(cluster_labels) * 100
    print(f"  类别 {i}: {count} 个样本 ({percentage:.2f}%)")


# ============================================================
# 第3步：数据准备 — 按聚类结果划分训练/测试数据
# ============================================================
print("\n1. 数据准备：按聚类结果划分数据...")

# --- 加载统一特征列表作为模型输入 ---
# 注意：不包含 WINDPOWER（它是预测目标），只使用与 Task2 一致的筛选特征
try:
    selected_features = load_selected_features()
    print(f"模型输入特征（来自 selected_features.json）: {selected_features}")
except FileNotFoundError:
    from common import compute_correlation
    _, _, selected_features = compute_correlation(df)
    print(f"模型输入特征（基于当前数据计算）: {selected_features}")

# X: 特征矩阵（输入变量），y: 目标向量（输出变量 = 风电功率）
X_all = df[selected_features].values
y_all = df['WINDPOWER'].values
clusters = df['cluster'].values

# --- 按聚类类别分别提取数据 ---
# clusters_data 结构: {cluster_id: {'X': 特征矩阵, 'y': 目标向量, 'indices': 原始索引}}
clusters_data = {}
for cluster_id in range(n_clusters):
    mask = clusters == cluster_id            # 布尔掩码：筛选当前类别的样本
    clusters_data[cluster_id] = {
        'X': X_all[mask],
        'y': y_all[mask],
        'indices': np.where(mask)[0]          # 保存原始索引，便于追溯
    }
    print(f"  类别 {cluster_id}: {len(clusters_data[cluster_id]['X'])} 个样本")


# ============================================================
# 第4步：模型训练与预测（按聚类遍历）
# ============================================================
# results 结构:
#   {cluster_id: {model_name: {
#       'model': 训练好的模型对象,
#       'predictions': 测试集预测值,
#       'actual': 测试集真实值,
#       'mae': 平均绝对误差,
#       'rmse': 均方根误差,
#       'r2': 决定系数
#   }}}

# 初始化结果存储
results = {}

print("\n2. 开始模型训练与预测...")

# 遍历每个聚类进行建模
for cluster_id in range(n_clusters):
    print(f"\n--- 处理聚类 {cluster_id} (样本数: {len(clusters_data[cluster_id]['X'])}) ---")

    X_cluster = clusters_data[cluster_id]['X']  # 当前聚类的特征矩阵
    y_cluster = clusters_data[cluster_id]['y']  # 当前聚类的目标向量

    # 安全检查：样本数过少（<2）无法训练/测试划分，直接跳过
    if len(X_cluster) < 2:
        print(f"  聚类 {cluster_id} 样本数太少，跳过建模")
        continue

    # --- 划分训练集和测试集（8:2 比例，固定随机种子保证可复现） ---
    X_train, X_test, y_train, y_test = train_test_split(
        X_cluster, y_cluster, test_size=0.2, random_state=42
    )

    print(f"  训练集大小: {X_train.shape[0]}, 测试集大小: {X_test.shape[0]}")

    # 初始化该聚类的结果字典
    results[cluster_id] = {}

    # ============================================================
    # 模型1：支持向量回归 (SVR)
    # ============================================================
    # 原理：利用 RBF 核函数将数据映射到高维空间，在高维空间寻找
    #       最优超平面，使得大部分样本落在 epsilon 不敏感区间内
    print(f"  2.1 训练SVR模型...")
    svr_model = SVR(kernel='rbf',      # RBF（径向基函数）核，适合非线性关系
                    C=1.0,             # 正则化系数：越大对误差惩罚越重，可能过拟合
                    epsilon=0.1)       # 不敏感区间：预测值与真实值偏差 < epsilon 不计入损失
    svr_model.fit(X_train, y_train)    # 训练模型
    y_pred_svr = svr_model.predict(X_test)  # 在测试集上预测

    # 计算三项评估指标
    svr_mae = mean_absolute_error(y_test, y_pred_svr)          # MAE：误差绝对值的均值
    svr_rmse = np.sqrt(mean_squared_error(y_test, y_pred_svr)) # RMSE：MSE 的平方根，与原始量纲一致
    svr_r2 = r2_score(y_test, y_pred_svr)                       # R²：决定系数，越接近1越好

    # 存储结果
    results[cluster_id]['SVR'] = {
        'model': svr_model,
        'predictions': y_pred_svr,
        'actual': y_test,
        'mae': svr_mae,
        'rmse': svr_rmse,
        'r2': svr_r2
    }
    print(f"    MAE: {svr_mae:.4f}, RMSE: {svr_rmse:.4f}, R2: {svr_r2:.4f}")

    # ============================================================
    # 模型2：BP 神经网络（多层感知机回归器）
    # ============================================================
    # 原理：多层前馈神经网络，通过反向传播算法（Backpropagation）
    #       逐层更新权重，最小化预测误差
    # 结构：输入层 → 隐藏层1(32) → 隐藏层2(32) → 输出层(1)
    print(f"  2.2 训练BP神经网络模型...")
    bp_model = MLPRegressor(
        hidden_layer_sizes=(32, 32),     # 两个隐藏层，各32个神经元
        activation='relu',               # ReLU 激活函数：f(x)=max(0,x)，缓解梯度消失
        solver='adam',                   # Adam 优化器：自适应学习率，收敛快
        learning_rate_init=0.001,        # 初始学习率
        max_iter=500,                    # 最大迭代次数（epochs）
        random_state=42                  # 固定随机种子，保证结果可复现
    )
    bp_model.fit(X_train, y_train)
    y_pred_bp = bp_model.predict(X_test)

    # 计算评估指标
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

    # ============================================================
    # 模型3：线性回归
    # ============================================================
    # 原理：最小二乘法拟合线性关系 y = w₁x₁ + w₂x₂ + ... + wₙxₙ + b
    # 优点：计算速度快，可解释性强；作为复杂模型的性能基准
    print(f"  2.3 训练Linear回归模型...")
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


# ============================================================
# 第5步：模型评估与可视化对比
# ============================================================
print(f"\n3. 模型评估与对比")
print("-" * 60)

# --- 创建汇总结果表格 ---
# 将嵌套的 results 字典扁平化为 DataFrame，便于展示和绘图
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
    print("各模型性能指标汇总:")
    print(summary_df.round(4))  # 保留 4 位小数

    # --- 确保输出目录存在 ---
    ensure_dir('RW4(SOLO)/task4_model_comparison.png')

    # --- 绘制模型性能对比柱状图（1行3列） ---
    # 三个子图分别展示 MAE、RMSE、R²，横轴为聚类类别，每组柱子为不同模型
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for i, metric in enumerate(['MAE', 'RMSE', 'R2']):
        # 数据透视：行=聚类ID，列=模型名，值=指标值
        pivot_data = summary_df.pivot(index='Cluster', columns='Model', values=metric)
        if pivot_data is not None and not pivot_data.empty:
            pivot_data.plot(kind='bar', ax=axes[i])      # 柱状图
            axes[i].set_title(f'{metric} 对比')
            axes[i].set_ylabel(metric)
            axes[i].legend()
            axes[i].tick_params(axis='x', rotation=45)    # X轴标签旋转45°
            axes[i].grid(True, alpha=0.3)                 # 半透明网格线

    plt.tight_layout()
    plt.savefig('RW4(SOLO)/task4_model_comparison.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("已保存: task4_model_comparison.png")

    # --- 绘制预测值 vs 真实值曲线图（每个聚类 × 每个模型） ---
    # 横轴为测试样本索引，纵轴为归一化功率值
    # 蓝色线=真实值，橙色线=模型预测值
    for cluster_id in results:
        for model_name in results[cluster_id]:
            if results[cluster_id][model_name] is not None:
                actual = results[cluster_id][model_name]['actual']        # 真实值数组
                pred = results[cluster_id][model_name]['predictions']     # 预测值数组

                out_path = f'RW4(SOLO)/task4_prediction_{cluster_id}_{model_name}.png'
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
# 第6步：总体性能汇总
# ============================================================
print(f"\n4. 总体性能汇总")
print("-" * 40)

# 按聚类遍历，输出每个类下各模型的完整指标
for cluster_id in results:
    print(f"\n聚类 {cluster_id} 的模型性能:")
    for model_name in results[cluster_id]:
        if results[cluster_id][model_name] is not None:
            perf = results[cluster_id][model_name]
            print(f"  {model_name}: MAE={perf['mae']:.4f}, RMSE={perf['rmse']:.4f}, R2={perf['r2']:.4f}")


# ============================================================
# 第7步：跨聚类模型对比分析
# ============================================================
print(f"\n5. 模型对比分析")
print("-" * 40)

# 对每种模型，汇总其在不同聚类上的表现，计算平均指标
for model_name in ['SVR', 'BP', 'Linear']:
    # 检查该模型是否在所有聚类中都存在且非空
    model_exists_in_all_clusters = True
    for cluster_id in results:
        if model_name not in results[cluster_id] or results[cluster_id][model_name] is None:
            model_exists_in_all_clusters = False
            break

    if model_exists_in_all_clusters:
        print(f"\n{model_name} 模型跨聚类性能:")
        cluster_maes = []    # 收集各聚类的 MAE
        cluster_rmses = []   # 收集各聚类的 RMSE
        cluster_r2s = []     # 收集各聚类的 R²

        for cluster_id in results:
            perf = results[cluster_id][model_name]
            cluster_maes.append(perf['mae'])
            cluster_rmses.append(perf['rmse'])
            cluster_r2s.append(perf['r2'])
            print(f"  聚类 {cluster_id}: MAE={perf['mae']:.4f}, RMSE={perf['rmse']:.4f}, R2={perf['r2']:.4f}")

        # 计算跨聚类的平均性能指标
        avg_mae = np.mean(cluster_maes)
        avg_rmse = np.mean(cluster_rmses)
        avg_r2 = np.mean(cluster_r2s)
        print(f"  平均性能: MAE={avg_mae:.4f}, RMSE={avg_rmse:.4f}, R2={avg_r2:.4f}")


# ============================================================
# 结束
# ============================================================
print("\n" + "=" * 60)
print("任务4完成！")
print("=" * 60)
print("生成的文件:")
print("  1. task4_model_comparison.png         — 模型性能对比图")
print("  2. task4_prediction_*_*.png          — 各聚类各模型预测效果图")
print("=" * 60)
