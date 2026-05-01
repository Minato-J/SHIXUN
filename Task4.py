"""
风电场风电功率预测
================================
任务目标：建立风电功率预测模型，对每一类数据分别训练，评估不同模型的预测性能，对比不同模型的预测效果。
采用方法：支持向量回归(SVR)、BP神经网络、线性回归(Linear)
评价指标：MAE、RMSE、R²

依赖: 需先运行 Task3.py 生成 data_with_clusters.csv
      若该文件缺失，将基于 data_normalized.csv 使用一致的逻辑重新聚类。
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# 导入公共模块
from common import (
    ensure_dir, kmeans_cluster, load_selected_features,
)

# ============================================================
# 主程序
# ============================================================
print("=" * 60)
print("任务4：风电场风电功率预测")
print("=" * 60)

# 优先加载含聚类标签的数据集，否则回退到归一化数据集并重新聚类
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
# 获取聚类标签：优先读取已有列，否则使用与 task3 一致的特征和逻辑重新聚类
# ============================================================
if 'cluster' in df.columns:
    print("检测到已有 cluster 列，直接使用。")
    cluster_labels = df['cluster'].values
    n_clusters = len(np.unique(cluster_labels))
    print(f"聚类数: {n_clusters}")
else:
    print("未检测到 cluster 列，使用与 task3 一致的逻辑重新聚类...")

    # 加载统一特征列表
    try:
        selected_features = load_selected_features()
        print(f"从 selected_features.json 读取筛选特征: {selected_features}")
    except FileNotFoundError:
        from common import compute_correlation
        _, _, selected_features = compute_correlation(df)

    # 聚类特征 = 筛选特征 + WINDPOWER
    cluster_features = list(dict.fromkeys(selected_features + ['WINDPOWER']))
    X_cluster = df[cluster_features].values

    cluster_result = kmeans_cluster(X_cluster)
    cluster_labels = cluster_result['labels']
    n_clusters = cluster_result['optimal_k']

df['cluster'] = cluster_labels

print(f"\n聚类结果统计:")
for i in range(n_clusters):
    count = sum(cluster_labels == i)
    percentage = count / len(cluster_labels) * 100
    print(f"  类别 {i}: {count} 个样本 ({percentage:.2f}%)")

# ============================================================
# 1. 数据准备：按聚类结果划分数据（使用与 task2 一致的特征）
# ============================================================
print("\n1. 数据准备：按聚类结果划分数据...")

# 加载统一特征列表作为模型输入（不包含 WINDPOWER，它是预测目标）
try:
    selected_features = load_selected_features()
    print(f"模型输入特征（来自 selected_features.json）: {selected_features}")
except FileNotFoundError:
    from common import compute_correlation
    _, _, selected_features = compute_correlation(df)
    print(f"模型输入特征（基于当前数据计算）: {selected_features}")

X_all = df[selected_features].values
y_all = df['WINDPOWER'].values
clusters = df['cluster'].values

# 按类别分别准备数据
clusters_data = {}
for cluster_id in range(n_clusters):
    mask = clusters == cluster_id
    clusters_data[cluster_id] = {
        'X': X_all[mask],
        'y': y_all[mask],
        'indices': np.where(mask)[0]
    }
    print(f"  类别 {cluster_id}: {len(clusters_data[cluster_id]['X'])} 个样本")

# 2. 初始化结果存储
results = {}

print("\n2. 开始模型训练与预测...")

# 遍历每个聚类进行建模
for cluster_id in range(n_clusters):
    print(f"\n--- 处理聚类 {cluster_id} (样本数: {len(clusters_data[cluster_id]['X'])}) ---")

    X_cluster = clusters_data[cluster_id]['X']
    y_cluster = clusters_data[cluster_id]['y']

    if len(X_cluster) < 2:  # 如果某个聚类样本太少，跳过
        print(f"  聚类 {cluster_id} 样本数太少，跳过建模")
        continue

    # 分割训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X_cluster, y_cluster, test_size=0.2, random_state=42
    )

    print(f"  训练集大小: {X_train.shape[0]}, 测试集大小: {X_test.shape[0]}")

    # 初始化该聚类的结果字典
    results[cluster_id] = {}

    # 2.1 支持向量回归 (SVR)
    print(f"  2.1 训练SVR模型...")
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

    # 2.2 BP神经网络
    print(f"  2.2 训练BP神经网络模型...")
    bp_model = MLPRegressor(
        hidden_layer_sizes=(32, 32),  # 2层，每层32个神经元
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

    # 2.3 线性回归模型
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
# 3. 模型评估与对比
# ============================================================
print(f"\n3. 模型评估与对比")
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
    print("各模型性能指标汇总:")
    print(summary_df.round(4))

    # 确保输出目录存在
    ensure_dir('RW4(SOLO)/task4_model_comparison.png')

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
    plt.savefig('RW4(SOLO)/task4_model_comparison.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("已保存: task4_model_comparison.png")

    # 绘制预测值vs真实值曲线图（每个聚类和模型）
    for cluster_id in results:
        for model_name in results[cluster_id]:
            if results[cluster_id][model_name] is not None:
                actual = results[cluster_id][model_name]['actual']
                pred = results[cluster_id][model_name]['predictions']

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
# 4. 总体性能汇总
# ============================================================
print(f"\n4. 总体性能汇总")
print("-" * 40)

for cluster_id in results:
    print(f"\n聚类 {cluster_id} 的模型性能:")
    for model_name in results[cluster_id]:
        if results[cluster_id][model_name] is not None:
            perf = results[cluster_id][model_name]
            print(f"  {model_name}: MAE={perf['mae']:.4f}, RMSE={perf['rmse']:.4f}, R2={perf['r2']:.4f}")

# ============================================================
# 5. 模型对比分析
# ============================================================
print(f"\n5. 模型对比分析")
print("-" * 40)

for model_name in ['SVR', 'BP', 'Linear']:
    model_exists_in_all_clusters = True
    for cluster_id in results:
        if model_name not in results[cluster_id] or results[cluster_id][model_name] is None:
            model_exists_in_all_clusters = False
            break

    if model_exists_in_all_clusters:
        print(f"\n{model_name} 模型跨聚类性能:")
        cluster_maes = []
        cluster_rmses = []
        cluster_r2s = []

        for cluster_id in results:
            perf = results[cluster_id][model_name]
            cluster_maes.append(perf['mae'])
            cluster_rmses.append(perf['rmse'])
            cluster_r2s.append(perf['r2'])
            print(f"  聚类 {cluster_id}: MAE={perf['mae']:.4f}, RMSE={perf['rmse']:.4f}, R2={perf['r2']:.4f}")

        avg_mae = np.mean(cluster_maes)
        avg_rmse = np.mean(cluster_rmses)
        avg_r2 = np.mean(cluster_r2s)
        print(f"  平均性能: MAE={avg_mae:.4f}, RMSE={avg_rmse:.4f}, R2={avg_r2:.4f}")

print("\n" + "=" * 60)
print("任务4完成！")
print("=" * 60)
print("生成的文件:")
print("  1. task4_model_comparison.png         — 模型性能对比图")
print("  2. task4_prediction_*_*.png          — 各聚类各模型预测效果图")
print("=" * 60)
