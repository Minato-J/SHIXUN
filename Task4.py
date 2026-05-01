"""
风电场风电功率预测
================================
任务目标：建立风电功率预测模型，对每一类数据分别训练，评估不同模型的预测性能，对比不同模型的预测效果。
采用方法：支持向量回归(SVR)、BP神经网络、LSTM神经网络
评价指标：MAE、RMSE、R²

依赖: 需先运行 Task3.py 生成 data_with_clusters.csv
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import warnings
warnings.filterwarnings('ignore')

# 导入公共模块
from common import (
    ensure_dir, load_selected_features, create_sequences,
)

# ============================================================
# 常量配置
# ============================================================
SEQ_LENGTH = 12           # LSTM 序列长度（~3 小时，15 分钟采样间隔）
LSTM_HIDDEN = 64          # LSTM 隐藏单元数
LSTM_DROPOUT = 0.2        # Dropout 比例
LSTM_EPOCHS = 100         # LSTM 训练轮数
LSTM_BATCH_SIZE = 32      # 批大小
LSTM_LR = 0.001           # 学习率
LSTM_PATIENCE = 10        # 早停耐心值
TEST_SIZE = 0.2           # 测试集比例
PLOT_SAMPLES = 300        # 预测曲线图中绘制的测试样本数（避免过于密集）


# ============================================================
# LSTM 模型定义 (PyTorch)
# ============================================================
class LSTMPredictor(nn.Module):
    """
    LSTM 风电功率预测模型
    结构: LSTM(64) → Dropout(0.2) → Linear(64, 1)
    输入: (batch, seq_len, n_features)
    输出: (batch,) — 预测功率值
    """
    def __init__(self, input_size, hidden_size=LSTM_HIDDEN, dropout=LSTM_DROPOUT):
        super(LSTMPredictor, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)          # (batch, seq_len, hidden)
        out = out[:, -1, :]            # 取最后一个时间步的输出
        out = self.dropout(out)
        out = self.fc(out)             # (batch, 1)
        return out.squeeze(-1)         # (batch,)


# ============================================================
# LSTM 训练函数
# ============================================================
def train_lstm(model, train_loader, X_val_tensor, y_val_tensor, epochs, lr, patience, device):
    """训练 LSTM 模型，含早停机制"""
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    best_val_loss = float('inf')
    patience_counter = 0
    best_state = None
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            pred = model(X_batch)
            loss = criterion(pred, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # 验证
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val_tensor)
            val_loss = criterion(val_pred, y_val_tensor).item()
        
        if (epoch + 1) % 20 == 0:
            print(f"    LSTM Epoch {epoch+1}/{epochs} — train_loss: {train_loss/len(train_loader):.6f}, val_loss: {val_loss:.6f}")
        
        # 早停
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"    LSTM 早停于 Epoch {epoch+1} (最佳 val_loss: {best_val_loss:.6f})")
                break
    
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


# ============================================================
# 主程序
# ============================================================
print("=" * 60)
print("任务4：风电场风电功率预测")
print("=" * 60)

# 加载含聚类标签的数据集
df = pd.read_csv('data_with_clusters.csv')
df['DATATIME'] = pd.to_datetime(df['DATATIME'])
print(f"加载 data_with_clusters.csv: {df.shape[0]} 行 × {df.shape[1]} 列")

cluster_labels = df['cluster'].values
n_clusters = len(np.unique(cluster_labels))
print(f"聚类数: {n_clusters}")

print(f"\n聚类结果统计:")
for i in range(n_clusters):
    count = sum(cluster_labels == i)
    percentage = count / len(cluster_labels) * 100
    print(f"  类别 {i}: {count} 个样本 ({percentage:.2f}%)")

# ============================================================
# 1. 数据准备：按聚类结果划分数据，按时间顺序划分训练/测试集
# ============================================================
print("\n1. 数据准备：按聚类结果按时序划分数据...")

# 加载统一特征列表
try:
    selected_features = load_selected_features()
    print(f"模型输入特征（来自 selected_features.json）: {selected_features}")
except FileNotFoundError:
    from common import compute_correlation
    _, _, selected_features = compute_correlation(df)
    print(f"模型输入特征（基于当前数据计算）: {selected_features}")

# 按类别分别准备数据（按时序排序后划分）
clusters_data = {}
for cluster_id in range(n_clusters):
    mask = cluster_labels == cluster_id
    cluster_df = df[mask].sort_values('DATATIME').reset_index(drop=True)
    
    X = cluster_df[selected_features].values.astype(np.float32)
    y = cluster_df['WINDPOWER'].values.astype(np.float32)
    datatimes = cluster_df['DATATIME'].values
    
    n_samples = len(X)
    split_idx = int(n_samples * (1 - TEST_SIZE))
    
    clusters_data[cluster_id] = {
        'X_train': X[:split_idx],
        'X_test': X[split_idx:],
        'y_train': y[:split_idx],
        'y_test': y[split_idx:],
        'time_train': datatimes[:split_idx],
        'time_test': datatimes[split_idx:],
        'n_total': n_samples,
        'n_train': split_idx,
        'n_test': n_samples - split_idx,
    }
    print(f"  类别 {cluster_id}: 总 {n_samples} → 训练 {split_idx} / 测试 {n_samples - split_idx}")

# 2. 初始化结果存储
results = {}
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\nPyTorch 设备: {device}")

print("\n2. 开始模型训练与预测...")

for cluster_id in range(n_clusters):
    data = clusters_data[cluster_id]
    X_train_raw = data['X_train']
    X_test_raw = data['X_test']
    y_train_raw = data['y_train']
    y_test_raw = data['y_test']
    n_train, n_test = data['n_train'], data['n_test']
    
    print(f"\n--- 聚类 {cluster_id} (训练: {n_train}, 测试: {n_test}) ---")
    
    results[cluster_id] = {}
    
    # ----- 2.0 准备 LSTM 序列数据（提前准备，以便对齐评估）-----
    # 训练序列
    X_train_seq, y_train_seq = create_sequences(X_train_raw, y_train_raw, SEQ_LENGTH)
    # 测试序列（前缀 seq_length 个训练点以构建第一个测试序列）
    X_test_prefix = np.vstack([X_train_raw[-SEQ_LENGTH:], X_test_raw])
    y_test_prefix = np.hstack([y_train_raw[-SEQ_LENGTH:], y_test_raw])
    X_test_seq, y_test_seq = create_sequences(X_test_prefix, y_test_prefix, SEQ_LENGTH)
    
    print(f"  LSTM 序列: 训练 {X_train_seq.shape[0]} 条, 测试 {X_test_seq.shape[0]} 条 (seq_len={SEQ_LENGTH})")
    
    # ============================================================
    # 2.1 支持向量回归 (SVR)
    # ============================================================
    print(f"  2.1 训练 SVR 模型...")
    svr_model = SVR(kernel='rbf', C=1.0, epsilon=0.1)
    svr_model.fit(X_train_raw, y_train_raw)
    y_pred_svr = svr_model.predict(X_test_raw)
    
    svr_mae = mean_absolute_error(y_test_raw, y_pred_svr)
    svr_rmse = np.sqrt(mean_squared_error(y_test_raw, y_pred_svr))
    svr_r2 = r2_score(y_test_raw, y_pred_svr)
    
    results[cluster_id]['SVR'] = {
        'model': svr_model,
        'predictions': y_pred_svr,
        'actual': y_test_raw,
        'mae': svr_mae,
        'rmse': svr_rmse,
        'r2': svr_r2,
        'time_test': data['time_test'],
    }
    print(f"    MAE: {svr_mae:.4f}, RMSE: {svr_rmse:.4f}, R²: {svr_r2:.4f}")
    
    # ============================================================
    # 2.2 BP 神经网络
    # ============================================================
    print(f"  2.2 训练 BP 神经网络...")
    bp_model = MLPRegressor(
        hidden_layer_sizes=(32, 32),
        activation='relu',
        solver='adam',
        learning_rate_init=0.001,
        max_iter=1000,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=10,
        random_state=42
    )
    bp_model.fit(X_train_raw, y_train_raw)
    y_pred_bp = bp_model.predict(X_test_raw)
    
    bp_mae = mean_absolute_error(y_test_raw, y_pred_bp)
    bp_rmse = np.sqrt(mean_squared_error(y_test_raw, y_pred_bp))
    bp_r2 = r2_score(y_test_raw, y_pred_bp)
    
    results[cluster_id]['BP'] = {
        'model': bp_model,
        'predictions': y_pred_bp,
        'actual': y_test_raw,
        'mae': bp_mae,
        'rmse': bp_rmse,
        'r2': bp_r2,
        'time_test': data['time_test'],
    }
    print(f"    MAE: {bp_mae:.4f}, RMSE: {bp_rmse:.4f}, R²: {bp_r2:.4f}")
    
    # ============================================================
    # 2.3 LSTM 神经网络
    # ============================================================
    print(f"  2.3 训练 LSTM 神经网络...")
    
    # 转换为 PyTorch 张量（取训练集的 10% 作为验证集用于早停）
    n_val = max(1, int(len(X_train_seq) * 0.1))
    X_train_lstm = torch.tensor(X_train_seq[:-n_val], dtype=torch.float32)
    y_train_lstm = torch.tensor(y_train_seq[:-n_val], dtype=torch.float32)
    X_val_lstm = torch.tensor(X_train_seq[-n_val:], dtype=torch.float32).to(device)
    y_val_lstm = torch.tensor(y_train_seq[-n_val:], dtype=torch.float32).to(device)
    
    X_test_lstm = torch.tensor(X_test_seq, dtype=torch.float32).to(device)
    
    train_dataset = TensorDataset(X_train_lstm, y_train_lstm)
    train_loader = DataLoader(train_dataset, batch_size=LSTM_BATCH_SIZE, shuffle=True)
    
    lstm_model = LSTMPredictor(
        input_size=X_train_seq.shape[2],
        hidden_size=LSTM_HIDDEN,
        dropout=LSTM_DROPOUT
    ).to(device)
    
    lstm_model = train_lstm(
        lstm_model, train_loader, X_val_lstm, y_val_lstm,
        epochs=LSTM_EPOCHS, lr=LSTM_LR, patience=LSTM_PATIENCE, device=device
    )
    
    # LSTM 预测
    lstm_model.eval()
    with torch.no_grad():
        y_pred_lstm_tensor = lstm_model(X_test_lstm)
    y_pred_lstm = y_pred_lstm_tensor.cpu().numpy()
    
    lstm_mae = mean_absolute_error(y_test_seq, y_pred_lstm)
    lstm_rmse = np.sqrt(mean_squared_error(y_test_seq, y_pred_lstm))
    lstm_r2 = r2_score(y_test_seq, y_pred_lstm)
    
    results[cluster_id]['LSTM'] = {
        'model': lstm_model,
        'predictions': y_pred_lstm,
        'actual': y_test_seq,
        'mae': lstm_mae,
        'rmse': lstm_rmse,
        'r2': lstm_r2,
        'time_test': data['time_test'][SEQ_LENGTH:],  # LSTM 测试集从第 seq_length 个时间点开始
    }
    print(f"    MAE: {lstm_mae:.4f}, RMSE: {lstm_rmse:.4f}, R²: {lstm_r2:.4f}")

# ============================================================
# 3. 模型评估与对比
# ============================================================
print(f"\n3. 模型评估与对比")
print("=" * 60)

# 创建汇总表格
summary_results = []
for cluster_id in results:
    for model_name in results[cluster_id]:
        perf = results[cluster_id][model_name]
        summary_results.append({
            'Cluster': cluster_id,
            'Model': model_name,
            'MAE': perf['mae'],
            'RMSE': perf['rmse'],
            'R²': perf['r2']
        })

summary_df = pd.DataFrame(summary_results)
print("各模型性能指标汇总:")
print(summary_df.round(4).to_string(index=False))

# 确保输出目录存在
ensure_dir('RW4/task4_model_comparison.png')

# 绘制模型性能对比柱状图
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for i, metric in enumerate(['MAE', 'RMSE', 'R²']):
    pivot_data = summary_df.pivot(index='Cluster', columns='Model', values=metric)
    if pivot_data is not None and not pivot_data.empty:
        pivot_data.plot(kind='bar', ax=axes[i])
        axes[i].set_title(f'{metric} 对比')
        axes[i].set_ylabel(metric)
        axes[i].legend(loc='best')
        axes[i].tick_params(axis='x', rotation=0)
        axes[i].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('RW4/task4_model_comparison.png', dpi=200, bbox_inches='tight')
plt.close()
print("已保存: RW4/task4_model_comparison.png")

# 绘制预测值 vs 真实值时间曲线图（每个聚类和模型）
print("\n绘制预测曲线图...")
for cluster_id in results:
    for model_name in results[cluster_id]:
        perf = results[cluster_id][model_name]
        actual = perf['actual']
        pred = perf['predictions']
        times = perf['time_test']
        
        # 选择最后 PLOT_SAMPLES 个测试点绘制，使曲线可读
        n_plot = min(PLOT_SAMPLES, len(actual))
        idx_slice = slice(-n_plot, None)
        
        out_path = f'RW4/task4_prediction_{cluster_id}_{model_name}.png'
        ensure_dir(out_path)
        
        plt.figure(figsize=(14, 6))
        plt.plot(times[idx_slice], actual[idx_slice], label='真实值', alpha=0.8, linewidth=1.0)
        plt.plot(times[idx_slice], pred[idx_slice], label='预测值', alpha=0.8, linewidth=1.0)
        plt.xlabel('时间')
        plt.ylabel('功率 (归一化)')
        plt.title(f'聚类 {cluster_id} — {model_name} 预测效果 (MAE={perf["mae"]:.4f}, RMSE={perf["rmse"]:.4f}, R²={perf["r2"]:.4f})')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=30)
        plt.tight_layout()
        plt.savefig(out_path, dpi=200, bbox_inches='tight')
        plt.close()
        print(f"已保存: {out_path}")

# ============================================================
# 4. 总体性能汇总
# ============================================================
print(f"\n4. 总体性能汇总")
print("=" * 60)

for cluster_id in results:
    print(f"\n聚类 {cluster_id} 的模型性能:")
    for model_name in results[cluster_id]:
        perf = results[cluster_id][model_name]
        print(f"  {model_name}: MAE={perf['mae']:.4f}, RMSE={perf['rmse']:.4f}, R²={perf['r2']:.4f}")

# ============================================================
# 5. 跨聚类模型对比分析
# ============================================================
print(f"\n5. 跨聚类模型对比分析")
print("=" * 60)

for model_name in ['SVR', 'BP', 'LSTM']:
    print(f"\n{model_name} 模型跨聚类性能:")
    cluster_mae, cluster_rmse, cluster_r2 = [], [], []
    
    for cluster_id in results:
        if model_name in results[cluster_id]:
            perf = results[cluster_id][model_name]
            cluster_mae.append(perf['mae'])
            cluster_rmse.append(perf['rmse'])
            cluster_r2.append(perf['r2'])
            print(f"  聚类 {cluster_id}: MAE={perf['mae']:.4f}, RMSE={perf['rmse']:.4f}, R²={perf['r2']:.4f}")
    
    if cluster_mae:
        print(f"  平均: MAE={np.mean(cluster_mae):.4f}, RMSE={np.mean(cluster_rmse):.4f}, R²={np.mean(cluster_r2):.4f}")

# ============================================================
# 6. 最佳模型标注
# ============================================================
print(f"\n6. 最佳模型")
print("=" * 60)
for cluster_id in results:
    best_model = None
    best_r2 = -float('inf')
    for model_name in results[cluster_id]:
        if results[cluster_id][model_name]['r2'] > best_r2:
            best_r2 = results[cluster_id][model_name]['r2']
            best_model = model_name
    print(f"  聚类 {cluster_id}: {best_model} (R²={best_r2:.4f})")

overall_best = summary_df.loc[summary_df['R²'].idxmax()]
print(f"\n  全局最佳: 聚类 {int(overall_best['Cluster'])} - {overall_best['Model']} (R²={overall_best['R²']:.4f})")

print("\n" + "=" * 60)
print("任务4完成！")
print("=" * 60)
print("生成的文件:")
print("  1. RW4/task4_model_comparison.png     — 三模型 MAE/RMSE/R² 柱状对比图")
print("  2. RW4/task4_prediction_*_*.png      — 各聚类各模型预测-时间曲线图")
print("=" * 60)
