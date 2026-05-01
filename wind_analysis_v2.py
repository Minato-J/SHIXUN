"""
风电数据分析项目 - 新版文件管理系统（面向对象 + 时间戳输出目录）
================================
目录结构: task_outputs/TaskName/{timestamp}

修复说明:
  - 复用 common.py 公共模块消除代码重复
  - 移除未使用的 original_cwd / restore_original_dir 逻辑
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.decomposition import PCA
import warnings
from pathlib import Path
import os
from datetime import datetime

warnings.filterwarnings('ignore')

# 导入公共模块
from common import (
    ensure_dir, load_and_preprocess, dbscan_denoise, minmax_normalize,
    compute_correlation, kmeans_cluster,
    NUMERIC_COLS, CORRELATION_THRESHOLD,
)


class SimpleTaskFileManager:
    """
    简化版任务文件管理器
    目录结构: task_outputs/TaskName/{timestamp}
    """

    def __init__(self, base_dir="./task_outputs"):
        self.base_dir = Path(base_dir).resolve()
        self.base_dir.mkdir(exist_ok=True)
        self.current_task_dir = None

    def create_task_folder(self, task_name, use_timestamp=True):
        """
        为任务创建独立的文件夹
        目录结构: task_outputs/TaskName/{timestamp}
        """
        main_task_dir = self.base_dir / task_name
        main_task_dir.mkdir(exist_ok=True)

        if use_timestamp:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            task_dir = main_task_dir / timestamp
        else:
            task_dir = main_task_dir / "latest"

        task_dir.mkdir(exist_ok=True)

        self.current_task_dir = task_dir
        print(f"任务文件夹已创建: {task_dir}")

        return task_dir

    def get_current_task_folder(self):
        """获取当前任务文件夹路径"""
        return self.current_task_dir

    def switch_to_task_folder(self, task_name, use_timestamp=True):
        """
        创建任务文件夹并返回路径（不改变当前工作目录）
        """
        task_dir = self.create_task_folder(task_name, use_timestamp)
        return task_dir


class WindAnalysisNewStructure:
    """
    风电数据分析类，使用新的目录结构
    """

    def __init__(self, data_path=None):
        # 始终使用项目根目录下的 DATE.csv
        self.data_source = Path.cwd() / "DATE.csv"
        self.df = None
        self.file_manager = SimpleTaskFileManager()

    def task1_data_preprocessing(self):
        """任务1：数据预处理（使用公共模块）"""
        print("=" * 60)
        print("任务1：数据预处理")
        print("=" * 60)

        # 创建任务1的文件夹
        task_dir = self.file_manager.switch_to_task_folder("Task1")

        # 使用公共模块加载和预处理原始数据
        df_raw, df_clean = load_and_preprocess(str(self.data_source))

        # 描述性统计
        stats = df_raw[['WINDSPEED', 'WINDPOWER']].describe().T
        stats['range'] = stats['max'] - stats['min']
        print("\nWINDSPEED (风速, m/s) 和 WINDPOWER (有功功率, kW) 的描述性统计:")
        print(stats[['mean', 'std', 'min', '25%', '50%', '75%', 'max', 'range']].to_string())

        # DBSCAN 去噪（使用公共模块）
        df_denoised = dbscan_denoise(df_clean)

        # Min-Max 归一化（使用公共模块）
        df_normalized, _ = minmax_normalize(df_denoised)

        # 保存归一化后的数据到任务目录和根目录
        df_normalized.to_csv(str(task_dir / 'data_normalized.csv'), index=False)
        df_normalized.to_csv('data_normalized.csv', index=False)
        print("\n已保存: data_normalized.csv (归一化后的数据集)")

        # 绘制各属性波形图
        fig, axes = plt.subplots(5, 1, figsize=(14, 12), sharex=True)
        features = ['WINDSPEED', 'WINDPOWER', 'TEMPERATURE', 'HUMIDITY', 'PRESSURE']
        titles = ['风速 (m/s)', '有功功率 (kW)', '温度 (°C)', '湿度 (%)', '气压 (hPa)']
        colors = ['steelblue', 'firebrick', 'forestgreen', 'orange', 'purple']

        for ax, col, title, color in zip(axes, features, titles, colors):
            ax.plot(df_normalized['DATATIME'], df_normalized[col], color=color, linewidth=0.3, alpha=0.7)
            ax.set_ylabel(title)
            ax.grid(True, alpha=0.3)
            if col == 'WINDPOWER':
                ax.set_ylim(bottom=0)

        axes[-1].set_xlabel('DateTime')
        plt.tight_layout()
        out_path = str(task_dir / 'waveforms.png')
        plt.savefig(out_path, dpi=200)
        plt.close()
        print(f"已保存: {out_path}")

        # 保存预处理后的数据供后续任务使用
        self.df = df_normalized

        return df_normalized

    def task2_visualization_correlation(self):
        """任务2：可视化与相关性分析"""
        if self.df is None:
            raise ValueError("请先运行任务1的数据预处理")

        print("\n" + "=" * 60)
        print("任务2：可视化与相关性分析")
        print("=" * 60)

        task_dir = self.file_manager.switch_to_task_folder("Task2")

        # 相关性分析与特征筛选（使用公共模块）
        correlation_matrix, power_corr_sorted, selected_features = compute_correlation(self.df)

        # 绘制相关性热力图
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                    square=True, fmt='.3f', cbar_kws={'shrink': 0.8})
        plt.title('特征相关性热力图')
        plt.tight_layout()
        out_path = str(task_dir / 'correlation_heatmap.png')
        plt.savefig(out_path, dpi=200, bbox_inches='tight')
        plt.close()
        print(f"已保存: {out_path}")

        # 风向玫瑰图
        def wind_rose_plot(df, bins=24):
            wind_dir_deg = df['WINDDIRECTION'] * 360
            wind_dir_rad = np.radians(wind_dir_deg)

            angle_bins = np.linspace(0, 2 * np.pi, bins + 1)
            hist, _ = np.histogram(wind_dir_rad, bins=angle_bins)
            centers = (angle_bins[:-1] + angle_bins[1:]) / 2

            fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
            width = 2 * np.pi / bins
            bars = ax.bar(centers, hist, width=width, alpha=0.7, edgecolor='black')

            ax.set_theta_zero_location('N')
            ax.set_theta_direction(-1)
            ax.set_title('风向频率玫瑰图', pad=20, fontsize=16)
            ax.set_rlabel_position(0)

            plt.tight_layout()
            return fig, ax

        fig, ax = wind_rose_plot(self.df, bins=24)
        out_path = str(task_dir / 'wind_rose.png')
        plt.savefig(out_path, dpi=200, bbox_inches='tight')
        plt.close()
        print(f"已保存: {out_path}")

        return correlation_matrix, selected_features

    def task3_kmeans_clustering(self, selected_features):
        """任务3：K-means聚类分析（使用公共模块）"""
        if self.df is None:
            raise ValueError("请先运行任务1的数据预处理")

        print("\n" + "=" * 60)
        print("任务3：K-means聚类分析")
        print("=" * 60)

        task_dir = self.file_manager.switch_to_task_folder("Task3")

        # 聚类特征 = 筛选特征 + WINDPOWER（与分步脚本保持一致）
        cluster_features = list(dict.fromkeys(selected_features + ['WINDPOWER']))
        X_cluster = self.df[cluster_features].values
        print(f"聚类使用的特征: {cluster_features}")
        print(f"聚类数据形状: {X_cluster.shape}")

        # 使用公共模块执行聚类（含采样优化的轮廓系数 + K值选择图）
        cluster_result = kmeans_cluster(
            X_cluster,
            save_k_plot_path=str(task_dir / 'kmeans_k_selection.png')
        )

        cluster_labels = cluster_result['labels']
        optimal_k = cluster_result['optimal_k']

        # 将聚类结果添加到数据框
        df_with_clusters = self.df.copy()
        df_with_clusters['cluster'] = cluster_labels

        # PCA可视化
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_cluster)

        plt.figure(figsize=(12, 8))
        scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_labels, cmap='viridis', alpha=0.6, s=20)
        plt.xlabel(f'第一主成分 (解释方差比: {pca.explained_variance_ratio_[0]:.3f})')
        plt.ylabel(f'第二主成分 (解释方差比: {pca.explained_variance_ratio_[1]:.3f})')
        plt.title(f'K-means聚类结果可视化 (K={optimal_k})')
        plt.colorbar(scatter)
        plt.grid(True, alpha=0.3)
        out_path = str(task_dir / 'kmeans_clusters.png')
        plt.savefig(out_path, dpi=200, bbox_inches='tight')
        plt.close()
        print(f"已保存: {out_path}")

        # 风速-功率散点图
        plt.figure(figsize=(10, 6))
        scatter = plt.scatter(df_with_clusters['WINDSPEED'], df_with_clusters['WINDPOWER'],
                              c=cluster_labels, cmap='viridis', alpha=0.6, s=10)
        plt.xlabel('风速 (归一化)')
        plt.ylabel('功率 (归一化)')
        plt.title(f'风速-功率散点图 (按聚类着色, K={optimal_k})')
        plt.colorbar(scatter)
        plt.grid(True, alpha=0.3)
        out_path = str(task_dir / 'speed_power_clusters.png')
        plt.savefig(out_path, dpi=200, bbox_inches='tight')
        plt.close()
        print(f"已保存: {out_path}")

        return df_with_clusters, optimal_k

    def task4_wind_prediction(self, df_with_clusters, optimal_k, selected_features):
        """任务4：风电功率预测"""
        print("\n" + "=" * 60)
        print("任务4：风电功率预测模型")
        print("=" * 60)

        task_dir = self.file_manager.switch_to_task_folder("Task4")

        # 准备数据：根据聚类结果分割数据（使用与 task2 一致的特征）
        X = df_with_clusters[selected_features].values
        y = df_with_clusters['WINDPOWER'].values
        clusters = df_with_clusters['cluster'].values

        # 按聚类结果分割数据
        clusters_data = {}
        for cluster_id in range(optimal_k):
            mask = clusters == cluster_id
            clusters_data[cluster_id] = {
                'X': X[mask],
                'y': y[mask],
                'indices': np.where(mask)[0]
            }
            print(f"  类别 {cluster_id}: {len(clusters_data[cluster_id]['X'])} 个样本")

        # 初始化结果存储
        results = {}

        # 遍历每个聚类进行建模
        for cluster_id in range(optimal_k):
            print(f"\n--- 处理聚类 {cluster_id} ---")
            X_cluster = clusters_data[cluster_id]['X']
            y_cluster = clusters_data[cluster_id]['y']

            if len(X_cluster) < 2:
                print(f"  聚类 {cluster_id} 样本数太少，跳过建模")
                continue

            X_train, X_test, y_train, y_test = train_test_split(
                X_cluster, y_cluster, test_size=0.2, random_state=42
            )

            print(f"  训练集大小: {X_train.shape[0]}, 测试集大小: {X_test.shape[0]}")

            results[cluster_id] = {}

            # 1. SVR
            print(f"  1. 训练SVR模型...")
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

            # 2. BP
            print(f"  2. 训练BP神经网络模型...")
            bp_model = MLPRegressor(
                hidden_layer_sizes=(32, 32),
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

            # 3. 线性回归
            print(f"  3. 训练线性回归模型...")
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

        # 4. 模型评估与对比
        print(f"\n4. 模型评估与对比")
        print("-" * 60)

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
            print(summary_df.round(4))

            fig, axes = plt.subplots(1, 3, figsize=(18, 5))

            for i, metric in enumerate(['MAE', 'RMSE', 'R2']):
                try:
                    pivot_data = summary_df.pivot(index='Cluster', columns='Model', values=metric)
                    if pivot_data is not None and not pivot_data.empty:
                        pivot_data.plot(kind='bar', ax=axes[i])
                        axes[i].set_title(f'{metric} 对比')
                        axes[i].set_ylabel(metric)
                        axes[i].legend()
                        axes[i].tick_params(axis='x', rotation=45)
                        axes[i].grid(True, alpha=0.3)
                except Exception as e:
                    print(f"绘制 {metric} 图时出错: {e}")

            plt.tight_layout()
            out_path = str(task_dir / 'model_comparison.png')
            plt.savefig(out_path, dpi=200, bbox_inches='tight')
            plt.close()
            print(f"已保存: {out_path}")

            for cluster_id in results:
                for model_name in results[cluster_id]:
                    if results[cluster_id][model_name] is not None:
                        actual = results[cluster_id][model_name]['actual']
                        pred = results[cluster_id][model_name]['predictions']

                        plt.figure(figsize=(10, 6))
                        plt.plot(range(len(actual)), actual, label='真实值', alpha=0.7)
                        plt.plot(range(len(pred)), pred, label='预测值', alpha=0.7)
                        plt.xlabel('样本索引')
                        plt.ylabel('功率 (归一化)')
                        plt.title(f'聚类 {cluster_id} - {model_name} 模型预测效果')
                        plt.legend()
                        plt.grid(True, alpha=0.3)
                        out_path = str(task_dir / f'prediction_{cluster_id}_{model_name}.png')
                        plt.savefig(out_path, dpi=200, bbox_inches='tight')
                        plt.close()
                        print(f"已保存: {out_path}")

        return results

    def run_complete_analysis(self):
        """运行完整的分析流程"""
        print("开始运行完整的风电数据分析...")

        # 任务1：数据预处理
        df_normalized = self.task1_data_preprocessing()

        # 任务2：可视化与相关性分析
        correlation_matrix, selected_features = self.task2_visualization_correlation()

        # 任务3：K-means聚类分析
        df_with_clusters, optimal_k = self.task3_kmeans_clustering(selected_features)

        # 任务4：预测模型
        results = self.task4_wind_prediction(df_with_clusters, optimal_k, selected_features)

        print("\n" + "=" * 60)
        print("所有任务完成！")
        print("=" * 60)
        print("各任务的输出文件已保存在独立的子文件夹中:")
        print("目录结构: task_outputs/TaskName/Timestamp/")
        base_dir = self.file_manager.base_dir
        for main_task_dir in base_dir.iterdir():
            if main_task_dir.is_dir():
                print(f"  {main_task_dir.name}/")
                for sub_dir in main_task_dir.iterdir():
                    if sub_dir.is_dir():
                        print(f"    {sub_dir.name}: {len(list(sub_dir.glob('*')))} 个文件")
        print("=" * 60)

        return {
            'df_normalized': df_normalized,
            'correlation_matrix': correlation_matrix,
            'selected_features': selected_features,
            'df_with_clusters': df_with_clusters,
            'optimal_k': optimal_k,
            'results': results
        }


# 主程序入口
if __name__ == "__main__":
    analyzer = WindAnalysisNewStructure()
    results = analyzer.run_complete_analysis()
