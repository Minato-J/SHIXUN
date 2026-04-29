"""
风电数据分析项目 - 新版文件管理系统
目录结构: task_outputs/TaskName/{timestamp}
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
import warnings
from pathlib import Path
import os
from datetime import datetime

warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

class SimpleTaskFileManager:
    """
    简化版任务文件管理器
    目录结构: task_outputs/TaskName/{timestamp}
    """
    
    def __init__(self, base_dir="./task_outputs"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)
        self.current_task_dir = None
        self.original_cwd = os.getcwd()  # 保存原始工作目录
    
    def create_task_folder(self, task_name, use_timestamp=True):
        """
        为任务创建独立的文件夹
        目录结构: task_outputs/TaskName/{timestamp}
        
        Args:
            task_name: 任务名称 (如 "Task1", "Task2")
            use_timestamp: 是否使用时间戳作为子文件夹名
        """
        # 创建主任务文件夹
        main_task_dir = self.base_dir / task_name
        main_task_dir.mkdir(exist_ok=True)
        
        # 创建时间戳子文件夹
        if use_timestamp:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            task_dir = main_task_dir / timestamp
        else:
            task_dir = main_task_dir / "latest"
        
        task_dir.mkdir(exist_ok=True)
        
        # 设置环境变量供任务使用
        os.environ['TASK_OUTPUT_DIR'] = str(task_dir.absolute())
        
        self.current_task_dir = task_dir
        print(f"任务文件夹已创建: {task_dir}")
        
        return task_dir
    
    def get_current_task_folder(self):
        """获取当前任务文件夹路径"""
        return self.current_task_dir
    
    def switch_to_task_folder(self, task_name, use_timestamp=True):
        """
        切换到任务文件夹（改变当前工作目录）
        
        Args:
            task_name: 任务名称 (如 "Task1", "Task2")
            use_timestamp: 是否使用时间戳作为子文件夹名
        """
        task_dir = self.create_task_folder(task_name, use_timestamp)
        os.chdir(task_dir)
        return task_dir
    
    def restore_original_dir(self):
        """恢复到原始工作目录"""
        os.chdir(self.original_cwd)


class WindAnalysisNewStructure:
    """
    风电数据分析类，使用新的目录结构
    """
    
    def __init__(self, data_path=None):
        # 使用绝对路径确保能找到数据文件
        if data_path is None:
            self.data_path = Path(self.__class__.__module__).parent / "DATE.csv"  # 使用当前目录下的文件
        else:
            self.data_path = Path(data_path).resolve()  # 转换为绝对路径
        self.df = None
        self.file_manager = SimpleTaskFileManager()
        
    def task1_data_preprocessing(self):
        """任务1：数据预处理"""
        print("=" * 60)
        print("任务1：数据预处理")
        print("=" * 60)
        
        # 切换到任务1的文件夹 (目录结构: task_outputs/Task1/{timestamp})
        self.file_manager.switch_to_task_folder("Task1")
        
        # 使用绝对路径加载数据
        data_file_path = Path(self.file_manager.original_cwd) / "DATE.csv"
        df = pd.read_csv(data_file_path)
        print(f"数据集形状: {df.shape[0]} 行 × {df.shape[1]} 列")
        
        # 时间解析
        df['DATATIME'] = pd.to_datetime(df['DATATIME'])
        df = df.sort_values('DATATIME').reset_index(drop=True)
        
        # 描述性统计
        numeric_cols = ['WINDSPEED', 'WINDDIRECTION', 'TEMPERATURE', 'HUMIDITY', 'PRESSURE', 'WINDPOWER']
        stats = df[numeric_cols].describe().T
        stats['range'] = stats['max'] - stats['min']
        stats_table = stats[['mean', 'std', 'min', '25%', '50%', '75%', 'max', 'range']]
        print("\nWINDSPEED (风速, m/s) 和 WINDPOWER (有功功率, kW) 的描述性统计:")
        print(stats_table.to_string())
        
        # 缺失值处理
        missing = df.isnull().sum()
        if missing.sum() > 0:
            for col in df.columns:
                if col == 'DATATIME':
                    continue
                df[col] = df[col].fillna(method='ffill')
        
        # 物理异常值剔除
        df = df[df['WINDPOWER'] >= 0].reset_index(drop=True)
        df = df[df['WINDSPEED'] >= 0].reset_index(drop=True)
        
        # DBSCAN去噪
        X = df[['WINDSPEED', 'WINDPOWER']].values
        scaler_std = StandardScaler()
        X_scaled = scaler_std.fit_transform(X)
        
        sample_size = min(5000, len(X_scaled))
        idx_sample = np.random.choice(len(X_scaled), sample_size, replace=False)
        X_sample = X_scaled[idx_sample]
        
        neigh = NearestNeighbors(n_neighbors=10)
        neigh.fit(X_sample)
        distances, _ = neigh.kneighbors(X_sample)
        k_dist = np.sort(distances[:, -1])
        epsilon = np.percentile(k_dist, 95)
        
        db = DBSCAN(eps=epsilon, min_samples=10)
        labels = db.fit_predict(X_scaled)
        
        df_clean = df[labels != -1].copy().reset_index(drop=True)
        
        # 数据归一化
        scaler = MinMaxScaler()
        df_normalized = df_clean.copy()
        df_normalized[numeric_cols] = scaler.fit_transform(df_clean[numeric_cols])
        
        # 保存归一化后的数据
        df_normalized.to_csv('data_normalized.csv', index=False)
        print("\n已保存: data_normalized.csv (归一化后的数据集)")
        
        # 绘制各属性波形图
        fig, axes = plt.subplots(5, 1, figsize=(14, 12), sharex=True)
        features = ['WINDSPEED', 'WINDPOWER', 'TEMPERATURE', 'HUMIDITY', 'PRESSURE']
        titles  = ['风速 (m/s)', '有功功率 (kW)', '温度 (°C)', '湿度 (%)', '气压 (hPa)']
        colors  = ['steelblue', 'firebrick', 'forestgreen', 'orange', 'purple']

        for ax, col, title, color in zip(axes, features, titles, colors):
            ax.plot(df_normalized['DATATIME'], df_normalized[col], color=color, linewidth=0.3, alpha=0.7)
            ax.set_ylabel(title)
            ax.grid(True, alpha=0.3)
            if col == 'WINDPOWER':
                ax.set_ylim(bottom=0)

        axes[-1].set_xlabel('DateTime')
        plt.tight_layout()
        plt.savefig('waveforms.png', dpi=200)
        plt.close()
        print("已保存: waveforms.png")
        
        # 保存预处理后的数据供后续任务使用
        self.df = df_normalized
        
        # 恢复原始目录
        self.file_manager.restore_original_dir()
        return df_normalized
    
    def task2_visualization_correlation(self):
        """任务2：可视化与相关性分析"""
        if self.df is None:
            raise ValueError("请先运行任务1的数据预处理")
        
        print("\n" + "=" * 60)
        print("任务2：可视化与相关性分析")
        print("=" * 60)
        
        # 切换到任务2的文件夹 (目录结构: task_outputs/Task2/{timestamp})
        self.file_manager.switch_to_task_folder("Task2")
        
        numeric_cols = ['WINDSPEED', 'WINDDIRECTION', 'TEMPERATURE', 'HUMIDITY', 'PRESSURE', 'WINDPOWER']
        
        # 相关性分析
        correlation_matrix = self.df[numeric_cols].corr()
        power_corr = correlation_matrix['WINDPOWER'].drop('WINDPOWER').abs().sort_values(ascending=False)
        
        # 特征筛选（相关系数绝对值 >= 0.2）
        threshold = 0.2
        selected_features = power_corr[power_corr >= threshold].index.tolist()
        print(f"\n相关系数绝对值 >= {threshold} 的特征（用于后续建模）:")
        for feature in selected_features:
            print(f"  {feature}: {power_corr[feature]:.4f}")
        
        # 绘制相关性热力图
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
                    square=True, fmt='.3f', cbar_kws={'shrink': 0.8})
        plt.title('特征相关性热力图')
        plt.tight_layout()
        plt.savefig('correlation_heatmap.png', dpi=200, bbox_inches='tight')
        plt.close()
        print("已保存: correlation_heatmap.png")
        
        # 风向玫瑰图
        def wind_rose_plot(df, bins=24):
            wind_dir_deg = df['WINDDIRECTION'] * 360
            wind_dir_rad = np.radians(wind_dir_deg)
            
            angle_bins = np.linspace(0, 2*np.pi, bins+1)
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
        plt.savefig('wind_rose.png', dpi=200, bbox_inches='tight')
        plt.close()
        print("已保存: wind_rose.png")
        
        # 恢复原始目录
        self.file_manager.restore_original_dir()
        return correlation_matrix, selected_features
    
    def task3_kmeans_clustering(self, selected_features):
        """任务3：K-means聚类分析"""
        if self.df is None:
            raise ValueError("请先运行任务1的数据预处理")
        
        print("\n" + "=" * 60)
        print("任务3：K-means聚类分析")
        print("=" * 60)
        
        # 切换到任务3的文件夹 (目录结构: task_outputs/Task3/{timestamp})
        self.file_manager.switch_to_task_folder("Task3")
        
        X_cluster = self.df[selected_features].values
        print(f"聚类使用的特征: {selected_features}")
        print(f"聚类数据形状: {X_cluster.shape}")
        
        # 肘部法和轮廓系数法确定最优K值
        from sklearn.metrics import silhouette_score
        k_range = range(2, 11)
        sse = []
        silhouette_scores = []
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(X_cluster)
            sse.append(kmeans.inertia_)
            
            silhouette_avg = silhouette_score(X_cluster, kmeans.labels_)
            silhouette_scores.append(silhouette_avg)
        
        # 找到最优K值
        optimal_k_silhouette = k_range[np.argmax(silhouette_scores)]
        print(f"基于轮廓系数的最优K值: {optimal_k_silhouette}")
        
        # 绘制K值选择图
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        ax1.plot(k_range, sse, 'bo-')
        ax1.set_xlabel('聚类数 K')
        ax1.set_ylabel('SSE (误差平方和)')
        ax1.set_title('肘部法确定最优K值')
        ax1.grid(True, alpha=0.3)
        
        ax2.plot(k_range, silhouette_scores, 'ro-')
        ax2.set_xlabel('聚类数 K')
        ax2.set_ylabel('轮廓系数')
        ax2.set_title('轮廓系数法确定最优K值')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('kmeans_k_selection.png', dpi=200, bbox_inches='tight')
        plt.close()
        print("已保存: kmeans_k_selection.png")
        
        # 执行聚类
        kmeans_final = KMeans(n_clusters=optimal_k_silhouette, random_state=42, n_init=10)
        cluster_labels = kmeans_final.fit_predict(X_cluster)
        
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
        plt.title(f'K-means聚类结果可视化 (K={optimal_k_silhouette})')
        plt.colorbar(scatter)
        plt.grid(True, alpha=0.3)
        plt.savefig('kmeans_clusters.png', dpi=200, bbox_inches='tight')
        plt.close()
        print("已保存: kmeans_clusters.png")
        
        # 风速-功率散点图
        plt.figure(figsize=(10, 6))
        scatter = plt.scatter(df_with_clusters['WINDSPEED'], df_with_clusters['WINDPOWER'], c=cluster_labels, cmap='viridis', alpha=0.6, s=10)
        plt.xlabel('风速 (归一化)')
        plt.ylabel('功率 (归一化)')
        plt.title(f'风速-功率散点图 (按聚类着色, K={optimal_k_silhouette})')
        plt.colorbar(scatter)
        plt.grid(True, alpha=0.3)
        plt.savefig('speed_power_clusters.png', dpi=200, bbox_inches='tight')
        plt.close()
        print("已保存: speed_power_clusters.png")
        
        # 恢复原始目录
        self.file_manager.restore_original_dir()
        return df_with_clusters, optimal_k_silhouette
    
    def task4_wind_prediction(self, df_with_clusters, optimal_k):
        """任务4：风电功率预测"""
        print("\n" + "=" * 60)
        print("任务4：风电功率预测模型")
        print("=" * 60)
        
        # 切换到任务4的文件夹 (目录结构: task_outputs/Task4/{timestamp})
        self.file_manager.switch_to_task_folder("Task4")
        
        # 准备数据：根据聚类结果分割数据
        selected_features = ['WINDSPEED', 'TEMPERATURE', 'WINDDIRECTION', 'PRESSURE']
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
            
            # 1. 支持向量回归 (SVR)
            print(f"  1. 训练SVR模型...")
            svr_model = SVR(kernel='rbf', C=1.0, epsilon=0.1)
            svr_model.fit(X_train, y_train)
            y_pred_svr = svr_model.predict(X_test)
            
            # 计算SVR评估指标
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
            
            # 2. BP神经网络 (使用sklearn的MLPRegressor)
            print(f"  2. 训练BP神经网络模型...")
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
            
            # 计算BP评估指标
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
            
            # 3. 线性回归模型（作为LSTM的替代）
            print(f"  3. 训练线性回归模型...")
            linear_model = LinearRegression()
            linear_model.fit(X_train, y_train)
            y_pred_linear = linear_model.predict(X_test)
            
            # 计算线性回归评估指标
            linear_mae = mean_absolute_error(y_test, y_pred_linear)
            linear_rmse = np.sqrt(mean_squared_error(y_test, y_pred_linear))
            linear_r2 = r2_score(y_test, y_pred_linear)
            
            results[cluster_id]['Linear'] = {  # 使用Linear代替LSTM
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
            import pandas as pd
            summary_df = pd.DataFrame(summary_results)
            print(summary_df.round(4))
            
            # 绘制模型性能对比图
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
            plt.savefig('model_comparison.png', dpi=200, bbox_inches='tight')
            plt.close()
            print("已保存: model_comparison.png")
            
            # 绘制预测值vs真实值曲线图
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
                        plt.savefig(f'prediction_{cluster_id}_{model_name}.png', dpi=200, bbox_inches='tight')
                        plt.close()
                        print(f"已保存: prediction_{cluster_id}_{model_name}.png")
        
        # 恢复原始目录
        self.file_manager.restore_original_dir()
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
        results = self.task4_wind_prediction(df_with_clusters, optimal_k)
        
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