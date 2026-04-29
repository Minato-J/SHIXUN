"""
优化的风电数据分析完整项目
================================
整合四个任务并采用优化技术：
  1. 数据预处理（缺失值填充、异常值检测、数据归一化）
  2. 可视化与相关性分析
  3. K-means聚类分析
  4. 风电功率预测模型
优化技术：
  - 函数模块化
  - 结果缓存
  - 并行处理
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
from joblib import dump, load
import os
import hashlib
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing as mp

warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

class WindAnalysisOptimizer:
    """
    优化的风电数据分析类，包含缓存、并行处理等功能
    """
    
    def __init__(self, cache_dir='cache'):
        self.cache_dir = cache_dir
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
        self.df = None
        self.results = {}
        
    def _get_cache_key(self, func_name, params):
        """生成缓存键"""
        param_str = f"{func_name}_{str(sorted(params.items()))}"
        return hashlib.md5(param_str.encode()).hexdigest()
    
    def _load_from_cache(self, cache_key):
        """从缓存加载结果"""
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.joblib")
        if os.path.exists(cache_file):
            try:
                return load(cache_file)
            except:
                return None
        return None
    
    def _save_to_cache(self, cache_key, data):
        """保存结果到缓存"""
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.joblib")
        dump(data, cache_file)
    
    def load_and_preprocess_data(self, data_path='DATE.csv', force_recompute=False):
        """任务1：数据加载与预处理"""
        cache_key = self._get_cache_key('load_and_preprocess_data', {'data_path': data_path})
        
        if not force_recompute:
            cached_result = self._load_from_cache(cache_key)
            if cached_result is not None:
                print("从缓存加载预处理数据")
                self.df, stats_info = cached_result
                return self.df, stats_info
        
        print("=" * 60)
        print("任务1：数据加载与预处理")
        print("=" * 60)
        
        # 数据加载
        df = pd.read_csv(data_path)
        print(f"数据集形状: {df.shape[0]} 行 × {df.shape[1]} 列")
        
        # 时间解析
        df['DATATIME'] = pd.to_datetime(df['DATATIME'])
        df = df.sort_values('DATATIME').reset_index(drop=True)
        
        # 描述性统计
        numeric_cols = ['WINDSPEED', 'WINDDIRECTION', 'TEMPERATURE', 'HUMIDITY', 'PRESSURE', 'WINDPOWER']
        stats = df[numeric_cols].describe().T
        stats['range'] = stats['max'] - stats['min']
        stats_table = stats[['mean', 'std', 'min', '25%', '50%', '75%', 'max', 'range']]
        
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
        
        # 保存归一化参数
        normalization_params = {
            'scaler': scaler,
            'scaler_params': {col: {'min': scaler.data_min_[i], 'max': scaler.data_max_[i]} 
                             for i, col in enumerate(numeric_cols)}
        }
        
        self.df = df_normalized
        stats_info = {
            'original_shape': df.shape,
            'cleaned_shape': df_clean.shape,
            'final_shape': df_normalized.shape,
            'stats_table': stats_table,
            'normalization_params': normalization_params
        }
        
        # 保存到缓存
        self._save_to_cache(cache_key, (self.df, stats_info))
        
        print(f"预处理完成，最终数据形状: {self.df.shape}")
        return self.df, stats_info
    
    def visualization_and_correlation(self, force_recompute=False):
        """任务2：可视化与相关性分析"""
        if self.df is None:
            raise ValueError("请先运行数据预处理")
        
        cache_key = self._get_cache_key('visualization_and_correlation', {})
        
        if not force_recompute:
            cached_result = self._load_from_cache(cache_key)
            if cached_result is not None:
                print("从缓存加载可视化和相关性分析结果")
                return cached_result
        
        print("\n" + "=" * 60)
        print("任务2：可视化与相关性分析")
        print("=" * 60)
        
        numeric_cols = ['WINDSPEED', 'WINDDIRECTION', 'TEMPERATURE', 'HUMIDITY', 'PRESSURE', 'WINDPOWER']
        
        # 相关性分析
        correlation_matrix = self.df[numeric_cols].corr()
        power_corr = correlation_matrix['WINDPOWER'].drop('WINDPOWER').abs().sort_values(ascending=False)
        
        # 特征筛选（相关系数绝对值 >= 0.2）
        threshold = 0.2
        selected_features = power_corr[power_corr >= threshold].index.tolist()
        
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
        
        result = {
            'correlation_matrix': correlation_matrix,
            'selected_features': selected_features,
            'power_correlations': power_corr
        }
        
        self._save_to_cache(cache_key, result)
        return result
    
    def kmeans_clustering(self, selected_features=None, force_recompute=False):
        """任务3：K-means聚类分析"""
        if self.df is None:
            raise ValueError("请先运行数据预处理")
        
        if selected_features is None:
            # 如果没有提供特征列表，则需要先运行相关性分析
            _, _, _, selected_features = self.get_selected_features()
        
        cache_key = self._get_cache_key('kmeans_clustering', {'features': tuple(selected_features)})
        
        if not force_recompute:
            cached_result = self._load_from_cache(cache_key)
            if cached_result is not None:
                print("从缓存加载聚类分析结果")
                self.df['cluster'] = cached_result['cluster_labels']
                return cached_result
        
        print("\n" + "=" * 60)
        print("任务3：K-means聚类分析")
        print("=" * 60)
        
        X_cluster = self.df[selected_features].values
        print(f"聚类使用的特征: {selected_features}")
        print(f"聚类数据形状: {X_cluster.shape}")
        
        # 肘部法和轮廓系数法确定最优K值
        k_range = range(2, 11)
        sse = []
        silhouette_scores = []
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(X_cluster)
            sse.append(kmeans.inertia_)
            
            from sklearn.metrics import silhouette_score
            silhouette_avg = silhouette_score(X_cluster, kmeans.labels_)
            silhouette_scores.append(silhouette_avg)
        
        # 找到最优K值
        optimal_k_silhouette = k_range[np.argmax(silhouette_scores)]
        
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
        self.df['cluster'] = cluster_labels
        
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
        scatter = plt.scatter(self.df['WINDSPEED'], self.df['WINDPOWER'], c=cluster_labels, cmap='viridis', alpha=0.6, s=10)
        plt.xlabel('风速 (归一化)')
        plt.ylabel('功率 (归一化)')
        plt.title(f'风速-功率散点图 (按聚类着色, K={optimal_k_silhouette})')
        plt.colorbar(scatter)
        plt.grid(True, alpha=0.3)
        plt.savefig('speed_power_clusters.png', dpi=200, bbox_inches='tight')
        plt.close()
        print("已保存: speed_power_clusters.png")
        
        result = {
            'cluster_labels': cluster_labels,
            'optimal_k': optimal_k_silhouette,
            'kmeans_model': kmeans_final,
            'pca_model': pca
        }
        
        self._save_to_cache(cache_key, result)
        return result
    
    def predict_models_parallel(self, selected_features=None, force_recompute=False):
        """任务4：风电功率预测（使用并行处理）"""
        if self.df is None or 'cluster' not in self.df.columns:
            raise ValueError("请先运行数据预处理和聚类分析")
        
        if selected_features is None:
            # 如果没有提供特征列表，则需要先运行相关性分析
            _, _, _, selected_features = self.get_selected_features()
        
        cache_key = self._get_cache_key('predict_models', {'features': tuple(selected_features)})
        
        if not force_recompute:
            cached_result = self._load_from_cache(cache_key)
            if cached_result is not None:
                print("从缓存加载预测模型结果")
                return cached_result
        
        print("\n" + "=" * 60)
        print("任务4：风电功率预测模型（并行处理）")
        print("=" * 60)
        
        # 准备数据
        X = self.df[selected_features].values
        y = self.df['WINDPOWER'].values
        clusters = self.df['cluster'].values
        
        # 按聚类分割数据
        clusters_data = {}
        for cluster_id in np.unique(clusters):
            mask = clusters == cluster_id
            clusters_data[cluster_id] = {
                'X': X[mask],
                'y': y[mask],
                'indices': np.where(mask)[0]
            }
            print(f"  类别 {cluster_id}: {len(clusters_data[cluster_id]['X'])} 个样本")
        
        # 并行训练模型
        results = {}
        
        def train_cluster_models(cluster_id):
            """为单个聚类训练所有模型"""
            print(f"处理聚类 {cluster_id}...")
            X_cluster = clusters_data[cluster_id]['X']
            y_cluster = clusters_data[cluster_id]['y']
            
            if len(X_cluster) < 2:
                print(f"  聚类 {cluster_id} 样本数太少，跳过建模")
                return cluster_id, {}
            
            # 分割训练集和测试集
            X_train, X_test, y_train, y_test = train_test_split(
                X_cluster, y_cluster, test_size=0.2, random_state=42
            )
            
            cluster_results = {}
            
            # SVR模型
            print(f"  聚类 {cluster_id} - 训练SVR模型...")
            svr_model = SVR(kernel='rbf', C=1.0, epsilon=0.1)
            svr_model.fit(X_train, y_train)
            y_pred_svr = svr_model.predict(X_test)
            
            svr_mae = mean_absolute_error(y_test, y_pred_svr)
            svr_rmse = np.sqrt(mean_squared_error(y_test, y_pred_svr))
            svr_r2 = r2_score(y_test, y_pred_svr)
            
            cluster_results['SVR'] = {
                'model': svr_model,
                'predictions': y_pred_svr,
                'actual': y_test,
                'mae': svr_mae,
                'rmse': svr_rmse,
                'r2': svr_r2
            }
            
            # BP神经网络模型
            print(f"  聚类 {cluster_id} - 训练BP神经网络模型...")
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
            
            cluster_results['BP'] = {
                'model': bp_model,
                'predictions': y_pred_bp,
                'actual': y_test,
                'mae': bp_mae,
                'rmse': bp_rmse,
                'r2': bp_r2
            }
            
            # 线性回归模型（作为LSTM的替代）
            print(f"  聚类 {cluster_id} - 训练线性回归模型...")
            linear_model = LinearRegression()
            linear_model.fit(X_train, y_train)
            y_pred_linear = linear_model.predict(X_test)
            
            linear_mae = mean_absolute_error(y_test, y_pred_linear)
            linear_rmse = np.sqrt(mean_squared_error(y_test, y_pred_linear))
            linear_r2 = r2_score(y_test, y_pred_linear)
            
            cluster_results['Linear'] = {
                'model': linear_model,
                'predictions': y_pred_linear,
                'actual': y_test,
                'mae': linear_mae,
                'rmse': linear_rmse,
                'r2': linear_r2
            }
            
            return cluster_id, cluster_results
        
        # 使用线程池并行处理不同聚类
        with ThreadPoolExecutor(max_workers=mp.cpu_count()) as executor:
            futures = {executor.submit(train_cluster_models, cid): cid for cid in clusters_data.keys()}
            
            for future in as_completed(futures):
                cluster_id, cluster_res = future.result()
                results[cluster_id] = cluster_res
                print(f"聚类 {cluster_id} 的模型训练完成")
        
        # 生成汇总结果
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
            print("\n各模型性能指标汇总:")
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
        
        result = {
            'results': results,
            'summary_df': pd.DataFrame(summary_results) if summary_results else pd.DataFrame()
        }
        
        self._save_to_cache(cache_key, result)
        return result
    
    def get_selected_features(self):
        """获取筛选后的特征（辅助函数）"""
        if self.df is None:
            raise ValueError("请先运行数据预处理")
        
        numeric_cols = ['WINDSPEED', 'WINDDIRECTION', 'TEMPERATURE', 'HUMIDITY', 'PRESSURE', 'WINDPOWER']
        correlation_matrix = self.df[numeric_cols].corr()
        power_corr = correlation_matrix['WINDPOWER'].drop('WINDPOWER').abs().sort_values(ascending=False)
        
        threshold = 0.2
        selected_features = power_corr[power_corr >= threshold].index.tolist()
        
        return correlation_matrix, power_corr, threshold, selected_features
    
    def run_full_analysis(self, force_recompute=False):
        """运行完整的分析流程"""
        print("开始运行完整的风电数据分析...")
        
        # 任务1：数据预处理
        df, stats_info = self.load_and_preprocess_data(force_recompute=force_recompute)
        
        # 任务2：可视化与相关性分析
        vis_result = self.visualization_and_correlation(force_recompute=force_recompute)
        selected_features = vis_result['selected_features']
        
        # 任务3：K-means聚类分析
        cluster_result = self.kmeans_clustering(selected_features, force_recompute=force_recompute)
        
        # 任务4：预测模型（并行处理）
        predict_result = self.predict_models_parallel(selected_features, force_recompute=force_recompute)
        
        print("\n" + "=" * 60)
        print("所有任务完成！")
        print("=" * 60)
        print("生成的文件:")
        print("  1. correlation_heatmap.png   — 相关性热力图")
        print("  2. wind_rose.png             — 风向玫瑰图")
        print("  3. kmeans_k_selection.png    — K-means K值选择图")
        print("  4. kmeans_clusters.png       — K-means聚类结果")
        print("  5. speed_power_clusters.png  — 风速-功率聚类图")
        print("  6. model_comparison.png      — 模型性能对比图")
        print("  7. prediction_*.png          — 各模型预测效果图")
        print("=" * 60)
        
        return {
            'df': df,
            'stats_info': stats_info,
            'vis_result': vis_result,
            'cluster_result': cluster_result,
            'predict_result': predict_result
        }

# 主程序入口
if __name__ == "__main__":
    optimizer = WindAnalysisOptimizer()
    
    # 运行完整分析（如果缓存存在则使用缓存，否则重新计算）
    results = optimizer.run_full_analysis(force_recompute=False)