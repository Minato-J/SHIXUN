"""
================================================================================
风电场数据分析项目 — 新版文件管理系统（详细注释版）
================================================================================

【脚本功能概述】
本脚本是四任务整合脚本的面向对象版本，与 wind_analysis.py（函数式风格）
功能相同但架构更优。采用类封装 + 时间戳输出目录的设计，每次运行结果自动
归档到独立文件夹，避免覆盖历史输出。

【架构设计】
  ┌─ SimpleTaskFileManager 类
  │   职责：管理输出目录结构
  │   目录：task_outputs/TaskName/{YYYYMMDD_HHMMSS}/
  │   功能：自动创建任务文件夹、时间戳命名、路径管理
  │
  └─ WindAnalysisNewStructure 类
      职责：执行四个分析任务
      方法：
        task1_data_preprocessing()     — 数据预处理（缺失填充→DBSCAN→归一化）
        task2_visualization_correlation() — 可视化与相关性分析
        task3_kmeans_clustering()       — K-means聚类分析
        task4_wind_prediction()         — 风电功率预测（SVR/BP/Linear）
        run_complete_analysis()         — 编排全部任务流程

【目录结构】
  task_outputs/
  ├── Task1/{20260429_093754}/    # 任务一输出：data_normalized.csv, waveforms.png
  ├── Task2/{20260429_093911}/    # 任务二输出：correlation_heatmap.png, wind_rose.png
  ├── Task3/{20260429_110228}/    # 任务三输出：kmeans_k_selection.png 等
  └── Task4/{20260429_110228}/    # 任务四输出：model_comparison.png 等

【与 wind_analysis.py 的区别】
  - 本文件：面向对象风格（OOP），带时间戳输出目录，每次运行不覆盖历史
  - wind_analysis.py：函数式风格，固定输出目录（RW2/RW3/RW4(SOLO)），
    每次运行覆盖之前结果

【核心优化】
  - 复用 common.py 公共模块（消除代码重复）
  - 使用 pathlib.Path 处理路径（跨平台兼容）
  - 不改变当前工作目录（避免副作用）
  - 所有任务结果通过 self.df 在内存中传递（减少 CSV 中间读写）

【依赖说明】
  - 需要项目根目录下存在 DATE.csv 原始数据文件
  - 依赖 common.py 公共模块（load_and_preprocess、dbscan_denoise 等）
"""

# ===========================================================================
# 第一部分：导入依赖库
# ===========================================================================

import pandas as pd               # 数据处理：DataFrame、CSV 读写
import numpy as np                # 数值计算：数组运算、数学函数
import matplotlib.pyplot as plt   # 数据可视化：绘制波形图、散点图、柱状图
import seaborn as sns             # 统计可视化：相关性热力图
from sklearn.model_selection import train_test_split  # 数据集划分：训练/测试 8:2
from sklearn.svm import SVR                # 支持向量回归模型（RBF核）
from sklearn.neural_network import MLPRegressor  # BP神经网络回归器
from sklearn.linear_model import LinearRegression  # 线性回归模型（最小二乘法）
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# 评估指标：
#   MAE  — 平均绝对误差（越小越好，量纲与目标一致）
#   RMSE — 均方根误差（对异常值敏感，惩罚大误差）
#   R²   — 决定系数（越接近1拟合越好）
from sklearn.decomposition import PCA  # 主成分分析：高维聚类数据降维到2D可视化
import warnings                   # 警告控制
from pathlib import Path          # 现代路径处理：比 os.path 更简洁、跨平台
import os                         # 操作系统接口：兼容性保留
from datetime import datetime     # 日期时间：生成时间戳输出目录名

warnings.filterwarnings('ignore')  # 忽略所有警告（如收敛警告），保持输出整洁

# --- 从公共模块导入共享函数和常量 ---
# ensure_dir           : 递归创建目录（兼容旧路径风格）
# load_and_preprocess  : 加载原始数据 → 缺失值填充 → 物理异常剔除
# dbscan_denoise       : DBSCAN密度聚类去噪
# minmax_normalize     : Min-Max归一化（缩放到[0,1]）
# compute_correlation  : 皮尔逊相关系数矩阵 + 特征筛选
# kmeans_cluster       : K-means聚类（肘部法+轮廓系数自动选K）
# NUMERIC_COLS         : 六维数值特征列名列表常量
# CORRELATION_THRESHOLD: 相关性筛选阈值常量
from common import (
    ensure_dir, load_and_preprocess, dbscan_denoise, minmax_normalize,
    compute_correlation, kmeans_cluster,
    NUMERIC_COLS, CORRELATION_THRESHOLD,
)


# ===========================================================================
# 第二部分：SimpleTaskFileManager 类 — 任务文件管理器
# ===========================================================================

class SimpleTaskFileManager:
    """
    简化版任务文件管理器

    【设计意图】
    为每次运行创建独立的输出目录，使用时间戳确保不同运行的结果互不覆盖。
    目录结构：task_outputs/TaskName/{YYYYMMDD_HHMMSS}/

    【使用示例】
        fm = SimpleTaskFileManager()
        task_dir = fm.create_task_folder("Task1")
        # task_dir = Path("task_outputs/Task1/20260501_143025")
    """

    def __init__(self, base_dir="./task_outputs"):
        """
        初始化文件管理器

        参数:
            base_dir : 所有任务输出的根目录路径（默认为 ./task_outputs）
        """
        # Path.resolve() 将相对路径转为绝对路径，确保后续操作不受工作目录变化影响
        self.base_dir = Path(base_dir).resolve()
        self.base_dir.mkdir(exist_ok=True)     # 如果根目录不存在则创建
        self.current_task_dir = None           # 当前活动的任务目录

    def create_task_folder(self, task_name, use_timestamp=True):
        """
        为指定任务创建独立的输出文件夹

        参数:
            task_name     : 任务名称（如 "Task1", "Task2"），作为子目录名
            use_timestamp : 是否使用时间戳作为最终目录名（默认 True）
                            True  → task_outputs/Task1/20260501_143025/
                            False → task_outputs/Task1/latest/

        返回:
            task_dir : 创建的目录 Path 对象
        """
        # 先创建任务名目录（如 task_outputs/Task1/）
        main_task_dir = self.base_dir / task_name
        main_task_dir.mkdir(exist_ok=True)

        # 生成时间戳或使用 "latest" 作为子目录名
        if use_timestamp:
            # 格式：YYYYMMDD_HHMMSS（年月日_时分秒），便于排序和识别
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            task_dir = main_task_dir / timestamp
        else:
            task_dir = main_task_dir / "latest"

        task_dir.mkdir(exist_ok=True)

        self.current_task_dir = task_dir  # 更新当前活动目录
        print(f"任务文件夹已创建: {task_dir}")

        return task_dir

    def get_current_task_folder(self):
        """
        获取当前活动的任务文件夹路径

        返回:
            Path 对象或 None（如果尚未创建任何任务目录）
        """
        return self.current_task_dir

    def switch_to_task_folder(self, task_name, use_timestamp=True):
        """
        创建并切换到指定任务的文件夹（便捷方法）

        内部调用 create_task_folder，返回创建的目录路径。
        注意：不改变当前进程的工作目录（cwd），仅管理文件输出路径。
        """
        task_dir = self.create_task_folder(task_name, use_timestamp)
        return task_dir


# ===========================================================================
# 第三部分：WindAnalysisNewStructure 类 — 风电数据分析主类
# ===========================================================================

class WindAnalysisNewStructure:
    """
    风电数据分析主类（面向对象设计）

    【设计理念】
    将四个任务封装为独立方法，通过实例属性（self.df）在任务间传递数据，
    避免重复读写 CSV 文件。使用 SimpleTaskFileManager 管理输出目录。

    【使用示例】
        analyzer = WindAnalysisNewStructure()
        results = analyzer.run_complete_analysis()
    """

    def __init__(self, data_path=None):
        """
        初始化分析器

        参数:
            data_path : 原始数据文件路径（暂未使用，保留扩展接口）
                        始终使用项目根目录下的 DATE.csv
        """
        # Path.cwd() 获取当前工作目录（应为项目根目录）
        self.data_source = Path.cwd() / "DATE.csv"
        self.df = None                      # 存储预处理后的数据，在任务间传递
        self.file_manager = SimpleTaskFileManager()  # 文件管理器实例

    # ================================================================
    # 任务一：数据预处理
    # ================================================================
    def task1_data_preprocessing(self):
        """
        任务1：数据预处理（完整流程）

        处理步骤（调用 common.py 公共模块）：
          1) load_and_preprocess : 加载 DATE.csv → 缺失值前向填充 → 物理异常剔除
          2) describe()          : 对风速和功率进行描述性统计
          3) dbscan_denoise      : DBSCAN密度聚类去噪（自动估计 epsilon）
          4) minmax_normalize    : Min-Max归一化到 [0,1] 区间
          5) 保存归一化数据 + 绘制5属性波形图

        返回:
            df_normalized : 归一化后的 DataFrame（含 DATATIME 列）
        """
        print("=" * 60)
        print("任务1：数据预处理")
        print("=" * 60)

        # --- 创建任务1专属输出文件夹（带时间戳） ---
        task_dir = self.file_manager.switch_to_task_folder("Task1")

        # --- 步骤1：加载并初步清洗原始数据 ---
        # 返回值：df_raw（原始含异常值）、df_clean（剔除负风速/负功率后）
        df_raw, df_clean = load_and_preprocess(str(self.data_source))

        # --- 步骤2：描述性统计 ---
        # 仅关注核心变量：风速（WINDSPEED）和功率（WINDPOWER）
        stats = df_raw[['WINDSPEED', 'WINDPOWER']].describe().T
        stats['range'] = stats['max'] - stats['min']  # 计算极差
        print("\nWINDSPEED (风速, m/s) 和 WINDPOWER (有功功率, kW) 的描述性统计:")
        print(stats[['mean', 'std', 'min', '25%', '50%', '75%', 'max', 'range']].to_string())

        # --- 步骤3：DBSCAN 密度聚类去噪 ---
        # 原理：DBSCAN 基于样本密度自动识别并剔除离群噪声点
        # 详细说明参见 Comment for project/common.py 和 analysis.py
        df_denoised = dbscan_denoise(df_clean)

        # --- 步骤4：Min-Max 归一化 ---
        # 公式：x' = (x - min) / (max - min)，将值映射到 [0,1]
        df_normalized, _ = minmax_normalize(df_denoised)

        # --- 步骤5：保存结果 ---
        # 同时保存到任务目录（归档）和项目根目录（供其他脚本使用）
        df_normalized.to_csv(str(task_dir / 'data_normalized.csv'), index=False)
        df_normalized.to_csv('data_normalized.csv', index=False)
        print("\n已保存: data_normalized.csv (归一化后的数据集)")

        # --- 步骤6：绘制五属性时间序列波形图 ---
        # 5×1 子图布局，共享X轴（DATATIME），展示风速/功率/温度/湿度/气压的变化趋势
        fig, axes = plt.subplots(5, 1, figsize=(14, 12), sharex=True)
        features = ['WINDSPEED', 'WINDPOWER', 'TEMPERATURE', 'HUMIDITY', 'PRESSURE']
        titles = ['风速 (m/s)', '有功功率 (kW)', '温度 (°C)', '湿度 (%)', '气压 (hPa)']
        colors = ['steelblue', 'firebrick', 'forestgreen', 'orange', 'purple']

        for ax, col, title, color in zip(axes, features, titles, colors):
            ax.plot(df_normalized['DATATIME'], df_normalized[col],
                    color=color, linewidth=0.3, alpha=0.7)
            ax.set_ylabel(title)
            ax.grid(True, alpha=0.3)
            if col == 'WINDPOWER':
                ax.set_ylim(bottom=0)  # 功率不能为负，强制Y轴从0开始

        axes[-1].set_xlabel('DateTime')
        plt.tight_layout()
        out_path = str(task_dir / 'waveforms.png')
        plt.savefig(out_path, dpi=200)
        plt.close()
        print(f"已保存: {out_path}")

        # --- 将预处理数据缓存到实例属性，供后续任务使用 ---
        self.df = df_normalized

        return df_normalized

    # ================================================================
    # 任务二：可视化与相关性分析
    # ================================================================
    def task2_visualization_correlation(self):
        """
        任务2：可视化与相关性分析

        分析内容（调用 common.py 的 compute_correlation）：
          1) 皮尔逊相关系数热力图（6×6矩阵）
          2) 风向频率玫瑰图（极坐标柱状图，24分区）
          3) 特征筛选（|r| ≥ 阈值 保留的特征用于后续建模）

        返回:
            correlation_matrix : 6×6 皮尔逊相关系数矩阵
            selected_features  : 筛选后的特征名列表（如 ['WINDSPEED', 'PRESSURE', ...]）

        前置条件:
            必须先运行 task1_data_preprocessing() 设置 self.df
        """
        if self.df is None:
            raise ValueError("请先运行任务1的数据预处理")

        print("\n" + "=" * 60)
        print("任务2：可视化与相关性分析")
        print("=" * 60)

        task_dir = self.file_manager.switch_to_task_folder("Task2")

        # --- 相关性分析（调用公共模块） ---
        # compute_correlation 内部完成：
        #   1) 计算 6×6 皮尔逊相关系数矩阵
        #   2) 提取与 WINDPOWER 的相关系数并排序
        #   3) 按阈值筛选特征
        correlation_matrix, power_corr_sorted, selected_features = compute_correlation(self.df)

        # --- 绘制相关性热力图 ---
        # 详细原理参见 Comment for project/analysis.py 和 wind_analysis.py
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                    square=True, fmt='.3f', cbar_kws={'shrink': 0.8})
        plt.title('特征相关性热力图')
        plt.tight_layout()
        out_path = str(task_dir / 'correlation_heatmap.png')
        plt.savefig(out_path, dpi=200, bbox_inches='tight')
        plt.close()
        print(f"已保存: {out_path}")

        # --- 绘制风向玫瑰图 ---
        # 将归一化风向值（0~1）→ 角度（0°~360°）→ 极坐标柱状图
        def wind_rose_plot(df, bins=24):
            """
            风向频率玫瑰图（内部辅助函数）

            参数:
                df   : 包含 WINDDIRECTION 列的 DataFrame
                bins : 方向分区数（24 = 每15°一个区间）

            返回:
                fig, ax : matplotlib 图形和极坐标轴对象
            """
            # 归一化值 → 角度（0~360°）→ 弧度
            wind_dir_deg = df['WINDDIRECTION'] * 360
            wind_dir_rad = np.radians(wind_dir_deg)

            # 划分角度区间并统计各区间样本数
            angle_bins = np.linspace(0, 2 * np.pi, bins + 1)
            hist, _ = np.histogram(wind_dir_rad, bins=angle_bins)
            centers = (angle_bins[:-1] + angle_bins[1:]) / 2

            # 创建极坐标图
            fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
            width = 2 * np.pi / bins
            bars = ax.bar(centers, hist, width=width, alpha=0.7, edgecolor='black')

            # 设置极坐标方向：0°指向正北，顺时针递增（气象学惯例）
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

    # ================================================================
    # 任务三：K-means 聚类分析
    # ================================================================
    def task3_kmeans_clustering(self, selected_features):
        """
        任务3：K-means 聚类分析

        流程（调用 common.py 的 kmeans_cluster）：
          1) 使用筛选特征 + WINDPOWER 构建聚类特征矩阵
          2) K-means 聚类（肘部法 SSE + 轮廓系数自动选最优K）
          3) PCA 降维到2D进行可视化
          4) 风速-功率散点图按聚类着色

        参数:
            selected_features : 任务二中筛选出的特征列表

        返回:
            df_with_clusters : 增加了 cluster 列的 DataFrame
            optimal_k        : 最优聚类数 K

        详细原理参见 Comment for project/Task3.py
        """
        if self.df is None:
            raise ValueError("请先运行任务1的数据预处理")

        print("\n" + "=" * 60)
        print("任务3：K-means聚类分析")
        print("=" * 60)

        task_dir = self.file_manager.switch_to_task_folder("Task3")

        # --- 构建聚类特征矩阵 ---
        # 聚类特征 = 筛选特征 + WINDPOWER（目标与特征共同决定运行工况）
        # dict.fromkeys() 去重：保持顺序，去除可能的重复元素
        cluster_features = list(dict.fromkeys(selected_features + ['WINDPOWER']))
        X_cluster = self.df[cluster_features].values
        print(f"聚类使用的特征: {cluster_features}")
        print(f"聚类数据形状: {X_cluster.shape}")

        # --- 执行 K-means 聚类 ---
        # kmeans_cluster 内部包含：肘部法 SSE 计算、采样轮廓系数计算、
        # K值选择图自动保存
        cluster_result = kmeans_cluster(
            X_cluster,
            save_k_plot_path=str(task_dir / 'kmeans_k_selection.png')
        )

        cluster_labels = cluster_result['labels']    # 各样本的聚类标签
        optimal_k = cluster_result['optimal_k']      # 自动确定的最优聚类数

        # 将聚类结果添加到数据框（使用 copy 避免修改原始数据）
        df_with_clusters = self.df.copy()
        df_with_clusters['cluster'] = cluster_labels

        # --- PCA 降维到 2D 可视化 ---
        # 原理：将多维特征投影到方差最大的两个主成分方向上
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_cluster)

        plt.figure(figsize=(12, 8))
        scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1],
                              c=cluster_labels, cmap='viridis', alpha=0.6, s=20)
        plt.xlabel(f'第一主成分 (解释方差比: {pca.explained_variance_ratio_[0]:.3f})')
        plt.ylabel(f'第二主成分 (解释方差比: {pca.explained_variance_ratio_[1]:.3f})')
        plt.title(f'K-means聚类结果可视化 (K={optimal_k})')
        plt.colorbar(scatter)
        plt.grid(True, alpha=0.3)
        out_path = str(task_dir / 'kmeans_clusters.png')
        plt.savefig(out_path, dpi=200, bbox_inches='tight')
        plt.close()
        print(f"已保存: {out_path}")

        # --- 风速-功率聚类散点图 ---
        # 在原始风速-功率空间上展示聚类效果，直观判断聚类物理意义
        plt.figure(figsize=(10, 6))
        scatter = plt.scatter(df_with_clusters['WINDSPEED'],
                              df_with_clusters['WINDPOWER'],
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

    # ================================================================
    # 任务四：风电功率预测
    # ================================================================
    def task4_wind_prediction(self, df_with_clusters, optimal_k, selected_features):
        """
        任务4：风电功率预测模型

        方法：对每个聚类分别训练三种回归模型
          - SVR（支持向量回归，RBF核）
          - BP 神经网络（2×32隐藏层，ReLU，Adam）
          - 线性回归（最小二乘法，性能基准）

        评估指标：MAE、RMSE、R²
        输出：三模型柱状对比图 + 各模型预测 vs 真实值曲线

        参数:
            df_with_clusters : 含 cluster 列的 DataFrame
            optimal_k        : 聚类数量
            selected_features: 筛选后的特征列表

        返回:
            results : 嵌套字典 {cluster_id: {model_name: {...}}}

        详细原理参见 Comment for project/Task4.py
        """
        print("\n" + "=" * 60)
        print("任务4：风电功率预测模型")
        print("=" * 60)

        task_dir = self.file_manager.switch_to_task_folder("Task4")

        # --- 步骤1：按聚类分割数据 ---
        X = df_with_clusters[selected_features].values  # 特征矩阵
        y = df_with_clusters['WINDPOWER'].values         # 目标向量
        clusters = df_with_clusters['cluster'].values    # 聚类标签

        # 按聚类标签将数据划分为独立子集
        clusters_data = {}
        for cluster_id in range(optimal_k):
            mask = clusters == cluster_id    # 布尔掩码
            clusters_data[cluster_id] = {
                'X': X[mask],
                'y': y[mask],
                'indices': np.where(mask)[0]
            }
            print(f"  类别 {cluster_id}: {len(clusters_data[cluster_id]['X'])} 个样本")

        # --- 步骤2：对每个聚类训练三种模型 ---
        results = {}

        for cluster_id in range(optimal_k):
            print(f"\n--- 处理聚类 {cluster_id} ---")
            X_cluster = clusters_data[cluster_id]['X']
            y_cluster = clusters_data[cluster_id]['y']

            # 安全检查
            if len(X_cluster) < 2:
                print(f"  聚类 {cluster_id} 样本数太少，跳过建模")
                continue

            # 8:2 训练/测试划分
            X_train, X_test, y_train, y_test = train_test_split(
                X_cluster, y_cluster, test_size=0.2, random_state=42
            )

            print(f"  训练集大小: {X_train.shape[0]}, 测试集大小: {X_test.shape[0]}")

            results[cluster_id] = {}

            # ------ 模型1：支持向量回归 (SVR) ------
            # 原理：RBF核将数据映射高维，在高维空间寻找最优回归超平面
            print(f"  1. 训练SVR模型...")
            svr_model = SVR(kernel='rbf', C=1.0, epsilon=0.1)
            svr_model.fit(X_train, y_train)
            y_pred_svr = svr_model.predict(X_test)

            svr_mae = mean_absolute_error(y_test, y_pred_svr)
            svr_rmse = np.sqrt(mean_squared_error(y_test, y_pred_svr))
            svr_r2 = r2_score(y_test, y_pred_svr)

            results[cluster_id]['SVR'] = {
                'model': svr_model, 'predictions': y_pred_svr, 'actual': y_test,
                'mae': svr_mae, 'rmse': svr_rmse, 'r2': svr_r2
            }
            print(f"    MAE: {svr_mae:.4f}, RMSE: {svr_rmse:.4f}, R2: {svr_r2:.4f}")

            # ------ 模型2：BP 神经网络 ------
            # 结构：输入层 → 隐藏层1(32个神经元,ReLU) → 隐藏层2(32,ReLU) → 输出层(1)
            print(f"  2. 训练BP神经网络模型...")
            bp_model = MLPRegressor(
                hidden_layer_sizes=(32, 32), activation='relu', solver='adam',
                learning_rate_init=0.001, max_iter=500, random_state=42
            )
            bp_model.fit(X_train, y_train)
            y_pred_bp = bp_model.predict(X_test)

            bp_mae = mean_absolute_error(y_test, y_pred_bp)
            bp_rmse = np.sqrt(mean_squared_error(y_test, y_pred_bp))
            bp_r2 = r2_score(y_test, y_pred_bp)

            results[cluster_id]['BP'] = {
                'model': bp_model, 'predictions': y_pred_bp, 'actual': y_test,
                'mae': bp_mae, 'rmse': bp_rmse, 'r2': bp_r2
            }
            print(f"    MAE: {bp_mae:.4f}, RMSE: {bp_rmse:.4f}, R2: {bp_r2:.4f}")

            # ------ 模型3：线性回归 ------
            # y = w₁x₁ + w₂x₂ + ... + b，最小二乘法拟合
            print(f"  3. 训练线性回归模型...")
            linear_model = LinearRegression()
            linear_model.fit(X_train, y_train)
            y_pred_linear = linear_model.predict(X_test)

            linear_mae = mean_absolute_error(y_test, y_pred_linear)
            linear_rmse = np.sqrt(mean_squared_error(y_test, y_pred_linear))
            linear_r2 = r2_score(y_test, y_pred_linear)

            results[cluster_id]['Linear'] = {
                'model': linear_model, 'predictions': y_pred_linear, 'actual': y_test,
                'mae': linear_mae, 'rmse': linear_rmse, 'r2': linear_r2
            }
            print(f"    MAE: {linear_mae:.4f}, RMSE: {linear_rmse:.4f}, R2: {linear_r2:.4f}")

        # --- 步骤3：模型评估与可视化对比 ---
        print(f"\n4. 模型评估与对比")
        print("-" * 60)

        # 扁平化嵌套结果 → DataFrame
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

            # --- 三指标柱状对比图 ---
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

            # --- 各模型预测值 vs 真实值曲线图 ---
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

    # ================================================================
    # 编排方法：一键运行全流程
    # ================================================================
    def run_complete_analysis(self):
        """
        运行完整的四任务分析流程（编排方法）

        执行顺序：
          Task1 → 数据预处理
          Task2 → 可视化与相关性分析 → 获取筛选特征
          Task3 → K-means 聚类 → 获取聚类标签
          Task4 → 风电功率预测 → 获取模型结果

        返回:
            包含所有关键结果的字典：
            {
                'df_normalized': ...,
                'correlation_matrix': ...,
                'selected_features': ...,
                'df_with_clusters': ...,
                'optimal_k': ...,
                'results': ...
            }
        """
        print("开始运行完整的风电数据分析...")

        # 任务1：数据预处理（缺失填充 → DBSCAN去噪 → Min-Max归一化）
        df_normalized = self.task1_data_preprocessing()

        # 任务2：可视化与相关性分析（热力图、玫瑰图、特征筛选）
        correlation_matrix, selected_features = self.task2_visualization_correlation()

        # 任务3：K-means 聚类分析（肘部法+轮廓系数选K、PCA可视化）
        df_with_clusters, optimal_k = self.task3_kmeans_clustering(selected_features)

        # 任务4：风电功率预测（SVR/BP/Linear 三模型对比）
        results = self.task4_wind_prediction(df_with_clusters, optimal_k, selected_features)

        # --- 输出汇总信息 ---
        print("\n" + "=" * 60)
        print("所有任务完成！")
        print("=" * 60)
        print("各任务的输出文件已保存在独立的子文件夹中:")
        print("目录结构: task_outputs/TaskName/Timestamp/")

        # 遍历所有任务目录，统计各时间戳子目录的文件数
        base_dir = self.file_manager.base_dir
        for main_task_dir in base_dir.iterdir():
            if main_task_dir.is_dir():
                print(f"  {main_task_dir.name}/")
                for sub_dir in main_task_dir.iterdir():
                    if sub_dir.is_dir():
                        file_count = len(list(sub_dir.glob('*')))
                        print(f"    {sub_dir.name}: {file_count} 个文件")
        print("=" * 60)

        return {
            'df_normalized': df_normalized,
            'correlation_matrix': correlation_matrix,
            'selected_features': selected_features,
            'df_with_clusters': df_with_clusters,
            'optimal_k': optimal_k,
            'results': results
        }


# ===========================================================================
# 第四部分：主程序入口
# ===========================================================================
# Python 惯用写法：当脚本被直接运行时（而非 import），执行以下代码
if __name__ == "__main__":
    # 1. 实例化分析器对象（自动初始化文件管理器）
    analyzer = WindAnalysisNewStructure()

    # 2. 一键运行四个任务的完整分析流程
    results = analyzer.run_complete_analysis()
