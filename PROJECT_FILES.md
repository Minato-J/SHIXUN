# 项目文件清单（PROJECT FILES）

> 风电场数据分析与功率预测 — 实训项目完整目录说明  
> 更新日期：2026-05-01

---

## 一、Python 脚本（7个）

| 文件名 | 类型 | 任务归属 | 功能描述 |
|--------|------|----------|----------|
| `common.py` | 公共模块 | 所有任务 | 共享常量、工具函数（数据加载、DBSCAN去噪、归一化、K-means聚类、模型训练等），被所有脚本 `import` |
| `analysis.py` | 独立任务脚本 | Task 1 + Task 2 | 数据预处理（缺失值填充→异常剔除→DBSCAN去噪→Min-Max归一化）+ 可视化与相关性分析（散点矩阵、风向玫瑰图、热力图、特征筛选） |
| `Task3.py` | 独立任务脚本 | Task 3 | K-means 聚类分析（肘部法+轮廓系数法确定K值，PCA降维可视化，风速-功率聚类散点图） |
| `Task4.py` | 独立任务脚本 | Task 4 | 风电功率预测（SVR / BP神经网络 / 线性回归），按聚类类别分别训练，MAE/RMSE/R²评估对比 |
| `wind_analysis.py` | 整合脚本 | Task 1~4 | 全任务一键运行（函数式风格），单文件完成从原始数据到预测评估的全流程 |
| `wind_analysis_v2.py` | 整合脚本 | Task 1~4 | 全任务一键运行（面向对象 + 时间戳输出目录），输出到 `task_outputs/TaskName/{timestamp}/` |

### 脚本依赖链

```
DATE.csv ──→ analysis.py ──→ data_normalized.csv + selected_features.json
                                      ↓
                                 Task3.py ──→ data_with_clusters.csv
                                                  ↓
                                             Task4.py ──→ 预测结果图

wind_analysis.py / wind_analysis_v2.py ──→ 以上全流程（内部集成）
```

---

## 二、数据文件（5个）

| 文件名 | 格式 | 行数 | 列数 | 内容说明 |
|--------|------|------|------|----------|
| `DATE.csv` | CSV | ~39,000 | 7 | **原始风电场数据**。列：DATATIME, WINDSPEED(m/s), WINDDIRECTION(°), TEMPERATURE(°C), HUMIDITY(%), PRESSURE(hPa), WINDPOWER(kW)，采样间隔~15分钟 |
| `data_normalized.csv` | CSV | ~39,000 | 7 | **归一化数据集**（Task1输出）。经缺失填充、异常剔除、DBSCAN去噪、Min-Max归一化至[0,1]，**不含聚类标签列** |
| `data_with_clusters.csv` | CSV | ~39,000 | 8 | **带聚类标签的归一化数据**（Task3输出）。在 `data_normalized.csv` 基础上增加 `cluster` 列（0/1） |
| `statistical_features.csv` | CSV | 6 | 9 | **六维特征统计表**（Task2中间产物）。列：Feature, Mean, Std Dev, Min, Q1, Q2(Median), Q3, Max, Range |
| `selected_features.json` | JSON | — | — | **特征筛选结果**（Task2输出）。内容：`["WINDSPEED", "WINDDIRECTION", "TEMPERATURE", "PRESSURE"]`，供Task3/Task4保持特征一致 |

---

## 三、输出目录（6个）

### 3.1 `RW1/` — Task 1 可视化输出

| 文件 | 说明 |
|------|------|
| `scatter_plot.png` | 风速-功率散点图（原始数据） |
| `scatter_denoised.png` | DBSCAN去噪前后对比散点图 |
| `scatter_normalized.png` | Min-Max归一化后散点图 |
| `waveforms.png` | 五维特征时间序列波形图 |
| `timeseries_plot.png` | 风速+功率双轴时序图 |

### 3.2 `RW2/` — Task 2 可视化输出

| 文件 | 说明 |
|------|------|
| `correlation_heatmap.png` | 皮尔逊相关系数热力图（6×6矩阵） |
| `scatter_matrix.png` | 六维特征散点矩阵图（对角线为分布直方图） |
| `wind_rose.png` | 风向玫瑰图（15°分箱，极坐标） |

### 3.3 `RW3/` — Task 3 可视化输出

| 文件 | 说明 |
|------|------|
| `task3_kmeans_k_selection.png` | K值选择图（肘部法SSE曲线 + 轮廓系数曲线） |
| `task3_kmeans_clusters.png` | PCA二维主成分聚类可视化 |
| `task3_speed_power_clusters.png` | 风速-功率按聚类着色散点图 |

### 3.4 `RW4(SOLO)/` — Task 4 可视化输出

| 文件 | 说明 |
|------|------|
| `task4_model_comparison.png` | 三模型（SVR/BP/Linear）MAE/RMSE/R² 柱状对比图 |
| `task4_prediction_0_SVR.png` | 簇0（高风速工况）SVR预测 vs 实际值 |
| `task4_prediction_0_BP.png` | 簇0 BP神经网络预测 vs 实际值 |
| `task4_prediction_0_Linear.png` | 簇0 线性回归预测 vs 实际值 |
| `task4_prediction_0_LSTM.png` | 簇0 LSTM预测 vs 实际值 |
| `task4_prediction_1_SVR.png` | 簇1（低风速工况）SVR预测 vs 实际值 |
| `task4_prediction_1_BP.png` | 簇1 BP神经网络预测 vs 实际值 |
| `task4_prediction_1_Linear.png` | 簇1 线性回归预测 vs 实际值 |
| `task4_prediction_1_LSTM.png` | 簇1 LSTM预测 vs 实际值 |

### 3.5 `task_outputs/` — 时间戳版执行历史

由 `wind_analysis_v2.py` 生成的带时间戳输出目录：

```
task_outputs/
├── Task1/
│   ├── 20260429_093754/    # task1_data.csv, task1_output.txt
│   ├── 20260429_093909/    # data_normalized.csv, waveforms.png
│   └── 20260429_110225/    # data_normalized.csv, waveforms.png
├── Task2/
│   ├── 20260429_093754/    # task2_output.txt, task2_report.json
│   ├── 20260429_093911/
│   └── 20260429_110227/
├── Task3/
│   ├── 20260429_093911/
│   └── 20260429_110228/
└── Task4/
    （暂无执行记录）
```

### 3.6 `OTHER/` — 实训原始文档

| 文件 | 说明 |
|------|------|
| `人工智能实训 任务.docx` | 实训任务要求书（中文Word文档） |
| `人工智能实训-实验项目手册.pdf` | 实验项目手册（中文PDF） |

---

## 四、文档（4个）

| 文件名 | 说明 |
|--------|------|
| `README.md` | 项目概述、数据字段说明、任务概览、环境依赖、快速开始指南 |
| `REFACTOR_REPORT.md` | 代码重构报告：架构改进、性能优化（轮廓系数采样）、Bug修复记录、P0-P3优先级清单 |
| `KMEANS_K_SELECTION.md` | K-means K值选择方法论：肘部法+轮廓系数数学推导、采样优化策略、实验结果 |
| `PROJECT_FILES.md` | 本文件：项目全部文件/文件夹的用途与内容说明 |

---

## 五、辅助目录

| 目录/文件 | 说明 |
|-----------|------|
| `.git/` | Git版本控制仓库（自动生成） |
| `.idea/` | PyCharm IDE项目配置（自动生成） |
| `.vscode/` | VS Code编辑器配置（自动生成） |
| `__pycache__/` | Python字节码缓存（自动生成，可安全删除） |

---

## 六、完整数据流

```
                        ┌─────────────────────────────────────────┐
                        │             OTHER/ (实训文档)             │
                        └─────────────────────────────────────────┘

  DATE.csv                          analysis.py
  (原始39K条) ──────────────→  [Task1] 预处理 ──→ data_normalized.csv
                               [Task2] 分析   ──→ selected_features.json
                               [Task2] 统计   ──→ statistical_features.csv
                                      │              │
                                      ↓              ↓
                                  RW1/ (5图)     RW2/ (3图)
                                  
                    data_normalized.csv + selected_features.json
                                      │
                                      ↓
                                 Task3.py
                              [Task3] K-means ──→ data_with_clusters.csv
                                      │
                                      ↓
                                  RW3/ (3图)
                                  
                           data_with_clusters.csv
                                      │
                                      ↓
                                 Task4.py
                           [Task4] SVR/BP/Linear
                                      │
                                      ↓
                               RW4(SOLO)/ (9图)

═══════════════════════════════════════════════════════════
  整合执行:
    wind_analysis.py        → 以上全流程（输出到 RW1~RW4(SOLO)）
    wind_analysis_v2.py     → 以上全流程（输出到 task_outputs/）
```
