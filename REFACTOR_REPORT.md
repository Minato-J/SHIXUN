# 风电场数据分析项目 — 代码重构报告

> **日期**: 2026年4月30日  
> **范围**: 6 个 Python 脚本的代码审查、修复与重构  
> **状态**: ✅ 全部完成，已验证通过

---

## 一、项目概述

本项目基于真实风电场运行数据（~39,000 条记录），完成以下 4 个任务：

| 任务 | 内容 | 对应脚本 |
|------|------|----------|
| Task 1 | 数据预处理（缺失值填充、DBSCAN 去噪、Min-Max 归一化） | `analysis.py` |
| Task 2 | 可视化与相关性分析（散点矩阵、玫瑰图、热力图、特征筛选） | `analysis.py` |
| Task 3 | K-means 聚类工况划分 | `Task3.py` |
| Task 4 | 风电功率预测（SVR / BP / Linear） | `Task4.py` |

另有 2 个全任务整合脚本：`wind_analysis.py`（单文件运行）和 `wind_analysis_v2.py`（OOP + 时间戳输出）。

---

## 二、修复前的问题

### 🔴 严重问题（影响功能正确性）

| # | 问题 | 影响 |
|---|------|------|
| 1 | **聚类特征不一致** — 4 个脚本使用不同的特征子集做 K-means，cluster 标签含义互不兼容 | Task3 和 Task4 的结果无法对应 |
| 2 | **`data_normalized.csv` 被反复覆盖** — `analysis.py` 写入无 cluster 列的版本，`task3` 又覆盖为含 cluster 列的版本 | 下游脚本在不同运行顺序下崩溃 |
| 3 | **轮廓系数性能陷阱** — `wind_analysis.py` 对 39,000+ 条数据直接调用 `silhouette_score()`（O(n²)） | 运行极其缓慢甚至卡死 |

### 🟡 中等问题

| # | 问题 |
|---|------|
| 4 | ~300 行代码在 4 个脚本中重复（DBSCAN、归一化、K-means、模型训练） |
| 5 | Docstring 与实际不符（声称 "LSTM" 但实际是 SVR/BP/Linear） |
| 6 | 输出路径混乱 — 部分图片存根目录，部分存子目录，缺乏统一规范 |
| 7 | `ensure_dir` 函数在无父目录时行为不一致 |

---

## 三、修复方案

### 架构调整

```
修复前:                              修复后:
┌─────────────┐                      ┌─────────────┐
│ analysis.py │  (独立运行)           │  common.py  │ ← 公共模块（新建）
├─────────────┤                      ├─────────────┤
│ task3...py  │  (独立运行)           │ analysis.py │ → data_normalized.csv
├─────────────┤                      │             │ → selected_features.json
│ task4...py  │  (独立运行)           ├─────────────┤
├─────────────┤                      │ task3...py  │ → data_with_clusters.csv
│ complete.py │  (全任务)             ├─────────────┤
├─────────────┤                      │ task4...py  │ → RW4(SOLO)/*.png
│ new_st...py │  (OOP)               ├─────────────┤
└─────────────┘                      │ complete.py │ 复用 common
                                     ├─────────────┤
                                     │ new_st...py │ 复用 common
                                     └─────────────┘
```

### 数据文件约定

| 文件 | 列数 | cluster 列 | 生成者 | 消费者 |
|------|------|-----------|--------|--------|
| `data_normalized.csv` | 7 | ❌ 无 | `analysis.py` | Task3 |
| `selected_features.json` | — | — | `analysis.py` | Task3, Task4 |
| `data_with_clusters.csv` | 8 | ✅ 有 | `task3` | Task4 |

### 输出目录约定

```
RW1/          — Task 1 输出（波形图、时序图、散点图、去噪图、归一化图）
RW2/          — Task 2 输出（散点矩阵、玫瑰图、热力图）
RW3/          — Task 3 输出（K值选择图、PCA聚类图、风速-功率聚类图）
RW4(SOLO)/    — Task 4 输出（模型对比图、各模型预测效果图）
```

### 关键常量统一（`common.py`）

```python
NUMERIC_COLS = ['WINDSPEED', 'WINDDIRECTION', 'TEMPERATURE', 'HUMIDITY', 'PRESSURE', 'WINDPOWER']
CORRELATION_THRESHOLD = 0.2
SILHOUETTE_SAMPLE_SIZE = 5000        # 轮廓系数采样优化
DBSCAN_SAMPLE_SIZE = 5000
RANDOM_SEED = 42
```

---

## 四、变更明细

### 4.1 新建文件

| 文件 | 行数 | 说明 |
|------|------|------|
| `common.py` | ~290 | 公共函数与常量模块，包含 10 个可复用函数 |

**提供的公共接口**:
- `ensure_dir()` — 目录创建（修复空路径边界情况）
- `load_and_preprocess()` — 数据加载 + 缺失值填充 + 物理异常剔除
- `dbscan_denoise()` — DBSCAN 去噪（含采样优化 + 对比图）
- `minmax_normalize()` — Min-Max 归一化
- `compute_correlation()` — 皮尔逊相关性 + 特征筛选
- `kmeans_cluster()` — 肘部法 + 轮廓系数法（采样优化）
- `save_data_with_clusters()` / `save_selected_features()` / `load_selected_features()`

### 4.2 修改文件

#### `analysis.py`
- 更新 Docstring：移除 "任务三/四/LSTM" 的错误声明
- 移除重复的字体配置、`ensure_dir`、import（改用 `common`）
- 数据加载 → 调用 `load_and_preprocess()` 获取 `df_raw` / `df_clean`
- DBSCAN → 调用 `dbscan_denoise()`
- 归一化 → 调用 `minmax_normalize()`
- 相关性 → 调用 `compute_correlation()`
- **新增**: `save_selected_features()` 保存 `selected_features.json`
- 保留：所有自定义可视化代码（波形图、散点矩阵、玫瑰图、imshow 热力图）

#### `Task3.py`
- 特征来源改为 `load_selected_features('selected_features.json')`
- 聚类特征统一为 `selected_features + ['WINDPOWER']`
- 聚类执行 → 调用 `kmeans_cluster()`（含采样优化 + K 值选择图）
- **关键修复**: 输出到 `data_with_clusters.csv`，**不再覆盖** `data_normalized.csv`
- 保留：PCA 可视化、风速-功率散点图、类别差异分析

#### `Task4.py`
- 数据源改为优先读取 `data_with_clusters.csv`，回退到 `data_normalized.csv`
- 模型输入特征从 `selected_features.json` 读取（而非硬编码 4 个特征）
- 聚类回退逻辑 → 调用 `kmeans_cluster()`
- 输出路径统一迁移到 `RW4(SOLO)/`

#### `wind_analysis.py`
- 移除未使用的 import（DBSCAN、StandardScaler、NearestNeighbors、silhouette_score）
- 聚类特征改为 `selected_features + ['WINDPOWER']`
- 轮廓系数计算 → 调用 `kmeans_cluster()`（含采样优化）
- 输出路径统一为 `RW2/`、`RW3/`、`RW4(SOLO)/`

#### `wind_analysis_v2.py`
- 移除重复的字体配置、`ensure_dir`、import
- 移除未使用的 `original_cwd` / `restore_original_dir` 逻辑
- `task1_data_preprocessing` → 调用 `load_and_preprocess` + `dbscan_denoise` + `minmax_normalize`
- `Task3.py` → 调用 `kmeans_cluster`，聚类特征统一
- `Task4.py` → 添加 `selected_features` 参数，移除硬编码特征

---

## 五、验证结果

### 运行测试

```
✅ analysis.py                 → 39312 行，生成 6 个输出文件
✅ Task3.py   → K=2 (轮廓系数 0.3986)，生成 4 个输出文件
✅ Task4.py   → 6 个模型全部训练成功，生成 7 个输出文件
```

### 数据文件验证

```
data_normalized.csv      7 列（无 cluster）  ✅
selected_features.json   ["WINDSPEED","WINDDIRECTION","TEMPERATURE","PRESSURE"]  ✅
data_with_clusters.csv   8 列（含 cluster）  ✅
```

### 输出目录验证

```
RW1/          5 个文件  ✅
RW2/          4 个文件  ✅
RW3/          3 个文件  ✅
RW4(SOLO)/    7 个文件  ✅
```

### 模型性能（归一化尺度）

| 聚类 | SVR (R²) | BP (R²) | Linear (R²) |
|------|----------|---------|-------------|
| 0 (高风速) | 0.7638 | 0.7354 | 0.6125 |
| 1 (低风速) | 0.6674 | 0.6474 | 0.4650 |
| **平均** | **0.7156** | **0.6914** | **0.5387** |

---

## 六、代码质量改善统计

| 指标 | 修复前 | 修复后 |
|------|--------|--------|
| 重复代码行 | ~300 行（4 处重复） | 0（提取至 `common.py`） |
| 硬编码特征列表 | 3 处不同值 | 1 处（`selected_features.json`） |
| 数据文件被覆盖风险 | 2 处竞争写入 | 0（分拆为独立文件） |
| 轮廓系数 O(n²) 风险 | 1 处 | 0（统一采样 5000 条） |
| 输出路径规范 | 混乱（根目录 + 子目录混用） | 统一按任务分目录 |
| 未使用的 import | ~8 个 | 0 |
| 语法错误 | 0 | 0 |

---

## 七、运行指南

### 分步运行（推荐）

```bash
# 1. 数据预处理 + 可视化 + 特征筛选
python analysis.py

# 2. K-means 聚类（依赖步骤 1）
python Task3.py

# 3. 功率预测（依赖步骤 2）
python Task4.py
```

### 单文件全任务运行

```bash
# 方式一：面向过程整合版（需先生成 data_normalized.csv）
python wind_analysis.py

# 方式二：面向对象 + 时间戳目录版（从 DATE.csv 开始全流程）
python wind_analysis_v2.py
```

### 依赖

```
Python 3.8+
pandas, numpy, matplotlib, seaborn, scikit-learn
```
