# 重组后的分析框架

## 📁 目录结构

```
/home/hongyu/MD/1_partial/
├── modules/                          # 模块化代码库
│   ├── __init__.py                  # 包初始化
│   ├── geometry.py                  # 几何计算（平面、向量、衰减等）
│   └── output_handler.py            # 输出管理和相互作用强度计算
│
├── results/                          # 所有结果输出目录（自动创建）
│   ├── Compound_1/
│   │   ├── replica_1/
│   │   │   ├── Compound_1_replica_1_TimeSeries.csv
│   │   │   ├── Compound_1_replica_1_Stats.csv
│   │   │   └── Compound_1_replica_1_projection.png
│   │   ├── replica_2/
│   │   └── ...
│   │   ├── Compound_1_All_TimeSeries.csv    # 汇总时间序列
│   │   └── Compound_1_All_Stats.csv         # 汇总统计（包括相互作用强度）
│   ├── Compound_2/
│   └── ...
│
├── run_analysis_v2.py               # 主分析脚本（新版本）
├── run_analysis.py                  # 原始脚本（保留）
└── [数据目录]                        # 原始MD数据
    ├── 20251115_D2_ARI_cryoEM_rebuild/
    ├── 20251115_D2_Dopa_cryoEM_rebuild/
    └── ...
```

## 🚀 使用方法

### 运行新版本分析
```bash
cd /home/hongyu/MD/1_partial
python run_analysis_v2.py
```

### 输出文件说明

#### 时间序列文件 (TimeSeries.csv)
每一帧的详细数据：
- 基础：`Time`, `Replica`, `Global_Angle`
- 距离：`Dist_Phe389_Geo`, `Dist_Phe389_Weighted`, `Dist_Phe390_Geo`, `Dist_Phe390_Weighted`
- 角度：`C1_Angle_to_Phe389`...`C6_Angle_to_Phe390` (12个碳角度列)
- OBP：各个OBP残基的距离

#### 统计文件 (Stats.csv)
汇总统计数据，包括：
- **基础统计**：化合物ID、副本名、偏移量
- **距离统计**：几何距离和加权距离的平均值
- **角度统计**：每个碳与389/390平面的平均夹角
- **ELF权重**：`C1_Weight`...`C6_Weight`（每个碳的电子密度）
- **相互作用强度** ⭐：
  - `strength_389`: Phe389相互作用强度 (0-1)
  - `strength_390`: Phe390相互作用强度 (0-1)
  - `strength_combined`: 综合相互作用强度 (0-1)
  - `quality_score_389/390`: 质量分数 (0-1)
  - `major_contributor_389/390`: 主要贡献碳原子编号 (1-6)
  - `avg_angle_389/390`: 平均夹角 (度数)
  - `std_angle_389/390`: 角度标准差

#### 汇总文件 (All_Stats.csv)
所有副本的统计结果 + AVERAGE行

## 📊 三层加权机制

每个碳原子对相互作用的贡献由三个因素决定：

```
最终权重 = ELF权重 × 角度衰减 × 距离衰减

ELF权重：              角度衰减：           距离衰减：
电子云密度强弱        与T型的匹配度      到Phe平面的接近程度
(0-1)               (0-1)             (0-1)
```

## 🎯 相互作用强度的含义

### Strength (强度)
- **0.8-1.0**：优秀 (所有6个碳都有贡献)
- **0.5-0.8**：良好 (大部分碳有贡献)
- **0.2-0.5**：一般 (仅部分碳有贡献)
- **< 0.2**：较差 (接触不充分)

### Quality Score (质量分数)
最优碳原子的加权贡献程度，越接近1越好

## 📈 模块化架构优势

1. **模块化代码**
   - 几何计算独立（`geometry.py`）
   - 输出管理独立（`output_handler.py`）
   - 便于后续扩展和维护

2. **统一输出目录**
   - 所有结果在 `./results/` 下
   - 按化合物和副本组织
   - 易于管理和查询

3. **自动汇总**
   - 自动计算多个副本的平均值
   - 自动计算综合相互作用强度
   - 所有数据一目了然

## 🔍 关键指标查询

### 如何找到最优对接的化合物？
查看 `results/*/All_Stats.csv` 中的 `strength_combined` 列
- 最高值 = 最优对接

### 如何判断哪个碳贡献最大？
查看 `major_contributor_389` 和 `major_contributor_390` 列
- 值为 1-6，对应C1-C6

### 如何评估T型堆积质量？
查看 `avg_angle_389` 和 `avg_angle_390` 列
- 接近 90° = 完美T型
- 80-100° = 很好
- < 70° 或 > 110° = 偏离大

## 📝 后续分析建议

1. **分析相互作用强度与活性的关联**
   ```
   results/*/All_Stats.csv → strength_combined vs 活性数据
   ```

2. **比较不同化合物的对接模式**
   ```
   绘制相互作用强度分布
   查看主要贡献碳的分布
   分析角度分布
   ```

3. **细粒度分析（基于时间序列）**
   ```
   results/Compound_1/replica_1/TimeSeries.csv
   分析角度和距离随时间的变化
   识别稳定对接的时间窗口
   ```

## 🔧 自定义配置

修改 `run_analysis_v2.py` 中的配置区：

```python
OUTPUT_BASE_DIR = "./results"        # 修改输出目录
INTEGRATION_RADIUS = 1.5            # ELF积分半径
# ... 其他参数
```

## 📌 版本对比

| 特性 | run_analysis.py | run_analysis_v2.py |
|------|-----------------|-------------------|
| 代码组织 | 单文件 | 模块化 |
| 输出位置 | 根目录散落 | results 统一目录 |
| 角度计算 | ✓ | ✓ |
| 距离衰减 | ✓ | ✓ |
| 相互作用强度 | ✗ | ✓ |
| 自动汇总 | 部分 | 完整 |

## 💡 示例输出

```
>>> Processing (V2.0 Modularized)...
Analyzing: 20251115_D2_ARI_cryoEM_rebuild (3 reps)

======================================================================
综合相互作用强度分析
======================================================================
Phe389 相互作用强度: 0.758 (质量分数: 0.945)
Phe390 相互作用强度: 0.823 (质量分数: 0.978)
综合相互作用强度: 0.790

Phe389 主要贡献碳: C2
  平均夹角: 88.5° (标准差: 3.2°)

Phe390 主要贡献碳: C3
  平均夹角: 89.1° (标准差: 2.8°)
======================================================================

  -> Done 20251115_D2_ARI_cryoEM_rebuild
```
