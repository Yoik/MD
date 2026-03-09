# 使用指南 - run_analysis_v2.py

## 快速开始

### 1. 准备数据结构
你的工作目录应该包含：
```
/home/hongyu/MD/1_partial/
├── run_analysis_v2.py          # 主分析脚本
├── modules/
│   ├── __init__.py
│   ├── geometry.py             # 几何计算
│   └── output_handler.py       # 输出处理与强度计算
├── results/                    # 输出目录（自动创建）
└── 数据文件:
    ├── 20251115_D2_ARI_cryoEM_rebuild_gromacs_replicate_1_TimeSeries.csv
    ├── 20251115_D2_ARI_cryoEM_rebuild_gromacs_replicate_1_Stats.csv
    ├── ... (其他化合物和副本)
```

### 2. 运行脚本

```bash
cd /home/hongyu/MD/1_partial
python run_analysis_v2.py
```

### 3. 输出目录结构

脚本运行后，会生成如下结构：

```
results/
├── D2_ARI/
│   ├── cryoEM_rebuild/
│   │   ├── TimeSeries.csv           # 所有帧的详细数据
│   │   ├── Stats.csv                # 统计数据
│   │   └── projection.png           # 可视化
│   ├── gromacs_replicate_1/
│   │   ├── TimeSeries.csv
│   │   ├── Stats.csv
│   │   └── projection.png
│   ├── gromacs_replicate_2/
│   │   └── ...
│   ├── gromacs_replicate_3/
│   │   └── ...
│   ├── All_TimeSeries.csv           # 聚合：所有副本合并
│   └── All_Stats.csv                # 聚合：统计汇总 + AVERAGE行
├── D2_Dopa/
│   └── ...
└── ...
```

## 输出文件解释

### TimeSeries.csv - 逐帧数据
每一行代表一个MD模拟帧

| 列名 | 含义 |
|------|------|
| Frame | 帧编号 |
| C1_Angle_to_Phe389 | C1到Phe389平面的夹角 |
| ... | ... |
| C6_Angle_to_Phe390 | C6到Phe390平面的夹角 |
| C1_Weight_389 | C1在Phe389计算中的权重 |
| ... | ... |
| Weighted_Distance_Phe389 | Phe389的加权平均距离 |
| Weighted_Distance_Phe390 | Phe390的加权平均距离 |

### Stats.csv - 统计汇总
聚合每个副本的所有帧数据

| 统计量 | 说明 |
|------|------|
| C1_Avg_Angle_to_Phe389 | C1到Phe389夹角的平均值 |
| C1_Std_Angle_to_Phe389 | C1到Phe389夹角的标准差 |
| ... | ... |
| Weighted_Distance_Phe389_Mean | Phe389距离的平均值 |
| Weighted_Distance_Phe389_Std | Phe389距离的标准差 |
| Strength_389 | Phe389相互作用强度 (0-1) |
| Strength_390 | Phe390相互作用强度 (0-1) |
| Strength_Combined | 综合相互作用强度 (0-1) |
| Quality_Score_389 | Phe389的质量分数 |
| Major_Contributor_389 | Phe389的主要贡献碳 |

### All_Stats.csv - 全化合物汇总
包含：
- 所有副本的数据行（标记为 replicate_1, replicate_2, replicate_3）
- **AVERAGE行**：所有副本的平均值和标准差

## 相互作用强度指标解释

### Strength值 (0-1范围)

```
0.85-1.00  → 优秀   (T-stacking配体接近平面，多数碳呈90°角)
0.60-0.85  → 良好   (部分碳偏离，但整体方向正确)
0.35-0.60  → 一般   (多个碳偏离，角度分散)
0.00-0.35  → 较差   (大部分碳与T-stacking无关)
```

### Quality_Score (0-1范围)

衡量该Phe残基对相互作用的贡献质量：
- 基于所有6个碳的权重分布均匀性
- 高分数 = 多个碳均匀贡献
- 低分数 = 单个碳主导贡献

### Major_Contributor

贡献最大的碳原子编号（C1-C6）
- 如果 = "C1"：说明C1的权重最高
- 反映哪个碳位置最优

## 三层权重机制

每个碳对距离计算的贡献权重由三个因子组成：

$$\text{weight} = \text{ELF}_i \times \text{Angle\_Decay}_i \times \text{Distance\_Decay}_i$$

### 1. ELF权重
从电子密度立方体文件读取，反映该碳处的电子密度

### 2. 角度衰减 (Angle_Decay)
```
Angle_Decay = exp(-((|angle - 90°| / 30°)²))
```
- 角度 = 90° → 衰减因子 = 1.0（完美T-stacking）
- 角度 = 60° 或 120° → 衰减因子 ≈ 0.34
- 角度 = 0° 或 180° → 衰减因子 ≈ 0.0

### 3. 距离衰减 (Distance_Decay)
```
Distance_Decay = exp(-(distance / 2.0Å)²)
```
- 距离 = 0Å → 衰减因子 = 1.0（接近平面）
- 距离 = 2Å → 衰减因子 ≈ 0.37
- 距离 = 4Å → 衰减因子 ≈ 0.02

## 数据解释示例

### 高强度结果 (Strength_Combined ≈ 0.92)
```
C1_Avg_Angle_to_Phe389: 88.5°   ← 接近90°
C1_Avg_Angle_to_Phe390: 89.2°   ← 接近90°
Weighted_Distance_Phe389: 1.2Å  ← 接近平面
Weighted_Distance_Phe390: 1.5Å  ← 接近平面
Major_Contributor_389: C3        ← C3贡献最大
```
**解释**：配体与两个Phe呈优秀T-stacking，多数碳位置优良，接近富芳香环受体平面

### 低强度结果 (Strength_Combined ≈ 0.28)
```
C1_Avg_Angle_to_Phe389: 42.3°   ← 偏离90°很远
C2_Avg_Angle_to_Phe389: 138.2°  ← 偏离90°很远
Weighted_Distance_Phe389: 4.8Å  ← 远离平面
Quality_Score_389: 0.15         ← 贡献不均匀
```
**解释**：配体与Phe相互作用差，不是典型的T-stacking配置

## 批量分析

如果有多个化合物，脚本会按名称自动识别并分类：

```python
# 脚本自动识别的化合物格式：
# 20251115_D2_ARI_cryoEM_rebuild_...
#           └─┬─┘  └──┬───┘
#             │        └─ 化合物ID (D2_ARI, D2_Dopa, D2_S10, etc.)
#             └─ 化合物ID (D2_ARI)

# 副本识别：
# gromacs_replicate_1, gromacs_replicate_2, gromacs_replicate_3
# cryoEM_rebuild (单一实验结果)
```

## 常见问题

### Q: 如果某个副本没有ELF立方体文件怎么办？
A: 脚本会跳过该副本的ELF权重，使用等权重计算。日志会显示警告信息。

### Q: Angle_Decay和Distance_Decay都是1.0但Strength还是低，为什么？
A: 可能是ELF权重本身较低，说明该位置的电子密度较低，不是主要相互作用位点。

### Q: 如何修改角度或距离的衰减参数？
A: 编辑 `modules/geometry.py` 中的常数：
- `angle_sigma = 30.0` (当前参数：±30°时衰减50%)
- `distance_sigma = 2.0` (当前参数：±2Å时衰减50%)

### Q: All_Stats.csv中的AVERAGE行是如何计算的？
A: 对所有副本的对应统计量取平均值和标准差。例如：
```
Strength_Combined_AVERAGE = (副本1 + 副本2 + 副本3) / 3
Strength_Combined_STD = std([副本1, 副本2, 副本3])
```

## 与v1的主要改进

| 特性 | v1 | v2 |
|------|----|----|
| 三层权重 | ✓ | ✓ |
| 模块化代码 | ✗ | ✓ |
| 目录组织 | 扁平 | 按化合物/副本组织 |
| 自动聚合 | 手动 | 自动生成All_*.csv |
| 相互作用强度 | ✗ | ✓ (0-1量化) |
| 质量分数 | ✗ | ✓ (识别贡献均匀性) |
| 可维护性 | 低 | 高 (分离关注点) |

## 下一步

1. 运行脚本生成所有化合物的分析结果
2. 查看 `results/` 下各化合物的 `All_Stats.csv`
3. 比较不同化合物的 Strength_Combined 值
4. 关联到实验活性数据，寻找相互作用强度与活性的相关性
5. 识别最优的T-stacking几何配置

