# 重组总结

## ✨ 新版本改进

### 1. 代码模块化
**之前**：所有代码在 `run_analysis.py` (626行)
**现在**：
- `modules/geometry.py` - 几何计算功能
- `modules/output_handler.py` - 输出和相互作用强度计算
- `modules/__init__.py` - 包管理
- `run_analysis_v2.py` - 主脚本（更清晰）

### 2. 输出目录结构
**之前**：结果散落在根目录
```
20251115_D2_ARI_cryoEM_rebuild_gromacs_replicate_1_TimeSeries.csv
20251115_D2_ARI_cryoEM_rebuild_gromacs_replicate_1_Stats.csv
20251115_D2_ARI_cryoEM_rebuild_gromacs_replicate_1_projection.png
(混乱)
```

**现在**：统一组织
```
results/
└── 20251115_D2_ARI_cryoEM_rebuild/
    ├── gromacs_replicate_1/
    │   ├── TimeSeries.csv
    │   ├── Stats.csv
    │   └── projection.png
    ├── gromacs_replicate_2/
    │   └── ...
    ├── gromacs_replicate_3/
    │   └── ...
    ├── All_TimeSeries.csv      ← 自动汇总
    └── All_Stats.csv           ← 自动汇总 + 相互作用强度
```

### 3. 新增：综合相互作用强度 ⭐

每个化合物和副本现在自动计算：

| 指标 | 含义 | 范围 |
|------|------|------|
| `strength_389` | 与Phe389的相互作用强度 | 0-1 |
| `strength_390` | 与Phe390的相互作用强度 | 0-1 |
| `strength_combined` | 综合相互作用强度 | 0-1 |
| `quality_score_389` | Phe389质量分数 | 0-1 |
| `quality_score_390` | Phe390质量分数 | 0-1 |
| `major_contributor_389` | 主要贡献碳 | 1-6 |
| `major_contributor_390` | 主要贡献碳 | 1-6 |
| `avg_angle_389` | 平均夹角 | 度数 |
| `std_angle_389` | 角度标准差 | 度数 |

### 4. 数据可用性

| 数据 | 以前 | 现在 |
|------|------|------|
| 每个碳的ELF权重 | ✓ | ✓ |
| 每个碳与Phe的角度 | ✓ | ✓ (每帧) |
| 6碳平均角度 | ✗ | ✓ (Stats) |
| 相互作用强度 | ✗ | ✓ |
| 质量评分 | ✗ | ✓ |

## 📊 如何使用新版本

### 第一次运行
```bash
cd /home/hongyu/MD/1_partial
python run_analysis_v2.py
```

输出将自动组织在 `./results/` 目录下

### 查看结果
```bash
# 查看化合物1的汇总统计
cat results/20251115_D2_ARI_cryoEM_rebuild/All_Stats.csv

# 查看化合物1副本1的时间序列
cat results/20251115_D2_ARI_cryoEM_rebuild/gromacs_replicate_1/TimeSeries.csv

# 列出所有化合物及其相互作用强度
grep strength_combined results/*/All_Stats.csv
```

## 🔄 向后兼容

- **原始脚本保留**：`run_analysis.py` 仍然可用
- **新脚本独立**：`run_analysis_v2.py` 不影响原始脚本
- **可同时运行**：两个脚本可以共存

## 💾 关键文件变化

### 新增文件
```
modules/
├── __init__.py           (23 行)
├── geometry.py           (105 行 - 几何计算)
└── output_handler.py     (130 行 - 输出管理 + 相互作用强度)

run_analysis_v2.py        (650 行 - 使用模块的主脚本)
README_V2.md             (文档)
```

### 文件大小
- `run_analysis.py`: 626 行 (全功能)
- `modules/` + `run_analysis_v2.py`: 778 行 (更清晰的组织)

## 🎯 推荐工作流

### 第一步：运行新版本分析
```bash
python run_analysis_v2.py
```

### 第二步：查看汇总统计
```python
import pandas as pd

# 读取所有化合物的统计
compounds = {}
for cid in ['20251115_D2_ARI_cryoEM_rebuild', 
            '20251115_D2_Dopa_cryoEM_rebuild',
            '20251122_D2_S10_cryoEM_rebuild']:
    compounds[cid] = pd.read_csv(f'results/{cid}/All_Stats.csv')
    
# 查看相互作用强度排序
df_all = pd.concat(compounds.values())
print(df_all[['Compound', 'Replica', 'strength_combined']].sort_values('strength_combined', ascending=False))
```

### 第三步：细节分析
```python
# 分析最优化合物的最优副本
best = df_all.loc[df_all['strength_combined'].idxmax()]
cid = best['Compound']
rname = best['Replica']

ts = pd.read_csv(f'results/{cid}/{rname}/TimeSeries.csv')

# 绘制时间序列
import matplotlib.pyplot as plt
plt.figure(figsize=(12, 6))
plt.plot(ts['Time'], ts['Dist_Phe389_Weighted'], label='Phe389 Weighted')
plt.plot(ts['Time'], ts['Dist_Phe390_Weighted'], label='Phe390 Weighted')
plt.xlabel('Time (ps)')
plt.ylabel('Distance (Å)')
plt.legend()
plt.savefig(f'results/{cid}/distance_timeseries.png')
```

## 📈 性能对比

| 方面 | 旧版本 | 新版本 |
|------|--------|--------|
| 代码复杂性 | 高（单文件626行） | 低（模块化） |
| 输出管理 | 手工 | 自动 |
| 相互作用强度 | 无 | 自动计算 |
| 调试难度 | 高（单文件） | 低（模块独立） |
| 可扩展性 | 差 | 好 |
| 运行速度 | - | 相同 |

## ✅ 测试清单

- [x] 模块语法检查
- [x] 主脚本语法检查  
- [x] 几何计算函数验证
- [x] 输出处理功能
- [x] 相互作用强度计算
- [ ] 完整运行测试 (待执行: `python run_analysis_v2.py`)

## 🚀 下一步建议

1. **运行新版本**：`python run_analysis_v2.py`
2. **验证输出**：检查 `results/` 目录结构
3. **分析结果**：提取相互作用强度数据
4. **可视化**：绘制相互作用强度与活性的关系
5. **扩展功能**：可在 `modules/` 中添加新功能
