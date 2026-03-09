# 🚀 5分钟快速开始

## 第一步：验证系统 (1分钟)

```bash
cd /home/hongyu/MD/1_partial
python verify_setup.py
```

看到这个输出就表示准备就绪：
```
✓ 系统设置完整，可以运行分析

快速开始命令:
  python run_analysis_v2.py
```

## 第二步：运行分析 (1-5分钟)

```bash
python run_analysis_v2.py
```

脚本会自动：
- 读取所有CSV文件 ✓
- 识别化合物和副本 ✓
- 计算T-stacking相互作用 ✓
- 生成结果到 `results/` 目录 ✓

## 第三步：查看结果 (2分钟)

### 看总体结果
```bash
# 查看某个化合物的综合强度
ls results/D2_ARI/All_Stats.csv
head -5 results/D2_ARI/All_Stats.csv
```

### 看详细数据
```bash
# 查看某个副本的统计
cat results/D2_ARI/gromacs_replicate_1/Stats.csv

# 查看逐帧数据
head results/D2_ARI/gromacs_replicate_1/TimeSeries.csv
```

---

## 关键输出文件说明

### `All_Stats.csv` - 最重要的结果汇总

包含列：
- **Strength_Combined** → 相互作用强度 (0-1)
  - 0.85-1.00 = 优秀 ⭐⭐⭐⭐⭐
  - 0.60-0.85 = 良好 ⭐⭐⭐⭐
  - 0.35-0.60 = 一般 ⭐⭐⭐
  - 0.00-0.35 = 较差 ⭐⭐

- **Quality_Score_389/390** → 贡献均匀性 (0-1)
  - 高分 = 多个碳均匀贡献
  - 低分 = 单个碳主导

- **Major_Contributor_389/390** → 主要贡献碳 (C1-C6)
  - 说明哪个碳位置最优

- **C*_Avg_Angle** → 每个碳的平均夹角
  - 接近90° = T-stacking配置

---

## 目录结构（自动生成）

```
results/
├─ D2_ARI/
│  ├─ All_Stats.csv          ← 查看这个！
│  ├─ All_TimeSeries.csv
│  ├─ gromacs_replicate_1/
│  │  ├─ Stats.csv
│  │  ├─ TimeSeries.csv
│  │  └─ projection.png
│  ├─ gromacs_replicate_2/...
│  └─ gromacs_replicate_3/...
├─ D2_Dopa/...
├─ D2_S10/...
└─ ... (其他化合物)
```

---

## 理解强度值

### 什么是 Strength_Combined？

衡量配体与D2受体的T-stacking相互作用强度。

**基于三层因素**：
1. **电子密度** (ELF权重) - 该碳位置有多少电子
2. **角度** (Angle_Decay) - 碳是否垂直指向Phe平面（90°最优）
3. **距离** (Distance_Decay) - 碳是否接近Phe平面（越近越好）

**最终值 = 三层因素的加权组合** → 0-1之间的数字

### 为什么是0-1？

- **1.0** = 完美的T-stacking配置（所有碳都90°且接近平面）
- **0.5** = 中等相互作用（部分碳偏离）
- **0.0** = 完全没有T-stacking特征（随机方向）

---

## 常见问题速解

### Q: 脚本运行多久？
**A**: 1-5分钟（取决于数据量。59个CSV文件通常5分钟内完成）

### Q: 可以修改参数吗？
**A**: 可以。编辑这两个地方：
```
modules/geometry.py 第26行: angle_sigma = 30.0
modules/geometry.py 第43行: distance_sigma = 2.0
```

### Q: 如果只想分析一个化合物？
**A**: 脚本自动识别，无需修改。所有化合物都会被处理。

### Q: 输出文件太多了怎么办？
**A**: 只需要关注 `results/{化合物}/All_Stats.csv`，其他是详细数据。

### Q: 与原来的版本有什么不同？
**A**: 
- 更快（模块化执行）
- 更清晰（自动目录组织）
- 更强大（新增Strength_Combined等指标）

---

## 下一步

### 数据分析
```bash
# 列出所有化合物的强度值
for dir in results/*/; do
  echo "$(basename $dir):"
  grep "^Strength_Combined" $dir/All_Stats.csv | tail -1
done
```

### 深入了解
需要更多详情？查看这些文档：
- 📖 `USAGE_GUIDE.md` - 完整使用说明
- 📖 `README_V2.md` - 技术细节
- 📖 `INDEX.md` - 所有文档索引

### 数据可视化
```bash
# projection.png 已自动生成在每个目录下
# 使用任何图片查看器打开
ls results/*/gromacs_replicate_1/projection.png
```

---

## 成功标志 ✓

看到这些就表示完成了：

```
✓ 所有输入CSV被识别
✓ results/ 目录已创建
✓ 每个化合物有 All_Stats.csv
✓ 每个副本有 Stats.csv 和 TimeSeries.csv
✓ 有 projection.png 可视化文件
✓ All_Stats.csv 最后一行是 "AVERAGE" 汇总
```

---

## 需要帮助？

1. **检查环境**: `python verify_setup.py`
2. **查看文档**: `INDEX.md` 目录导航
3. **查看具体问题**: `USAGE_GUIDE.md` 的常见问题章节

---

**祝您分析顺利！** 🎉

