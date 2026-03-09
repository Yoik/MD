#!/usr/bin/env python3
"""
角度计算方式说明：
每个碳原子指向Phe质心的向量 与 Phe环平面法向量 的夹角

这个角度表示该碳原子是否以"T型"的方式接近Phe环
- 0° = 平行于Phe平面（不好，无法有效接触）
- 90° = 垂直于Phe平面（完美T型，最好）
- 180° = 也平行于平面（不好）
"""

import numpy as np

# 创建模拟数据：Phe环的法向量
phe_normal = np.array([0, 0, 1])  # 平面法向量（垂直于Phe平面）
phe_center = np.array([0, 0, 0])

# 模拟6个碳原子的位置（配体苯环上）
# 为简化，假设配体形成近似圆形
angles_config = np.linspace(0, 2*np.pi, 6, endpoint=False)
carbon_positions = np.array([
    [3*np.cos(a), 3*np.sin(a), 0.5 + 0.3*np.sin(2*a)] 
    for a in angles_config
])

# 计算每个碳的角度
carbon_angles = []
for i, c_pos in enumerate(carbon_positions):
    vec_to_c = c_pos - phe_center
    # 向量与平面法向量的夹角
    dot_prod = np.dot(vec_to_c, phe_normal)
    norm = np.linalg.norm(vec_to_c)
    cos_angle = dot_prod / norm
    angle_rad = np.arccos(np.clip(np.abs(cos_angle), -1, 1))
    angle_deg = np.degrees(angle_rad)
    carbon_angles.append(angle_deg)

carbon_angles = np.array(carbon_angles)

# 计算角度衰减因子
angle_deviations = np.abs(carbon_angles - 90.0)
angle_decay = np.exp(-((angle_deviations / 30.0) ** 2))

print("=" * 70)
print("配体碳原子与Phe389平面的角度分析")
print("=" * 70)
print()
print(f"Phe389 平面法向量: {phe_normal}")
print(f"Phe389 质心位置: {phe_center}")
print()
print("碳原子位置与角度:")
print("-" * 70)
print(f"{'碳原子':<8} {'X (Å)':<10} {'Y (Å)':<10} {'Z (Å)':<10} {'角度':<10} {'偏离90°':<10} {'衰减因子':<10}")
print("-" * 70)

for i, (pos, angle, dev, decay) in enumerate(zip(carbon_positions, carbon_angles, angle_deviations, angle_decay)):
    status = "✓ 完美T型" if dev < 10 else "△ 良好" if dev < 30 else "✗ 偏离大"
    print(f"C{i+1:<7} {pos[0]:>9.2f} {pos[1]:>9.2f} {pos[2]:>9.2f} {angle:>9.1f}° {dev:>9.1f}° {decay:>9.3f}  {status}")

print()
print("=" * 70)
print("角度意义解释:")
print("=" * 70)
print("""
• 0° 或 180°   = 碳原子位于Phe平面内（平行于平面）
  → 无法以T型方式接近，贡献权重很弱 (衰减 ≈ 0.01)

• 45°          = 碳原子与Phe平面成45°角
  → 次优接近方式，贡献权重中等 (衰减 ≈ 0.33)

• 90°          = 碳原子垂直指向/远离Phe平面（完美T型）
  → 最优接近方式，贡献权重最大 (衰减 ≈ 1.0) ✓✓✓

公式：衰减因子 = exp(-(|角度-90°|/30°)²)

例子：
  - 角度 85°   → 偏离 5°    → 衰减 = 0.97 ✓ 非常好
  - 角度 95°   → 偏离 5°    → 衰减 = 0.97 ✓ 非常好
  - 角度 75°   → 偏离 15°   → 衰减 = 0.75 ✓ 不错
  - 角度 60°   → 偏离 30°   → 衰减 = 0.37 △ 可以
  - 角度 45°   → 偏离 45°   → 衰减 = 0.10 ✗ 很差
""")

# 输出角度分布统计
print()
print("=" * 70)
print("统计摘要:")
print("=" * 70)
print(f"平均角度: {np.mean(carbon_angles):.1f}°")
print(f"角度范围: {np.min(carbon_angles):.1f}° ~ {np.max(carbon_angles):.1f}°")
print(f"平均衰减: {np.mean(angle_decay):.3f}")
print(f"最大衰减: {np.max(angle_decay):.3f} (C{np.argmax(angle_decay)+1})")
print(f"最小衰减: {np.min(angle_decay):.3f} (C{np.argmin(angle_decay)+1})")

perfect_count = np.sum(angle_deviations < 10)
good_count = np.sum((angle_deviations >= 10) & (angle_deviations < 30))
bad_count = np.sum(angle_deviations >= 30)

print()
print(f"完美T型 (偏离<10°): {perfect_count}/6 个碳")
print(f"良好接触 (偏离10-30°): {good_count}/6 个碳")
print(f"偏离较大 (偏离>30°): {bad_count}/6 个碳")
print()
print("=" * 70)
