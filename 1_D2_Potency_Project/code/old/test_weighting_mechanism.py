#!/usr/bin/env python3
"""
演示新的三层加权机制：
1. ELF 权重 (w) - 电子云密度
2. 角度衰减 - 与T型接触的偏离程度
3. 距离衰减 - 离Phe环平面的距离
"""

import numpy as np
import matplotlib.pyplot as plt

# 模拟6个碳原子的数据
n_carbons = 6

# 1. ELF权重 (示例值，归一化到0-1)
elf_weights = np.array([0.8, 0.9, 0.7, 0.5, 0.6, 0.85])
print("=" * 60)
print("ELF权重 (电子云密度):")
for i, w in enumerate(elf_weights):
    print(f"  C{i+1}: {w:.3f}")

# 2. 角度衰减因子 (模拟：有的C与T型接触匹配度好，有的差)
angle_deviations = np.array([5, 10, 25, 45, 35, 8])  # 单位：度
angle_decay = np.exp(-((angle_deviations / 30.0) ** 2))
print("\n角度偏离与衰减因子:")
for i, (dev, decay) in enumerate(zip(angle_deviations, angle_decay)):
    print(f"  C{i+1}: 偏离{dev:2d}° -> 衰减={decay:.3f}")

# 3. 距离衰减因子 (模拟：不同C到Phe平面的垂直距离)
perp_distances = np.array([0.5, 1.0, 2.0, 3.5, 2.8, 1.2])  # 单位：Ångström
distance_decay = np.exp(-((perp_distances / 2.0) ** 2))
print("\n垂直距离与衰减因子:")
for i, (dist, decay) in enumerate(zip(perp_distances, distance_decay)):
    print(f"  C{i+1}: {dist:.1f}Å -> 衰减={decay:.3f}")

# 4. 模拟的距离值
distances_to_phe = np.array([4.5, 4.2, 4.8, 5.1, 5.0, 4.3])

# 计算各种加权方式的加权平均距离
print("\n" + "=" * 60)
print("加权平均距离对比:")
print("=" * 60)

# 方法1: 仅ELF权重
avg_elf_only = np.average(distances_to_phe, weights=elf_weights)
print(f"1. ELF权重仅:           {avg_elf_only:.3f} Ångström")

# 方法2: ELF + 角度衰减
combined_2 = elf_weights * angle_decay
avg_elf_angle = np.average(distances_to_phe, weights=combined_2)
print(f"2. ELF + 角度衰减:       {avg_elf_angle:.3f} Ångström")

# 方法3: ELF + 距离衰减 (缺少角度信息)
combined_3 = elf_weights * distance_decay
avg_elf_dist = np.average(distances_to_phe, weights=combined_3)
print(f"3. ELF + 距离衰减:       {avg_elf_dist:.3f} Ångström")

# 方法4: ELF + 角度衰减 + 距离衰减 (完整的三层加权)
combined_4 = elf_weights * angle_decay * distance_decay
avg_all = np.average(distances_to_phe, weights=combined_4)
print(f"4. ELF + 角度 + 距离:    {avg_all:.3f} Ångström ✓ 推荐")

print("\n" + "=" * 60)
print("权重分布分析:")
print("=" * 60)

# 显示各碳原子的最终权重
final_weights = combined_4 / np.sum(combined_4)
print("\n碳原子的最终权重占比:")
for i, w in enumerate(final_weights):
    bar = "█" * int(w * 50)
    print(f"  C{i+1}: {w:.3f} {bar}")

# 找出贡献最大的碳
max_idx = np.argmax(combined_4)
print(f"\n🎯 最大贡献原子: C{max_idx+1}")
print(f"   ELF权重: {elf_weights[max_idx]:.3f}")
print(f"   角度衰减: {angle_decay[max_idx]:.3f} (偏离{angle_deviations[max_idx]}°)")
print(f"   距离衰减: {distance_decay[max_idx]:.3f} (距离{perp_distances[max_idx]:.1f}Å)")
print(f"   到Phe距离: {distances_to_phe[max_idx]:.2f}Å")

# 可视化
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 子图1: ELF权重
ax = axes[0, 0]
ax.bar(range(1, 7), elf_weights, color='steelblue', alpha=0.7, edgecolor='black')
ax.set_ylabel('权重值', fontsize=11)
ax.set_title('(A) ELF电子权重', fontsize=12, fontweight='bold')
ax.set_ylim(0, 1)
for i, v in enumerate(elf_weights):
    ax.text(i+1, v+0.05, f'{v:.2f}', ha='center', fontsize=10)

# 子图2: 角度衰减
ax = axes[0, 1]
colors = ['green' if d < 15 else 'orange' if d < 30 else 'red' for d in angle_deviations]
ax.bar(range(1, 7), angle_decay, color=colors, alpha=0.7, edgecolor='black')
ax.set_ylabel('衰减因子', fontsize=11)
ax.set_title('(B) 角度衰减 (T型接触)', fontsize=12, fontweight='bold')
ax.set_ylim(0, 1)
for i, (v, dev) in enumerate(zip(angle_decay, angle_deviations)):
    ax.text(i+1, v+0.05, f'{v:.2f}\n({dev}°)', ha='center', fontsize=9)

# 子图3: 距离衰减
ax = axes[1, 0]
colors = ['green' if d < 1.2 else 'orange' if d < 2.0 else 'red' for d in perp_distances]
ax.bar(range(1, 7), distance_decay, color=colors, alpha=0.7, edgecolor='black')
ax.set_ylabel('衰减因子', fontsize=11)
ax.set_title('(C) 距离衰减 (到平面的垂直距离)', fontsize=12, fontweight='bold')
ax.set_ylim(0, 1)
for i, (v, d) in enumerate(zip(distance_decay, perp_distances)):
    ax.text(i+1, v+0.05, f'{v:.2f}\n({d:.1f}Å)', ha='center', fontsize=9)

# 子图4: 最终组合权重
ax = axes[1, 1]
final_weights_display = combined_4 / np.sum(combined_4)
colors_final = plt.cm.RdYlGn(final_weights_display / final_weights_display.max())
bars = ax.bar(range(1, 7), final_weights_display, color=colors_final, alpha=0.7, edgecolor='black')
ax.set_ylabel('权重占比', fontsize=11)
ax.set_title('(D) 最终组合权重 (ELF×角度×距离)', fontsize=12, fontweight='bold')
ax.set_ylim(0, max(final_weights_display) * 1.2)
for i, v in enumerate(final_weights_display):
    ax.text(i+1, v+0.02, f'{v:.3f}', ha='center', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig('/home/hongyu/MD/1_partial/weighting_mechanism_demo.png', dpi=150, bbox_inches='tight')
print(f"\n📊 可视化已保存: weighting_mechanism_demo.png")

# 统计总结
print("\n" + "=" * 60)
print("三层加权机制的意义:")
print("=" * 60)
print("""
1️⃣  ELF权重: 反映电子云密度分布
   - 权重大 = 电子云密度高 = 与配体的π-π相互作用强

2️⃣  角度衰减: 与T型堆积(90°)的匹配程度
   - 完美T型(偏离<10°) -> 衰减≈1.0 ✓
   - 偏离30° -> 衰减≈0.37
   - 偏离45° -> 衰减≈0.03

3️⃣  距离衰减: 离Phe芳环平面的接近程度
   - 接近平面(<1Å) -> 衰减≈1.0 ✓
   - 距离2Å -> 衰减≈0.37
   - 距离3.5Å -> 衰减≈0.01

🎯 结果: 综合ELF强度、几何匹配度和空间接近度
   得到最有代表性的"加权平均距离"
""")
