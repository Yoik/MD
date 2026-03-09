#!/usr/bin/env python3
"""
test_ring_detection.py
测试新的环检测逻辑：优先检测融合环（吲哚/呋喃），否则检测苯环
"""

import numpy as np
from modules import RingMatcher

# 示例1：苯环
print("=" * 60)
print("测试 1: 纯苯环（C6H6）")
print("=" * 60)

benzene_coords = np.array([
    [1.0, 0.0, 0.0],      # C1
    [0.5, 0.866, 0.0],    # C2
    [-0.5, 0.866, 0.0],   # C3
    [-1.0, 0.0, 0.0],     # C4
    [-0.5, -0.866, 0.0],  # C5
    [0.5, -0.866, 0.0]    # C6
])
benzene_elements = ['C', 'C', 'C', 'C', 'C', 'C']

try:
    rm_benzene = RingMatcher(benzene_coords, benzene_elements)
    print(f"✓ 检测到环类型: {rm_benzene.ring_type}")
    print(f"  环大小: {rm_benzene.rings[0]['size']}")
    print()
except Exception as e:
    print(f"✗ 错误: {e}")
    print()

# 示例2：吲哚（6元苯环 + 5元吡咯共享2个原子）
print("=" * 60)
print("测试 2: 吲哚（融合6元+5元环，含N）")
print("=" * 60)

# 实际吲哚骨架坐标（更准确的键长）
indole_coords = np.array([
    # 苯环部分 (6个原子，标准苯环键长 1.4 Å)
    [0.0, 0.0, 0.0],      # C1
    [1.4, 0.0, 0.0],      # C2
    [2.1, 1.21, 0.0],     # C3
    [1.4, 2.42, 0.0],     # C4
    [0.0, 2.42, 0.0],     # C5
    [-0.7, 1.21, 0.0],    # C6
    # 吡咯环部分 (5个原子，共享C1和C6)
    # N连接到C1和C6
    [-1.4, 0.0, 0.0],     # N (连接C1和后面的C7)
    [-1.4, -1.3, 0.0],    # C7 (连接N和C8)
    [-0.1, -1.8, 0.0],    # C8 (连接C7和C1)
])
indole_elements = ['C', 'C', 'C', 'C', 'C', 'C', 'N', 'C', 'C']

try:
    rm_indole = RingMatcher(indole_coords, indole_elements)
    print(f"✓ 检测到环类型: {rm_indole.ring_type}")
    print(f"  环大小: {rm_indole.rings[0]['size']}")
    if 'six_ring' in rm_indole.rings[0]:
        print(f"  6元环原子数: {len(rm_indole.rings[0]['six_ring'])}")
        print(f"  5元环原子数: {len(rm_indole.rings[0]['five_ring'])}")
        print(f"  共享原子数: {len(rm_indole.rings[0]['shared_atoms'])}")
    print()
except Exception as e:
    print(f"✗ 错误: {e}")
    print()

# 示例3：呋喃（6元 + 5元含O）
print("=" * 60)
print("测试 3: 呋喃类结构（融合6元+5元环，含O）")
print("=" * 60)

furan_coords = np.array([
    # 苯环部分 (6个原子)
    [0.0, 0.0, 0.0],      # C1
    [1.4, 0.0, 0.0],      # C2
    [2.1, 1.21, 0.0],     # C3
    [1.4, 2.42, 0.0],     # C4
    [0.0, 2.42, 0.0],     # C5
    [-0.7, 1.21, 0.0],    # C6
    # 呋喃环部分 (5个原子，包括O，共享C1和C6)
    [-1.4, 0.0, 0.0],     # O
    [-1.4, -1.3, 0.0],    # C7
    [-0.1, -1.8, 0.0],    # C8
])
furan_elements = ['C', 'C', 'C', 'C', 'C', 'C', 'O', 'C', 'C']

try:
    rm_furan = RingMatcher(furan_coords, furan_elements)
    print(f"✓ 检测到环类型: {rm_furan.ring_type}")
    print(f"  环大小: {rm_furan.rings[0]['size']}")
    if 'six_ring' in rm_furan.rings[0]:
        print(f"  6元环原子数: {len(rm_furan.rings[0]['six_ring'])}")
        print(f"  5元环原子数: {len(rm_furan.rings[0]['five_ring'])}")
        print(f"  共享原子数: {len(rm_furan.rings[0]['shared_atoms'])}")
    print()
except Exception as e:
    print(f"✗ 错误: {e}")
    print()

print("=" * 60)
print("测试完成！")
print("=" * 60)
print("\n新逻辑优先级:")
print("1. 优先检测融合环（吲哚/呋喃）- 6元+5元共享结构")
print("2. 如果没有融合环，检测单独的苯环（6元）")
print("3. 如果都没有找到，返回空列表并抛出错误")
