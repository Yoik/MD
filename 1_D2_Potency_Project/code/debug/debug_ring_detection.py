#!/usr/bin/env python3
"""
debug_ring_detection.py
调试环检测逻辑
"""

import numpy as np
from scipy.spatial.distance import cdist
import itertools

# 吲哚坐标 - 正确的融合环结构
# 吲哚 = 苯并吡咯
# 结构：
#     4 - 5
#    /   \
#   3     6(N)
#    \   /
#     2-1-7
#       \
#        8

indole_coords = np.array([
    # 苯环部分 - 碳原子 (标准键长 ~1.4 Å)
    [0.0, 0.0, 0.0],       # C1 (索引0) - 苯环和吡咯的共享点
    [1.4, 0.0, 0.0],       # C2 (索引1)
    [2.1, 1.21, 0.0],      # C3 (索引2)
    [1.4, 2.42, 0.0],      # C4 (索引3)
    [0.0, 2.42, 0.0],      # C5 (索引4)
    [-0.7, 1.21, 0.0],     # C6 (索引5) - 苯环和吡咯的共享点
    # 吡咯环部分
    [-1.4, 0.0, 0.0],      # N (索引6) - 连接C6和C7
    [0.0, -1.4, 0.0],      # C7 (索引7) - 吡咯的第二个碳，连接C1
])
indole_elements = ['C', 'C', 'C', 'C', 'C', 'C', 'N', 'C']

print("吲哚结构（简化）：")
print("  4-5")
print(" /   \\")
print("3  6N")
print(" \\ / \\")
print("  2-1-7")
print("  (  \\")
print("   8 )")
print()

print("=" * 60)
print("调试吲哚结构的环检测")
print("=" * 60)

# 计算所有原子对之间的距离
dmat = cdist(indole_coords, indole_coords)
print("\n距离矩阵 (键长 1.1-1.7 Å 为有效键):")
print("    ", end="")
for i in range(len(indole_elements)):
    print(f"{indole_elements[i]:>5}", end="")
print()

for i in range(len(indole_elements)):
    print(f"{indole_elements[i]:>3}:", end="")
    for j in range(len(indole_elements)):
        d = dmat[i, j]
        if 1.1 <= d <= 1.7:
            print(f"{d:>5.2f}*", end="")
        else:
            print(f"{d:>5.2f}", end="")
    print()

# 找 6 元环
print("\n" + "=" * 60)
print("查找 6 元环（仅碳原子）")
print("=" * 60)

c_indices = [i for i, e in enumerate(indole_elements) if e == 'C']
print(f"碳原子索引: {c_indices}")

if len(c_indices) >= 6:
    c_coords = indole_coords[c_indices]
    c_dmat = cdist(c_coords, c_coords)
    c_adj = np.logical_and(c_dmat > 1.1, c_dmat < 1.7)
    
    found_6rings = 0
    for comb in itertools.combinations(range(len(c_indices)), 6):
        sub_idx = list(comb)
        curr_coords = c_coords[sub_idx]
        
        # 检查共平面性
        centered = curr_coords - curr_coords.mean(0)
        _, s, _ = np.linalg.svd(centered)
        
        # 检查连接性
        sub_adj = c_adj[np.ix_(sub_idx, sub_idx)]
        conn_ok = np.all(np.sum(sub_adj, axis=1) >= 2)
        
        if s[2] <= 0.3 and conn_ok:
            found_6rings += 1
            actual_indices = [c_indices[i] for i in sub_idx]
            print(f"  ✓ 找到 6 元环: {actual_indices} (SVD={s[2]:.3f}, 平面性OK)")

print(f"总共找到 {found_6rings} 个 6 元环")

# 找 5 元环
print("\n" + "=" * 60)
print("查找 5 元环（含杂原子N/O）")
print("=" * 60)

found_5rings = []
for comb in itertools.combinations(range(len(indole_coords)), 5):
    ring_idx = list(comb)
    ring_coords = indole_coords[ring_idx]
    ring_elems = [indole_elements[i] for i in ring_idx]
    
    # 必须包含N或O
    has_hetero = 'N' in ring_elems or 'O' in ring_elems
    if not has_hetero:
        continue
    
    # 构建邻接矩阵
    sub_dmat = cdist(ring_coords, ring_coords)
    sub_adj = np.logical_and(sub_dmat > 1.1, sub_dmat < 1.7)
    
    # 检查连接性
    conn_ok = np.all(np.sum(sub_adj, axis=1) == 2)
    if not conn_ok:
        continue
    
    # 检查共平面性
    centered = ring_coords - ring_coords.mean(0)
    _, s, _ = np.linalg.svd(centered)
    
    if s[2] <= 0.3:
        found_5rings.append({
            'indices': ring_idx,
            'elements': ring_elems,
            'svd': s[2]
        })
        print(f"  ✓ 找到 5 元环: {ring_idx} ({ring_elems}) (SVD={s[2]:.3f})")

print(f"总共找到 {len(found_5rings)} 个 5 元环")

# 检查融合关系
print("\n" + "=" * 60)
print("检查融合关系（6+5 共享原子）")
print("=" * 60)

if found_6rings > 0 and found_5rings:
    c_indices = [i for i, e in enumerate(indole_elements) if e == 'C']
    c_coords = indole_coords[c_indices]
    c_dmat = cdist(c_coords, c_coords)
    c_adj = np.logical_and(c_dmat > 1.1, c_dmat < 1.7)
    
    for comb in itertools.combinations(range(len(c_indices)), 6):
        sub_idx = list(comb)
        curr_coords = c_coords[sub_idx]
        
        centered = curr_coords - curr_coords.mean(0)
        _, s, _ = np.linalg.svd(centered)
        
        sub_adj = c_adj[np.ix_(sub_idx, sub_idx)]
        conn_ok = np.all(np.sum(sub_adj, axis=1) >= 2)
        
        if s[2] <= 0.3 and conn_ok:
            six_ring_global = [c_indices[i] for i in sub_idx]
            six_set = set(six_ring_global)
            
            for five_ring_info in found_5rings:
                five_set = set(five_ring_info['indices'])
                shared = six_set & five_set
                
                if len(shared) >= 2:
                    print(f"\n  ✓ 找到融合结构！")
                    print(f"    6元环: {six_ring_global}")
                    print(f"    5元环: {five_ring_info['indices']} ({five_ring_info['elements']})")
                    print(f"    共享原子: {list(shared)} ({[indole_elements[i] for i in shared]})")
