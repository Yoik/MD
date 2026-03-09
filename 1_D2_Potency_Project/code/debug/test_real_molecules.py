#!/usr/bin/env python3
"""
test_real_molecules.py
使用真实的 PDB 和 CUB 文件测试环检测
"""

import numpy as np
from modules import RingMatcher
import glob
import os

def get_ref_data_from_pdb(pdb_file):
    """从 PDB 文件中提取参考分子的原子坐标和元素"""
    coords = []
    elements = []
    try:
        with open(pdb_file) as f:
            for line in f:
                if line.startswith("ATOM") or line.startswith("HETATM"):
                    # PDB 格式的原子坐标和元素
                    atom_name = line[12:16].strip()
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                    
                    # 提取元素符号（通常在原子名的第一个字符）
                    # 但要处理特殊情况（如 CL, CA 等）
                    element = atom_name[0]
                    if len(atom_name) > 1 and atom_name[1].isalpha():
                        element = atom_name[0:2].strip()
                    
                    # 只关心配体的原子（非蛋白质、非水）
                    # 简单起见，只提取 C, N, O, S, H 等小分子常见元素
                    if element in ['C', 'N', 'O', 'S', 'F', 'Cl']:
                        coords.append([x, y, z])
                        elements.append(element)
    except Exception as e:
        print(f"  Error reading PDB: {e}")
        return None, None
    
    if coords:
        return np.array(coords), elements
    return None, None

def test_molecule(data_dir):
    """测试单个分子目录"""
    compound_name = os.path.basename(data_dir)
    print(f"\n{'='*60}")
    print(f"测试: {compound_name}")
    print(f"{'='*60}")
    
    # 查找同一级目录下的 PDB 文件（不在 charmm-gui 等子目录中）
    pdb_files = []
    for file in os.listdir(data_dir):
        if file.endswith('.pdb') and os.path.isfile(os.path.join(data_dir, file)):
            pdb_files.append(os.path.join(data_dir, file))
    
    if not pdb_files:
        print(f"✗ 未找到同级 PDB 文件")
        return False
    
    pdb_file = pdb_files[0]  # 取第一个
    print(f"使用 PDB: {os.path.basename(pdb_file)}")
    
    # 提取坐标
    coords, elements = get_ref_data_from_pdb(pdb_file)
    
    if coords is None or len(coords) < 5:
        print(f"✗ 无法提取有效坐标 (extracted {len(coords) if coords is not None else 0} atoms)")
        return False
    
    print(f"提取原子数: {len(coords)}")
    print(f"元素组成: {dict((e, elements.count(e)) for e in set(elements))}")
    
    # 尝试匹配环
    try:
        rm = RingMatcher(coords, elements)
        print(f"✓ 环检测成功！")
        print(f"  环类型: {rm.ring_type}")
        print(f"  环大小: {rm.rings[0]['size']}")
        
        if 'six_ring' in rm.rings[0]:
            print(f"  检测到融合环结构:")
            print(f"    - 6元环大小: {len(rm.rings[0]['six_ring'])}")
            print(f"    - 5元环大小: {len(rm.rings[0]['five_ring'])}")
            print(f"    - 共享原子数: {len(rm.rings[0]['shared_atoms'])}")
        
        return True
    
    except ValueError as e:
        print(f"✗ 环检测失败: {e}")
        return False
    except Exception as e:
        print(f"✗ 错误: {e}")
        return False


# 主程序
print("=" * 60)
print("实际分子环检测测试")
print("=" * 60)

root = "/home/hongyu/MD/1_D2_Potency_Project"

# 测试所有带有 CUB 文件的目录
cub_files = glob.glob(os.path.join(root, "*_*", "*.cub"))
tested_dirs = set()
success_count = 0
fail_count = 0

for cub_file in sorted(cub_files):
    data_dir = os.path.dirname(cub_file)
    
    # 避免重复测试
    if data_dir in tested_dirs:
        continue
    tested_dirs.add(data_dir)
    
    if test_molecule(data_dir):
        success_count += 1
    else:
        fail_count += 1

print(f"\n{'='*60}")
print(f"测试总结")
print(f"{'='*60}")
print(f"成功: {success_count}")
print(f"失败: {fail_count}")
print(f"总计: {success_count + fail_count}")
