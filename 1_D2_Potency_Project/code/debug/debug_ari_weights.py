#!/usr/bin/env python3
"""
debug_ari_weights_v3.py
最终版诊断脚本：
1. 集成 Voronoi/Bond Exclusion (通过调用新的 CubeParser)
2. 集成 PDB-Cube 原子顺序强制对齐 (复刻主脚本的修复逻辑)
3. 显示最终的权重分配是否合理
"""

import MDAnalysis as mda
import numpy as np
import os
import glob
from scipy.spatial.distance import cdist
from modules.ring_matcher import RingMatcher
from modules.cube_parser import CubeParser
from modules.geometry import get_aromatic_ring_data

# === 配置 ===
TARGET_COMPOUND = "ARI"
OFFSET = -33
PHE389_RESID = 389
INTEGRATION_RADIUS = 1.5 # 保持与主脚本一致
BOHR_TO_ANGSTROM = 0.52917721067

def get_ref_data_from_pdb(pdb_file):
    c = []; e = []; names = []
    with open(pdb_file) as f:
        for l in f:
            if l.startswith("ATOM") or l.startswith("HETATM"):
                n = l[12:16].strip()
                # 提取所有重原子用于环检测
                if n[0] in ['C', 'N', 'O', 'S']:
                    if n not in ['CA', 'C', 'N', 'O']: # 排除骨架
                        c.append([float(l[30:38]),float(l[38:46]),float(l[46:54])])
                        e.append(n[0])
    
    # 为了显示名字，再读一遍（简化处理）
    names_full = []
    c_full = []
    with open(pdb_file) as f:
        for l in f:
            if l.startswith("ATOM") or l.startswith("HETATM"):
                n = l[12:16].strip()
                if n[0] in ['C', 'N', 'O', 'S'] and n not in ['CA', 'C', 'N', 'O']:
                    names_full.append(n)
                    c_full.append([float(l[30:38]),float(l[38:46]),float(l[46:54])])
    
    return np.array(c), e, names_full, np.array(c_full)

def find_substituents(atom, all_atoms, ring_atom_indices):
    """查找取代基 (复用 V2 逻辑)"""
    dists = cdist([atom.position], all_atoms.positions)[0]
    neighbor_indices = np.where(dists < 2.05)[0]
    
    subs = []
    for idx in neighbor_indices:
        neighbor = all_atoms[idx]
        if neighbor.index == atom.index: continue
        if neighbor.index in ring_atom_indices: continue
        if neighbor.name.startswith('H') or neighbor.name[0].isdigit(): continue
        
        element = neighbor.name.strip('0123456789')
        if element.upper().startswith('CL'): element = 'Cl'
        elif element.upper().startswith('BR'): element = 'Br'
        else: element = element[0]
        subs.append(f"{element}")
        
    return "+".join(subs) if subs else "-"

def main():
    root = "."
    print(f"\n🔍 DEBUGGING WEIGHTS (V3 - Final Check) FOR: {TARGET_COMPOUND}\n" + "="*80)

    # 1. 查找路径
    all_dirs = glob.glob(os.path.join(root, "*"))
    c_dir = next((d for d in all_dirs if TARGET_COMPOUND.lower() in os.path.basename(d).lower() and os.path.isdir(d)), None)
    if not c_dir: print("❌ Folder not found"); return
    
    pdb = glob.glob(os.path.join(c_dir, "*.pdb"))[0]
    cubs = glob.glob(os.path.join(c_dir, "*.cub"))[0]
    xtc = glob.glob(os.path.join(c_dir, "**", "merged.xtc"), recursive=True)[0]
    rd = os.path.dirname(xtc)
    topo = [os.path.join(rd, f) for f in os.listdir(rd) if f.endswith(".tpr")][0]

    # 2. 读取数据
    print(f"📘 PDB: {os.path.basename(pdb)}")
    print(f"🧊 Cube: {os.path.basename(cubs)}")
    
    # ref_c: 仅坐标, ref_e: 元素, ref_names: 原子名, ref_c_full: 对应names的坐标
    ref_c, ref_e, ref_names, ref_c_full = get_ref_data_from_pdb(pdb)
    
    cp = CubeParser(cubs)
    raw_int = cp.get_carbon_integrals(INTEGRATION_RADIUS) # 这里会自动用新的 Voronoi 逻辑
    
    # =========================================================================
    # 【注入修复逻辑】 原子顺序对齐
    # =========================================================================
    print("-" * 80)
    print("🛠️  Applying Atom Order Alignment (PDB vs Cube)...")
    
    # 1. 获取 PDB 碳原子
    pdb_c_indices = [i for i, e in enumerate(ref_e) if e == 'C']
    pdb_c_coords = ref_c[pdb_c_indices]
    
    # 2. 获取 Cube 碳原子
    cube_c_coords = []
    if hasattr(cp, 'atoms'):
        for atom in cp.atoms:
            if atom['Z'] == 6:
                pos = atom['pos_bohr'] * BOHR_TO_ANGSTROM if cp.is_header_bohr else atom['pos_bohr']
                cube_c_coords.append(pos)
    
    # 3. 匹配
    if len(pdb_c_coords) == len(cube_c_coords) and len(pdb_c_coords) > 0:
        dmat = cdist(pdb_c_coords, cube_c_coords)
        mapping = np.argmin(dmat, axis=1)
        min_dists = np.min(dmat, axis=1)
        max_err = np.max(min_dists)
        
        if max_err < 0.5:
            # 重排！
            raw_int_aligned = raw_int[mapping]
            
            # 检查是否真的发生了重排
            if not np.array_equal(mapping, np.arange(len(mapping))):
                print(f"✅  FIX APPLIED: Reordered Cube data to match PDB (Max error: {max_err:.4f} A)")
                print(f"    Mapping: {mapping}")
            else:
                print(f"ℹ️  Order was already correct (Max error: {max_err:.4f} A)")
            
            raw_int = raw_int_aligned
        else:
            print(f"⚠️  ALIGNMENT FAILED: Max distance {max_err:.2f}A is too large!")
    else:
        print(f"⚠️  COUNT MISMATCH: PDB Carbons ({len(pdb_c_coords)}) != Cube Carbons ({len(cube_c_coords)})")
    print("-" * 80)
    # =========================================================================

    # 3. 运行 RingMatcher
    u = mda.Universe(topo, xtc)
    target_resid = PHE389_RESID + OFFSET
    phe389 = u.select_atoms(f"resid {target_resid}")
    phe_center, _ = get_aromatic_ring_data(phe389)
    if phe_center is None: phe_center = phe389.center_of_mass()
    
    lig_res = u.select_atoms("resname LIG LIG1 LDP R5F DRG UNK").residues[0]
    
    # 初始化 RingMatcher
    rm = RingMatcher(ref_c, ref_e)
    
    u.trajectory[0]
    matched_atoms, cube_idxs, _ = rm.match(lig_res.atoms, phe_center)
    
    if matched_atoms is None: print("❌ Match Failed"); return

    # 4. 打印结果
    print(f"\n📊 Final Weights Check (Voronoi + Alignment):")
    print(f"{'Ref Name':<10} | {'Sim Name':<10} | {'Substituents':<15} | {'ELF Wgt':<10} | {'Dist Phe389':<12}")
    print("-" * 80)

    # 获取 PDB 中对应环原子的名称
    # rm.ref_ring_idx 是 ref_c 中的索引
    # 我们需要从 ref_names 找到对应的名字
    # ref_c 和 ref_names 是对齐的 (get_ref_data_from_pdb 保证)
    
    ring_global_indices = [a.index for a in matched_atoms]
    dists = np.linalg.norm(matched_atoms.positions - phe_center, axis=1)
    
    # 使用对齐后的 raw_int 获取权重
    weights = raw_int[cube_idxs]
    max_w = np.max(raw_int) # 全局最大值

    for i in range(len(matched_atoms)):
        ref_idx = rm.ref_ring_idx[i]
        r_name = ref_names[ref_idx]
        
        s_atom = matched_atoms[i]
        s_name = s_atom.name
        subs_str = find_substituents(s_atom, lig_res.atoms, ring_global_indices)
        
        w_val = weights[i]
        w_norm = w_val / max_w
        d_val = dists[i]
        
        highlight = ""
        if "Cl" in subs_str: highlight = " <-- Has Cl"
        
        print(f"{r_name:<10} | {s_name:<10} | {subs_str:<15} | {w_norm:.3f}      | {d_val:.2f} Å      {highlight}")

    print("-" * 80)
    print("验证标准：")
    print("1. [Order Fix] 上方应显示 'FIX APPLIED' 或 'Order was already correct'。")
    print("2. [Voronoi Fix] 连着 Cl 的碳 (Has Cl) 权重不应再是全场最高 (1.000)。")
    print("   如果它们降到了 0.4-0.6 左右，且其他位置有更高值，说明修复完美！")

if __name__ == "__main__":
    main()