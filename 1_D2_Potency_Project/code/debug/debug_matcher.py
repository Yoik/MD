#!/usr/bin/env python3
"""
debug_matcher.py
专门用于调试 RingMatcher 匹配失败的原因
"""

import MDAnalysis as mda
import numpy as np
import os
import glob
from scipy.spatial.distance import cdist
from modules.ring_matcher import RingMatcher

# === 这里填入你想要调试的化合物 ID ===
TARGET_COMPOUNDS = [
    "20251122_D2_S10_cryoEM_rebuild",
    "20251202_D2_S84_boltz",
    "20251203_D2_R10_cryoEM_rebuild"
]

def get_ref_data(root_dir, cid):
    """读取 PDB 参考文件"""
    c_dir = os.path.join(root_dir, cid)
    pdb = glob.glob(os.path.join(c_dir, "*.pdb"))
    if not pdb: return None, None
    # 找到 reference PDB (排除 step7 和 topol)
    ref_pdb = next((p for p in pdb if "step7" not in p and "topol" not in p), pdb[0])
    
    print(f"\n[Ref] Loading Reference PDB: {os.path.basename(ref_pdb)}")
    
    c = []; e = []
    with open(ref_pdb) as f:
        for l in f:
            if l.startswith("ATOM") or l.startswith("HETATM"):
                name = l[12:16].strip()
                # 仅保留重原子逻辑
                if name[0] in ['C', 'N', 'O', 'S'] and name not in ['CA', 'C', 'N', 'O']:
                    c.append([float(l[30:38]), float(l[38:46]), float(l[46:54])])
                    e.append(name[0])
    return np.array(c), e

def debug_fingerprint(rm, md_atoms, candidate_indices):
    """
    深度调试指纹匹配过程，打印详细的邻居信息
    """
    print("\n  >>> 🔍 正在调试指纹匹配 (Fingerprint Debug) ...")
    
    # 1. 参考环信息
    ref_indices = rm.ref_ring_idx
    ref_coords = rm.ref_coords[ref_indices]
    ref_dmat = cdist(rm.ref_coords, rm.ref_coords)
    
    print(f"  [Ref Info] Ring Elements: {rm.ring_elements}")
    print("  [Ref Neighbors] (Threshold < 2.0 Å):")
    
    ref_n_counts = []
    for i, r_idx in enumerate(ref_indices):
        # 计算参考环的邻居
        neighbors = [n for n in range(rm.n_ref) 
                     if n not in ref_indices and ref_dmat[r_idx, n] < 2.0]
        ref_n_counts.append(len(neighbors))
        print(f"    Ref Atom {i} ({rm.ring_elements[i]}): {len(neighbors)} neighbors (Indices: {neighbors})")

    # 2. MD 候选环信息
    cand_atoms = md_atoms[candidate_indices]
    full_atoms = md_atoms # 整个配体
    cand_global_ids = [a.index for a in cand_atoms]
    cand_coords = cand_atoms.positions
    
    print(f"\n  [MD Candidate] Indices: {candidate_indices}")
    print(f"  [MD Neighbors] (Threshold < 2.2 Å, Excluding H):")
    
    dmat_md = cdist(cand_coords, full_atoms.positions)
    
    md_n_counts = []
    
    for i in range(len(cand_atoms)):
        atom = cand_atoms[i]
        # 找邻居
        nearby = np.where(dmat_md[i] < 2.2)[0]
        
        valid_neighbors = []
        ignored_neighbors = []
        
        for n_idx in nearby:
            n_atom = full_atoms[n_idx]
            if n_atom.index in cand_global_ids: continue # 排除环内
            
            # 这里的过滤逻辑要和你的 ring_matcher.py 完全一致
            name_upper = n_atom.name.upper()
            is_h = name_upper.startswith('H') or (name_upper[0].isdigit() and 'H' in name_upper)
            is_heavy = name_upper[0] in ['C', 'N', 'O', 'S', 'F', 'P', 'I', 'B', 'Cl']
            
            if is_h or not is_heavy:
                ignored_neighbors.append(n_atom.name)
                continue
            
            valid_neighbors.append(n_atom.name)
            
        md_n_counts.append(len(valid_neighbors))
        print(f"    MD Atom {i} ({atom.name}): {len(valid_neighbors)} neighbors. "
              f"Valid: {valid_neighbors} | Ignored: {ignored_neighbors}")

    # 3. 尝试匹配并打印分数
    print("\n  [Matching Attempt] Comparing Ref vs MD:")
    
    matched_pairs = []
    
    for i in range(len(ref_indices)):
        r_e = rm.ring_elements[i]
        r_count = ref_n_counts[i]
        
        print(f"    Target Ref {i} ({r_e}, Neigh={r_count}) matches:")
        
        best_match = None
        min_penalty = float('inf')
        
        for j in range(len(cand_atoms)):
            md_e = cand_atoms[j].name[0]
            md_count = md_n_counts[j]
            
            # 元素检查
            if md_e != r_e: continue
            
            # 邻居计数检查
            diff = abs(md_count - r_count)
            penalty = 5.0 if diff > 0 else 0.0
            
            status = "✅" if penalty == 0 else "❌ Penalty!"
            print(f"      -> MD Atom {j} ({cand_atoms[j].name}, Neigh={md_count}): Diff={diff} | {status}")
            
            if penalty < min_penalty:
                min_penalty = penalty
                best_match = j
                
        matched_pairs.append(best_match)

    print(f"\n  [Result] Final Mapping Indices: {matched_pairs}")
    if len(set(matched_pairs)) != len(ref_indices):
        print("  ❌ 匹配失败！存在重复映射或未找到映射。")
    else:
        print("  ✅ 匹配理论上应该成功。")

def main():
    root_dir = "."
    
    for cid in TARGET_COMPOUNDS:
        c_dir = os.path.join(root_dir, cid)
        if not os.path.isdir(c_dir):
            print(f"Skipping {cid} (not found)")
            continue
            
        print(f"\n{'='*60}")
        print(f"DEBUGGING: {cid}")
        print(f"{'='*60}")
        
        # 1. 准备 RingMatcher
        ref_c, ref_e = get_ref_data(root_dir, cid)
        if ref_c is None:
            print("No Ref Data found.")
            continue
            
        try:
            rm = RingMatcher(ref_c, ref_e)
            print(f"[Ref Ring Detected] Type: {rm.ring_type}, Size: {len(rm.ref_ring_idx)}")
        except Exception as e:
            print(f"RingMatcher Init Failed: {e}")
            continue

        # 2. 准备 MD 数据 (只读第一个 xtc 的第一帧)
        xtcs = glob.glob(os.path.join(c_dir, "**", "merged.xtc"), recursive=True)
        if not xtcs: continue
        xtc = xtcs[0]
        rd = os.path.dirname(xtc)
        tps = [os.path.join(rd, f) for f in os.listdir(rd) if f.endswith(".tpr")]
        topo = tps[0]
        
        print(f"[MD] Loading: {os.path.basename(topo)} / {os.path.basename(xtc)}")
        u = mda.Universe(topo, xtc)
        lig = u.select_atoms("resname LIG LIG1 LDP R5F DRG UNK")
        if not lig:
            lig = u.atoms[0].residue.atoms # Fallback
            
        print(f"[MD] Ligand Atoms: {len(lig)}")
        
        # 3. 手动运行查找逻辑 (模拟 _match_fused_system 的前半部分)
        # 这里为了简单，我们直接用 RingMatcher 内部的逻辑找候选环
        
        # 重新实现简化的找环逻辑以获取 candidate_indices
        md_atoms = lig
        md_coords = md_atoms.positions
        heavy_mask = [a.name[0] in ['C', 'N', 'O', 'S'] for a in md_atoms]
        heavy_indices_local = [i for i, x in enumerate(heavy_mask) if x]
        heavy_coords = md_coords[heavy_indices_local]
        
        dmat = cdist(heavy_coords, heavy_coords)
        # 使用你修改后的宽阈值 1.9
        adj = np.logical_and(dmat > 1.1, dmat < 1.9)
        
        def find_rings_local(target_len):
            found = []
            seen = set()
            def dfs(s, c, p):
                if len(p) == target_len: return p if adj[c, s] else None
                for n in np.where(adj[c])[0]:
                    if n == s and len(p) < target_len - 1: continue
                    if n not in p:
                        r = dfs(s, n, p + [n])
                        if r: return r
                return None
            for i in range(len(heavy_indices_local)):
                if np.sum(adj[i]) >= 2:
                    res = dfs(i, i, [i])
                    if res:
                        s = tuple(sorted(res))
                        if s not in seen: found.append(set(res)); seen.add(s)
            return found

        rings_6 = find_rings_local(6)
        rings_5 = find_rings_local(5)
        
        print(f"[MD Topology] Found {len(rings_6)} 6-rings, {len(rings_5)} 5-rings.")
        
        candidate_found = False
        for r6 in rings_6:
            for r5 in rings_5:
                shared = r6.intersection(r5)
                if len(shared) == 2:
                    fused_set = r6.union(r5)
                    if len(fused_set) == 9:
                        print("\n[Candidate Found!] Found a 6+5 fused system.")
                        candidate_indices_local = list(fused_set)
                        # 映射回全局索引
                        real_indices = [heavy_indices_local[i] for i in candidate_indices_local]
                        
                        # === 调用诊断函数 ===
                        debug_fingerprint(rm, lig, real_indices)
                        candidate_found = True
        
        if not candidate_found:
            print("❌ 在 MD 中未找到任何符合 6+5 拓扑的环结构！(可能是键长阈值 1.9 还是不够？)")

if __name__ == "__main__":
    main()