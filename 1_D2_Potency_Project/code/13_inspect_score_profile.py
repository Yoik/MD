#!/usr/bin/env python3
import matplotlib
matplotlib.use('Agg')
import MDAnalysis as mda
import numpy as np
import glob
import os
import sys
import pandas as pd
from tqdm import tqdm

# ================= 复用现有模块 =================
try:
    from modules.qm_loader import find_ligand
    from modules.sequence_aligner import OffsetCalculator
    from modules import (
        get_aromatic_ring_data,
        calculate_carbon_angles_and_decay,
        calculate_distance_decay,
        calculate_combined_weight
    )
    from src.config import init_config
except ImportError as e:
    print(f"Error: 请确保在项目根目录下运行。\n{e}")
    sys.exit(1)

# ================= 配置 =================
config = init_config()
SCORE_TARGET_BW = "6.51" # 使用 6.51 的得分
CONTACT_CUTOFF = 5.0 

def analyze_profile(c_dir, cid, aligner):
    # 寻找轨迹
    xtcs = glob.glob(os.path.join(c_dir, "**", "merged.xtc"), recursive=True)
    if not xtcs: return None

    # 存储所有帧的 Top 5 得分
    all_top_scores = []
    # 存储所有帧的 有效原子数量 (Score > 0.1)
    all_atom_counts = []

    for xtc in xtcs:
        rd = os.path.dirname(xtc)
        tps = [f for f in os.listdir(rd) if f.endswith(".tpr") and "production" in f]
        if not tps: tps = [f for f in os.listdir(rd) if f.endswith(".tpr")]
        if not tps: continue
        
        try:
            u = mda.Universe(os.path.join(rd, tps[0]), xtc)
        except: continue

        real_score_id = aligner.get_real_residue_ids(u, [SCORE_TARGET_BW])
        if not real_score_id: continue
        res_score = u.select_atoms(f"resid {real_score_id[0]}")
        
        ligand = find_ligand(u)
        if not ligand: continue
        
        base_w = np.ones(len(ligand.atoms))

        stride = 10 
        for ts in u.trajectory[::stride]:
            c_ring, n_ring = get_aromatic_ring_data(res_score)
            if c_ring is None: continue
            
            lp = ligand.atoms.positions
            
            # 计算得分
            _, ang_dec = calculate_carbon_angles_and_decay(lp, c_ring, n_ring)
            _, dist_dec = calculate_distance_decay(lp, c_ring, n_ring)
            scores = calculate_combined_weight(base_w, ang_dec, dist_dec)
            
            # 距离截断
            dists = np.linalg.norm(lp - c_ring, axis=1)
            scores[dists > CONTACT_CUTOFF] = 0.0
            
            if np.max(scores) < 0.01: continue
            
            # 1. 获取 Top 5 得分
            sorted_s = np.sort(scores)[::-1] # 从大到小
            top_5 = np.zeros(5)
            n_copy = min(len(sorted_s), 5)
            top_5[:n_copy] = sorted_s[:n_copy]
            all_top_scores.append(top_5)
            
            # 2. 统计有效原子数 (Score > 0.1)
            count = np.sum(scores > 0.1)
            all_atom_counts.append(count)

    if not all_top_scores: return None

    avg_top_5 = np.mean(all_top_scores, axis=0)
    avg_count = np.mean(all_atom_counts)
    
    return avg_top_5, avg_count

def main():
    aligner = OffsetCalculator()
    root = "."
    compounds = sorted(glob.glob(os.path.join(root, "*")))
    
    results = []
    
    print(f"{'Compound':<20} | {'Rank1':<6} | {'Rank2':<6} | {'Rank3':<6} | {'Rank4':<6} | {'Rank5':<6} | {'Count(>0.1)':<12}")
    print("-" * 80)

    for c_dir in compounds:
        if not os.path.isdir(c_dir): continue
        if any(x in c_dir for x in ["modules", "results", "__pycache__"]): continue
        cid = os.path.basename(c_dir)
        
        # 重点关注这几个
        if not any(x in cid for x in ["ARI", "S10", "UNC", "Dopa", "BRE"]): continue
        
        ret = analyze_profile(c_dir, cid, aligner)
        if ret:
            prof, count = ret
            print(f"{cid[:20]:<20} | {prof[0]:.3f}  | {prof[1]:.3f}  | {prof[2]:.3f}  | {prof[3]:.3f}  | {prof[4]:.3f}  | {count:.2f}")

if __name__ == "__main__":
    main()