#!/usr/bin/env python3
import matplotlib
matplotlib.use('Agg')
import MDAnalysis as mda
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import sys
import argparse
from tqdm import tqdm

# ================= 复用现有模块 =================
try:
    from modules.qm_loader import load_cube_and_map, find_ligand
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
OUTPUT_BASE_DIR = config.get_path("paths.result_dir")
TARGET_BW = "6.51" 
INTEGRATION_RADIUS = config.get_float("data.integration_radius")

def analyze_score_ranks(c_dir, cid, aligner):
    print(f"\n>>> Processing: {cid}")
    
    # 1. 尝试加载 QC 权重 (保持与之前逻辑一致)
    cubs = glob.glob(os.path.join(c_dir, "*.cub"))
    pdbs = glob.glob(os.path.join(c_dir, "*.pdb"))
    ref_pdb = next((p for p in pdbs if "step7" not in p and "topol" not in p and "QC" not in p), None)
    
    qc_weights_map = None
    if cubs and ref_pdb:
        try:
            qm_data = load_cube_and_map(cubs[0], ref_pdb, INTEGRATION_RADIUS)
            raw_integrals = qm_data['integrals']
            if len(raw_integrals) > 0:
                qc_weights_map = raw_integrals / np.max(raw_integrals)
        except: pass
    
    # 2. 寻找轨迹
    xtcs = glob.glob(os.path.join(c_dir, "**", "merged.xtc"), recursive=True)
    if not xtcs: return None, None

    # 存储每一帧的 Top 10 得分
    all_ranks_matrix = [] 

    for xtc in xtcs:
        rd = os.path.dirname(xtc)
        tps = [os.path.join(rd, f) for f in os.listdir(rd) if f.endswith(".tpr")]
        topo = next((t for t in tps if "production" in t), tps[0] if tps else None)
        if not topo: continue

        try:
            u = mda.Universe(topo, xtc)
        except: continue

        real_target_ids = aligner.get_real_residue_ids(u, [TARGET_BW])
        if not real_target_ids: continue
        target_res = u.select_atoms(f"resid {real_target_ids[0]}")
        
        ligand = find_ligand(u)
        if not ligand: continue

        base_weights = np.ones(len(ligand.atoms))
        if qc_weights_map is not None:
            n_min = min(len(base_weights), len(qc_weights_map))
            base_weights[:n_min] = qc_weights_map[:n_min]

        # 抽样处理
        stride = 5 
        for ts in u.trajectory[::stride]:
            c2, n2 = get_aromatic_ring_data(target_res)
            if c2 is None: continue
            
            lp_ring = ligand.atoms.positions
            
            # 计算得分
            _, ang_dec = calculate_carbon_angles_and_decay(lp_ring, c2, n2)
            _, dist_dec = calculate_distance_decay(lp_ring, c2, n2)
            atom_scores = calculate_combined_weight(base_weights, ang_dec, dist_dec)
            
            # --- 关键步骤：排序 ---
            # 从大到小排序
            sorted_scores = np.sort(atom_scores)[::-1]
            
            # 只取前 10 名 (如果不够 10 个用 0 补齐)
            top_k = 10
            padded_scores = np.zeros(top_k)
            n_real = min(len(sorted_scores), top_k)
            padded_scores[:n_real] = sorted_scores[:n_real]
            
            # 只有当第一名得分 > 0.01 (有基本接触) 时才记录，避免纯噪音干扰平均值
            if padded_scores[0] > 0.01:
                all_ranks_matrix.append(padded_scores)

    if not all_ranks_matrix:
        return None, None
        
    # 计算平均值
    # shape: [N_frames, 10]
    mat = np.array(all_ranks_matrix)
    mean_curve = np.mean(mat, axis=0)
    std_curve = np.std(mat, axis=0)
    
    return mean_curve, std_curve

def main():
    aligner = OffsetCalculator()
    root = "."
    results = {}
    
    all_dirs = sorted(glob.glob(os.path.join(root, "*")))
    
    # 颜色库
    colors = plt.cm.tab20(np.linspace(0, 1, len(all_dirs)))
    color_idx = 0

    plt.figure(figsize=(10, 6))

    for c_dir in all_dirs:
        if not os.path.isdir(c_dir): continue
        if any(x in c_dir for x in ["modules", "results", "__pycache__"]): continue
        cid = os.path.basename(c_dir)
        
        mean_curve, std_curve = analyze_score_ranks(c_dir, cid, aligner)
        
        if mean_curve is not None:
            # X 轴: Rank 1, Rank 2 ...
            x = np.arange(1, 11)
            
            # 线条粗细区分 Dopa
            lw = 3.0 if "Dopa" in cid else 1.5
            alpha = 1.0 if "Dopa" in cid else 0.6
            ls = '-'
            
            # 归一化曲线？(可选)
            # 如果你想看相对比例，可以除以 Rank 1 的分值:
            # mean_curve = mean_curve / mean_curve[0] 
            
            plt.plot(x, mean_curve, label=cid, linewidth=lw, alpha=alpha, linestyle=ls)
            # 也可以画出误差带，但分子多了会乱，这里暂不画
            
            print(f"    Top 3 Scores: {mean_curve[0]:.3f}, {mean_curve[1]:.3f}, {mean_curve[2]:.3f}")
            
            # 检查 Rank 3 vs Rank 1 的比率
            ratio_3_1 = mean_curve[2] / (mean_curve[0] + 1e-6)
            print(f"    Ratio (Rank3 / Rank1): {ratio_3_1:.2f}")

    plt.title("Distribution of Atom Scores by Rank (Top 10 Atoms)")
    plt.xlabel("Atom Rank (1 = Highest Score)")
    plt.ylabel("Average Score (Absolute Value)")
    plt.xticks(np.arange(1, 11))
    plt.grid(True, linestyle='--', alpha=0.5)
    
    # 画一条 0.7 * Max 的参考线 (假设 Max=1.0)
    # 这是一个动态阈值参考区
    # plt.axhline(y=0.7, color='r', linestyle='--', label='Fixed Threshold 0.7 (Example)')
    
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    out_file = os.path.join(OUTPUT_BASE_DIR, "score_rank_distribution.png")
    plt.savefig(out_file, dpi=300)
    print(f"\n[Done] Analysis plot saved to: {out_file}")

if __name__ == "__main__":
    main()