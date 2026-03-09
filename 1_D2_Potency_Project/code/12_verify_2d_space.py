#!/usr/bin/env python3
import matplotlib
matplotlib.use('Agg')
import MDAnalysis as mda
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import sys
import seaborn as sns
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

# ================= 参数配置 =================
config = init_config()
OUTPUT_BASE_DIR = config.get_path("paths.result_dir")

# 核心设定：
# 1. 用于【算分/选原子】的靶点：6.51 (主Pi-Pi位点)
SCORE_TARGET_BW = "6.51"
# 2. 用于【定方向】的参考向量：6.51 -> 6.52 (Helix 6 轴)
VECTOR_BW_START = "6.51"
VECTOR_BW_END   = "6.52"

# 距离截断 (只考虑接触面附近的原子)
CONTACT_CUTOFF = 5.0 

def calculate_principal_axis(positions, weights):
    """ PCA 计算主轴 """
    weights = weights.reshape(-1, 1)
    if np.sum(weights) < 1e-6: return None
    center = np.sum(positions * weights, axis=0) / np.sum(weights)
    centered = positions - center
    w_pos = centered * np.sqrt(weights)
    cov = np.dot(w_pos.T, w_pos)
    eigvals, eigvecs = np.linalg.eigh(cov)
    return eigvecs[:, -1]

def analyze_compound(c_dir, cid, aligner):
    print(f"\n>>> Processing: {cid}")
    
    # 寻找轨迹
    xtcs = glob.glob(os.path.join(c_dir, "**", "merged.xtc"), recursive=True)
    if not xtcs: return []

    frame_data = [] # 存每一帧的 (Angle, Sum)

    for xtc in xtcs:
        rd = os.path.dirname(xtc)
        tps = [f for f in os.listdir(rd) if f.endswith(".tpr") and "production" in f]
        if not tps: tps = [f for f in os.listdir(rd) if f.endswith(".tpr")]
        if not tps: continue
        
        try:
            u = mda.Universe(os.path.join(rd, tps[0]), xtc)
        except: continue

        # 1. 找 6.51 (用于得分)
        real_score_id = aligner.get_real_residue_ids(u, [SCORE_TARGET_BW])
        if not real_score_id: continue
        res_score = u.select_atoms(f"resid {real_score_id[0]}")
        
        # 2. 找 6.51->6.52 (用于向量)
        real_vec_ids = aligner.get_real_residue_ids(u, [VECTOR_BW_START, VECTOR_BW_END])
        if not real_vec_ids or len(real_vec_ids) < 2: continue
        atom_vec_start = u.select_atoms(f"resid {real_vec_ids[0]} and name CA")
        atom_vec_end   = u.select_atoms(f"resid {real_vec_ids[1]} and name CA")
        
        ligand = find_ligand(u)
        if not ligand: continue
        
        # 权重初始化
        base_w = np.ones(len(ligand.atoms))

        # 抽样遍历
        stride = 10 
        for ts in u.trajectory[::stride]:
            # A. 准备数据
            c_ring, n_ring = get_aromatic_ring_data(res_score) # 6.51 的环
            if c_ring is None: continue
            
            if len(atom_vec_start)==0 or len(atom_vec_end)==0: continue
            p_s = atom_vec_start.positions[0]
            p_e = atom_vec_end.positions[0]
            ref_vec = p_e - p_s # 6.51 -> 6.52
            norm_ref = np.linalg.norm(ref_vec)
            if norm_ref < 0.1: continue
            ref_vec_u = ref_vec / norm_ref

            lp = ligand.atoms.positions
            
            # B. 计算得分 (基于 6.51)
            _, ang_dec = calculate_carbon_angles_and_decay(lp, c_ring, n_ring)
            _, dist_dec = calculate_distance_decay(lp, c_ring, n_ring)
            scores = calculate_combined_weight(base_w, ang_dec, dist_dec)
            
            # 距离截断
            dists = np.linalg.norm(lp - c_ring, axis=1)
            scores[dists > CONTACT_CUTOFF] = 0.0
            
            # 至少要有接触
            if np.max(scores) < 0.01: continue
            
            # === C. 计算 Sum (Y轴特征) ===
            # 直接加和所有大于 0.01 的分数，代表接触“总量”
            total_sum = np.sum(scores[scores > 0.01])

            # === D. 计算 Angle (X轴特征) ===
            # Top-3 筛选
            valid_mask = scores > 0.01
            valid_idx = np.where(valid_mask)[0]
            
            angle_val = None
            if len(valid_idx) >= 2:
                # 取 Top 3
                sub_scores = scores[valid_idx]
                k = 3
                if len(sub_scores) > k:
                    top_k_local = np.argsort(sub_scores)[-k:]
                    final_idx = valid_idx[top_k_local]
                else:
                    final_idx = valid_idx
                
                eff_pos = lp[final_idx]
                eff_w   = scores[final_idx]
                
                lig_axis = calculate_principal_axis(eff_pos, eff_w)
                if lig_axis is not None:
                    dot = np.clip(np.dot(lig_axis, ref_vec_u), -1, 1)
                    angle_val = np.degrees(np.arccos(np.abs(dot)))
            
            if angle_val is not None:
                frame_data.append([angle_val, total_sum])

    return np.array(frame_data)

def main():
    aligner = OffsetCalculator()
    root = "."
    
    # 颜色映射
    plt.figure(figsize=(10, 8))
    
    compounds = sorted(glob.glob(os.path.join(root, "*")))
    
    summary_data = []

    for c_dir in compounds:
        if not os.path.isdir(c_dir): continue
        if any(x in c_dir for x in ["modules", "results", "__pycache__"]): continue
        cid = os.path.basename(c_dir)
        
        data = analyze_compound(c_dir, cid, aligner)
        if len(data) == 0: continue
        
        # 取平均值画散点 (代表该分子的中心位置)
        mean_angle = np.mean(data[:, 0])
        mean_sum = np.mean(data[:, 1])
        
        # 记录用于打印
        summary_data.append({
            "Name": cid, 
            "Angle": mean_angle, 
            "Sum": mean_sum,
            "N_Frames": len(data)
        })
        
        # 设定样式
        color = 'gray'
        marker = 'o'
        size = 100
        label = cid
        
        if "Dopa" in cid:
            color = 'green'; marker = '*'; size = 300
        elif "ARI" in cid:
            color = 'red'; marker = 's'; size = 150
        elif "S10" in cid or "UNC" in cid:
            color = 'blue'; marker = '^'; size = 150
            
        plt.scatter(mean_angle, mean_sum, c=color, s=size, marker=marker, alpha=0.8, edgecolors='k', label=label)

    # 绘制参考区域 (根据你的新数据调整)
    # 垂直界限: 50度
    plt.axvline(x=50, color='k', linestyle='--', alpha=0.3)
    # Sum 界限: 1.2 (区分 Dopa/ROT 和 ARI/S10)
    plt.axhline(y=1.2, color='k', linestyle='--', alpha=0.3)
    
    # 标注区域含义 (基于最新数据的假设)
    plt.text(70, 0.8, "Region 1: Dopa/ROT Type\n(Vertical & Modest Contact)\nHigh Angle / Low Sum (<1.2)", color='green', fontsize=10)
    plt.text(20, 1.8, "Region 2: ARI/S10 Type\n(Horizontal & Huge Contact)\nLow Angle / High Sum (>1.4)", color='red', fontsize=10)
    
    plt.xlabel("Orientation Angle vs Helix 6 Axis (Degrees)\n0=Parallel(Horizontal), 90=Perpendicular(Vertical)")
    plt.ylabel("Total Interaction Score (Sum)")
    plt.title("2D Classification: Orientation vs Interaction Sum")
    plt.grid(True, linestyle='--', alpha=0.5)
    
    # 打印详细数据
    print("\n=== Verification Summary ===")
    print(f"{'Compound':<20} | {'Angle':<10} | {'Sum':<10} | {'Type Estimate'}")
    print("-" * 60)
    for item in summary_data:
        t = "Unknown"
        if item['Angle'] > 50 and item['Sum'] < 1.3:
            t = "Dopa-like"
        elif item['Angle'] < 50 and item['Sum'] > 1.3:
            t = "ARI/S10-like"
        elif item['Angle'] < 50 and item['Sum'] < 1.3:
            t = "ROT-like (Unique)"
            
        print(f"{item['Name'][:20]:<20} | {item['Angle']:<10.2f} | {item['Sum']:<10.3f} | {t}")
        plt.text(item['Angle']+1, item['Sum']+0.01, item['Name'][:10], fontsize=8)

    out_file = os.path.join(OUTPUT_BASE_DIR, "verify_2d_classification.png")
    plt.savefig(out_file, dpi=300)
    print(f"\n[Done] Plot saved to: {out_file}")

if __name__ == "__main__":
    main()