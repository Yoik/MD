#!/usr/bin/env python3
import matplotlib
matplotlib.use('Agg')
import MDAnalysis as mda
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import sys
from tqdm import tqdm

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
    sys.exit(1)

config = init_config()
OUTPUT_BASE_DIR = config.get_path("paths.result_dir")
SCORE_TARGET_BW = "6.51"
VECTOR_BW_START = "6.51"
VECTOR_BW_END   = "6.52"
CONTACT_CUTOFF = 5.0 

def calculate_principal_axis(positions, weights):
    weights = weights.reshape(-1, 1)
    if np.sum(weights) < 1e-6: return None
    center = np.sum(positions * weights, axis=0) / np.sum(weights)
    centered = positions - center
    w_pos = centered * np.sqrt(weights)
    cov = np.dot(w_pos.T, w_pos)
    eigvals, eigvecs = np.linalg.eigh(cov)
    return eigvecs[:, -1]

def analyze_compound(c_dir, cid, aligner):
    xtcs = glob.glob(os.path.join(c_dir, "**", "merged.xtc"), recursive=True)
    if not xtcs: return []
    frame_data = [] 

    for xtc in xtcs:
        rd = os.path.dirname(xtc)
        tps = [f for f in os.listdir(rd) if f.endswith(".tpr") and "production" in f]
        if not tps: tps = [f for f in os.listdir(rd) if f.endswith(".tpr")]
        if not tps: continue
        try: u = mda.Universe(os.path.join(rd, tps[0]), xtc)
        except: continue

        real_score_id = aligner.get_real_residue_ids(u, [SCORE_TARGET_BW])
        real_vec_ids = aligner.get_real_residue_ids(u, [VECTOR_BW_START, VECTOR_BW_END])
        if not real_score_id or not real_vec_ids or len(real_vec_ids)<2: continue
        
        res_score = u.select_atoms(f"resid {real_score_id[0]}")
        atom_vec_start = u.select_atoms(f"resid {real_vec_ids[0]} and name CA")
        atom_vec_end   = u.select_atoms(f"resid {real_vec_ids[1]} and name CA")
        ligand = find_ligand(u)
        if not ligand: continue
        
        base_w = np.ones(len(ligand.atoms))
        
        for ts in u.trajectory[::10]:
            c_ring, n_ring = get_aromatic_ring_data(res_score)
            if c_ring is None: continue
            
            p_s = atom_vec_start.positions[0]; p_e = atom_vec_end.positions[0]
            ref_vec = p_e - p_s; norm_ref = np.linalg.norm(ref_vec)
            if norm_ref < 0.1: continue
            ref_vec_u = ref_vec / norm_ref

            lp = ligand.atoms.positions
            _, ang_dec = calculate_carbon_angles_and_decay(lp, c_ring, n_ring)
            _, dist_dec = calculate_distance_decay(lp, c_ring, n_ring)
            scores = calculate_combined_weight(base_w, ang_dec, dist_dec)
            dists = np.linalg.norm(lp - c_ring, axis=1)
            scores[dists > CONTACT_CUTOFF] = 0.0
            
            s_sum = np.sum(scores[scores > 0.01])
            s_max = np.max(scores)
            
            if s_sum < 0.01: continue
            
            # === 核心特征 ===
            # Y轴: 专注度 (Max / Sum)
            focus_ratio = s_max / s_sum
            
            # X轴: 角度
            angle_val = None
            valid_mask = scores > 0.01
            valid_idx = np.where(valid_mask)[0]
            if len(valid_idx) >= 2:
                sub_scores = scores[valid_idx]
                k = 3
                if len(sub_scores) > k:
                    top_k_local = np.argsort(sub_scores)[-k:]
                    final_idx = valid_idx[top_k_local]
                else: final_idx = valid_idx
                eff_pos = lp[final_idx]; eff_w = scores[final_idx]
                lig_axis = calculate_principal_axis(eff_pos, eff_w)
                if lig_axis is not None:
                    dot = np.clip(np.dot(lig_axis, ref_vec_u), -1, 1)
                    angle_val = np.degrees(np.arccos(np.abs(dot)))
            
            if angle_val is not None:
                frame_data.append([angle_val, focus_ratio])

    return np.array(frame_data)

def main():
    aligner = OffsetCalculator()
    root = "."
    plt.figure(figsize=(10, 8))
    compounds = sorted(glob.glob(os.path.join(root, "*")))
    summary_data = []

    for c_dir in compounds:
        if not os.path.isdir(c_dir): continue
        if any(x in c_dir for x in ["modules", "results", "__pycache__"]): continue
        cid = os.path.basename(c_dir)
        
        data = analyze_compound(c_dir, cid, aligner)
        if len(data) == 0: continue
        
        mean_angle = np.mean(data[:, 0])
        mean_focus = np.mean(data[:, 1])
        summary_data.append({"Name": cid, "Angle": mean_angle, "Focus": mean_focus})
        
        color = 'gray'; marker = 'o'; size = 100
        if "Dopa" in cid: color = 'green'; marker = '*'; size = 300
        elif "ARI" in cid or "BRE" in cid: color = 'red'; marker = 's'; size = 150
        elif "S10" in cid or "UNC" in cid: color = 'blue'; marker = '^'; size = 150
            
        plt.scatter(mean_angle, mean_focus, c=color, s=size, marker=marker, alpha=0.8, edgecolors='k', label=cid)

    # 绘制参考线
    plt.axvline(x=50, color='k', linestyle='--', alpha=0.3)
    plt.axhline(y=0.35, color='k', linestyle='--', alpha=0.3) # 专注度阈值
    
    plt.text(70, 0.45, "Zone A: Dopa (Vertical & Focused)", color='green')
    plt.text(20, 0.45, "Zone B: S10/UNC (Horizontal & Focused)", color='blue')
    plt.text(20, 0.20, "Zone C: ARI/BRE (Horizontal & Diffuse)", color='red')

    plt.xlabel("Orientation Angle (Degrees)"); plt.ylabel("Focus Ratio (Max / Sum)")
    plt.title("Final Separation: Angle vs Focus")
    plt.savefig(os.path.join(OUTPUT_BASE_DIR, "14_verify_final_separation.png"), dpi=300)
    print(f"[Done] Saved to 14_verify_final_separation.png")
    
    print(f"{'Compound':<20} | {'Angle':<8} | {'Focus(Max/Sum)':<15}")
    for item in summary_data:
        print(f"{item['Name'][:20]:<20} | {item['Angle']:<8.2f} | {item['Focus']:<15.3f}")

if __name__ == "__main__":
    main()