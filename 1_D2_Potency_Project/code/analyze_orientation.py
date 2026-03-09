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

# ================= 核心参数配置 =================
config = init_config()
OUTPUT_BASE_DIR = config.get_path("paths.result_dir")

# 【修改点 1】 定义参考向量的起止 BW 编号
REF_START_BW = "6.51"
REF_END_BW   = "6.52"

# 距离截断与PCA阈值
CONTACT_CUTOFF = 5.0     
SCORE_THRESHOLD_RATIO = 0.7  

def calculate_principal_axis(positions, weights):
    """ 计算加权点云的第一主轴 (PCA) """
    weights = weights.reshape(-1, 1)
    total_weight = np.sum(weights)
    if total_weight < 1e-6: return None
    center = np.sum(positions * weights, axis=0) / total_weight
    centered_pos = positions - center
    weighted_pos = centered_pos * np.sqrt(weights)
    covariance_matrix = np.dot(weighted_pos.T, weighted_pos)
    eigvals, eigvecs = np.linalg.eigh(covariance_matrix)
    return eigvecs[:, -1] # 最大特征值对应的特征向量

def analyze_compound(c_dir, cid, aligner):
    print(f"\n>>> Processing: {cid}")
    
    # 1. 尝试加载 QC 权重
    cubs = glob.glob(os.path.join(c_dir, "*.cub"))
    pdbs = glob.glob(os.path.join(c_dir, "*.pdb"))
    ref_pdb = next((p for p in pdbs if "step7" not in p and "topol" not in p and "QC" not in p), None)
    
    qc_weights_map = None
    if cubs and ref_pdb:
        try:
            qm_data = load_cube_and_map(cubs[0], ref_pdb, 5.0) # hardcode radius for simplicity
            raw_integrals = qm_data['integrals']
            if len(raw_integrals) > 0:
                qc_weights_map = raw_integrals / np.max(raw_integrals)
            print(f"    [QC] Using QM electron density weights.")
        except:
            print(f"    [Warn] QM load failed, using Geometry only.")
    
    # 2. 寻找轨迹
    xtcs = glob.glob(os.path.join(c_dir, "**", "merged.xtc"), recursive=True)
    if not xtcs: return None

    all_angles = []

    for xtc in xtcs:
        rd = os.path.dirname(xtc)
        tps = [os.path.join(rd, f) for f in os.listdir(rd) if f.endswith(".tpr")]
        topo = next((t for t in tps if "production" in t), tps[0] if tps else None)
        if not topo: continue

        try:
            u = mda.Universe(topo, xtc)
        except: continue

        # ==========================================
        # 【修改点 2】 获取 6.51 和 6.52 的真实 ID
        # ==========================================
        real_ids = aligner.get_real_residue_ids(u, [REF_START_BW, REF_END_BW])
        
        # 必须两个都找到才能连线
        if not real_ids or len(real_ids) < 2:
            print(f"    [Skip] Could not find both BW {REF_START_BW} and {REF_END_BW} in topo.")
            continue
            
        id_start, id_end = real_ids[0], real_ids[1]
        
        # 选择 CA 原子
        atom_start = u.select_atoms(f"resid {id_start} and name CA")
        atom_end   = u.select_atoms(f"resid {id_end} and name CA")
        
        # 为了计算 Score (判定有效原子)，我们依然需要 6.51 的环平面信息
        # 假设 6.51 就是 REF_START_BW
        res_651 = u.select_atoms(f"resid {id_start}")

        if len(atom_start) == 0 or len(atom_end) == 0:
            print("    [Skip] CA atoms missing.")
            continue

        ligand = find_ligand(u)
        if not ligand: continue

        # 准备基础权重
        base_weights = np.ones(len(ligand.atoms))
        if qc_weights_map is not None:
            n_min = min(len(base_weights), len(qc_weights_map))
            base_weights[:n_min] = qc_weights_map[:n_min]

        stride = 5 
        for ts in u.trajectory[::stride]:
            # A. 计算参考向量 (6.51 CA -> 6.52 CA)
            # 这代表了 Helix 的局部方向
            p_start = atom_start.positions[0]
            p_end   = atom_end.positions[0]
            ref_vec = p_end - p_start
            
            # 归一化参考向量
            norm_ref = np.linalg.norm(ref_vec)
            if norm_ref < 0.1: continue
            ref_vec_u = ref_vec / norm_ref

            # B. 计算配体有效原子 (PCA) - 逻辑保持不变以确保只取接触面
            # 依然使用相对于 6.51 芳香环的 Decay 来筛选原子
            c2, n2 = get_aromatic_ring_data(res_651)
            if c2 is None: continue
            
            lp_ring = ligand.atoms.positions
            
            # 距离/角度 Decay
            _, ang_dec = calculate_carbon_angles_and_decay(lp_ring, c2, n2)
            _, dist_dec = calculate_distance_decay(lp_ring, c2, n2)
            atom_scores = calculate_combined_weight(base_weights, ang_dec, dist_dec)
            
            # 距离截断
            dists_to_center = np.linalg.norm(lp_ring - c2, axis=1)
            atom_scores[dists_to_center > CONTACT_CUTOFF] = 0.0
            
            # 筛选 Top Cloud
            current_max = np.max(atom_scores)
            if current_max < 0.01: continue
            
            eff_idx = np.where(atom_scores > (current_max * SCORE_THRESHOLD_RATIO))[0]
            if len(eff_idx) < 2: continue
            
            eff_coords = lp_ring[eff_idx]
            eff_weights = atom_scores[eff_idx]
            
            # PCA 计算配体主轴
            lig_vec = calculate_principal_axis(eff_coords, eff_weights)
            if lig_vec is None: continue
            
            # C. 【修改点 3】 计算两个向量的夹角
            # lig_vec 是轴 (无方向)，ref_vec 是向量 (有方向)
            # 但我们只关心它们是不是平行，所以取 abs(dot)
            # |cos(theta)| = |v1 . v2|
            
            dot_val = np.clip(np.dot(lig_vec, ref_vec_u), -1, 1)
            angle_rad = np.arccos(np.abs(dot_val)) # 范围 [0, pi/2]
            angle_deg = np.degrees(angle_rad)      # 范围 [0, 90]
            
            # 0度 = 平行于 Helix 走向
            # 90度 = 垂直于 Helix 走向
            
            all_angles.append(angle_deg)
            
    if all_angles:
        print(f"    -> Collected {len(all_angles)} frames. Mean Angle vs CA-CA Vector: {np.mean(all_angles):.2f}")
    return all_angles

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--filter", type=str, default="", help="Filter compound name")
    args = parser.parse_args()
    
    aligner = OffsetCalculator()
    root = "."
    results = {}
    
    all_dirs = sorted(glob.glob(os.path.join(root, "*")))
    for c_dir in all_dirs:
        if not os.path.isdir(c_dir): continue
        if any(x in c_dir for x in ["modules", "results", "__pycache__"]): continue
        cid = os.path.basename(c_dir)
        if args.filter and args.filter not in cid: continue
        
        angles = analyze_compound(c_dir, cid, aligner)
        if angles: results[cid] = angles

    if results:
        plt.figure(figsize=(10, 6))
        bins = np.linspace(0, 90, 45)
        for cid, angles in results.items():
            lw = 2.5 if "Dopa" in cid else 1.5
            plt.hist(angles, bins=bins, histtype='step', linewidth=lw, label=cid, density=True)

        plt.title(f"Ligand Orientation vs {REF_START_BW}-{REF_END_BW} CA Vector")
        plt.xlabel(f"Angle with {REF_START_BW}-{REF_END_BW} CA Vector (0=Parallel, 90=Perpendicular)")
        plt.ylabel("Density")
        plt.xlim(0, 90)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        out_path = os.path.join(OUTPUT_BASE_DIR, "orientation_vs_caca_vector.png")
        plt.savefig(out_path, dpi=300)
        print(f"\n[Done] Saved to: {out_path}")

if __name__ == "__main__":
    main()