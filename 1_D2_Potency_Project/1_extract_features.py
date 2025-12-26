#!/usr/bin/env python3
import matplotlib
matplotlib.use('Agg')
import gc
import MDAnalysis as mda
from MDAnalysis.lib.distances import distance_array
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
import os
import sys
import argparse
from scipy.spatial.distance import cdist
import io
import tempfile
import warnings

# RDKit 必须引入
from rdkit import Chem

# 导入模块
try:
    # 引入模块
    from modules.qm_loader import load_cube_and_map, save_qc_structure, find_ligand
    from modules.cube_parser import CubeParser
    from modules.ring_matcher import RingMatcher
    from modules.sequence_aligner import OffsetCalculator
    from modules import (
        get_aromatic_ring_data,
        calculate_plane_normal,
        calculate_carbon_angles_and_decay,
        calculate_distance_decay,
        calculate_combined_weight,
        calculate_interaction_strength,
        OutputHandler
    )
    from src.config import init_config
except ImportError as e:
    print(f"Error: 模块导入失败: {e}")
    sys.exit(1)

# ================= 核心配置区 =================
config = init_config()
INTEGRATION_RADIUS = config.get_float("data.integration_radius")
OUTPUT_BASE_DIR = config.get_path("paths.result_dir")
QC_OUTPUT_DIR = config.get_path("paths.qc_output_dir")

# 直接读取 BW 编号列表 (Config 中是字符串列表)
PHE_BW_LIST = config.get_list("residues.phe_residues")     # 如 ["6.48", "6.50"]
OBP_BW_LIST = config.get_list("residues.obp_residues")     # 如 ["3.32", ...]
PLANE_BW_LIST = config.get_list("residues.plane_residues") # 如 ["5.46", ...]

# ================= 核心工具函数：RDKit 映射 =================

def get_rdkit_mapping(ref_pdb_path, mda_ligand_atoms):
    """
    计算从 Reference PDB (QM) 到 MD Analysis Ligand 的原子索引映射。
    """
    
    # --- 内部辅助：骨架化函数 ---
    def get_skeleton(mol):
        m = Chem.Mol(mol)
        for b in m.GetBonds():
            b.SetBondType(Chem.BondType.SINGLE)
            b.SetIsAromatic(False)
        for a in m.GetAtoms():
            a.SetIsAromatic(False)
        return m
    # ---------------------------

    # 1. 【元素清洗】
    if not hasattr(mda_ligand_atoms.universe.atoms, 'elements'):
        mda_ligand_atoms.universe.add_TopologyAttr('elements')
        
    valid_elems = set(['H', 'C', 'N', 'O', 'S', 'F', 'P', 'CL', 'BR', 'I', 'B', 'SI', 'FE', 'ZN', 'MG', 'CA', 'NA', 'K', 'LI'])
    
    for atom in mda_ligand_atoms:
        original_elem = atom.element.upper() if atom.element else ""
        if original_elem not in valid_elems:
            name = atom.name.upper()
            guess = "".join(filter(str.isalpha, name))
            if len(guess) > 1 and guess[:2] in valid_elems: atom.element = guess[:2]
            elif len(guess) > 0 and guess[0] in valid_elems: atom.element = guess[0]
            else: atom.element = 'C'

    # 2. 加载 Reference PDB
    ref_mol = Chem.MolFromPDBFile(ref_pdb_path, removeHs=True, sanitize=False)
    if not ref_mol:
        print(f"     [Map Error] Failed to load Ref PDB: {ref_pdb_path}")
        return None, None

    # 3. 将 MD Ligand 转换为 RDKit Mol
    target_mol = None
    tmp_path = None
    
    try:
        with tempfile.NamedTemporaryFile(suffix=".pdb", delete=False) as tmp:
            tmp_path = tmp.name
            
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mda_ligand_atoms.write(tmp_path)
        
        target_mol = Chem.MolFromPDBFile(tmp_path, removeHs=True, sanitize=False)
        
    except Exception as e:
        print(f"     [Map Error] MD Ligand to RDKit conversion failed: {e}")
        return None, None
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try: os.remove(tmp_path)
            except: pass

    if not target_mol:
        print(f"     [Map Error] Target Mol is None after conversion.")
        return None, None

    # 4. 【核心步骤】骨架匹配
    try:
        try:
            ref_mol.UpdatePropertyCache(strict=False)
            target_mol.UpdatePropertyCache(strict=False)
        except: pass
        ref_mol = Chem.RemoveHs(ref_mol, sanitize=False)
        target_mol = Chem.RemoveHs(target_mol, sanitize=False)

        ref_skel = get_skeleton(ref_mol)
        target_skel = get_skeleton(target_mol)

        if target_skel.HasSubstructMatch(ref_skel):
            match = target_skel.GetSubstructMatch(ref_skel)
            
            # 检查原子数一致性 (Info 级别)
            if len(match) != ref_mol.GetNumAtoms() or ref_mol.GetNumAtoms() != target_mol.GetNumAtoms():
                if ref_mol.GetNumAtoms() > target_mol.GetNumAtoms():
                     print(f"     [Map Warn] Ref larger than Target? Ref: {ref_mol.GetNumAtoms()}, Target: {target_mol.GetNumAtoms()}")
            
            mapping = {}
            for ref_idx, target_idx in enumerate(match):
                mapping[ref_idx] = target_idx
            return mapping, ref_mol
        else:
            print(f"     [Map Fail] Topology mismatch (Skeleton).")
            return None, None

    except Exception as e:
        print(f"     [Map Error] Skeleton match crashed: {e}")
        return None, None

# ================= 辅助函数 =================

def get_dopa_global_max(root_dir):
    print(">>> Searching for Dopa reference (Global Normalization Standard)...")
    all_dirs = glob.glob(os.path.join(root_dir, "*"))
    for c_dir in all_dirs:
        if not os.path.isdir(c_dir): continue
        if "dopa" in os.path.basename(c_dir).lower():
            cubs = glob.glob(os.path.join(c_dir, "*.cub"))
            if cubs:
                try:
                    cp = CubeParser(cubs[0])
                    integrals = cp.get_carbon_integrals(INTEGRATION_RADIUS)
                    if len(integrals) > 0:
                        g_max = np.max(integrals)
                        print(f"    [Global Ref] Dopa Max = {g_max:.4f} ({os.path.basename(c_dir)})")
                        return g_max
                except: pass
    print("    [WARN] Dopa reference not found! Using 1.0.")
    return 1.0

def align_xy(lig, obp, whole):
    c = lig.mean(0); u,s,vh = np.linalg.svd(lig-c)
    return np.dot(lig-c, vh.T)[:,:2], np.dot(obp-c, vh.T)[:,:2], np.dot(whole-c, vh.T)[:,:2]

def save_3d_structure_pdb(lig_coords, lig_names, obp_coords, obp_names, obp_indices, out_file, cid, ring_atom_indices=None, ring_weights=None):
    try:
        with open(out_file, 'w') as f:
            f.write(f"REMARK Compound: {cid}\n")
            atom_num = 1
            for i, (coord, name) in enumerate(zip(lig_coords, lig_names)):
                in_ring = False; w_val = 0.0
                if ring_atom_indices is not None and i in ring_atom_indices:
                    in_ring = True
                    w_idx = ring_atom_indices.index(i)
                    if ring_weights is not None and w_idx < len(ring_weights):
                        w_val = ring_weights[w_idx]
                b_factor = 50.0 if in_ring else 0.0
                f.write(f"ATOM  {atom_num:5d}  {name:4s} LIG A   1    {coord[0]:8.3f}{coord[1]:8.3f}{coord[2]:8.3f}{w_val:6.2f}{b_factor:6.2f}           C\n")
                atom_num += 1
            for i, (coord, name, res_idx) in enumerate(zip(obp_coords, obp_names, obp_indices)):
                f.write(f"ATOM  {atom_num:5d}  CA  RES B{i+1:3d}    {coord[0]:8.3f}{coord[1]:8.3f}{coord[2]:8.3f}  1.00  0.00           C\n")
                atom_num += 1
            f.write("END\n")
    except Exception as e:
        print(f"Error saving PDB: {e}")

# ================= 主处理逻辑 =================

def process_replicate(xtc, topo, qm_data, 
                      ref_ring_indices, ref_geo_indices, ring_type_info, 
                      cid, rep_name, offset_calc, output_handler, global_max):
    try: 
        u = mda.Universe(topo, xtc)
    except Exception as e: 
        print(f"     [Error] Universe load failed for {rep_name}: {e}")
        return None, None, None
    
    # 【新增功能】打印偏移量 & 氨基酸单字母检查 (3.32, 6.51, 6.52)
    # ======================================================================
    print(f"     [Check] Residue Verification:")
    
    # 1. 获取当前受体的标准信息
    # identify_receptor 会返回 (ReceptorKey, SimSeq, SimResids)
    rec_key, _, _ = offset_calc.identify_receptor(u, verbose=True)
    
    if rec_key:
        # 获取该受体的 BW 映射表 (BW -> Std ID)
        bw_map = offset_calc.db[rec_key]['bw_map']
        
        # 我们要检查的目标 BW 编号
        check_targets = ["3.32", "6.51", "6.52"]
        
        for bw in check_targets:
            # A. 获取标准序列 ID (从 YAML 数据库)
            # 注意: 如果 yaml 里没写这个 BW，就没法算 offset
            std_id = bw_map.get(bw) 
            if isinstance(std_id, str): std_id = int(std_id)
            
            # B. 获取模拟真实 ID (通过比对计算)
            real_ids = offset_calc.get_real_residue_ids(u, [bw])
            
            if real_ids and std_id:
                rid = real_ids[0] # 真实模拟 ID
                
                # C. 获取氨基酸单字母
                try:
                    # 从 Universe 中选出该残基
                    res = u.select_atoms(f"resid {rid}").residues[0]
                    res_name_3 = res.resname # 三字母 (如 ASP)
                    # 复用 offset_calc 里的字典转单字母
                    res_name_1 = offset_calc.three_to_one.get(res_name_3, '?')
                    
                    # D. 计算偏移量
                    current_offset = rid - std_id
                    
                    # 打印结果
                    # 格式: BW编号: Std -> Sim (Offset) | 氨基酸
                    print(f"       - BW {bw}: Std {std_id:<3} -> Sim {rid:<3} (Offset {current_offset:+d}) | AA: {res_name_1} ({res_name_3})")
                    
                except Exception as e:
                    print(f"       - BW {bw}: Sim ID {rid} found, but failed to read residue info.")
            else:
                status = "Not in DB" if not std_id else "Not in Sim"
                print(f"       - BW {bw}: Check Failed ({status})")
    else:
        print("       - [Error] Could not identify receptor, skipping check.")
    
    print("-" * 50)
    
    # 注意：这里直接传 Universe 和 BW 列表，会自动识别受体
    real_phe_ids = offset_calc.get_real_residue_ids(u, PHE_BW_LIST)
    real_obp_ids = offset_calc.get_real_residue_ids(u, OBP_BW_LIST)
    real_plane_ids = offset_calc.get_real_residue_ids(u, PLANE_BW_LIST)

    # 简单检查 (防止没找到残基)
    if not real_phe_ids or not real_obp_ids:
        print(f"     [Error] Failed to map BW residues to Simulation IDs.")
        return None, None, None
        
    # 为了兼容后面代码，计算一个名义上的 offset (仅用于统计输出，不参与核心计算)
    # 假设 PHE_BW_LIST[0] 是 6.48 (F389)，我们看它映射到了哪
    offset = 0 # 默认值
    if real_phe_ids:
        # 这里只是粗略估算，不再用于原子选择
        offset = real_phe_ids[0] - 389
    
    # 找到配体
    lig_res = find_ligand(u)
    if not lig_res: 
        print(f"     [Error] Ligand not found in MD topology.")
        return None, None, None

    # 1. 建立映射
    mapping, _ = get_rdkit_mapping(qm_data['pdb_path'], lig_res.atoms)
    
    if mapping is None:
        print(f"     [Map Fail] RDKit mapping failed completely for {rep_name}.")
        return None, None, None

    # ======================================================================
    # 【核心 QC 保存逻辑】(只在 Output/Log 阶段执行)
    # ======================================================================
    if not os.path.exists(QC_OUTPUT_DIR):
        os.makedirs(QC_OUTPUT_DIR)

    # A. 保存 REF 结构 (全分子积分，保持不变)
    ref_path = os.path.join(QC_OUTPUT_DIR, f"{cid}_QC_REF.pdb")
    if not os.path.exists(ref_path): 
        save_qc_structure(ref_path, qm_data, normalize=True, global_max=global_max)

    # B. 保存 MD MAP 结构 (仅高亮 Ring Atoms)
    md_map_path = os.path.join(QC_OUTPUT_DIR, f"{cid}_QC_MD_MAP.pdb")
    try:
        integrals = qm_data['integrals']
        norm_integrals = integrals / global_max
        
        # 初始化权重为 0
        md_weights = np.zeros(len(lig_res.atoms))
        
        # 【关键修改】只将 ref_ring_indices 中的原子权重传递给 MD
        # 这样打开 PDB 时，只有参与计算的部分是亮色的
        count_mapped = 0
        for ref_idx, md_idx in mapping.items():
            if ref_idx in ref_ring_indices and ref_idx < len(norm_integrals):
                md_weights[md_idx] = norm_integrals[ref_idx]
                count_mapped += 1
        
        # Universe 级别添加属性
        univ = lig_res.universe
        if not hasattr(univ.atoms, 'tempfactors'):
            univ.add_TopologyAttr('tempfactors')
            
        # 备份 & 赋值
        orig_vals = None
        if hasattr(lig_res.atoms, 'tempfactors'):
             orig_vals = lig_res.atoms.tempfactors.copy()
        
        lig_res.atoms.tempfactors = md_weights
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            lig_res.atoms.write(md_map_path)
            
        # 恢复
        if orig_vals is not None:
            lig_res.atoms.tempfactors = orig_vals
            
        # 打印日志
        if "replicate_1" in rep_name:
            print(f"     [QC] Saved Ring-Specific MD Verification: {md_map_path}")
            print(f"          (Highlighted {count_mapped} atoms, Ring Type: {ring_type_info})")
            
    except Exception as e:
        print(f"     [QC Error] Failed to save MD MAP PDB: {e}")
    # ======================================================================

    md_ring_indices = []
    md_geo_indices = []
    extracted_ring_weights = []

    for r_idx in ref_ring_indices:
        if r_idx in mapping:
            md_idx = mapping[r_idx]
            md_ring_indices.append(md_idx)
            raw_int = qm_data['integrals'][r_idx]
            extracted_ring_weights.append(raw_int / global_max)
            
    for g_idx in ref_geo_indices:
        if g_idx in mapping:
            md_geo_indices.append(mapping[g_idx])

    if not md_ring_indices or not md_geo_indices:
        print(f"     [Error] Ring atoms mapped to nowhere!")
        return None, None, None

    ring_weights_np = np.array(extracted_ring_weights)
    
    if rep_name == "gromacs_replicate_1":
        print(f"     [Ring] Type: {ring_type_info}")
        print(f"            Interaction Atoms: {len(md_ring_indices)}")
        print(f"            Geometry Center Atoms: {len(md_geo_indices)}")

    # 2. Prepare AtomGroups
    obp_atoms = u.select_atoms(f"resid {' '.join(map(str, real_obp_ids))} and name CA")
    plane_res = u.select_atoms(f"resid {' '.join(map(str, real_plane_ids))} and name CA")
    
    if len(real_phe_ids) >= 1:
        r389 = u.select_atoms(f"resid {real_phe_ids[0]}")
    else:
        r389 = u.select_atoms("none") # 空选择
        
    if len(real_phe_ids) >= 2:
        r390 = u.select_atoms(f"resid {real_phe_ids[1]}")
    else:
        r390 = u.select_atoms("none")

    global_ring_indices = [lig_res.atoms[i].index for i in md_ring_indices]
    ring_ag = u.atoms[global_ring_indices]
    global_geo_indices = [lig_res.atoms[i].index for i in md_geo_indices]
    geo_ag = u.atoms[global_geo_indices]

    data = []; vis_accum = np.zeros((len(real_obp_ids)+2, 2)); cnt = 0

    feature_vectors = []
    all_angles_389 = []; all_angles_390 = []
    all_dist_decays_389 = []; all_dist_decays_390 = []

    # 3. Trajectory Loop

    MAX_ATOMS = 9
    N_ATOM_FEAT = 6 # 每个原子的特征数 = 14 (距离) + 1 (389得分) + 1 (390得分) = 16

    padded_weights = np.zeros(MAX_ATOMS, dtype=np.float32)
    n_real = min(len(ring_weights_np), MAX_ATOMS)
    padded_weights[:n_real] = ring_weights_np[:n_real]

    for ts in u.trajectory:
        lp_ring = ring_ag.positions # ligand ring positions
        lp_geo = geo_ag.positions
        # lc_geo = lp_geo.mean(0) # Ligand Center (Geometry Atoms)
        safe_w = np.maximum(ring_weights_np, 0.0)

        # A. 几何角度计算
        ln = calculate_plane_normal(lp_geo)
        pn = calculate_plane_normal(plane_res.positions)
        dot_val = np.clip(np.dot(ln, pn), -1, 1)
        ga = np.degrees(np.arccos(dot_val)); ga = ga if ga<=90 else 180-ga
        ml_cos_angle = np.abs(dot_val)
        
        # 2. 计算 389 相互作用 (保留每个原子的得分)
        c1, n1 = get_aromatic_ring_data(r389)

        atom_scores_389 = np.zeros(len(lp_ring)) # [N_atoms]
        atom_dists_389 = np.ones(len(lp_ring)) * 100.0 # 默认距离 100
        atom_angles_389 = np.zeros(len(lp_ring))

        ml_389 = [0.0, 0.0, 0.0]
        if c1 is not None:
            raw_ang_389, ang_dec = calculate_carbon_angles_and_decay(lp_ring, c1, n1)
            raw_dist_389, dist_dec = calculate_distance_decay(lp_ring, c1, n1)

            atom_angles_389 = raw_ang_389
            atom_dists_389 = raw_dist_389

            # 这里直接拿到每个原子的加权分
            atom_scores_389 = calculate_combined_weight(safe_w, ang_dec, dist_dec)

            # comb = calculate_combined_weight(safe_w, ang_dec, dist_dec)
            s_sum = np.sum(atom_scores_389); s_max = np.max(atom_scores_389)
            ml_389 = [s_sum, s_max, s_max/(s_sum+1e-6)]

            # 用于最后画图/统计 (辅助)
            all_angles_389.append(ang_dec) # 这里存一下原始角度decay用于统计
            all_dist_decays_389.append(dist_dec)

        # 3. 计算 390 相互作用 (保留每个原子的得分)
        c2, n2 = get_aromatic_ring_data(r390)

        atom_scores_390 = np.zeros(len(lp_ring)) # [N_atoms]
        atom_dists_390 = np.ones(len(lp_ring)) * 100.0
        atom_angles_390 = np.zeros(len(lp_ring))

        ml_390 = [0.0, 0.0, 0.0]
        if c2 is not None:
            raw_ang_390, ang_dec = calculate_carbon_angles_and_decay(lp_ring, c2, n2)
            raw_dist_390, dist_dec = calculate_distance_decay(lp_ring, c2, n2)

            atom_angles_390 = raw_ang_390
            atom_dists_390 = raw_dist_390

            # 这里直接拿到每个原子的加权分
            atom_scores_390 = calculate_combined_weight(safe_w, ang_dec, dist_dec)

            # comb = calculate_combined_weight(safe_w, ang_dec, dist_dec)
            s_sum = np.sum(atom_scores_390); s_max = np.max(atom_scores_390)
            ml_390 = [s_sum, s_max, s_max/(s_sum+1e-6)]

            # 用于最后画图/统计 (辅助)
            all_angles_390.append(ang_dec)
            all_dist_decays_390.append(dist_dec)

        # C. 构建特征向量

        # 当前帧的原子特征矩阵: [N_atoms, 14]
        # 列: [Dist_1...Dist_12, Score_389, Score_390]
        current_frame_atoms = np.column_stack([
            atom_dists_389, atom_angles_389,
            atom_dists_390, atom_angles_390,
            atom_scores_389, atom_scores_390
        ])

        # 初始化距离特征矩阵（N_ATOM_FEAT x MAX_ATOMS）
        padded_frame = np.ones((MAX_ATOMS, N_ATOM_FEAT), dtype=np.float32) * 100.0 # 填充为大值
        padded_frame[:, [1, 3, 4, 5]] = 0.0  # 得分部分填充为0

        # 复制当前帧数据到填充矩阵
        n_real = min(len(lp_ring), MAX_ATOMS)
        padded_frame[:n_real, :] = current_frame_atoms[:n_real, :]

        # 展平为一维向量
        # 原子部分: 9 atoms x 6 features = 54维
        flat_atoms = padded_frame.flatten()  # 54维

        # 全局部分：Angle (1) + 389 scores (3) + 390 scores (3) = 7
        global_feats = np.concatenate([[ml_cos_angle], ml_389, ml_390])  # 7维

        # 最终特征向量: 54 + 7 + 9 = 70维
        feat = np.concatenate([flat_atoms, global_feats])  # 70维
        feature_vectors.append(feat)

        # D. 可视化对齐 & 数据记录
        p1 = c1 if c1 is not None else [0,0,0]
        p2 = c2 if c2 is not None else [0,0,0]
        tpos = [a.position for a in obp_atoms] + [p1, p2]
        
        _, pxy, _ = align_xy(lp_geo, np.array(tpos), lig_res.atoms.positions)
        vis_accum += pxy; cnt += 1
        
        # E. 记录时间序列数据
        row = {
            "Time": ts.time, "Replica": rep_name, "Global_Angle": ga,
            "Phe389_Score_Sum": ml_389[0], "Phe389_Score_Max": ml_389[1],
            "Phe390_Score_Sum": ml_390[0], "Phe390_Score_Max": ml_390[1]
            }
        data.append(row)

    df = pd.DataFrame(data)
    if feature_vectors:
        output_handler.save_features(np.array(feature_vectors, dtype=np.float32))

    if cnt > 0:
        axy = vis_accum / cnt
        ref_file = os.path.join(OUTPUT_BASE_DIR, "REFERENCE_DOPA_OBP.npy")
        is_dopa = "dopa" in cid.lower()
        if is_dopa and (not os.path.exists(ref_file) or rep_name == "replicate_1"):
            np.save(ref_file, axy)
            target_obp = axy
        elif os.path.exists(ref_file):
            target_obp = np.load(ref_file)
        
        proj_path = output_handler.save_projection(None)
        pdb_path = str(proj_path).replace('.png', '_structure.pdb')
        save_3d_structure_pdb(
            lig_res.atoms.positions, [a.name for a in lig_res.atoms],
            obp_atoms.positions, [f"RES{i}" for i in range(len(real_obp_ids))], real_obp_ids,
            pdb_path, cid,
            ring_atom_indices=md_ring_indices, 
            ring_weights=extracted_ring_weights
        )

    stats = {
        "Compound": cid, "Replica": rep_name, "Offset": offset,
        "Phe389_Score_Sum_Mean": df["Phe389_Score_Sum"].mean(), 
        "Phe390_Score_Sum_Mean": df["Phe390_Score_Sum"].mean()
    }
    
    strength = None
    if all_angles_389 and all_angles_390:
        strength = calculate_interaction_strength(
            safe_w, np.mean(all_angles_389, 0), np.mean(all_angles_390, 0),
            np.mean(all_dist_decays_389, 0), np.mean(all_dist_decays_390, 0)
        )
        if strength: stats.update(strength)

    u.trajectory.close(); del u, feature_vectors, data; plt.close('all'); gc.collect()
    return df, stats, strength

def main():
    # 【新增】参数解析
    parser = argparse.ArgumentParser(description="Extract Features from MD Trajectories")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing feature files if they exist.")
    args = parser.parse_args()
    
    root = "."
    if not os.path.exists(OUTPUT_BASE_DIR): os.makedirs(OUTPUT_BASE_DIR)
    GLOBAL_MAX = get_dopa_global_max(root)
    aligner = OffsetCalculator()
    all_dirs = glob.glob(os.path.join(root, "*"))
    all_dirs.sort(key=lambda x: (not "dopa" in os.path.basename(x).lower(), x))
    
    for c_dir in all_dirs:
        if not os.path.isdir(c_dir): continue
        if any(x in c_dir for x in ["modules", "results", "__pycache__"]): continue
        
        cid = os.path.basename(c_dir)
        cubs = glob.glob(os.path.join(c_dir, "*.cub"))
        pdbs = glob.glob(os.path.join(c_dir, "*.pdb"))
        ref_pdb = next((p for p in pdbs if "step7" not in p and "topol" not in p and "QC" not in p), None)
        
        if not cubs or not ref_pdb: 
            print(f"[Skip] {cid}: Missing .cub or reference .pdb")
            continue
        
        print(f"Processing: {cid}")
        
        qm_data = load_cube_and_map(cubs[0], ref_pdb, INTEGRATION_RADIUS)
        if not qm_data: 
            print(f"  [Skip] QM data loading failed for {cid}")
            continue

        try:
            rm = RingMatcher(qm_data['coords'], qm_data['elements'])
            ref_ring_indices = rm.ref_ring_idx 
            ring_type = rm.ring_type
            if rm.rings and 'six_ring' in rm.rings[0]:
                ref_geo_indices = rm.rings[0]['six_ring']
            else:
                ref_geo_indices = rm.ref_ring_idx
        except Exception as e:
            print(f"  [Error] Ring matching failed on Ref PDB: {e}")
            continue

        xtcs = glob.glob(os.path.join(c_dir, "**", "merged.xtc"), recursive=True)
        if not xtcs:
            print(f"  [Skip] No merged.xtc found for {cid}")
            continue

        ts_list = []; stat_list = []
        
        for xtc in xtcs:
            rd = os.path.dirname(xtc) # 例如 .../charmm-gui-6321/gromacs_replicate_1
            
            # 1. 获取基础 replicate 名 (gromacs_replicate_1)
            base_rn = os.path.basename(rd) 
            
            # 2. 获取上一级 charmm-gui 名 (charmm-gui-6321...)
            charmm_gui_id = os.path.basename(os.path.dirname(rd))
            
            # 3. 组合成唯一名称！
            # 结果示例: "charmm-gui-6321_gromacs_replicate_1"
            # 这样输出文件夹就会变成 data/features/Dopa/charmm-gui-6321_gromacs_replicate_1/
            unique_rn = f"{charmm_gui_id}_{base_rn}"
            
            tps = [os.path.join(rd, f) for f in os.listdir(rd) if f.endswith(".tpr")]
            topo = next((t for t in tps if "production" in t), tps[0] if tps else None)
            
            if topo:
                # 使用 unique_rn 初始化 OutputHandler
                output_handler = OutputHandler(cid, unique_rn, OUTPUT_BASE_DIR)

                # 【新增】检查文件是否存在
                if not args.overwrite and output_handler.check_features_exist():
                    print(f"  [Skip] Features exist for {unique_rn} (Use --overwrite to force)")
                    continue
                
                ts, st, strength = process_replicate(
                    xtc, topo, qm_data, 
                    ref_ring_indices, ref_geo_indices, ring_type,
                    cid, unique_rn, aligner, output_handler, GLOBAL_MAX
                )
                if ts is not None:
                    output_handler.save_timeseries(ts)
                    output_handler.save_stats(pd.DataFrame([st]))
                    ts_list.append(ts); stat_list.append(st)
            else:
                print(f"  [Skip] No suitable .tpr found in {base_rn}")

            gc.collect()
            
        if ts_list:
            OutputHandler.aggregate_timeseries(ts_list, OUTPUT_BASE_DIR, cid)
            OutputHandler.aggregate_stats(stat_list, OUTPUT_BASE_DIR, cid)

if __name__ == "__main__":
    main()