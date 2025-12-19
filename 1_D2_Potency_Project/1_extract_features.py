#!/usr/bin/env python3
"""
run_analysis_v3_robust.py
基于 1_1_extract_cub_features.py 的 QC 流程重构
结合 RingMatcher 精准提取环区电子特征
【功能更新】
1. QC_MD_MAP.pdb 仅高亮实际参与计算的环原子 (ref_ring_indices)，其他原子 B-factor 为 0。
2. 保持 QC_REF.pdb 显示全分子积分，作为原始数据参考。
"""

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
except ImportError as e:
    print(f"Error: 模块导入失败: {e}")
    sys.exit(1)

# ================= 核心配置区 =================
INTEGRATION_RADIUS = 1.5      
OUTPUT_BASE_DIR = "./data/features"
QC_OUTPUT_DIR = "./data/qc_structures" 

STANDARD_SEQUENCE = """
MDPLNLSWYDDDLERQNWSRPFNGSDGKADRPHYNYYATLLTLLIAVIVFGNVLVCMAVS
REKALQTTTNYLIVSLAVADLLVATLVMPWVVYLEVVGEWKFSRIHCDIFVTLDVMMCTA
SILNLCAISIDRYTAVAMPMLYNTRYSSKRRVTVMIAIVWVLSFTISCPLLFGLNNADQN
ECIIANPAFVVYSSIVSFYVPFIVTLLVYIKIYIVLRKRRKRVNTKRSSRAFRAHLRAPL
KGNCTHPEDMKLCTVIMKSNGSFPVNRRRVEAARRAQELEMEMLSSTSPPERTRYSPIPP
SHHQLTLPDPSHHGLHSTPDSPAKPEKNGHAKDHPKIAKIFEIQTMPNGKTRTSLKTMSR
RKLSQQKEKKATQMLAIVLGVFIICWLPFFITHILNIHCDCNIPPVLYSAFTWLGYVNSA
VNPIIYTTFNIEFRKAFLKILSC
"""

PHE_RESIDUES_STD = [389, 390]
OBP_RESIDUES_STD = [114, 115, 118, 119, 190, 193, 194, 197, 386, 389, 390, 393, 412, 416]
PLANE_RESIDUES_STD = [198, 163, 76, 127]

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

    offset = offset_calc.calculate_offset(u, 389)
    if offset is None: 
        print(f"     [Error] Sequence alignment failed (Offset is None).")
        return None, None, None

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
    obp_ids = [x + offset for x in OBP_RESIDUES_STD]
    plane_ids = [x + offset for x in PLANE_RESIDUES_STD]
    obp_atoms = u.select_atoms(f"resid {' '.join(map(str, obp_ids))} and name CA")
    plane_res = u.select_atoms(f"resid {' '.join(map(str, plane_ids))} and name CA")
    
    t389 = 389 + offset; t390 = 390 + offset
    r389 = u.select_atoms(f"resid {t389}"); r390 = u.select_atoms(f"resid {t390}")
    
    global_ring_indices = [lig_res.atoms[i].index for i in md_ring_indices]
    ring_ag = u.atoms[global_ring_indices]
    global_geo_indices = [lig_res.atoms[i].index for i in md_geo_indices]
    geo_ag = u.atoms[global_geo_indices]

    data = []; vis_accum = np.zeros((len(OBP_RESIDUES_STD)+2, 2)); cnt = 0
    feature_vectors = []
    all_angles_389 = []; all_angles_390 = []
    all_dist_decays_389 = []; all_dist_decays_390 = []

    # 3. Trajectory Loop

    # constant features defined
    MAX_ATOMS = 9
    N_OBP = len(OBP_RESIDUES_STD) # 14 个 OBP 残基
    N_ATOM_FEAT = N_OBP + 2 # 每个原子的特征数 = 14 (距离) + 1 (389得分) + 1 (390得分) = 16

    for ts in u.trajectory:
        lp_ring = ring_ag.positions # ligand ring positions
        lp_geo = geo_ag.positions
        lc_geo = lp_geo.mean(0) # Ligand Center (Geometry Atoms)
        safe_w = np.maximum(ring_weights_np, 0.0)

        # A. 几何角度计算
        ln = calculate_plane_normal(lp_geo)
        pn = calculate_plane_normal(plane_res.positions)
        dot_val = np.clip(np.dot(ln, pn), -1, 1)
        ga = np.degrees(np.arccos(dot_val)); ga = ga if ga<=90 else 180-ga
        ml_cos_angle = np.abs(dot_val)
        
        # B. 原子级特征提取
        # 1. 计算芳香环原子与OBP的距离矩阵[N_atoms x N_OBP]
        dists_obp = distance_array(lp_ring, obp_atoms.positions, box=u.dimensions)

        # 容错：如果 OBP 残基缺失，补 100.0
        if dists_obp.shape[1] < N_OBP:
            padded_dists = np.ones((len(lp_ring), N_OBP)) * 100.0
            padded_dists[:, :dists_obp.shape[1]] = dists_obp
            dists_obp = padded_dists

        # 2. 计算 389 相互作用 (保留每个原子的得分)
        c1, n1 = get_aromatic_ring_data(r389)
        atom_scores_389 = np.zeros(len(lp_ring)) # [N_atoms]
        ml_389 = [0.0, 0.0, 0.0]
        if c1 is not None:
            _, ang_dec = calculate_carbon_angles_and_decay(lp_ring, c1, n1)
            _, dist_dec = calculate_distance_decay(lp_ring, c1, n1)

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
        ml_390 = [0.0, 0.0, 0.0]
        if c2 is not None:
            _, ang_dec = calculate_carbon_angles_and_decay(lp_ring, c2, n2)
            _, dist_dec = calculate_distance_decay(lp_ring, c2, n2)

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
        current_frame_atoms = np.column_stack([dists_obp, atom_scores_389, atom_scores_390])

        # 初始化距离特征矩阵（N_ATOM_FEAT x MAX_ATOMS）
        padded_frame = np.ones((MAX_ATOMS, N_ATOM_FEAT), dtype=np.float32) * 100.0 # 填充为大值
        padded_frame[:, N_OBP:] = 0.0  # 得分部分填充为0

        # 复制当前帧数据到填充矩阵
        n_real = min(len(lp_ring), MAX_ATOMS)
        padded_frame[:n_real, :] = current_frame_atoms[:n_real, :]

        # 展平为一维向量
        # 原子部分: 9 atoms x 14 features = 126
        flat_atoms = padded_frame.flatten()  # 126维

        # 全局部分：Angle (1) + 389 scores (3) + 390 scores (3) = 7
        global_feats = np.concatenate([[ml_cos_angle], ml_389, ml_390])  # 7维

        # 最终特征向量: 126 + 7 = 133维
        feat = np.concatenate([flat_atoms, global_feats])  # 133维
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
            obp_atoms.positions, [f"RES{i}" for i in range(len(obp_ids))], obp_ids,
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
    root = "."
    if not os.path.exists(OUTPUT_BASE_DIR): os.makedirs(OUTPUT_BASE_DIR)
    GLOBAL_MAX = get_dopa_global_max(root)
    aligner = OffsetCalculator(STANDARD_SEQUENCE)
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
            rd = os.path.dirname(xtc); rn = os.path.basename(rd)
            tps = [os.path.join(rd, f) for f in os.listdir(rd) if f.endswith(".tpr")]
            topo = next((t for t in tps if "production" in t), tps[0] if tps else None)
            
            if topo:
                output_handler = OutputHandler(cid, rn, OUTPUT_BASE_DIR)
                ts, st, strength = process_replicate(
                    xtc, topo, qm_data, 
                    ref_ring_indices, ref_geo_indices, ring_type,
                    cid, rn, aligner, output_handler, GLOBAL_MAX
                )
                if ts is not None:
                    output_handler.save_timeseries(ts)
                    output_handler.save_stats(pd.DataFrame([st]))
                    ts_list.append(ts); stat_list.append(st)
            else:
                print(f"  [Skip] No suitable .tpr found in {rn}")

            gc.collect()
            
        if ts_list:
            OutputHandler.aggregate_timeseries(ts_list, OUTPUT_BASE_DIR, cid)
            OutputHandler.aggregate_stats(stat_list, OUTPUT_BASE_DIR, cid)

if __name__ == "__main__":
    main()