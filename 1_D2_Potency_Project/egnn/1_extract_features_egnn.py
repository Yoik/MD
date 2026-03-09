#!/usr/bin/env python3
import os
import sys
import glob
import argparse
import gc
import warnings
import numpy as np
import torch
import MDAnalysis as mda
from rdkit import Chem
from tqdm import tqdm
import tempfile

# 引入项目模块
try:
    from src.config import init_config
    from src.featurizer import PhysicsFeaturizer # <--- 引入新模块
    from modules.qm_loader import load_cube_and_map, find_ligand
    from modules.cube_parser import CubeParser
    from modules.ring_matcher import RingMatcher
except ImportError as e:
    print(f"Error: 模块导入失败: {e}")
    sys.exit(1)

# ================= 配置加载 =================
config = init_config()
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

INTEGRATION_RADIUS = config.get_float("data.integration_radius")
OUTPUT_BASE_DIR = config.get_path("paths.result_dir")
# REF_PDB_PATH, OBP_BW_LIST 等已移入 Featurizer

# ================= 原版 Mapping 函数 (严格保留你的版本) =================
def get_rdkit_mapping(ref_pdb_path, mda_ligand_atoms):
    """
    计算从 Reference PDB (QM) 到 MD Analysis Ligand 的原子索引映射。
    (保持原版逻辑，未添加额外增强)
    """
    def get_skeleton(mol):
        m = Chem.Mol(mol)
        for b in m.GetBonds():
            b.SetBondType(Chem.BondType.SINGLE)
            b.SetIsAromatic(False)
        for a in m.GetAtoms():
            a.SetIsAromatic(False)
        return m

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
    if not ref_mol: return None, None

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
    except: return None, None
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try: os.remove(tmp_path)
            except: pass

    if not target_mol: return None, None

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
            mapping = {}
            for ref_idx, target_idx in enumerate(match):
                mapping[ref_idx] = target_idx
            return mapping, ref_mol
        else:
            return None, None
    except: return None, None

def get_dopa_global_max(root_dir):
    print(">>> Searching for Dopa reference...")
    all_dirs = glob.glob(os.path.join(root_dir, "*"))
    for c_dir in all_dirs:
        if not os.path.isdir(c_dir): continue
        if "dopa" in os.path.basename(c_dir).lower():
            cubs = glob.glob(os.path.join(c_dir, "*.cub"))
            if cubs:
                try:
                    cp = CubeParser(cubs[0])
                    integrals = cp.get_carbon_integrals(INTEGRATION_RADIUS)
                    if len(integrals) > 0: return np.max(integrals)
                except: pass
    return 1.0

# ================= 核心处理逻辑 (改为调用 Featurizer) =================

def process_compound_replicates(cid, c_dir, featurizer, global_max, args):
    """
    处理单个化合物的所有副本
    """
    # 1. 准备 QM 数据
    cubs = glob.glob(os.path.join(c_dir, "*.cub"))
    if not cubs:
        print(f"  [Skip] {cid}: No .cub file found (QM density required).")
        return
    pdbs = glob.glob(os.path.join(c_dir, "*.pdb"))
    if not pdbs:
        print(f"  [Skip] {cid}: No .pdb file found (QM reference required).")
        return
    qm_ref_pdb = next((p for p in pdbs if "step7" not in p and "topol" not in p and "QC" not in p), None)
    if not qm_ref_pdb:
        print(f"  [Skip] {cid}: No valid reference .pdb found (excluding step7/topol/QC).")
        return
    if not cubs or not qm_ref_pdb: return

    print(f"Processing Compound: {cid}")
    qm_data = load_cube_and_map(cubs[0], qm_ref_pdb, INTEGRATION_RADIUS)
    if not qm_data: 
        print(f"  [Skip] {cid}: Failed to load QM data from .cub or map to reference.")
        return

    # 2. 准备 Ring Matcher (逻辑来自原脚本)
    qm_ring_indices = []
    try:
        rm = RingMatcher(qm_data['coords'], qm_data['elements'])
        if rm.rings and 'six_ring' in rm.rings[0]:
            qm_ring_indices = rm.rings[0]['six_ring']
        else:
            qm_ring_indices = rm.ref_ring_idx
    except: pass

    xtcs = glob.glob(os.path.join(c_dir, "**", "merged.xtc"), recursive=True)
    
    for xtc in xtcs:
        rd = os.path.dirname(xtc)
        charmm_gui_id = os.path.basename(os.path.dirname(rd))
        base_rn = os.path.basename(rd)
        unique_rn = f"{charmm_gui_id}_{base_rn}"
        
        save_dir = os.path.join(OUTPUT_BASE_DIR, cid, unique_rn)
        if not os.path.exists(save_dir): os.makedirs(save_dir)
        save_path = os.path.join(save_dir, "graph_features.pt")
        
        if os.path.exists(save_path) and not args.overwrite:
            print(f"  [Skip] Exists: {unique_rn}")
            continue

        tps = [os.path.join(rd, f) for f in os.listdir(rd) if f.endswith(".tpr")]
        topo = next((t for t in tps if "production" in t), tps[0] if tps else None)
        
        if not topo: continue

        try:
            u = mda.Universe(topo, xtc)
            lig_res = find_ligand(u)
            if not lig_res: continue
            
            # 3. 计算 Raw Weights (基于 Mapping)
            mapping, _ = get_rdkit_mapping(qm_data['pdb_path'], lig_res.atoms)
            if mapping is None: 
                print(f"  [Skip] Mapping failed for {unique_rn}")
                continue
                
            integrals = qm_data['integrals']
            norm_integrals = integrals / global_max
            raw_weights = np.zeros(len(lig_res.atoms))
            
            # Map Weights
            for qm_idx, md_idx in mapping.items():
                if qm_idx < len(norm_integrals):
                    raw_weights[md_idx] = norm_integrals[qm_idx]
            
            # Map Ring Indices
            md_ring_indices = []
            for r_idx in qm_ring_indices:
                if r_idx in mapping: md_ring_indices.append(mapping[r_idx])

            # 4. 遍历帧并调用 Featurizer
            graph_list = []
            stride = 5  # 可调整的帧间隔，减少计算量
            for ts in tqdm(u.trajectory[::stride], desc=f"  Extr. {unique_rn}", leave=False):
                try:
                    # === 调用 Featurizer (传入 Raw Weights，内部做衰减) ===
                    data = featurizer.process_frame(u, lig_res, raw_weights, md_ring_indices)
                    if data: graph_list.append(data)
                except Exception:
                    pass

            if graph_list:
                torch.save(graph_list, save_path)
                print(f"  [Saved] {len(graph_list)} frames -> {save_path}")
                
            del u
            gc.collect()
            
        except Exception as e:
            print(f"  [Error] Processing {unique_rn}: {e}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    # 初始化 Featurizer
    print("Initializing Physics Featurizer...")
    featurizer = PhysicsFeaturizer(config)

    root = PROJECT_ROOT
    if not os.path.exists(OUTPUT_BASE_DIR): os.makedirs(OUTPUT_BASE_DIR)
    
    GLOBAL_MAX = get_dopa_global_max(root)
    print(f"Global Normalization Factor: {GLOBAL_MAX:.4f}")

    all_dirs = glob.glob(os.path.join(root, "*"))
    all_dirs.sort()

    for c_dir in all_dirs:
        if not os.path.isdir(c_dir): continue
        if any(x in c_dir for x in ["modules", "results", "__pycache__", "data", "saved_models", "src"]): continue
        
        cid = os.path.basename(c_dir)
        process_compound_replicates(cid, c_dir, featurizer, GLOBAL_MAX, args)

if __name__ == "__main__":
    main()