import os
import sys
import glob
import torch
import numpy as np
import pickle
import MDAnalysis as mda
from MDAnalysis.lib.distances import distance_array
from rdkit import Chem
from rdkit import RDLogger
import warnings
import tempfile

# 屏蔽 RDKit 干扰日志
RDLogger.DisableLog('rdApp.*')

# 引入项目模块
from src.model import EfficiencyPredictor

# 尝试导入必要的模块
try:
    from modules.qm_loader import load_cube_and_map
    from modules.ring_matcher import RingMatcher
    from modules.sequence_aligner import OffsetCalculator
    from modules import (
        get_aromatic_ring_data,
        calculate_plane_normal,
        calculate_carbon_angles_and_decay,
        calculate_distance_decay,
        calculate_combined_weight
    )
except ImportError as e:
    print(f"[Critical Error] Module import failed: {e}")
    sys.exit(1)

# ================= 配置参数 =================
MODEL_PATH = "saved_models/best_model_mccv.pth" 
SCALER_PATH = "saved_models/scaler.pkl"
INPUT_ROOT = "predict"

INPUT_DIM = 151
INTEGRATION_RADIUS = 1.5 
GLOBAL_MAX_REF = 1.0 

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

OBP_RESIDUES_STD = [114, 115, 118, 119, 190, 193, 194, 197, 386, 389, 390, 393, 412, 416]
PHE_RESIDUES_STD = [389, 390]
PLANE_RESIDUES_STD = [198, 163, 76, 127]
MAX_ATOMS = 9
N_OBP = len(OBP_RESIDUES_STD)

# ================= 极简工具函数 =================

def find_ligand_boltz(u):
    """
    针对 Boltz 预测结果的极简配体搜索
    只找 LIG 或 LIG1
    """
    # 1. 优先找 Boltz 标准命名
    ligands = u.select_atoms("resname LIG LIG1")
    if len(ligands) > 0:
        return ligands.residues[0]
    
    # 2. 兜底：找任何非蛋白的小分子 (防止 Boltz 改名)
    # 假设配体残基名不是常见氨基酸
    protein_residues = [
        "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
        "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL"
    ]
    candidates = u.select_atoms(f"not resname {' '.join(protein_residues)} and not resname WAT SOL TIP3")
    
    # 过滤掉单原子 (如离子)
    real_cands = [r for r in candidates.residues if len(r.atoms) > 3]
    
    if real_cands:
        # 返回原子数最多的那个 (通常是配体)
        real_cands.sort(key=lambda r: len(r.atoms), reverse=True)
        return real_cands[0]
        
    return None

def convert_cif_to_pdb(cif_path, pdb_path):
    """(保留) 手动解析 CIF 转换为 PDB"""
    try:
        with open(cif_path, 'r') as f: lines = f.readlines()
        loop_indices = {}; headers = []; data_start_idx = -1; in_atom_site_loop = False
        for i, line in enumerate(lines):
            line = line.strip()
            if line.startswith('loop_'):
                j = i + 1; is_atom = False; temp_headers = []
                while j < len(lines) and lines[j].strip().startswith('_'):
                    key = lines[j].strip()
                    if key.startswith('_atom_site.'):
                        is_atom = True; temp_headers.append(key.replace('_atom_site.', ''))
                    j += 1
                if is_atom:
                    in_atom_site_loop = True; headers = temp_headers
                    for k, h in enumerate(headers): loop_indices[h] = k
                    data_start_idx = j; break
        if not in_atom_site_loop: return False
        def get_val(parts, key1, key2=None, default=''):
            idx = loop_indices.get(key1, loop_indices.get(key2, -1))
            if idx != -1 and idx < len(parts): return parts[idx]
            return default
        atom_count = 0
        with open(pdb_path, 'w') as f:
            for i in range(data_start_idx, len(lines)):
                line = lines[i].strip()
                if not line or line.startswith('#') or line.startswith('loop_'): break
                parts = line.split()
                if len(parts) < len(headers): continue 
                atom_name = get_val(parts, 'auth_atom_id', 'label_atom_id', 'C')
                res_name = get_val(parts, 'auth_comp_id', 'label_comp_id', 'UNK')[:3]
                chain_id = get_val(parts, 'auth_asym_id', 'label_asym_id', 'A')[:1]
                res_seq = get_val(parts, 'auth_seq_id', 'label_seq_id', '1')
                try: x, y, z = float(get_val(parts, 'Cartn_x')), float(get_val(parts, 'Cartn_y')), float(get_val(parts, 'Cartn_z'))
                except: x,y,z = 0.0,0.0,0.0
                element = get_val(parts, 'type_symbol', default='C')
                atom_count += 1
                fmt_name = f" {atom_name:<3}" if len(atom_name)<4 else f"{atom_name[:4]:<4}"
                f.write(f"ATOM  {atom_count:5d} {fmt_name} {res_name:<3} {chain_id:1s}{res_seq:4s}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          {element:>2s}\n")
        return atom_count > 0
    except Exception: return False

class RobustMatcher:
    def __init__(self, ref_pdb_path):
        # 1. 强力去氢：即使 RDKit 没识别出 H，只要名字带 H 就删
        raw_mol = Chem.MolFromPDBFile(ref_pdb_path, removeHs=False, sanitize=False)
        if not raw_mol:
            self.ref_skel = None; return

        # 使用 RDKit 的可编辑分子对象删除氢
        rw_mol = Chem.RWMol(raw_mol)
        atoms_to_remove = []
        for atom in raw_mol.GetAtoms():
            # 判定标准：原子序数为1 OR 名字以 H 开头 (针对不规范 PDB)
            if atom.GetAtomicNum() == 1 or atom.GetSymbol() == 'H' or atom.GetPDBResidueInfo().GetName().strip().startswith('H'):
                atoms_to_remove.append(atom.GetIdx())
        
        # 倒序删除防止索引错乱
        atoms_to_remove.sort(reverse=True)
        for idx in atoms_to_remove:
            rw_mol.RemoveAtom(idx)
            
        self.ref_mol = rw_mol.GetMol()
        self.ref_skel = self.get_skeleton(self.ref_mol)
        self.ref_atoms = self.ref_mol.GetNumAtoms()

    def get_skeleton(self, mol):
        m = Chem.Mol(mol)
        try: m.UpdatePropertyCache(strict=False)
        except: pass
        for b in m.GetBonds():
            b.SetBondType(Chem.BondType.SINGLE); b.SetIsAromatic(False)
        for a in m.GetAtoms(): a.SetIsAromatic(False)
        return m

    def match(self, target_atom_group):
        if not self.ref_skel: return None, None, None
        
        # 1. 过滤 Target 中的 H (Boltz 通常没有，但以防万一)
        sel = target_atom_group.select_atoms("not (name H* or name 1H* or name 2H* or name LP*)")
        
        # 2. 转 RDKit
        with tempfile.NamedTemporaryFile(suffix=".pdb", mode='w+', delete=True) as tmp:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                sel.write(tmp.name)
            target_mol = Chem.MolFromPDBFile(tmp.name, removeHs=True, sanitize=False)
            
        if not target_mol: return None, None, None

        # 3. 骨架匹配
        target_skel = self.get_skeleton(target_mol)
        matches = target_skel.GetSubstructMatches(self.ref_skel, uniquify=False)
        
        if not matches:
            # print(f"    [Debug] Match failed. Ref Heavy: {self.ref_atoms}, Target Heavy: {target_mol.GetNumAtoms()}")
            return None, None, None

        best_match = matches[0]
        mapped_ag_atoms = []
        ref_indices = []
        
        for ref_idx, target_idx in enumerate(best_match):
            mapped_ag_atoms.append(sel[target_idx])
            ref_indices.append(ref_idx)
            
        return mapped_ag_atoms, ref_indices, None

# ================= 智能文件识别 =================
def identify_inputs(input_dir):
    if not os.path.exists(input_dir): return None, None, None, "No input folder"
    files = os.listdir(input_dir)
    cub = next((f for f in files if f.endswith('.cub')), None)
    pdbs = [f for f in files if f.endswith('.pdb') or f.endswith('.cif')]
    
    if not cub or not pdbs: return None, None, None, "Missing files"
    cub_path = os.path.join(input_dir, cub)
    
    # 简单策略：Ref通常有 ref/qm 字样，或者文件较小(只有配体)
    # Target通常有 boltz/pred 字样，或者文件较大(复合物)
    pdbs.sort(key=lambda f: os.path.getsize(os.path.join(input_dir, f)))
    ref_pdb = None; target_pdb = None
    
    for f in pdbs:
        path = os.path.join(input_dir, f)
        name = f.lower()
        if "ref" in name or "qm" in name or "lig" in name: ref_pdb = path
        if "boltz" in name or "pred" in name or "complex" in name: target_pdb = path

    # 兜底：最小的是 Ref，最大的是 Target
    if not ref_pdb: ref_pdb = os.path.join(input_dir, pdbs[0])
    if not target_pdb: target_pdb = os.path.join(input_dir, pdbs[-1])

    return cub_path, ref_pdb, target_pdb, None

# ================= 特征提取核心 =================
def extract_robust_features(cub_path, ref_pdb_path, target_pdb_path, aligner):
    # 1. QM
    qm_data = load_cube_and_map(cub_path, ref_pdb_path, INTEGRATION_RADIUS)
    if not qm_data: return None

    # 2. Ref Rings
    try:
        rm = RingMatcher(qm_data['coords'], qm_data['elements'])
        ref_ring_indices = rm.ref_ring_idx
        ref_geo_indices = rm.rings[0]['six_ring'] if (rm.rings and 'six_ring' in rm.rings[0]) else rm.ref_ring_idx
    except: return None

    # 3. Target (Boltz)
    u = None; temp_pdb = None
    try:
        if target_pdb_path.lower().endswith('.cif'):
            fd, temp_pdb = tempfile.mkstemp(suffix=".pdb"); os.close(fd)
            if not convert_cif_to_pdb(target_pdb_path, temp_pdb): 
                if temp_pdb: os.remove(temp_pdb)
                return None
            u = mda.Universe(temp_pdb)
        else:
            u = mda.Universe(target_pdb_path)
    except: return None

    # 4. Align
    offset = aligner.calculate_offset(u, 389)
    if offset is None: 
        if temp_pdb: os.remove(temp_pdb)
        return None
    
    lig_res = find_ligand_boltz(u)
    if not lig_res:
        if temp_pdb: os.remove(temp_pdb)
        return None

    # 5. Match
    try:
        matcher = RobustMatcher(ref_pdb_path)
        mapped_ag, ref_match_indices, _ = matcher.match(lig_res.atoms)
        if not mapped_ag: raise ValueError("Match fail")
        ref_to_target_map = {r: t for r, t in zip(ref_match_indices, mapped_ag)}
    except:
        if temp_pdb: os.remove(temp_pdb)
        return None

    # 6. Transfer & Compute
    # (Extract coordinates and weights)
    md_ring = []; md_geo = []; w = []
    for r in ref_ring_indices:
        if r in ref_to_target_map:
            md_ring.append(ref_to_target_map[r]); w.append(qm_data['integrals'][r]/GLOBAL_MAX_REF)
    for g in ref_geo_indices:
        if g in ref_to_target_map: md_geo.append(ref_to_target_map[g])
        
    if not md_ring: 
        if temp_pdb: os.remove(temp_pdb)
        return None

    lp_ring = np.array([a.position for a in md_ring])
    lp_geo = np.array([a.position for a in md_geo])
    safe_w = np.maximum(np.array(w), 0.0)

    # Residues
    def get_ca(ids): 
        # 增加容错：如果某个残基没找到 CA，就不计入，防止报错
        pos = []
        for i in ids:
            sel = u.select_atoms(f"resid {i+offset} and name CA")
            if len(sel) > 0: pos.append(sel[0].position)
            else: pos.append([0,0,0]) # 缺失补0
        return np.array(pos)

    obp_pos = get_ca(OBP_RESIDUES_STD)
    plane_pos = get_ca(PLANE_RESIDUES_STD)
    r389 = u.select_atoms(f"resid {389 + offset}")
    r390 = u.select_atoms(f"resid {390 + offset}")

    # Features
    ml_cos = 0.0
    if len(lp_geo)>=3 and np.any(plane_pos):
        ln = calculate_plane_normal(lp_geo)
        pn = calculate_plane_normal(plane_pos)
        ml_cos = np.abs(np.clip(np.dot(ln, pn), -1, 1))

    dists = distance_array(lp_ring, obp_pos)
    # Pad if OBP residues missing
    if dists.shape[1] < N_OBP:
        pad = np.ones((len(lp_ring), N_OBP))*100.0
        pad[:, :dists.shape[1]] = dists
        dists = pad

    def score(res):
        c, n = get_aromatic_ring_data(res)
        if c is None: return [0,0,0], np.zeros(len(lp_ring))
        _, ang = calculate_carbon_angles_and_decay(lp_ring, c, n)
        _, dist = calculate_distance_decay(lp_ring, c, n)
        s = calculate_combined_weight(safe_w, ang, dist)
        return [np.sum(s), np.max(s), np.max(s)/(np.sum(s)+1e-6)], s

    m389, s389 = score(r389)
    m390, s390 = score(r390)

    if temp_pdb: os.remove(temp_pdb)

    # Flatten
    N_FEAT = N_OBP + 2
    cur = np.column_stack([dists, s389, s390])
    pad_frame = np.ones((MAX_ATOMS, N_FEAT), dtype=np.float32) * 100.0
    pad_frame[:, N_OBP:] = 0.0
    n = min(len(lp_ring), MAX_ATOMS)
    pad_frame[:n, :] = cur[:n, :]
    
    return np.concatenate([pad_frame.flatten(), [ml_cos], m389, m390])

# ================= 主程序 =================
def main():
    device = torch.device("cpu")
    print(f">>> Initializing Simplified Predictor...")
    
    if not os.path.exists(MODEL_PATH): print("Model missing."); return
    model = EfficiencyPredictor(input_dim=INPUT_DIM).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    with open(SCALER_PATH, 'rb') as f: scaler = pickle.load(f)
    aligner = OffsetCalculator(STANDARD_SEQUENCE)
    
    if not os.path.exists(INPUT_ROOT): print("Input root missing."); return
    dirs = [d for d in glob.glob(os.path.join(INPUT_ROOT, "*")) if os.path.isdir(d)]
    
    print("\n" + "="*95)
    print(f"{'Compound':<20} | {'Score':<10} | {'Status':<15} | {'Info (Ref -> Target)'}")
    print("-" * 95)

    for c_dir in dirs:
        c_name = os.path.basename(c_dir)
        cub, ref, tgt, err = identify_inputs(os.path.join(c_dir, "input"))
        
        if err:
            print(f"{c_name:<20} | {'N/A':<10} | {err:<15} | -")
            continue
            
        rname = os.path.basename(ref); tname = os.path.basename(tgt)
        
        try:
            feat = extract_robust_features(cub, ref, tgt, aligner)
            if feat is None:
                print(f"{c_name:<20} | {'N/A':<10} | {'Match Fail':<15} | {rname} -> {tname}")
                continue
                
            proc = feat.copy()
            for i in range(MAX_ATOMS):
                s, e = i*16, i*16+14
                proc[s:e] = 1.0 / (proc[s:e] + 1e-6)
            
            inp = torch.from_numpy(scaler.transform(proc.reshape(1,-1))).float().unsqueeze(0).to(device)
            with torch.no_grad(): score = model(inp)["pred"].item()
            
            print(f"{c_name:<20} | {score:<10.2f} | {'Success':<15} | {rname} -> {tname}")
            
        except Exception as e:
            print(f"{c_name:<20} | {'Error':<10} | {'Crash':<15} | {str(e)[:40]}")

if __name__ == "__main__":
    main()