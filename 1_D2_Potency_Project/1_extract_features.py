#!/usr/bin/env python3
"""
run_analysis_v2.py
改进的主分析脚本 - 模块化设计，输出到统一目录

功能：
1. 自动对齐序列并计算偏移量
2. 提取ELF电子密度（立方体文件）
3. 计算配体与Phe环的三层加权相互作用
4. 输出详细时间序列和统计数据
5. 计算综合相互作用强度
"""

import matplotlib
matplotlib.use('Agg')

import MDAnalysis as mda
from MDAnalysis.lib.distances import distance_array
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
import os
import itertools
from scipy.spatial.distance import cdist

from modules import (
    get_aromatic_ring_data,
    calculate_plane_normal,
    calculate_carbon_angles_and_decay,
    calculate_distance_decay,
    calculate_combined_weight,
    calculate_weighted_average_distance,
    calculate_interaction_strength,
    format_interaction_strength,
    OutputHandler
)

try:
    from Bio import Align
except ImportError:
    print("Error: Please install biopython! Run: pip install biopython")
    exit()
# ==============================================================================
# 1. 用户配置区
# ==============================================================================

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
OBP_RESIDUES_STD = [114, 115, 118, 119, 190, 193, 194, 197, 386, 393, 412, 416]
PLANE_RESIDUES_STD = [198, 163, 76, 127]
ANCHOR_RESID_STD = 389

T_STACK_DIST_CUTOFF = 6.5 
T_STACK_ANGLE_CENTER = 90.0
T_STACK_ANGLE_TOL = 30.0 
INTEGRATION_RADIUS = 1.5 
GLOBAL_DOPA_MAX_INTEGRAL = 1.0 
BOHR_TO_ANGSTROM = 0.52917721067

MANUAL_ATOM_OVERRIDES = {
    "unc": ['C16', 'C13', 'C8', 'C7', 'C12', 'C15']
}

OUTPUT_BASE_DIR = "./data/features"

# ==============================================================================
# 2. 序列对齐模块
# ==============================================================================

class OffsetCalculator:
    def __init__(self, standard_seq_str):
        self.ref_seq = "".join(standard_seq_str.split())
        self.three_to_one = {
            'ALA':'A', 'ARG':'R', 'ASN':'N', 'ASP':'D', 'CYS':'C',
            'GLN':'Q', 'GLU':'E', 'GLY':'G', 'HIS':'H', 'HSD':'H', 'HSE':'H', 'HSP':'H',
            'ILE':'I', 'LEU':'L', 'LYS':'K', 'MET':'M', 'PHE':'F',
            'PRO':'P', 'SER':'S', 'THR':'T', 'TRP':'W', 'TYR':'Y', 'VAL':'V'
        }

    def get_sim_sequence(self, u):
        protein = u.select_atoms("protein and name CA")
        resnames = protein.resnames
        resids = protein.resids
        
        seq_str = ""
        valid_indices = []
        
        for i, res in enumerate(resnames):
            code = self.three_to_one.get(res, 'X')
            seq_str += code
            valid_indices.append(resids[i])
            
        return seq_str, valid_indices

    def calculate_offset(self, u, target_std_id):
        sim_seq, sim_resids = self.get_sim_sequence(u)
        
        aligner = Align.PairwiseAligner()
        aligner.mode = 'global'
        aligner.match_score = 2
        aligner.mismatch_score = -1
        aligner.open_gap_score = -0.5
        aligner.extend_gap_score = -0.1
        
        alignments = aligner.align(self.ref_seq, sim_seq)
        best_aln = alignments[0]
        
        mapping = {}
        aligned_ref = best_aln.aligned[0]
        aligned_sim = best_aln.aligned[1]
        
        for (r_start, r_end), (s_start, s_end) in zip(aligned_ref, aligned_sim):
            length = r_end - r_start
            
            for i in range(length):
                r_idx = r_start + i
                s_idx = s_start + i
                std_res_num = r_idx + 1
                
                if s_idx < len(sim_resids):
                    sim_resid_val = sim_resids[s_idx]
                    mapping[std_res_num] = sim_resid_val
        
        if target_std_id in mapping:
            found_sim_resid = mapping[target_std_id]
            offset = found_sim_resid - target_std_id
            print(f"     [Align Info] Std {target_std_id} aligns to Sim {found_sim_resid}. Offset = {offset}")
            return offset
        else:
            print(f"     [Align Error] Residue {target_std_id} is a GAP in the simulation!")
            return None

# ==============================================================================
# 3. Cube和Ring匹配类（保持不变）
# ==============================================================================

class CubeParser:
    def __init__(self, filepath):
        self.filepath = filepath
        self.data = None; self.origin = None; self.spacing = None; self.dims = None
        self.atom_lines = []; self.is_header_bohr = True 
        self._load()
    def _load(self):
        try:
            with open(self.filepath, 'r') as f:
                lines = f.readlines()
                parts = lines[2].split(); natoms = int(parts[0])
                if natoms > 0: self.is_header_bohr = True
                else: self.is_header_bohr = False; natoms = abs(natoms)
                self.origin = np.array([float(x) for x in parts[1:4]])
                nx = int(lines[3].split()[0]); vx = np.array([float(x) for x in lines[3].split()[1:4]])
                ny = int(lines[4].split()[0]); vy = np.array([float(x) for x in lines[4].split()[1:4]])
                nz = int(lines[5].split()[0]); vz = np.array([float(x) for x in lines[5].split()[1:4]])
                self.dims = (nx, ny, nz); self.spacing = np.array([vx[0], vy[1], vz[2]]) 
                self.atom_lines = lines[6 : 6 + natoms]
                data_start = 6 + natoms
                raw_data = []
                for line in lines[data_start:]: raw_data.extend([float(x) for x in line.split()])
                self.data = np.array(raw_data).reshape(self.dims)
        except Exception as e: print(f"     [Cube Error] {e}"); self.data = None
    def get_carbon_integrals(self, radius=1.5):
        if self.data is None: return np.array([])
        origin_ang = self.origin * BOHR_TO_ANGSTROM
        spacing_ang = self.spacing * BOHR_TO_ANGSTROM
        integrals = []; nx, ny, nz = self.dims
        for line in self.atom_lines:
            parts = line.split()
            if int(parts[0]) == 6:
                raw_coord = np.array([float(x) for x in parts[2:5]])
                atom_coord_ang = raw_coord * BOHR_TO_ANGSTROM if self.is_header_bohr else raw_coord
                min_idx = np.maximum(np.floor((atom_coord_ang - radius - origin_ang) / spacing_ang).astype(int), 0)
                max_idx = np.minimum(np.ceil((atom_coord_ang + radius - origin_ang) / spacing_ang).astype(int) + 1, [nx, ny, nz])
                if np.any(min_idx >= max_idx): integrals.append(0.0); continue
                local_data = self.data[min_idx[0]:max_idx[0], min_idx[1]:max_idx[1], min_idx[2]:max_idx[2]]
                ix = np.arange(min_idx[0], max_idx[0]); iy = np.arange(min_idx[1], max_idx[1]); iz = np.arange(min_idx[2], max_idx[2])
                X, Y, Z = np.meshgrid(ix, iy, iz, indexing='ij')
                grid_pos_x = origin_ang[0] + X * spacing_ang[0]
                grid_pos_y = origin_ang[1] + Y * spacing_ang[1]
                grid_pos_z = origin_ang[2] + Z * spacing_ang[2]
                dist_sq = (grid_pos_x - atom_coord_ang[0])**2 + (grid_pos_y - atom_coord_ang[1])**2 + (grid_pos_z - atom_coord_ang[2])**2
                mask = dist_sq < (radius**2)
                integrals.append(np.sum(local_data[mask]))
        return np.array(integrals)

class RingMatcher:
    def __init__(self, ref_coords, ref_elements):
        self.ref_coords = ref_coords; self.ref_elements = ref_elements; self.n_ref = len(ref_coords)
        self.ref_ring_idx = self._find_single_ring(ref_coords, ref_elements)
        if self.ref_ring_idx is None:
            c_indices = [i for i, e in enumerate(ref_elements) if e == 'C']
            if len(c_indices) == 6: self.ref_ring_idx = c_indices
            else: raise ValueError(f"No 6-ring found")
        self.ref_neigh_idx = []
        dmat = cdist(ref_coords, ref_coords)
        ring_set = set(self.ref_ring_idx)
        for i in range(self.n_ref):
            if i not in ring_set and np.min(dmat[i, self.ref_ring_idx]) < 1.70: self.ref_neigh_idx.append(i)
    def _find_single_ring(self, coords, elements):
        c_indices = [i for i, e in enumerate(elements) if e == 'C']
        if len(c_indices) < 6: return None
        sub_coords = coords[c_indices]; dmat = cdist(sub_coords, sub_coords); adj = np.logical_and(dmat > 1.1, dmat < 1.65)
        for comb in itertools.combinations(range(len(c_indices)), 6):
            sub_idx = list(comb); curr_coords = sub_coords[sub_idx]
            if np.linalg.svd(curr_coords - curr_coords.mean(0))[1][2] > 0.3: continue 
            sub_adj = adj[np.ix_(sub_idx, sub_idx)]
            if np.all(np.sum(sub_adj, axis=1) >= 2): return [c_indices[i] for i in self._order_ring_indices(sub_idx, sub_adj)]
        return None
    def _order_ring_indices(self, indices, sub_adj):
        ordered = [indices[0]]; current = 0; used = {0}
        for _ in range(5):
            for n in np.where(sub_adj[current])[0]:
                if n not in used: ordered.append(indices[n]); used.add(n); current = n; break
        return ordered
    def match(self, md_atoms, anchor_com):
        md_coords = md_atoms.positions; md_c_indices = [i for i, a in enumerate(md_atoms) if a.name.startswith('C')]
        if len(md_c_indices) < 6: return None, None
        md_c_coords = md_coords[md_c_indices]; dmat = cdist(md_c_coords, md_c_coords); adj = np.logical_and(dmat > 1.1, dmat < 1.70)
        found_rings = []
        seen = set()
        def dfs(s,c,p):
            if len(p)==6: return p if adj[c,s] else None
            for n in np.where(adj[c])[0]:
                if n==s and len(p)<5: continue
                if n not in p: 
                    r = dfs(s,n,p+[n])
                    if r: return r
            return None
        for i in range(len(md_c_indices)):
            res = dfs(i,i,[i])
            if res:
                s=tuple(sorted(res)); 
                if s not in seen: found_rings.append(list(res)); seen.add(s)
        if not found_rings: return None, None
        
        best_ring = None; min_dist = float('inf')
        for r in found_rings:
            g = [md_c_indices[i] for i in r]; cent = md_coords[g].mean(0); d = np.linalg.norm(cent - anchor_com)
            if d < min_dist: min_dist = d; best_ring = g
        
        ref_ring = self.ref_coords[self.ref_ring_idx]; md_ring = md_coords[best_ring]
        ref_neigh = self.ref_coords[self.ref_neigh_idx] if self.ref_neigh_idx else None
        best_p = None; min_score = float('inf')
        shifts = [list(range(6))[i:]+list(range(6))[:i] for i in range(6)]; perms = shifts + [s[::-1] for s in shifts]
        for p in perms:
            tgt = md_ring[list(p)]; Pc=ref_ring.mean(0); Qc=tgt.mean(0); H=np.dot((ref_ring-Pc).T,(tgt-Qc))
            U,S,Vt=np.linalg.svd(H); R=np.dot(Vt.T,U.T)
            if np.linalg.det(R)<0: Vt[2,:]*=-1; R=np.dot(Vt.T,U.T)
            t=Qc-np.dot(Pc,R.T)
            rms = np.mean(np.linalg.norm((np.dot(ref_ring,R.T)+t)-tgt,axis=1))
            if rms > 0.5: continue
            score = rms
            if ref_neigh is not None:
                tn = np.dot(ref_neigh,R.T)+t; dchk = cdist(tn, md_coords)
                score = np.mean(np.min(dchk,axis=1))
            if score < min_score: min_score = score; best_p = p
        
        if best_p is None: return None, None
        ref_c_idxs = [i for i, e in enumerate(self.ref_elements) if e == 'C']
        cube_idxs = [{idx:rank for rank,idx in enumerate(ref_c_idxs)}[i] for i in self.ref_ring_idx]
        return md_atoms[[best_ring[i] for i in best_p]], cube_idxs

# ==============================================================================
# 4. 辅助函数
# ==============================================================================

def get_ref_data_from_pdb(pdb_file):
    c=[]; e=[]
    try:
        with open(pdb_file) as f:
            for l in f:
                if l.startswith("ATOM") or l.startswith("HETATM"):
                    n=l[12:16].strip(); 
                    if n.startswith("C") and not n.startswith("CL") and not n.startswith("CA"):
                        c.append([float(l[30:38]),float(l[38:46]),float(l[46:54])]); e.append(n[0])
    except: return None, None
    return np.array(c), e

def get_dopa_max_integral(root):
    mx=1.0; fd=False
    for d in glob.glob(os.path.join(root,"*")):
        if os.path.isdir(d) and "dopa" in os.path.basename(d).lower():
            cubs = glob.glob(os.path.join(d,"*.cub"))
            if cubs:
                cp = CubeParser(cubs[0]); ints = cp.get_carbon_integrals(INTEGRATION_RADIUS)
                if len(ints)>0: mx=np.max(ints); fd=True; print(f"    [GLOBAL STD] Dopa Max = {mx:.2f}")
            break
    if not fd: print("    [WARN] Dopa Max not found, using 1.0")
    return mx

def align_xy(lig, obp, whole):
    c = lig.mean(0); u,s,vh = np.linalg.svd(lig-c)
    return np.dot(lig-c, vh.T)[:,:2], np.dot(obp-c, vh.T)[:,:2], np.dot(whole-c, vh.T)[:,:2]

def calc_kabsch_2d(mobile, target):
    """
    计算将 mobile 点集对齐到 target 点集的旋转矩阵 R 和平移向量 t
    使得: mobile @ R.T + t ≈ target
    """
    # 1. 计算质心
    pm = np.mean(mobile, axis=0)
    pt = np.mean(target, axis=0)
    
    # 2. 去中心化
    m = mobile - pm
    t_coords = target - pt
    
    # 3. 计算协方差矩阵
    H = np.dot(m.T, t_coords)
    
    # 4. SVD 分解
    U, S, Vt = np.linalg.svd(H)
    
    # 5. 计算旋转矩阵
    R = np.dot(Vt.T, U.T)
    
    # 6. 处理反射 (保证是旋转而不是镜像)
    if np.linalg.det(R) < 0:
        Vt[1, :] *= -1
        R = np.dot(Vt.T, U.T)
        
    # 7. 计算平移
    t = pt - np.dot(pm, R)
    
    return R, t

def plot_proj(lig, obp, whole, w, labs, out, cid, suf):
    fig, ax = plt.subplots(figsize=(10,10)); ax.set_aspect('equal')
    ax.scatter(whole[:,0], whole[:,1], c='lightgray', s=30, zorder=1)
    sc = ax.scatter(lig[:,0], lig[:,1], c=w, cmap='coolwarm', vmin=0, vmax=1.0, s=500, edgecolors='k', zorder=2)
    loop = np.vstack([lig, lig[0]]); ax.plot(loop[:,0], loop[:,1], 'k-', lw=2, zorder=2)
    for i,p in enumerate(lig): ax.text(p[0],p[1],f"{w[i]:.2f}",ha='center',va='center',color='w',fontweight='bold')
    for i,p in enumerate(obp):
        col = 'red' if labs[i] in ['389','390'] else 'blue'; mk = '^' if col=='red' else 'o'
        ax.scatter(p[0],p[1], c=col, marker=mk, s=150, alpha=0.7); ax.text(p[0]+0.2,p[1]+0.2,labs[i],color=col)
    plt.colorbar(sc, label='Norm. ELF'); plt.title(f"{cid} Projection - {suf}")
    plt.savefig(out, dpi=150, bbox_inches='tight'); plt.close()

def find_ligand(u):
    p = u.select_atoms("resname LIG LIG1 LDP R5F DRG UNK")
    if len(p)>0: return p.residues[0]
    cands = [r for r in u.residues if r.resname not in ["TIP3","SOL","WAT","SOD","CLA","POT","ZN","POPC","POPE","CHL"] and len(r.atoms)>3]
    if not cands: return None
    cands.sort(key=lambda r: len(r.atoms), reverse=True); return cands[0]

# ==============================================================================
# 5. 主处理函数
# ==============================================================================

def process_replicate(xtc, topo, cube_d, ref_d, cid, rep_name, offset_calc, output_handler):
    """处理单个副本的轨迹"""
    cp, raw_int = cube_d; ref_c, ref_e = ref_d
    try: u = mda.Universe(topo, xtc)
    except: return None, None, None

    offset = offset_calc.calculate_offset(u, 389) 
    if offset is None: return None, None, None 

    lig_res = find_ligand(u)
    if not lig_res: return None, None, None
    
    # --- 定义原有残基选择 ---
    t389 = 389 + offset; t390 = 390 + offset
    r389 = u.select_atoms(f"resid {t389}"); r390 = u.select_atoms(f"resid {t390}")
    
    if len(r389)>0: 
        c, _ = get_aromatic_ring_data(r389)
        anchor = c if c is not None else r389.center_of_mass()
    else: return None, None, None

    # --- 【新增】定义 D3.32 和 W6.48 的选择 ---
    t114 = 114 + offset
    t386 = 386 + offset
    
    # D3.32: 尝试选择侧链羧基原子，如果找不到则回退到 CA
    r114_group = u.select_atoms(f"resid {t114} and name CG OD1 OD2")
    if len(r114_group) == 0:
        r114_group = u.select_atoms(f"resid {t114} and name CA")
    
    # W6.48: 选择整个残基用于计算环质心
    r386_group = u.select_atoms(f"resid {t386}")
    
    # 配体氮原子: 用于寻找碱性氮
    lig_nitrogens = lig_res.atoms.select_atoms("name N*")

    # --- 环匹配逻辑 (保持不变) ---
    manual = MANUAL_ATOM_OVERRIDES.get("unc") if "unc" in cid.lower() else None
    matched = None; w = None
    if manual:
        matched = lig_res.atoms.select_atoms(f"name {' '.join(manual)}")
        if len(matched)==6:
            try:
                rm = RingMatcher(ref_c, ref_e); matched, c_idx = rm.match(matched, anchor)
                if matched: w = raw_int[c_idx]/GLOBAL_DOPA_MAX_INTEGRAL
            except: pass
    else:
        cands = lig_res.atoms.select_atoms("name C*")
        if len(cands)>5:
            try:
                rm = RingMatcher(ref_c, ref_e); matched, c_idx = rm.match(cands, anchor)
                if matched: w = raw_int[c_idx]/GLOBAL_DOPA_MAX_INTEGRAL
            except: pass
            
    if not matched: return None, None, None

    # 定义几何计算用的残基
    obp_ids = [x + offset for x in OBP_RESIDUES_STD]
    plane_ids = [x + offset for x in PLANE_RESIDUES_STD]
    
    obp_atoms = u.select_atoms(f"resid {' '.join(map(str, obp_ids))} and name CA")
    plane_res = u.select_atoms(f"resid {' '.join(map(str, plane_ids))} and name CA")
    if len(plane_res)==0: return None, None, None

    # 数据容器
    data = []; vis_accum = np.zeros((len(OBP_RESIDUES_STD)+2, 2)); cnt = 0
    vis_labs = [str(r) for r in OBP_RESIDUES_STD] + ["389", "390"]
    
    all_angles_389 = []; all_angles_390 = []
    all_distance_decays_389 = []; all_distance_decays_390 = []
    
    feature_vectors = [] 

    for ts in u.trajectory:
        lp = matched.positions
        lc_geo = lp.mean(0)
        safe_w = np.maximum(w, 0.0)
        
        # --- 几何特征: 平面二面角 ---
        ln = calculate_plane_normal(lp)
        pn = calculate_plane_normal(plane_res.positions)
        dot_val = np.clip(np.dot(ln, pn), -1, 1)
        ga = np.degrees(np.arccos(dot_val)); ga = ga if ga<=90 else 180-ga
        ml_cos_angle = np.abs(dot_val)
        
        # --- OBP 距离特征 ---
        dobp = distance_array(lc_geo[None,:], obp_atoms.positions, box=u.dimensions)[0]
        if len(dobp) < len(OBP_RESIDUES_STD):
            padded_dobp = np.ones(len(OBP_RESIDUES_STD)) * 20.0
            padded_dobp[:len(dobp)] = dobp
            dobp = padded_dobp

# --- 【新增】配体碱性氮相关特征计算 ---
        # 改动点1：默认值设为 0.0，对应"忽略"和"无信号"的状态
        dist_n_114 = 0.0
        dist_n_386 = 0.0
        
        if len(lig_nitrogens) > 0:
            # 1. 寻找配体上距离 D3.32 (114) 最近的氮原子 (逻辑不变，用于定位)
            if len(r114_group) > 0:
                # 计算所有 N 到 D3.32 所有原子的距离矩阵
                dmat_114 = cdist(lig_nitrogens.positions, r114_group.positions)
                # 找到每个 N 到 D3.32 的最小距离
                min_dists = np.min(dmat_114, axis=1)
                # 确定哪个 N 是最近的 ("Basic Nitrogen")
                closest_n_idx = np.argmin(min_dists)
                basic_n_pos = lig_nitrogens.positions[closest_n_idx]
                
                # 特征1: 【关键修改】虽然算出了位置，但强制将特征值设为 0.0 (Masking)
                # 这样模型就无法利用盐桥距离来"作弊"
                dist_n_114 = 0.0
            else:
                basic_n_pos = None

            # 2. 计算该氮原子到 W6.48 (386) 苯环质心的距离，并应用高斯衰减
            if basic_n_pos is not None and len(r386_group) > 0:
                c386, _ = get_aromatic_ring_data(r386_group)
                if c386 is None: # 如果提取失败，回退到几何中心
                    c386 = r386_group.center_of_mass()
                
                # 计算原始欧氏距离
                raw_dist_386 = np.linalg.norm(basic_n_pos - c386)
                
                # 特征2: 【关键修改】应用高斯衰减 exp(-(d/4)^2)
                # 将距离差异转化为显著的信号强度差异 (0~1)
                # 3.5A -> ~0.46 (强), 5.5A -> ~0.15 (弱)
                dist_n_386 = np.exp(-((raw_dist_386 / 4.0) ** 2))
                
        # --- Phe 389 电子特征 ---
        c1, n1 = get_aromatic_ring_data(r389)
        d1_geo = 999.0; d1_w = 999.0
        c1_angles = np.zeros(6); angles_389 = None; dist_decay_389 = None
        ml_phe389_stats = [0.0, 0.0, 0.0] 
        
        if c1 is not None:
            d1_geo = np.linalg.norm(lc_geo - c1)
            dists_to_c1 = np.linalg.norm(lp - c1, axis=1)
            angles_389, angle_decay_389 = calculate_carbon_angles_and_decay(lp, c1, n1)
            distances_389, dist_decay_389 = calculate_distance_decay(lp, c1, n1)
            c1_angles = angles_389
            
            combined_weights_1 = calculate_combined_weight(safe_w, angle_decay_389, dist_decay_389)
            s_sum = np.sum(combined_weights_1)
            s_max = np.max(combined_weights_1)
            s_conc = s_max / (s_sum + 1e-6)
            ml_phe389_stats = [s_sum, s_max, s_conc]
            d1_w = calculate_weighted_average_distance(dists_to_c1, combined_weights_1) if s_sum > 0 else np.mean(dists_to_c1)
            
            all_angles_389.append(angles_389)
            all_distance_decays_389.append(dist_decay_389)

        # --- Phe 390 电子特征 ---
        c2, n2 = get_aromatic_ring_data(r390)
        d2_geo = 999.0; d2_w = 999.0
        c2_angles = np.zeros(6); angles_390 = None; dist_decay_390 = None
        ml_phe390_stats = [0.0, 0.0, 0.0]

        if c2 is not None:
            d2_geo = np.linalg.norm(lc_geo - c2)
            dists_to_c2 = np.linalg.norm(lp - c2, axis=1)
            angles_390, angle_decay_390 = calculate_carbon_angles_and_decay(lp, c2, n2)
            distances_390, dist_decay_390 = calculate_distance_decay(lp, c2, n2)
            c2_angles = angles_390
            combined_weights_2 = calculate_combined_weight(safe_w, angle_decay_390, dist_decay_390)
            
            s_sum = np.sum(combined_weights_2)
            s_max = np.max(combined_weights_2)
            s_conc = s_max / (s_sum + 1e-6)
            ml_phe390_stats = [s_sum, s_max, s_conc]
            d2_w = calculate_weighted_average_distance(dists_to_c2, combined_weights_2) if s_sum > 0 else np.mean(dists_to_c2)
            
            all_angles_390.append(angles_390)
            all_distance_decays_390.append(dist_decay_390)

        # --- 构建当前帧的特征向量 ---
        # 顺序: 
        # [Dist_1..Dist_M (12个), Cos_Angle (1个), P1_Sum..Conc (3个), P2_Sum..Conc (3个), Dist_N_114 (1个), Dist_N_386 (1个)]
        # 总共 19 + 2 = 21 维
        current_frame_features = np.concatenate([
            dobp,               # 几何距离 (12维)
            [ml_cos_angle],     # 几何角度 (1维)
            ml_phe389_stats,    # Phe389 电子特征 (3维)
            ml_phe390_stats,    # Phe390 电子特征 (3维)
            [dist_n_114],       # 【新增】碱性氮到D3.32距离 (1维)
            [dist_n_386]        # 【新增】碱性氮到W6.48环距离 (1维)
        ])
        feature_vectors.append(current_frame_features)

        # Vis Accumulation
        tpos = [a.position for a in obp_atoms] + [c1 if c1 is not None else [0,0,0], c2 if c2 is not None else [0,0,0]]
        _, pxy, _ = align_xy(lp, np.array(tpos), lig_res.atoms.positions)
        vis_accum += pxy; cnt += 1
        
        # CSV Data Row
        row = {
            "Time": ts.time, "Replica": rep_name, "Global_Angle": ga, 
            "Dist_Phe389_Geo": d1_geo, "Dist_Phe389_Weighted": d1_w, 
            "Dist_Phe390_Geo": d2_geo, "Dist_Phe390_Weighted": d2_w,
            "Phe389_Score_Sum": ml_phe389_stats[0], "Phe389_Score_Max": ml_phe389_stats[1],
            "Phe390_Score_Sum": ml_phe390_stats[0], "Phe390_Score_Max": ml_phe390_stats[1],
            # 【新增】CSV记录
            "Dist_N_to_D332": dist_n_114,
            "Dist_N_to_W648_Ring": dist_n_386
        }
        for i in range(6):
            row[f"C{i+1}_Angle_to_Phe389"] = c1_angles[i]
            row[f"C{i+1}_Angle_to_Phe390"] = c2_angles[i]
        
        for i, rid in enumerate(OBP_RESIDUES_STD): row[f"Dist_Res_{rid}"] = dobp[i]
        data.append(row)
        
    # --- 保存结果 ---
    df = pd.DataFrame(data)
    
    # 1. 保存 .npy 特征矩阵 (ML用)
    if feature_vectors:
        feat_array = np.array(feature_vectors, dtype=np.float32)
        output_handler.save_features(feat_array)

    # 2. 保存投影图
    if cnt > 0:
        # 当前轨迹的平均口袋坐标 (axy) 和配体坐标 (lxy)
        # 注意：lxy 来自最后一帧的 align_xy，代表了配体的形状和自身坐标系
        # axy 是在这个自身坐标系下，口袋的平均位置
        axy = vis_accum / cnt
        lxy, _, wxy = align_xy(lp, np.array([[0,0,0]]), lig_res.atoms.positions)
        
        # --- [新增] 全局对齐逻辑 START ---
        ref_file = os.path.join(OUTPUT_BASE_DIR, "REFERENCE_DOPA_OBP.npy")
        is_dopa = "dopa" in cid.lower()
        
        target_obp = None
        
        # 策略：如果是 Dopa，存为基准；如果不是，尝试读取基准
        if is_dopa:
            # 只有当这是第一个 Dopa 副本时才保存，避免覆盖
            if not os.path.exists(ref_file) or rep_name == "replicate_1":
                np.save(ref_file, axy)
                print(f"     [Ref] Saved current pocket as Global Reference to {ref_file}")
            target_obp = axy # 自身就是基准
        elif os.path.exists(ref_file):
            target_obp = np.load(ref_file)
        else:
            print("     [Ref] Warning: No Dopa reference found yet. Projection will be unaligned.")

        # 如果有基准，进行对齐
        if target_obp is not None:
            try:
                # 计算将 当前口袋(axy) 对齐到 基准口袋(target_obp) 的变换
                # 注意：axy 和 target_obp 必须点数相同且顺序一致 (OBP_RESIDUES_STD 保证了这一点)
                R, t = calc_kabsch_2d(axy, target_obp)
                
                # 应用变换到口袋 (仅为了验证，理论上变换后 axy ≈ target_obp)
                axy = np.dot(axy, R.T) + t
                
                # 【关键】应用同样的变换到配体
                # 这会将配体移动到“标准口袋坐标系”中正确的位置
                lxy = np.dot(lxy, R.T) + t
                
                # 应用变换到全原子背景 (如果有的话)
                if wxy is not None:
                     wxy = np.dot(wxy, R.T) + t
                     
            except Exception as e:
                print(f"     [Align Error] {e}")
        # --- [新增] 全局对齐逻辑 END ---

        proj_path = output_handler.save_projection(None)
        plot_proj(lxy, axy, wxy, w, vis_labs, str(proj_path), cid, rep_name)

    # 3. 计算总体统计 (Stats CSV)
    strength = None
    if all_angles_389 and all_angles_390:
        avg_angles_389 = np.mean(all_angles_389, axis=0)
        avg_angles_390 = np.mean(all_angles_390, axis=0)
        avg_dist_decay_389 = np.mean(all_distance_decays_389, axis=0)
        avg_dist_decay_390 = np.mean(all_distance_decays_390, axis=0)
        
        strength = calculate_interaction_strength(
            safe_w, avg_angles_389, avg_angles_390,
            avg_dist_decay_389, avg_dist_decay_390
        )
        # print(format_interaction_strength(strength)) # Optional print

    stats = {
        "Compound": cid, "Replica": rep_name, "Offset": offset,
        "Global_Angle_Mean": df["Global_Angle"].mean(), 
        "Phe389_Score_Sum_Mean": df["Phe389_Score_Sum"].mean(), 
        "Phe390_Score_Sum_Mean": df["Phe390_Score_Sum"].mean(),
        # 【新增】统计平均值
        "Dist_N_to_D332_Mean": df["Dist_N_to_D332"].mean(),
        "Dist_N_to_W648_Ring_Mean": df["Dist_N_to_W648_Ring"].mean()
    }
    
    if strength: stats.update(strength)
    
    return df, stats, strength

# ==============================================================================
# 6. 主函数
# ==============================================================================

def main():
    global GLOBAL_DOPA_MAX_INTEGRAL
    root = "."
    gmx = get_dopa_max_integral(root)
    GLOBAL_DOPA_MAX_INTEGRAL = gmx
    aligner = OffsetCalculator(STANDARD_SEQUENCE)
    
    print("\n>>> Processing (V2.0 Modularized + Dopa Priority)...")
    
    # 1. 获取所有待处理目录
    all_dirs = glob.glob(os.path.join(root, "*"))
    
    # 2. 【关键修改】自定义排序：让包含 "dopa" (忽略大小写) 的目录排在最前面
    # lambda x: (False, x) 会排在 (True, x) 前面，因为 False < True
    # 我们希望 "dopa" 排前面，所以当 "dopa" 不在 x 中时返回 True (排后面)
    all_dirs.sort(key=lambda x: (not "dopa" in os.path.basename(x).lower(), x))

    for c_dir in all_dirs:
        if not os.path.isdir(c_dir): continue
        # 排除非数据目录
        if any(x in c_dir for x in ["run_analysis", "modules", "results", "__pycache__"]): continue
        
        cid = os.path.basename(c_dir)
        
        cubs = glob.glob(os.path.join(c_dir, "*.cub")); 
        if not cubs: continue
        
        pdb = glob.glob(os.path.join(c_dir, "*.pdb"))
        pdb = next((p for p in pdb if "step7" not in p and "topol" not in p), None)
        if not pdb: print(f"[Skip] {cid} no ref PDB"); continue
        
        ref_d = get_ref_data_from_pdb(pdb); ref_e = ref_d[1]
        cp = CubeParser(cubs[0]); ri = cp.get_carbon_integrals(INTEGRATION_RADIUS)
        
        # 简单的长度检查，防止 cube 和 pdb 不匹配
        if ref_e.count('C') != len(ri): 
             # 尝试宽容处理或者跳过
             pass 
        
        cube_d = (cp, ri)
        
        xtcs = glob.glob(os.path.join(c_dir, "**", "merged.xtc"), recursive=True)
        if not xtcs: print(f"[Skip] {cid} no merged.xtc"); continue
        
        print(f"Analyzing: {cid} ({len(xtcs)} reps)")
        ts_list = []; stat_list = []; strength_list = []
        
        for xtc in xtcs:
            rd = os.path.dirname(xtc); rn = os.path.basename(rd)
            tps = [os.path.join(rd, f) for f in os.listdir(rd) if f.endswith(".tpr")]
            topo = next((t for t in tps if "production" in t), tps[0] if tps else None)
            
            if topo:
                output_handler = OutputHandler(cid, rn, OUTPUT_BASE_DIR)
                # 调用处理函数（包含您刚才修改的对齐绘图逻辑）
                ts, st, strength = process_replicate(xtc, topo, cube_d, ref_d, cid, rn, aligner, output_handler)
                
                if ts is not None:
                    output_handler.save_timeseries(ts)
                    output_handler.save_stats(pd.DataFrame([st]))
                    ts_list.append(ts); stat_list.append(st)
                    if strength:
                        strength_list.append(strength)
        
        if ts_list:
            OutputHandler.aggregate_timeseries(ts_list, OUTPUT_BASE_DIR, cid)
            OutputHandler.aggregate_stats(stat_list, OUTPUT_BASE_DIR, cid)
            print(f"  -> Done {cid}")

if __name__ == "__main__":
    main()
