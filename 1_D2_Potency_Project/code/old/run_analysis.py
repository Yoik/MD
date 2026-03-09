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

# [NEW] 引入 Biopython 进行序列比对
try:
    from Bio import Align
    from Bio.SeqUtils import seq1
except ImportError:
    print("Error: Please install biopython! Run: pip install biopython")
    exit()

# ==============================================================================
# 1. 用户配置区 (请务必修改这里)
# ==============================================================================

# [CRITICAL] 标准参考序列 (对应你认为 389 是 6.51 的那个序列)
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

# 关键残基 (标准序列中的编号)
PHE_RESIDUES_STD = [389, 390]
OBP_RESIDUES_STD = [114, 115, 118, 119, 190, 193, 194, 197, 386, 393, 412, 416]
PLANE_RESIDUES_STD = [198, 163, 76, 127]

# 锚点残基 (用于计算 Offset 的基准点，选一个保守的，比如 389)
ANCHOR_RESID_STD = 389

# 其他配置
T_STACK_DIST_CUTOFF = 6.5 
T_STACK_ANGLE_CENTER = 90.0
T_STACK_ANGLE_TOL = 30.0 
INTEGRATION_RADIUS = 1.5 
GLOBAL_DOPA_MAX_INTEGRAL = 1.0 
BOHR_TO_ANGSTROM = 0.52917721067

# UNC 手动原子
MANUAL_ATOM_OVERRIDES = {
    "unc": ['C16', 'C13', 'C8', 'C7', 'C12', 'C15']
}

# ==============================================================================
# 2. 序列对齐模块 (Offset Auto-Calculator - Fixed)
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
            code = self.three_to_one.get(res, 'X') # Fallback to X
            seq_str += code
            valid_indices.append(resids[i])
            
        return seq_str, valid_indices

    def calculate_offset(self, u, target_std_id):
        """
        [FIX] 使用 aligned 属性代替 coordinates，更加稳健
        """
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
        
        # aligned[0] 是标准序列的片段区间 [(start, end), ...]
        # aligned[1] 是仿真序列的片段区间 [(start, end), ...]
        # 两者一一对应
        aligned_ref = best_aln.aligned[0]
        aligned_sim = best_aln.aligned[1]
        
        for (r_start, r_end), (s_start, s_end) in zip(aligned_ref, aligned_sim):
            # 区间长度是一样的
            length = r_end - r_start
            
            for i in range(length):
                # 0-based index in Standard String
                r_idx = r_start + i
                # 0-based index in Sim String
                s_idx = s_start + i
                
                # 转换回 Standard PDB Numbering (1-based)
                std_res_num = r_idx + 1
                
                # 找到 Sim 序列对应的真实 Resid
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
# 3. 核心功能类 (Cube, Matcher)
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

def calculate_plane_normal(c): return np.linalg.svd(c-c.mean(0))[2][2,:]

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

def get_aromatic_ring_data(rg):
    if len(rg)==0: return None, None
    rn = rg.resnames[0]; ra = None
    if rn in ['PHE','TYR']: ra = rg.atoms.select_atoms("name CG CD1 CD2 CE1 CE2 CZ")
    elif rn == 'TRP': ra = rg.atoms.select_atoms("name CD2 CE2 CE3 CZ2 CZ3 CH2")
    elif rn in ['HIS','HSD','HSE','HSP']: ra = rg.atoms.select_atoms("name CG ND1 CD2 CE1 NE2")
    exp = 5 if rn.startswith('HIS') else 6
    if ra and len(ra)==exp: return ra.center_of_mass(), calculate_plane_normal(ra.positions)
    side = rg.atoms.select_atoms("not name N CA C O")
    return (side.center_of_mass() if len(side)>0 else rg.center_of_mass()), None

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

def align_xy(lig, obp, whole):
    c = lig.mean(0); u,s,vh = np.linalg.svd(lig-c)
    return np.dot(lig-c, vh.T)[:,:2], np.dot(obp-c, vh.T)[:,:2], np.dot(whole-c, vh.T)[:,:2]

def find_ligand(u):
    p = u.select_atoms("resname LIG LIG1 LDP R5F DRG UNK")
    if len(p)>0: return p.residues[0]
    cands = [r for r in u.residues if r.resname not in ["TIP3","SOL","WAT","SOD","CLA","POT","ZN","POPC","POPE","CHL"] and len(r.atoms)>3]
    if not cands: return None
    cands.sort(key=lambda r: len(r.atoms), reverse=True); return cands[0]

# ==============================================================================
# 5. Process
# ==============================================================================
def process_replicate(xtc, topo, cube_d, ref_d, cid, rep_name, offset_calc):
    cp, raw_int = cube_d; ref_c, ref_e = ref_d
    try: u = mda.Universe(topo, xtc)
    except: return None, None

    # [NEW] Calculate Offset
    offset = offset_calc.calculate_offset(u, ANCHOR_RESID_STD)
    if offset is None: return None, None 

    lig_res = find_ligand(u)
    if not lig_res: return None, None
    
    # Anchor using dynamic offset
    t389 = 389 + offset; t390 = 390 + offset
    r389 = u.select_atoms(f"resid {t389}"); r390 = u.select_atoms(f"resid {t390}")
    
    # Get anchor COM
    if len(r389)>0: 
        c, _ = get_aromatic_ring_data(r389)
        anchor = c if c is not None else r389.center_of_mass()
    else: return None, None

    # Match Ring
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
            
    if not matched: return None, None

    # Define Residues
    obp_ids = [x + offset for x in OBP_RESIDUES_STD]
    plane_ids = [x + offset for x in PLANE_RESIDUES_STD]
    
    obp_atoms = u.select_atoms(f"resid {' '.join(map(str, obp_ids))} and name CA")
    plane_res = u.select_atoms(f"resid {' '.join(map(str, plane_ids))} and name CA")
    if len(plane_res)==0: return None, None

    # Loop
    data = []; vis_accum = np.zeros((len(OBP_RESIDUES_STD)+2, 2)); cnt = 0
    vis_labs = [str(r) for r in OBP_RESIDUES_STD] + ["389", "390"]

# [修改核心] 循环计算逻辑
    for ts in u.trajectory:
        lp = matched.positions
        
        # 1. 基础几何中心 (用于计算 OBP 距离和 Global Angle)
        lc_geo = lp.mean(0)
        
        # 2. 准备权重 (防止负数)
        safe_w = np.maximum(w, 0.0)
        sum_w = np.sum(safe_w)
        
        # 3. [NEW] 计算配体苯环的平面法向量
        ln = calculate_plane_normal(lp)
        pn = calculate_plane_normal(plane_res.positions)
        ga = np.degrees(np.arccos(np.clip(np.dot(ln, pn), -1, 1))); ga = ga if ga<=90 else 180-ga
        
        # 4. [NEW] 为每个碳原子计算多维衰减因子
        #    注意：角度衰减因子会在389/390分别计算，因为需要Phe的平面法向量
        #    这里先计算距离衰减因子作为占位符
        angle_decay_factors = np.ones(6)  # 稍后在389/390处计算
        distance_decay_factors = np.ones(6)
        
        # --- Phe 389 计算 ---
        c1, n1 = get_aromatic_ring_data(r389)
        d1_geo = 999.0   # 纯几何距离
        d1_w = 999.0     # 电子加权平均距离
        a1 = 180.0
        c1_angles = np.zeros(6)  # 每个碳与389平面的角度
        
        if c1 is not None:
            # A. 几何距离 (质心对质心)
            d1_geo = np.linalg.norm(lc_geo - c1)
            
            # B. [修正] 电子加权平均距离 + 角度衰减 + 距离衰减
            # 计算配体6个碳原子分别到 389 中心的距离 -> 得到一个 (6,) 的数组
            dists_to_c1 = np.linalg.norm(lp - c1, axis=1) 
            
            # [NEW] 计算每个碳与Phe389平面的角度和衰减因子
            if n1 is not None:
                # 向量：从Phe质心到每个碳原子
                vec_to_atoms = lp - c1
                
                # 角度：该向量与Phe平面法向量的夹角
                # 范围 [0°, 90°]，其中0°表示平行于平面，90°表示垂直于平面（T型）
                dot_products = np.dot(vec_to_atoms, n1)  # 标量积的向量
                norms = np.linalg.norm(vec_to_atoms, axis=1)  # 每个向量的长度
                cos_angles = dot_products / (norms + 1e-10)  # 避免除零
                angles_rad = np.arccos(np.clip(np.abs(cos_angles), -1, 1))
                c1_angles = np.degrees(angles_rad)  # 转换为度数
                
                # 角度衰减：T型接触对应90°（垂直于平面）
                # 偏离90°越远，衰减越厉害
                angle_deviation_1 = np.abs(c1_angles - 90.0)
                angle_decay_factors_1 = np.exp(-((angle_deviation_1 / 30.0) ** 2))
                
                # 距离衰减：垂直距离
                perp_dists_1 = np.abs(np.dot(vec_to_atoms, n1))
                distance_decay_factors_1 = np.exp(-((perp_dists_1 / 2.0) ** 2))
            else:
                # 如果无法获取平面法向量，使用均匀权重
                angle_decay_factors_1 = np.ones(6)
                distance_decay_factors_1 = 1.0 / (1.0 + (dists_to_c1 / 3.0))
            
            # 组合权重：电子云权重 × 角度衰减因子 × 距离衰减因子
            combined_weights_1 = safe_w * angle_decay_factors_1 * distance_decay_factors_1
            sum_combined_w1 = np.sum(combined_weights_1)
            
            if sum_combined_w1 > 0:
                # 加权平均：综合考虑电子云强度、角度偏离和距离
                d1_w = np.average(dists_to_c1, weights=combined_weights_1)
            else:
                d1_w = np.mean(dists_to_c1) # 如果没有权重数据，退化为算术平均
            
            # 角度计算（配体平面与Phe389平面的夹角）
            if n1 is not None: 
                ang = np.degrees(np.arccos(np.clip(np.dot(ln, n1), -1, 1)))
                a1 = min(abs(90-ang), abs(90-(180-ang)))
            
            contrib = combined_weights_1 * dists_to_c1
            max_idx = np.argmax(contrib)
            print(f"Phe389 最大贡献原子: C{max_idx+1}, ELF权重={safe_w[max_idx]:.3f}, 与平面角度={c1_angles[max_idx]:.1f}°, 角衰减={angle_decay_factors_1[max_idx]:.3f}, 距离衰减={distance_decay_factors_1[max_idx]:.3f}, 距离={dists_to_c1[max_idx]:.3f}")

        # --- Phe 390 计算 (同理) ---
        c2, n2 = get_aromatic_ring_data(r390)
        d2_geo = 999.0; d2_w = 999.0; a2 = 180.0
        c2_angles = np.zeros(6)  # 每个碳与390平面的角度
        
        if c2 is not None:
            d2_geo = np.linalg.norm(lc_geo - c2)
            
            # [修正] 电子加权平均距离 + 角度衰减 + 距离衰减
            dists_to_c2 = np.linalg.norm(lp - c2, axis=1)
            
            # [NEW] 计算每个碳与Phe390平面的角度和衰减因子
            if n2 is not None:
                # 向量：从Phe质心到每个碳原子
                vec_to_atoms = lp - c2
                
                # 角度：该向量与Phe平面法向量的夹角
                dot_products = np.dot(vec_to_atoms, n2)
                norms = np.linalg.norm(vec_to_atoms, axis=1)
                cos_angles = dot_products / (norms + 1e-10)
                angles_rad = np.arccos(np.clip(np.abs(cos_angles), -1, 1))
                c2_angles = np.degrees(angles_rad)  # 转换为度数
                
                # 角度衰减：T型接触对应90°（垂直于平面）
                angle_deviation_2 = np.abs(c2_angles - 90.0)
                angle_decay_factors_2 = np.exp(-((angle_deviation_2 / 30.0) ** 2))
                
                # 距离衰减：垂直距离
                perp_dists_2 = np.abs(np.dot(vec_to_atoms, n2))
                distance_decay_factors_2 = np.exp(-((perp_dists_2 / 2.0) ** 2))
            else:
                angle_decay_factors_2 = np.ones(6)
                distance_decay_factors_2 = 1.0 / (1.0 + (dists_to_c2 / 3.0))
            
            # 组合权重：电子云权重 × 角度衰减因子 × 距离衰减因子
            combined_weights_2 = safe_w * angle_decay_factors_2 * distance_decay_factors_2
            sum_combined_w2 = np.sum(combined_weights_2)
            
            if sum_combined_w2 > 0:
                d2_w = np.average(dists_to_c2, weights=combined_weights_2)
            else:
                d2_w = np.mean(dists_to_c2)

            if n2 is not None: 
                ang = np.degrees(np.arccos(np.clip(np.dot(ln, n2), -1, 1)))
                a2 = min(abs(90-ang), abs(90-(180-ang)))
            
            contrib = combined_weights_2 * dists_to_c2
            max_idx = np.argmax(contrib)
            print(f"Phe390 最大贡献原子: C{max_idx+1}, ELF权重={safe_w[max_idx]:.3f}, 与平面角度={c2_angles[max_idx]:.1f}°, 角衰减={angle_decay_factors_2[max_idx]:.3f}, 距离衰减={distance_decay_factors_2[max_idx]:.3f}, 距离={dists_to_c2[max_idx]:.3f}")

        # OBP 距离 (保持使用几何中心 lc_geo)
        dobp = distance_array(lc_geo[None,:], obp_atoms.positions, box=u.dimensions)[0]
        
        # Vis (保持不变)
        tpos = [a.position for a in obp_atoms] + [c1 if c1 is not None else [0,0,0], c2 if c2 is not None else [0,0,0]]
        _, pxy, _ = align_xy(lp, np.array(tpos), lig_res.atoms.positions)
        vis_accum += pxy; cnt += 1
        
        # [修改] 保存新的列名，区分 Geo 和 Weighted，添加每个碳与Phe平面的角度
        row = {
            "Time": ts.time, "Replica": rep_name, "Global_Angle": ga, 
            "Dist_Phe389_Geo": d1_geo, "Dist_Phe389_Weighted": d1_w, "AngDev_Phe389": a1, 
            "Dist_Phe390_Geo": d2_geo, "Dist_Phe390_Weighted": d2_w, "AngDev_Phe390": a2
        }
        # 添加每个碳与389平面的角度
        for i in range(6):
            row[f"C{i+1}_Angle_to_Phe389"] = c1_angles[i] if 'c1_angles' in locals() else 0.0
        # 添加每个碳与390平面的角度
        for i in range(6):
            row[f"C{i+1}_Angle_to_Phe390"] = c2_angles[i] if 'c2_angles' in locals() else 0.0
        
        for i, rid in enumerate(OBP_RESIDUES_STD): row[f"Dist_Res_{rid}"] = dobp[i] if i < len(dobp) else 999.
        data.append(row)
        
    df = pd.DataFrame(data)
    if cnt > 0:
        axy = vis_accum/cnt
        lxy, _, wxy = align_xy(lp, np.array([[0,0,0]]), lig_res.atoms.positions)
        plot_proj(lxy, axy, wxy, w, vis_labs, f"{cid}_{rep_name}_projection.png", cid, rep_name)

    stats = {
            "Compound": cid, "Replica": rep_name, "Offset": offset,
            "Global_Angle_Mean": df["Global_Angle"].mean(), "Global_Angle_SD": df["Global_Angle"].std(),
            
            # 389
            "Dist_Phe389_Geo_Mean": df["Dist_Phe389_Geo"].mean(), 
            "Dist_Phe389_Weighted_Mean": df["Dist_Phe389_Weighted"].mean(), # <--- 新指标
            "AngDev_Phe389_Mean": df["AngDev_Phe389"].mean(),
            
            # 390
            "Dist_Phe390_Geo_Mean": df["Dist_Phe390_Geo"].mean(), 
            "Dist_Phe390_Weighted_Mean": df["Dist_Phe390_Weighted"].mean(), # <--- 新指标
            "AngDev_Phe390_Mean": df["AngDev_Phe390"].mean()
        }
    # 添加每个碳与389/390平面的平均角度
    for i in range(6):
        stats[f"C{i+1}_Avg_Angle_to_Phe389"] = df[f"C{i+1}_Angle_to_Phe389"].mean() if f"C{i+1}_Angle_to_Phe389" in df.columns else 0.0
        stats[f"C{i+1}_Avg_Angle_to_Phe390"] = df[f"C{i+1}_Angle_to_Phe390"].mean() if f"C{i+1}_Angle_to_Phe390" in df.columns else 0.0
    
    for rid in OBP_RESIDUES_STD: stats[f"Dist_Res_{rid}"] = df[f"Dist_Res_{rid}"].mean()
    for i, val in enumerate(w): stats[f"C{i+1}_Weight"] = val
    return df, stats

def main():
    global GLOBAL_DOPA_MAX_INTEGRAL
    root = "."
    gmx = get_dopa_max_integral(root)
    GLOBAL_DOPA_MAX_INTEGRAL = gmx
    aligner = OffsetCalculator(STANDARD_SEQUENCE)
    
    print("\n>>> Processing (V17.2 Auto-Align Fixed)...")
    for c_dir in glob.glob(os.path.join(root, "*")):
        if not os.path.isdir(c_dir): continue
        if "run_analysis" in c_dir: continue
        cid = os.path.basename(c_dir)
        
        cubs = glob.glob(os.path.join(c_dir, "*.cub")); 
        if not cubs: continue
        
        pdb = glob.glob(os.path.join(c_dir, "*.pdb"))
        pdb = next((p for p in pdb if "step7" not in p and "topol" not in p), None)
        if not pdb: print(f"[Skip] {cid} no ref PDB"); continue
        
        ref_d = get_ref_data_from_pdb(pdb); ref_e = ref_d[1]
        cp = CubeParser(cubs[0]); ri = cp.get_carbon_integrals(INTEGRATION_RADIUS)
        if ref_e.count('C') != len(ri): print(f"[Error] {cid} size mismatch"); continue
        cube_d = (cp, ri)
        
        xtcs = glob.glob(os.path.join(c_dir, "**", "merged.xtc"), recursive=True)
        if not xtcs: print(f"[Skip] {cid} no merged.xtc"); continue
        
        print(f"Analyzing: {cid} ({len(xtcs)} reps)")
        ts_list = []; stat_list = []
        
        for xtc in xtcs:
            rd = os.path.dirname(xtc); rn = os.path.basename(rd)
            tps = [os.path.join(rd, f) for f in os.listdir(rd) if f.endswith(".tpr")]
            topo = next((t for t in tps if "production" in t), tps[0] if tps else None)
            
            if topo:
                ts, st = process_replicate(xtc, topo, cube_d, ref_d, cid, rn, aligner)
                if ts is not None:
                    ts.to_csv(f"{cid}_{rn}_TimeSeries.csv", index=False)
                    pd.DataFrame([st]).to_csv(f"{cid}_{rn}_Stats.csv", index=False)
                    ts_list.append(ts); stat_list.append(st)
        
        if ts_list:
            full = pd.concat(ts_list); full["Time"] = full["Time"].round(2)
            agg_m = full.groupby("Time").mean(numeric_only=True).reset_index()
            agg_s = full.groupby("Time").std(numeric_only=True).reset_index()
            res = agg_m.copy()
            for c in agg_m.columns: 
                if c!="Time": res[f"{c}_SD"] = agg_s[c]
            res.to_csv(f"{cid}_All_TimeSeries.csv", index=False)
            
            sum_df = pd.DataFrame(stat_list)
            avg = sum_df.mean(numeric_only=True); avg["Replica"]="AVERAGE"; avg["Compound"]=cid
            pd.concat([sum_df, pd.DataFrame([avg])], ignore_index=True).to_csv(f"{cid}_All_Stats.csv", index=False)
            print(f"  -> Done {cid}")

if __name__ == "__main__":
    main()