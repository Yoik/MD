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
import gc
import MDAnalysis as mda
from MDAnalysis.lib.distances import distance_array
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
import os
from scipy.spatial.distance import cdist

# 导入模块化组件
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
from modules.cube_parser import CubeParser
from modules.ring_matcher import RingMatcher
from modules.sequence_aligner import OffsetCalculator
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

MANUAL_ATOM_OVERRIDES = {}

OUTPUT_BASE_DIR = "./data/features"

# ==============================================================================
# 2. 序列对齐模块（已移到 sequence_aligner.py）
# ==============================================================================

# ==============================================================================
# 3. Cube和Ring匹配类（已移到 cube_parser.py 和 ring_matcher.py）
# ==============================================================================

# ==============================================================================
# 4. 辅助函数
# ==============================================================================

def get_ref_data_from_pdb(pdb_file):
    c=[]; e=[]
    try:
        with open(pdb_file) as f:
            for l in f:
                if l.startswith("ATOM") or l.startswith("HETATM"):
                    atom_name = l[12:16].strip()
                    # 提取所有重原子（C, N, O, S）- 用于环检测和ELF积分
                    # 排除主链原子 (CA, C, N, O)
                    if atom_name[0] in ['C', 'N', 'O', 'S']:
                        if atom_name not in ['CA', 'C', 'N', 'O']:
                            c.append([float(l[30:38]),float(l[38:46]),float(l[46:54])])
                            e.append(atom_name[0])
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

def plot_proj(lig, obp, whole, w, labs, out, cid, suf, ring_type="BENZENE", lig_atom_names=None):
    fig, ax = plt.subplots(figsize=(10,10)); ax.set_aspect('equal')
    ax.scatter(whole[:,0], whole[:,1], c='lightgray', s=30, zorder=1)
    sc = ax.scatter(lig[:,0], lig[:,1], c=w, cmap='coolwarm', vmin=0, vmax=1.0, s=500, edgecolors='k', zorder=2)
    loop = np.vstack([lig, lig[0]]); ax.plot(loop[:,0], loop[:,1], 'k-', lw=2, zorder=2)
    
    # 显示每个环原子的权重和原子名称
    for i,p in enumerate(lig):
        atom_label = f"{lig_atom_names[i]}" if lig_atom_names and i < len(lig_atom_names) else ""
        weight_text = f"{w[i]:.2f}"
        display_text = f"{atom_label}\n{weight_text}" if atom_label else weight_text
        ax.text(p[0],p[1], display_text, ha='center', va='center', color='w', fontweight='bold', fontsize=9)
    
    for i,p in enumerate(obp):
        col = 'red' if labs[i] in ['389','390'] else 'blue'; mk = '^' if col=='red' else 'o'
        ax.scatter(p[0],p[1], c=col, marker=mk, s=150, alpha=0.7); ax.text(p[0]+0.2,p[1]+0.2,labs[i],color=col)
    
    plt.colorbar(sc, label='Norm. ELF')
    title = f"{cid} Projection ({ring_type}) - {suf}"
    plt.title(title)
    plt.savefig(out, dpi=150, bbox_inches='tight')
    # 关键修改：显式关闭特定的 figure 对象，并清理内存
    plt.close(fig) 
    plt.clf()
    plt.cla()

def save_3d_structure_pdb(lig_coords, lig_names, obp_coords, obp_names, obp_indices, out_file, cid, ring_atom_indices=None, ring_weights=None):
    """
    保存 3D 结构为 PDB 文件，包括配体原子和 OBP 残基原子
    可用于 PyMOL、VMD 等可视化软件进行交互式查看
    
    参数：
    - lig_coords: 配体原子坐标 (N, 3)
    - lig_names: 配体原子名称列表
    - obp_coords: OBP 残基原子坐标 (M, 3)
    - obp_names: OBP 原子名称/标签列表
    - obp_indices: OBP 残基索引列表
    - out_file: 输出 PDB 文件路径
    - cid: 化合物 ID
    - ring_atom_indices: 环原子索引列表（用于 B-factor 标记）
    - ring_weights: 环原子权重列表（用于 occupancy 字段）
    """
    try:
        with open(out_file, 'w') as f:
            f.write("REMARK Generated 3D structure with ring atoms and OBP residues\n")
            f.write(f"REMARK Compound: {cid}\n")
            
            atom_num = 1
            
            # 写入配体原子
            for i, (coord, name) in enumerate(zip(lig_coords, lig_names)):
                # 检查该原子是否在环中
                in_ring = i in ring_atom_indices if ring_atom_indices is not None else False
                b_factor = 50.0 if in_ring else 0.0  # B-factor 标记环原子
                occupancy = ring_weights[ring_atom_indices.index(i)] if (in_ring and ring_weights is not None) else 1.0
                
                x, y, z = coord
                f.write(f"ATOM  {atom_num:5d}  {name:4s} LIG A   1    {x:8.3f}{y:8.3f}{z:8.3f}{occupancy:6.2f}{b_factor:6.2f}           C\n")
                atom_num += 1
            
            # 写入 OBP 残基原子（用不同的链标记）
            for i, (coord, name, res_idx) in enumerate(zip(obp_coords, obp_names, obp_indices)):
                x, y, z = coord
                # OBP 原子的 B-factor = 0，occupancy = 1.0
                f.write(f"ATOM  {atom_num:5d}  CA  RES B{i+1:3d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C\n")
                atom_num += 1
            
            f.write("END\n")
        
        print(f"     [3D] Saved 3D structure: {out_file}")
        return True
    except Exception as e:
        print(f"     [3D] Error saving 3D structure: {e}")
        return False

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

# --- 环匹配逻辑 + 环类型打印 + 环原子映射 ---
    matched = None; w = None; ring_type_info = "unknown"
    
    # 1. 创建 RingMatcher 实例
    try:
        rm = RingMatcher(ref_c, ref_e)
        ring_type_info = rm.ring_type
    except:
        ring_type_info = "unknown"
    
    # 2. 【核心修改】使用 rm.match() 替代原本的手动距离匹配
    # 原来的代码在这里尝试手动平移和 cdist 查找，导致了错误。
    # 现在直接调用我们写好的 match 方法：
    if ring_type_info != "unknown":
        # match 方法返回：(MD原子对象, Cube积分索引列表, MD原子索引列表)
        # 它内部会自动处理：拓扑查找、Kabsch旋转对齐、RMSD筛选
        matched_atoms, cube_idxs, _ = rm.match(lig_res.atoms, anchor)
        
        if matched_atoms is not None:
            matched = matched_atoms
            
            # 根据返回的 cube_idxs 获取对应的电子密度权重
            # raw_int 是所有重原子的积分数组
            if cube_idxs:
                try:
                    w = raw_int[cube_idxs] / GLOBAL_DOPA_MAX_INTEGRAL
                except IndexError:
                    # 极其罕见的情况：索引越界，回退到默认
                    w = np.ones(len(matched)) * 0.5
            else:
                w = np.ones(len(matched)) * 0.5

            # --- 【新增】确定几何中心原子索引列表 ---
            geo_center_indices = []
    
            # 1. 优先判断是否为单苯环 (Benzene)
            if hasattr(rm, 'ring_type') and rm.ring_type == 'benzene':
                # 如果是苯环，直接使用所有原子
                geo_center_indices = list(range(len(matched)))
                
            # 2. 如果是稠环 (Indole/Thiophene/Furan)，需要筛选
            elif hasattr(rm, 'rings') and len(rm.rings) > 0:
                try:
                    # 从参考环信息中获取“苯环部分”的原始原子索引
                    # rm.rings[0] 是当前选中的参考环信息
                    ref_six_set = set(rm.rings[0].get('six_ring', []))
                    
                    if ref_six_set:
                        # 动态筛选：遍历 MD 原子对应的 Ref 索引
                        for i, ref_idx in enumerate(rm.ref_ring_idx):
                            if ref_idx in ref_six_set:
                                geo_center_indices.append(i)
                    else:
                        # 理论上稠环必须有 six_ring 信息，如果没有就是异常
                        print(f"     [Error] {cid}: Fused ring detected but 'six_ring' data is missing!")
                        
                except Exception as e:
                    print(f"     [Error] {cid}: Geo center logic failed with exception: {e}")

            # 3. 【核心修改】严格检查：如果筛选失败，报错并退出，绝不凑合！
            if not geo_center_indices:
                print(f"     [Result Error] {cid}: Could not identify Benzene atoms! (RingType: {getattr(rm, 'ring_type', 'Unknown')})")
                print(f"     -> Matched atoms: {len(matched)}")
                print(f"     -> Ref Ring Idx: {rm.ref_ring_idx}")
                if hasattr(rm, 'rings') and len(rm.rings) > 0:
                    print(f"     -> Six Ring Info: {rm.rings[0].get('six_ring', 'None')}")
                return None, None, None
            else:
                geo_center_indices = list(range(len(matched)))
        else:
            # 匹配失败 (RMSD过大或找不到对应的环)
            matched = None
            w = None

    # 打印识别到的环类型
    if rep_name == "gromacs_replicate_1":  # 只在第一个副本打印一次
        if ring_type_info == "indole":
            ring_desc = "INDOLE (6+5 fused aromatic, N-heterocycle)"
        elif ring_type_info == "furan":
            ring_desc = "FURAN (6+5 fused aromatic, O-heterocycle)"
        elif ring_type_info == "thiophene":
            ring_desc = "THIOPHENE (6+5 fused aromatic, S-heterocycle)"
        else:
            ring_desc = "BENZENE (6-membered)"
        
        print(f"     [Ring] {cid}: {ring_desc}")
        
        # 打印所有环原子及其权重信息
        if matched is not None and w is not None:
            atom_info = []
            for i, atom in enumerate(matched):
                weight_val = w[i] if i < len(w) else 0.0
                atom_info.append(f"{atom.name}({weight_val:.3f})")
            print(f"            Ring atoms (all {len(matched)}): {', '.join(atom_info)}")
            
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
    
    # 如果环匹配失败，使用默认权重（全1）
    if w is None:
        w = np.ones(6)

    for ts in u.trajectory:
        lp = matched.positions
        
        # 【修改】只使用苯环部分的原子计算几何中心
        # 这样 OBP 距离 (dobp) 和 Phe 距离 (d1_geo) 都会基于苯环中心
        if 'geo_center_indices' in locals() and geo_center_indices:
            lc_geo = lp[geo_center_indices].mean(0)
        else:
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
        c1_angles = None; angles_389 = None; dist_decay_389 = None
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
        c2_angles = None; angles_390 = None; dist_decay_390 = None
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
        
        # 记录可变数量的角度（根据实际环原子数量）
        if c1_angles is not None:
            for i, angle in enumerate(c1_angles):
                row[f"C{i+1}_Angle_to_Phe389"] = angle
        if c2_angles is not None:
            for i, angle in enumerate(c2_angles):
                row[f"C{i+1}_Angle_to_Phe390"] = angle
        
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
        
        # 准备环类型和原子名称用于绘图
        ring_type_display = ring_type_info.upper() if ring_type_info else "BENZENE"
        lig_atom_names = [atom.name for atom in matched] if matched else None
        
        plot_proj(lxy, axy, wxy, w, vis_labs, str(proj_path), cid, rep_name, 
                  ring_type=ring_type_display, lig_atom_names=lig_atom_names)
        
        # 保存 3D 结构可视化 (使用平均坐标)
        if all_angles_389 and obp_atoms is not None:  # 确保有数据
            try:
                # 使用配体的 3D 坐标 (从最后一帧)
                lig_coords_3d = lig_res.atoms.positions
                lig_atom_names_full = [atom.name for atom in lig_res.atoms]
                
                # OBP 原子的 3D 坐标
                obp_coords_3d = obp_atoms.positions
                obp_atom_names = [f"RES{i+1}" for i in range(len(OBP_RESIDUES_STD))]
                
                # 环原子的索引 (在配体中)
                ring_atom_indices = []
                if matched is not None:
                    for ring_atom in matched:
                        for i, lig_atom in enumerate(lig_res.atoms):
                            if lig_atom.index == ring_atom.index:
                                ring_atom_indices.append(i)
                                break
                
                # 生成 PDB 文件
                pdb_path = str(proj_path).replace('.png', '_structure.pdb')
                save_3d_structure_pdb(
                    lig_coords_3d, lig_atom_names_full,
                    obp_coords_3d, obp_atom_names, obp_ids,
                    pdb_path, cid,
                    ring_atom_indices=ring_atom_indices,
                    ring_weights=w if matched else None
                )
            except Exception as e:
                print(f"     [3D] Warning: Could not generate 3D structure: {e}")

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
    
    # 1. 关闭 MDAnalysis 轨迹文件句柄（非常重要！）
    if 'u' in locals():
        u.trajectory.close()
        del u  # 删除对象引用
        
    # 2. 清理巨大的临时列表
    del feature_vectors
    del data
    del all_angles_389
    del all_angles_390
    
    # 3. 强制清理 Matplotlib 的缓存（防止画图内存泄漏）
    plt.close('all')
    
    # 4. 手动触发一次垃圾回收
    gc.collect()
    
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
        cp = CubeParser(cubs[0])
        # 获取所有重原子的积分（而不仅仅是碳）
        # CubeParser 内部会计算所有重原子（原子序数 >= 6）
        ri = cp.get_carbon_integrals(INTEGRATION_RADIUS)
        
        if len(ri) == 0: 
            print(f"[Skip] {cid} no heavy atoms in cube file")
            continue
        
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

            print(f"     [GC] Cleaning up memory after {rn}...")
            gc.collect()
        
        if ts_list:
            OutputHandler.aggregate_timeseries(ts_list, OUTPUT_BASE_DIR, cid)
            OutputHandler.aggregate_stats(stat_list, OUTPUT_BASE_DIR, cid)
            print(f"  -> Done {cid}")

if __name__ == "__main__":
    main()
