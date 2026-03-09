import torch
import numpy as np
import MDAnalysis as mda
from MDAnalysis.lib.distances import distance_array
import os
import glob
import pickle
from tqdm import tqdm
from scipy.optimize import minimize
from src.model import EfficiencyPredictor
from scipy.stats import gaussian_kde

# 引入解释性分析工具
try:
    from captum.attr import IntegratedGradients
except ImportError:
    print("[ERROR] Please install captum: pip install captum")
    exit()

# ================= 配置参数 =================
# 【请修改这里】指向一个包含受体结构的 PDB 文件 (用于获取锚点坐标)
# 最好是 Dopa 的结构，或者是你 MD 的第一帧
REFERENCE_PDB = "data/step5_input.pdb" 

# 核心过滤参数 (新增)
SCORE_THRESHOLD = 0.80  # 找回 Dopa 的关键
ATTN_THRESHOLD = 1.0    # 关注被注意力机制选中的帧
TOP_K_ATOMS = 3         # 消除原子数量偏差

DATA_DIR = "data/features"
MODEL_PATH = "saved_models/best_model_mccv.pth" 
SCALER_PATH = "saved_models/scaler.pkl"
INPUT_DIM = 151
BF_THRESHOLD = 0   # 你想要的阈值

# 维度参数 (根据你最新的特征提取脚本调整)
# Input Dim = 151
# Atom Block = 144 (9 atoms * 16 features)
# Per Atom Features = 16 (14 Distances + 2 Scores)
N_ATOMS = 9
N_ATOM_FEAT = 16
N_DIST = 14

# OBP 残基列表 (必须与提取特征时的顺序一致)
# 这里我们需要残基 ID 来从 PDB 里找坐标
OBP_RESIDUES = [114, 115, 118, 119, 190, 193, 194, 197, 386, 389, 390, 393, 412, 416]
def get_anchor_coordinates(pdb_path, residue_ids):
    """从 PDB 中提取 14 个锚点残基的 CA 原子坐标"""
    try:
        u = mda.Universe(pdb_path)
        # 注意：这里假设 PDB 的残基编号与 MD 一致。
        # 如果不一致，你可能需要用 sequence alignment 或手动校正
        # 这里简单起见，假设 residue numbers 匹配
        coords = []
        names = []
        for rid in residue_ids:
            # 尝试选 CA，如果没有 CA 选 C 或 N
            sel = u.select_atoms(f"resid {rid} and name CA")
            if len(sel) == 0:
                print(f"[Warn] CA for resid {rid} not found, trying center of mass.")
                sel = u.select_atoms(f"resid {rid}")
                pos = sel.center_of_mass()
            else:
                pos = sel.positions[0]
            coords.append(pos)
            names.append(f"R{rid}")
            
        return np.array(coords), names
    except Exception as e:
        print(f"[Error] Failed to load anchors from PDB: {e}")
        return None, None

def get_residue_atomgroup(u, residue_ids):
    sel = u.select_atoms(f"resid {' '.join(map(str, residue_ids))}")
    if len(sel) == 0:
        raise ValueError("No residue atoms found for given residue_ids.")
    return sel


def reconstruction_loss(target_point, anchor_coords, target_distances):
    """
    损失函数：计算当前点到锚点的距离，与目标距离的差异
    Loss = Sum( (Calculated_Dist - Target_Dist)^2 )
    """
    # target_point: [x, y, z]
    # anchor_coords: [12, 3]
    # target_distances: [12]
    
    current_dists = np.linalg.norm(anchor_coords - target_point, axis=1)
    diff = current_dists - target_distances
    return np.sum(diff ** 2)

def reconstruct_point_robust(target_dists, anchor_coords, center_guess, max_tries=5):
    """
    尝试多次不同的初始位置来进行三维重构，防止陷入几何对称陷阱。
    """
    best_pos = None
    best_loss = float('inf')
    
    for i in range(max_tries):
        # 第1次尝试：使用几何中心（保留原有逻辑）
        if i == 0:
            current_guess = center_guess
        # 后续尝试：在中心附近随机“踹一脚”（关键修改！）
        else:
            # 在 +/- 5.0 埃范围内随机扰动 X, Y, Z
            noise = np.random.uniform(-5.0, 5.0, size=3)
            current_guess = center_guess + noise

        # 执行优化
        res = minimize(
            reconstruction_loss, 
            current_guess, 
            args=(anchor_coords, target_dists),
            method='L-BFGS-B',
            tol=1e-4
        )

        # 记录最好的结果
        if res.success:
            if res.fun < best_loss:
                best_loss = res.fun
                best_pos = res.x
            
            # 如果误差极小（说明找到了完美解），可以直接提前返回
            if best_loss < 0.1:
                return best_pos

    return best_pos

def main():
    device = torch.device("cpu")
    
    # === 修复点：使用局部变量承接全局配置 ===
    # 将全局的 REFERENCE_PDB 赋值给局部变量 ref_path
    ref_path = REFERENCE_PDB 
    
    # 1. 获取锚点坐标
    print(f"Loading Anchor Coordinates from {ref_path}...")
    
    # 检查路径是否存在
    if not os.path.exists(ref_path):
        print(f"Error: Reference PDB not found: {ref_path}")
        # 自动容错逻辑
        # 尝试在 data 目录下随便找一个 pdb 文件作为参考
        pdbs = glob.glob("data/**/*.pdb", recursive=True)
        if pdbs:
            ref_path = pdbs[0] # 更新局部变量
            print(f"-> Auto-selected fallback PDB: {ref_path}")
        else:
            print("[Critical] No PDB files found in 'data/' to extract anchors.")
            return

    # 使用 ref_path 继续后续操作
    anchor_coords, anchor_names = get_anchor_coordinates(ref_path, OBP_RESIDUES)
    
    if anchor_coords is None: 
        print("Failed to load anchors. Exiting.")
        return
        
    print(f"Anchors loaded. Shape: {anchor_coords.shape}")
    # 2. 加载模型 & 筛选高贡献距离 (复用之前的逻辑)
    print("Loading Model & Scanning Atoms...")
    model = EfficiencyPredictor(input_dim=INPUT_DIM).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    
    with open(SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)

    files = glob.glob(os.path.join(DATA_DIR, "*", "*", "*_features.npy"))
    ig = IntegratedGradients(lambda x: model(x)["pred"])
    
    target_distance_vectors = []
    
    # === 修复后的循环逻辑 ===
    print(f"Scanning files with Trajectory Score > {SCORE_THRESHOLD}...")
    
    for f_path in tqdm(files):
        try:
            raw_data = np.load(f_path) # Shape: [Frames, 151]
            n_frames = raw_data.shape[0]

            # 1. 预处理 (保持不变)
            proc_data = raw_data.copy()
            for i in range(N_ATOMS):
                start = i * 16
                end_dist = start + 14 
                proc_data[:, start:end_dist] = 1.0 / (proc_data[:, start:end_dist] + 1e-6)
            
            norm_data = scaler.transform(proc_data)
            input_tensor = torch.from_numpy(norm_data).float().unsqueeze(0).to(device)
            
            # 2. 获取每帧的详细评分
            with torch.no_grad():
                out = model(input_tensor)
                traj_score = out["pred"].item()         # 整条轨迹的总分
                frame_scores = out["frame_scores"].squeeze().cpu().numpy() # [Frames] 每一帧的分数
                
            # 3. 筛选逻辑：只处理高分轨迹
            if traj_score > SCORE_THRESHOLD: 
                
                # 【关键修复】找出这一条轨迹里，哪些帧是真正的“高分姿态”
                # 例如：只取单帧分数 > 0.8 的帧
                high_quality_indices = np.where(frame_scores > 0.8)[0]
                
                if len(high_quality_indices) == 0: continue

                # 【降采样】为了防止点太多，每个文件最多只取 20 帧
                # 如果你想看更密的云，可以把 20 改成 50 或 100
                if len(high_quality_indices) > 20:
                    selected_indices = np.random.choice(high_quality_indices, size=20, replace=False)
                else:
                    selected_indices = high_quality_indices
                
                # 4. 遍历选中的帧 (不再是 0 了！)
                for frame_idx in selected_indices:
                    
                    # 提取这一帧的所有原子距离
                    # raw_data[frame_idx]  <--- 这里动起来了！
                    frame_feats = raw_data[frame_idx] 
                    
                    for i in range(N_ATOMS):
                        start = i * 16
                        atom_feat = frame_feats[start : start+16]
                        
                        atom_dists = atom_feat[:14] # 距离部分
                        elec_score = atom_feat[14] + atom_feat[15] # 电子分
                        
                        # 再次过滤：只画靠近口袋且有贡献的原子
                        # 距离 < 8.0 且 电子得分 > 0.1 (排除无效噪点)
                        if np.min(atom_dists) < 8.0 and elec_score > 0.1:
                            target_distance_vectors.append(atom_dists)

        except Exception as e:
            # print(f"Error processing {f_path}: {e}")
            continue        
    if not target_distance_vectors:
        print("No valid atoms found.")
        return

    all_targets = np.concatenate(target_distance_vectors, axis=0)
    print(f"\nReconstructing 3D positions for {len(all_targets)} points...")
    print("This may take a minute (Optimization loop)...")

    # 3. 核心：三维重建 (Multilateration)
    reconstructed_points = []
    
    # 初始猜测：锚点的几何中心 (作为起跑线)
    initial_guess = np.mean(anchor_coords, axis=0)

    print(f"Base Center Guess: {initial_guess}")

    # 为了速度，如果点太多，可以降采样
    if len(all_targets) > 2000:
        indices = np.random.choice(len(all_targets), 2000, replace=False)
        targets_to_process = all_targets[indices]
    else:
        targets_to_process = all_targets

    for target_dists in tqdm(targets_to_process, desc="Reconstructing 3D"):
        # 它会自动尝试多个起点，打破对称性陷阱
        pos = reconstruct_point_robust(target_dists, anchor_coords, initial_guess)
        
        if pos is not None:
            reconstructed_points.append(pos)

    reconstructed_points = np.array(reconstructed_points)
    
    # 4. 计算密度 (作为 B-factor)
    print("Calculating Cloud Density (for B-factors)...")
    xyz = reconstructed_points.T
    try:
        kde = gaussian_kde(xyz)
        densities = kde(xyz)
        # 归一化到 0.00 - 100.00
        b_factors = 100 * (densities - densities.min()) / (densities.max() - densities.min())
    except:
        print("KDE failed (points might be coplanar), using constant B-factor.")
        b_factors = np.ones(len(reconstructed_points)) * 50.0

    # 5. 保存为 PDB
    output_pdb = "Carbon_Cloud_Reconstructed.pdb"
    print(f"\nSaving {len(reconstructed_points)} points to {output_pdb}...")
    
    with open(output_pdb, 'w') as f:
        f.write("REMARK  GENERATED BY AI RECONSTRUCTION\n")
        # 先把锚点写进去 (作为参考骨架)，Chain B
        atom_idx = 1
        for i, pos in enumerate(anchor_coords):
            # ATOM      1  CA  ASP B 115 ...
            f.write(f"ATOM  {atom_idx:5d}  CA  RES B{OBP_RESIDUES[i]:3d}    "
                    f"{pos[0]:8.3f}{pos[1]:8.3f}{pos[2]:8.3f}  1.00  0.00           C\n")
            atom_idx += 1
            
        # 再把重构的云点写进去，Chain A，Residue CLD (Cloud)
        for i, pos in enumerate(reconstructed_points):
            # 用 B-factor 100 标记，方便 PyMOL 着色
            bf = b_factors[i]
            if bf < BF_THRESHOLD: continue  # 只写入高密度点
            f.write(f"ATOM  {atom_idx:5d}  C   CLD A   1    "
                    f"{pos[0]:8.3f}{pos[1]:8.3f}{pos[2]:8.3f}  1.00 {bf:6.2f}           C\n")
            atom_idx += 1
            
        f.write("END\n")

    print("Done! Open 'Carbon_Cloud_Reconstructed.pdb' in PyMOL.")
    print("Tip: Use 'show spheres' for Chain A and 'show sticks' for Chain B.")

if __name__ == "__main__":
    main()