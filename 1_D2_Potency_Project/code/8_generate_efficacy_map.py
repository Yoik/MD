import torch
import numpy as np
import MDAnalysis as mda
import os
import glob
import pickle
from tqdm import tqdm
from scipy.optimize import minimize
from src.model import EfficiencyPredictor

# ================= 配置参数 =================
REFERENCE_PDB = "data/step5_input.pdb"  # 用于获取锚点坐标
DATA_DIR = "data/features"
MODEL_PATH = "saved_models/best_model_mccv.pth" 
SCALER_PATH = "saved_models/scaler.pkl"
INPUT_DIM = 151

# 采样配置：每种等级抽取多少帧进行重构 (太大会跑很慢)
SAMPLES_PER_LEVEL = 500 

# 残基锚点 (必须与特征提取时一致)
OBP_RESIDUES = [114, 115, 118, 119, 190, 193, 194, 197, 386, 389, 390, 393, 412, 416]

# ================= 工具函数 =================
def get_anchor_coordinates(pdb_path, residue_ids):
    try:
        u = mda.Universe(pdb_path)
        coords = []
        for rid in residue_ids:
            sel = u.select_atoms(f"resid {rid} and name CA")
            if len(sel) == 0: sel = u.select_atoms(f"resid {rid}") # Fallback
            if len(sel) == 0: return None # Fail
            coords.append(sel.center_of_mass())
        return np.array(coords)
    except: return None

def reconstruction_loss(target_point, anchor_coords, target_distances):
    # loss = sum((dist_calc - dist_target)^2)
    current_dists = np.linalg.norm(anchor_coords - target_point, axis=1)
    return np.sum((current_dists - target_distances) ** 2)

def reconstruct_position(anchor_coords, target_dists, initial_guess):
    res = minimize(
        reconstruction_loss, 
        initial_guess, 
        args=(anchor_coords, target_dists),
        method='L-BFGS-B', tol=1e-3
    )
    return res.x if res.success else None

# ================= 主程序 =================
def main():
    device = torch.device("cpu")
    
    # 1. 加载资源
    print(f"Loading resources...")
    # 自动寻找 PDB
    ref_path = REFERENCE_PDB
    if not os.path.exists(ref_path):
        pdbs = glob.glob("data/**/*.pdb", recursive=True)
        if pdbs: ref_path = pdbs[0]
        else: print("[Error] No PDB found."); return

    anchor_coords = get_anchor_coordinates(ref_path, OBP_RESIDUES)
    if anchor_coords is None: print("Anchor load failed."); return
    
    model = EfficiencyPredictor(input_dim=INPUT_DIM).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    
    with open(SCALER_PATH, 'rb') as f: scaler = pickle.load(f)

    # 2. 全量预测与收集
    print("Scanning dataset for efficacy distribution...")
    all_frames = [] # Store tuples: (score, distance_vector_of_atoms)
    
    files = glob.glob(os.path.join(DATA_DIR, "*", "*", "*_features.npy"))
    
    for f_path in tqdm(files, desc="Predicting"):
        try:
            raw_data = np.load(f_path) # [Frames, 151]
            
            # 预处理输入给模型
            proc_data = raw_data.copy()
            for i in range(9): # 9 atoms
                proc_data[:, i*16 : i*16+14] = 1.0 / (proc_data[:, i*16 : i*16+14] + 1e-6)
            
            norm_data = scaler.transform(proc_data)
            tensor = torch.from_numpy(norm_data).float().to(device) # [F, 151]
            
            # 批量预测
            with torch.no_grad():
                out = model(tensor.unsqueeze(0)) # Add batch dim: [1, F, 151]
                scores = out["pred"].squeeze(0).numpy() # [F]
            
            # 收集数据 (只收集环区原子距离)
            # raw_data: [F, 151], 前 9*16 列是原子特征
            # 我们需要 raw distances: raw_data[:, atom_idx*16 : atom_idx*16+14]
            
            # 为了节省内存，我们只存 High/Med/Low 的样本索引，或者直接存
            # 这里简单处理：存所有，后面采样
            
            for i, score in enumerate(scores):
                # 提取这一帧所有原子的距离向量 [9, 14]
                frame_dists = []
                for a in range(9):
                    d = raw_data[i, a*16 : a*16+14]
                    # 简单过滤：太远的点不要 (溶剂中的点)
                    if np.min(d) < 12.0: 
                        frame_dists.append(d)
                
                if frame_dists:
                    all_frames.append((score, np.array(frame_dists)))

        except Exception as e: continue

    print(f"Total valid frames collected: {len(all_frames)}")
    
    # 3. 分层采样
    print("Stratifying and Sampling...")
    # 定义区间
    bins = {
        "HIGH": [], # Score > 90
        "MED":  [], # 60 < Score <= 90
        "LOW":  []  # Score <= 60
    }
    
    for score, dists in all_frames:
        if score > 90: bins["HIGH"].append((score, dists))
        elif score > 60: bins["MED"].append((score, dists))
        else: bins["LOW"].append((score, dists))
        
    print(f"Counts -> HIGH: {len(bins['HIGH'])}, MED: {len(bins['MED'])}, LOW: {len(bins['LOW'])}")
    
    samples_to_reconstruct = []
    
    # 从每个 bin 随机抽样
    for label, data_list in bins.items():
        if not data_list: continue
        n_sample = min(len(data_list), SAMPLES_PER_LEVEL)
        indices = np.random.choice(len(data_list), n_sample, replace=False)
        for idx in indices:
            score, atoms_dists = data_list[idx]
            # atoms_dists shape: [N_atoms, 14]
            for atom_d in atoms_dists:
                samples_to_reconstruct.append({
                    "label": label,
                    "score": score,
                    "dists": atom_d
                })

    # 4. 3D 重构
    print(f"Reconstructing {len(samples_to_reconstruct)} atom positions...")
    print("This involves geometric optimization, please wait...")
    
    results = []
    center_guess = np.mean(anchor_coords, axis=0)
    
    for item in tqdm(samples_to_reconstruct):
        pos = reconstruct_position(anchor_coords, item['dists'], center_guess)
        if pos is not None:
            item['pos'] = pos
            results.append(item)
            
    # 5. 输出 PDB
    out_file = "Efficacy_Map.pdb"
    print(f"Saving to {out_file}...")
    
    with open(out_file, 'w') as f:
        f.write("REMARK   EFFICACY DISTRIBUTION MAP\n")
        f.write("REMARK   CHAIN A: HIGH SCORE (>90)\n")
        f.write("REMARK   CHAIN B: MED  SCORE (60-90)\n")
        f.write("REMARK   CHAIN C: LOW  SCORE (<60)\n")
        f.write("REMARK   B-FACTOR COLUMN = PREDICTED SCORE\n")
        
        atom_idx = 1
        # 先写锚点 (Chain Z)
        for i, xyz in enumerate(anchor_coords):
            f.write(f"ATOM  {atom_idx:5d}  CA  REF Z{i+1:3d}    "
                    f"{xyz[0]:8.3f}{xyz[1]:8.3f}{xyz[2]:8.3f}  1.00  0.00           C\n")
            atom_idx += 1
            
        # 写数据点
        # 映射 label 到 chain
        chain_map = {"HIGH": "A", "MED": "B", "LOW": "C"}
        
        for res in results:
            chn = chain_map[res['label']]
            x, y, z = res['pos']
            s = res['score']
            
            # 限制 score 显示范围防止 PDB 格式溢出
            s_disp = max(0.0, min(99.99, s))
            
            f.write(f"ATOM  {atom_idx:5d}  C   MAP {chn}{atom_idx % 9999:4d}    "
                    f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00 {s_disp:6.2f}           C\n")
            atom_idx += 1
            
        f.write("END\n")
        
    print("\nDone! Visualization Instructions:")
    print("1. Open 'Efficacy_Map.pdb' in PyMOL.")
    print("2. Use this command to color by chain (Efficacy Zones):")
    print("   color green, chain A  (High Potency Zone)")
    print("   color yellow, chain B (Medium Potency Zone)")
    print("   color red, chain C    (Low Potency Zone)")
    print("3. Or color by exact score:")
    print("   spectrum b, red_yellow_green, chain A+B+C")

if __name__ == "__main__":
    main()