import torch
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import os
import glob
import pickle
from tqdm import tqdm
from scipy.stats import gaussian_kde

# 引入解释性分析工具
try:
    from captum.attr import IntegratedGradients
except ImportError:
    print("[ERROR] Please install captum: pip install captum")
    exit()

from src.model import EfficiencyPredictor

# ================= 配置参数 =================
DATA_DIR = "data/features"
MODEL_PATH = "saved_models/best_model_mccv.pth" 
SCALER_PATH = "saved_models/scaler.pkl"
INPUT_DIM = 151

# OBP 残基列表
OBP_LABELS = [
    "V114", "D115", "M118", "P119", 
    "D190", "S193", "V194", "F197", 
    "H386", "F389", "F390", "H393", 
    "W412", "Y416"
]

def main():
    device = torch.device("cpu")
    
    # 1. 加载模型 & 数据 (与之前相同)
    print(f"Loading model...")
    model = EfficiencyPredictor(input_dim=INPUT_DIM).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    
    with open(SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)

    files = glob.glob(os.path.join(DATA_DIR, "*", "*", "*_features.npy"))
    ig = IntegratedGradients(lambda x: model(x)["pred"])
    good_atom_locations = [] 

    print("Scanning for 'Sweet Spot' Atom Cloud...")
    
    # --- 筛选高贡献原子逻辑 (同前) ---
    for f_path in tqdm(files):
        try:
            raw_data = np.load(f_path)
            proc_data = raw_data.copy()
            n_atoms = 9
            for i in range(n_atoms):
                start = i * 16; end_dist = start + 14
                proc_data[:, start:end_dist] = 1.0 / (proc_data[:, start:end_dist] + 1e-6)
            
            norm_data = scaler.transform(proc_data)
            input_tensor = torch.from_numpy(norm_data).float().unsqueeze(0).to(device)
            input_tensor.requires_grad = True
            
            attributions = ig.attribute(input_tensor, n_steps=5)
            attr_np = attributions.squeeze(0).detach().numpy()
            
            atom_attrs = attr_np[:, :144].reshape(-1, 9, 16)
            atom_raw_dists = raw_data[:, :144].reshape(-1, 9, 16)[:, :, :14]
            
            # 使用更严格的阈值 (Top 5%) 以获得更干净的核心云
            atom_importance = np.sum(atom_attrs, axis=2)
            threshold = np.percentile(atom_importance, 95)
            
            if threshold <= 0: continue
            
            mask = atom_importance > threshold
            selected_dists = atom_raw_dists[mask]
            
            # 过滤 Padding
            real_atom_mask = np.min(selected_dists, axis=1) < 15.0
            good_atom_locations.append(selected_dists[real_atom_mask])

        except Exception:
            continue
            
    if not good_atom_locations:
        print("No valid atoms found.")
        return

    all_locations = np.concatenate(good_atom_locations, axis=0)
    df_loc = pd.DataFrame(all_locations, columns=OBP_LABELS)
    
    # =========================================================
    # 3D 可视化部分
    # =========================================================
    print(f"\nGeneratin 3D Phase Space for {len(all_locations)} points...")

    # 1. 自动寻找最重要的 3 个轴 (方差最小 = 限制最死 = 最重要)
    # 比如 D115 必须是 3.5A (方差小)，而 V194 可以是 3-10A (方差大)
    variances = df_loc.var().sort_values()
    top_3_res = variances.index[:3].tolist() # e.g. ['D115', 'S193', 'F197']
    print(f"Selected Axes (Most Constrained): {top_3_res}")
    
    x_data = df_loc[top_3_res[0]]
    y_data = df_loc[top_3_res[1]]
    z_data = df_loc[top_3_res[2]]

    # 简单过滤极值 (只看口袋内部)
    mask = (x_data < 10) & (y_data < 10) & (z_data < 10)
    x_data, y_data, z_data = x_data[mask], y_data[mask], z_data[mask]

    # 2. 计算 3D 密度 (作为颜色)
    # 为了速度，先降采样算密度，再插值 (如果点太多)
    xyz = np.vstack([x_data, y_data, z_data])
    try:
        density = gaussian_kde(xyz)(xyz)
    except:
        density = np.ones_like(x_data) # Fallback

    # 3. 绘图
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Scatter plot with density coloring (Inferno colormap looks 'hot')
    sc = ax.scatter(x_data, y_data, z_data, c=density, cmap='inferno', 
                    s=10, alpha=0.6, marker='o', edgecolors='none')

    # 美化坐标轴
    ax.set_xlabel(f"Dist to {top_3_res[0]} (Å)", fontsize=12, labelpad=10)
    ax.set_ylabel(f"Dist to {top_3_res[1]} (Å)", fontsize=12, labelpad=10)
    ax.set_zlabel(f"Dist to {top_3_res[2]} (Å)", fontsize=12, labelpad=10)
    
    ax.set_title(f"3D 'Sweet Spot' Cloud\n(Defining the Perfect Efficacy Pocket)", fontsize=16, fontweight='bold')
    
    # 添加颜色条
    cbar = plt.colorbar(sc, shrink=0.6, pad=0.1)
    cbar.set_label('Interaction Probability Density', rotation=270, labelpad=15)

    # 4. 保存多视角
    angles = [(30, 45), (60, 120), (20, -60)] # (elev, azim)
    for i, (elev, azim) in enumerate(angles):
        ax.view_init(elev=elev, azim=azim)
        plt.tight_layout()
        fname = f"3D_Atom_Cloud_View_{i+1}.png"
        plt.savefig(fname, dpi=300)
        print(f"Saved 3D view: {fname}")

    print("Done! Check the '3D_Atom_Cloud_View_*.png' files.")

if __name__ == "__main__":
    main()