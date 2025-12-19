import torch
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob
import pickle
from tqdm import tqdm
from matplotlib.gridspec import GridSpec

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
INPUT_DIM = 133  # DeepSets (9*14 + 7)

# OBP 残基列表 (对应前12维距离)
OBP_LABELS = [
    "V114", "D115", "M118", "P119", 
    "D190", "S193", "V194", "F197", 
    "H386", "H393", "W412", "Y416"
]

def main():
    device = torch.device("cpu") 
    
    # 1. 加载模型
    print(f"Loading model from {MODEL_PATH}...")
    model = EfficiencyPredictor(input_dim=INPUT_DIM).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    
    # 2. 加载 Scaler
    with open(SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)

    # 3. 准备数据文件
    files = glob.glob(os.path.join(DATA_DIR, "*", "*", "*_features.npy"))
    print(f"Found {len(files)} trajectory files.")

    # 4. 定义归因函数
    ig = IntegratedGradients(lambda x: model(x)["pred"])

    # 容器：存储"黄金原子"的原始距离坐标
    good_atom_locations = [] 

    print("Scanning for High-Value Carbon Clouds...")
    
    for f_path in tqdm(files):
        try:
            # === 数据加载与预处理 (与之前一致) ===
            raw_data = np.load(f_path) 
            proc_data = raw_data.copy()
            
            n_atoms = 9
            n_feat = 14
            for i in range(n_atoms):
                start = i * n_feat
                end_dist = start + 12
                # 距离取倒数
                proc_data[:, start:end_dist] = 1.0 / (proc_data[:, start:end_dist] + 1e-6)
            
            norm_data = scaler.transform(proc_data)
            
            input_tensor = torch.from_numpy(norm_data).float().unsqueeze(0).to(device)
            input_tensor.requires_grad = True
            
            # === 运行 Integrated Gradients ===
            attributions = ig.attribute(input_tensor, n_steps=5) 
            attr_np = attributions.squeeze(0).detach().numpy()   
            
            # === 原子筛选 ===
            atom_attrs = attr_np[:, :126].reshape(-1, 9, 14) 
            atom_raw_dists = raw_data[:, :126].reshape(-1, 9, 14)[:, :, :12]
            
            # 计算总贡献
            atom_importance = np.sum(atom_attrs, axis=2) 
            
            # 阈值策略：Top 10%
            threshold = np.percentile(atom_importance, 90)
            if threshold <= 0: continue
            
            mask = atom_importance > threshold
            selected_dists = atom_raw_dists[mask]
            
            # 过滤 Padding
            real_atom_mask = np.min(selected_dists, axis=1) < 15.0
            good_atom_locations.append(selected_dists[real_atom_mask])

        except Exception as e:
            continue
            
    if not good_atom_locations:
        print("No valid atoms found. Check model or data.")
        return

    # 合并数据
    all_locations = np.concatenate(good_atom_locations, axis=0)
    print(f"\nIdentified {len(all_locations)} high-contribution atomic states.")
    
    # =========================================================
    # 核心修改：三图合一面板 (Combined Panel)
    # =========================================================
    print("Generating Atom Cloud Panel...")
    
    fig = plt.figure(figsize=(20, 14))
    # 定义 2行 2列 的网格
    # height_ratios=[1.2, 1]: 上面那行稍微高一点
    gs = GridSpec(2, 2, figure=fig, height_ratios=[1.2, 1], wspace=0.2, hspace=0.3)
    
    df_loc = pd.DataFrame(all_locations, columns=OBP_LABELS)

    # --- 图 A: 空间指纹 (占据第一行整行) ---
    ax1 = fig.add_subplot(gs[0, :]) # span all columns
    df_long = df_loc.melt(var_name="Residue", value_name="Distance (Å)")
    sns.violinplot(x="Residue", y="Distance (Å)", data=df_long, inner="quartile", palette="viridis", ax=ax1)
    ax1.set_title("A. Spatial Fingerprint of High-Efficacy Carbon Atoms", fontsize=16, fontweight='bold', loc='left')
    ax1.set_ylabel("Distance to Residue (Å)", fontsize=12)
    ax1.set_xlabel("")
    ax1.set_ylim(0, 20) # 聚焦在 10A 以内
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    
    # --- 图 B: 关键残基定位 (左下角) ---
    # 自动寻找方差最小（最保守）的两个残基作为锚点，或者手动指定
    # 这里我们用方差排序，找"定位最准"的两个残基
    variances = df_loc.var().sort_values()
    best_res = variances.index[:2].tolist() # 比如 ['D115', 'S193']
    x_res, y_res = best_res[0], best_res[1]
    
    ax2 = fig.add_subplot(gs[1, 0])
    
    x_vals = df_loc[x_res]
    y_vals = df_loc[y_res]
    # 简单过滤极值以便绘图好看
    mask = (x_vals < 12) & (y_vals < 12)
    
    # KDE Plot
    sns.kdeplot(x=x_vals[mask], y=y_vals[mask], fill=True, cmap="inferno", thresh=0.05, ax=ax2)
    # 叠加少量散点增加质感
    ax2.scatter(x_vals[mask][::5], y_vals[mask][::5], s=2, color='cyan', alpha=0.15)
    
    ax2.set_title(f"B. Optimal Position Map: {x_res} vs {y_res}", fontsize=14, fontweight='bold', loc='left')
    ax2.set_xlabel(f"Distance to {x_res} (Å)", fontsize=12)
    ax2.set_ylabel(f"Distance to {y_res} (Å)", fontsize=12)
    ax2.grid(True, alpha=0.2)

    # --- 图 C: 理想坐标条 (右下角) ---
    ax3 = fig.add_subplot(gs[1, 1])
    ideal_profile = df_loc.mean().to_frame().T
    
    # 画 Heatmap
    sns.heatmap(ideal_profile, annot=True, fmt=".1f", cmap="Blues_r", cbar=True, 
                cbar_kws={'label': 'Mean Distance (Å)', 'orientation': 'horizontal'},
                annot_kws={"size": 11, "weight": "bold"}, ax=ax3)
    
    ax3.set_title("C. Ideal 'Carbon Vector' (Mean Distance Profile)", fontsize=14, fontweight='bold', loc='left')
    ax3.set_yticks([]) # 不需要 y 轴标签
    ax3.set_xlabel("Residue ID", fontsize=12)

    # 保存
    plt.tight_layout()
    output_file = "Global_Atom_Cloud_Panel.png"
    plt.savefig(output_file, dpi=300)
    print(f"Panel saved to: {output_file}")

if __name__ == "__main__":
    main()