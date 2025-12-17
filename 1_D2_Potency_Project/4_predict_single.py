import sys
import os
import glob
import torch
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle
from src.model import EfficiencyPredictor

# ================= 配置参数 =================
LABEL_FILE = "data/labels.csv"
RESULT_DIR = "data/features"
MODEL_PATH = "saved_models/best_model_mccv.pth" 
SCALER_PATH = "saved_models/scaler.pkl"
INPUT_DIM = 19  
# ===========================================

def analyze_compound_components(compound_name, model, scaler, device, true_val):
    print(f"\nAnalyzing Components for: {compound_name}")
    
    # 1. 寻找文件
    compound_folder_pattern = os.path.join(RESULT_DIR, f"*{compound_name}*")
    matching_folders = glob.glob(compound_folder_pattern)
    if not matching_folders:
        print(f"Error: No folder found for {compound_name}")
        return

    npy_files = []
    for folder in matching_folders:
        npy_search = os.path.join(folder, "*", "*_features.npy")
        npy_files.extend(glob.glob(npy_search))
    
    if not npy_files:
        print("No .npy files found.")
        return

    # 2. 准备绘图 (三个 Y 轴有点挤，我们画两张图或者用双Y轴)
    # 方案：图1画 分数和门控，图2画 Attention
    fig, (ax1, ax3) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    ax2 = ax1.twinx()  # Ax1 的双胞胎，画 Gate

    total_geom = []
    total_gate = []
    total_final = []
    total_attn = []

    for f_path in npy_files:
        try:
            raw_data = np.load(f_path)
            
            # === 预处理 ===
            data_proc = raw_data.copy()
            dist_indices = list(range(12)) 
            data_proc[:, dist_indices] = 1.0 / (data_proc[:, dist_indices] + 1e-6)
            data_proc = scaler.transform(data_proc)
            
            input_tensor = torch.from_numpy(data_proc).float().unsqueeze(0).to(device)
            
            # === 模型推理 (接收 4 个返回值) ===
            model.eval()
            with torch.no_grad():
                # 【关键修改】接收 4 个值
                _, final_scores, gate_vals, attn_weights = model(input_tensor)
                
                gate_np = gate_vals.squeeze().cpu().numpy().flatten()
                final_np = final_scores.squeeze().cpu().numpy().flatten()
                attn_np = attn_weights.squeeze().cpu().numpy().flatten()
                
                # 反推几何分 (Final = Geom * Gate)
                # Geom = Final / (Gate + epsilon)
                geom_np = final_np / (gate_np + 1e-6)

            # 收集数据 (不做缩放，直接用原始模型输出的 0-1)
            total_geom.extend(geom_np) 
            total_gate.extend(gate_np)        
            total_final.extend(final_np)
            total_attn.extend(attn_np)
            
        except Exception as e:
            print(f"Error processing {f_path}: {e}")

    if not total_final:
        print("No data processed.")
        return

    # 3. 绘制曲线 (平滑处理)
    window = 100
    df = pd.DataFrame({
        'Geom': total_geom, 
        'Gate': total_gate, 
        'Final': total_final,
        'Attn': total_attn
    })
    df_smooth = df.rolling(window=window).mean()

    # --- 子图 1: 效能与门控 ---
    # 左轴: 效能 (0-1)
    
    l1, = ax1.plot(df_smooth['Geom'], color='green', linestyle='--', alpha=0.5, label='Geometry Score (Affinity)')
    l2, = ax1.plot(df_smooth['Final'], color='blue', linewidth=2, label='Final Prediction (Efficacy)')
    ax1.axhline(y=true_val, color='red', linestyle=':', label=f'True Efficacy ({true_val})')
    
    ax1.set_ylabel('Efficacy Score (0-1)', color='blue')
    ax1.set_ylim(-0.1, 1.2)
    ax1.set_title(f"Dual-Stream Analysis: {compound_name}", fontsize=14)

    # 右轴: 门控 (0-1)
    l3, = ax2.plot(df_smooth['Gate'], color='orange', linewidth=2, linestyle='-', alpha=0.7, label='Electronic Gate')
    ax2.set_ylabel('Gate Probability', color='orange')
    ax2.set_ylim(0, 1.1)

    # 图例
    lines = [l1, l2, l3]
    ax1.legend(lines, [l.get_label() for l in lines], loc='upper center', ncol=3)
    ax1.grid(True, alpha=0.3)

    # --- 子图 2: 注意力权重 ---
    # 注意力通常很稀疏，峰值很高
    ax3.plot(df['Attn'], color='purple', alpha=0.3, label='Raw Attention') # 原始噪点
    ax3.plot(df_smooth['Attn'], color='purple', linewidth=2, label='Smoothed Attention') # 平滑曲线
    ax3.set_ylabel('Attention Weight', color='purple')
    ax3.set_xlabel('Frame Index')
    ax3.legend(loc='upper right')
    ax3.grid(True, alpha=0.3)
    ax3.set_title("Temporal Attention Distribution (Key Frames)", fontsize=12)

    plt.tight_layout()
    save_name = f"analysis_components_{compound_name}.png"
    plt.savefig(save_name, dpi=300)
    print(f"Analysis saved to: {save_name}")
    print(f"  -> Avg Geometry Score: {np.mean(total_geom):.4f}")
    print(f"  -> Avg Gate Value:     {np.mean(total_gate):.4f}")
    print(f"  -> Avg Final Score:    {np.mean(total_final):.4f}")

def main():
    if len(sys.argv) < 2:
        print("Usage: python 4_analyze_trajectory.py <CompoundName>")
        return
    target_compound = sys.argv[1]
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 加载模型 (确保是 19 维)
    model = EfficiencyPredictor(input_dim=INPUT_DIM).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    
    with open(SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)

    # 获取 True Label (不需要除以100了，已经是0-1)
    df = pd.read_csv(LABEL_FILE)
    row = df[df['Compound'].astype(str).str.contains(target_compound, case=False)]
    true_val = float(row.iloc[0]['Efficacy']) if len(row) > 0 else 0.0

    analyze_compound_components(target_compound, model, scaler, device, true_val)

if __name__ == "__main__":
    main()