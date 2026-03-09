import torch
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pickle
import glob
from tqdm import tqdm

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

# 【关键参数】输入维度 151
INPUT_DIM = 151 

# 14 个残基 (对应特征提取中的顺序)
OBP_LABELS = [
    "V114", "D115", "M118", "P119", 
    "D190", "S193", "V194", "F197", 
    "H386", "F389", "F390", "H393", "W412", "Y416"
]

# === 动态生成 151 维特征名称 ===
FEATURE_NAMES = []

# A. 原子特征 (0-143): 9个原子 * 16个特征 (14距离 + 2电子)
for i in range(9):
    # 1. 距离特征 (0-13)
    for res in OBP_LABELS:
        FEATURE_NAMES.append(f"Atom{i}_{res}_Dist")
    # 2. 电子特征 (14-15)
    FEATURE_NAMES.append(f"Atom{i}_Phe389_Score")
    FEATURE_NAMES.append(f"Atom{i}_Phe390_Score")

# B. 全局特征 (144-150): 7个
GLOBAL_NAMES = [
    "Global_Angle_Cos",
    "Global_Phe389_Sum", "Global_Phe389_Max", "Global_Phe389_Conc",
    "Global_Phe390_Sum", "Global_Phe390_Max", "Global_Phe390_Conc"
]
FEATURE_NAMES.extend(GLOBAL_NAMES)

print(f"Total Feature Names generated: {len(FEATURE_NAMES)} (Target: 151)")

def main():
    device = torch.device("cpu") 
    
    print(f"Loading model from {MODEL_PATH} ...")
    if not os.path.exists(MODEL_PATH):
        print("Model file not found.")
        return

    # 1. 初始化模型
    model = EfficiencyPredictor(input_dim=INPUT_DIM).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    
    # 2. 加载 Scaler
    with open(SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)

    # 3. 寻找数据
    files = glob.glob(os.path.join(DATA_DIR, "*", "*", "*_features.npy"))
    print(f"Found {len(files)} trajectory files.")
    if not files: return

    # 4. 定义 IG 目标函数
    ig = IntegratedGradients(lambda x: model(x)["pred"])

    # 容器
    all_feature_importances = [] 
    all_score_corrs = []  # 改名：与 Frame Score 的相关性
    all_attn_corrs = []   # 与 Attention 的相关性

    print("Running Interpretation Loop...")
    for file_path in tqdm(files):
        try:
            raw_data = np.load(file_path) # [T, 151]
            if raw_data.shape[0] == 0: continue
            
            # 预处理 (距离取倒数 + 标准化)
            data_proc = raw_data.copy()
            n_atoms = 9; n_feat = 16; n_dist = 14
            for i in range(n_atoms):
                start = i * n_feat
                end_dist = start + n_dist
                data_proc[:, start:end_dist] = 1.0 / (data_proc[:, start:end_dist] + 1e-6)

            data_proc = scaler.transform(data_proc)
            input_tensor = torch.from_numpy(data_proc).float().unsqueeze(0).to(device)
            input_tensor.requires_grad = True

            # A. 归因分析 (IG)
            attributions = ig.attribute(input_tensor, n_steps=10)
            traj_imp = attributions.sum(dim=1).squeeze(0).detach().numpy()
            all_feature_importances.append(traj_imp)

            # B. 提取中间变量 (Score & Attn)
            with torch.no_grad():
                out = model(input_tensor)
                scores = out["frame_scores"].squeeze().numpy() # [T]
                attns = out["attn"].squeeze().numpy()          # [T]
            
            # C. 计算相关性
            # 我们想知道：哪个特征变大时，Frame Score 也会变大？
            features = data_proc
            if scores.ndim==0: scores=np.array([scores])
            if attns.ndim==0: attns=np.array([attns])

            sc_c = []; at_c = []
            for i in range(INPUT_DIM):
                feat_col = features[:, i]
                # 计算 Pearson 相关系数
                if np.std(feat_col) < 1e-6 or np.std(scores) < 1e-6: sc_c.append(0)
                else: sc_c.append(np.corrcoef(feat_col, scores)[0, 1])

                if np.std(feat_col) < 1e-6 or np.std(attns) < 1e-6: at_c.append(0)
                else: at_c.append(np.corrcoef(feat_col, attns)[0, 1])
            
            all_score_corrs.append(sc_c)
            all_attn_corrs.append(at_c)

        except Exception:
            continue

    if not all_feature_importances: return

    # 5. 聚合结果
    avg_importances = np.mean(all_feature_importances, axis=0)
    avg_score_drivers = np.mean(all_score_corrs, axis=0)
    avg_attn_drivers = np.mean(all_attn_corrs, axis=0)

    # =========================================================
    # 核心修改：打印数据 + 绘图
    # =========================================================
    print("\n" + "="*60)
    print("INTERPRETATION REPORT")
    print("="*60)

    fig, axes = plt.subplots(1, 3, figsize=(26, 12))

    # --- Plot A: Global Importance ---
    analyze_and_plot(
        ax=axes[0], 
        values=avg_importances, 
        title="A. Global Feature Importance (IG)", 
        xlabel="Integrated Gradients Contribution",
        print_label="Global Importance"
    )

    # --- Plot B: Frame Score Drivers ---
    analyze_and_plot(
        ax=axes[1], 
        values=avg_score_drivers, 
        title="B. Frame Score Drivers (Correlation)", 
        xlabel="Correlation with Frame Score (Quality)",
        print_label="Frame Quality Drivers",
        color_theme='purple'
    )

    # --- Plot C: Attention Drivers ---
    analyze_and_plot(
        ax=axes[2], 
        values=avg_attn_drivers, 
        title="C. Attention Drivers (Correlation)", 
        xlabel="Correlation with Attention Weight",
        print_label="Attention Drivers",
        color_theme='purple'
    )

    plt.tight_layout()
    plt.savefig("Global_Interpretation_Panel.png", dpi=300)
    print(f"\n[Output] Plot saved to Global_Interpretation_Panel.png")

# ==============================================================================
# 通用打印与绘图函数
# ==============================================================================
def analyze_and_plot(ax, values, title, xlabel, print_label, color_theme='red', top_k=20):
    # 1. 数据整理
    df = pd.DataFrame({'Feature': FEATURE_NAMES, 'Value': values})
    df['Abs_Value'] = df['Value'].abs()
    
    # 2. 排序
    df_sorted = df.sort_values('Abs_Value', ascending=False).head(top_k)
    
    # 3. 【新增】控制台打印
    print(f"\n>>> Top 10 {print_label}:")
    print("-" * 50)
    # 格式化打印：索引 | 特征名 | 数值
    print(f"{'Rank':<4} | {'Feature Name':<30} | {'Value':<8}")
    print("-" * 50)
    for idx, (i, row) in enumerate(df_sorted.head(10).iterrows()):
        print(f"{idx+1:<4} | {row['Feature']:<30} | {row['Value']:.4f}")
    
    # 4. 绘图
    if color_theme == 'red':
        palette = ['#d62728' if x > 0 else '#1f77b4' for x in df_sorted['Value']]
    else:
        palette = ['#9467bd' if x > 0 else '#2ca02c' for x in df_sorted['Value']]

    sns.barplot(x='Value', y='Feature', data=df_sorted, palette=palette, hue='Feature', legend=False, ax=ax)
    
    ax.set_title(title, fontsize=14, weight='bold', pad=15)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel("")
    ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8, alpha=0.6)
    ax.grid(axis='x', linestyle='--', alpha=0.5)

if __name__ == "__main__":
    main()