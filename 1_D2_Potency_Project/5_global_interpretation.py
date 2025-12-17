import torch
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg') # 服务器端绘图
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pickle
import glob
from tqdm import tqdm

# 检查 Captum 库
try:
    from captum.attr import IntegratedGradients
except ImportError:
    print("[ERROR] 'captum' library not found. Please install it: pip install captum")
    exit()

from src.model import EfficiencyPredictor

# ==============================================================================
# 配置参数
# ==============================================================================
DATA_DIR = "data/features"
MODEL_PATH = "saved_models/best_model_mccv.pth" 
SCALER_PATH = "saved_models/scaler.pkl"

# 特征名称 (对应 19 维输入)
# 定义具体的残基映射
OBP_LABELS = [
    "V114 (Dist_1)", "D115 (Dist_2)", "M118 (Dist_3)", "P119 (Dist_4)", 
    "D190 (Dist_5)", "S193 (Dist_6)", "V194 (Dist_7)", "F197 (Dist_8)", 
    "H386 (Dist_9)", "H393 (Dist_10)", "W412 (Dist_11)", "Y416 (Dist_12)"
]

FEATURE_NAMES = OBP_LABELS + \
                ["Angle_Cos"] + \
                ["Phe389_Sum", "Phe389_Max", "Phe389_Conc"] + \
                ["Phe390_Sum", "Phe390_Max", "Phe390_Conc"]
# ==============================================================================
# 主程序
# ==============================================================================
def main():
    device = torch.device("cpu")
    
    print(f"Loading model from {MODEL_PATH} ...")
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file not found at {MODEL_PATH}")
        return

    # 1. 初始化模型
    model = EfficiencyPredictor(input_dim=19).to(device)
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    except Exception as e:
        print(f"\n[CRITICAL ERROR] Model architecture mismatch!\n{e}")
        return

    model.eval()
    
    # 2. 加载 Scaler
    if not os.path.exists(SCALER_PATH):
        print("Error: Scaler file not found.")
        return
        
    with open(SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)

    # 3. 寻找数据文件
    search_path = os.path.join(DATA_DIR, "*", "*", "*_features.npy")
    files = glob.glob(search_path)
    print(f"Found {len(files)} trajectory files for analysis.")
    
    if len(files) == 0:
        print("No feature files found.")
        return

    # 4. 定义 Forward 函数 (只返回预测值)
    def forward_func(inputs):
        macro_pred, _, _, _ = model(inputs)
        return macro_pred

    ig = IntegratedGradients(forward_func)

    # 存储结果容器
    all_feature_importances = [] 
    all_gate_corrs = []          
    all_attn_corrs = []          

    print("Running Global Interpretation...")
    for file_path in tqdm(files):
        try:
            # === 数据预处理 ===
            raw_data = np.load(file_path)
            if raw_data.shape[0] == 0: continue
            
            data_proc = raw_data.copy()
            data_proc[:, :12] = 1.0 / (data_proc[:, :12] + 1e-6) # 距离倒数
            data_proc = scaler.transform(data_proc)
            
            input_tensor = torch.from_numpy(data_proc).float().unsqueeze(0).to(device)
            input_tensor.requires_grad = True

            # === A. IG 归因分析 ===
            attributions = ig.attribute(input_tensor, n_steps=10)
            traj_importance = attributions.sum(dim=1).squeeze(0).detach().numpy()
            all_feature_importances.append(traj_importance)

            # === B. 提取机制变量 ===
            with torch.no_grad():
                _, _, gate_vals, attn_weights = model(input_tensor)
            
            gates = gate_vals.squeeze().numpy()
            attns = attn_weights.squeeze().numpy()
            features = data_proc
            
            # 维度保护
            if gates.ndim == 0: gates = np.expand_dims(gates, 0)
            if attns.ndim == 0: attns = np.expand_dims(attns, 0)
            
            # === C. 计算相关性 ===
            gate_c = []
            attn_c = []
            for i in range(19):
                feat_col = features[:, i]
                if np.std(feat_col) < 1e-6:
                    gate_c.append(0); attn_c.append(0)
                else:
                    gate_c.append(np.corrcoef(feat_col, gates)[0, 1])
                    attn_c.append(np.corrcoef(feat_col, attns)[0, 1])
            
            all_gate_corrs.append(gate_c)
            all_attn_corrs.append(attn_c)

        except Exception as e:
            # print(f"Warning: {e}")
            continue

    if not all_feature_importances:
        print("No valid data processed.")
        return

    # === 5. 聚合数据 ===
    avg_importances = np.mean(all_feature_importances, axis=0)
    avg_gate_drivers = np.mean(all_gate_corrs, axis=0)
    avg_attn_drivers = np.mean(all_attn_corrs, axis=0)

    # === 6. 组合绘图 (1行3列) ===
    print("\nGenerating Combined Plot...")
    
    # 创建一个宽幅画布: 24x10 英寸
    fig, axes = plt.subplots(1, 3, figsize=(24, 10))
    
    # 绘制子图 1: 全局重要性 (IG)
    plot_on_axis(axes[0], avg_importances, 
                 title="A. Global Feature Importance\n(Determinants of Efficacy)", 
                 xlabel="Integrated Gradients (Contribution)", 
                 color_logic='contribution')

    # 绘制子图 2: 门控驱动 (Gate)
    plot_on_axis(axes[1], avg_gate_drivers, 
                 title="B. Gate Drivers\n(What opens the Electronic Gate?)", 
                 xlabel="Correlation with Gate Value", 
                 color_logic='correlation')

    # 绘制子图 3: 注意力驱动 (Attention)
    plot_on_axis(axes[2], avg_attn_drivers, 
                 title="C. Attention Drivers\n(What attracts Temporal Attention?)", 
                 xlabel="Correlation with Attention Weight", 
                 color_logic='correlation')

    plt.tight_layout()
    output_file = "Global_Interpretation_Panel.png"
    plt.savefig(output_file, dpi=300)
    print(f"Saved combined plot: {output_file}")

# ==============================================================================
# 通用子图绘制函数
# ==============================================================================
def plot_on_axis(ax, values, title, xlabel, color_logic='correlation'):
    # 准备数据框并排序
    df = pd.DataFrame({'Feature': FEATURE_NAMES, 'Value': values})
    df['Abs_Value'] = df['Value'].abs()
    df = df.sort_values('Abs_Value', ascending=False)
    
    # 配色方案
    if color_logic == 'contribution':
        # 红蓝配色: 红=正贡献, 蓝=负贡献
        palette = ['#d62728' if x > 0 else '#1f77b4' for x in df['Value']]
    else:
        # 紫绿配色: 紫=正相关, 绿=负相关
        palette = ['#9467bd' if x > 0 else '#2ca02c' for x in df['Value']]

    # 绘图
    sns.barplot(x='Value', y='Feature', data=df, palette=palette, hue='Feature', legend=False, ax=ax)
    
    # 装饰
    ax.set_title(title, fontsize=14, weight='bold', pad=15)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel("") # 去掉 Y 轴标签，节省空间
    ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8, alpha=0.6)
    ax.grid(axis='x', linestyle='--', alpha=0.5)
    
    # 调整刻度字体
    ax.tick_params(axis='y', labelsize=10)
    ax.tick_params(axis='x', labelsize=10)

if __name__ == "__main__":
    main()