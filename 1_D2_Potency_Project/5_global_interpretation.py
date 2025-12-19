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
    device = torch.device("cpu") # 解释性分析通常在 CPU 上做以免显存溢出
    
    print(f"Loading model from {MODEL_PATH} ...")
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file not found at {MODEL_PATH}")
        return

    # 1. 初始化模型
    # 注意：如果你刚刚修改了模型架构（比如加了 Conv1d），请确保这里初始化的结构
    # 与 saved_models/best_model_mccv.pth 中的权重匹配。
    model = EfficiencyPredictor(input_dim=19).to(device)
    
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    except Exception as e:
        print(f"\n[CRITICAL ERROR] Model architecture mismatch!\n{e}")
        print("提示: 如果你刚修改了 src/model.py，请先重新运行 2_train_model.py 生成新的权重文件。")
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

    # 4. 定义 Forward 函数 (适配 Captum)
    def forward_func(inputs):
        # 【修复 1】正确处理字典返回
        out_dict = model(inputs)
        return out_dict["pred"]  # 只返回用于求梯度的预测值

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
            
            # 转换为 Tensor [1, Frames, 19]
            input_tensor = torch.from_numpy(data_proc).float().unsqueeze(0).to(device)
            input_tensor.requires_grad = True

            # === A. IG 归因分析 ===
            attributions = ig.attribute(input_tensor, n_steps=10)
            
            # 聚合：对时间维度求和 [1, T, 19] -> [19]
            traj_importance = attributions.sum(dim=1).squeeze(0).detach().numpy()
            all_feature_importances.append(traj_importance)

            # === B. 提取机制变量 ===
            with torch.no_grad():
                # 【修复 2】正确解包字典
                out_dict = model(input_tensor)
                gate_vals = out_dict["gate"]    # [1, T, 1]
                attn_weights = out_dict["attn"] # [1, T, 1]
            
            gates = gate_vals.squeeze().numpy()
            attns = attn_weights.squeeze().numpy()
            features = data_proc
            
            # 维度保护 (防止单帧数据导致维度消失)
            if gates.ndim == 0: gates = np.expand_dims(gates, 0)
            if attns.ndim == 0: attns = np.expand_dims(attns, 0)
            
            # === C. 计算相关性 ===
            gate_c = []
            attn_c = []
            for i in range(19):
                feat_col = features[:, i]
                # 防止方差为0导致的除零错误
                if np.std(feat_col) < 1e-6 or np.std(gates) < 1e-6:
                    gate_c.append(0)
                else:
                    gate_c.append(np.corrcoef(feat_col, gates)[0, 1])

                if np.std(feat_col) < 1e-6 or np.std(attns) < 1e-6:
                    attn_c.append(0)
                else:
                    attn_c.append(np.corrcoef(feat_col, attns)[0, 1])
            
            all_gate_corrs.append(gate_c)
            all_attn_corrs.append(attn_c)

        except Exception as e:
            # 打开这个 print 可以看到具体的报错，调试时很有用
            # print(f"Warning processing {file_path}: {e}")
            continue

    if not all_feature_importances:
        print("No valid data processed. (Check if model load failed or data is empty)")
        return

    # === 5. 聚合数据 ===
    avg_importances = np.mean(all_feature_importances, axis=0)
    avg_gate_drivers = np.mean(all_gate_corrs, axis=0)
    avg_attn_drivers = np.mean(all_attn_corrs, axis=0)

    # === 6. 组合绘图 (1行3列) ===
    print("\nGenerating Combined Plot...")
    
    fig, axes = plt.subplots(1, 3, figsize=(24, 10))
    
    plot_on_axis(axes[0], avg_importances, 
                 title="A. Global Feature Importance\n(Determinants of Efficacy)", 
                 xlabel="Integrated Gradients (Contribution)", 
                 color_logic='contribution')

    plot_on_axis(axes[1], avg_gate_drivers, 
                 title="B. Gate Drivers\n(What opens the Electronic Gate?)", 
                 xlabel="Correlation with Gate Value", 
                 color_logic='correlation')

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
    df = pd.DataFrame({'Feature': FEATURE_NAMES, 'Value': values})
    df['Abs_Value'] = df['Value'].abs()
    df = df.sort_values('Abs_Value', ascending=False)
    
    if color_logic == 'contribution':
        palette = ['#d62728' if x > 0 else '#1f77b4' for x in df['Value']]
    else:
        palette = ['#9467bd' if x > 0 else '#2ca02c' for x in df['Value']]

    sns.barplot(x='Value', y='Feature', data=df, palette=palette, hue='Feature', legend=False, ax=ax)
    
    ax.set_title(title, fontsize=14, weight='bold', pad=15)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel("")
    ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8, alpha=0.6)
    ax.grid(axis='x', linestyle='--', alpha=0.5)
    ax.tick_params(axis='y', labelsize=10)
    ax.tick_params(axis='x', labelsize=10)

if __name__ == "__main__":
    main()