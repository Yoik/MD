import torch
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # <--- 【关键修改1】防止在服务器上画图崩溃
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pickle
import glob
from tqdm import tqdm
from captum.attr import IntegratedGradients
from src.model import EfficiencyPredictor

# --- 配置 ---
DATA_DIR = "data/features"      # 数据目录
MODEL_PATH = "saved_models/best_model.pth"
SCALER_PATH = "saved_models/scaler.pkl"
POCKET_ATOM_NUM = 12

# 定义特征名称
FEATURE_NAMES = [f"Dist_{i+1}" for i in range(POCKET_ATOM_NUM)] + \
                ["Angle_Cos"] + \
                ["Phe389_Sum", "Phe389_Max", "Phe389_Conc"] + \
                ["Phe390_Sum", "Phe390_Max", "Phe390_Conc"] + \
                ["Dist_N_D114", "Dist_N_W386"]

def main():
    device = torch.device("cpu") # 解释性分析推荐用 CPU
    
    # 1. 加载模型和标准化器
    print("Loading model...")
    model = EfficiencyPredictor(input_dim=21).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    
    with open(SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)

    # 2. 扫描所有 .npy 文件
    search_path = os.path.join(DATA_DIR, "*", "*", "*_features.npy")
    files = glob.glob(search_path)
    print(f"Found {len(files)} trajectory files.")

    # 存储器
    all_feature_importances = [] # 存储每个文件的特征贡献 (IG)
    all_attention_corrs = []     # 存储每个文件的特征-注意力相关性

    # 3. 定义 Captum 需要的前向函数 (只返回预测值)
    def forward_func(inputs):
        macro_pred, _, _ = model(inputs)
        return macro_pred

    ig = IntegratedGradients(forward_func)

    # 4. 循环分析所有文件
    print("Running Global Interpretation...")
    for file_path in tqdm(files):
        try:
            # --- A. 数据预处理 ---
            raw_data = np.load(file_path)
            data_proc = raw_data.copy()
            data_proc[:, :POCKET_ATOM_NUM] = 1.0 / (data_proc[:, :POCKET_ATOM_NUM] + 1e-6)
            data_proc = scaler.transform(data_proc)
            
            input_tensor = torch.from_numpy(data_proc).float().unsqueeze(0).to(device) # [1, Frames, 19]
            input_tensor.requires_grad = True

            # --- B. 问题1: 哪些特征决定了最终效能? (Integrated Gradients) ---
            # 计算归因
            attributions = ig.attribute(input_tensor, n_steps=20) # n_steps=20 以加快速度
            # 在时间轴(Frames)上求和，得到该轨迹整体的特征贡献
            traj_importance = attributions.sum(dim=1).squeeze(0).detach().numpy()
            all_feature_importances.append(traj_importance)

            # --- C. 问题2: 哪些特征决定了注意力权重? (Correlation Analysis) ---
            # 获取模型对每一帧的注意力权重
            with torch.no_grad():
                _, _, attn_weights = model(input_tensor) 
                # attn_weights shape: [1, Frames, 1]
            
            weights = attn_weights.squeeze().numpy() # [Frames]
            features = data_proc # [Frames, 19]
            
            # 计算每一列特征与注意力权重的皮尔逊相关系数
            # 如果某特征越大，注意力越高，相关性为正
            corrs = []
            for i in range(21):
                if np.std(features[:, i]) < 1e-6: # 防止方差为0除以0
                    corrs.append(0)
                else:
                    corr = np.corrcoef(features[:, i], weights)[0, 1]
                    corrs.append(corr)
            all_attention_corrs.append(corrs)

        except Exception as e:
            print(f"Error processing {os.path.basename(file_path)}: {e}")

    # 5. 聚合结果
    # [N_files, 19] -> 平均 -> [19]
    avg_importances = np.mean(all_feature_importances, axis=0)
    avg_attn_drivers = np.mean(all_attention_corrs, axis=0)

    # 6. 绘图 1: 全局特征重要性 (Global Feature Importance)
    plot_bar_chart(
        values=avg_importances, 
        title="Global Feature Importance (What determines Efficacy?)",
        ylabel="Contribution to Efficacy (Integrated Gradients)",
        filename="Global_Feature_Importance.png",
        color_logic=True
    )

    # 7. 绘图 2: 注意力驱动因子 (Attention Drivers)
    plot_bar_chart(
        values=avg_attn_drivers, 
        title="Attention Drivers (What makes a frame 'Important'?)",
        ylabel="Correlation with Attention Weight",
        filename="Global_Attention_Drivers.png",
        color_logic=False # 这里不需要红蓝反转逻辑，直接看相关性
    )

def plot_bar_chart(values, title, ylabel, filename, color_logic=False):
    plt.figure(figsize=(14, 8))
    
    df = pd.DataFrame({'Feature': FEATURE_NAMES, 'Value': values})
    
    # 排序
    df['Abs_Value'] = df['Value'].abs()
    df = df.sort_values('Abs_Value', ascending=False)
    
    # 颜色
    if color_logic:
        # 红色=正贡献(增效)，蓝色=负贡献(减效)
        colors = ['red' if x > 0 else 'blue' for x in df['Value']]
    else:
        # 紫色=正相关(特征值大引关注)，绿色=负相关(特征值小引关注)
        colors = ['purple' if x > 0 else 'green' for x in df['Value']]

    sns.barplot(x='Value', y='Feature', data=df, palette=colors)
    
    plt.title(title, fontsize=16)
    plt.xlabel(ylabel, fontsize=12)
    plt.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
    plt.grid(axis='x', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    print(f"Saved plot: {filename}")

if __name__ == "__main__":
    main()