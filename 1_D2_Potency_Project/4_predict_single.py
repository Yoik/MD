import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from src.model import EfficiencyPredictor
import glob
import os
import pickle

# --- 配置 ---
# 修改为您想测试的化合物名称（需与文件夹匹配）
TEST_COMPOUND = "Dopa" 
RESULT_DIR = "data/features"
MODEL_PATH = "saved_models/best_model.pth"
SCALER_PATH = "saved_models/scaler.pkl"
INPUT_DIM = 19
POCKET_ATOM_NUM = 12

def main():
    device = torch.device("cpu")
    
    # 1. 加载模型
    model = EfficiencyPredictor(INPUT_DIM).to(device)
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    except FileNotFoundError:
        print(f"Error: Model file not found at {MODEL_PATH}")
        return
    model.eval()
    
    # 2. 加载标准化器
    try:
        with open(SCALER_PATH, 'rb') as f:
            scaler = pickle.load(f)
    except FileNotFoundError:
        print(f"Error: Scaler file not found at {SCALER_PATH}")
        return

    # 3. 寻找文件
    # 模糊匹配文件夹，再找 npy
    search_path = os.path.join(RESULT_DIR, f"*{TEST_COMPOUND}*", "*", "*_features.npy")
    files = glob.glob(search_path)
    
    if not files:
        print(f"No feature files found for compound: {TEST_COMPOUND}")
        return

    # 默认分析第一个找到的文件
    file_path = files[0]
    print(f"Analyzing: {file_path}")
    
    # 4. 数据预处理
    raw_data = np.load(file_path)
    data_proc = raw_data.copy()
    data_proc[:, :POCKET_ATOM_NUM] = 1.0 / (data_proc[:, :POCKET_ATOM_NUM] + 1e-6)
    data_proc = scaler.transform(data_proc)
    
    input_tensor = torch.from_numpy(data_proc).float().unsqueeze(0)
    
    # 5. 推理 (修复了这里的解包错误)
    with torch.no_grad():
        # 现在模型返回3个值: 预测值, 单帧分数, 注意力权重
        macro_pred, frame_scores, attn_weights = model(input_tensor)
        
    # 6. 结果处理
    predicted_efficacy = macro_pred.item() * 100 # 还原到 0-100
    scores = frame_scores.squeeze().numpy()
    attentions = attn_weights.squeeze().numpy()
    
    print(f"Model Predicted Efficacy: {predicted_efficacy:.2f}")

    # --- 保存预测结果 ---
    res_df = pd.DataFrame({
        "Compound": [TEST_COMPOUND],
        "File": [os.path.basename(file_path)],
        "Predicted_Efficacy": [predicted_efficacy]
    })
    res_csv = "prediction_results.csv"
    res_df.to_csv(res_csv, mode='a', header=not os.path.exists(res_csv), index=False)
    print(f"Result saved to {res_csv}")

    # --- 找出关键帧 (基于注意力权重) ---
    # 注意力越大的帧，说明模型认为它对最终效能贡献越大
    top_attn_indices = attentions.argsort()[-5:][::-1]
    
    print("\n" + "="*50)
    print("Top 5 Most Important Frames (Highest Attention):")
    for idx in top_attn_indices:
        print(f"Frame {idx:4d} | Score: {scores[idx]:.4f} | Attention: {attentions[idx]:.4f}")
    print("="*50 + "\n")

    # --- 画图 ---
    plt.figure(figsize=(10, 6))
    
    # 1. 直方图 (不变)
    sns.histplot(scores, kde=True, bins=50, color='dodgerblue', alpha=0.6, label='Frame Scores')
    
    # 2. 线条修正
    # A. 算术平均线 (旧的，代表如果不加 Attention 的结果)
    simple_mean = np.mean(scores)
    plt.axvline(x=simple_mean, color='green', linestyle=':', linewidth=2, 
                label=f'Simple Mean: {simple_mean:.3f}')
    
    # B. 模型预测线 (新的，代表 Attention 加权后的结果)
    # macro_pred 是模型直接输出的加权结果 (0-1)，需要 *100 还原吗？
    # 注意：scores 已经是 numpy array 了，而 macro_pred 是 Tensor
    weighted_pred = macro_pred.item() # 0-1 之间的值
    
    plt.axvline(x=weighted_pred, color='red', linestyle='--', linewidth=2, 
                label=f'Model Prediction (Weighted): {weighted_pred:.3f}')    
    plt.title(f"Microscopic Efficiency Distribution: {TEST_COMPOUND}\n(Predicted Macro Efficacy: {predicted_efficacy:.1f})")
    plt.xlabel("Frame Efficiency Score (0.0 - 1.0)")
    plt.ylabel("Frame Count")
    plt.legend()
    
    plot_name = f"{TEST_COMPOUND}_dist.png"
    plt.savefig(plot_name, dpi=300)
    print(f"Distribution plot saved to {plot_name}")
    # plt.show() # 如果在服务器无界面环境，可注释掉

if __name__ == "__main__":
    main()