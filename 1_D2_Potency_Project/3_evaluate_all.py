import sys
import os
import glob
import torch
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # <--- 【关键修改1】防止在服务器上画图崩溃
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from src.model import EfficiencyPredictor
# --- 配置参数 ---
LABEL_FILE = "data/labels.csv"
RESULT_DIR = "data/features"
MODEL_PATH = "saved_models/best_model_mccv.pth"
SCALER_PATH = "saved_models/scaler.pkl"

# 物理参数 (必须与训练时一致)
POCKET_ATOM_NUM = 12
INPUT_DIM = 19

def load_and_predict_compound(compound_name, model, scaler, device):
    """
    寻找指定化合物的所有 npy 文件，进行预测并返回平均值
    """
    # 1. 模糊匹配文件夹
    compound_folder_pattern = os.path.join(RESULT_DIR, f"*{compound_name}*")
    matching_folders = glob.glob(compound_folder_pattern)
    
    if not matching_folders:
        return None, "Folder Not Found"
    
    # 2. 寻找所有副本的 .npy 文件
    files = []
    for folder in matching_folders:
        npy_search = os.path.join(folder, "*", "*_features.npy")
        files.extend(glob.glob(npy_search))
        
    if not files:
        return None, "No Feature Files"
    
    # 3. 逐个文件预测
    preds = []
    for f in files:
        try:
            # 加载
            raw_data = np.load(f)
            
            # 预处理 (倒数 + 标准化)
            data_proc = raw_data.copy()
            
            # 【关键修改】定义需要取倒数的列 (与 dataset.py 保持一致)
            dist_indices = list(range(12)) 
            
            # 对这些距离特征取倒数 (转化为亲和力强度)
            data_proc[:, dist_indices] = 1.0 / (data_proc[:, dist_indices] + 1e-6)
            
            # 标准化
            data_proc = scaler.transform(data_proc)
            
            # 转 Tensor
            input_tensor = torch.from_numpy(data_proc).float().unsqueeze(0).to(device)
                        
            # 推理
            with torch.no_grad():
                out_dict = model(input_tensor) # 1. 拿到字典
                macro_pred = out_dict["pred"]  # 2. 提取预测值
                
            # 还原到 0-100
            preds.append(macro_pred.item() * 100)
            
        except Exception as e:
            print(f"Error reading {f}: {e}")
            
    if not preds:
        return None, "Prediction Failed"
        
    # 返回所有副本预测的平均值
    return np.mean(preds), "Success"

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. 加载资源
    print("Loading model and scaler...")
    model = EfficiencyPredictor(INPUT_DIM).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    
    with open(SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)
        
    # 2. 读取标签表
    df = pd.read_csv(LABEL_FILE)
    print(f"Loaded labels for {len(df)} compounds.")
    
    results = []
    
    # 3. 循环评估
    print("\nEvaluating all compounds...")
    print(f"{'Compound':<15} | {'True':<8} | {'Pred':<8} | {'Diff':<8} | {'Status'}")
    print("-" * 60)
    
    for idx, row in df.iterrows():
        name = str(row['Compound']).strip()
        true_val = float(row['Efficacy'])
        
        pred_val, status = load_and_predict_compound(name, model, scaler, device)
        
        if pred_val is not None:
            diff = pred_val - true_val
            print(f"{name:<15} | {true_val:<8.2f} | {pred_val:<8.2f} | {diff:<+8.2f} | {status}")
            
            results.append({
                "Compound": name,
                "True_Efficacy": true_val,
                "Predicted_Efficacy": pred_val,
                "Difference": diff,
                "Abs_Error": abs(diff)
            })
        else:
            print(f"{name:<15} | {true_val:<8.2f} | {'N/A':<8} | {'N/A':<8} | {status}")

    # 4. 统计分析
    if not results:
        print("\nNo valid predictions generated. Check your data folders.")
        return

    res_df = pd.DataFrame(results)
    
    # 计算整体指标
    mse = np.mean(res_df['Difference'] ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(res_df['Abs_Error'])
    corr = res_df['True_Efficacy'].corr(res_df['Predicted_Efficacy'])
    
    print("\n" + "="*30)
    print("  Overall Evaluation Report")
    print("="*30)
    print(f"Compounds Evaluated: {len(res_df)} / {len(df)}")
    print(f"RMSE (Root Mean Sq Error): {rmse:.2f}")
    print(f"MAE  (Mean Abs Error):     {mae:.2f}")
    print(f"Pearson Correlation (R):   {corr:.4f}")
    
    # 5. 找出偏差最大的 Top 5
    print("\n>>> Top 5 Largest Deviations (Worst Predictions):")
    worst_5 = res_df.sort_values("Abs_Error", ascending=False).head(5)
    print(worst_5[['Compound', 'True_Efficacy', 'Predicted_Efficacy', 'Difference']].to_string(index=False))
    
    # 6. 保存详细结果
    res_df.to_csv("evaluation_report.csv", index=False)
    print("\nDetailed report saved to 'evaluation_report.csv'")
    
    # 7. 画图：真实值 vs 预测值
    plt.figure(figsize=(8, 8))
    sns.scatterplot(data=res_df, x="True_Efficacy", y="Predicted_Efficacy", s=100, color='blue')
    
    # 画对角线 (完美预测线)
    min_val = min(res_df["True_Efficacy"].min(), res_df["Predicted_Efficacy"].min()) - 5
    max_val = max(res_df["True_Efficacy"].max(), res_df["Predicted_Efficacy"].max()) + 5
    plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', label='Perfect Prediction')
    
    # 标注点名称
    for i, row in res_df.iterrows():
        plt.text(row['True_Efficacy']+1, row['Predicted_Efficacy'], row['Compound'], fontsize=9)
        
    plt.title(f"True vs Predicted Efficacy (R={corr:.2f})")
    plt.xlabel("Experimental Efficacy")
    plt.ylabel("Model Predicted Efficacy")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.savefig("evaluation_plot.png", dpi=300)
    print("Plot saved to 'evaluation_plot.png'")
    # plt.show()

if __name__ == "__main__":
    main()