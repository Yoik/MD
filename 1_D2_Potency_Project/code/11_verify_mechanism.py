#!/usr/bin/env python3
import matplotlib
matplotlib.use('Agg')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os
import sys
from scipy.stats import pearsonr

# 配置路径 (根据你的环境自动调整)
BASE_DIR = "."
FEATURE_DIR = os.path.join(BASE_DIR, "data", "features")
RESULT_CSV = os.path.join(FEATURE_DIR, "loocv_results.csv")

def verify_mechanism():
    print(f"--- Starting Mechanism Verification ---\n")
    
    # 1. 加载预测结果
    if not os.path.exists(RESULT_CSV):
        print(f"[Error] Results file not found: {RESULT_CSV}")
        print("Please run 2_train_model.py first.")
        return

    df_res = pd.read_csv(RESULT_CSV)
    # 创建字典方便查询: Compound -> Pred_Diff_Score
    pred_map = dict(zip(df_res['Compound'], df_res['Pred_Diff_Score']))
    
    # 2. 收集特征数据
    compound_stats = []
    
    # 遍历所有化合物目录
    compound_dirs = sorted(glob.glob(os.path.join(FEATURE_DIR, "*")))
    
    for c_dir in compound_dirs:
        if not os.path.isdir(c_dir): continue
        cid = os.path.basename(c_dir)
        
        # 排除非化合物目录
        if cid in ["1225", "plots", "csv", "__pycache__"]: continue
        
        # 寻找 .npy 文件
        npy_files = glob.glob(os.path.join(c_dir, "**", "*.npy"), recursive=True)
        if not npy_files: continue
        
        # 加载所有 replicate 的特征
        angles = []
        for f in npy_files:
            try:
                data = np.load(f)
                # === 关键点：假设 Orientation 是最后一列 ===
                # 检查维度
                if data.shape[1] < 62:
                    print(f"[Warn] {cid}: Feature dim {data.shape[1]} < 62. Orientation feature might be missing.")
                    continue
                
                # 取最后一列 (Lig_H6_Orientation)
                # 之前的脚本逻辑是 0-90度
                feat_vals = data[:, -1]
                angles.extend(feat_vals)
            except Exception as e:
                print(f"[Error] Loading {f}: {e}")
        
        if not angles: continue
        
        mean_angle = np.mean(angles)
        std_angle = np.std(angles)
        
        # 获取该分子的预测分
        # 注意: 你的文件夹名可能叫 '2025..._ARI_...'，但 csv 里可能只叫 'ARI'
        # 我们尝试模糊匹配
        pred_score = None
        short_name = None
        for name in pred_map.keys():
            if name in cid:
                pred_score = pred_map[name]
                short_name = name
                break
        
        if pred_score is not None:
            compound_stats.append({
                "Compound": short_name,
                "Full_Name": cid,
                "Mean_Angle": mean_angle,
                "Std_Angle": std_angle,
                "Pred_Score": pred_score
            })

    if not compound_stats:
        print("[Error] No valid feature data found. Please check your .npy files.")
        return

    df_stats = pd.DataFrame(compound_stats)
    
    # 3. 验证物理假设 (Physics Check)
    print(f"\n>>> 1. Geometric Hypothesis Check (Physics):")
    target_compounds = ["ARI", "Dopa", "ROT", "UNC", "S84"]
    
    for name in target_compounds:
        row = df_stats[df_stats["Compound"] == name]
        if not row.empty:
            angle = row.iloc[0]["Mean_Angle"]
            std = row.iloc[0]["Std_Angle"]
            print(f"   {name:<5} : Angle = {angle:.2f}° ± {std:.2f}")
            
            if name == "ARI":
                print(f"          -> Expect Low/Parallel (Target: ~0-30°)")
            if name == "Dopa":
                print(f"          -> Expect High/Perpendicular (Target: ~60-90°)")
    
    # 4. 验证模型机制 (Model Check)
    print(f"\n>>> 2. Model Mechanism Check (Correlation):")
    
    # 计算相关系数
    x = df_stats["Mean_Angle"]
    y = df_stats["Pred_Score"]
    corr, p_val = pearsonr(x, y)
    
    print(f"   Pearson Correlation (Angle vs Score): {corr:.3f} (p={p_val:.4f})")
    
    if corr > 0.3:
        print(f"   [CONCLUSION] ✅ The model learned: 'Perpendicular is BETTER'.")
        print(f"                (Larger angle -> Higher score)")
    elif corr < -0.3:
        print(f"   [CONCLUSION] ❌ The model learned: 'Parallel is BETTER'.")
        print(f"                (Smaller angle -> Higher score)")
    else:
        print(f"   [CONCLUSION] ❓ No linear relationship found. Relationship might be complex.")

    # 5. 画图
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df_stats, x="Mean_Angle", y="Pred_Score", s=100, hue="Compound", palette="tab10")
    
    # 拟合线
    sns.regplot(data=df_stats, x="Mean_Angle", y="Pred_Score", scatter=False, color="gray", line_kws={"linestyle": "--"})
    
    # 标注特殊点
    for i, row in df_stats.iterrows():
        plt.text(row["Mean_Angle"]+1, row["Pred_Score"]+0.01, row["Compound"], fontsize=9)

    plt.title(f"Model Mechanism: Orientation vs Predicted Score\nCorr = {corr:.2f} (Positive = Vertical is Better)")
    plt.xlabel("Ligand-H6 Orientation Angle (Degrees)\n0=Parallel, 90=Perpendicular")
    plt.ylabel("Model Predicted Score (Pred_Diff_Score)\nHigher is Better")
    plt.grid(True, linestyle='--', alpha=0.5)
    
    out_file = "mechanism_verification.png"
    plt.savefig(out_file, dpi=300)
    print(f"\n[Done] Plot saved to: {out_file}")
    print("Please check the plot to see if ARI is at bottom-left and Dopa is at top-right.")

if __name__ == "__main__":
    verify_mechanism()