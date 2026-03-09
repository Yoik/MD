import pandas as pd
import numpy as np
import glob
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import euclidean
import seaborn as sns
from math import pi

# ==============================================================================
# 1. 配置
# ==============================================================================
DATA_DIR = "."
LABEL_FILE = "labels.csv"

# 物理指纹特征 (13个关键指标)
FINGERPRINT_COLS = [
    "Global_Angle_Mean", 
    "Dist_Phe389_Mean", "AngDev_Phe389_Mean",
    "Dist_Phe390_Mean", "AngDev_Phe390_Mean",
    "Dist_Res_114_Mean", "Dist_Res_115_Mean", "Dist_Res_118_Mean",
    "Dist_Res_190_Mean", "Dist_Res_193_Mean", 
    "Dist_Res_386_Mean", "Dist_Res_412_Mean", "Dist_Res_416_Mean",
    "C1_Weight", "C4_Weight"
]

# ==============================================================================
# 2. 数据加载
# ==============================================================================
def load_summary_stats():
    if not os.path.exists(LABEL_FILE):
        print(f"[Error] {LABEL_FILE} not found."); return None

    labels_df = pd.read_csv(LABEL_FILE)
    files = glob.glob(os.path.join(DATA_DIR, "*_All_Stats.csv"))
    
    data_list = []
    print("Loading summary stats...")
    
    for f in files:
        fname = os.path.basename(f)
        df = pd.read_csv(f)
        avg_row = df[df['Replica'] == 'AVERAGE'].copy()
        if avg_row.empty: continue
        
        efficacy = None
        cpd_name = None
        for _, row in labels_df.iterrows():
            if str(row['Compound']) in fname: 
                efficacy = row['Efficacy']
                cpd_name = row['Compound']
                break
        
        if efficacy is not None:
            avg_row['Efficacy'] = efficacy
            avg_row['Compound_Name'] = cpd_name
            data_list.append(avg_row)
            
    if not data_list: return None
    return pd.concat(data_list, ignore_index=True)

# ==============================================================================
# 3. 核心分析
# ==============================================================================
def analyze_similarity(df):
    # --- 准备数据 ---
    for col in FINGERPRINT_COLS:
        if col not in df.columns: df[col] = 0.0
            
    X = df[FINGERPRINT_COLS].values
    compounds = df['Compound_Name'].values
    
    # --- 标准化 (Z-Score) ---
    # 必须标准化，否则距离会被大数值特征(如角度)主导
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # --- 找到 Dopa ---
    dopa_idx = -1
    for i, name in enumerate(compounds):
        if "Dopa" in name: dopa_idx = i; break
    
    if dopa_idx == -1: print("[Error] Dopa not found!"); return

    dopa_vec = X_scaled[dopa_idx]
    
    # --- 计算距离 ---
    distances = []
    for i in range(len(X_scaled)):
        d = euclidean(X_scaled[i], dopa_vec)
        distances.append(d)
    
    df['Dist_to_Dopa'] = distances
    
    # ==========================================================================
    # 打印输出 (The Print Output)
    # ==========================================================================
    print("\n" + "="*80)
    print(f"{'Compound':<15} | {'Efficacy':<10} | {'Dist to Dopa':<15} | {'Interpretation'}")
    print("-" * 80)
    
    # 按距离排序
    df_sorted = df.sort_values(by='Dist_to_Dopa')
    
    for _, row in df_sorted.iterrows():
        dist = row['Dist_to_Dopa']
        eff = row['Efficacy']
        
        # 简单解读
        interp = ""
        if "Dopa" in row['Compound_Name']: interp = "REFERENCE"
        elif dist < 3.0: interp = "High Similarity (Active-like)"
        elif dist > 5.0: interp = "High Deviation (Inactive-like)"
        else: interp = "Intermediate"
        
        print(f"{row['Compound_Name']:<15} | {eff:<10.2f} | {dist:<15.4f} | {interp}")
        
    print("-" * 80)
    
    # 计算相关性
    corr = df['Dist_to_Dopa'].corr(df['Efficacy'])
    print(f"\n[Hypothesis Check] Correlation (Distance vs Efficacy): {corr:.4f}")
    if corr < -0.7:
        print("  -> STRONG NEGATIVE CORRELATION detected!")
        print("  -> Conclusion: 'Looking like Dopamine' strongly predicts High Efficacy.")
    else:
        print("  -> Correlation is weak. The relationship might be non-linear.")

    # ==========================================================================
    # 偏差分析 (Why is the worst compound bad?)
    # ==========================================================================
    worst_compound = df_sorted.iloc[-1] # 距离最远的那个
    worst_idx = df_sorted.index[-1] # 原始索引
    
    # 找出原始矩阵中对应的行 (标准化后的)
    # df_sorted 已经乱序，需要用 Compound Name 找回 X_scaled 的索引
    w_name = worst_compound['Compound_Name']
    
    # 找到该化合物在 X_scaled 中的位置
    w_vec_idx = np.where(compounds == w_name)[0][0]
    w_vec = X_scaled[w_vec_idx]
    
    # 计算每个特征的绝对偏差 |Compound - Dopa|
    diffs = np.abs(w_vec - dopa_vec)
    
    # 排序偏差
    sorted_diff_indices = np.argsort(diffs)[::-1] # 降序
    
    print(f"\n[Deviation Analysis] Why is {w_name} so different from Dopa?")
    print(f"Top 5 Features deviating from Dopamine's conformation:")
    for i in range(5):
        feat_idx = sorted_diff_indices[i]
        feat_name = FINGERPRINT_COLS[feat_idx]
        deviation_score = diffs[feat_idx]
        
        # 判断是偏大还是偏小
        raw_val_cpd = df[df['Compound_Name']==w_name][feat_name].values[0]
        raw_val_dopa = df[df['Compound_Name']=='Dopa'][feat_name].values[0]
        direction = "LARGER" if raw_val_cpd > raw_val_dopa else "SMALLER"
        
        print(f"  {i+1}. {feat_name:<20} (Dev: {deviation_score:.2f} sigma) -> {direction} than Dopa")

    # ==========================================================================
    # 画图 (保留)
    # ==========================================================================
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df, x='Dist_to_Dopa', y='Efficacy', hue='Compound_Name', s=150)
    sns.regplot(data=df, x='Dist_to_Dopa', y='Efficacy', scatter=False, color='gray', line_kws={'linestyle':'--'})
    plt.title(f"Dopa Similarity Hypothesis (Corr: {corr:.2f})")
    plt.savefig("dopa_similarity_v2.png", dpi=300)
    print("\nSaved plot to dopa_similarity_v2.png")

if __name__ == "__main__":
    df = load_summary_stats()
    if df is not None:
        analyze_similarity(df)