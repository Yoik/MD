import pandas as pd
import numpy as np
import glob
import os
import matplotlib.pyplot as plt
import seaborn as sns

# ==============================================================================
# 1. 配置
# ==============================================================================
DATA_DIR = "."
LABEL_FILE = "labels.csv"

# 这是一个极其关键的特征组合
# 114 (D3.32): 负责结合 (Binding Anchor)
# 412 (W7.40): 负责激活 (Activation Switch / NPxxY region)
X_FEATURE = "Dist_Res_114"
Y_FEATURE = "Dist_Res_412"

# 辅助特征 (T-stacking)
Z_FEATURE = "Dist_Phe389_Mean"

def load_data():
    if not os.path.exists(LABEL_FILE): return None
    labels_df = pd.read_csv(LABEL_FILE)
    # 将 labels 转为字典方便查找: "Dopa" -> 100.0
    label_map = dict(zip(labels_df['Compound'], labels_df['Efficacy']))
    
    files = glob.glob(os.path.join(DATA_DIR, "*_All_Stats.csv"))
    if not files:
        print("[Error] No *_All_Stats.csv files found."); return None

    data_list = []
    
    for f in files:
        df = pd.read_csv(f)
        # 只取 AVERAGE 行
        avg_row = df[df['Replica'] == 'AVERAGE'].copy()
        if avg_row.empty: continue
        
        # 匹配效能
        fname = os.path.basename(f)
        found_match = False
        for cpd_name, eff in label_map.items():
            if str(cpd_name) in fname:
                avg_row['Efficacy'] = eff
                avg_row['Compound'] = cpd_name # 规范化名字
                found_match = True
                break
        
        if found_match:
            data_list.append(avg_row)
            
    if not data_list: return None
    return pd.concat(data_list, ignore_index=True)

def analyze_tstack_quality(df):
    # 1. 准备 ELF 特征 (取 C1-C6 的平均值作为整个环的电子密度指标)
    elf_cols = [f"C{i}_Weight" for i in range(1, 7)]
    # 检查列是否存在
    valid_elf_cols = [c for c in elf_cols if c in df.columns]
    if valid_elf_cols:
        df['Ring_ELF_Mean'] = df[valid_elf_cols].mean(axis=1)
    else:
        df['Ring_ELF_Mean'] = 0.0
        print("[Warn] No ELF weight columns found!")

    # 2. 准备几何特征
    # 我们关注 Phe389 (6.51)
    # 距离: Dist_Phe389_Mean
    # 角度偏差: AngDev_Phe389_Mean (越小越接近垂直/T型)
    
    print("\n" + "="*80)
    print(f"{'Compound':<10} | {'Eff':<6} | {'Dist-389':<8} | {'AngDev':<8} | {'ELF-Ring':<8}")
    print("-" * 80)
    
    df_sort = df.sort_values(by='Efficacy', ascending=False)
    for _, row in df_sort.iterrows():
        print(f"{row['Compound']:<10} | {row['Efficacy']:<6.1f} | {row['Dist_Phe389_Mean']:<8.2f} | {row['AngDev_Phe389_Mean']:<8.1f} | {row['Ring_ELF_Mean']:<8.3f}")

    # ==========================================================================
    # 可视化 1: 几何甜点区 (Geometry Sweet Spot)
    # X=距离, Y=角度偏差, Color=效能
    # ==========================================================================
    plt.figure(figsize=(10, 8))
    sns.set(style="whitegrid")
    
    # 绘制散点
    sns.scatterplot(
        data=df, 
        x='Dist_Phe389_Mean', 
        y='AngDev_Phe389_Mean', 
        hue='Efficacy', 
        size='Efficacy',
        sizes=(100, 600),
        palette='viridis', 
        edgecolor='black',
        alpha=0.9
    )
    
    # 标注名字
    for i in range(len(df)):
        row = df.iloc[i]
        plt.text(
            row['Dist_Phe389_Mean']+0.02, 
            row['AngDev_Phe389_Mean']+0.2, 
            row['Compound'], 
            fontsize=11, 
            weight='bold'
        )

    plt.title("T-Stacking Geometry Quality: F389 (6.51)", fontsize=16)
    plt.xlabel("Centroid Distance to F389 [Å]", fontsize=12)
    plt.ylabel("Angle Deviation from 90° [deg]", fontsize=12)
    
    # 反转Y轴，因为角度偏差越小越好（越接近T型）
    plt.gca().invert_yaxis()
    
    plt.savefig("tstack_geometry_f389.png", dpi=300)
    print("\nSaved tstack_geometry_f389.png")

    # ==========================================================================
    # 可视化 2: 电子效应 (ELF) vs 效能
    # ==========================================================================
    plt.figure(figsize=(8, 6))
    sns.regplot(data=df, x='Ring_ELF_Mean', y='Efficacy', color='purple', scatter_kws={'s': 100})
    
    for i in range(len(df)):
        row = df.iloc[i]
        plt.text(row['Ring_ELF_Mean'], row['Efficacy']+2, row['Compound'], fontsize=10)

    plt.title("Electronic Effect: Ring Electron Density vs Efficacy", fontsize=14)
    plt.xlabel("Mean ELF Integral on Ligand Ring (Normalized)", fontsize=12)
    plt.ylabel("Efficacy (%)", fontsize=12)
    
    plt.savefig("tstack_elf_correlation.png", dpi=300)
    print("Saved tstack_elf_correlation.png")

if __name__ == "__main__":
    df = load_data()
    if df is not None:
        analyze_tstack_quality(df)