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

# [FIX] 修正列名：OBP 残基在 Stats.csv 中没有 _Mean 后缀
X_FEATURE = "Dist_Res_114"   # Binding (TM3)
Y_FEATURE = "Dist_Res_412"   # Activation (TM7)
Z_FEATURE = "Dist_Phe389_Mean" # T-stacking (有 _Mean 后缀)

def load_data():
    if not os.path.exists(LABEL_FILE):
        print(f"[Error] {LABEL_FILE} not found."); return None
    
    labels_df = pd.read_csv(LABEL_FILE)
    # 将 labels 转为字典方便查找: "Dopa" -> 100.0
    label_map = dict(zip(labels_df['Compound'], labels_df['Efficacy']))
    
    files = glob.glob(os.path.join(DATA_DIR, "*_All_Stats.csv"))
    if not files:
        print("[Error] No *_All_Stats.csv files found."); return None

    data_list = []
    print(f"Loading {len(files)} summary files...")
    
    for f in files:
        df = pd.read_csv(f)
        # 只取 AVERAGE 行
        avg_row = df[df['Replica'] == 'AVERAGE'].copy()
        if avg_row.empty: continue
        
        # 匹配效能
        # 逻辑：文件名包含 label_map 中的 key
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
        else:
            print(f"  [Warn] No label match for {fname}")
            
    if not data_list: return None
    return pd.concat(data_list, ignore_index=True)

def plot_landscape(df):
    # 检查列是否存在
    print("\n[Data Columns Check]")
    missing = []
    for col in [X_FEATURE, Y_FEATURE, Z_FEATURE]:
        if col not in df.columns:
            missing.append(col)
        else:
            print(f"  OK: {col} (Range: {df[col].min():.2f} - {df[col].max():.2f})")
            
    if missing:
        print(f"\n[Error] Missing columns: {missing}")
        print("Available columns:", df.columns.tolist())
        return

    # 设置绘图风格
    sns.set(style="whitegrid")
    
    # ==========================================================================
    # 图 1: Binding (114) vs Activation (412)
    # ==========================================================================
    plt.figure(figsize=(10, 8))
    
    # 绘制散点
    scatter = sns.scatterplot(
        data=df, 
        x=X_FEATURE, 
        y=Y_FEATURE, 
        hue='Efficacy', 
        size='Efficacy',
        sizes=(100, 600),
        palette='coolwarm_r', # 蓝(低) -> 红(高)
        edgecolor='black',
        alpha=0.8
    )
    
    # 标注名字
    for i in range(len(df)):
        row = df.iloc[i]
        plt.text(
            row[X_FEATURE]+0.05, 
            row[Y_FEATURE]+0.05, 
            row['Compound'], 
            fontsize=12, 
            weight='bold',
            color='black'
        )
        
    # 添加辅助线 (均值)
    plt.axvline(df[X_FEATURE].mean(), color='gray', linestyle='--', alpha=0.5)
    plt.axhline(df[Y_FEATURE].mean(), color='gray', linestyle='--', alpha=0.5)
    
    # 区域标注 (可选)
    plt.xlabel(f"Binding Anchor (TM3 D114) Distance [Å]", fontsize=12)
    plt.ylabel(f"Activation Switch (TM7 W412) Distance [Å]", fontsize=12)
    plt.title(f"Efficacy Landscape: Binding vs Activation", fontsize=16)
    
    # 保存
    out_file = "landscape_binding_vs_activation.png"
    plt.savefig(out_file, dpi=300, bbox_inches='tight')
    print(f"\n[Success] Plot saved to {out_file}")
    
    # ==========================================================================
    # 数值分析输出
    # ==========================================================================
    print("\n[Detailed Feature Analysis]")
    # 格式化打印
    header = f"{'Compound':<10} | {'Eff':<6} | {'D114':<8} | {'W412':<8} | {'F389':<8}"
    print("-" * len(header))
    print(header)
    print("-" * len(header))
    
    # 按效能降序
    df_sort = df.sort_values(by='Efficacy', ascending=False)
    for _, row in df_sort.iterrows():
        print(f"{row['Compound']:<10} | {row['Efficacy']:<6.1f} | {row[X_FEATURE]:<8.2f} | {row[Y_FEATURE]:<8.2f} | {row[Z_FEATURE]:<8.2f}")
        
    # 自动结论
    try:
        dopa = df[df['Compound']=='Dopa'].iloc[0]
        ari = df[df['Compound']=='ARI'].iloc[0]
        
        print("\n[Mechanism Hypothesis Check: Dopa vs ARI]")
        print(f"  Target: Explain why ARI is partial agonist despite high affinity.")
        
        d_bind = abs(dopa[X_FEATURE] - ari[X_FEATURE])
        d_act = abs(dopa[Y_FEATURE] - ari[Y_FEATURE])
        
        print(f"  1. Binding Site (D114) Diff:    {d_bind:.2f} Å")
        print(f"  2. Activation Site (W412) Diff: {d_act:.2f} Å")
        
        if d_act > d_bind:
            print("\n  *** CONCLUSION: SUPPORTED ***")
            print("  ARI binds similarly to Dopa (small D114 diff), but fails to engage the")
            print("  TM7 activation switch W412 (large diff). This structural uncoupling")
            print("  is the likely cause of its low efficacy.")
        else:
            print("\n  *** CONCLUSION: UNCLEAR ***")
            print("  The difference is not primarily in TM7. Check other switches (e.g. TM6/TM5).")
            
    except IndexError:
        pass

if __name__ == "__main__":
    df = load_data()
    if df is not None:
        plot_landscape(df)