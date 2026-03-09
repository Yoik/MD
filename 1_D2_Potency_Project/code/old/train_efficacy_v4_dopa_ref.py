import matplotlib
matplotlib.use('Agg') # 必须放在最前面

import pandas as pd
import numpy as np
import glob
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor

# ==============================================================================
# 配置
# ==============================================================================
DATA_DIR = "."
LABEL_FILE = "labels.csv"

# 我们要对比的特征对
COMPARE_FEATURES = [
    # 1. 389号残基：几何 vs 电子加权
    "Dist_Phe389_Geo", "Dist_Phe389_Weighted",
    # 2. 390号残基：几何 vs 电子加权
    "Dist_Phe390_Geo", "Dist_Phe390_Weighted",
    # 3. 参照组：TM7 激活开关
    "Dist_Res_412"
]

def load_and_calc_deviations():
    if not os.path.exists(LABEL_FILE): return None, None
    labels_df = pd.read_csv(LABEL_FILE)
    label_map = dict(zip(labels_df['Compound'], labels_df['Efficacy']))
    
    files = glob.glob(os.path.join(DATA_DIR, "*_All_Stats.csv"))
    rows = []
    
    for f in files:
        df = pd.read_csv(f)
        # 兼容列名：有些可能有 _Mean 后缀，有些没有
        # 我们把所有列名统一去掉 _Mean 后缀以便处理
        df.columns = [c.replace('_Mean', '') for c in df.columns]
        
        avg_row = df[df['Replica'] == 'AVERAGE'].copy()
        if avg_row.empty: continue
        
        fname = os.path.basename(f)
        for name, eff in label_map.items():
            if str(name) in fname:
                avg_row['Compound'] = name
                avg_row['Efficacy'] = eff
                rows.append(avg_row)
                break
                
    if not rows: return None, None
    df_raw = pd.concat(rows, ignore_index=True)
    
    # 检查 Dopa
    if 'Dopa' not in df_raw['Compound'].values:
        print("Error: Dopa not found.")
        return None, None
        
    dopa_row = df_raw[df_raw['Compound'] == 'Dopa'].iloc[0]
    
    # 计算偏差
    df_delta = pd.DataFrame()
    df_delta['Compound'] = df_raw['Compound']
    df_delta['Efficacy'] = df_raw['Efficacy']
    
    feat_names = []
    for feat in COMPARE_FEATURES:
        if feat in df_raw.columns:
            # 计算绝对偏差 |Val - Dopa|
            df_delta[feat] = np.abs(df_raw[feat] - dopa_row[feat])
            feat_names.append(feat)
        else:
            print(f"Warning: Feature {feat} not found in CSV.")
            
    return df_delta, feat_names

def analyze_importance(df, features):
    # 训练一个简单的 RF 来评估特征重要性
    # 这次我们使用全部数据训练（不留一），只为了看特征排名
    X = df[features].values
    y = df['Efficacy'].values
    
    rf = RandomForestRegressor(n_estimators=500, random_state=42)
    rf.fit(X, y)
    
    # 获取重要性
    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    print("\n" + "="*60)
    print("HYPOTHESIS TEST: Electronic-Weighted vs Geometric Distance")
    print("="*60)
    print(f"{'Rank':<5} | {'Feature Name':<30} | {'Importance':<10}")
    print("-" * 60)
    
    for i in range(len(features)):
        idx = indices[i]
        feat = features[idx]
        score = importances[idx]
        
        # 标记胜出者
        mark = ""
        if "Weighted" in feat: mark = "<-- ELECTRONIC"
        if "Geo" in feat: mark = "<-- GEOMETRIC"
        
        print(f"{i+1:<5} | {feat:<30} | {score:.4f}     {mark}")
        
    print("-" * 60)

    # 专门对比 389
    try:
        imp_geo = importances[features.index("Dist_Phe389_Geo")]
        imp_wei = importances[features.index("Dist_Phe389_Weighted")]
        
        print(f"\n[Direct Comparison: Phe389]")
        print(f"  Geometric Distance Importance: {imp_geo:.4f}")
        print(f"  Weighted  Distance Importance: {imp_wei:.4f}")
        
        if imp_wei > imp_geo:
            print("\n  *** CONCLUSION: SUCCESS ***")
            print("  The model relies MORE on the 'Electronic Weighted Distance'.")
            print("  This confirms that the positioning of the high-density electron cloud")
            print("  is more critical than the simple geometric centroid.")
        else:
            print("\n  *** CONCLUSION: NEUTRAL ***")
            print("  Geometric distance seems sufficient. The electronic shift might be subtle.")
            
    except ValueError:
        pass

    # 绘图
    plt.figure(figsize=(8, 6))
    sns.barplot(x=importances[indices], y=np.array(features)[indices], palette='viridis')
    plt.title("Feature Importance: Weighted vs Geometric")
    plt.xlabel("Random Forest Importance")
    plt.tight_layout()
    plt.savefig("weighted_validation.png", dpi=300)
    print("\nSaved plot to weighted_validation.png")

if __name__ == "__main__":
    df, feats = load_and_calc_deviations()
    if df is not None:
        analyze_importance(df, feats)