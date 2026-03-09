import pandas as pd
import numpy as np
import glob
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance
from sklearn.metrics import r2_score, mean_absolute_error

# ==============================================================================
# 1. 配置
# ==============================================================================
DATA_DIR = "."
LABEL_FILE = "labels.csv"
CORRELATION_THRESHOLD = 0.98 # 稍微放宽，保留更多细微差异

# 基础特征名 (脚本会自动寻找对应的 _SD 列)
BASE_FEATURES = [
    "Global_Angle", 
    "Dist_Phe389", "AngDev_Phe389", 
    "Dist_Phe390", "AngDev_Phe390",
    "Dist_Res_114", "Dist_Res_115", "Dist_Res_118", "Dist_Res_119",
    "Dist_Res_190", "Dist_Res_193", "Dist_Res_194", "Dist_Res_197",
    "Dist_Res_386", "Dist_Res_393", "Dist_Res_412", "Dist_Res_416",
    # Weight 本身就是积分值，波动性可能意义不大，但也加上试试
    "C1_Weight", "C2_Weight", "C3_Weight", "C4_Weight", "C5_Weight", "C6_Weight"
]

# ==============================================================================
# 2. 工具函数
# ==============================================================================
def load_data_with_dynamics():
    if not os.path.exists(LABEL_FILE): return None, None

    labels_df = pd.read_csv(LABEL_FILE)
    files = glob.glob(os.path.join(DATA_DIR, "*_All_TimeSeries.csv"))
    
    # 自动构建特征列表：Mean + SD
    # 你的CSV里可能有 "Dist_Phe389" (均值) 和 "Dist_Phe389_SD" (标准差)
    # 我们先读一个文件来看看列名
    if not files: return None, None
    sample_df = pd.read_csv(files[0])
    
    # 动态构建可用特征列表
    final_features = []
    for base in BASE_FEATURES:
        if base in sample_df.columns:
            final_features.append(base) # 静态 (均值)
        
        sd_col = f"{base}_SD"
        if sd_col in sample_df.columns:
            final_features.append(sd_col) # 动态 (波动)
            
    print(f"  [Features] Auto-detected {len(final_features)} features (Static + Dynamic)")

    all_data = []
    for f in files:
        fname = os.path.basename(f)
        efficacy = None
        cpd_name = None
        
        for _, row in labels_df.iterrows():
            if row['Compound'] in fname: 
                efficacy = row['Efficacy']
                cpd_name = row['Compound']
                break
        
        if efficacy is None: continue
            
        df = pd.read_csv(f)
        # 填充缺失
        for col in final_features:
            if col not in df.columns: df[col] = 0.0
            
        # 下采样
        df_sub = df[final_features].iloc[::10].copy()
        df_sub['Efficacy'] = efficacy
        df_sub['Compound'] = cpd_name
        
        all_data.append(df_sub)

    return pd.concat(all_data, ignore_index=True), final_features

def drop_correlated_features(df, features, threshold=0.95):
    df_feat = df[features]
    corr_matrix = df_feat.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    return [f for f in features if f not in to_drop]

# ==============================================================================
# 3. 训练与评估
# ==============================================================================
def train_loco(df, features):
    # 去重
    selected_features = drop_correlated_features(df, features, CORRELATION_THRESHOLD)
    
    compounds = df['Compound'].unique()
    results = []
    
    # 使用 RF (在V2中表现最好)
    rf = RandomForestRegressor(n_estimators=200, max_depth=15, random_state=42)
    
    print(f"\nStarting LOCO CV with Dynamic Features ({len(selected_features)} inputs)...")
    print(f"{'Test Compound':<15} | {'True':<6} | {'Pred':<6} | {'Error':<6}")
    print("-" * 45)

    importances_accum = np.zeros(len(selected_features))

    for test_cpd in compounds:
        train_df = df[df['Compound'] != test_cpd]
        test_df = df[df['Compound'] == test_cpd]
        
        X_train = train_df[selected_features].values
        y_train = train_df['Efficacy'].values
        
        X_test = test_df[selected_features].values
        y_test = test_df['Efficacy'].values
        true_val = y_test[0]

        rf.fit(X_train, y_train)
        preds = rf.predict(X_test)
        score = np.mean(preds)
        
        print(f"{test_cpd:<15} | {true_val:<6.1f} | {score:<6.1f} | {abs(score-true_val):<6.1f}")
        
        results.append({
            "Compound": test_cpd, "True": true_val, "Pred": score, "Error": abs(score-true_val)
        })
        importances_accum += rf.feature_importances_

    # ==========================================================================
    # 结果分析
    # ==========================================================================
    res_df = pd.DataFrame(results)
    mae = mean_absolute_error(res_df['True'], res_df['Pred'])
    print(f"\nOverall MAE (Dynamic RF): {mae:.2f}")

    # 绘图 - 特征重要性
    avg_imp = importances_accum / len(compounds)
    imp_df = pd.DataFrame({'Feature': selected_features, 'Importance': avg_imp})
    imp_df = imp_df.sort_values(by='Importance', ascending=False)
    
    # 区分静态和动态特征颜色
    imp_df['Type'] = imp_df['Feature'].apply(lambda x: 'Dynamic (SD)' if '_SD' in x else 'Static (Mean)')
    
    plt.figure(figsize=(10, 8))
    sns.barplot(data=imp_df.head(20), x='Importance', y='Feature', hue='Type', dodge=False)
    plt.title("Feature Importance: Static vs Dynamic")
    plt.tight_layout()
    plt.savefig('feature_importance_dynamic.png', dpi=300)
    print("Saved feature_importance_dynamic.png")

    print("\n[Top 5 Features]")
    for i in range(5):
        row = imp_df.iloc[i]
        print(f"  {i+1}. {row['Feature']} ({row['Type']})")

if __name__ == "__main__":
    df, feats = load_data_with_dynamics()
    if df is not None:
        train_loco(df, feats)