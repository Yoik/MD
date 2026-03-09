import numpy as np
import pandas as pd
# import torch  <-- 不需要 torch 了
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import mean_squared_error
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# 复用你的 Dataset 代码
from src.dataset import prepare_data, TrajectoryDataset

def main():
    print(">>> Running Sanity Check: Random Forest on Averaged Features...")
    
    # 1. 加载数据
    try:
        train_ds, test_ds = prepare_data(
            label_file="data/labels.csv", 
            result_dir="data/features", 
            pocket_atom_num=12, 
            save_scaler_path=None, # RF不需要归一化
            window_size=100, stride=20
        )
    except Exception as e:
        print(f"Data Load Error: {e}"); return

    # 手动合并列表
    all_features = train_ds.features + test_ds.features
    all_labels = train_ds.labels + test_ds.labels
    all_ids = train_ds.ids + test_ds.ids
    
    unique_compounds = sorted(list(set(all_ids)))
    
    X_flat = []
    y_flat = []
    names = []
    
    # 排除列表 (诊断性排除)
    blacklist = ["BRE", "UNC"]
    candidates = [c for c in unique_compounds if c not in blacklist]
    print(f"Candidates: {len(candidates)}")

    for cmpd in candidates:
        # 找到该化合物所有的 Slice 索引
        indices = [i for i, x in enumerate(all_ids) if x == cmpd]
        
        if not indices: continue

        # 取出该化合物所有 Slice 的特征 (List of numpy arrays)
        feats = [all_features[i] for i in indices] 
        
        # 拼接并取平均 -> 变成一个 [151] 的向量
        # 修正：使用 numpy 操作
        if len(feats) > 0:
            # feats[0] shape is likely (Frames, 151)
            # concatenate on axis 0 to merge all frames from all slices
            feats_concat = np.concatenate(feats, axis=0) # [Total_Frames, 151]
            feats_mean = np.mean(feats_concat, axis=0)   # [151]
            
            # Label
            label_val = all_labels[indices[0]]
            # 如果 label 是 tensor，转 numpy；如果是 float，直接用
            if hasattr(label_val, 'item'):
                label = label_val.item() * 100
            else:
                label = label_val * 100
            
            X_flat.append(feats_mean)
            y_flat.append(label)
            names.append(cmpd)
        
    X = np.array(X_flat)
    y = np.array(y_flat)
    
    # 3. LOO-CV with Random Forest
    loo = LeaveOneOut()
    preds = []
    trues = []
    
    print("\nStarting RF-LOO...")
    print(f"{'Compound':<10} | {'True':<6} | {'Pred':<6} | {'Diff':<6}")
    print("-" * 40)
    
    for train_ix, test_ix in loo.split(X):
        X_train, X_test = X[train_ix], X[test_ix]
        y_train, y_test = y[train_ix], y[test_ix]
        
        # 简单的随机森林
        rf = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=5)
        rf.fit(X_train, y_train)
        
        pred = rf.predict(X_test)[0]
        preds.append(pred)
        trues.append(y_test[0])
        
        name = names[test_ix[0]]
        diff = pred - y_test[0]
        print(f"{name:<10} | {y_test[0]:<6.2f} | {pred:<6.2f} | {diff:<6.2f}")

    # 4. 统计结果
    preds = np.array(preds)
    trues = np.array(trues)
    
    rmse = np.sqrt(mean_squared_error(trues, preds))
    r, p = stats.pearsonr(trues, preds)
    
    print("\n" + "="*40)
    print("Random Forest Sanity Check Results")
    print("="*40)
    print(f"RMSE : {rmse:.4f}")
    print(f"R    : {r:.4f}")
    
    # 画图
    plt.figure(figsize=(6,6))
    sns.regplot(x=trues, y=preds)
    for i, txt in enumerate(names):
        plt.text(trues[i]+1, preds[i]+1, txt)
    plt.xlabel('True Efficacy')
    plt.ylabel('RF Predicted')
    plt.title(f'Random Forest Baseline\nR={r:.3f}, RMSE={rmse:.3f}')
    plt.savefig('sanity_check_rf.png')
    print("Plot saved to sanity_check_rf.png")

if __name__ == "__main__":
    main()