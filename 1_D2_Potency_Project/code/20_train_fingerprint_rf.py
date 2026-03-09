import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import mean_squared_error
from scipy import stats
from src.dataset import prepare_data

def get_vertical_features(vec):
    """
    指纹提取 V7: Vertical Logic (Correction based on User Feedback)
    
    Topology Update:
    - 390 is DEEP (Switch).
    - 393 is SHALLOW (Head Anchor for Dopa OH-group).
    - 193/194 are SIDE (TM5 Anchors for UNC/ROT).
    """
    
    # --- 1. 提取基础读数 (Min Z-Score) ---
    
    # TM5 Side Anchors (193 & 194)
    # UNC 可能抓 194，ROT 可能抓 193。我们取两者的最强者，代表"侧面锚定能力"
    vals_193 = [vec[a*16 + 5] for a in range(9) if (a*16+5) < len(vec)]
    min_193 = min(vals_193) if vals_193 else 0.0
    
    vals_194 = [vec[a*16 + 6] for a in range(9) if (a*16+6) < len(vec)]
    min_194 = min(vals_194) if vals_194 else 0.0
    
    side_anchor_strength = min(min_193, min_194) # Z-score 越负越强
    
    # TM6 Switch (390) - The Deep Toggle
    vals_390 = [vec[a*16 + 10] for a in range(9) if (a*16+10) < len(vec)]
    min_390 = min(vals_390) if vals_390 else 0.0
    
    # TM6 Shallow Anchor (393) - The Head Positioner (Critical for Dopa)
    vals_393 = [vec[a*16 + 11] for a in range(9) if (a*16+11) < len(vec)]
    min_393 = min(vals_393) if vals_393 else 0.0

    # --- 2. 构建物理逻辑特征 ---
    
    # Feature 1: Switch Quality (开关质量)
    # 惩罚 ARI 的死锁 (-1.2)，奖励 Dopa 的适度 (-0.4)
    # 使用绝对偏差
    optimal_switch = -0.4
    switch_deviation = abs(min_390 - optimal_switch)
    
    # Feature 2: Dopa Path (Vertical Integrity)
    # 只有当 390 (Deep) 和 393 (Shallow) 同时结合时，才算打通了垂直路径
    # 我们把这两个原始值都放进去，RF 会学会 "If 393 is Good AND Switch is Good -> High Score"
    
    # Feature 3: UNC Path (Side Leverage)
    # 如果 390 结合了，且 Side Anchor 极强，也算打通了路径
    
    # 最终特征向量：
    # [开关偏差, 浅层锚点(393), 侧面锚点(Max 193/194), 原始开关值]
    return np.array([switch_deviation, min_393, side_anchor_strength, min_390])

def main():
    print(">>> Training RF with 'Vertical Logic' (Corrected 393)...")
    
    try:
        train_ds, test_ds = prepare_data(
            label_file="data/labels.csv", 
            result_dir="data/features", 
            pocket_atom_num=12, 
            save_scaler_path=None, 
            window_size=100, stride=20
        )
    except Exception as e:
        print(f"[DATA ERROR] {e}"); return

    all_features = train_ds.features + test_ds.features
    all_labels = train_ds.labels + test_ds.labels
    all_ids = train_ds.ids + test_ds.ids
    
    X_list = []
    y_list = []
    id_list = []
    
    print("Extracting features: Switch_Dev, Shallow_393, Side_Max(193/194)...")
    for i, feats in enumerate(all_features):
        for frame_idx in range(feats.shape[0]):
            fp = get_vertical_features(feats[frame_idx])
            X_list.append(fp)
            l = all_labels[i]
            val = l.item() * 100 if hasattr(l, 'item') else l * 100
            y_list.append(val)
            id_list.append(all_ids[i])
            
    X = np.array(X_list)
    y = np.array(y_list)
    ids = np.array(id_list)
    
    feat_names = ["Switch_Dev", "Shallow_393", "Side_Max", "Raw_390"]
    
    # LOO-CV
    unique_compounds = sorted(list(set(ids)))
    loo_results = []
    
    # 增加树的深度，因为现在逻辑有点复杂（分支逻辑）
    rf_params = {'n_estimators': 300, 'max_depth': 6, 'min_samples_leaf': 3, 'random_state': 42, 'n_jobs': -1}
    
    for test_cmpd in unique_compounds:
        test_mask = (ids == test_cmpd)
        train_mask = ~test_mask
        
        X_train, y_train = X[train_mask], y[train_mask]
        X_test, y_test = X[test_mask], y[test_mask]
        
        rf = RandomForestRegressor(**rf_params)
        rf.fit(X_train, y_train)
        
        pred_mean = np.mean(rf.predict(X_test))
        true_val = y_test[0]
        
        loo_results.append({
            "Compound": test_cmpd,
            "True": true_val,
            "Pred": pred_mean
        })
        print(f"  -> {test_cmpd:<5}: True={true_val:.1f}, Pred={pred_mean:.1f}")

    # 结果
    df_res = pd.DataFrame(loo_results)
    r, p = stats.pearsonr(df_res["True"], df_res["Pred"])
    rmse = np.sqrt(mean_squared_error(df_res["True"], df_res["Pred"]))
    
    print("\n" + "="*40)
    print(f"Final Results (Vertical Logic)")
    print(f"Pearson R : {r:.4f}")
    print(f"RMSE      : {rmse:.4f}")
    print("="*40)
    print(df_res)
    
    # Feature Importance
    rf_final = RandomForestRegressor(**rf_params)
    rf_final.fit(X, y)
    importances = rf_final.feature_importances_
    
    print("\nFeature Importances:")
    sorted_idx = np.argsort(importances)[::-1]
    for idx in sorted_idx:
        print(f"  {feat_names[idx]:<15}: {importances[idx]:.4f}")

if __name__ == "__main__":
    main()