import numpy as np
import pandas as pd
from scipy import stats
from src.dataset import prepare_data

def main():
    print(">>> Hunting for the Golden Feature...")
    
    # 1. 加载数据
    try:
        train_ds, test_ds = prepare_data(
            label_file="data/labels.csv", 
            result_dir="data/features", 
            pocket_atom_num=12, 
            save_scaler_path=None, 
            window_size=100, stride=20
        )
    except Exception as e:
        print(e); return

    # 2. 准备数据
    all_features = train_ds.features + test_ds.features
    all_labels = train_ds.labels + test_ds.labels
    all_ids = train_ds.ids + test_ds.ids
    
    # 诊断：我们可以选择是否剔除 BRE 和 UNC
    # 建议先剔除，看看在“正常分子”中谁最有效
    blacklist = ["BRE", "UNC"] 
    
    data_map = {} # {compound_name: [list_of_vectors]}
    label_map = {}
    
    for i, cmpd in enumerate(all_ids):
        if cmpd in blacklist: continue
        
        if cmpd not in data_map:
            data_map[cmpd] = []
            l = all_labels[i]
            label_map[cmpd] = l.item() * 100 if hasattr(l, 'item') else l * 100
            
        data_map[cmpd].append(all_features[i])

    # 3. 计算每个化合物的特征均值
    # X: [N_compounds, 151]
    # y: [N_compounds]
    compounds = sorted(data_map.keys())
    print(f"Analyzing {len(compounds)} compounds: {compounds}")
    
    X_means = []
    y_vals = []
    
    for cmpd in compounds:
        # 拼接所有切片并取平均
        feats = np.concatenate(data_map[cmpd], axis=0)
        mean_feat = np.mean(feats, axis=0) # [151]
        X_means.append(mean_feat)
        y_vals.append(label_map[cmpd])
        
    X = np.array(X_means) # [N_cmpd, 151]
    y = np.array(y_vals)  # [N_cmpd]
    
    # 4. 暴力扫描：计算每一个特征与 y 的相关性
    n_features = X.shape[1]
    correlations = []
    
    OBP_RESIDUES = [114, 115, 118, 119, 190, 193, 194, 197, 386, 389, 390, 393, 412, 416]

    print(f"\nScanning {n_features} features for correlation with Efficacy...")
    print(f"{'Rank':<5} | {'FeatIdx':<8} | {'Pearson R':<10} | {'Description'}")
    print("-" * 65)
    
    for i in range(n_features):
        feat_col = X[:, i]
        # 计算 Pearson R
        r, p = stats.pearsonr(feat_col, y)
        
        # 生成描述
        desc = "Unknown"
        if i < 144:
            atom_idx = i // 16
            feat_type = i % 16
            if feat_type < 14:
                res_id = OBP_RESIDUES[feat_type]
                desc = f"LigAtom {atom_idx} - Res {res_id} Dist"
            elif feat_type == 14:
                desc = f"LigAtom {atom_idx} - Phe389 Score"
            elif feat_type == 15:
                desc = f"LigAtom {atom_idx} - Phe390 Score"
        elif i == 144:
            desc = "Global Geometry (Cos)"
        elif 145 <= i <= 147:
            desc = f"Global Phe389 (Type {i-145})"
        elif 148 <= i <= 150:
            desc = f"Global Phe390 (Type {i-148})"
            
        correlations.append((i, r, desc))
        
    # 5. 排序并展示
    # 按 R 的绝对值排序
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)
    
    for rank, (idx, r, desc) in enumerate(correlations[:20]):
        print(f"{rank+1:<5} | {idx:<8} | {r:10.4f} | {desc}")
        
    # 6. 看看电子特征排哪里了
    print("\n--- Electronic Features Rank ---")
    elec_feats = [c for c in correlations if "Score" in c[2] or "Global Phe" in c[2]]
    for i, (idx, r, desc) in enumerate(elec_feats[:5]):
        print(f"Top {i+1}: Idx {idx} | R = {r:.4f} | {desc}")

if __name__ == "__main__":
    main()