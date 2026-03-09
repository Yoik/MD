import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from src.dataset import prepare_data, TrajectoryDataset

def main():
    print(">>> Analyzing Feature Importance & Distributions...")
    
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

    all_features = train_ds.features + test_ds.features
    all_labels = train_ds.labels + test_ds.labels
    all_ids = train_ds.ids + test_ds.ids
    
    # 2. 整理为 (N_samples, 151) 的矩阵
    # 这里我们不再取平均，而是使用所有的 Frame，看看细节分布
    X_list = []
    y_list = []
    id_list = []
    
    for i, feat in enumerate(all_features):
        # feat: [T, 151]
        X_list.append(feat)
        # label: scalar -> repeat T times
        l = all_labels[i]
        if hasattr(l, 'item'): l = l.item() * 100
        else: l = l * 100
        y_list.extend([l] * feat.shape[0])
        id_list.extend([all_ids[i]] * feat.shape[0])
        
    X = np.concatenate(X_list, axis=0)
    y = np.array(y_list)
    ids = np.array(id_list)
    
    print(f"Total Data Points (Frames): {X.shape}")
    
    # 3. 训练 RF 获取特征重要性
    print("Training RF to find key features...")
    rf = RandomForestRegressor(n_estimators=50, max_depth=5, n_jobs=-1, random_state=42)
    rf.fit(X, y)
    
    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    # 打印前 10 个最重要的特征
    print("\nTop 10 Most Important Features:")
    print(f"{'Rank':<5} | {'Feat Index':<10} | {'Importance':<10} | {'Description'}")
    print("-" * 50)
    
    # 特征描述映射
    # 0-143: Atom Pairs (9 atoms * 16 feats) -> 实际上是 9个原子 x (14距离 + 2电子) ?
    # 你的 extract_features: 
    #   padded_frame: [MAX_ATOMS(9), N_ATOM_FEAT(14+2=16)] -> Flatten = 144
    #   global: [cos(1), 389(3), 390(3)] = 7
    #   Total = 151
    
    # 我们不仅要打印Index，还要尝试解释它
    # 假设 Input Dim = 151
    # 0-143: Atom Features. 
    #    - Index i corresponds to Atom floor(i/16)
    #    - Remainder i%16: 0-13 are distances to OBP residues, 14 is 389_score, 15 is 390_score
    # 144: Global Cos
    # 145-147: Global 389
    # 148-150: Global 390
    
    OBP_RESIDUES = [114, 115, 118, 119, 190, 193, 194, 197, 386, 389, 390, 393, 412, 416]
    
    top_features = []
    
    for f in range(10):
        idx = indices[f]
        score = importances[idx]
        
        desc = "Unknown"
        if idx < 144:
            atom_idx = idx // 16
            feat_type = idx % 16
            if feat_type < 14:
                res_id = OBP_RESIDUES[feat_type]
                desc = f"LigAtom {atom_idx} - Res {res_id} Dist"
            elif feat_type == 14:
                desc = f"LigAtom {atom_idx} - Phe389 Score"
            elif feat_type == 15:
                desc = f"LigAtom {atom_idx} - Phe390 Score"
        elif idx == 144:
            desc = "Global Geometry (Cos)"
        elif 145 <= idx <= 147:
            desc = f"Global Phe389 (Type {idx-145})"
        elif 148 <= idx <= 150:
            desc = f"Global Phe390 (Type {idx-148})"
            
        print(f"{f+1:<5} | {idx:<10} | {score:.4f}     | {desc}")
        top_features.append((idx, desc))

    # 4. 可视化：对比 Dopa (High) vs ARI (Low) 在第一重要特征上的分布
    # 看看它们是否重叠
    
    best_feat_idx = indices[0]
    best_feat_desc = top_features[0][1]
    
    print(f"\nVisualizing distribution for Top Feature: {best_feat_desc}")
    
    target_compounds = ["Dopa", "ARI", "ROT", "UNC"]
    plot_data = []
    
    for cmpd in target_compounds:
        mask = (ids == cmpd)
        if np.sum(mask) == 0: continue
        values = X[mask, best_feat_idx]
        for v in values:
            plot_data.append({"Compound": cmpd, "Value": v})
            
    df_plot = pd.DataFrame(plot_data)
    
    plt.figure(figsize=(10, 6))
    sns.kdeplot(data=df_plot, x="Value", hue="Compound", fill=True, common_norm=False, palette="Set1")
    plt.title(f"Feature Distribution: {best_feat_desc}\n(Does High Efficacy separate from Low Efficacy?)")
    plt.savefig("feature_dist_analysis.png")
    print("Plot saved to feature_dist_analysis.png")

if __name__ == "__main__":
    main()