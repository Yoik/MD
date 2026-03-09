import numpy as np
import pandas as pd
from src.dataset import prepare_data

def main():
    print(">>> Verifying the 'TM5 Side Anchor' Hypothesis (Res 193, 194, 197)...")
    
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
        print(f"[DATA ERROR] {e}"); return

    all_features = train_ds.features + test_ds.features
    all_ids = train_ds.ids + test_ds.ids
    
    # 2. 定义对比组
    # Group A: 强效能且可能利用 TM5 的分子
    group_high = ["UNC", "ROT", "Dopa", "R10"]
    # Group B: 弱效能或缺乏 TM5 锚点的分子
    group_low = ["ARI", "BRE", "S84"]
    
    data_map = {}
    for cmpd in group_high + group_low:
        indices = [i for i, x in enumerate(all_ids) if x == cmpd]
        if indices:
            feats = [all_features[i] for i in indices]
            data_map[cmpd] = np.mean(np.concatenate(feats, axis=0), axis=0)

    # 3. 聚焦 TM5 残基
    # OBP_RESIDUES_STD = [114, 115, 118, 119, 190, 193, 194, 197, ...]
    # Index 5 = 193
    # Index 6 = 194
    # Index 7 = 197
    tm5_indices = [5, 6, 7] 
    tm5_names = ["193", "194", "197"]
    
    print(f"\n{'Compound':<10} | {'Res 193 (Dist Z)':<18} | {'Res 194 (Dist Z)':<18} | {'Res 197 (Dist Z)':<18} | {'Avg TM5 Interaction'}")
    print("-" * 90)
    
    for cmpd in group_high + group_low:
        if cmpd not in data_map: continue
        
        vec = data_map[cmpd]
        
        # 我们寻找该分子与 TM5 最亲密的原子（Minimum Distance Z-Score）
        # 也就是说，只要有一个原子紧紧抓住了 194，就算抓住了
        
        scores = []
        best_atoms = []
        
        for i, res_idx in enumerate(tm5_indices):
            # 提取所有原子到该残基的距离 (0, 16, 32... + res_idx)
            atom_dists = []
            for a in range(9): # 假设前9个是重原子
                feat_idx = a * 16 + res_idx
                if feat_idx < len(vec):
                    atom_dists.append(vec[feat_idx])
            
            # 取最小值（最强接触）
            min_dist = min(atom_dists)
            scores.append(min_dist)
            
            # 记录是哪个原子抓的
            best_atom = np.argmin(atom_dists)
            best_atoms.append(f"A{best_atom}")

        avg_tm5 = np.mean(scores)
        
        # 格式化输出
        s1 = f"{scores[0]:6.2f} ({best_atoms[0]})"
        s2 = f"{scores[1]:6.2f} ({best_atoms[1]})"
        s3 = f"{scores[2]:6.2f} ({best_atoms[2]})"
        
        print(f"{cmpd:<10} | {s1:<18} | {s2:<18} | {s3:<18} | {avg_tm5:6.2f}")

    print("-" * 90)
    print("Interpretation:")
    print("  - Negative Z-Score (< -1.0) = STRONG Interaction (Close Distance)")
    print("  - Positive Z-Score (> 1.0)  = WEAK/NO Interaction")
    print("Hypothesis Check:")
    print("  - UNC/ROT/Dopa should have highly negative scores (grabbing TM5).")
    print("  - ARI/BRE should have positive scores (missing TM5 anchor).")

if __name__ == "__main__":
    main()