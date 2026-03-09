import numpy as np
import torch
from src.dataset import prepare_data, TrajectoryDataset

def main():
    print(">>> diagnosing UNC Atom Mapping...")
    
    # 1. 加载数据 (不做任何 Masking，看原始数据)
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

    # 合并
    all_features = train_ds.features + test_ds.features
    all_ids = train_ds.ids + test_ds.ids
    
    # 2. 找到 UNC 的数据
    target_compound = "Dopa"
    
    indices = [i for i, x in enumerate(all_ids) if x == target_compound]
    
    if not indices:
        print(f"Error: {target_compound} not found in dataset!")
        return
        
    print(f"Found {len(indices)} slices for {target_compound}")
    
    # 取第一个 slice 来分析 (通常这就够了，因为 Mapping 在整个轨迹中通常是一致的)
    # feat shape: [Frames, 151]
    # 我们把所有 slice 拼起来看平均值
    feats_list = [all_features[i] for i in indices]
    feats_concat = np.concatenate(feats_list, axis=0) # [Total_Frames, 151]
    
    print(f"Total Frames analyzing: {feats_concat.shape[0]}")
    
    # 3. 逐个原子检查
    # 特征结构: 9个原子，每个原子16维
    # [Atom 0 (0-15)], [Atom 1 (16-31)], ...
    
    N_ATOMS = 6
    FEAT_PER_ATOM = 16
    
    print("\n" + "="*60)
    print(f"{'Atom Idx':<10} | {'Status':<10} | {'Avg Dist to Res118':<20} | {'Avg Phe390 Score':<20}")
    print("-" * 60)
    
    active_atoms = 0
    
    for i in range(N_ATOMS):
        start = i * FEAT_PER_ATOM
        end = start + FEAT_PER_ATOM
        
        # 取出该原子的所有特征列
        atom_feats = feats_concat[:, start:end]
        
        # 检查1：距离特征 (前14列)
        # 在 extract_features.py 中，如果原子没匹配到，填充值通常是 100.0
        # Res 118 的距离是第 10 列 (Residue 118 index is 2 in OBP list... wait, let's just check mean)
        # 我们直接看所有距离的均值。如果均值接近 100，说明是 Padding
        dist_mean = np.mean(atom_feats[:, :14])
        
        # 检查2：电子特征 (最后2列)
        # 如果没匹配到，填充值是 0.0
        score_390_mean = np.mean(atom_feats[:, 15]) # Index 15 is 390 score
        
        # 判断状态
        is_missing = (dist_mean > 90.0) and (score_390_mean == 0.0)
        
        status = "MISSING" if is_missing else "ACTIVE"
        if not is_missing: active_atoms += 1
        
        print(f"{i:<10} | {status:<10} | {dist_mean:<20.4f} | {score_390_mean:<20.4f}")
        
    print("-" * 60)
    print(f"Summary: Found {active_atoms} active atoms out of {N_ATOMS}")
    
    if active_atoms < 9:
        print(f"\n[CRITICAL WARNING] {target_compound} has missing atoms!")
        print("This explains why the prediction is low. The model sees 'empty air' where atoms should be.")
    else:
        print(f"\n[INFO] {target_compound} looks structurally complete.")

if __name__ == "__main__":
    main()