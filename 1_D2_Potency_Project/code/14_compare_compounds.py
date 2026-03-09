import numpy as np
import pandas as pd
from src.dataset import prepare_data

def main():
    print(">>> Comparative Analysis: Finding the Missing Link...")
    
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

    # 2. 提取特定化合物的特征均值
    target_compounds = ["ROT", "BRE", "UNC", "S84"]
    # ROT: 标杆 (High)
    # BRE: 迷惑项 (Low, but looks High to Phe390)
    # UNC: 迷惑项 (High, but looks Low to Phe390)
    # S84: 标杆 (Low)
    
    data_map = {}
    
    # 合并数据
    all_features = train_ds.features + test_ds.features
    all_ids = train_ds.ids + test_ds.ids
    
    for cmpd in target_compounds:
        indices = [i for i, x in enumerate(all_ids) if x == cmpd]
        if not indices:
            print(f"Warning: {cmpd} not found."); continue
            
        feats = [all_features[i] for i in indices]
        # 拼接并取平均 -> [151]
        mean_feat = np.mean(np.concatenate(feats, axis=0), axis=0)
        data_map[cmpd] = mean_feat

    if "ROT" not in data_map: return

    # 3. 定义特征描述列表
    OBP_RESIDUES = [114, 115, 118, 119, 190, 193, 194, 197, 386, 389, 390, 393, 412, 416]
    feat_descs = []
    for i in range(151):
        if i < 144:
            atom_idx = i // 16
            feat_type = i % 16
            if feat_type < 14:
                desc = f"Atom {atom_idx} - Res {OBP_RESIDUES[feat_type]} Dist"
            elif feat_type == 14: desc = f"Atom {atom_idx} - Phe389 Score"
            elif feat_type == 15: desc = f"Atom {atom_idx} - Phe390 Score"
        elif i == 144: desc = "Global Cos"
        elif 145 <= i <= 147: desc = f"Global 389 Type {i-145}"
        elif 148 <= i <= 150: desc = f"Global 390 Type {i-148}"
        feat_descs.append(desc)

    # =======================================================
    # 分析 1: 为什么 BRE (Low) 被误判为 High?
    # 寻找: ROT 和 BRE 在 Phe390 上相似，但在哪个特征上差异最大？
    # =======================================================
    print("\n" + "="*80)
    print("ANALYSIS 1: BRE (False Positive) vs ROT (True Positive)")
    print("Why is BRE bad despite hitting Phe390?")
    print("Looking for features where ROT is Strong but BRE is Weak (or vice versa)...")
    print("="*80)
    
    if "BRE" in data_map:
        diffs = []
        vec_rot = data_map["ROT"]
        vec_bre = data_map["BRE"]
        
        for i in range(151):
            # 我们关注差异大的特征
            # Z-Score 空间：差值 > 1.0 说明有显著物理差异
            diff = vec_rot[i] - vec_bre[i]
            diffs.append((i, diff, vec_rot[i], vec_bre[i]))
            
        # 按绝对差异排序
        diffs.sort(key=lambda x: abs(x[1]), reverse=True)
        
        print(f"{'Rank':<5} | {'Feat':<35} | {'Diff':<8} | {'ROT (High)':<10} | {'BRE (Low)':<10}")
        print("-" * 80)
        for rank, (idx, d, v_rot, v_bre) in enumerate(diffs[:15]):
            print(f"{rank+1:<5} | {feat_descs[idx]:<35} | {d:8.4f} | {v_rot:10.4f} | {v_bre:10.4f}")

    # =======================================================
    # 分析 2: 为什么 UNC (High) 被误判为 Low?
    # 寻找: UNC 在哪里像 ROT? (既然它不像 Phe390)
    # =======================================================
    print("\n" + "="*80)
    print("ANALYSIS 2: UNC (False Negative) vs ROT (True Positive)")
    print("Does UNC hit ANY residue that ROT also hits?")
    print("="*80)
    
    if "UNC" in data_map:
        diffs_unc = []
        vec_rot = data_map["ROT"]
        vec_unc = data_map["UNC"]
        
        for i in range(151):
            # 我们想找 UNC 和 ROT "长得像" 的特征
            # 即 diff 接近 0，且绝对值比较大（说明都有强相互作用）
            # 或者我们找 UNC 最突出的特征
            diff = vec_rot[i] - vec_unc[i]
            # 我们这里还是看差异，看看 UNC 到底死在哪
            diffs_unc.append((i, diff, vec_rot[i], vec_unc[i]))
            
        diffs_unc.sort(key=lambda x: abs(x[1]), reverse=True)
        
        print(f"{'Rank':<5} | {'Feat':<35} | {'Diff':<8} | {'ROT (High)':<10} | {'UNC (High)':<10}")
        print("-" * 80)
        for rank, (idx, d, v_rot, v_unc) in enumerate(diffs_unc[:15]):
            print(f"{rank+1:<5} | {feat_descs[idx]:<35} | {d:8.4f} | {v_rot:10.4f} | {v_unc:10.4f}")

if __name__ == "__main__":
    main()