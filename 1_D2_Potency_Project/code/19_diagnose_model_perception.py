import numpy as np
import pandas as pd
from src.dataset import prepare_data

def main():
    print(">>> Diagnosing Model Perception: What does the model ACTUALLY see?")
    
    # 1. 加载数据 (这一步会自动进行 Z-Score 归一化)
    # 我们必须看归一化后的数据，因为这才是模型看到的
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
    
    # 2. 选取典型代表
    # High Efficacy: ROT, UNC, R10
    # Low Efficacy: BRE, ARI, S84
    # Outlier: Lisu (Predicted 37, True 14 -> High False Positive)
    targets = ["ROT", "UNC", "BRE", "Lisu", "S84"]
    
    data_map = {}
    for cmpd in targets:
        indices = [i for i, x in enumerate(all_ids) if x == cmpd]
        if indices:
            # 拼接所有帧
            feats = np.concatenate([all_features[i] for i in indices], axis=0)
            # 计算平均特征向量 (Mean Profile)
            data_map[cmpd] = np.mean(feats, axis=0)
            
    # 3. 定义我们要检查的“传感器”
    # OBP List: 0:114, 5:193, 6:194, 7:197, 10:390, 11:393
    sensors = {
        #"Anchor (114)": 0,
        "Side (193)": 5,
        "Side (194)": 6,
        "Side (197)": 7,
        "Switch (390)": 10,
        "Stab (393)": 11,
        "Elec (390Sc)": 15
    }
    
    print("\n" + "="*100)
    print(f"{'Compound':<10} | {'True':<5} | {'Pred':<5} | {'Anch(114)':<10} | {'Side(193)':<10} | {'Side(194)':<10} | {'Swit(390)':<10} | {'Swit(393)':<10} | {'Elec390':<10}")
    print("-" * 100)
    
    # 手动填入刚才 LOO 的预测值，方便对比
    pred_map = {
        "ROT": 36.4, "UNC": 7.6, "BRE": 18.4, "Lisu": 37.4, "S84": 12.2
    }
    true_map = {
        "ROT": 51.6, "UNC": 43.1, "BRE": 0.7, "Lisu": 14.8, "S84": 6.9
    }

    for cmpd in targets:
        if cmpd not in data_map: continue
        
        vec = data_map[cmpd]
        
        # 对于每个传感器，我们要找 9 个原子里“最近”的那个（Min Z-Score）
        # 因为模型只要有一个原子抓住了就算抓住了
        readings = {}
        for name, feat_idx in sensors.items():
            # 提取 9 个原子在该特征上的值
            vals = []
            for a in range(9):
                # 电子积分(15) 是越大越好(负分?) -> 注意 Z-score 负值通常代表原始值小
                # 距离特征: Z-score 越负 -> 距离越近 -> 结合越强
                # 电子特征: 原始值越负越好 -> Z-score 越负越好
                # 所以我们统一找 "Min Z-Score"
                
                idx = a * 16 + feat_idx
                if idx < len(vec):
                    vals.append(vec[idx])
            
            readings[name] = min(vals) if vals else 99.9

        # 打印行
        # 标红/标绿逻辑：
        # 强信号 (Strong) < -0.5
        # 弱信号 (Weak)   > 0.0
        row_str = f"{cmpd:<10} | {true_map.get(cmpd,0):<5.1f} | {pred_map.get(cmpd,0):<5.1f} | "
        
        for k in sensors.keys():
            val = readings[k]
            mark = ""
            if val < -1.0: mark = "++" # 极强
            elif val < -0.5: mark = "+" # 强
            elif val > 0.5: mark = "X"  # 极弱/丢失
            
            row_str += f"{val:6.2f}{mark:<3} | "
            
        print(row_str)

    print("-" * 100)
    print("Diagnosis Guide:")
    print("1. Z-Score < -0.5 : Good Contact (Strong Signal)")
    print("2. Z-Score > 0.0  : Bad Contact (Weak/No Signal)")
    print("3. Check UNC's 'Anch(114)': If this is > 0, UNC is floating away from the anchor.")
    print("4. Check Lisu's 'Side': If this is < -1, Lisu is falsely grabbing the side lock.")

if __name__ == "__main__":
    main()