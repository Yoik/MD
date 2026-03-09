import torch
import numpy as np
import os
import glob
import pandas as pd

# ================= 配置 =================
DATA_DIR = "data/features"
TARGETS = ["Dopa", "BRE", "S84", "ARI"]

def get_feature_stats(name):
    # 模糊匹配查找文件夹
    found_dir = None
    if not os.path.exists(DATA_DIR): return None
    for d in os.listdir(DATA_DIR):
        if name.lower() in d.lower():
            found_dir = os.path.join(DATA_DIR, d)
            break
    
    if not found_dir: return None

    files = glob.glob(os.path.join(found_dir, "*", "graph_features.pt"))
    if not files: return None
    
    offsets = []
    cos_angles = []
    
    for f in files:
        try:
            data_list = torch.load(f, weights_only=False)
            for data in data_list:
                # global_attr: [Cos1, Cos2, Offset]
                g = data.global_attr.numpy().flatten()
                offsets.append(g[2])
                cos_angles.append(g[0])
        except: pass
        
    return {
        "Compound": name,
        "Avg_Offset": np.mean(offsets),
        "Avg_Cos_W648": np.mean(cos_angles)
    }

def main():
    print(f"{'Compound':<10} {'Avg_Offset':<15} {'Avg_Cos_W648':<15}")
    print("-" * 45)
    for t in TARGETS:
        stats = get_feature_stats(t)
        if stats:
            print(f"{stats['Compound']:<10} {stats['Avg_Offset']:<15.4f} {stats['Avg_Cos_W648']:<15.4f}")

if __name__ == "__main__":
    main()