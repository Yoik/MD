import numpy as np
import glob
import os
import pandas as pd
import shutil

# --- 配置 ---
FEATURE_DIR = "data/features"  # 现有的特征目录
LABEL_FILE = "data/labels.csv" # 现有的标签文件
MUTATION_FACTOR = 0.9          # 突变后效能保留比例 (90%)

# Phe390 在特征向量中的索引 
# 旧版结构 (19维): [Dist_1..12, Cos, P1(389), P2(390)]
# 新版结构 (21维): [Dist_1..12, Cos, P1(389), P2(390), Dist_N_D114, Dist_N_W386]
# 索引 16, 17, 18 依然对应 P2(390)
PHE390_INDICES = [16, 17, 18]

def main():
    print(f"Loading labels from {LABEL_FILE}...")
    df = pd.read_csv(LABEL_FILE)
    
    new_rows = []
    created_count = 0
    
    # 遍历现有的每一个化合物
    for index, row in df.iterrows():
        original_name = row['Compound']
        original_efficacy = float(row['Efficacy'])
        
        # 1. 跳过已经是虚拟突变体的数据 (防止递归生成)
        if "Virtual_F390A" in original_name:
            continue
            
        # 2. 定义新化合物的名称和效能
        new_name = f"Virtual_F390A_{original_name}"
        new_efficacy = original_efficacy * MUTATION_FACTOR
        
        print(f"Processing: {original_name} ({original_efficacy}) -> {new_name} ({new_efficacy:.2f})")
        
        # 3. 寻找该化合物的所有 .npy 文件
        # 假设结构: data/features/CompoundName/Replicate_X/xxx.npy
        search_pattern = os.path.join(FEATURE_DIR, f"*{original_name}*", "*", "*_features.npy")
        files = glob.glob(search_pattern)
        
        if not files:
            print(f"  [Warn] No feature files found for {original_name}, skipping.")
            continue
            
        # 4. 生成虚拟数据
        for f in files:
            # 读取
            data = np.load(f)
            
            # 修改 (Mask Phe390)
            mutant_data = data.copy()
            mutant_data[:, PHE390_INDICES] = 0.0
            
            # 构造新保存路径
            # 我们需要模仿原有的目录结构，以便 dataset.py 能读到
            # 原: data/features/2025_Dopa/replicate_1/feat.npy
            # 新: data/features/Virtual_F390A_Dopa/replicate_1/feat.npy
            
            # 获取相对路径部分
            rel_path = os.path.relpath(f, FEATURE_DIR) # e.g., 2025_Dopa/replicate_1/feat.npy
            path_parts = rel_path.split(os.sep)
            
            # 将第一层文件夹名加上前缀
            if "Virtual_F390A" not in path_parts[0]:
                 path_parts[0] = f"Virtual_F390A_{path_parts[0]}"
            
            new_rel_path = os.path.join(*path_parts)
            save_path = os.path.join(FEATURE_DIR, new_rel_path)
            
            # 创建文件夹
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            # 保存
            np.save(save_path, mutant_data)
            created_count += 1
            
        # 5. 记录到新标签列表
        new_rows.append({
            "Compound": new_name,
            "Efficacy": new_efficacy
        })
        
    print(f"\nGenerated {created_count} virtual feature files.")

    # 6. 更新 CSV
    if new_rows:
        new_df = pd.DataFrame(new_rows)
        # 过滤掉已经存在的，避免重复添加
        existing_names = df['Compound'].values
        new_df = new_df[~new_df['Compound'].isin(existing_names)]
        
        if not new_df.empty:
            final_df = pd.concat([df, new_df], ignore_index=True)
            final_df.to_csv(LABEL_FILE, index=False)
            print(f"Updated {LABEL_FILE} with {len(new_df)} new mutant entries.")
        else:
            print("No new entries needed for CSV (already existed).")
    else:
        print("No new data generated.")

if __name__ == "__main__":
    main()