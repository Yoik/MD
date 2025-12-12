# src/dataset.py (更新版)
import numpy as np
import pandas as pd
import glob
import os
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pickle

class TrajectoryDataset(Dataset):
    def __init__(self, features_list, labels_list, compound_ids):
        self.features = features_list
        self.labels = labels_list
        self.ids = compound_ids
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        traj = torch.from_numpy(self.features[idx]).float()
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return traj, label, self.ids[idx]

def prepare_data(label_file, result_dir, pocket_atom_num, save_scaler_path=None):
    # 1. 读取标签，修改列名匹配您的 CSV
    # 您的 CSV 列名是: Compound, Efficacy
    df = pd.read_csv(label_file)
    
    X_all, y_all, ids_all = [], [], []
    
    print(f"Scanning data for {len(df)} compounds...")

    for idx, row in df.iterrows():
        cmpd_name = str(row['Compound']).strip()  # 例如 "S10"
        
        # 2. 标签标准化建议
        # 因为您的 Efficacy 是 0-100 的数值，建议直接除以 100 归一化到 0-1
        # 这样模型更容易收敛 (使用 MSELoss 时)
        val = float(row['Efficacy']) / 100.0 
        
        # 3. 模糊匹配文件夹路径
        # 寻找 result_dir 下包含 cmpd_name 的文件夹
        # 例如: results/*S10*
        compound_folder_pattern = os.path.join(result_dir, f"*{cmpd_name}*")
        matching_folders = glob.glob(compound_folder_pattern)
        
        if not matching_folders:
            print(f"  [Warning] No folder found for compound: {cmpd_name}")
            continue
            
        # 假设匹配到了文件夹，继续在这个文件夹下找 .npy
        # 结构: results/FullName/Replica/Features.npy
        found_data = False
        for folder in matching_folders:
            npy_search = os.path.join(folder, "*", "*_features.npy")
            files = glob.glob(npy_search)
            
            for f in files:
                try:
                    data = np.load(f)
                    # 简单检查数据有效性
                    if data.shape[0] > 0:
                        X_all.append(data)
                        y_all.append(val)
                        ids_all.append(cmpd_name)
                        found_data = True
                except:
                    print(f"    Error loading {f}")

        if not found_data:
             print(f"  [Warning] Folder found but no .npy for: {cmpd_name}")

    print(f"Total trajectories loaded: {len(X_all)}")
    
    if len(X_all) == 0:
        raise ValueError("No data loaded! Please check paths.")

    # 4. 划分数据集 (按索引划分)
    indices = np.arange(len(X_all))
    train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42)
    
    # 5. 特征工程 (倒数 + 标准化)
    M = pocket_atom_num
    
    # 仅用训练集计算均值方差
    train_features_concat = np.concatenate([X_all[i] for i in train_idx], axis=0)
    
    # 距离倒数处理 (前 M 列)
    train_features_concat[:, :M] = 1.0 / (train_features_concat[:, :M] + 1e-6)
    
    scaler = StandardScaler()
    scaler.fit(train_features_concat)
    
    if save_scaler_path:
        with open(save_scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
            
    # 应用到所有数据
    X_processed = []
    for x in X_all:
        x_new = x.copy()
        x_new[:, :M] = 1.0 / (x_new[:, :M] + 1e-6)
        x_new = scaler.transform(x_new)
        X_processed.append(x_new)

    # 6. 返回 Dataset
    train_dataset = TrajectoryDataset(
        [X_processed[i] for i in train_idx],
        [y_all[i] for i in train_idx],
        [ids_all[i] for i in train_idx]
    )
    
    test_dataset = TrajectoryDataset(
        [X_processed[i] for i in test_idx],
        [y_all[i] for i in test_idx],
        [ids_all[i] for i in test_idx]
    )
    
    return train_dataset, test_dataset