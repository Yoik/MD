# src/dataset.py

import numpy as np
import pandas as pd
import glob
import os
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pickle

# TrajectoryDataset 类定义保持不变
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

# ==============================================================================
# 完整修改版 prepare_data
# ==============================================================================
def prepare_data(label_file, result_dir, pocket_atom_num, save_scaler_path=None, 
                 window_size=200, stride=50): # <--- 1. 新增参数：窗口大小和步长
    # 1. 读取标签
    df = pd.read_csv(label_file)
    
    X_all, y_all, ids_all = [], [], []
    
    print(f"Scanning data for {len(df)} compounds...")
    print(f"  -> Slicing Strategy: Window={window_size}, Stride={stride}")

    for idx, row in df.iterrows():
        cmpd_name = str(row['Compound']).strip()  # 例如 "S10"
        val = float(row['Efficacy']) / 100.0 
        
        # 模糊匹配文件夹路径
        compound_folder_pattern = os.path.join(result_dir, f"*{cmpd_name}*")
        matching_folders = glob.glob(compound_folder_pattern)
        
        if not matching_folders:
            # 没跑完的数据跳过不报错，保持安静
            # print(f"  [Warning] No folder found for compound: {cmpd_name}")
            continue
            
        found_data = False
        for folder in matching_folders:
            npy_search = os.path.join(folder, "*", "*_features.npy")
            files = glob.glob(npy_search)
            
            for f in files:
                try:
                    data = np.load(f)
                    
                    # === 2. 核心修改：切片逻辑整合 ===
                    # 只要轨迹长度允许，就按步长滑动切片
                    if data.shape[0] > window_size:
                        n_frames = data.shape[0]
                        # 比如: 0~200, 50~250, 100~300 ...
                        # 每一个切片都被视为一个独立的数据点，但拥有相同的 Compound ID
                        for start in range(0, n_frames - window_size + 1, stride):
                            end = start + window_size
                            slice_data = data[start:end, :]
                            
                            X_all.append(slice_data)
                            y_all.append(val)
                            ids_all.append(cmpd_name) # 关键：ID 也要复制，后续才能正确划分
                        
                        found_data = True
                    else:
                        # 如果轨迹比窗口还短，则保留原始数据（保底）
                        if data.shape[0] > 0:
                            X_all.append(data)
                            y_all.append(val)
                            ids_all.append(cmpd_name)
                            found_data = True
                except:
                    print(f"    Error loading {f}")

        if not found_data:
             print(f"  [Warning] Folder found but no .npy for: {cmpd_name}")

    print(f"Total slices generated: {len(X_all)}") # 这里可以看到数据量翻倍的效果
    
    if len(X_all) == 0:
        raise ValueError("No data loaded! Please check paths.")

    # 4. 划分数据集 (按化合物 ID 划分) - 【确保数据不泄露】
    # 4.1. 拿到所有唯一的化合物名字
    # set(ids_all) 会自动去重，所以切片多了也没关系，名字还是那几个
    unique_ids = list(set(ids_all))
    
    # 4.2. 对“名字”进行切分
    # 这里的 0.2 是指 20% 的化合物作为测试集（例如 15个里选3个）
    train_ids, test_ids = train_test_split(unique_ids, test_size=0.2, random_state=42)
    
    # 4.3. 转成集合
    train_id_set = set(train_ids)
    
    # 4.4. 根据名字把数据的索引归位
    train_idx = []
    test_idx = []
    
    for i, cmpd_id in enumerate(ids_all):
        # 这里的 ids_all 长度是切片后的总长度
        # 只要切片的 ID 属于训练集，该切片就进训练集
        if cmpd_id in train_id_set:
            train_idx.append(i)
        else:
            test_idx.append(i)

    print(f"Split Strategy: Grouped by Compound ID (Slicing Enabled)")
    print(f"Train: {len(train_ids)} compounds -> {len(train_idx)} slices")
    print(f"Test : {len(test_ids)} compounds -> {len(test_idx)} slices")
    
    # 5. 特征工程 (倒数 + 标准化)
    # 0-11: OBP 残基距离
    dist_indices = list(range(12)) 
    
    # 仅用训练集计算均值方差 (避免数据泄露)
    train_features_concat = np.concatenate([X_all[i] for i in train_idx], axis=0)
    
    # 对指定列取倒数
    train_features_concat[:, dist_indices] = 1.0 / (train_features_concat[:, dist_indices] + 1e-6)
    
    scaler = StandardScaler()
    scaler.fit(train_features_concat)
    
    if save_scaler_path:
        with open(save_scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
            
    # 应用到所有数据
    X_processed = []
    for x in X_all:
        x_new = x.copy()
        x_new[:, dist_indices] = 1.0 / (x_new[:, dist_indices] + 1e-6)
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