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

class RankingDataset(Dataset):
    def __init__(self, query_feats, query_labels, query_ids, ref_feats_list, ref_value=1.0):
        """
        :param query_feats: 待测化合物的切片列表
        :param ref_feats_list: 参考化合物（已切片、已归一化）的列表
        :param ref_value: 参考化合物的效能值 (通常为 1.0 或 100.0)
        """
        self.query_feats = query_feats
        self.query_labels = query_labels
        self.query_ids = query_ids
        self.ref_feats = ref_feats_list  # 保存所有参考片段供随机采样
        self.ref_value = ref_value

        # 将参考特征列表转换为张量
        if len(ref_feats_list) > 0:
            self.ref_tensor = torch.from_numpy(np.stack(ref_feats_list)).float()
        else:
            raise ValueError("Reference list is empty!")
    
    def __len__(self):
        return len(self.query_feats)
    
    def __getitem__(self, idx):
        # 1. 获取 Query (待测)
        q_traj = torch.from_numpy(self.query_feats[idx]).float()
        q_score = self.query_labels[idx]
        
        # 2. 获取 Reference (参考) - 随机采样一个片段以增加鲁棒性
        # 注意：这里假设 ref_feats 不为空
        # rand_ref_idx = np.random.randint(len(self.ref_feats))
        r_traj = self.ref_tensor
        
        # 3. 生成 Ranking Label (用于 MarginRankingLoss)
        # y = 1 表示 Query >= Ref (强或相当)
        # y = -1 表示 Query < Ref (弱)
        # 这里阈值可以根据需要微调，比如 q_score >= self.ref_value * 0.9
        if q_score >= self.ref_value:
            rank_label = 1.0
        else:
            rank_label = -1.0
            
        return {
            'query_feat': q_traj,
            'ref_feat': r_traj,
            'rank_label': torch.tensor(rank_label, dtype=torch.float32),
            'query_score': torch.tensor(q_score, dtype=torch.float32), # 保留绝对值用于监控
            'compound_id': self.query_ids[idx]
        }
    
# ==============================================================================
# 完整修改版 prepare_data
# ==============================================================================
def prepare_data(label_file, result_dir, ref_feature_path,  # <--- [新增参数]
                 pocket_atom_num, save_scaler_path=None, 
                 window_size=200, stride=50):
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

    # === 3. 处理参考化合物特征 ===
    print(f"Processing Reference Data from dir: {ref_feature_path}")
    
    ref_raw_list = []
    
    # 1. 判断是文件夹还是文件
    if os.path.isdir(ref_feature_path):
        # 递归搜索文件夹下所有的 features.npy
        # 您的目录结构似乎是: 20251115_.../charmm-gui-..._replicate_1/..._features.npy
        search_pattern = os.path.join(ref_feature_path, "**", "*_features.npy")
        ref_files = glob.glob(search_pattern, recursive=True)
        print(f"  -> Found {len(ref_files)} reference files.")
    else:
        # 兼容旧逻辑：如果是文件直接加载
        ref_files = [ref_feature_path]

    if not ref_files:
        raise ValueError(f"No .npy files found in reference path: {ref_feature_path}")

    # 2. 循环加载并合并
    for f in ref_files:
        try:
            data = np.load(f)
            # 简单的形状检查
            if data.ndim == 2 and data.shape[1] > 0:
                ref_raw_list.append(data)
            else:
                print(f"    [Warning] Skipping invalid file: {f}")
        except Exception as e:
            print(f"    [Error] Failed to load {f}: {e}")

    if not ref_raw_list:
        raise ValueError("Reference data is empty!")

    # 3. 将所有 Reference 轨迹拼接成一个巨大的 Pool
    # 这样做的好处是切片时可以跨文件处理（如果是长轨迹）或者分别处理
    # 为了简单起见，我们对每个独立的文件分别切片，然后收集到一起
    
    ref_slices = []
    total_ref_frames = 0
    
    for ref_data in ref_raw_list:
        n_frames = ref_data.shape[0]
        total_ref_frames += n_frames
        
        # 对每个文件分别切片
        if n_frames > window_size:
            for start in range(0, n_frames - window_size + 1, stride):
                end = start + window_size
                slice_data = ref_data[start:end, :]
                ref_slices.append(slice_data)
        else:
            # 如果文件本身比窗口短，但不是空的，作为一个切片
            if n_frames > 0:
                ref_slices.append(ref_data)

    print(f"  -> Loaded {total_ref_frames} frames from reference.")
    print(f"  -> Generated {len(ref_slices)} reference slices.")
    
    if len(ref_slices) == 0:
        raise ValueError("Reference slices is empty after processing!")
    
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
    
    # 5. 特征工程
    dist_indices = list(range(12)) 
    
    # 5.1 仅用训练集 Fit Scaler
    train_features_concat = np.concatenate([X_all[i] for i in train_idx], axis=0)
    train_features_concat[:, dist_indices] = 1.0 / (train_features_concat[:, dist_indices] + 1e-6)
    
    scaler = StandardScaler()
    scaler.fit(train_features_concat)
    
    if save_scaler_path:
        with open(save_scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
            
    # 5.2 定义一个辅助函数来处理数据 (应用倒数 + Transform)
    def process_features_list(feature_list, scaler, dist_idxs):
        processed = []
        for x in feature_list:
            x_new = x.copy()
            x_new[:, dist_idxs] = 1.0 / (x_new[:, dist_idxs] + 1e-6) # 倒数
            x_new = scaler.transform(x_new) # 归一化
            processed.append(x_new)
        return processed

    # [修改] 对 Query (X_all) 进行处理
    X_processed = process_features_list(X_all, scaler, dist_indices)
    
    # [新增] 对 Reference (ref_slices) 进行完全相同的处理
    # 注意：必须使用同一个 scaler！
    ref_processed = process_features_list(ref_slices, scaler, dist_indices)

    # ================= [修改] 返回新的 Dataset =================
    # 6. 返回 Dataset
    # 传入 ref_processed 和 ref_value (假设 label 里的 100 是 1.0)
    
    train_dataset = RankingDataset(
        query_feats=[X_processed[i] for i in train_idx],
        query_labels=[y_all[i] for i in train_idx],
        query_ids=[ids_all[i] for i in train_idx],
        ref_feats_list=ref_processed,  # [新增]
        ref_value=1.0                  # [新增] 1.0 = 100%
    )
    
    test_dataset = RankingDataset(
        query_feats=[X_processed[i] for i in test_idx],
        query_labels=[y_all[i] for i in test_idx],
        query_ids=[ids_all[i] for i in test_idx],
        ref_feats_list=ref_processed,  # [新增] 测试集也用同样的 Reference
        ref_value=1.0
    )
    
    return train_dataset, test_dataset