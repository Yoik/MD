import sys
import os
import matplotlib
matplotlib.use('Agg')  # 防止服务器绘图报错

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import ShuffleSplit
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# 复用你的项目代码
from src.dataset import prepare_data, TrajectoryDataset
from src.model import EfficiencyPredictor

# ================= 配置参数 =================
LABEL_FILE = "data/labels.csv"
RESULT_DIR = "data/features"
SCALER_SAVE_PATH = "saved_models/scaler.pkl"

# 物理参数
POCKET_ATOM_NUM = 12
INPUT_DIM = 151
LEARNING_RATE = 0.002
DROPOUT_RATE = 0.2
WEIGHT_DECAY = 1e-4
NUM_EPOCHS = 60      # 评估时可以稍微减少轮数以加快速度
BATCH_SIZE = 32
N_SPLITS = 20        # 为了速度，评估轮次可以设为 10 (原为 20)
TEST_SIZE = 1

# === 【关键设置】我们要测试的时间点 ===
# 假设 90ns 是 100%。我们测试以下比例的数据量。
# "Start": 强制只取前 1-2 个切片，模拟"刚跑完几个帧"
# 0.11: 大约 10ns (10/90)
# 0.22: 大约 20ns (20/90)
DURATION_FRACTIONS = [0.01, 0.11, 0.22, 0.33, 0.44, 0.55, 1.0] 
DURATION_LABELS    = ["Start(<1ns)", "10ns", "20ns", "50ns", "Full(90ns)"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_and_evaluate(dataset, candidates, always_train_cmpds, fraction_label):
    """
    对给定的数据集进行 MCCV 交叉验证，返回平均 R 和 RMSE
    """
    rs = ShuffleSplit(n_splits=N_SPLITS, test_size=TEST_SIZE, random_state=42)
    loocv_results = []
    
    # 获取所有 ID 用于索引查找
    all_ids = np.array(dataset.ids)
    
    # 进度条
    pbar = tqdm(total=N_SPLITS, desc=f"Eval {fraction_label}", leave=False)
    
    for round_idx, (train_idx_cand, test_idx_cand) in enumerate(rs.split(candidates)):
        # 1. 划分化合物
        train_cand_names = [candidates[i] for i in train_idx_cand]
        test_cmpds = [candidates[i] for i in test_idx_cand]
        train_cmpds = train_cand_names + always_train_cmpds
        
        # 2. 获取对应的样本索引
        # 注意：这里 dataset 已经是截断过的数据了，直接查名字即可
        train_indices = [i for i, x in enumerate(dataset.ids) if x in train_cmpds]
        
        if len(train_indices) == 0:
            continue

        # 3. 训练
        train_loader = DataLoader(Subset(dataset, train_indices), batch_size=BATCH_SIZE, shuffle=True)
        
        model = EfficiencyPredictor(input_dim=INPUT_DIM, dropout_rate=DROPOUT_RATE).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        criterion = nn.MSELoss()
        
        model.train()
        for epoch in range(NUM_EPOCHS):
            for traj, label, _ in train_loader:
                traj, label = traj.to(device), label.to(device)
                optimizer.zero_grad()
                out = model(traj)
                loss = criterion(out["pred"].squeeze(), label.squeeze())
                loss.backward()
                optimizer.step()
        
        # 4. 测试 (逐个化合物)
        model.eval()
        with torch.no_grad():
            for cmpd_name in test_cmpds:
                cmpd_indices = [i for i, x in enumerate(dataset.ids) if x == cmpd_name]
                if len(cmpd_indices) == 0: continue
                
                cmpd_loader = DataLoader(Subset(dataset, cmpd_indices), batch_size=BATCH_SIZE, shuffle=False)
                
                slice_preds = []
                ground_truth = None
                
                for traj, label, _ in cmpd_loader:
                    traj = traj.to(device)
                    out = model(traj)
                    slice_preds.extend(out["pred"].cpu().numpy().flatten())
                    if ground_truth is None: ground_truth = label[0].item()
                
                if slice_preds:
                    pred_val = np.mean(slice_preds) * 100 # 还原比例
                    true_val = ground_truth * 100
                    loocv_results.append({
                        "True": true_val, "Pred": pred_val, "Compound": cmpd_name
                    })
        pbar.update(1)
    pbar.close()

    # 统计本轮 Metrics
    if not loocv_results: return None, None
    
    df = pd.DataFrame(loocv_results)
    # 按化合物聚合 (因为 MCCV 可能会多次测同一个化合物)
    summary = df.groupby('Compound').mean().reset_index()
    
    if len(summary) < 2: return 0.0, 0.0
    
    rmse = np.sqrt(np.mean((summary['True'] - summary['Pred'])**2))
    corr = summary['True'].corr(summary['Pred'])
    return corr, rmse

def main():
    print(">>> Loading Full Dataset...")
    train_ds, test_ds = prepare_data(
        label_file=LABEL_FILE, 
        result_dir=RESULT_DIR, 
        pocket_atom_num=POCKET_ATOM_NUM, 
        save_scaler_path=SCALER_SAVE_PATH,
        window_size=100, stride=20
    )
    
    # 合并全量数据
    full_features = train_ds.features + test_ds.features
    full_labels = train_ds.labels + test_ds.labels
    full_ids = train_ds.ids + test_ds.ids
    
    # 建立索引映射：Compound Name -> [List of Indices]
    # 假设数据是按时间顺序加载的 (extract_features 通常是 append 模式)
    unique_compounds = sorted(list(set(full_ids)))
    idx_map = {name: [] for name in unique_compounds}
    for idx, name in enumerate(full_ids):
        idx_map[name].append(idx)
        
    print(f"Total Compounds: {len(unique_compounds)}")
    for name in unique_compounds:
        print(f"  {name}: {len(idx_map[name])} slices")

    # 划分 训练/测试 候选名单 (固定不变)
    ref_compounds = ["Dopa", "ARI"]
    always_train_cmpds = [c for c in ref_compounds if c in unique_compounds]
    candidates = [c for c in unique_compounds if c not in always_train_cmpds]
    
    final_metrics = []

    print("\n>>> Starting Duration Benchmark...")
    print(f"Testing Durations: {DURATION_LABELS}")
    
    for fraction, label in zip(DURATION_FRACTIONS, DURATION_LABELS):
        print(f"\n--- Testing Duration: {label} (Fraction: {fraction}) ---")
        
        # 1. 构建截断数据集
        subset_indices = []
        total_slices_used = 0
        
        for name in unique_compounds:
            all_indices = idx_map[name] # 假设这是按时间排序的
            n_total = len(all_indices)
            
            # 计算截断点
            cut_point = int(n_total * fraction)
            # 保证至少有 1 个切片 (模拟 "Start/1帧")
            if cut_point < 1: cut_point = 1
            
            selected = all_indices[:cut_point]
            subset_indices.extend(selected)
            total_slices_used += len(selected)
            
        print(f"  Using {total_slices_used} / {len(full_ids)} total slices.")
        
        # 创建临时 Dataset 对象
        # 注意：这里为了节省内存，不复制 features，而是通过 Subset 或重新封装
        # 但 TrajectoryDataset 需要 list，所以我们筛选一下 list
        sub_features = [full_features[i] for i in subset_indices]
        sub_labels = [full_labels[i] for i in subset_indices]
        sub_ids = [full_ids[i] for i in subset_indices]
        
        sub_dataset = TrajectoryDataset(sub_features, sub_labels, sub_ids)
        
        # 2. 运行评估
        r, rmse = train_and_evaluate(sub_dataset, candidates, always_train_cmpds, label)
        
        if r is not None:
            print(f"  Result -> R: {r:.4f}, RMSE: {rmse:.4f}")
            final_metrics.append({
                "Duration": label,
                "Fraction": fraction,
                "Pearson_R": r,
                "RMSE": rmse
            })
        else:
            print("  Result -> Failed (Insufficient data?)")

    # ================= 绘图 =================
    if not final_metrics: return

    df_res = pd.DataFrame(final_metrics)
    print("\n>>> Final Summary:")
    print(df_res)
    
    # 保存结果
    df_res.to_csv("benchmark_duration_results.csv", index=False)

    # 画双轴图
    fig, ax1 = plt.subplots(figsize=(10, 6))

    color = 'tab:blue'
    ax1.set_xlabel('Simulation Duration')
    ax1.set_ylabel('Pearson R', color=color)
    ax1.plot(df_res['Duration'], df_res['Pearson_R'], marker='o', color=color, linewidth=2, label='Correlation (R)')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_ylim(0, 1.0)
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()  # 实例化第二个轴
    color = 'tab:red'
    ax2.set_ylabel('RMSE', color=color)
    ax2.plot(df_res['Duration'], df_res['RMSE'], marker='s', linestyle='--', color=color, linewidth=2, label='RMSE')
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title('Impact of Simulation Duration on Model Performance')
    fig.tight_layout()
    plt.savefig('benchmark_duration_plot.png', dpi=300)
    print("\nPlot saved to 'benchmark_duration_plot.png'")

if __name__ == "__main__":
    main()