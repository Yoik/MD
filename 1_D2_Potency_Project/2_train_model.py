import sys
import os
import matplotlib
matplotlib.use('Agg')  # <--- 【关键修改1】防止在服务器上画图崩溃

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import ShuffleSplit
from src.dataset import prepare_data, TrajectoryDataset
from src.model import EfficiencyPredictor

# ==============================================================================
# 配置参数
# ==============================================================================
LABEL_FILE = "data/labels.csv"
RESULT_DIR = "data/features"
MODEL_SAVE_PATH = "saved_models/best_model_mccv.pth" 
SCALER_SAVE_PATH = "saved_models/scaler.pkl"

# 物理参数
POCKET_ATOM_NUM = 12
INPUT_DIM = 19       # 13(Geom) + 6(Elec) + Attention
# 训练参数
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-4
NUM_EPOCHS = 45      # 推荐 40-50
BATCH_SIZE = 32
N_SPLITS = 20        # 20 轮交叉验证
TEST_SIZE = 2        # 每次随机选 2 个化合物做测试 (Dopa 除外)

def augment_trajectory(traj, noise_std=0.01):
    """
    traj: [B, T, F]
    """
    noise = torch.randn_like(traj) * noise_std
    return traj + noise

def attention_entropy(attn_weights, eps=1e-8):
    """
    attn_weights: [B, T, 1]
    """
    p = attn_weights.squeeze(-1)
    entropy = -torch.sum(p * torch.log(p + eps), dim=1)
    return entropy.mean()

# ==============================================================================
# 主程序
# ==============================================================================
def main():
    # 1. 准备数据
    print("Preparing data with slicing...")
    try:
        train_ds, test_ds = prepare_data(
            label_file=LABEL_FILE, 
            result_dir=RESULT_DIR, 
            pocket_atom_num=POCKET_ATOM_NUM, 
            save_scaler_path=SCALER_SAVE_PATH,
            window_size=100, 
            stride=20
        )
    except Exception as e:
        print(f"\n[DATA ERROR] Failed to load data: {e}")
        import traceback
        traceback.print_exc()
        return

    # 2. 整合全量数据 (用于自定义划分)
    full_features = train_ds.features + test_ds.features
    full_labels = train_ds.labels + test_ds.labels
    full_ids = train_ds.ids + test_ds.ids
    full_dataset = TrajectoryDataset(full_features, full_labels, full_ids)
    
    # 获取所有唯一的化合物 ID
    all_ids = full_dataset.ids
    unique_compounds = sorted(list(set(all_ids)))

    compound2idx = {name: i for i, name in enumerate(unique_compounds)}
    num_compounds = len(unique_compounds)
    print(f"Compound embedding vocab size: {num_compounds}")

    
    # === [关键修复]：分离 Reference (Dopa) 和 候选化合物 ===
    ref_compound = "Dopa"
    if ref_compound in unique_compounds:
        # 候选者：排除 Dopa 的其他化合物
        candidates = [c for c in unique_compounds if c != ref_compound]
        # Dopa 永远在训练集
        always_train_cmpds = [ref_compound]
    else:
        candidates = unique_compounds
        always_train_cmpds = []
        print(f"[Warning] '{ref_compound}' not found in dataset!")

    print(f"\n>>> Dataset Ready.")
    print(f"Total Slices: {len(full_dataset)}")
    print(f"Total Compounds: {len(unique_compounds)}")
    print(f"Splittable Candidates: {len(candidates)} (Dopa excluded from test)")
    print("-" * 50)
    print(f"Starting MCCV ({N_SPLITS} rounds)...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # === [关键修复]：初始化分割器和结果容器 ===
    # 定义随机分割器
    rs = ShuffleSplit(n_splits=N_SPLITS, test_size=TEST_SIZE, random_state=42)
    
    # 初始化结果列表 (替代原来的 compound_results 字典)
    loocv_results = []

    # === 4. 开始 MCCV 循环 ===
    # 注意：这里是对 candidates 进行分割，而不是 unique_compounds
    for round_idx, (train_idx_cand, test_idx_cand) in enumerate(rs.split(candidates)):
        
        # 4.1 构建本轮的化合物名单
        # 从索引映射回名字
        train_cand_names = [candidates[i] for i in train_idx_cand]
        test_cmpds = [candidates[i] for i in test_idx_cand]
        
        # 训练集 = 被选中的候选者 + Dopa
        train_cmpds = train_cand_names + always_train_cmpds
        
        print(f"\n--- Round {round_idx+1}/{N_SPLITS} ---")
        # print(f"  Train: {len(train_cmpds)} compounds")
        # print(f"  Test : {test_cmpds}")

        # 根据 ID 找切片索引
        train_indices = [i for i, x in enumerate(full_dataset.ids) if x in train_cmpds]
        test_indices = [i for i, x in enumerate(full_dataset.ids) if x in test_cmpds]
        
        # 构建 DataLoader
        train_loader = DataLoader(Subset(full_dataset, train_indices), batch_size=BATCH_SIZE, shuffle=True)
        # 注意：test_loader 这里其实不需要了，因为下面是按化合物单独测试的，但保留也没事
        
        # 初始化模型
        model = EfficiencyPredictor(input_dim=INPUT_DIM, attn_dropout=0.2).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        criterion = nn.MSELoss()

        # 4.2 训练 (Train)
        model.train()
        for epoch in range(NUM_EPOCHS):
            for traj, label, cmpd_name in train_loader:
                traj, label = traj.to(device), label.to(device)

                traj = augment_trajectory(traj, noise_std=0.01)

                optimizer.zero_grad()
                out = model(traj)

                pred = out["pred"]
                frame_scores = out["frame_scores"]
                gate_val = out["gate"]
                attn_weights = out["attn"]

                # 1️⃣ 原有 macro-level loss
                mse_loss = criterion(pred.squeeze(), label.squeeze())

                # 2️⃣ 新增：轨迹均值约束
                frame_mean = frame_scores.mean(dim=1).squeeze(-1)
                mean_loss = criterion(frame_mean, label.squeeze())

                # 3️⃣ 注意力熵正则
                # attn_weights: [B, T, 1]
                T = attn_weights.shape[1]

                H_max = torch.log(torch.tensor(T, device=attn_weights.device, dtype=torch.float))
                H_target = 0.6 * H_max   # 经验值：0.5~0.7 都可以

                H = attention_entropy(attn_weights)

                ent_loss = (H - H_target).pow(2)

                # 4️⃣ 总 loss
                loss = (
                    mse_loss
                    + 1.0 * mean_loss
                    + 0.01 * ent_loss
                )

                loss.backward()
                optimizer.step()
                if epoch == 0:
                    print(
                        "pred mean:", pred.mean().item(),
                        "frame mean:", frame_mean.mean().item(),
                        "label mean:", label.mean().item()
                    )

        # 4.3 测试 (Test) - 按化合物逐个评估
        model.eval()
        with torch.no_grad():
            for cmpd_name in test_cmpds:
                # 找出该化合物对应的所有切片索引
                cmpd_indices = [i for i, x in enumerate(full_dataset.ids) if x == cmpd_name]
                if len(cmpd_indices) == 0: continue
                
                # 专属 Loader
                cmpd_subset = Subset(full_dataset, cmpd_indices)
                cmpd_loader = DataLoader(cmpd_subset, batch_size=BATCH_SIZE, shuffle=False)
                
                slice_preds = []
                ground_truth = None
                
                for traj, label, cmpd_name_batch in cmpd_loader:
                    traj = traj.to(device)
                    out = model(traj)
                    pred = out["pred"]

                    slice_preds.extend(
                        pred.detach().cpu().numpy().flatten()
                    )
                    if ground_truth is None and len(label) > 0:
                        ground_truth = label[0].item()
                
                if len(slice_preds) > 0:
                    avg_pred = np.mean(slice_preds)
                    true_val_100 = ground_truth * 100
                    pred_val_100 = avg_pred * 100
                    diff = pred_val_100 - true_val_100
                    
                    loocv_results.append({
                        "Compound": cmpd_name,
                        "True": true_val_100,
                        "Pred": pred_val_100,
                        "Diff": diff,
                        "Round": round_idx
                    })

    # ==========================================
    # 5. 汇总报告与可视化
    # ==========================================
    print("\n" + "="*50)
    print("MCCV Final Report")
    print("="*50)
    
    # 转换为 DataFrame
    df_res = pd.DataFrame(loocv_results)
    
    if df_res.empty:
        print("Error: No results generated.")
        return

    # 计算整体指标
    rmse = np.sqrt(np.mean((df_res['True'] - df_res['Pred'])**2))
    corr = df_res['True'].corr(df_res['Pred'])
    print(f"Overall RMSE : {rmse:.4f}")
    print(f"Pearson R    : {corr:.4f}")

    # === 绘图部分 (Mean ± SEM) ===
    import matplotlib.pyplot as plt
    import seaborn as sns
    from scipy import stats

    print("\nGenerating Correlation Plot with Error Bars...")
    
    # 聚合计算 Mean 和 SEM
    summary = df_res.groupby('Compound').agg({
        'Pred': ['mean', 'sem', 'count'],
        'True': 'first'
    }).reset_index()
    summary.columns = ['Compound', 'Pred_Mean', 'Pred_SEM', 'Count', 'True_Val']
    
    print("\n--- Compound Performance Summary ---")
    print(summary[['Compound', 'True_Val', 'Pred_Mean', 'Pred_SEM', 'Count']].to_string(index=False))

    # 计算聚合后的 R 和 RMSE
    r_pearson, p_value = stats.pearsonr(summary['True_Val'], summary['Pred_Mean'])
    rmse_agg = np.sqrt(np.mean((summary['True_Val'] - summary['Pred_Mean'])**2))

    # 绘图
    plt.figure(figsize=(10, 8))
    sns.set_style("whitegrid")
    
    plt.errorbar(
        x=summary['True_Val'], 
        y=summary['Pred_Mean'], 
        yerr=summary['Pred_SEM'], 
        fmt='o', color='royalblue', ecolor='gray', 
        elinewidth=1.5, capsize=4, alpha=0.8,
        label='Compounds (Mean ± SEM)'
    )
    
    min_val = min(summary['True_Val'].min(), summary['Pred_Mean'].min()) - 5
    max_val = max(summary['True_Val'].max(), summary['Pred_Mean'].max()) + 5
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.6, label='Ideal Fit')
    
    for i, row in summary.iterrows():
        plt.text(row['True_Val']+1, row['Pred_Mean']+1, row['Compound'], fontsize=9, alpha=0.7)
        
    plt.title(f"MCCV Prediction ({N_SPLITS} rounds)\nPearson R = {r_pearson:.3f} | RMSE = {rmse_agg:.3f}")
    plt.xlabel("Experimental Efficacy (%)")
    plt.ylabel("Predicted Efficacy (%)")
    plt.legend()
    
    plt.savefig("efficacy_correlation_plot_sem_entropy_0001.png", dpi=300, bbox_inches='tight')
    print("\nPlot saved to: efficacy_correlation_plot_sem_entropy_0001.png")
    
    # === 6. 全量重训并保存模型 ===
    print("\nRetraining Final Model on ALL data...")

    final_loader = DataLoader(
        full_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    final_model = EfficiencyPredictor(
        input_dim=INPUT_DIM,
        attn_dropout=0.2
    ).to(device)

    optimizer = torch.optim.Adam(
        final_model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    criterion = nn.MSELoss()

    final_model.train()
    for epoch in range(NUM_EPOCHS):
        for traj, label, cmpd_name in final_loader:
            traj = traj.to(device)
            label = label.to(device)

            optimizer.zero_grad()

            # ✅ 正确接收 dict 输出
            out = final_model(traj)
            pred = out["pred"]

            loss = criterion(pred.squeeze(), label.squeeze())
            loss.backward()
            optimizer.step()

    torch.save(final_model.state_dict(), MODEL_SAVE_PATH)
    print(f"Final model saved to: {MODEL_SAVE_PATH}")
    print("Done.")

# 别忘了调用 main
if __name__ == "__main__":
    main()