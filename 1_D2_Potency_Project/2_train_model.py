import sys
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import LeaveOneOut 
from scipy import stats 
from src.dataset import prepare_data, TrajectoryDataset
from src.model import EfficiencyPredictor
from src.config import init_config

# ================= 配置参数 =================
config = init_config()

LABEL_FILE = config.get_path("paths.label_file") #··标签文件路径
RESULT_DIR = config.get_path("paths.result_dir") #··结果输出目录
MODEL_SAVE_PATH = config.get_path("paths.model_path") #··模型保存路径
SCALER_SAVE_PATH = config.get_path("paths.scaler_path") #··标准化器保存路径

POCKET_ATOM_NUM = config.get_int("data.pocket_atom_num") #··口袋原子数量
INPUT_DIM = config.get_int("data.input_dim_features") #··输入特征维度
LEARNING_RATE = config.get_float("training.learning_rate") #··学习率
DROPOUT_RATE = config.get_float("training.dropout_rate") #··Dropout 比例
WEIGHT_DECAY = config.get_float("training.weight_decay") #··权重衰减
NUM_EPOCHS = config.get_int("training.num_epochs") #··训练轮数 
BATCH_SIZE = config.get_int("training.batch_size") #··批量大小

# 【新增】稀疏惩罚系数
# 值越大，模型删特征越狠；值越小，模型保留特征越多
# 建议 0.001 - 0.005
L1_LAMBDA = config.get_float("training.l1_lambda") #··L1 稀疏惩罚系数

def main():
    print("Preparing data...")
    try:
        train_ds, test_ds = prepare_data(
            label_file=LABEL_FILE, 
            result_dir=RESULT_DIR, 
            pocket_atom_num=POCKET_ATOM_NUM, 
            save_scaler_path=SCALER_SAVE_PATH,
            window_size=100, stride=20
        )
    except Exception as e:
        print(f"[DATA ERROR] {e}"); return

    # 1. 创建 Full Dataset
    full_dataset = TrajectoryDataset(
        train_ds.features + test_ds.features,
        train_ds.labels + test_ds.labels,
        train_ds.ids + test_ds.ids
    )
    
    unique_compounds = sorted(list(set(full_dataset.ids)))
    print(f"Total Compounds: {len(unique_compounds)}")
    
    # ================= 2. 药效团特征滤镜 (Hard Constraints) =================
    # 我们依然保留 Hard Mask，用来强制屏蔽 Y416/V114 等作弊特征
    # 但对于其他区域，我们全部设为 1，让 Dynamic Mask 自己去筛选
    
    print("\n[STRATEGY] Hybrid Masking Strategy")
    print("1. Hard Mask: Strictly ban Y416 & V114 (Cheating Features)")
    print("2. Dynamic Mask: Let model learn importance for everything else")
    
    hard_mask = np.ones(INPUT_DIM, dtype=np.float32)
    n_atoms = 9
    
    # 禁止的残基索引: 0(V114), 13(Y416)
    # 我们只屏蔽这两个，其他的全开，看模型自己选谁
    banned_indices = [0, 13] 
    
    for i in range(n_atoms):
        base = i * 16
        for idx in banned_indices:
            hard_mask[base + idx] = 0.0
            
    # 转为 Tensor
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hard_mask_t = torch.from_numpy(hard_mask).float().to(device)
    hard_mask_t = hard_mask_t.unsqueeze(0).unsqueeze(0) # [1, 1, 151]
    
    print(f"Hard Mask initialized. Banned features set to 0.")
    # ======================================================================

    ref_compounds = ["Dopa","UNC","BRE","ARI"]
    always_train_cmpds = [c for c in ref_compounds if c in unique_compounds]
    candidates = [c for c in unique_compounds if c not in always_train_cmpds]

    rs = LeaveOneOut()
    loocv_results = []

    print(f"Starting LOO-CV with Dynamic Masking...")

    for round_idx, (train_idx_cand, test_idx_cand) in enumerate(rs.split(candidates)):
        
        test_cmpd_name = candidates[test_idx_cand[0]]
        train_cmpds = [candidates[i] for i in train_idx_cand] + always_train_cmpds
        
        train_indices = [i for i, x in enumerate(full_dataset.ids) if x in train_cmpds]
        test_indices = [i for i, x in enumerate(full_dataset.ids) if x == test_cmpd_name]
        
        if not test_indices: continue

        # Z-Score
        train_labels_raw = [full_dataset.labels[i] for i in train_indices]
        y_mean = np.mean(train_labels_raw)
        y_std = np.std(train_labels_raw) + 1e-6 
        
        y_mean_t = torch.tensor(y_mean, device=device, dtype=torch.float)
        y_std_t = torch.tensor(y_std, device=device, dtype=torch.float)
        
        train_loader = DataLoader(Subset(full_dataset, train_indices), batch_size=BATCH_SIZE, shuffle=True)
        
        model = EfficiencyPredictor(input_dim=INPUT_DIM, dropout_rate=DROPOUT_RATE).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        criterion = nn.MSELoss() 

        model.train()
        for epoch in range(NUM_EPOCHS):
            for traj, label, _ in train_loader:
                traj, label = traj.to(device), label.to(device)
                
                # 【应用 Hard Mask】(先屏蔽作弊的)
                traj = traj * hard_mask_t 
                # 注意：Dynamic Mask 是在 model.forward 内部应用的
                
                target_z = (label - y_mean_t) / y_std_t

                optimizer.zero_grad()
                out = model(traj)
                pred_z = out["pred"].squeeze()
                learned_mask = out["mask"] # 获取当前学到的 mask
                
                # 1. 预测误差
                mse_loss = criterion(pred_z, target_z.squeeze())
                
                # 2. 【核心】L1 稀疏惩罚 (逼迫模型关掉不用的特征)
                # 加上一个小权重的 L1 Loss
                l1_loss = torch.mean(learned_mask) 
                
                loss = mse_loss + L1_LAMBDA * l1_loss
                
                loss.backward()
                optimizer.step()

        # 测试
        model.eval()
        slice_preds_real = []
        ground_truth = None
        
        test_loader = DataLoader(Subset(full_dataset, test_indices), batch_size=BATCH_SIZE, shuffle=False)
        
        with torch.no_grad():
            for traj, label, _ in test_loader:
                traj = traj.to(device)
                traj = traj * hard_mask_t # Hard mask
                
                out = model(traj)
                pred_z = out["pred"].detach().cpu().numpy().flatten()
                
                pred_real = (pred_z * y_std + y_mean) * 100
                slice_preds_real.extend(pred_real)
                if ground_truth is None: ground_truth = label[0].item() * 100

        if slice_preds_real:
            avg_pred = np.mean(slice_preds_real)
            avg_pred = np.clip(avg_pred, 0, 120)
            diff = avg_pred - ground_truth
            
            loocv_results.append({
                "Compound": test_cmpd_name,
                "True": ground_truth,
                "Pred": avg_pred,
                "Diff": diff
            })

    # ================= 报告 =================
    print("\n" + "="*50)
    print("LOO-CV Final Report (Dynamic Masking)")
    print("="*50)
    
    df_res = pd.DataFrame(loocv_results)
    if df_res.empty: return

    summary = df_res.groupby('Compound').agg({
        'Pred': ['mean', 'sem', 'count'],
        'True': 'first'
    }).reset_index()
    summary.columns = ['Compound', 'Pred_Mean', 'Pred_SEM', 'Count', 'True_Val']
    
    print("\n--- Compound Performance Summary ---")
    print(summary[['Compound', 'True_Val', 'Pred_Mean', 'Pred_SEM', 'Count']].to_string(index=False))

    r_pearson, p_value = stats.pearsonr(summary['True_Val'], summary['Pred_Mean'])
    rmse_agg = np.sqrt(np.mean((summary['True_Val'] - summary['Pred_Mean'])**2))
    
    print(f"\nAggregated Pearson R : {r_pearson:.4f}")
    print(f"Aggregated RMSE      : {rmse_agg:.4f}")

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
        
    plt.title(f"Loo Prediction \nPearson R = {r_pearson:.3f} | RMSE = {rmse_agg:.3f}")
    plt.xlabel("Experimental Efficacy (%)")
    plt.ylabel("Predicted Efficacy (%)")
    plt.legend()
    
    plt.savefig("efficacy_correlation_plot_sem_entropy.png", dpi=300, bbox_inches='tight')
    print("\nPlot saved to: efficacy_correlation_plot_sem_entropy.png")

    # ================= 查看模型到底选了谁 =================
    print("\nRetraining on ALL data to inspect learned mask...")
    final_loader = DataLoader(full_dataset, batch_size=BATCH_SIZE, shuffle=True)
    final_model = EfficiencyPredictor(input_dim=INPUT_DIM, dropout_rate=DROPOUT_RATE).to(device)
    optimizer = torch.optim.Adam(final_model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    
    all_labels = [l for l in full_dataset.labels]
    final_mean = np.mean(all_labels)
    final_std = np.std(all_labels)
    y_mean_t = torch.tensor(final_mean, device=device, dtype=torch.float)
    y_std_t = torch.tensor(final_std, device=device, dtype=torch.float)

    final_model.train()
    for epoch in range(NUM_EPOCHS):
        for traj, label, _ in final_loader:
            traj, label = traj.to(device), label.to(device)
            traj = traj * hard_mask_t
            target_z = (label - y_mean_t) / y_std_t
            
            optimizer.zero_grad()
            out = final_model(traj)
            loss = nn.MSELoss()(out["pred"].squeeze(), target_z.squeeze()) + L1_LAMBDA * torch.mean(out["mask"])
            loss.backward()
            optimizer.step()
            
    # 打印学到的 Mask
    final_mask = torch.sigmoid(final_model.feature_mask_logits).detach().cpu().numpy()
    
    # 1. 直接读取 BW 编号作为标签
    OBP_LABELS = config.get_list("residues.obp_residues")
    PHE_LABELS = config.get_list("residues.phe_residues")

    print(f"Loaded {len(OBP_LABELS)} OBP labels from config: {OBP_LABELS}")

    # === 动态生成 151 维特征名称 ===
    FEATURE_NAMES = []

    # A. 原子特征 (0-143): 9个原子 * (N个距离 + 2个电子)
    for i in range(9):
        # 1. 距离特征 (BW 编号)
        for bw_label in OBP_LABELS:
            FEATURE_NAMES.append(f"Atom{i}_{bw_label}_Dist") # 例如 Atom0_3.32_Dist
        
        # 2. 电子特征 (Phe 389/390 -> 6.48/6.51)
        if len(PHE_LABELS) >= 1:
            FEATURE_NAMES.append(f"Atom{i}_{PHE_LABELS[0]}_Score")
        if len(PHE_LABELS) >= 2:
            FEATURE_NAMES.append(f"Atom{i}_{PHE_LABELS[1]}_Score")

    # 1. 全局角度 (1维)
    FEATURE_NAMES.append("Global_Cos_Angle")
    
    # 2. Phe1 全局统计 (3维: Sum, Max, Norm)
    p1 = PHE_LABELS[0] if len(PHE_LABELS) > 0 else "Phe1"
    FEATURE_NAMES.extend([f"{p1}_Global_Sum", f"{p1}_Global_Max", f"{p1}_Global_Norm"])
    
    # 3. Phe2 全局统计 (3维: Sum, Max, Norm)
    p2 = PHE_LABELS[1] if len(PHE_LABELS) > 1 else "Phe2"
    FEATURE_NAMES.extend([f"{p2}_Global_Sum", f"{p2}_Global_Max", f"{p2}_Global_Norm"])
    # ==========================================================

    # 简单检查一下维度是否对齐
    if len(FEATURE_NAMES) != len(final_mask):
        print(f"[Warning] Name mismatch! Names: {len(FEATURE_NAMES)}, Mask: {len(final_mask)}")
        # 防止再次报错，补齐 Unknown
        while len(FEATURE_NAMES) < len(final_mask):
            FEATURE_NAMES.append(f"Unknown_Feat_{len(FEATURE_NAMES)}")
            print(f"Added placeholder feature: {FEATURE_NAMES[-1]}")
    # 排序并打印 Top 15
    indices = np.argsort(final_mask)[::-1]
    print("\n>>> Top 15 Features Chosen by Dynamic Mask:")
    print(f"{'Rank':<5} | {'Feature':<20} | {'Mask Value':<10}")
    print("-" * 45)
    for i in range(15):
        idx = indices[i]
        print(f"{i+1:<5} | {FEATURE_NAMES[idx]:<20} | {final_mask[idx]:.4f}")

    torch.save(final_model.state_dict(), MODEL_SAVE_PATH)
    print("\nModel saved.")

if __name__ == "__main__":
    main()