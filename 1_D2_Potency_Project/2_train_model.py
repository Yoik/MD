import sys
import os
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
NUM_EPOCHS = 40      # 推荐 40-50
BATCH_SIZE = 32
N_SPLITS = 20        # 20 轮交叉验证
TEST_SIZE = 2        # 每次随机选 2 个化合物做测试 (Dopa 除外)

# ==============================================================================
# 主程序
# ==============================================================================
def main():
    # 1. 准备数据
    print("Preparing data with slicing...")
    try:
        # 修正：完整传入所有必要参数
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
        return

    # 2. 整合全量数据 (用于自定义划分)
    full_features = train_ds.features + test_ds.features
    full_labels = train_ds.labels + test_ds.labels
    full_ids = train_ds.ids + test_ds.ids
    full_dataset = TrajectoryDataset(full_features, full_labels, full_ids)
    
    # 获取所有化合物名称
    all_unique_ids = sorted(list(set(full_dataset.ids)))
    print(f"\n>>> Dataset Ready. Total Compounds: {len(all_unique_ids)}")
    
    # === 3. 核心逻辑：锁定 Dopa ===
    target_anchors = ["Dopa", "ARI"]
    
    # 检查这些锚点是否都在数据里
    valid_anchors = [a for a in target_anchors if a in all_unique_ids]
    
    if len(valid_anchors) > 0:
        # 候选池：剔除所有锚点
        candidates = [c for c in all_unique_ids if c not in valid_anchors]
        # 锚点：永远进入训练集
        always_train_cmpds = valid_anchors
        print(f"[Fixed Anchor] {valid_anchors} are LOCKED in the training set (Range Defined).")
    else:
        candidates = all_unique_ids
        always_train_cmpds = []
        print(f"[Warning] Anchors not found. Running standard random split.")    
    unique_compounds = np.array(candidates)
    
    # 定义划分器 (对剩下的候选者进行抽签)
    rs = ShuffleSplit(n_splits=N_SPLITS, test_size=TEST_SIZE, random_state=42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"MCCV Strategy: {N_SPLITS} rounds | Device: {device}")
    
    # 存储结果容器
    compound_results = {name: [] for name in all_unique_ids}
    compound_truths = {name: 0.0 for name in all_unique_ids}

    # === 4. 开始 MCCV 循环 ===
    for round_idx, (train_idx_cand, test_idx_cand) in enumerate(rs.split(unique_compounds)):
        # 训练集 = 本轮抽到的候选者 + 锁定的 Dopa
        train_cmpds = list(unique_compounds[train_idx_cand]) + always_train_cmpds
        # 测试集 = 本轮抽到的测试者 (绝对没有 Dopa)
        test_cmpds = list(unique_compounds[test_idx_cand])
        
        print(f"\n--- Round {round_idx+1}/{N_SPLITS} ---")
        # print(f"  Train: {train_cmpds}") # 调试用，平时可注释
        # print(f"  Test : {test_cmpds}")

        # 根据 ID 找切片索引
        train_indices = [i for i, x in enumerate(full_dataset.ids) if x in train_cmpds]
        test_indices = [i for i, x in enumerate(full_dataset.ids) if x in test_cmpds]
        
        # 构建 DataLoader
        train_loader = DataLoader(Subset(full_dataset, train_indices), batch_size=BATCH_SIZE, shuffle=True)
        test_loader = DataLoader(Subset(full_dataset, test_indices), batch_size=BATCH_SIZE, shuffle=False)

        # 初始化模型
        model = EfficiencyPredictor(input_dim=INPUT_DIM).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        criterion = nn.MSELoss()

        # 4.1 训练 (Train)
        model.train()
        for epoch in range(NUM_EPOCHS):
            for traj, label, _ in train_loader:
                traj, label = traj.to(device), label.to(device)
                optimizer.zero_grad()
                
                # 【重要】接收 4 个返回值 (pred, scores, gate, attn)
                pred, _, _, _ = model(traj)
                
                loss = criterion(pred.squeeze(), label.squeeze())
                loss.backward()
                optimizer.step()

        # 4.2 测试 (Test)
        model.eval()
        with torch.no_grad():
            # 临时存本轮的预测，防止同一个化合物有多个切片
            round_preds_map = {name: [] for name in test_cmpds}
            
            for traj, label, ids in test_loader:
                traj = traj.to(device)
                pred, _, _, _ = model(traj) # 接收 4 个返回值
                
                preds_np = pred.squeeze().cpu().numpy().flatten()
                labels_np = label.cpu().numpy().flatten()
                
                for p, l, name in zip(preds_np, labels_np, ids):
                    round_preds_map[name].append(p)
                    compound_truths[name] = l # 记录真实值

            # 输出本轮结果
            for name in test_cmpds:
                if round_preds_map[name]:
                    avg_val = np.mean(round_preds_map[name])
                    compound_results[name].append(avg_val)
                    print(f"  Test: {name:<5} | True: {compound_truths[name]:.2f} | Pred: {avg_val:.2f}")

    # === 5. 最终汇总报告 (严格按照你的格式要求) ===
    print("\n" + "="*70)
    print("MCCV Final Average Report")
    print("="*70)
    # 表头包含 Tested N times
    print(f"{'Compound':<10} | {'True':<8} | {'Avg Pred':<8} | {'Std Dev':<8} | {'Diff':<8} | {'Tested N times'}")
    print("-" * 70)
    
    final_stats_for_rmse = []
    
    for name in all_unique_ids:
        true_val = compound_truths[name]
        preds = compound_results[name]
        count = len(preds)
        
        if count > 0:
            avg_pred = np.mean(preds)
            std_dev = np.std(preds)
            diff = avg_pred - true_val
            
            print(f"{name:<10} | {true_val:<8.2f} | {avg_pred:<8.2f} | {std_dev:<8.2f} | {diff:<+8.2f} | {count}")
            
            final_stats_for_rmse.append({"True": true_val, "Pred": avg_pred})
        else:
            # 对于 Dopa (Anchor)，它没有测试数据
            if name in target_anchors:
                 print(f"{name:<10} | {true_val:<8.2f} | {'(Anchor)':<8} | {'0.00':<8} | {'----':<8} | {0} (Train Only)")
            else:
                 print(f"{name:<10} | {true_val:<8.2f} | {'N/A':<8} | {'0.00':<8} | {'----':<8} | {0}")

    # 计算整体指标
    if final_stats_for_rmse:
        df_res = pd.DataFrame(final_stats_for_rmse)
        rmse = np.sqrt(np.mean((df_res['True'] - df_res['Pred'])**2))
        corr = df_res['True'].corr(df_res['Pred'])
        print("-" * 70)
        print(f"Overall RMSE : {rmse:.4f}")
        print(f"Pearson R    : {corr:.4f}")

    # === 6. 全量重训并保存 (Retrain Final Model) ===
    print("\nRetraining Final Model on ALL data...")
    # 使用全量数据 (包含 Dopa)
    final_loader = DataLoader(full_dataset, batch_size=BATCH_SIZE, shuffle=True)
    final_model = EfficiencyPredictor(input_dim=INPUT_DIM).to(device)
    optimizer = torch.optim.Adam(final_model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    
    final_model.train()
    for epoch in range(NUM_EPOCHS):
        for traj, label, _ in final_loader:
            traj, label = traj.to(device), label.to(device)
            optimizer.zero_grad()
            # 同样接收 4 个返回值
            pred, _, _, _ = final_model(traj)
            loss = nn.MSELoss()(pred.squeeze(), label.squeeze())
            loss.backward()
            optimizer.step()
            
    torch.save(final_model.state_dict(), MODEL_SAVE_PATH)
    print(f"Final model saved to: {MODEL_SAVE_PATH}")
    print("Done.")

if __name__ == "__main__":
    main()