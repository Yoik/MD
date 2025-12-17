import sys
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from src.dataset import prepare_data, TrajectoryDataset
from src.model import EfficiencyPredictor

# --- 配置参数 ---
LABEL_FILE = "data/labels.csv"
RESULT_DIR = "data/features"
MODEL_SAVE_PATH = "saved_models/best_model.pth"
SCALER_SAVE_PATH = "saved_models/scaler.pkl"

# 物理参数配置
POCKET_ATOM_NUM = 12
INPUT_DIM = 21

# 训练超参数 (针对切片后的数据调整)
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-4   # 增加正则化，防止过拟合
NUM_EPOCHS = 50       # 每轮 LOOCV 训练的轮数 (数据多了，轮数可以少一点)
BATCH_SIZE = 32       # 切片后数据量大了，可以使用更大的 Batch

def main():
    # 1. 准备数据 (带切片)
    # 注意：prepare_data 默认会按 8:2 切分，但为了做 LOOCV，我们会把它们合并回来
    print("Preparing data with slicing...")
    try:
        # window_size=200, stride=50 意味着每200帧切一段，每隔50帧取样一次
        train_ds_fixed, test_ds_fixed = prepare_data(
            LABEL_FILE, RESULT_DIR, POCKET_ATOM_NUM, SCALER_SAVE_PATH,
            window_size=200, stride=50
        )
    except ValueError as e:
        print(f"Data Error: {e}")
        return

    # 2. 合并数据集以进行 LOOCV
    # 我们需要拿回所有的特征、标签和ID，自己控制划分
    full_features = train_ds_fixed.features + test_ds_fixed.features
    full_labels = train_ds_fixed.labels + test_ds_fixed.labels
    full_ids = train_ds_fixed.ids + test_ds_fixed.ids
    
    full_dataset = TrajectoryDataset(full_features, full_labels, full_ids)
    
    # 获取所有唯一的化合物 ID
    all_ids = full_dataset.ids
    unique_compounds = sorted(list(set(all_ids)))
    
    print(f"\n>>> Dataset Ready.")
    print(f"Total Slices: {len(full_dataset)}")
    print(f"Total Compounds: {len(unique_compounds)}")
    print("-" * 50)
    print(f"Starting LOOCV (Leave-One-Out Cross Validation)...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    loocv_results = [] # 存储每一轮的结果

    # ==========================================
    # 3. LOOCV 循环
    # ==========================================
    for i, test_cmpd in enumerate(unique_compounds):
        print(f"\n[{i+1}/{len(unique_compounds)}] Testing on Compound: {test_cmpd}")
        
        # 3.1 动态划分训练/测试索引
        # 凡是 ID 不等于当前测试化合物的，都做训练
        train_indices = [j for j, x in enumerate(all_ids) if x != test_cmpd]
        test_indices = [j for j, x in enumerate(all_ids) if x == test_cmpd]
        
        # 创建 Subset
        train_sub = Subset(full_dataset, train_indices)
        test_sub = Subset(full_dataset, test_indices)
        
        # DataLoader
        train_loader = DataLoader(train_sub, batch_size=BATCH_SIZE, shuffle=True)
        # 测试集 Batch 设大一点没关系，只是为了预测
        test_loader = DataLoader(test_sub, batch_size=BATCH_SIZE, shuffle=False)
        
        # 3.2 初始化模型 (每一轮都要重置模型权重！)
        model = EfficiencyPredictor(input_dim=INPUT_DIM).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        criterion = nn.MSELoss()
        
        # 3.3 训练当前 Fold
        model.train()
        for epoch in range(NUM_EPOCHS):
            epoch_loss = 0
            for traj, label, _ in train_loader:
                traj, label = traj.to(device), label.to(device)
                
                optimizer.zero_grad()
                pred, _, _ = model(traj)
                loss = criterion(pred.squeeze(), label.squeeze())
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            
            # (可选) 打印训练进度，为了版面整洁这里就不每轮都打印了
            # if (epoch+1) % 10 == 0:
            #     print(f"    Epoch {epoch+1}: Loss {epoch_loss/len(train_loader):.4f}")

        # 3.4 测试当前 Fold
        model.eval()
        fold_preds = []
        fold_truth = 0.0
        
        with torch.no_grad():
            for traj, label, _ in test_loader:
                traj, label = traj.to(device), label.to(device)
                pred, _, _ = model(traj)
                
                # 收集预测值 (可能有多个切片)
                fold_preds.extend(pred.squeeze().cpu().numpy().flatten())
                # 记录真实标签 (所有切片的标签应该是一样的)
                if len(label) > 0:
                    fold_truth = label[0].item()
        
        # 计算该化合物的平均预测值
        avg_pred = np.mean(fold_preds)
        
        # 还原到 0-100 量级便于观察
        truth_100 = fold_truth * 100
        pred_100 = avg_pred * 100
        diff = pred_100 - truth_100
        
        print(f"  -> True: {truth_100:.2f} | Pred: {pred_100:.2f} | Diff: {diff:.2f}")
        
        loocv_results.append({
            "Compound": test_cmpd,
            "True": truth_100,
            "Pred": pred_100,
            "Diff": diff
        })

    # ==========================================
    # 4. 汇总报告
    # ==========================================
    print("\n" + "="*50)
    print("LOOCV Final Report")
    print("="*50)
    df_res = pd.DataFrame(loocv_results)
    
    # 计算整体指标
    rmse = np.sqrt(np.mean(df_res['Diff']**2))
    mae = np.mean(np.abs(df_res['Diff']))
    corr = df_res['True'].corr(df_res['Pred'])
    
    print(df_res.to_string(index=False, float_format="%.2f"))
    print("-" * 50)
    print(f"Overall RMSE : {rmse:.4f}")
    print(f"Overall MAE  : {mae:.4f}")
    print(f"Pearson R    : {corr:.4f}")
    
    # 保存 LOOCV 结果
    df_res.to_csv("loocv_results.csv", index=False)
    print(f"Detailed results saved to 'loocv_results.csv'")

    # ==========================================
    # 5. 最终全量训练 (Retrain on All Data)
    # ==========================================
    print("\n" + "="*50)
    print("Retraining Final Model on ALL data...")
    print("="*50)
    
    final_loader = DataLoader(full_dataset, batch_size=BATCH_SIZE, shuffle=True)
    final_model = EfficiencyPredictor(input_dim=INPUT_DIM).to(device)
    final_optim = torch.optim.Adam(final_model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    final_criterion = nn.MSELoss()
    
    final_model.train()
    for epoch in range(NUM_EPOCHS):
        total_loss = 0
        for traj, label, _ in final_loader:
            traj, label = traj.to(device), label.to(device)
            final_optim.zero_grad()
            pred, _, _ = final_model(traj)
            loss = final_criterion(pred.squeeze(), label.squeeze())
            loss.backward()
            final_optim.step()
            total_loss += loss.item()
            
        if (epoch+1) % 10 == 0:
            print(f"Epoch {epoch+1}/{NUM_EPOCHS} | Loss: {total_loss/len(final_loader):.5f}")

    # 保存最终模型
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    torch.save(final_model.state_dict(), MODEL_SAVE_PATH)
    print(f"Final model saved to: {MODEL_SAVE_PATH}")
    print("Done.")

if __name__ == "__main__":
    main()