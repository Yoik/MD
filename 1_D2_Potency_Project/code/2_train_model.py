import sys
import os
import json
import copy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import random
from tqdm import tqdm
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import LeaveOneOut 
from scipy import stats 
from src.dataset import prepare_data, RankingDataset
from src.model import EfficiencyPredictor
from src.config import init_config
from src.analysis.feature_names import get_feature_names
from src.analysis.mask_analysis import collapse_mask, save_collapsed_csv
from src.utils.logger import Logger
from src.utils.seed import seed_everything
from src.analysis.mask_trainer import train_final_mask_model



# ================= 配置参数 =================
config = init_config()

LABEL_FILE = config.get_path("paths.label_file") #··标签文件路径
RESULT_DIR = config.get_path("paths.result_dir") #··结果输出目录
MODEL_SAVE_PATH = config.get_path("paths.model_path") #··模型保存路径
SCALER_SAVE_PATH = config.get_path("paths.scaler_path") #··标准化器保存路径

REF_FEATURE_PATH = config.get_path("paths.ref_feature_path") #··参考特征路径

POCKET_ATOM_NUM = config.get_int("data.pocket_atom_num") #··口袋原子数量
INPUT_DIM = config.get_int("data.input_dim_features") #··输入特征维度
LEARNING_RATE = config.get_float("training.learning_rate") #··学习率
DROPOUT_RATE = config.get_float("training.dropout_rate") #··Dropout 比例
WEIGHT_DECAY = config.get_float("training.weight_decay") #··权重衰减
NUM_EPOCHS = config.get_int("training.num_epochs") #··训练轮数 
BATCH_SIZE = config.get_int("training.batch_size") #··批量大小
seed = config.get_int("misc.random_seed") #··随机种子
# 【新增】稀疏惩罚系数
# 值越大，模型删特征越狠；值越小，模型保留特征越多
# 建议 0.001 - 0.005
L1_LAMBDA = config.get_float("training.l1_lambda") #··L1 稀疏惩罚系数

# ================= 创建输出目录 =================
from datetime import datetime 

FEATURE_DIR = RESULT_DIR
# 创建带日期时间的输出目录，例如 2025-12-27_23-50-12 
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") 
OUTPUT_DIR = os.path.join(RESULT_DIR, timestamp) 
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 这一行执行后，所有的 print() 都会同时出现在屏幕和 log.txt 里
sys.stdout = Logger(os.path.join(OUTPUT_DIR, "training_log.txt"))

# ================= 设置随机种子 =================

def main():
    seed_everything(seed)
    # ================= 数据准备 =================
    print("Preparing data...")
    try:
        train_ds, test_ds = prepare_data(
            label_file=LABEL_FILE, 
            result_dir=RESULT_DIR, 
            ref_feature_path=REF_FEATURE_PATH,
            pocket_atom_num=POCKET_ATOM_NUM, 
            save_scaler_path=SCALER_SAVE_PATH,
            window_size=100, stride=20
        )
    except Exception as e:
        print(f"[DATA ERROR] {e}"); return

    # ================= LOO-CV =================
    all_query_feats = train_ds.query_feats + test_ds.query_feats
    all_query_labels = train_ds.query_labels + test_ds.query_labels
    all_query_ids = train_ds.query_ids + test_ds.query_ids
    
    shared_ref_feats_by_pose = train_ds.ref_feats_by_pose

    # Reference 现在是按 pose 分组的多个 tensor
    shared_ref_tensors = train_ds.ref_tensors
    shared_ref_pose_ids = train_ds.ref_pose_ids
    print(f"Reference poses: {shared_ref_pose_ids}")

    # 将 Reference 按 pose 存储为字典，方便后续随机采样    
    full_ids = all_query_ids
    unique_compounds = sorted(list(set(full_ids)))
    print(f"Total Compounds: {len(unique_compounds)}")

    # ================= Masking =================

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ======================================================================

    ref_compounds = ["Dopa"]
    always_train_cmpds = [c for c in ref_compounds if c in unique_compounds]
    candidates = [c for c in unique_compounds if c not in always_train_cmpds]

    rs = LeaveOneOut()
    loocv_results = []

    all_fold_losses = {} # 记录每一轮的 Loss 曲线

    print(f"Starting LOO-CV with Dynamic Masking...")

    total_rounds = len(list(rs.split(candidates))) # 预计算总轮数
    splitter = rs.split(candidates) # 重新创建生成器

    for round_idx, (train_idx_cand, test_idx_cand) in enumerate(splitter):
        
        test_cmpd_name = candidates[test_idx_cand[0]]
        # 打印当前测试化合物名称
        print(f"\n>>> Round {round_idx+1} / {len(candidates)} | Testing: {test_cmpd_name}")

        current_fold_losses = [] # 记录当前轮的 Loss 曲线

        train_cmpds = [candidates[i] for i in train_idx_cand] + always_train_cmpds
        train_indices = [i for i, x in enumerate(full_ids) if x in train_cmpds]
        test_indices = [i for i, x in enumerate(full_ids) if x == test_cmpd_name]
        
        if not test_indices: continue

        current_fold_losses = {'train': [], 'test': []}

        # 1. 准备训练集
        curr_train_ds = RankingDataset(
            query_feats=[all_query_feats[i] for i in train_indices],
            query_labels=[all_query_labels[i] for i in train_indices],
            query_ids=[all_query_ids[i] for i in train_indices],
            ref_feats_by_pose=shared_ref_feats_by_pose,
            ref_value=1.0 
        )
        train_loader = DataLoader(curr_train_ds, batch_size=BATCH_SIZE, shuffle=True)
        
        # 2. 准备测试集 Tensor (用于计算 Test Loss)
        _tmp_test_ds = RankingDataset(
            query_feats=[all_query_feats[i] for i in test_indices],
            query_labels=[all_query_labels[i] for i in test_indices],
            query_ids=[all_query_ids[i] for i in test_indices],
            ref_feats_by_pose=shared_ref_feats_by_pose, 
            ref_value=1.0
        )
        test_query_tensor = torch.from_numpy(np.stack(_tmp_test_ds.query_feats)).float().to(device)
        # 准备测试集的 Label
        _test_raw_labels = torch.tensor(_tmp_test_ds.query_labels, dtype=torch.float32).to(device)
        test_rank_target = torch.where(
            _test_raw_labels >= (1.0 - 0.05),
            torch.tensor(1.0).to(device),
            torch.tensor(-1.0).to(device)
        )

        # 3. [修改] 仅预加载 Tensor，不要在这里计算分数！
        # all_ref_tensor = curr_train_ds.ref_tensor.to(device)
        all_ref_tensors = [t.to(device) for t in curr_train_ds.ref_tensors]

        model = EfficiencyPredictor(input_dim=INPUT_DIM, dropout_rate=DROPOUT_RATE).to(device)

        # Adam 优化器
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        criterion = nn.MarginRankingLoss(margin=0.3)        
        # ================ 训练 =================

        # ================= 学习率衰减调度器 =================
        # 每 20 个 Epoch 学习率减半 (60个Epoch会衰减2次: 20, 40)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

        # ================= 初始化最佳记录变量 =================
        best_loss = float('inf')
        best_model_wts = copy.deepcopy(model.state_dict())
        # ====================================================

        model.train()
        epoch_bar = tqdm(range(NUM_EPOCHS), desc=f"Training {test_cmpd_name}", leave=False)

        for epoch in epoch_bar:
            # --- 1. 训练阶段 (Training) ---
            model.train() 
            batch_losses = [] 
            
            # [重要修改] 删掉这里的 with torch.no_grad()... 代码块
            # Ref 的计算必须进下面的循环，才能有梯度！

            for batch in train_loader:
                q_traj = batch['query_feat'].to(device)
                rank_label = batch['rank_label'].to(device)

                optimizer.zero_grad()
                
                # [方案一核心]：在循环内计算 Reference，且带有梯度！
                # 这会让计算量变大 (回到了吃显存的状态)，但数学逻辑是正确的
                ref_means = []
                for ref_t in all_ref_tensors:
                    out_ref = model.forward_one(ref_t)
                    ref_means.append(torch.mean(out_ref["pred"]))
                score_ref_mean = torch.stack(ref_means).mean()

                # B. 计算 Query
                out_q = model.forward_one(q_traj)
                score_q = out_q["pred"].view(-1)
                
                # C. Loss 计算
                # 此时 score_ref_mean 是带有梯度的，模型可以去优化它！
                score_ref_batch = score_ref_mean.expand_as(score_q)
                mask_loss = torch.mean(out_q["mask"]) 
                
                rank_loss = criterion(score_q, score_ref_batch, rank_label)
                loss = rank_loss + L1_LAMBDA * mask_loss                
                
                loss.backward()
                optimizer.step()
                batch_losses.append(loss.item())

            avg_train_loss = np.mean(batch_losses)

            # --- 2. 验证阶段 (Test Loss) ---
            model.eval() 
            with torch.no_grad():
                # 计算 Test Query
                out_test_q = model.forward_one(test_query_tensor)
                score_test_q = out_test_q["pred"].view(-1)
                
                # [为了严谨] 在 Eval 模式下重新算一遍 Ref 基准分
                # 因为 Eval 模式下 Dropout 行为不同，分数会有微小差异
                ref_means_eval = []
                for ref_t in all_ref_tensors:
                    out_ref_eval = model.forward_one(ref_t)
                    ref_means_eval.append(torch.mean(out_ref_eval["pred"]))
                score_ref_eval = torch.stack(ref_means_eval).mean()

                score_ref_test = score_ref_eval.expand_as(score_test_q)
                
                test_mask_loss = torch.mean(out_test_q['mask'])
                
                # 计算 Loss
                test_loss = criterion(score_test_q, score_ref_test, test_rank_target) \
                            + L1_LAMBDA * test_mask_loss
                
                val_loss_val = test_loss.item()

                # ================= [新增 3] 学习率衰减 & 保存最佳模型 =================
                # B. 检查是否是最佳 Loss
                if val_loss_val < best_loss:
                    best_loss = val_loss_val
                    # 深拷贝当前这一刻的模型参数，暂存在内存中
                    best_model_wts = copy.deepcopy(model.state_dict())
            # A. 更新学习率 (每个 Epoch 结束时调用)
            scheduler.step()

            # --- 3. 记录与显示 ---
            current_fold_losses['train'].append(avg_train_loss)
            current_fold_losses['test'].append(val_loss_val)
            
            epoch_bar.set_postfix({
                'T_loss': f"{avg_train_loss:.4f}", 
                'V_loss': f"{val_loss_val:.4f}"
            })

        all_fold_losses[test_cmpd_name] = current_fold_losses # 保存当前轮的 Loss 曲线

        # ================= 恢复最佳模型权重 =================
        model.load_state_dict(best_model_wts)
        print(f"  -> Best Test Loss: {best_loss:.4f} (Restored weights)")
        # ================= 测试 (Relative Scoring) =================
        model.eval()
        diff_scores = []
        ground_truth_eff = None

        curr_test_ds = RankingDataset(
            query_feats=[all_query_feats[i] for i in test_indices],
            query_labels=[all_query_labels[i] for i in test_indices],
            query_ids=[all_query_ids[i] for i in test_indices],
            ref_feats_by_pose=shared_ref_feats_by_pose,
            ref_value=1.0
        )
        
        test_loader = DataLoader(curr_test_ds, batch_size=BATCH_SIZE, shuffle=False)
                
        # 先在测试开始前算好一个 eval 模式下的 reference score（宏观）
        with torch.no_grad():
            ref_means_test = []
            for ref_t in all_ref_tensors:
                out_ref = model.forward_one(ref_t)
                ref_means_test.append(torch.mean(out_ref["pred"]))
            score_ref_test_scalar = torch.stack(ref_means_test).mean().item()

        for batch in test_loader:
            q_traj = batch['query_feat'].to(device)
            out_q = model.forward_one(q_traj)
            s_q = out_q["pred"].detach().cpu().numpy().flatten()

            diff = s_q - score_ref_test_scalar
            diff_scores.extend(diff)

            if ground_truth_eff is None:
                ground_truth_eff = batch['query_score'][0].item() * 100
        # 根据 diff_scores 计算最终的预测类别
        if diff_scores:
            avg_diff = np.mean(diff_scores)
            if avg_diff > 0.05: pred_cls = "Stronger/Equal"
            elif avg_diff < -0.05: pred_cls = "Weaker"
            else: pred_cls = "Similar"
            
            # 真实的相对类别 (Reference = 100%)
            if ground_truth_eff >= 90: true_cls = "Stronger/Equal"
            elif ground_truth_eff <= 70: true_cls = "Weaker" # 假设70以下算弱
            else: true_cls = "Similar"
            
            loocv_results.append({
                "Compound": test_cmpd_name,
                "True_Eff": ground_truth_eff,
                "Pred_Diff_Score": avg_diff,
                "Pred_Class": pred_cls,
                "True_Class": true_cls,
                "Correct": 1 if pred_cls == true_cls else 0
            })

    # ================= 结果保存 =================
    print("\n" + "="*50)
    print("LOO-CV Final Report (Ranking Mode)")
    print("="*50)
    
    if loocv_results:
        # 1. 保存 CSV
        df_res = pd.DataFrame(loocv_results)
        csv_path = os.path.join(OUTPUT_DIR, "loocv_results.csv")
        df_res.to_csv(csv_path, index=False)
        print(f"[Save] Results saved to: {csv_path}")

        # 聚合结果
        summary = df_res.groupby('Compound').agg({
            'Pred_Diff_Score': 'mean',
            'Correct': 'first'
        }).reset_index()

        print("\n--- Compound Performance Summary ---")
        print(summary.sort_values(by='Pred_Diff_Score', ascending=False).to_string(index=False))

        # 2. 保存 Loss 曲线 JSON
        json_path = os.path.join(OUTPUT_DIR, "loocv_losses.json")
        with open(json_path, 'w') as f:
            json.dump(all_fold_losses, f)
        print(f"[Save] Loss history saved to: {json_path}")

    # ================= 查看模型到底选了谁 =================
    print("\nRetraining on ALL data to inspect learned mask...")

    final_model, final_mask = train_final_mask_model(
        all_query_feats=all_query_feats,
        all_query_labels=all_query_labels,
        all_query_ids=all_query_ids,
        shared_ref_feats_by_pose=shared_ref_feats_by_pose,
        shared_ref_tensors=shared_ref_tensors,
        device=device,
        input_dim=INPUT_DIM,
        dropout_rate=DROPOUT_RATE,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        num_epochs=NUM_EPOCHS,
        l1_lambda=L1_LAMBDA
    )

    FEATURE_NAMES = get_feature_names()

    assert len(FEATURE_NAMES) == len(final_mask), \
        f"Feature name mismatch: {len(FEATURE_NAMES)} vs {len(final_mask)}"

    collapsed_rows = collapse_mask(FEATURE_NAMES, final_mask, agg="mean")

    print("\n>>> Top 15 Feature types by mask (collapsed):")
    print(f"{'Rank':<5} | {'Feature':<25} | {'Mask Value':<10} | {'Count':<5}")
    print("-" * 65)
    for i, (feat, val, cnt) in enumerate(collapsed_rows[:15], start=1):
        print(f"{i:<5} | {feat:<25} | {val:<10.4f} | {cnt:<5}")

    collapsed_csv_path = os.path.join(OUTPUT_DIR, "feature_importance_collapsed.csv")
    save_collapsed_csv(collapsed_rows, collapsed_csv_path)
            
    # =============================================================
    torch.save(final_model.state_dict(), MODEL_SAVE_PATH)
    print("\nModel saved.")

if __name__ == "__main__":
    main()