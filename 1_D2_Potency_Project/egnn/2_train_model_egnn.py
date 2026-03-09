import os
import sys
import torch
import torch.nn as nn
import numpy as np
import pandas as pd

from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
from torch_geometric.data import Batch
from tqdm import tqdm

# 引入项目模块
try:
    from src.config import init_config
    from src.dataset import MolGraphDataset, PairwiseGraphDataset, get_pairwise_loader
    from src.model import DeltaEGNN
except ImportError as e:
    print(f"Error: 模块导入失败: {e}")
    sys.exit(1)

# ================= 1. 加载配置 =================
config = init_config()
RESULT_DIR = config.get_path("paths.result_dir")
LABEL_FILE = config.get_path("paths.label_file")

# 训练超参
LR = config.get_float("training.learning_rate", 5e-4)
EPOCHS = config.get_int("training.num_epochs", 60)
BATCH_SIZE = config.get_int("training.batch_size", 64)
ENSEMBLE_RUNS = 5  # 固定 5 折系综

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed(seed):
    """确保可复现性"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

def train_one_epoch(model, loader, optimizer):
    model.train()
    total_loss = 0
    count = 0
    loss_fn = nn.MSELoss()

    for batch_a, batch_b, delta_y, _, _ in loader:
        batch_a = batch_a.to(DEVICE)
        batch_b = batch_b.to(DEVICE)
        delta_y = delta_y.to(DEVICE)
        
        optimizer.zero_grad()
        
        # Siamese Network
        pred_a, pred_b = model(batch_a, batch_b)
        
        # === [修复点] ===
        # pred_a - pred_b 的形状是 [Batch, 1]
        # delta_y 的形状是 [Batch] -> 必须变成 [Batch, 1] 才能正确计算 MSE
        loss = loss_fn(pred_a - pred_b, delta_y.view(-1, 1))
        # ===============
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item() * delta_y.size(0)
        count += delta_y.size(0)
        
    return total_loss / count if count > 0 else 0
# ================= [关键] 分布均值评估逻辑 =================

def compute_mean_score(model, frames_list, batch_size=64):
    """
    计算一个化合物所有帧的平均绝对分数。
    这消除了单帧随机采样的方差，反映物理稳态效能。
    """
    model.eval()
    scores = []
    
    # 批量推理以加速
    for i in range(0, len(frames_list), batch_size):
        batch_frames = frames_list[i : i+batch_size]
        batch = Batch.from_data_list(batch_frames).to(DEVICE)
        
        with torch.no_grad():
            # 技巧：DeltaEGNN 也是 Siamese，我们输入 (batch, batch)
            # 输出 (score, score)，我们只取第一个
            pred_score, _ = model(batch, batch)
            scores.extend(pred_score.cpu().numpy().flatten())
            
    return np.mean(scores)

def evaluate_loo_distribution(models, test_cmpd, train_cmpds, base_dataset):
    """
    使用【分布均值】策略进行评估
    Pred = True_Ref + (Mean_Score_Test - Mean_Score_Ref)
    """
    # 1. 获取测试化合物的所有帧
    # 假设 base_dataset.data_map 存储了 {cmpd_name: [Data, Data...]}
    # 如果 dataset 实现不同，需相应调整
    test_frames = base_dataset.data_map[test_cmpd]
    
    ensemble_final_preds = []
    
    # 2. 遍历 5 个模型
    for model in models:
        # A. 算出测试化合物的"物理稳态分"
        score_test = compute_mean_score(model, test_frames)
        
        # B. 锚定推理 (Anchored Inference)
        # 用训练集中每个化合物作为 Reference，推导出一个预测值，然后取平均
        fold_preds = []
        for ref_cmpd in train_cmpds:
            ref_frames = base_dataset.data_map[ref_cmpd]
            true_ref = base_dataset.label_map[ref_cmpd]
            
            # 算出参考化合物的"物理稳态分"
            score_ref = compute_mean_score(model, ref_frames)
            
            # 核心公式
            delta = score_test - score_ref
            pred = true_ref + delta
            fold_preds.append(pred)
            
        # 当前模型给出的最终预测 (平均了所有锚点)
        ensemble_final_preds.append(np.mean(fold_preds))
        
    # 3. 返回 5 个模型预测值的平均
    return np.mean(ensemble_final_preds), base_dataset.label_map[test_cmpd]

# ================= 主流程 =================

def main():
    print(f"Loading Dataset from {RESULT_DIR}...")
    if not os.path.exists(LABEL_FILE):
        print(f"Error: Label file not found at {LABEL_FILE}")
        return

    base_ds = MolGraphDataset(RESULT_DIR, LABEL_FILE)
    all_cmpds = sorted(base_ds.get_compounds())
    print(f"Compounds ({len(all_cmpds)}): {all_cmpds}")

    if len(all_cmpds) < 2: return

    final_results = []
    
    # === Phase 1: LOO-CV (验证模式) ===
    # 目的：评估模型性能 (RMSE, R, Acc)
    # 策略：不保存模型到硬盘，只在内存中流转
    print(f"\n{'='*10} Phase 1: LOO Cross-Validation (Distribution Mean) {'='*10}")
    
    for i, test_cmpd in enumerate(tqdm(all_cmpds, desc="LOO CV")):
        print(f"\n>>> LOO Round {i+1}/{len(all_cmpds)}: Testing {test_cmpd}")
        
        # 划分数据集
        train_cmpds = [c for c in all_cmpds if c != test_cmpd]
        train_ds = PairwiseGraphDataset(base_ds, compound_list=train_cmpds, mode='train')
        train_loader = get_pairwise_loader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
        
        current_round_models = []
        
        # 训练 5 个系综模型
        for run in tqdm(range(ENSEMBLE_RUNS), desc=" Ensemble", leave=False):
            set_seed(42 + run)
            model = DeltaEGNN(config).to(DEVICE)
            
            # 架构检查 (只在第一个跑的时候打印)
            if i == 0 and run == 0:
                print(f"  [Arch Check] Layers: {model.n_layers}")
                if hasattr(model, 'att_pool'):
                    print("  [Arch Check] WARN: Attention Pooling detected.")
                else:
                    print("  [Arch Check] Pooling: Mean Pooling (Correct).")

            optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
            
            # 训练循环
            for epoch in tqdm(range(EPOCHS), desc=" Epochs", leave=False):
                loss = train_one_epoch(model, train_loader, optimizer)
                scheduler.step(loss)
            
            # 保存到内存列表
            current_round_models.append(model)
        
        # 评估 (使用分布均值策略)
        pred_val, true_val = evaluate_loo_distribution(current_round_models, test_cmpd, train_cmpds, base_ds)
        
        diff = pred_val - true_val
        print(f"  -> Result: True={true_val:.4f}, Pred={pred_val:.4f}, Diff={diff:+.4f}")
        final_results.append({"Compound": test_cmpd, "True": true_val, "Pred": pred_val})

    # === Phase 2: Production (生产模式) ===
    # 目的：生成最终可用的模型文件
    # 策略：使用全量数据重新训练，并保存权重
    print(f"\n{'='*10} Phase 2: Finalizing Production Models (All Data) {'='*10}")
    
    full_train_ds = PairwiseGraphDataset(base_ds, compound_list=all_cmpds, mode='train')
    full_loader = get_pairwise_loader(full_train_ds, batch_size=BATCH_SIZE, shuffle=True)
    
    for run in range(ENSEMBLE_RUNS):
        print(f"Training Production Model {run+1}/{ENSEMBLE_RUNS}...")
        set_seed(42 + run)
        model = DeltaEGNN(config).to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
        
        for epoch in range(EPOCHS):
            train_one_epoch(model, full_loader, optimizer)
            
        # 保存权重
        save_name = f"model_ensemble_{run}.pth"
        save_path = os.path.join(RESULT_DIR, save_name)
        torch.save(model.state_dict(), save_path)
        print(f"  [Saved] {save_path}")

    # === Phase 3: Final Report ===
    print(f"\n{'='*10} Final LOO-CV Report {'='*10}")
    df = pd.DataFrame(final_results)
    
    rmse = np.sqrt(mean_squared_error(df["True"], df["Pred"]))
    p_corr, _ = pearsonr(df["True"], df["Pred"])
    
    # 计算 Pairwise Accuracy
    n_correct = 0; n_total = 0
    vals = df.to_dict('records')
    for i in range(len(vals)):
        for j in range(i+1, len(vals)):
            if np.sign(vals[i]["True"] - vals[j]["True"]) == np.sign(vals[i]["Pred"] - vals[j]["Pred"]):
                n_correct += 1
            n_total += 1
    
    acc = n_correct / n_total if n_total > 0 else 0
    
    print(df)
    print("-" * 30)
    print(f"RMSE: {rmse:.4f}")
    print(f"Pearson R: {p_corr:.4f}")
    print(f"Pairwise Accuracy: {acc:.2%}")
    
    df.to_csv(os.path.join(RESULT_DIR, "loo_results_final.csv"), index=False)

if __name__ == "__main__":
    main()