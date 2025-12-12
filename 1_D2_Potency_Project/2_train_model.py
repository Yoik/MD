import sys
import os
# 1. 强制将当前目录加入路径，解决 ModuleNotFoundError
sys.path.append(os.getcwd())

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.dataset import prepare_data
from src.model import EfficiencyPredictor

# --- 配置参数 ---
LABEL_FILE = "data/labels.csv"
RESULT_DIR = "data/features"
MODEL_SAVE_PATH = "saved_models/best_model.pth"
SCALER_SAVE_PATH = "saved_models/scaler.pkl"

# 物理参数配置
POCKET_ATOM_NUM = 12  # 您定义的口袋几何距离特征数量 (根据实际npy列数调整)
INPUT_DIM = 19        # 总特征维度 (距离 + 角度 + 电子统计量)

# 训练超参数
LEARNING_RATE = 0.001
NUM_EPOCHS = 100      # 增加轮数，因为我们有了 Scheduler 防止过拟合
STEP_SIZE = 30        # 每 30 轮衰减一次学习率
GAMMA = 0.1           # 衰减倍率 (lr = lr * 0.1)

def main():
    # 1. 准备数据
    print("Preparing data...")
    try:
        # 这一步会自动处理标签归一化(0-1)和特征标准化
        train_ds, test_ds = prepare_data(LABEL_FILE, RESULT_DIR, POCKET_ATOM_NUM, SCALER_SAVE_PATH)
    except ValueError as e:
        print(f"Data Error: {e}")
        return

    # Batch Size 设为 1 (处理变长轨迹)
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)
    
    # 2. 初始化模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = EfficiencyPredictor(input_dim=INPUT_DIM).to(device)
    
    # 定义优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # 【新增】定义学习率调度器 (防止后期震荡)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=STEP_SIZE, gamma=GAMMA)
    
    # 定义损失函数
    criterion = nn.MSELoss()
    
    # 3. 训练循环
    best_test_loss = float('inf')
    
    print(f"{'Epoch':^5} | {'Train MSE':^10} | {'Test MSE':^10} | {'LR':^8} | {'Note'}")
    print("-" * 55)
    
    for epoch in range(NUM_EPOCHS):
        # === 训练阶段 ===
        model.train()
        train_loss_sum = 0
        
        for traj, label, _ in train_loader:
            traj, label = traj.to(device), label.to(device)
            
            optimizer.zero_grad()
            
            # 模型返回: (macro_pred, scores, attn_weights)
            # 我们只需要第一个 macro_pred 参与 Loss 计算
            pred, _, _ = model(traj) 
            
            loss = criterion(pred.squeeze(), label.squeeze())
            loss.backward()
            optimizer.step()
            
            train_loss_sum += loss.item()
            
        avg_train_loss = train_loss_sum / len(train_loader)
        
        # === 验证阶段 (新增) ===
        model.eval()
        test_loss_sum = 0
        with torch.no_grad():
            for traj, label, _ in test_loader:
                traj, label = traj.to(device), label.to(device)
                
                pred, _, _ = model(traj)
                loss = criterion(pred.squeeze(), label.squeeze())
                test_loss_sum += loss.item()
                
        avg_test_loss = test_loss_sum / len(test_loader)
        
        # === 学习率更新 ===
        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step()
        
        # === 保存最佳模型 (基于 Test Loss) ===
        save_msg = ""
        if avg_test_loss < best_test_loss:
            best_test_loss = avg_test_loss
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            save_msg = "*" # 标记这一轮保存了模型
            
        print(f"{epoch+1:^5} | {avg_train_loss:.5f}    | {avg_test_loss:.5f}   | {current_lr:.1e} | {save_msg}")
            
    print("-" * 55)
    print(f"Training finished.")
    print(f"Best Test MSE: {best_test_loss:.5f}")
    print(f"Model saved to: {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    # 确保保存目录存在
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    main()