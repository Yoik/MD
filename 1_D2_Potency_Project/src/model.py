import torch
import torch.nn as nn
import torch.nn.functional as F

class EfficiencyPredictor(nn.Module):
    def __init__(self, input_dim):
        super(EfficiencyPredictor, self).__init__()
        
        # === 1. 特征提取器 (Feature Extractor) ===
        # 先把物理特征映射到高维空间，但不急着压缩成1个数
        self.feature_net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        
        # === 2. 效能打分头 (Score Head) ===
        # 预测该帧的“潜在效能值” (0-1)
        self.score_head = nn.Sequential(
            nn.Linear(64, 1),
            nn.Sigmoid() # 限制在 0-1
        )
        
        # === 3. 注意力门控网络 (Attention Gating) ===
        # 预测该帧的“权重/重要性” (0-1)
        # 物理含义：这一帧构象出现的概率，或者对宏观性质的贡献度
        self.attention_net = nn.Sequential(
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, 1) # 输出该帧的未归一化权重
        )

    def forward(self, x):
            # x 可能的形状: 
            # [Frames, Features] (旧的单条数据)
            # [Batch, Frames, Features] (训练时 Batch=1，Captum解释时 Batch=50)
            
            # 1. 维度统一化处理
            if x.dim() == 2:
                x = x.unsqueeze(0) # 变成 [1, Frames, Features]
                
            batch_size, num_frames, num_feats = x.size()
            
            # 2. 展平 (Flatten) --- 这是修复报错的关键！
            # 将 [Batch, Frames, Feats] 压扁成 [Batch*Frames, Feats]
            # 这样 BatchNorm1d 就会把所有帧都视为独立的样本进行归一化，而不会混淆维度
            x_flat = x.reshape(-1, num_feats)
            
            # 3. 提取特征
            feats_flat = self.feature_net(x_flat) # 输出 [Batch*Frames, 64]
            
            # 4. 还原形状
            # 变回 [Batch, Frames, 64] 以便计算注意力
            feats = feats_flat.view(batch_size, num_frames, -1)
            
            # 5. 计算分数和注意力
            scores = self.score_head(feats) # [Batch, Frames, 1]
            attn_logits = self.attention_net(feats) # [Batch, Frames, 1]
            
            # 6. 注意力归一化
            # 在 Frames 维度 (dim=1) 做 Softmax，确保每条轨迹的权重和为 1
            attn_weights = F.softmax(attn_logits, dim=1)
            
            # 7. 聚合 (Weighted Sum)
            # [Batch, Frames, 1] * [Batch, Frames, 1] -> sum -> [Batch, 1]
            macro_pred = torch.sum(scores * attn_weights, dim=1)
            
            # 8. 返回结果
            # macro_pred.squeeze(-1) 变成 [Batch]
            return macro_pred.squeeze(-1), scores, attn_weights