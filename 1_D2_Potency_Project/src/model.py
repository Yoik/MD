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
        # x: [Frames, Features] (Batch size = 1)
        x = x.squeeze(0) 
        
        # 1. 提取高维特征
        feats = self.feature_net(x) # [Frames, 64]
        
        # 2. 计算单帧效能分 (Microscopic Efficacy)
        scores = self.score_head(feats) # [Frames, 1]
        
        # 3. 计算注意力权重 (Attention Weights)
        attn_logits = self.attention_net(feats) # [Frames, 1]
        
        # 使用 Softmax 归一化权重，使所有帧的权重之和为 1
        # 这样就变成了“加权平均”，而不是简单平均
        attn_weights = F.softmax(attn_logits, dim=0) 
        
        # 4. 聚合 (Weighted Sum)
        # 宏观效能 = Σ (单帧效能 * 该帧权重)
        macro_pred = torch.sum(scores * attn_weights)
        
        return macro_pred, scores, attn_weights