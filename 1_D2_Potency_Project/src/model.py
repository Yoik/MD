import torch
import torch.nn as nn
import torch.nn.functional as F

class EfficiencyPredictor(nn.Module):
    def __init__(self, input_dim):
        super(EfficiencyPredictor, self).__init__()
        
        # 19维特征：0-12几何，13-18电子
        self.geom_dim = 13
        self.elec_dim = 6
        self.input_dim = input_dim # 19
        
        # === 1. 亲和力通道 (Geometry Stream) ===
        # 判断结合姿态 (0-1)
        self.geom_net = nn.Sequential(
            nn.Linear(self.geom_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid() 
        )
        
        # === 2. 电子效能门控 (Electronic Gate) ===
        # 判断激活开关 (0-1)
        self.elec_gate = nn.Sequential(
            nn.Linear(self.elec_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid() 
        )

        # === 3. 时间注意力机制 (Temporal Attention) ===
        # 判断这一帧的重要性 (权重)
        # 输入是全量特征 (19维)，因为它需要综合判断
        self.attn_net = nn.Sequential(
            nn.Linear(self.input_dim, 32),
            nn.Tanh(), # Tanh 通常用于 Attention 之前的激活
            nn.Linear(32, 1) # 输出 logit，后面接 Softmax
        )
        
    def forward(self, x):
        # x shape: [Batch, Frames, 19]
        batch_size, num_frames, _ = x.shape
        
        # 1. 拆分特征
        x_geom = x[:, :, :13]   # 几何
        x_elec = x[:, :, 13:]   # 电子
        
        # 展平处理
        x_geom_flat = x_geom.reshape(-1, self.geom_dim)
        x_elec_flat = x_elec.reshape(-1, self.elec_dim)
        x_flat = x.reshape(-1, self.input_dim) # Attention 用全量
        
        # 2. 计算各流分数
        # Geom: [Batch, Frames, 1]
        geom_score = self.geom_net(x_geom_flat).view(batch_size, num_frames, 1)
        # Gate: [Batch, Frames, 1]
        gate_val = self.elec_gate(x_elec_flat).view(batch_size, num_frames, 1)
        
        # 3. 计算每一帧的瞬时效能
        # Frame_Score = 几何分 * 电子门控
        frame_scores = geom_score * gate_val
        
        # 4. 计算注意力权重
        # Attn_Logits: [Batch, Frames, 1]
        attn_logits = self.attn_net(x_flat).view(batch_size, num_frames, 1)
        # Softmax over 'Frames' dimension (dim=1)
        # 这样每一条轨迹的所有帧权重加起来等于 1.0
        attn_weights = F.softmax(attn_logits, dim=1)
        
        # 5. 加权求和 (Weighted Sum Pooling)
        # 以前是 Mean()，现在是 Sum( Score * Weight )
        weighted_scores = frame_scores * attn_weights
        macro_pred = torch.sum(weighted_scores, dim=1).squeeze(-1)
        
        # 返回 4 个值：最终预测, 逐帧分数, 门控值, 注意力权重
        return macro_pred, frame_scores, gate_val, attn_weights