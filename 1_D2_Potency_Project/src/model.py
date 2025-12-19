import torch
import torch.nn as nn
import torch.nn.functional as F

class AtomSetEncoder(nn.Module):
    """
    DeepSets 模块：处理变长/无序的原子集合
    输入: [Batch, Frames, N_atoms, N_feats]
    输出: [Batch, Frames, Hidden_Dim * 2] (Sum + Max)
    """
    def __init__(self, atom_in_dim, hidden_dim):
        super(AtomSetEncoder, self).__init__()
        
        # === 单原子评委 (Shared MLP) ===
        # 对每个原子独立打分，学习 "位置 + 电子" 的综合价值
        self.atom_mlp = nn.Sequential(
            nn.Linear(atom_in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim), # 小batch下 LayerNorm 更稳
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
    def forward(self, x):
        # x: [Batch, Frames, 9, 14]
        b, f, n, d = x.shape
        
        # 合并 Batch 和 Frames 维度以便并行处理
        x_flat = x.view(b * f, n, d) 
        
        # 1. 局部处理: 给每个原子打分
        # atom_feats: [B*F, 9, Hidden]
        atom_feats = self.atom_mlp(x_flat)
        
        # 2. 全局聚合 (Permutation Invariant)
        # Sum Pooling: 感知 "数量" (有多少好原子)
        pool_sum = torch.sum(atom_feats, dim=1) # [B*F, Hidden]
        # Max Pooling: 感知 "质量" (最好的那个有多好)
        pool_max = torch.max(atom_feats, dim=1)[0] # [B*F, Hidden]
        
        # 3. 融合
        ring_repr = torch.cat([pool_sum, pool_max], dim=1) # [B*F, Hidden*2]
        
        # 恢复维度
        ring_repr = ring_repr.view(b, f, -1)
        return ring_repr
    
class EfficiencyPredictor(nn.Module):
    def __init__(self, input_dim = 133, dropout_rate=0.2):
        super(EfficiencyPredictor, self).__init__()
        
        # 配置参数
        # 原子部分：9个原子，每个原子16维特征 -> 144维
        # 全局部分：7维（1个角度 + 6个电子特征） -> 7维
        self.n_atoms = 9
        self.atom_feat_dim = 16
        self.global_feat_dim = 7

        self.atom_hidden_dim = 64 
        # DeepSets 内部隐藏维度 = 64(SUM) + 64(MAX) = 128维
        self.ring_dim = self.atom_hidden_dim * 2

        # 最终输入维度 = 原子集合编码(144) + 全局特征(7) = 151维
        self.frame_dim = self.ring_dim + self.global_feat_dim
        
        # === 1. 原子集合编码器 ===
        self.atom_encoder = AtomSetEncoder(self.atom_feat_dim, self.atom_hidden_dim)

        # === 2. 帧效能评估器 ===
        self.frame_scorer = nn.Sequential(
            nn.Linear(self.frame_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()  # 输出范围 [0, 1]
        )

        # === 3. 时间注意力 模块 ===
        self.attn_net = nn.Sequential(
            nn.Linear(self.frame_dim, 32),
            nn.Tanh(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        # x shape: [Batch, Frames, 133]
        batch_size, num_frames, _ = x.shape
        
        # --- A. 拆分特征 ---
        # 1. 拆分特征
        x_atoms_flat = x[:, :, :self.n_atoms * self.atom_feat_dim]
        x_global = x[:, :, self.n_atoms * self.atom_feat_dim:]
        
        # 2. 重塑原子特征形状: [B, F, 9, 14]
        x_atoms = x_atoms_flat.view(batch_size, num_frames, self.n_atoms, self.atom_feat_dim)

        # --- B. 提取单帧特征 ---
        # 1. 通过DeepSets编码原子集合
        ring_feats = self.atom_encoder(x_atoms)  # [B, F, 128]

        # 2. 拼接全局特征
        frame_feats = torch.cat([ring_feats, x_global], dim=-1)  #
        
        # --- C. 计算分数与注意力 ---
        # 1. 每一帧的效能分数 (0-1)
        # 代表：这一帧的构象+电子状态，理论上能激活受体吗？
        frame_scores = self.frame_scorer(frame_feats) # [B, F, 1]
        
        # 2. 每一帧的注意力权重
        # 代表：这一帧是噪音还是关键帧？
        attn_logits = self.attn_net(frame_feats) # [B, F, 1]
        attn_weights = F.softmax(attn_logits, dim=1) # 在时间轴归一化
        
        # --- D. 宏观聚合 ---
        # 加权求和: Sum(Score * Weight)
        weighted_scores = frame_scores * attn_weights
        macro_pred = torch.sum(weighted_scores, dim=1).squeeze(-1) # [Batch]
        
        # 为了兼容之前的训练代码，返回 4 个值
        # gate_val 这里已经融入 frame_scores 了，返回 frame_scores 作为替代
        return {
            "pred": macro_pred,           # 对应 out["pred"]
            "frame_scores": frame_scores, # 对应 out["frame_scores"]
            "gate": frame_scores,         # 对应 out["gate"] (在DeepSets架构中，gate已融入frame_scores)
            "attn": attn_weights          # 对应 out["attn"]
        }