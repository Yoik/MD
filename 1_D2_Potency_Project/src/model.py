import torch
import torch.nn as nn
import torch.nn.functional as F

class EfficiencyPredictor(nn.Module):
    def __init__(self, input_dim, num_compounds, embed_dim=8, attn_dropout=0.2):
        super(EfficiencyPredictor, self).__init__()
        
        # 19维特征：0-12几何，13-18电子
        self.geom_dim = 13
        self.elec_dim = 6
        self.input_dim = input_dim # 19

        self.num_compounds = num_compounds
        self.embed_dim = embed_dim
        self.attn_input_dim = self.input_dim + self.embed_dim

        self.compound_embedding = nn.Embedding(num_compounds, embed_dim)
        self.frame_scale = nn.Parameter(torch.tensor(1.0))

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
            nn.Linear(self.attn_input_dim, 32),
            nn.Tanh(),
            nn.Dropout(p=attn_dropout),
            nn.Linear(32, 1)
        )

        # 多尺度聚合融合：输入 2 个标量（attn_sum / topk）
        self.agg_fuse = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )
        self.topk_k = 5

        
    def forward(self, x, compound_idx):
        # x shape: [Batch, Frames, 19]
        # compound_ids shape: [Batch]
        # x shape: [Batch, Frames, 19]
        batch_size, num_frames, _ = x.shape
        
        # 1. 拆分特征
        x_geom = x[:, :, :13]   # 几何
        x_elec = x[:, :, 13:]   # 电子
        
        # 展平处理
        x_geom_flat = x_geom.reshape(-1, self.geom_dim)
        x_elec_flat = x_elec.reshape(-1, self.elec_dim)

        # compound_idx: [B]
        # embed: [B, embed_dim]
        embed = self.compound_embedding(compound_idx)

        # expand to frames: [B, T, embed_dim]
        embed_expanded = embed.unsqueeze(1).expand(batch_size, num_frames, self.embed_dim)

        # concat for attention input: [B, T, input_dim + embed_dim]
        x_attn = torch.cat([x, embed_expanded], dim=-1)

        # flatten for attn_net
        x_attn_flat = x_attn.reshape(-1, self.attn_input_dim)
        
        # 2. 计算各流分数
        # Geom: [Batch, Frames, 1]
        geom_score = self.geom_net(x_geom_flat).view(batch_size, num_frames, 1)
        # Gate: [Batch, Frames, 1]
        gate_val = self.elec_gate(x_elec_flat).view(batch_size, num_frames, 1)
        
        # 3. 计算每一帧的瞬时效能
        # Frame_Score = 几何分 * 电子门控
        frame_scores = self.frame_scale * geom_score * gate_val
        
        # 4. 计算注意力权重
        # Attn_Logits: [Batch, Frames, 1]
        attn_logits = self.attn_net(x_attn_flat).view(batch_size, num_frames, 1)
        attn_weights = F.softmax(attn_logits, dim=1)
        
        # 5. 加权求和 (Weighted Sum Pooling)
        # 1) attention 加权和
        attn_sum = torch.sum(frame_scores * attn_weights, dim=1)  # [B, 1]
        
        # 2) mean pooling
        # mean_pool = torch.mean(frame_scores, dim=1)               # [B, 1]

        # 3) top-k pooling（按 frame_scores 选）
        k = min(self.topk_k, num_frames)
        topk_vals, _ = torch.topk(frame_scores.squeeze(-1), k=k, dim=1)  # [B, k]
        # topk_pool = topk_vals.mean(dim=1, keepdim=True)                   # [B, 1]

        # 融合
        # agg_feat = torch.cat([attn_sum, topk_pool], dim=1)     # [B, 3]
        # macro_pred = self.agg_fuse(agg_feat).squeeze(-1)                  # [B]
        macro_pred = attn_sum.squeeze(-1)

        traj_mean = frame_scores.mean(dim=1)

        topk_mean = topk_vals.mean(dim=1, keepdim=True)

        # 返回 4 个值：最终预测, 逐帧分数, 门控值, 注意力权重
        return {
            "pred": macro_pred,
            "frame_scores": frame_scores,
            "gate": gate_val,
            "attn": attn_weights
        }
