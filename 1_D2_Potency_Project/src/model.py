import torch
import torch.nn as nn
import torch.nn.functional as F

class EfficiencyPredictor(nn.Module):
    def __init__(self, input_dim, attn_dropout=0.2):
        super(EfficiencyPredictor, self).__init__()
        
        self.geom_dim = 13
        self.elec_dim = 6
        self.input_dim = input_dim 

        # 1. 定义时序卷积 (用于捕捉动态过程)
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(in_channels=input_dim, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(32)
        )
        
        # Attention 输入维度 = 原始特征(19) + 时序特征(32)
        self.attn_input_dim = self.input_dim + 32  
        self.frame_scale = nn.Parameter(torch.tensor(3.0)) # 保持之前的高初始值

        # ============================================================
        # 【核心修改 1】：角色互换
        # ============================================================
        
        # A. 几何流 (Geometry Stream) -> 降级为 "Position Gate" (位置门控)
        # 变浅、变窄。迫使它只学 "在不在口袋里" 这种简单逻辑。
        self.geom_gate = nn.Sequential(
            nn.Linear(self.geom_dim, 32), # 简单映射
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()  # 输出 0~1 的概率：Is present?
        )
        
        # B. 电子流 (Electronic Stream) -> 升级为 "Efficacy Scorer" (效能打分)
        # 变深、变宽。让它承担拟合药效的核心任务。
        self.elec_net = nn.Sequential(
            nn.Linear(self.elec_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid() # 输出 0~1 的强度：How strong?
        )
        
        # C. 初始化 (体现你的假设)
        # 几何门控：初始化为 "半开" (0.5)，因为它只是个条件
        nn.init.normal_(self.geom_gate[-2].weight, mean=0.0, std=0.1)
        nn.init.constant_(self.geom_gate[-2].bias, 0.0)
        
        # 电子效能：初始化为 "高" (2.0 -> Sigmoid ≈ 0.88)
        # 因为我们假设是活性模拟，只要几何位置对了，电子大概率是好的
        nn.init.constant_(self.elec_net[-2].bias, 2.0)


        # ============================================================
        # 注意力机制 (保持不变，或者让它更关注几何)
        # ============================================================
        self.attn_net = nn.Sequential(
            nn.Linear(self.attn_input_dim, 32),
            nn.Tanh(),
            nn.Dropout(p=attn_dropout),
            nn.Linear(32, 1)
        )

    def forward(self, x, compound_idx=None):
        # x shape: [Batch, Frames, 19]
        batch_size, num_frames, _ = x.shape
        
        # 1. 拆分特征
        x_geom = x[:, :, :13]
        x_elec = x[:, :, 13:]
        
        x_geom_flat = x_geom.reshape(-1, self.geom_dim)
        x_elec_flat = x_elec.reshape(-1, self.elec_dim)

        # 2. 计算两条流
        # Geom -> Gate (位置因子)
        pos_gate = self.geom_gate(x_geom_flat).view(batch_size, num_frames, 1)
        
        # Elec -> Score (效能因子)
        elec_score = self.elec_net(x_elec_flat).view(batch_size, num_frames, 1)
        
        # 3. 物理融合：最终分数 = 基础系数 * 位置因子 * 电子效能
        # 逻辑：如果位置不对(pos_gate->0)，电子再好也没用；如果位置对了，看电子好不好。
        frame_scores = self.frame_scale * pos_gate * elec_score
        
        # 4. 时序与注意力 (和之前保持一致)
        x_permuted = x.permute(0, 2, 1)
        temporal_feat = self.temporal_conv(x_permuted).permute(0, 2, 1)
        x_attn_combined = torch.cat([x, temporal_feat], dim=2)
        
        x_attn_flat = x_attn_combined.reshape(-1, self.attn_input_dim)
        attn_logits = self.attn_net(x_attn_flat).view(batch_size, num_frames, 1)
        attn_weights = F.softmax(attn_logits, dim=1)
        
        # 5. 加权求和
        attn_sum = torch.sum(frame_scores * attn_weights, dim=1)
        macro_pred = attn_sum.squeeze(-1)

        return {
            "pred": macro_pred,
            "frame_scores": frame_scores,
            "gate": pos_gate,       # 这里返回的是位置门控
            "elec": elec_score,     # 这里返回的是电子得分，方便后续分析
            "attn": attn_weights
        }