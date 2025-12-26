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
        # 原子部分：9个原子，每个原子6维特征 -> 54维
        # 全局部分：7维（1个角度 + 6个电子特征） -> 7维
        self.n_atoms = 9
        self.atom_feat_dim = 6
        self.global_feat_dim = 8

        self.atom_hidden_dim = 64 
        # DeepSets 内部隐藏维度 = 64(SUM) + 64(MAX) = 128维
        self.ring_dim = self.atom_hidden_dim * 2

        # 最终输入维度 = 原子集合编码(144) + 全局特征(7) = 151维
        self.frame_dim = self.ring_dim + self.global_feat_dim

        # === 【核心修改】动态特征掩码 (Dynamic Feature Mask) ===
        # 初始化为 0.5 (sigmoid(0) = 0.5)，表示“不确定是否有用”
        # 模型会自己学习把它推向 1 (有用) 或 0 (无用)
        self.atom_mask_logits = nn.Parameter(torch.ones(self.atom_feat_dim) * 2.0)

        self.global_mask_logits = nn.Parameter(torch.ones(self.global_feat_dim) * 2.0)
        
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
            #nn.Sigmoid()  # 输出范围 [0, 1]
        )

        # === 3. 时间注意力 模块 ===
        self.attn_net = nn.Sequential(
            nn.Linear(self.frame_dim, 32),
            nn.Tanh(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(32, 1)
        )

    def forward_one(self, x):
        # x input shape: [Batch, Frames, 9*6 + 7] = [B, F, 61] (假设输入是拼接好的)
        batch_size, num_frames, _ = x.shape
        
        # ============================================================
        # 4. 【核心修改】先拆分，再 Mask
        # ============================================================
        
        # --- A. 拆分特征 ---
        # 算出原子特征的总长度: 9 * 6 = 54
        num_atom_feats = self.n_atoms * self.atom_feat_dim
        
        # 切片：原子部分
        x_atoms_flat = x[:, :, :num_atom_feats] # [B, F, 54]
        # 切片：全局部分
        x_global = x[:, :, num_atom_feats:]     # [B, F, 7]
        
        # --- B. 变形原子特征 ---
        # [B, F, 54] -> [B, F, 9, 6]
        # 这样我们才能对最后一维(6个特征)应用共享 Mask
        x_atoms = x_atoms_flat.view(batch_size, num_frames, self.n_atoms, self.atom_feat_dim)
        
        # --- C. 应用共享 Mask (Broadcasting) ---
        # 生成 Mask (0~1)
        atom_mask = torch.sigmoid(self.atom_mask_logits)     # [6]
        global_mask = torch.sigmoid(self.global_mask_logits) # [7]
        
        # 广播乘法：
        # atom_mask [6] 会自动扩充为 [1, 1, 1, 6] 应用到每个原子
        x_atoms_masked = x_atoms * atom_mask.view(1, 1, 1, -1)
        
        # global_mask [7] 扩充为 [1, 1, 7]
        x_global_masked = x_global * global_mask.view(1, 1, -1)
        
        # ============================================================
        # 5. 进入网络骨架（DeepSets 聚合）
        # ============================================================
        # 这里的输入 x_atoms_masked 已经是“加权”过的了
        # 但它依然是无序的集合，DeepSets 会公平处理
        ring_feats = self.atom_encoder(x_atoms_masked) # Output: [B, F, 128]

        # 拼接全局特征
        frame_feats = torch.cat([ring_feats, x_global_masked], dim=-1) # [B, F, 135]
        
        # --- 后续预测逻辑不变 ---
        frame_scores = self.frame_scorer(frame_feats)
        attn_logits = self.attn_net(frame_feats)
        attn_weights = F.softmax(attn_logits, dim=1)
        
        weighted_scores = frame_scores * attn_weights
        macro_pred = torch.sum(weighted_scores, dim=1).squeeze(-1)
        
        # 为了兼容训练代码，构造 mask 返回值
        # 我们把两个 mask 拼起来返回，方便你看 feature importance
        # 注意：这里需要把 atom_mask 复制 9 份拼回去，才能对齐原来的名字列表
        full_atom_mask = atom_mask.repeat(self.n_atoms) # [54]
        full_mask = torch.cat([full_atom_mask, global_mask], dim=0) # [61]
        
        return {
            "pred": macro_pred,
            "mask": full_mask, # 这样你的 feature_importance.csv 代码依然能跑
            "frame_scores": frame_scores,
            "attn_weights": attn_weights
        }
    # ==========================================================================
    ### 主 forward 支持双输入
    # ==========================================================================
    def forward(self, x, x_ref=None):
        """
        x: Query 化合物特征
        x_ref: (可选) Reference 化合物特征。如果提供，则执行孪生网络模式。
        """
        # 1. 计算 Query 的结果
        out_query = self.forward_one(x)
        
        # 2. 如果提供了 Ref
        if x_ref is not None:
            # 检查维度：如果是 4 维 [Batch, N_Refs, Frames, Feats]，说明是“全量模式”
            if x_ref.dim() == 4:
                batch_size, n_refs, frames, feats = x_ref.shape
                
                # A. 展平为大 Batch 进行并行计算
                # [B * N, F, D]
                x_ref_flat = x_ref.view(-1, frames, feats)
                
                # B. 计算所有片段的分数
                out_ref_flat = self.forward_one(x_ref_flat)
                scores_flat = out_ref_flat['pred'] # [B * N]
                
                # C. 变回形状并取平均
                scores_grouped = scores_flat.view(batch_size, n_refs)
                score_ref_mean = scores_grouped.mean(dim=1) # [B] -> 得到平均基准分
                
                # 构造一个类似的返回字典 (mask 等取第一个即可)
                out_ref = {
                    "pred": score_ref_mean, # 平均后的分数
                    "mask": out_ref_flat['mask'] # mask 是共享参数，一样的
                }
                
                return out_query, out_ref
                
            else:
                # 兼容旧的一对一模式
                out_ref = self.forward_one(x_ref)
                return out_query, out_ref        
        # 3. 只有 Query (推理模式)
        return out_query