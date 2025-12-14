"""
sequence_aligner.py
序列对齐和残基偏移计算模块
"""

from Bio import Align


class OffsetCalculator:
    """
    使用 Biopython 进行序列对齐，计算模拟轨迹中的残基与标准序列的偏移量
    """
    
    def __init__(self, standard_seq_str):
        """
        初始化对齐计算器
        
        Args:
            standard_seq_str: 标准蛋白质序列（单字母或多字母代码）
        """
        # 移除空格和换行符
        self.ref_seq = "".join(standard_seq_str.split())
        
        # 三字母到单字母的转换表
        self.three_to_one = {
            'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
            'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'HSD': 'H', 
            'HSE': 'H', 'HSP': 'H',
            'ILE': 'I', 'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F',
            'PRO': 'P', 'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V'
        }

    def get_sim_sequence(self, u):
        """
        从 MDAnalysis Universe 对象提取模拟轨迹的蛋白质序列
        
        Args:
            u: MDAnalysis Universe 对象
            
        Returns:
            tuple: (序列字符串, 残基 ID 列表)
        """
        protein = u.select_atoms("protein and name CA")
        resnames = protein.resnames
        resids = protein.resids
        
        seq_str = ""
        valid_indices = []
        
        for i, res in enumerate(resnames):
            code = self.three_to_one.get(res, 'X')
            seq_str += code
            valid_indices.append(resids[i])
        
        return seq_str, valid_indices

    def calculate_offset(self, u, target_std_id):
        """
        计算标准序列中的残基在模拟序列中的偏移量
        
        Args:
            u: MDAnalysis Universe 对象
            target_std_id: 标准序列中的目标残基号（通常从 1 开始）
            
        Returns:
            int: 偏移量 (sim_resid - std_resid)，如果无法对齐则返回 None
        """
        sim_seq, sim_resids = self.get_sim_sequence(u)
        
        # 设置对齐参数
        aligner = Align.PairwiseAligner()
        aligner.mode = 'global'
        aligner.match_score = 2
        aligner.mismatch_score = -1
        aligner.open_gap_score = -0.5
        aligner.extend_gap_score = -0.1
        
        # 执行对齐
        alignments = aligner.align(self.ref_seq, sim_seq)
        best_aln = alignments[0]
        
        # 构建映射：标准序列残基号 -> 模拟序列残基号
        mapping = {}
        aligned_ref = best_aln.aligned[0]
        aligned_sim = best_aln.aligned[1]
        
        for (r_start, r_end), (s_start, s_end) in zip(aligned_ref, aligned_sim):
            length = r_end - r_start
            
            for i in range(length):
                r_idx = r_start + i
                s_idx = s_start + i
                std_res_num = r_idx + 1
                
                if s_idx < len(sim_resids):
                    sim_resid_val = sim_resids[s_idx]
                    mapping[std_res_num] = sim_resid_val
        
        # 查找目标残基的偏移量
        if target_std_id in mapping:
            found_sim_resid = mapping[target_std_id]
            offset = found_sim_resid - target_std_id
            print(f"     [Align Info] Std {target_std_id} aligns to Sim {found_sim_resid}. Offset = {offset}")
            return offset
        else:
            print(f"     [Align Error] Residue {target_std_id} is a GAP in the simulation!")
            return None
