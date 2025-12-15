"""
modules/rdkit_matcher.py (v2.1 - Fix Virtual Sites Crash)
基于 RDKit 的子结构对齐与映射模块
修复：明确排除 MD 结构中的 Lone Pair (LP) 虚拟原子，防止 RDKit 崩溃。
"""

import os
import numpy as np
import tempfile
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolAlign

class RDKitMatcher:
    def __init__(self, ref_pdb_path):
        """
        初始化：加载参考结构 (作为 Query 子结构)
        """
        if not os.path.exists(ref_pdb_path):
            raise FileNotFoundError(f"Reference PDB not found: {ref_pdb_path}")
            
        # 1. 加载参考分子 (QM 结构)
        self.ref_mol = Chem.MolFromPDBFile(ref_pdb_path, removeHs=True, sanitize=False)
        try:
            Chem.SanitizeMol(self.ref_mol)
        except:
            print("Warning: RDKit sanitization failed for Ref PDB, using raw structure.")
            
        if self.ref_mol is None:
             raise ValueError(f"RDKit failed to load Reference PDB: {ref_pdb_path}")

        self.ref_atoms_count = self.ref_mol.GetNumAtoms()

    def match(self, md_atom_group, anchor_com=None):
        """
        核心方法：将 Reference (子结构) 映射到 MD AtomGroup (整体)
        """
        # 1. 将 MDAnalysis AtomGroup 转为 RDKit Mol
        with tempfile.NamedTemporaryFile(suffix=".pdb", delete=False) as tmp:
            # 【关键修改】排除 H 以及 LP (Lone Pair) 虚拟原子
            # name LP* : 排除 LP1, LP2 等孤对电子
            # name MW* : 排除虚拟水位点 (如适用)
            selection_str = "not (name H* or name 1H* or name 2H* or name 3H* or name LP* or name MW*)"
            
            md_heavy = md_atom_group.select_atoms(selection_str)
            
            if len(md_heavy) == 0: 
                # 如果过滤太狠导致为空，回退到原始组（虽然这可能会再次导致崩溃，但至少能提示）
                md_heavy = md_atom_group
                
            md_heavy.write(tmp.name)
            tmp_path = tmp.name
        
        # 使用 sanitize=False 防止因键级缺失导致的加载失败
        target_mol = Chem.MolFromPDBFile(tmp_path, removeHs=True, sanitize=False)
        os.remove(tmp_path)
        
        if target_mol is None:
            # print("RDKit failed to parse MD PDB structure.")
            return None, None, None

        # 2. 子结构搜索
        if target_mol.GetNumAtoms() < self.ref_atoms_count:
             return None, None, None

        try:
            matches = target_mol.GetSubstructMatches(self.ref_mol, uniquify=False)
        except Exception:
            return None, None, None
        
        if not matches:
            return None, None, None

        # 3. 筛选最佳匹配 (Best Match Selection)
        best_match_indices = None
        best_score = float('inf')
        
        ref_conf = self.ref_mol.GetConformer()
        ref_coords = ref_conf.GetPositions()
        
        target_conf = target_mol.GetConformer()
        target_coords = target_conf.GetPositions()

        for match_indices in matches:
            # --- 评分标准 A: 锚点距离 ---
            current_match_coords = target_coords[list(match_indices)]
            center = np.mean(current_match_coords, axis=0)
            
            dist_score = 0.0
            if anchor_com is not None:
                dist_score = np.linalg.norm(center - anchor_com)
                if dist_score > 3.0: 
                    continue
            
            # --- 评分标准 B: 3D 姿态差异 (RMSD) ---
            atom_map = list(zip(range(len(match_indices)), match_indices))
            try:
                # 使用 Ref 作为 Probe，Target 作为 Reference 进行对齐计算 RMSD
                rmsd = rdMolAlign.AlignMol(self.ref_mol, target_mol, atomMap=atom_map)
            except:
                # 如果 Align 失败，尝试手动计算坐标 RMSD
                # P: Ref (Centered), Q: MD_Sub (Centered)
                P = ref_coords - np.mean(ref_coords, axis=0)
                Q = current_match_coords - np.mean(current_match_coords, axis=0)
                # 简单距离差近似
                diff = P - Q
                rmsd = np.sqrt((diff * diff).sum() / len(P))
            
            final_score = rmsd + (dist_score * 0.1) 
            
            if final_score < best_score:
                best_score = final_score
                best_match_indices = match_indices

        if best_match_indices is None:
            return None, None, None

        # 4. 构建返回数据
        final_md_atoms = []
        final_cube_idxs = []
        
        for ref_idx, md_idx in enumerate(best_match_indices):
            # md_idx 是 target_mol (md_heavy) 里的索引
            atom = md_heavy[md_idx]
            final_md_atoms.append(atom)
            final_cube_idxs.append(ref_idx)
        
        import MDAnalysis
        final_atom_group = MDAnalysis.Core.groups.AtomGroup(final_md_atoms)
        
        return final_atom_group, final_cube_idxs, [a.index for a in final_md_atoms]