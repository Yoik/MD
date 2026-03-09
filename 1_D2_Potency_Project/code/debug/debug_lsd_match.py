#!/usr/bin/env python3
import os
import sys
import numpy as np
import MDAnalysis as mda
from rdkit import Chem
from rdkit.Chem import rdFMCS

# 确保能导入 modules
sys.path.append(".") 
try:
    from modules.qm_loader import find_ligand
except ImportError:
    print("Error: 找不到 modules.qm_loader，请确保在 1_D2_Potency_Project 目录下运行")
    sys.exit(1)

# ================= 配置 =================
TARGET_DIR = "20251216_D2_LSD_cryoEM_rebuild"
# =======================================

def get_files(folder_name):
    root = "."
    folder_path = os.path.join(root, folder_name)
    
    # 找 Ref PDB
    pdbs = [p for p in os.listdir(folder_path) if p.endswith(".pdb")]
    ref_pdb = next((p for p in pdbs if "step7" not in p and "topol" not in p and "_out" not in p and "QC" not in p), None)
    if ref_pdb: ref_pdb = os.path.join(folder_path, ref_pdb)

    # 找 MD GRO
    gros = []
    for r, d, f in os.walk(folder_path):
        for file in f:
            if file.endswith(".gro"):
                gros.append(os.path.join(r, file))
    
    target_gro = None
    for g in gros:
        if "production" in g or "npt" in g:
            target_gro = g
            break
    if not target_gro and gros: target_gro = gros[0]

    return ref_pdb, target_gro

def debug_lsd_logic():
    print(f">>> Investigating: {TARGET_DIR}")
    ref_path, gro_path = get_files(TARGET_DIR)
    
    if not ref_path or not gro_path:
        print("Missing files.")
        return

    print(f"Ref: {os.path.basename(ref_path)}")
    print(f"Gro: {os.path.basename(gro_path)}")

    # 1. 加载 Ref
    ref_mol = Chem.MolFromPDBFile(ref_path, removeHs=True, sanitize=False)
    print(f"\n[Ref PDB] Heavy Atoms: {ref_mol.GetNumHeavyAtoms()}")
    
    # 2. 模拟 qm_loader.validate_md_mapping 的加载逻辑
    print(f"\n[MD Gro] Loading using QM_Loader logic...")
    try:
        u = mda.Universe(gro_path)
        if not hasattr(u.atoms, 'elements'):
            u.add_TopologyAttr('elements')
            
        # === 复用 qm_loader 的核心函数 ===
        ligand_res = find_ligand(u)
        
        if not ligand_res:
            print("Error: find_ligand failed to find any ligand!")
            return
            
        print(f"  Found Ligand Residue: {ligand_res.resname} (ResID: {ligand_res.resid})")
        print(f"  Total Atoms in Residue: {len(ligand_res.atoms)}")
        
        sel = ligand_res.atoms
        
        # 打印一下原子明细，看看那 28 个重原子是谁
        print("\n  --- Atom Composition (Before Cleaning) ---")
        elements_count = {}
        atom_names = []
        for a in sel:
            elem = a.element.upper() if a.element else "?"
            name = a.name
            if elem not in elements_count: elements_count[elem] = 0
            elements_count[elem] += 1
            atom_names.append(f"{name}({elem})")
        
        print(f"  Elements found: {elements_count}")
        # 如果包含非重原子，这里看不出来，我们看 RDKit 转换后的
        
        # 执行元素清洗 (这是 qm_loader 里的内联逻辑，我们这里手动执行以复现)
        valid_elems = set(['H', 'C', 'N', 'O', 'S', 'F', 'P', 'CL', 'BR', 'I', 'B', 'SI', 'FE', 'ZN', 'MG', 'CA', 'NA', 'K', 'LI'])
        fixed_count = 0
        for atom in sel:
            original_elem = atom.element.upper()
            if original_elem not in valid_elems:
                name = atom.name.upper()
                guess = "".join(filter(str.isalpha, name))
                if len(guess) > 2: guess = guess[:2]
                
                if guess in valid_elems: 
                    atom.element = guess
                    fixed_count += 1
                elif len(guess) > 0 and guess[0] in valid_elems: 
                    atom.element = guess[0]
                    fixed_count += 1
                else: 
                    atom.element = 'C'
                    fixed_count += 1
        
        if fixed_count > 0:
            print(f"  [Info] Fixed {fixed_count} element assignments (imitating qm_loader).")

        # 转换为 RDKit
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".pdb", delete=False) as tmp:
            sel.write(tmp.name)
            tmp_path = tmp.name
        
        target_mol = Chem.MolFromPDBFile(tmp_path, removeHs=True, sanitize=False)
        os.remove(tmp_path)
        
        print(f"\n[MD RDKit] Heavy Atoms: {target_mol.GetNumHeavyAtoms()}")
        
        # 3. 深入比对
        # 打印 MD 分子中的所有原子名，找出多余的 4 个（LSD 应该是 ~24 个）
        print("\n  --- MD Heavy Atoms List ---")
        md_atom_names = [a.GetPDBResidueInfo().GetName().strip() for a in target_mol.GetAtoms()]
        print(f"  {md_atom_names}")
        
        # 骨架化匹配测试
        def get_skeleton(mol):
            m = Chem.Mol(mol)
            for b in m.GetBonds():
                b.SetBondType(Chem.BondType.SINGLE)
                b.SetIsAromatic(False)
            for a in m.GetAtoms():
                a.SetIsAromatic(False)
            return m

        ref_skel = get_skeleton(ref_mol)
        target_skel = get_skeleton(target_mol)
        
        if target_skel.HasSubstructMatch(ref_skel):
            print("\n  [Success] Skeleton Match Passed!")
        else:
            print("\n  [FAIL] Skeleton Match Failed.")
            # 尝试 MCS 诊断
            mcs = rdFMCS.FindMCS([ref_skel, target_skel], bondCompare=rdFMCS.BondCompare.CompareAny)
            print(f"  MCS Common Atoms: {mcs.numAtoms}")
            print(f"  Ref Atoms: {ref_skel.GetNumAtoms()}")
            print(f"  Target Atoms: {target_skel.GetNumAtoms()}")
            print("  ==> 如果 MCS < Ref，说明 Ref 的骨架连接在 MD 中被打断了。")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    debug_lsd_logic()