#!/usr/bin/env python3
"""
1_1_extract_cub_features.py (v4.0 - RDKit Integration)

QC 工具（核心驱动脚本）：
1. 自动寻找 Dopa 目录计算全局最大积分值 (Global Max)，统一物理标准。
2. 调用 qm_loader 使用 RDKit 进行基于子结构的拓扑对齐。
3. 生成两个关键 QC 文件：
   - {CID}_QC_REF.pdb: 带有归一化 ELF 权重的参考结构。
   - {CID}_QC_MD_MAP.pdb: 映射到 MD 坐标系并带有权重的结构 (用于验证对齐是否准确)。
"""

import os
import glob
import numpy as np
import sys

# 导入模块
try:
    from modules.qm_loader import load_cube_and_map, save_qc_structure, validate_md_mapping
    from modules.cube_parser import CubeParser
except ImportError as e:
    print(f"Error: 无法导入必要模块。请确保 'modules' 文件夹完整且包含 updated qm_loader.py。")
    print(f"Details: {e}")
    sys.exit(1)

# ================= 配置区 =================
INTEGRATION_RADIUS = 1.5
OUTPUT_DIR = "./data/qc_structures"
ROOT_DIR = "." 
# ==========================================

def get_dopa_global_max(root_dir):
    """
    遍历目录寻找含有 'dopa' 的文件夹，计算其最大积分值作为全局标准。
    使用了体积修正后的物理积分。
    """
    print(">>> Searching for Dopa reference (Global Normalization Standard)...")
    all_dirs = glob.glob(os.path.join(root_dir, "*"))
    
    # 优先找名字里带 dopa 的目录
    for c_dir in all_dirs:
        if not os.path.isdir(c_dir): continue
        dirname = os.path.basename(c_dir).lower()
        
        if "dopa" in dirname:
            cubs = glob.glob(os.path.join(c_dir, "*.cub"))
            if cubs:
                print(f"    Found Dopa dir: {os.path.basename(c_dir)}")
                try:
                    cp = CubeParser(cubs[0])
                    # 获取积分 (CubeParser 已包含体积修正)
                    integrals = cp.get_carbon_integrals(INTEGRATION_RADIUS)
                    if len(integrals) > 0:
                        g_max = np.max(integrals)
                        print(f"    [Global Ref] Dopa Max Integral = {g_max:.4f}")
                        return g_max
                except Exception as e:
                    print(f"    [Error] Reading Dopa cube failed: {e}")
    
    print("    [WARN] Dopa reference not found! Using 1.0 as default (Normalization might be inconsistent).")
    return 1.0

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    
    # 1. 获取全局归一化标准 (Dopa Max)
    # 这一步非常重要，确保所有分子的 B-factor 都在同一个标尺上
    GLOBAL_MAX = get_dopa_global_max(ROOT_DIR)
        
    print(f"\n>>> Starting QM Feature Extraction & RDKit Mapping Check")
    print(f">>> Normalization Factor: {GLOBAL_MAX:.4f}")
    print(f">>> Output Directory:     {OUTPUT_DIR}\n")
    
    all_dirs = glob.glob(os.path.join(ROOT_DIR, "*"))
    all_dirs.sort()
    
    success_count = 0
    fail_count = 0
    
    for c_dir in all_dirs:
        if not os.path.isdir(c_dir): continue
        # 排除非数据目录
        if any(x in c_dir for x in ["modules", "results", "__pycache__", "data", "efficacy_models", "saved_models"]): continue
        
        cid = os.path.basename(c_dir)
        
        # 寻找输入文件
        cubs = glob.glob(os.path.join(c_dir, "*.cub"))
        if not cubs: continue
        
        # 寻找 Ref PDB
        pdbs = glob.glob(os.path.join(c_dir, "*.pdb"))
        # 排除 MD 生成的中间文件
        ref_pdb = next((p for p in pdbs if "step7" not in p and "topol" not in p and "_out" not in p and "QC" not in p), None)
        
        if not ref_pdb:
            print(f"[Skip] {cid}: No reference PDB found (required for QM mapping).")
            continue
            
        print(f"Processing: {cid}")
        
        # 2. 加载 QM 数据 & 建立映射关系
        # 注意：这里调用的是 modules/qm_loader.py 里的函数
        # 它内部会读取 PDB 路径，供后续 RDKit 使用
        qm_data = load_cube_and_map(cubs[0], ref_pdb, INTEGRATION_RADIUS)
        
        if qm_data:
            # 3. 保存 QC REF (基于 QM 坐标)
            # 这是一个单纯的数值检查文件
            ref_qc_filename = f"{cid}_QC_REF.pdb"
            ref_qc_path = os.path.join(OUTPUT_DIR, ref_qc_filename)
            save_qc_structure(ref_qc_path, qm_data, normalize=True, global_max=GLOBAL_MAX)
            
            # 4. 验证 MD 映射 (基于 RDKit 对齐)
            # 寻找目标 GRO/TPR 文件
            gros = glob.glob(os.path.join(c_dir, "**", "*.gro"), recursive=True)
            target_gro = None
            
            # 优先找 production 或 npt，因为它们代表平衡后的结构
            for g in gros:
                if "production" in g or "npt" in g:
                    target_gro = g
                    break
            if not target_gro and gros: target_gro = gros[0]
            
            if target_gro:
                md_qc_filename = f"{cid}_QC_MD_MAP.pdb"
                md_qc_path = os.path.join(OUTPUT_DIR, md_qc_filename)
                
                # 【核心调用】
                # 这里会触发 qm_loader -> RDKitMatcher -> match
                # 能够处理原子乱序、原子数不一致、对称性翻转
                success = validate_md_mapping(qm_data, target_gro, md_qc_path, global_max=GLOBAL_MAX)
                
                if success:
                    print(f"  [OK] RDKit Mapping successful. QC files saved.")
                    success_count += 1
                else:
                    print(f"  [FAIL] RDKit Mapping failed. Check if Ref PDB is a valid substructure of MD Ligand.")
                    fail_count += 1
            else:
                print(f"  [Skip] No .gro file found for mapping check.")
            
            print("-" * 50)
            
    print(f"\n>>> Process Completed.")
    print(f"    Successful: {success_count}")
    print(f"    Failed:     {fail_count}")

if __name__ == "__main__":
    main()