import MDAnalysis as mda
import os
import glob
import sys
import warnings

# 抑制 MDAnalysis 的一些警告
warnings.filterwarnings("ignore")

# 尝试导入你的模块
try:
    from modules.sequence_aligner import OffsetCalculator
except ImportError as e:
    print(f"Error: 无法导入 modules.sequence_aligner。请确保在项目根目录下运行。\n{e}")
    sys.exit(1)

# ================= 配置区 =================
STANDARD_SEQUENCE = """
MDPLNLSWYDDDLERQNWSRPFNGSDGKADRPHYNYYATLLTLLIAVIVFGNVLVCMAVS
REKALQTTTNYLIVSLAVADLLVATLVMPWVVYLEVVGEWKFSRIHCDIFVTLDVMMCTA
SILNLCAISIDRYTAVAMPMLYNTRYSSKRRVTVMIAIVWVLSFTISCPLLFGLNNADQN
ECIIANPAFVVYSSIVSFYVPFIVTLLVYIKIYIVLRKRRKRVNTKRSSRAFRAHLRAPL
KGNCTHPEDMKLCTVIMKSNGSFPVNRRRVEAARRAQELEMEMLSSTSPPERTRYSPIPP
SHHQLTLPDPSHHGLHSTPDSPAKPEKNGHAKDHPKIAKIFEIQTMPNGKTRTSLKTMSR
RKLSQQKEKKATQMLAIVLGVFIICWLPFFITHILNIHCDCNIPPVLYSAFTWLGYVNSA
VNPIIYTTFNIEFRKAFLKILSC
"""

# 我们关心的关键残基 (标准编号)
# 118: Asp (Anchor), 390: Phe (Switch), 194: Asp (suspected shift)
CRITICAL_RESIDS_STD = [118, 390, 194] 

def check_residue_at_index(u, std_resid, offset):
    """
    检查在 Universe 中，(Std_ID + Offset) 位置到底是什么残基
    """
    target_resid = std_resid + offset
    
    # MDAnalysis 选择语句
    sel = u.select_atoms(f"resid {target_resid}")
    
    if len(sel) == 0:
        return target_resid, "MISSING", "N/A"
    
    # 获取第一个原子的残基信息
    res = sel.residues[0]
    return target_resid, res.resname, res.segid

def main():
    root = "."  # 既然文件夹就在当前目录下，这里用 "." 是对的
    print(">>> Debugging Residue Mapping (Folder Name Fuzzy Match)...")
    
    # 初始化比对器
    aligner = OffsetCalculator(STANDARD_SEQUENCE)
    
    all_dirs = glob.glob(os.path.join(root, "*"))
    # 排序
    all_dirs.sort(key=lambda x: (not "dopa" in os.path.basename(x).lower(), x))
    
    # 我们要诊断的目标关键词
    TARGETS = ["ROT", "UNC", "BRE", "Dopa", "S84", "ARI", "R10", "S10"]
    
    found_any = False
    
    for c_dir in all_dirs:
        if not os.path.isdir(c_dir): continue
        if any(x in c_dir for x in ["modules", "results", "__pycache__"]): continue
        
        cid = os.path.basename(c_dir)
        
        # === 修正：模糊匹配 ===
        # 只要文件夹名字包含 "UNC" (不区分大小写)，就进去检查
        matched_target = None
        for t in TARGETS:
            if t.lower() in cid.lower():
                matched_target = t
                break
        
        if not matched_target:
            continue
            
        found_any = True
        print(f"\n{'='*60}")
        print(f"Checking Compound: {cid} (Matched '{matched_target}')")
        
        # === 1. 寻找 Simulation Topology ===
        # 你的 extract_features 是找 merged.xtc 所在的目录
        xtcs = glob.glob(os.path.join(c_dir, "**", "merged.xtc"), recursive=True)
        
        if not xtcs:
            # 备用逻辑：如果你只跑了第一步没合并，可能只有 gro/pdb
            # 尝试直接找 step7_*.gro 或 production.tpr
            print(f"  [Info] No merged.xtc found. Searching for TPR/GRO directly...")
            tps = glob.glob(os.path.join(c_dir, "**", "*.tpr"), recursive=True)
            gros = glob.glob(os.path.join(c_dir, "**", "*.gro"), recursive=True)
            
            topo = next((t for t in tps if "production" in t), tps[0] if tps else None)
            traj = next((g for g in gros if "step7" in g), gros[0] if gros else None)
        else:
            xtc = xtcs[0]
            rd = os.path.dirname(xtc)
            tps = [os.path.join(rd, f) for f in os.listdir(rd) if f.endswith(".tpr")]
            topo = next((t for t in tps if "production" in t), tps[0] if tps else None)
            traj = xtc

        if not topo or not traj:
            print(f"  [Skip] Could not find Topology (.tpr) or Trajectory (.xtc/.gro)")
            continue
            
        print(f"  Top:  {os.path.basename(topo)}")
        print(f"  Traj: {os.path.basename(traj)}")

        # === 2. 加载 Universe 并计算 Offset ===
        try:
            u = mda.Universe(topo, traj)
            
            # 核心步骤：计算偏移量
            offset = aligner.calculate_offset(u, 389)
            
            if offset is None:
                print(f"  [CRITICAL ERROR] Offset calculation FAILED!")
                print("  Sequence alignment failed. Anchor residue (389) might be missing.")
                continue
                
            print(f"  [Alignment] Calculated Offset: {offset}")
            
            # === 3. 检查关键残基 ===
            print(f"  {'-'*55}")
            print(f"  {'Std ID':<8} | {'Calc ID':<8} | {'Found ResName':<15} | {'Verdict'}")
            print(f"  {'-'*55}")
            
            for std_id in CRITICAL_RESIDS_STD:
                actual_id, resname, segid = check_residue_at_index(u, std_id, offset)
                
                # 判定逻辑
                verdict = "OK"
                if std_id == 118:
                    if resname != "ASP": verdict = "WRONG! (Exp: ASP)"
                elif std_id == 390:
                    if resname != "PHE": verdict = "WRONG! (Exp: PHE)"
                elif std_id == 194:
                    if resname == "ASP": verdict = "SUSPICIOUS (If 118 missed)"
                
                print(f"  {std_id:<8} | {actual_id:<8} | {resname:<15} | {verdict}")
                
        except Exception as e:
            print(f"  [Error] Processing failed: {e}")

    if not found_any:
        print("\n[Warning] Still found no folders matching UNC, ROT, etc.")
        print("Please check if folder names contain these strings.")

if __name__ == "__main__":
    main()