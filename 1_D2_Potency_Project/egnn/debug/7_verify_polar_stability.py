import MDAnalysis as mda
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
import seaborn as sns

from modules.sequence_aligner import OffsetCalculator
from src.config import init_config

# =========================
# 初始化配置
# =========================
config = init_config()

# 定义极性锚点 (Polar Anchors)
ANCHOR_BW = {
    # 疏水/深层核心 (用于对比)
    "W6.48": ["6.48", "6.51", "6.52"], 
    # 极性/侧壁锚点 (我们关注的焦点)
    "TM5_Polar": ["5.42", "5.46", "5.43"],
    "TM6_Polar": ["6.55"]
}

TARGETS = ["UNC", "Dopa", "BRE", "S84"]

# =========================
# 主函数
# =========================
def analyze_stability(name):
    print("\n==============================")
    print(f"Analyzing target: {name}")
    print("==============================")

    # -------- 1. 路径查找 (复用你的逻辑) --------
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
    
    # 自动尝试 data 和 results 目录，增加鲁棒性
    possible_roots = [
        PROJECT_ROOT, 
        os.path.join(PROJECT_ROOT, "data"), 
        os.path.join(PROJECT_ROOT, "..", "data")
    ]
    
    found_dir = None
    for root in possible_roots:
        if not os.path.exists(root): continue
        # 你的文件夹命名模式: 2025*_D2_{name}_*
        candidates = glob.glob(os.path.join(root, "2025*_D2_*"))
        for d in candidates:
            if os.path.isdir(d) and f"_D2_{name}_" in os.path.basename(d):
                found_dir = d
                break
        if found_dir: break

    if not found_dir:
        print(f"[ERROR] Cannot find simulation directory for {name}")
        return None

    print(f"[INFO] Simulation root: {found_dir}")

    # -------- 2. 找 Replicates --------
    replicate_dirs = sorted(
        d for d in glob.glob(os.path.join(found_dir, "**/gromacs_replicate_*"), recursive=True)
        if os.path.isdir(d)
    )

    if not replicate_dirs:
        print("[ERROR] No gromacs_replicate_* directories found")
        return None

    # 存储所有 Replicates 的距离数据，用于合并统计
    all_reps_data = {k: [] for k in ANCHOR_BW.keys()}

    # -------- 3. 逐 Replicate 计算 --------
    for rep_dir in replicate_dirs:
        print(f"  -> Processing: {os.path.basename(rep_dir)}")

        xtc = os.path.join(rep_dir, "merged.xtc")
        tprs = glob.glob(os.path.join(rep_dir, "step7_3.tpr"))

        if not os.path.exists(xtc): continue
        if len(tprs) != 1: continue
        tpr = tprs[0]

        try:
            u = mda.Universe(tpr, xtc)
            aligner = OffsetCalculator()

            # 获取锚点真实 ID
            anchor_real_ids = {}
            for k, bw_list in ANCHOR_BW.items():
                rids = aligner.get_real_residue_ids(u, bw_list)
                if rids: anchor_real_ids[k] = rids

            # 获取配体
            lig_ag = u.select_atoms("resname LIG LIG1 UNK LDP and not name H*")
            if len(lig_ag) == 0: continue

            # === 全轨迹扫描 (Stride=10) ===
            for ts in u.trajectory[::10]:
                lig_pos = lig_ag.positions
                
                for k, rids in anchor_real_ids.items():
                    # 锚点原子 (排除骨架，只看侧链极性/疏水部分)
                    anchor_ag = u.select_atoms(f"resid {' '.join(map(str, rids))} and not name H* and not name N C O CA")
                    if len(anchor_ag) == 0: 
                        # Fallback: 如果选不到侧链(如Gly)，就选全重原子
                        anchor_ag = u.select_atoms(f"resid {' '.join(map(str, rids))} and not name H*")
                    
                    if len(anchor_ag) == 0: continue

                    # 计算最小距离
                    # 1. 计算质心距离? 不，计算"最近原子距离"更符合相互作用物理
                    # (质心距离对长分子不准，我们关心的是有没有任何部分接触)
                    dmat = mda.lib.distances.distance_array(lig_pos, anchor_ag.positions)
                    min_dist = np.min(dmat)
                    
                    all_reps_data[k].append(min_dist)

        except Exception as e:
            print(f"    [Error] {e}")

    # -------- 4. 统计分析 --------
    stats = {}
    for k, dists in all_reps_data.items():
        if len(dists) == 0:
            stats[k] = None
        else:
            arr = np.array(dists)
            stats[k] = {
                "mean": np.mean(arr),
                "std": np.std(arr),
                "contact_ratio": np.sum(arr < 3.5) / len(arr) * 100, # <3.5Å 视为接触
                "data": arr
            }
            
    return stats

def main():
    print("Running Polar Stability Analysis...")
    
    summary_table = []
    
    # 准备绘图
    plt.figure(figsize=(12, 5))
    
    for i, target in enumerate(TARGETS):
        res = analyze_stability(target)
        if not res: continue
        
        # 我们重点关注 TM5_Polar 的稳定性
        tm5_stats = res.get("TM5_Polar")
        w648_stats = res.get("W6.48")
        
        if tm5_stats:
            row = {
                "Compound": target,
                "TM5_Mean": tm5_stats['mean'],
                "TM5_Std": tm5_stats['std'],
                "TM5_Contact%": tm5_stats['contact_ratio'],
                "W648_Mean": w648_stats['mean'] if w648_stats else 0
            }
            summary_table.append(row)
            
            # 画图: TM5 距离分布
            sns.kdeplot(tm5_stats['data'], label=f"{target} (std={tm5_stats['std']:.2f})", fill=True, alpha=0.3)
            
    # 绘图装饰
    plt.axvline(x=3.5, color='r', linestyle='--', label="H-Bond Cutoff (3.5Å)")
    plt.title("Distribution of Minimum Distance to TM5 Polar Anchors (5.42/5.46)")
    plt.xlabel("Distance (Å)")
    plt.ylabel("Density")
    plt.xlim(1.5, 10.0)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("debug_polar_stability_check.png", dpi=300)
    print("\n[Output] Plot saved to debug_polar_stability_check.png")
    
    # 打印最终表格
    print("\n" + "="*80)
    print(f"{'Compound':<10} | {'TM5 Mean':<10} {'TM5 Std':<10} {'Contact%':<10} | {'W6.48 Mean':<10} | {'Diagnosis'}")
    print("-" * 80)
    
    for row in summary_table:
        # 自动诊断
        diag = ""
        if row['TM5_Contact%'] > 80 and row['TM5_Std'] < 0.6:
            diag = "LOCKED (Enthalpy)"
        elif row['TM5_Std'] > 0.8:
            diag = "UNSTABLE (Entropy)"
        elif row['TM5_Contact%'] < 20:
            diag = "NO INTERACTION"
        else:
            diag = "MODERATE"
            
        print(f"{row['Compound']:<10} | {row['TM5_Mean']:<10.2f} {row['TM5_Std']:<10.2f} {row['TM5_Contact%']:<10.1f} | {row['W648_Mean']:<10.2f} | {diag}")
    print("="*80)

if __name__ == "__main__":
    main()