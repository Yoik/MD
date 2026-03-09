import MDAnalysis as mda
import numpy as np
import os
import glob

from modules.sequence_aligner import OffsetCalculator
from src.config import init_config


# =========================
# 初始化配置
# =========================
config = init_config()

ANCHOR_BW = {
    "W6.48": ["6.48", "6.51", "6.52"],
    "TM5_Polar": ["5.42", "5.46", "5.43"],
    "TM6_Polar": ["6.55"]
}

TARGETS = ["UNC", "Dopa", "BRE", "S84"]


# =========================
# 主函数
# =========================
def check_distance(name):
    print("\n==============================")
    print(f"Analyzing target: {name}")
    print("==============================")

    # -------- 项目根目录 --------
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

    print(f"[INFO] Searching under: {PROJECT_ROOT}")

    # -------- 找 simulation 根目录 --------
    found_dir = None
    for d in glob.glob(os.path.join(PROJECT_ROOT, "2025*_D2_*")):
        if os.path.isdir(d) and f"_D2_{name}_" in os.path.basename(d):
            found_dir = d
            break

    if not found_dir:
        raise RuntimeError(f"Cannot find simulation directory for {name}")

    print(f"[INFO] Found simulation root: {found_dir}")

    # -------- 找 replicate --------
    replicate_dirs = sorted(
        d for d in glob.glob(
            os.path.join(found_dir, "**/gromacs_replicate_*"),
            recursive=True
        )
        if os.path.isdir(d)
    )

    if not replicate_dirs:
        raise RuntimeError("No gromacs_replicate_* directories found")

    print(f"[INFO] Found {len(replicate_dirs)} replicates")

    # -------- 逐 replicate 计算 --------
    for rep_dir in replicate_dirs:
        print(f"\n--- Replicate: {os.path.basename(rep_dir)} ---")

        xtc = os.path.join(rep_dir, "merged.xtc")
        tprs = glob.glob(os.path.join(rep_dir, "step7_3.tpr"))

        if not os.path.exists(xtc):
            print("[WARNING] merged.xtc not found, skipping")
            continue
        if len(tprs) != 1:
            print("[WARNING] tpr not unique, skipping")
            continue

        tpr = tprs[0]

        print(f"[INFO] xtc: {xtc}")
        print(f"[INFO] tpr: {tpr}")

        # -------- Universe --------
        u = mda.Universe(tpr, xtc)
        aligner = OffsetCalculator()

        # -------- Anchor mapping --------
        anchor_real_ids = {}
        for k, bw_list in ANCHOR_BW.items():
            rids = aligner.get_real_residue_ids(u, bw_list)
            if rids:
                anchor_real_ids[k] = rids
                print(f"[INFO] Anchor {k} → resid {rids}")
            else:
                print(f"[WARNING] Anchor {k} not found")

        if not anchor_real_ids:
            print("[WARNING] No anchors mapped, skipping replicate")
            continue

        # -------- Ligand --------
        lig_ag = u.select_atoms("resname LIG1 or resname UNK or resname LDP")
        if len(lig_ag) == 0:
            print("[WARNING] No ligand atoms, skipping replicate")
            continue

        print(f"[INFO] Ligand atoms: {len(lig_ag)}")

        # -------- 距离计算 --------
        avg_dists = {k: [] for k in anchor_real_ids}

        for ts in u.trajectory[:10]:  # 前 10 帧
            lig_pos = lig_ag.positions

            for k, rids in anchor_real_ids.items():
                anchor_ag = u.select_atoms(
                    f"resid {' '.join(map(str, rids))} and not name H*"
                )
                if len(anchor_ag) == 0:
                    continue

                center = anchor_ag.center_of_mass()
                dists = np.linalg.norm(lig_pos - center, axis=1)
                avg_dists[k].append(np.min(dists))

        # -------- 输出 --------
        print("[RESULT] Mean minimal distances (first 10 frames):")
        for k, v in avg_dists.items():
            if v:
                print(f"  {k:12s}: {np.mean(v):6.2f} Å")
            else:
                print(f"  {k:12s}: N/A")


# =========================
# 入口
# =========================
if __name__ == "__main__":
    for t in TARGETS:
        try:
            check_distance(t)
        except Exception as e:
            print(f"[ERROR] {t}: {e}")
