import MDAnalysis as mda
from MDAnalysis.lib.distances import distance_array
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import os

# ==============================================================================
# ⚠️ 用户手动指定区
# ==============================================================================
UNC_DIR = "./20251123_D2_UNC_Boltz"
OFFSET = -29  # UNC 的正确偏移

# 你刚刚点选的 6 个原子 (顺序不重要，质心是一样的)
MANUAL_ATOMS = ['C16', 'C13', 'C8', 'C7', 'C12', 'C15']

# 关键指纹残基
FINGERPRINT_RESIDUES = [114, 389, 190, 118] 

# ==============================================================================
# 核心逻辑
# ==============================================================================
def find_traj_files(base_dir):
    for root, dirs, files in os.walk(base_dir):
        xtc = next((os.path.join(root, f) for f in files if f.endswith(".xtc") and "step7" in f), None)
        tpr = next((os.path.join(root, f) for f in files if f.endswith(".tpr") and "step7" in f), None)
        if xtc and tpr: return xtc, tpr
    return None, None

def main():
    print(f"Diagnosing UNC (Manual Mode) ...")
    print(f"Target Atoms: {MANUAL_ATOMS}")
    
    xtc, tpr = find_traj_files(UNC_DIR)
    if not xtc: print("Files not found"); return
    
    u = mda.Universe(tpr, xtc)
    
    # 1. 直接选择这 6 个原子
    # 注意：为了防止选到其他残基的同名原子，限定 resname
    sel_string = f"resname LIG LIG1 LDP R5F and name {' '.join(MANUAL_ATOMS)}"
    target_ring_atoms = u.select_atoms(sel_string)
    
    if len(target_ring_atoms) != 6:
        print(f"[Error] Found {len(target_ring_atoms)} atoms instead of 6!")
        print("Check if resname is correct (LIG/LIG1/LDP/R5F).")
        # 尝试列出找到的原子
        for a in target_ring_atoms: print(f"  Found: {a.name} in res {a.resname} {a.resid}")
        return

    print(f"Successfully selected {len(target_ring_atoms)} atoms.")
    ring_center = target_ring_atoms.center_of_mass()

    # 2. 计算并打印距离
    print("\n" + "="*60)
    print(f"{'Target':<15} | {'Dist (A)':<15} | {'Status'}")
    print("-" * 60)
    
    pocket_coords = [] # 用于绘图
    pocket_labels = []
    
    for rid in FINGERPRINT_RESIDUES:
        target_resid = rid + OFFSET
        ag = u.select_atoms(f"resid {target_resid} and name CA")
        
        status = "MISSING"
        dist_val = 999.9
        
        if len(ag) > 0:
            pos = ag.positions[0]
            # 计算 PBC 距离
            dist_val = distance_array(ring_center[np.newaxis, :], pos[np.newaxis, :], box=u.dimensions)[0][0]
            
            pocket_coords.append(pos)
            pocket_labels.append(f"{rid}")
            
            # 简单的状态判断
            if rid == 389: # 6.51
                status = "ANCHOR (Expected ~5-15)"
            elif rid == 114:
                status = "FAR (Expected ~30-40)"
            else:
                status = ""
        
        print(f"Res {rid:<11} | {dist_val:.2f}           | {status}")
    print("="*60)

    # ==========================================================================
    # 3. 3D 可视化 (生成 png)
    # ==========================================================================
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # 画口袋残基
    pocket_coords = np.array(pocket_coords)
    if len(pocket_coords) > 0:
        ax.scatter(pocket_coords[:,0], pocket_coords[:,1], pocket_coords[:,2], 
                   c='blue', s=150, label='Pocket Residues (CA)')
        for i, txt in enumerate(pocket_labels):
            ax.text(pocket_coords[i,0], pocket_coords[i,1], pocket_coords[i,2], txt, color='blue', fontsize=12, weight='bold')

    # 画手动指定的苯环 (绿色)
    r_coords = target_ring_atoms.positions
    ax.scatter(r_coords[:,0], r_coords[:,1], r_coords[:,2], c='green', s=100, label='Manual Ring')
    
    # 连线 (闭合环)
    # 注意：原子列表顺序可能是乱的，直接连线可能会画成五角星。
    # 这里为了画图好看，简单按空间距离排序连线
    from scipy.spatial import ConvexHull
    try:
        hull = ConvexHull(r_coords[:,:2]) # 投影到2D找凸包顺序
        order = hull.vertices
        ordered_coords = r_coords[order]
        loop = np.vstack([ordered_coords, ordered_coords[0]])
        ax.plot(loop[:,0], loop[:,1], loop[:,2], c='green', lw=3)
    except:
        pass # 如果投影失败就不连线了

    # 画配体其他原子 (灰色背景)
    all_ligand = u.select_atoms("resname LIG LIG1 LDP R5F and not name H*")
    other_coords = all_ligand.positions
    ax.scatter(other_coords[:,0], other_coords[:,1], other_coords[:,2], c='gray', s=20, alpha=0.3)

    # 设置视角
    all_points = np.vstack([pocket_coords, r_coords])
    mid = np.mean(all_points, axis=0)
    max_range = np.max(np.ptp(all_points, axis=0)) / 2.0 + 5.0

    ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
    ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
    ax.set_zlim(mid[2] - max_range, mid[2] + max_range)

    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    ax.legend()
    plt.title(f"Manual Ring Check: {MANUAL_ATOMS}")
    
    out_file = "unc_manual_check.png"
    plt.savefig(out_file, dpi=150)
    print(f"\n[Visual] Saved 3D check image to {out_file}")

if __name__ == "__main__":
    main()