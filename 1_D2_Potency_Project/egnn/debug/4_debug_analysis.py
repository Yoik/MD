import torch
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# ================= 配置 =================
DATA_DIR = "data/features"
TARGETS = ["Dopa", "BRE", "S84", "ARI", "UNC"]  # 对比组
# 这里的 7084 对应的文件夹名可能需要你确认，如果叫 R10 或其他，请修改
# 如果你有 7084 的数据，建议加进去对比，它是 BRE 的刚性对照

def analyze_compound(name):
    print(f"\nAnalyzing {name}...")
    
    # 1. 寻找文件
    # 模糊匹配文件夹
    found_dir = None
    all_dirs = glob.glob(os.path.join(DATA_DIR, "*"))
    for d in all_dirs:
        if name.lower() in os.path.basename(d).lower():
            found_dir = d
            break
            
    if not found_dir:
        print(f"  [Error] Folder not found for {name}")
        return None

    files = glob.glob(os.path.join(found_dir, "*", "graph_features.pt"))
    if not files:
        print(f"  [Error] No .pt files found in {found_dir}")
        return None
        
    # 加载所有帧
    all_frames = []
    for f in files:
        try:
            frames = torch.load(f, weights_only=False)
            all_frames.extend(frames)
        except Exception as e:
            print(f"  Load error: {e}")

    if not all_frames:
        return None

    print(f"  Loaded {len(all_frames)} frames.")
    
    # ================= 诊断 1: S84 崩塌之谜 (有效性检查) =================
    valid_frames = 0
    zero_weight_frames = 0
    total_ligand_atoms = 0
    
    positions = [] # 记录配体几何中心
    weights_sum = [] # 记录每一帧的总电子权重
    
    for data in all_frames:
        x = data.x
        pos = data.pos
        
        # 解析特征 (假设第 10 列是 is_ligand, 第 9 列是 electronic_weight)
        # x dim: [Type(9), Weight(1), IsLig(1), Res(N)]
        is_ligand = x[:, 10] == 1
        weights = x[:, 9]
        
        lig_w = weights[is_ligand]
        lig_pos = pos[is_ligand]
        
        if len(lig_pos) == 0: continue
        
        # 统计
        total_w = lig_w.sum().item()
        weights_sum.append(total_w)
        
        if total_w > 0.01: # 阈值：只要有一点点权重就算有效
            valid_frames += 1
        else:
            zero_weight_frames += 1
            
        # 记录几何中心 (Geometric Center)
        # 用几何中心代表“骨架位置”，看它晃不晃
        center = lig_pos.mean(dim=0).numpy()
        positions.append(center)
        
        total_ligand_atoms = max(total_ligand_atoms, len(lig_pos))

    # ================= 诊断 2: BRE 虚高之谜 (空间稳定性) =================
    positions = np.array(positions)
    if len(positions) > 1:
        # 计算质心的标准差 (RMSF_COM) -> 反映整体乱动程度
        center_std = np.std(positions, axis=0) # [std_x, std_y, std_z]
        rmsf_com = np.linalg.norm(center_std)
    else:
        rmsf_com = 0.0

    stats = {
        "name": name,
        "total_frames": len(all_frames),
        "valid_ratio": valid_frames / len(all_frames) * 100,
        "avg_weight": np.mean(weights_sum) if weights_sum else 0,
        "rmsf_com": rmsf_com,  # 核心指标：越大约乱
        "atom_count": total_ligand_atoms
    }
    
    print(f"  -> Valid Frames: {stats['valid_ratio']:.1f}%")
    print(f"  -> Avg Elec Weight: {stats['avg_weight']:.4f}")
    print(f"  -> Spatial Instability (RMSF): {stats['rmsf_com']:.4f} Å")
    
    return stats, positions

def main():
    results = []
    plot_data = {}
    
    for target in TARGETS:
        res = analyze_compound(target)
        if res:
            stats, pos = res
            results.append(stats)
            plot_data[target] = pos
            
    print("\n" + "="*50)
    print(f"{'Compound':<10} {'Valid%':<10} {'Weight':<10} {'Instability(Å)':<15} {'Diagnosis'}")
    print("-" * 65)
    
    for r in results:
        # 自动诊断逻辑
        diag = "Normal"
        if r['valid_ratio'] < 10:
            diag = "COLLAPSED (Alignment Fail?)"
        elif r['rmsf_com'] > 1.5: # 假设阈值，需根据 Dopa 调整
            diag = "UNSTABLE (Tail Wagging?)"
        elif r['rmsf_com'] < 0.5 and r['valid_ratio'] > 90:
            diag = "RIGID (Good Binder)"
            
        print(f"{r['name']:<10} {r['valid_ratio']:<10.1f} {r['avg_weight']:<10.4f} {r['rmsf_com']:<15.4f} {diag}")
    print("="*50)
    
    # 简单的可视化：画出质心分布图 (XY平面)
    plt.figure(figsize=(8, 8))
    for name, pos in plot_data.items():
        if len(pos) == 0: continue
        # 假设 Z 轴是通道方向，我们看 XY 平面的散布
        # 减去均值，让它们叠在一起比较“散布范围”
        centered_pos = pos - pos.mean(axis=0)
        plt.scatter(centered_pos[:, 0], centered_pos[:, 1], s=5, alpha=0.5, label=name)
        
        # 画个圈表示 1 sigma 范围
        std = np.std(centered_pos, axis=0)
        circle = plt.Circle((0, 0), np.linalg.norm(std[:2]), color='black', fill=False, linestyle='--', alpha=0.3)
        plt.gca().add_patch(circle)

    plt.title("Ligand Center-of-Mass Distribution (Relative to Mean)\nVerify: Is BRE more scattered than Dopa?")
    plt.xlabel("Delta X (Å)")
    plt.ylabel("Delta Y (Å)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.savefig("debug_spatial_distribution.png", dpi=300)
    print("\nSaved scatter plot to debug_spatial_distribution.png")

if __name__ == "__main__":
    main()