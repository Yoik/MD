import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from scipy.spatial.distance import cdist
import sys
import os
import glob
import itertools

# ==============================================================================
# 全局配置
# ==============================================================================
# 邻居原子样式
ATOM_PROPS = {
    1:  {'symbol': 'H',  'color': 'lightgray', 'size': 40},
    7:  {'symbol': 'N',  'color': 'blue',      'size': 100},
    8:  {'symbol': 'O',  'color': 'red',       'size': 100},
    9:  {'symbol': 'F',  'color': 'lime',      'size': 100},
    16: {'symbol': 'S',  'color': 'yellow',    'size': 120},
    17: {'symbol': 'Cl', 'color': 'green',     'size': 140},
    35: {'symbol': 'Br', 'color': 'darkred',   'size': 150},
    53: {'symbol': 'I',  'color': 'purple',    'size': 160},
}
DEFAULT_PROP = {'symbol': 'X', 'color': 'pink', 'size': 80}

# ELF 热力图配色方案 (Blue=Low -> Red=High)
CMAP_NAME = 'coolwarm' 

# 全局变量：Dopa 的最大值 (运行时填充)
GLOBAL_DOPA_MAX = 1.0

# ==============================================================================
# 1. CUBE 解析器 (含单位修正)
# ==============================================================================
class CubeAnalyzer:
    def __init__(self, filepath):
        self.filepath = filepath
        self.data = None; self.origin = None; self.spacing = None; self.dims = None
        self.is_header_bohr = True
        self._load()

    def _load(self):
        try:
            with open(self.filepath, 'r') as f:
                lines = f.readlines()
                parts = lines[2].split()
                natoms = int(parts[0])
                
                # 单位判断
                if natoms > 0: self.is_header_bohr = True
                else: self.is_header_bohr = False; natoms = abs(natoms)
                
                self.origin = np.array([float(x) for x in parts[1:4]])
                nx = int(lines[3].split()[0]); vx = np.array([float(x) for x in lines[3].split()[1:4]])
                ny = int(lines[4].split()[0]); vy = np.array([float(x) for x in lines[4].split()[1:4]])
                nz = int(lines[5].split()[0]); vz = np.array([float(x) for x in lines[5].split()[1:4]])
                self.dims = (nx, ny, nz); self.spacing = np.array([vx[0], vy[1], vz[2]]) 
                
                self.atom_lines = lines[6 : 6 + natoms]
                data_start = 6 + natoms
                raw_data = []
                for line in lines[data_start:]: raw_data.extend([float(x) for x in line.split()])
                self.data = np.array(raw_data).reshape(self.dims)
                
                # 解析原子坐标 (转为 Angstrom 用于绘图)
                self.coords = []; self.elements = []
                scale = 0.529177 if self.is_header_bohr else 1.0
                for line in self.atom_lines:
                    p = line.split()
                    self.elements.append(int(p[0]))
                    self.coords.append(np.array([float(x) for x in p[2:5]]) * scale)
                self.coords = np.array(self.coords)
                self.elements = np.array(self.elements)
                
        except Exception as e:
            print(f"  [Error] Load failed {self.filepath}: {e}")
            self.data = None

    def get_atom_elf(self, atom_idx):
        """获取指定原子位置的 ELF 值"""
        if self.data is None: return 0.0
        
        # 重新解析原始行以获取未缩放坐标 (用于查 grid)
        line = self.atom_lines[atom_idx]
        parts = line.split()
        coord = np.array([float(x) for x in parts[2:5]])
        
        # 如果是 Angstrom (header<0)，转 Bohr 查 Grid
        if not self.is_header_bohr:
            coord = coord * 1.889726 # Angstrom -> Bohr
            
        grid_idx = np.round((coord - self.origin) / self.spacing).astype(int)
        if np.all(grid_idx >= 0) and np.all(grid_idx < self.dims):
            return self.data[grid_idx[0], grid_idx[1], grid_idx[2]]
        return 0.0

# ==============================================================================
# 2. 几何识别 (找苯环 & 邻居)
# ==============================================================================
def find_benzene_ring(coords, elements):
    c_indices = np.where(elements == 6)[0]
    if len(c_indices) < 6: return None, None
    c_coords = coords[c_indices]
    dmat = cdist(c_coords, c_coords)
    adj = np.logical_and(dmat > 1.2, dmat < 1.55)
    
    def dfs(start, curr, path):
        if len(path) == 6: return path if adj[curr, start] else None
        for n in np.where(adj[curr])[0]:
            if n == start and len(path) < 5: continue
            if n not in path:
                res = dfs(start, n, path + [n])
                if res: return res
        return None

    ring_local = None
    for i in range(len(c_indices)):
        res = dfs(i, i, [i])
        if res: ring_local = res; break
    
    if ring_local is None:
        if len(c_indices) == 6: ring_local = list(range(6))
        else: return None, None
            
    ring_global_idx = c_indices[ring_local]
    all_dmat = cdist(coords, coords[ring_global_idx])
    neighbor_mask = np.any((all_dmat < 1.6) & (all_dmat > 0.1), axis=1)
    for idx in ring_global_idx: neighbor_mask[idx] = False
    neighbor_idx = np.where(neighbor_mask)[0]
    return ring_global_idx, neighbor_idx

# ==============================================================================
# 3. 绘图 (热力图风格)
# ==============================================================================
def plot_elf_heatmap(analyzer, ring_idx, neigh_idx, ring_elfs, output_file):
    coords = analyzer.coords
    ring_coords = coords[ring_idx]
    neigh_coords = coords[neigh_idx] if len(neigh_idx) > 0 else np.empty((0,3))
    neigh_elements = analyzer.elements[neigh_idx] if len(neigh_idx) > 0 else []
    
    # 归一化 ELF (相对于 Dopa Max)
    norm_elfs = ring_elfs / GLOBAL_DOPA_MAX
    # 截断一下防止过饱和 (虽然理论上不会超过 1.0太多)
    norm_elfs = np.clip(norm_elfs, 0, 1.0)
    
    # 颜色映射
    cmap = plt.get_cmap(CMAP_NAME)
    norm = mcolors.Normalize(vmin=0, vmax=1.0)
    
    # 设置绘图
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_aspect('equal')
    
    # 1. 画键
    plot_atoms = np.vstack([ring_coords, neigh_coords]) if len(neigh_coords)>0 else ring_coords
    dmat = cdist(plot_atoms, plot_atoms)
    for i in range(len(plot_atoms)):
        for j in range(i+1, len(plot_atoms)):
            if dmat[i,j] < 1.65:
                ax.plot([plot_atoms[i,0], plot_atoms[j,0]], 
                        [plot_atoms[i,1], plot_atoms[j,1]], 'k-', lw=3, zorder=1, alpha=0.3)

    # 2. 画苯环原子 (根据强度着色)
    # 使用 scatter 画圈
    sc = ax.scatter(ring_coords[:,0], ring_coords[:,1], 
                    c=norm_elfs, cmap=cmap, norm=norm, 
                    s=600, label='Benzene Carbon', zorder=2, edgecolors='k', linewidth=1.5)
    
    # 在圈内标注数值
    for i, (coord, val) in enumerate(zip(ring_coords, norm_elfs)):
        # 字体颜色根据背景深浅自动调整
        text_color = 'white' if (val > 0.7 or val < 0.3) else 'black'
        ax.text(coord[0], coord[1], f"{val:.2f}", 
                ha='center', va='center', fontsize=9, color=text_color, weight='bold', zorder=3)

    # 3. 画邻居原子
    legend_elements = set()
    if len(neigh_coords) > 0:
        for i, coord in enumerate(neigh_coords):
            at_num = neigh_elements[i]
            # 忽略碳氢 (如果是纯碳氢邻居，一般不画或者画小点，这里只画杂原子和碳)
            if at_num == 6: 
                prop = {'symbol': 'C', 'color': 'gray', 'size': 80}
            else:
                prop = ATOM_PROPS.get(at_num, DEFAULT_PROP)
            
            ax.scatter(coord[0], coord[1], 
                       c=prop['color'], s=prop['size'], alpha=0.9, zorder=2, edgecolors='k')
            
            # 标签偏移
            ax.text(coord[0]+0.15, coord[1]+0.15, prop['symbol'], 
                    fontsize=11, color=prop['color'], weight='bold', zorder=3)

    # 4. 调整布局
    all_points_2d = plot_atoms[:,:2]
    mid_x, mid_y = np.mean(all_points_2d, axis=0)
    span = np.max(np.ptp(all_points_2d, axis=0)) / 2.0 + 1.2
    ax.set_xlim(mid_x - span, mid_x + span)
    ax.set_ylim(mid_y - span, mid_y + span)
    
    # 添加 Colorbar
    cbar = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(f'Norm. ELF Intensity (1.0 = Dopa Max)', rotation=270, labelpad=15)
    
    plt.title(f"ELF-pi Distribution (Top View)\n{os.path.basename(analyzer.filepath)}", fontsize=14)
    ax.axis('off') # 关闭坐标轴刻度，更像结构式
    
    print(f"  -> Saving {output_file}")
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()

# ==============================================================================
# Main
# ==============================================================================
if __name__ == "__main__":
    target_cubes = []
    # 递归搜索
    for root, dirs, files in os.walk("."):
        for f in files:
            if f.endswith(".cub"):
                target_cubes.append(os.path.join(root, f))
    
    if not target_cubes:
        print("No .cub files found."); sys.exit()

    # --- 1. 预扫描 Dopa Max ---
    print(">>> 1. Finding Global Max (Dopa)...")
    dopa_max = 0.0
    for cub in target_cubes:
        if "dopa" in os.path.basename(os.path.dirname(cub)).lower() or "dopa" in os.path.basename(cub).lower():
            try:
                analyzer = CubeAnalyzer(cub)
                ring_idx, _ = find_benzene_ring(analyzer.coords, analyzer.elements)
                if ring_idx is not None:
                    vals = [analyzer.get_atom_elf(idx) for idx in ring_idx]
                    local_max = np.max(vals)
                    print(f"    Found Dopa: {os.path.basename(cub)}, Max={local_max:.4f}")
                    if local_max > dopa_max: dopa_max = local_max
            except: pass
    
    if dopa_max == 0:
        print("    [Warn] Dopa not found, using 1.0")
        dopa_max = 1.0
    else:
        print(f"    Global Standard (Dopa Max) = {dopa_max:.4f}")
    
    GLOBAL_DOPA_MAX = dopa_max

    # --- 2. 处理所有文件 ---
    print("\n>>> 2. Generating Heatmaps...")
    for cub_file in target_cubes:
        try:
            print(f"Processing {os.path.basename(cub_file)}...")
            analyzer = CubeAnalyzer(cub_file)
            ring_idx, neigh_idx = find_benzene_ring(analyzer.coords, analyzer.elements)
            
            if ring_idx is None:
                print("  [Skip] No benzene ring found.")
                continue
            
            # 获取苯环6个原子的 ELF
            ring_elfs = np.array([analyzer.get_atom_elf(idx) for idx in ring_idx])
            
            # 画图
            out_png = os.path.splitext(cub_file)[0] + "_elf_heatmap.png"
            plot_elf_heatmap(analyzer, ring_idx, neigh_idx, ring_elfs, out_png)
            
        except Exception as e:
            print(f"  [Error] {e}")