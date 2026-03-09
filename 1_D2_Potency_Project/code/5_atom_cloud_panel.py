import sys
import os
import matplotlib
matplotlib.use('Agg') # 服务器端绘图
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader

# 引入项目模块
from src.dataset import prepare_data, RankingDataset
from src.model import EfficiencyPredictor
from src.config import init_config

def extract_3d_data(model, data_loader, device):
    """
    提取原子相对于 6.51 和 6.52 的双重几何特征 + 重要性
    """
    model.eval()
    points = [] # [d1, a1, d2, a2, importance]
    
    print("Extracting geometric features...")
    with torch.no_grad():
        for batch in tqdm(data_loader):
            x = batch['query_feat'].to(device)
            batch_size, num_frames, _ = x.shape
            
            # 1. 预处理 (拆分 & Mask)
            num_atom_feats = model.n_atoms * model.atom_feat_dim
            x_atoms = x[:, :, :num_atom_feats].view(batch_size, num_frames, model.n_atoms, model.atom_feat_dim)
            
            # 应用 Shared Mask
            if hasattr(model, 'atom_mask_logits'):
                atom_mask = torch.sigmoid(model.atom_mask_logits).view(1, 1, 1, -1)
                x_atoms_masked = x_atoms * atom_mask
            else:
                x_atoms_masked = x_atoms # Fallback
            
            # 2. 计算重要性
            x_flat = x_atoms_masked.view(batch_size * num_frames, model.n_atoms, model.atom_feat_dim)
            atom_embeddings = model.atom_encoder.atom_mlp(x_flat)
            atom_importance = torch.norm(atom_embeddings, dim=2) # [B*F, 9]
            
            # 3. 提取几何坐标
            # 原始特征 (未 Mask 的): 
            # 0: d_6.51, 1: a_6.51, 2: d_6.52, 3: a_6.52
            raw_atoms_flat = x_atoms.view(batch_size * num_frames, model.n_atoms, model.atom_feat_dim)
            
            d1 = raw_atoms_flat[:, :, 0].flatten() # Dist 6.51
            a1 = raw_atoms_flat[:, :, 1].flatten() # Angle 6.51
            d2 = raw_atoms_flat[:, :, 2].flatten() # Dist 6.52
            a2 = raw_atoms_flat[:, :, 3].flatten() # Angle 6.52
            imp = atom_importance.flatten()
            
            # Stack
            pts = torch.stack([d1, a1, d2, a2, imp], dim=1)
            points.append(pts.cpu().numpy())
            
    return np.concatenate(points, axis=0)

def plot_3d_cone_cloud(data, residue_name, col_dist, col_angle, output_path):
    """
    将 (r, theta) 转换为伪 3D 坐标并绘图
    """
    # 过滤高分点 (Top 20%)
    threshold = np.percentile(data[:, 4], 80)
    indices = data[:, 4] > threshold
    subset = data[indices]
    
    r = subset[:, col_dist]
    theta = subset[:, col_angle] # 假设是弧度或Cos值，这里假设是原始角度值(如果是Cos需要arccos)
    # 注意：如果特征提取时用了 np.cos(angle)，这里需要反算。
    # 通常 MD 提取直接是角度(弧度或度)。假设是弧度。
    # 为了保险，我们直接用 r 和 theta 作图，不强求转换笛卡尔，除非确定单位。
    
    # 转换为圆柱坐标投影 (假设 theta 是与法向量夹角)
    # Z = 垂直距离 (Projection along Normal)
    # R = 水平距离 (Projection on Plane)
    # 如果 feature 是 cos(theta)，则 z = r * cos_theta
    # 如果 feature 是 theta(弧度)，则 z = r * cos(theta)
    
    # 假设特征是弧度 (常见做法)
    z_local = r * np.cos(theta)
    rho_local = r * np.sin(theta)
    
    # 为了 3D 效果，随机生成方位角 phi (0~2pi)
    phi = np.random.uniform(0, 2*np.pi, size=len(r))
    x_3d = rho_local * np.cos(phi)
    y_3d = rho_local * np.sin(phi)
    z_3d = z_local
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # 绘制原点 (Residue Center)
    ax.scatter([0], [0], [0], c='red', s=200, marker='*', label=f'{residue_name} Center')
    
    # 绘制法向量轴
    ax.plot([0, 0], [0, 0], [0, np.max(z_3d)], c='red', linestyle='--', alpha=0.5, label='Normal Vector')
    
    # 绘制原子云
    p = ax.scatter(x_3d, y_3d, z_3d, c=subset[:, 4], cmap='viridis', s=5, alpha=0.4)
    
    ax.set_xlabel('X (Angstrom)')
    ax.set_ylabel('Y (Angstrom)')
    ax.set_zlabel('Z (Angstrom, along Normal)')
    ax.set_title(f'3D Reconstructed Pharmacophore Cloud\nRelative to {residue_name}')
    fig.colorbar(p, label='Importance')
    plt.legend()
    plt.savefig(output_path)
    print(f"Saved {output_path}")

def plot_dual_distance_3d(data, output_path):
    """
    X=Dist6.51, Y=Dist6.52, Z=Importance
    """
    # 过滤
    threshold = np.percentile(data[:, 4], 80)
    subset = data[indices := data[:, 4] > threshold]
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    p = ax.scatter(subset[:, 0], subset[:, 2], subset[:, 4], c=subset[:, 4], cmap='magma', s=5, alpha=0.5)
    
    ax.set_xlabel('Dist to 6.51')
    ax.set_ylabel('Dist to 6.52')
    ax.set_zlabel('Importance Score')
    ax.set_title('Dual-Residue Interaction Landscape')
    
    fig.colorbar(p, label='Importance')
    plt.savefig(output_path)
    print(f"Saved {output_path}")

def main():
    config = init_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 加载数据 (Train + Test 全量)
    print("Loading data...")
    try:
        train_ds, test_ds = prepare_data(
            label_file=config.get_path("paths.label_file"), 
            result_dir=config.get_path("paths.result_dir"), 
            ref_feature_path=config.get_path("paths.ref_feature_path"),
            pocket_atom_num=config.get_int("data.pocket_atom_num"), 
            save_scaler_path=config.get_path("paths.scaler_path")
        )
    except:
        print("Data load failed.")
        return

    all_feats = train_ds.query_feats + test_ds.query_feats
    all_labels = train_ds.query_labels + test_ds.query_labels
    all_ids = train_ds.query_ids + test_ds.query_ids
    
    ds = RankingDataset(all_feats, all_labels, all_ids, train_ds.ref_feats)
    loader = DataLoader(ds, batch_size=32, shuffle=False)
    
    # 加载模型
    model_path = config.get_path("paths.model_path")
    model = EfficiencyPredictor(input_dim=config.get_int("data.input_dim_features"))
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    
    # 提取数据
    # columns: [d1, a1, d2, a2, imp]
    data = extract_3d_data(model, loader, device)
    
    # 绘图 1: 6.51 坐标系云
    plot_3d_cone_cloud(data, "6.51", col_dist=0, col_angle=1, 
                       output_path=os.path.join(config.get_path("paths.result_dir"), 'cloud_3d_651.png'))
    
    # 绘图 2: 6.52 坐标系云
    plot_3d_cone_cloud(data, "6.52", col_dist=2, col_angle=3, 
                       output_path=os.path.join(config.get_path("paths.result_dir"), 'cloud_3d_652.png'))
                       
    # 绘图 3: 双距离景观
    plot_dual_distance_3d(data, 
                          output_path=os.path.join(config.get_path("paths.result_dir"), 'landscape_3d_dual.png'))

if __name__ == "__main__":
    main()