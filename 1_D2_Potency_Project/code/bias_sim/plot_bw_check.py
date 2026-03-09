import matplotlib.pyplot as plt
import numpy as np

def plot_bw_correlation():
    try:
        # 读取 PLUMED driver 生成的文件
        # 格式: time, phi_B, dist_BW
        data = np.loadtxt("ANALYSIS_BW", comments="#")
        
        phi_B = data[:, 1]       # 第二列是角度
        dist_BW = data[:, 2] * 10.0  # 第三列是距离 (nm -> Angstrom)
        
        plt.figure(figsize=(8, 6))
        
        # 散点图：横轴是翻转角，纵轴是 B-W 距离
        # alpha 设置透明度，方便看点的密度
        plt.scatter(phi_B, dist_BW, s=5, alpha=0.1, c='darkblue')
        
        plt.xlabel(r"Phe389 $\chi_2$ Angle (rad)")
        plt.ylabel(r"Distance Phe389 - Trp386 ($\AA$)")
        plt.title("Correlation: Phe389 State vs. W Contact")
        plt.grid(True, alpha=0.3)
        
        # 标出关键区域
        plt.axvline(x=1.3, color='red', linestyle='--', alpha=0.5, label='State 1 (+75°)')
        plt.axvline(x=-1.8, color='green', linestyle='--', alpha=0.5, label='State 2 (-100°)')
        
        # 标出可能的接触阈值 (T型接触通常 < 6.0 A)
        plt.axhline(y=6.0, color='gray', linestyle=':', label='Contact Threshold (~6Å)')
        
        plt.legend()
        plt.tight_layout()
        plt.savefig("check_BW_distance.png")
        print("图片已保存为: check_BW_distance.png")
        
    except Exception as e:
        print(f"绘图出错: {e}")
        print("请确保已经运行了 'plumed driver' 并生成了 'ANALYSIS_BW' 文件。")

if __name__ == "__main__":
    plot_bw_correlation()