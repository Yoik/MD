import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import subprocess
import os

def reconstruct_fes_from_bias(phi, bias, nbins=100):
    """
    用 OPES bias 构建 1D effective free energy
    """
    bins = np.linspace(-np.pi, np.pi, nbins + 1)
    centers = 0.5 * (bins[:-1] + bins[1:])

    F = np.zeros(nbins)
    for i in range(nbins):
        mask = (phi >= bins[i]) & (phi < bins[i+1])
        if np.any(mask):
            F[i] = -np.mean(bias[mask])
        else:
            F[i] = np.nan

    # 去掉空 bin
    valid = ~np.isnan(F)
    centers = centers[valid]
    F = F[valid]

    # 归零
    F -= np.min(F)
    barrier = np.max(F)

    return centers, F, barrier


def plot_monitor():
    try:
        # ===== 读取 COLVAR =====
        data = np.loadtxt("COLVAR", comments="#")
        if data.shape[0] < 10:
            print("数据太少，稍后再看...")
            return

        time_ns = data[:, 0] / 1000.0
        phi_B = data[:, 1]
        dist_AB = data[:, 2] * 10.0
        dist_BC = data[:, 3] * 10.0

        # ===== 重建 FES =====
        bias = data[:, 4]   # opes.bias

        phi_fes, F_fes, barrier = reconstruct_fes_from_bias(phi_B, bias)

        # ===== 画图 =====
        fig = plt.figure(figsize=(10, 12))

        # --- Panel 1: Chi2 翻转 ---
        ax1 = plt.subplot(3, 1, 1)
        ax1.plot(time_ns, phi_B, '.', markersize=1)
        ax1.axhline(1.5, color='r', ls='--', alpha=0.3)
        ax1.axhline(-1.5, color='r', ls='--', alpha=0.3)
        ax1.set_ylabel("Chi2 (rad)")
        ax1.set_title("Phe389 Flipping Monitor")
        ax1.grid(alpha=0.3)

        # --- Panel 2: 距离 ---
        ax2 = plt.subplot(3, 1, 2, sharex=ax1)
        ax2.plot(time_ns, dist_BC, label="Dist B–C (Inducer)", color="orange")
        ax2.plot(time_ns, dist_AB, label="Dist A–B (Stacking)", color="green", alpha=0.6)
        ax2.set_ylabel("Distance (Å)")
        ax2.legend()
        ax2.grid(alpha=0.3)

        # --- Panel 3: FES ---
        ax3 = plt.subplot(3, 1, 3)
        if phi_fes is not None:
            ax3.plot(phi_fes, F_fes, lw=2)
            ax3.set_title(f"OPES Free Energy  ΔF ≈ {barrier:.1f} kJ/mol")
        else:
            ax3.text(0.5, 0.5, "FES not available",
                     ha="center", va="center", transform=ax3.transAxes)

        ax3.set_xlabel("Phe389 Chi2 (rad)")
        ax3.set_ylabel("Free Energy (kJ/mol)")
        ax3.grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig("monitor_status.png")
        plt.close()

        print(f"当前模拟时间: {time_ns[-1]:.2f} ns", end="")
        if barrier is not None:
            print(f" | ΔF ≈ {barrier:.1f} kJ/mol")
        else:
            print("")

    except Exception as e:
        print("监控绘图失败:", e)


if __name__ == "__main__":
    plot_monitor()
