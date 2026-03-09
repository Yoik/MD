import os
import json
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from src.config import init_config
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-i", type=str, default=None)
args = parser.parse_args()

config = init_config()
RESULT_DIR = args.i or config.get_path("paths.result_dir")

# 初始化配置
# config = init_config()
# RESULT_DIR = config.get_path("paths.result_dir")

def main():
    print(">>> Starting Plotting Pipeline...")
    
    # 设置画布：使用 GridSpec 实现非均匀布局 (2行2列)
    # 上面两张图并排，下面一张图跨列
    fig = plt.figure(figsize=(20, 14))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.2]) # 下面的图稍微高一点
    
    sns.set_style("whitegrid")
    
    # =================================================================
    # 图 1 (左上): 排名相关性 (Ranking Correlation)
    # =================================================================
    ax1 = fig.add_subplot(gs[0, 0])
    csv_path = os.path.join(RESULT_DIR, "loocv_results.csv")
    
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        summary = df.groupby('Compound').agg({
            'Pred_Diff_Score': 'mean',
            'True_Eff': 'first',
            'Correct': 'first'
        }).reset_index()

        accuracy = summary['Correct'].mean()
        rho, p = stats.spearmanr(summary['True_Eff'], summary['Pred_Diff_Score'])
        
        sns.scatterplot(
            data=summary, x='True_Eff', y='Pred_Diff_Score', 
            hue='Correct', palette={1: 'green', 0: 'red'}, 
            s=150, style='Correct', markers={1: 'o', 0: 'X'}, ax=ax1
        )
        
        ax1.axhline(0, color='gray', linestyle='--', label='Ref (Dopa)')
        
        # 标注
        for i, row in summary.iterrows():
            ax1.text(
                row['True_Eff'] + 0.5, row['Pred_Diff_Score'], 
                row['Compound'], fontsize=11, fontweight='bold', alpha=0.8
            )
            
        ax1.set_title(f"A. Ranking Correlation\nSpearman Rho = {rho:.3f} | Acc = {accuracy:.0%}", fontsize=16)
        ax1.set_xlabel("Experimental Efficacy (%)", fontsize=14)
        ax1.set_ylabel("Predicted Score Diff (Tanh Scale)", fontsize=14)
        ax1.legend(loc='upper left')
    else:
        ax1.text(0.5, 0.5, "No Ranking CSV Found", ha='center', fontsize=14)

    # =================================================================
    # 图 2 (右上): Loss 曲线 (Test Loss)
    # =================================================================
    ax2 = fig.add_subplot(gs[0, 1])
    json_path = os.path.join(RESULT_DIR, "loocv_losses.json")
    
    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            all_fold_losses = json.load(f)
            
        compounds = list(all_fold_losses.keys())
        palette = sns.color_palette("husl", len(compounds))
        
        for i, (cmpd_name, losses) in enumerate(all_fold_losses.items()):
            # 优先画 Test Loss
            if isinstance(losses, dict):
                loss_data = losses.get('test', [])
                label_prefix = "Test"
            else:
                loss_data = losses
                
            if loss_data:
                ax2.plot(loss_data, label=f"{cmpd_name}", linewidth=2, alpha=0.8, color=palette[i])

        ax2.set_title("B. LOO-CV Validation Loss (OOD Detection)", fontsize=16)
        ax2.set_xlabel("Epoch", fontsize=14)
        ax2.set_ylabel("Test Loss", fontsize=14)
        ax2.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0., title="Hold-out Cmpd")
    else:
        ax2.text(0.5, 0.5, "No Loss JSON Found", ha='center', fontsize=14)

    # =================================================================
    # 图 3 (下方): Top 15 特征重要性 (Feature Importance)
    # =================================================================
    ax3 = fig.add_subplot(gs[1, :]) # 跨越两列
    feat_path = os.path.join(RESULT_DIR, "feature_importance.csv")
    
    if os.path.exists(feat_path):
        df_feat = pd.read_csv(feat_path)
        # 取前 15 名
        top_df = df_feat.head(15).copy()
        
        # 绘制水平条形图
        sns.barplot(
            data=top_df, x="Mask_Value", y="Feature", 
            palette="viridis", ax=ax3, edgecolor='black'
        )
        
        ax3.set_title("C. Top 15 Learned Pharmacophore Features (Dynamic Mask)", fontsize=16)
        ax3.set_xlabel("Importance Weight (0.0 - 1.0)", fontsize=14)
        ax3.set_ylabel("Feature Name", fontsize=14)
        ax3.set_xlim(0, 1.0) # 权重范围固定在 0-1
        
        # 在柱子上标数值
        for i, v in enumerate(top_df["Mask_Value"]):
            ax3.text(v + 0.01, i, f"{v:.4f}", color='black', va='center', fontweight='bold')
            
    else:
        ax3.text(0.5, 0.5, "No Feature CSV Found\n(Run training again with updated script)", ha='center', fontsize=14)

    # =================================================================
    # 保存
    # =================================================================
    plt.tight_layout()
    save_path = os.path.join(RESULT_DIR, "final_analysis_panel.png")
    plt.savefig(save_path, dpi=300)
    print(f"[Plot] Panel saved to: {save_path}")
    plt.close()

if __name__ == "__main__":
    main()