#!/usr/bin/env python3
"""
train_efficacy_model.py
基于T-stacking相互作用强度预测D2受体激动剂效能

功能:
1. 从run_analysis_v2.py的结果中提取特征 (Strength_Combined等)
2. 加载效能标签 (labels.csv)
3. 训练多个机器学习模型
4. 评估模型性能
5. 可视化预测结果
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # 在导入pyplot之前设置后端
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import (
    train_test_split, cross_val_score, LeaveOneOut, KFold
)
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import warnings

warnings.filterwarnings('ignore')

# 设置绘图风格
sns.set_style("whitegrid")
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']

class EfficacyPredictor:
    """效能预测模型"""
    
    def __init__(self, results_dir='./results', labels_file='labels.csv', output_dir='./efficacy_models'):
        """
        初始化预测器
        
        参数:
        - results_dir: 分析结果目录
        - labels_file: 效能标签文件
        - output_dir: 模型输出目录
        """
        self.results_dir = results_dir
        self.labels_file = labels_file
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        self.features_df = None
        self.labels_df = None
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.models = {}
        self.predictions = {}
        self.metrics = {}
        self.cv_predictions = {}  # 用于存储交叉验证预测
        self.loo_predictions = {}  # 用于存储LOO预测
        
        print("=" * 80)
        print("效能预测模型初始化")
        print("=" * 80)
    
    def load_labels(self):
        """加载效能标签"""
        print("\n【第一步】加载效能标签...")
        self.labels_df = pd.read_csv(self.labels_file)
        print(f"✓ 已加载 {len(self.labels_df)} 个化合物的效能标签")
        print(f"  效能范围: {self.labels_df['Efficacy'].min():.2f} - {self.labels_df['Efficacy'].max():.2f}")
        return self.labels_df
    
    def extract_features(self):
        """从分析结果中提取特征"""
        print("\n【第二步】提取T-stacking特征...")
        
        features_list = []
        
        # 遍历results目录
        if not os.path.exists(self.results_dir):
            print("⚠ results/ 目录不存在，尝试先运行分析...")
            os.system('python run_analysis_v2.py')
        
        # 查找所有All_Stats.csv文件
        for compound_dir in os.listdir(self.results_dir):
            compound_path = os.path.join(self.results_dir, compound_dir)
            if not os.path.isdir(compound_path):
                continue
            
            stats_file = os.path.join(compound_path, 'All_Stats.csv')
            if not os.path.exists(stats_file):
                print(f"⚠ 缺失 {compound_dir}/All_Stats.csv")
                continue
            
            try:
                # 读取统计文件
                stats_df = pd.read_csv(stats_file)
                
                # 取AVERAGE行 (最后一行)
                avg_row = stats_df[stats_df['Replica'] == 'AVERAGE']
                if len(avg_row) == 0:
                    avg_row = stats_df.iloc[[-1]]
                
                # 获取AVERAGE行的数据
                avg_row_dict = avg_row.iloc[0].to_dict()
                
                # 提取关键特征
                feature = {
                    'Compound_ID': compound_dir,
                    'Strength_389': self._safe_extract_value(avg_row_dict, 'strength_389'),
                    'Strength_390': self._safe_extract_value(avg_row_dict, 'strength_390'),
                    'Strength_Combined': self._safe_extract_value(avg_row_dict, 'strength_combined'),
                    'Quality_Score_389': self._safe_extract_value(avg_row_dict, 'quality_score_389'),
                    'Quality_Score_390': self._safe_extract_value(avg_row_dict, 'quality_score_390'),
                    'Avg_Angle_389': self._safe_extract_value(avg_row_dict, 'avg_angle_389'),
                    'Avg_Angle_390': self._safe_extract_value(avg_row_dict, 'avg_angle_390'),
                    'Weighted_Distance_389': self._safe_extract_value(avg_row_dict, 'Dist_Phe389_Weighted_Mean'),
                    'Weighted_Distance_390': self._safe_extract_value(avg_row_dict, 'Dist_Phe390_Weighted_Mean'),
                }
                
                # 过滤有效特征
                valid_features = {}
                for k, v in feature.items():
                    if v is not None:
                        try:
                            val_float = float(v)
                            if not np.isnan(val_float):
                                valid_features[k] = val_float
                        except (ValueError, TypeError):
                            pass
                
                if len(valid_features) > 1:  # 至少有化合物ID和一个特征
                    valid_features['Compound_ID'] = compound_dir
                    features_list.append(valid_features)
                    strength = valid_features.get('Strength_Combined', 'N/A')
                    if isinstance(strength, float):
                        print(f"✓ {compound_dir:<20} - 已提取特征 (Strength_Combined: {strength:.3f})")
                    else:
                        print(f"✓ {compound_dir:<20} - 已提取特征")
            
            except Exception as e:
                print(f"✗ {compound_dir}: {str(e)}")
        
        self.features_df = pd.DataFrame(features_list)
        print(f"\n✓ 已提取 {len(self.features_df)} 个化合物的特征")
        return self.features_df
    
    def _safe_extract_value(self, data_dict, key):
        """安全地从字典中提取值"""
        # 精确匹配首先
        if key in data_dict:
            try:
                val = data_dict[key]
                if val is not None and val != '' and str(val).lower() != 'nan':
                    return float(val)
            except (ValueError, TypeError):
                pass
        
        # 然后尝试模糊匹配
        for k, v in data_dict.items():
            if isinstance(k, str) and key.lower() in k.lower():
                try:
                    if v is not None and v != '' and str(v).lower() != 'nan':
                        return float(v)
                except (ValueError, TypeError):
                    pass
        
        return None
    
    def prepare_data(self):
        """准备训练数据"""
        print("\n【第三步】准备训练数据...")
        
        # 合并特征和标签
        # 需要匹配化合物名称
        merged_data = []
        
        for _, label_row in self.labels_df.iterrows():
            compound_name = label_row['Compound']
            efficacy = label_row['Efficacy']
            
            # 在features_df中查找匹配的化合物
            matching_features = self.features_df[
                self.features_df['Compound_ID'].str.contains(compound_name, case=False, na=False)
            ]
            
            if len(matching_features) > 0:
                # 取第一个匹配
                feature_row = matching_features.iloc[0].to_dict()
                feature_row['Efficacy'] = efficacy
                feature_row['Compound_Name'] = compound_name
                merged_data.append(feature_row)
                print(f"✓ {compound_name:<25} - 效能: {efficacy:>7.2f}")
            else:
                print(f"✗ {compound_name:<25} - 未找到特征数据")
        
        merged_df = pd.DataFrame(merged_data)
        print(f"\n✓ 已合并 {len(merged_df)} 个样本的特征和标签")
        
        # 选择特征列
        feature_cols = [col for col in merged_df.columns 
                       if col not in ['Compound_ID', 'Compound_Name', 'Efficacy']]
        
        self.X = merged_df[feature_cols].values
        self.y = merged_df['Efficacy'].values
        
        print(f"✓ 特征矩阵形状: {self.X.shape}")
        print(f"✓ 特征列: {', '.join(feature_cols)}")
        
        # 数据分割
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )
        
        # 标准化
        scaler = StandardScaler()
        self.X_train = scaler.fit_transform(self.X_train)
        self.X_test = scaler.transform(self.X_test)
        self.scaler = scaler
        
        print(f"✓ 训练集: {len(self.X_train)} | 测试集: {len(self.X_test)}")
        
        return merged_df
    
    def train_models(self):
        """训练多个模型 - 使用多层交叉验证"""
        print("\n【第四步】训练模型...")
        
        # 定义模型
        models_to_train = {
            'Linear Regression': LinearRegression(),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42, learning_rate=0.1),
            'SVR': SVR(kernel='rbf', C=100, epsilon=0.1)
        }
        
        for name, model in models_to_train.items():
            print(f"\n训练 {name}...")
            
            # 1. Leave-One-Out (LOO) 交叉验证 - 在完整数据集上进行
            loo = LeaveOneOut()
            loo_predictions = np.zeros_like(self.y, dtype=float)
            
            for train_idx, test_idx in loo.split(self.X):
                X_train_loo = self.X[train_idx]
                X_test_loo = self.X[test_idx]
                y_train_loo = self.y[train_idx]
                y_test_loo = self.y[test_idx]
                
                model_loo = type(model)(**model.get_params())
                model_loo.fit(X_train_loo, y_train_loo)
                loo_predictions[test_idx] = model_loo.predict(X_test_loo)
            
            loo_r2 = r2_score(self.y, loo_predictions)
            loo_rmse = np.sqrt(mean_squared_error(self.y, loo_predictions))
            loo_mae = mean_absolute_error(self.y, loo_predictions)
            
            # 2. K-Fold 交叉验证 - 调整分组数以适应小数据集
            n_splits = min(3, len(self.X))  # 7个样本时用3-fold
            kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)
            
            kfold_predictions = np.zeros_like(self.y, dtype=float)
            for train_idx, test_idx in kfold.split(self.X):
                X_train_kf = self.X[train_idx]
                X_test_kf = self.X[test_idx]
                y_train_kf = self.y[train_idx]
                
                model_kf = type(model)(**model.get_params())
                model_kf.fit(X_train_kf, y_train_kf)
                kfold_predictions[test_idx] = model_kf.predict(X_test_kf)
            
            kfold_r2 = r2_score(self.y, kfold_predictions)
            kfold_rmse = np.sqrt(mean_squared_error(self.y, kfold_predictions))
            kfold_mae = mean_absolute_error(self.y, kfold_predictions)
            
            # 3. 标准训练/测试划分 - 用于参考
            model.fit(self.X_train, self.y_train)
            y_pred_train = model.predict(self.X_train)
            y_pred_test = model.predict(self.X_test)
            
            r2_train = r2_score(self.y_train, y_pred_train)
            r2_test = r2_score(self.y_test, y_pred_test)
            rmse_test = np.sqrt(mean_squared_error(self.y_test, y_pred_test))
            mae_test = mean_absolute_error(self.y_test, y_pred_test)
            
            self.models[name] = model
            self.predictions[name] = {
                'y_pred_train': y_pred_train,
                'y_pred_test': y_pred_test
            }
            self.cv_predictions[name] = loo_predictions
            self.metrics[name] = {
                'R2_Train': r2_train,
                'R2_Test': r2_test,
                'RMSE_Test': rmse_test,
                'MAE_Test': mae_test,
                'KFold_CV_R2': kfold_r2,
                'KFold_CV_RMSE': kfold_rmse,
                'KFold_CV_MAE': kfold_mae,
                'LOO_R2': loo_r2,
                'LOO_RMSE': loo_rmse,
                'LOO_MAE': loo_mae
            }
            
            print(f"  ✓ R² (训练): {r2_train:.4f} | R² (测试): {r2_test:.4f}")
            print(f"  ✓ RMSE: {rmse_test:.4f} | MAE: {mae_test:.4f}")
            print(f"  ✓ K-Fold CV R²: {kfold_r2:.4f}")
            print(f"  ✓ Leave-One-Out R²: {loo_r2:.4f}")
        
        return self.metrics
    
    def evaluate_models(self):
        """评估和比较模型"""
        print("\n【第五步】模型评估...")
        
        metrics_df = pd.DataFrame(self.metrics).T
        print("\n全面的模型性能对比 (多种验证方式):")
        print(metrics_df.to_string())
        
        # 基于Leave-One-Out R²选择最佳模型（最严格的评估）
        best_model_name = metrics_df['LOO_R2'].idxmax()
        best_loo_r2 = metrics_df.loc[best_model_name, 'LOO_R2']
        best_kfold_r2 = metrics_df.loc[best_model_name, 'KFold_CV_R2']
        best_test_r2 = metrics_df.loc[best_model_name, 'R2_Test']
        
        print(f"\n✓ 最佳模型: {best_model_name}")
        print(f"  Leave-One-Out R²: {best_loo_r2:.4f} (最严格的评估)")
        print(f"  K-Fold CV R²: {best_kfold_r2:.4f}")
        print(f"  测试集 R²: {best_test_r2:.4f}")
        
        print(f"\n✅ 交叉验证方式对比:")
        print(f"  - Leave-One-Out: 最严格，评估每个样本作为测试集")
        print(f"  - K-Fold (3-fold): 平衡的交叉验证，使用全部数据")
        print(f"  - 测试集 (25% hold-out): 简单的hold-out验证")
        
        return metrics_df
    
    def plot_results(self, merged_df):
        """绘制结果可视化"""
        print("\n【第六步】可视化预测结果...")
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('T-Stacking 相互作用强度 vs 效能预测', fontsize=16, fontweight='bold')
        
        # 1. 特征与效能的关系
        ax = axes[0, 0]
        if 'Strength_Combined' in merged_df.columns:
            ax.scatter(merged_df['Strength_Combined'], merged_df['Efficacy'], s=100, alpha=0.6, color='steelblue')
            z = np.polyfit(merged_df['Strength_Combined'].dropna(), 
                          merged_df.loc[merged_df['Strength_Combined'].notna(), 'Efficacy'], 1)
            p = np.poly1d(z)
            x_line = np.linspace(merged_df['Strength_Combined'].min(), merged_df['Strength_Combined'].max(), 100)
            ax.plot(x_line, p(x_line), "r--", alpha=0.8, linewidth=2)
            
            corr = merged_df['Strength_Combined'].corr(merged_df['Efficacy'])
            ax.set_title(f'Strength_Combined vs 效能 (r={corr:.3f})', fontweight='bold')
            ax.set_xlabel('Strength_Combined')
            ax.set_ylabel('效能 (%)')
            ax.grid(True, alpha=0.3)
        
        # 2. 模型预测对比
        ax = axes[0, 1]
        best_model_name = max(self.metrics, key=lambda x: self.metrics[x]['R2_Test'])
        y_pred = self.predictions[best_model_name]['y_pred_test']
        
        ax.scatter(self.y_test, y_pred, s=100, alpha=0.6, color='green')
        ax.plot([self.y_test.min(), self.y_test.max()], 
               [self.y_test.min(), self.y_test.max()], 'r--', lw=2)
        
        r2 = self.metrics[best_model_name]['R2_Test']
        ax.set_title(f'预测值 vs 真实值 ({best_model_name})\nR²={r2:.4f}', fontweight='bold')
        ax.set_xlabel('True Efficacy')
        ax.set_ylabel('Predicted Efficacy')
        ax.grid(True, alpha=0.3)
        
        # 3. 模型R²对比
        ax = axes[1, 0]
        r2_values = [self.metrics[m]['R2_Test'] for m in self.models.keys()]
        colors = ['green' if v == max(r2_values) else 'steelblue' for v in r2_values]
        ax.barh(list(self.models.keys()), r2_values, color=colors, alpha=0.7)
        ax.set_xlabel('R² (Test Set)', fontweight='bold')
        ax.set_title('模型性能对比', fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        for i, v in enumerate(r2_values):
            ax.text(v + 0.01, i, f'{v:.4f}', va='center')
        
        # 4. 残差分析
        ax = axes[1, 1]
        residuals = self.y_test - y_pred
        ax.scatter(y_pred, residuals, s=100, alpha=0.6, color='orange')
        ax.axhline(y=0, color='r', linestyle='--', lw=2)
        ax.set_xlabel('Predicted Efficacy')
        ax.set_ylabel('Residuals')
        ax.set_title('残差分析', fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_file = os.path.join(self.output_dir, 'prediction_results.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"✓ 已保存可视化: {output_file}")
        plt.close()
    
    def save_results(self, merged_df, metrics_df):
        """保存结果到CSV"""
        print("\n【第七步】保存结果...")
        
        # 保存特征和预测
        results_data = merged_df.copy()
        loo_results = merged_df.copy()
        for model_name in self.models.keys():
            y_pred = self.predictions[model_name]['y_pred_test']
            # 只保存测试集的预测
            results_data[f'Pred_{model_name}'] = np.nan
            test_indices = np.arange(len(self.y_test))
            results_data.loc[results_data.index[-len(test_indices):], f'Pred_{model_name}'] = y_pred
            loo_pred = self.cv_predictions[model_name]  # LOO 的预测值
            loo_results[f'LOO_Pred_{model_name}'] = loo_pred

        output_file = os.path.join(self.output_dir, 'efficacy_predictions.csv')
        results_data.to_csv(output_file, index=False)
        print(f"✓ 已保存预测结果: {output_file}")
        
        loo_file = os.path.join(self.output_dir, 'efficacy_predictions_LOO.csv')
        loo_results.to_csv(loo_file, index=False)
        print(f"✓ 已保存 LOO 预测结果: {loo_file}")
        # 保存模型指标
        metrics_file = os.path.join(self.output_dir, 'model_metrics.csv')
        metrics_df.to_csv(metrics_file)
        print(f"✓ 已保存模型指标: {metrics_file}")
    
    def run_pipeline(self):
        """运行完整的预测流程"""
        try:
            self.load_labels()
            self.extract_features()
            merged_df = self.prepare_data()
            self.train_models()
            metrics_df = self.evaluate_models()
            self.plot_results(merged_df)
            self.save_results(merged_df, metrics_df)
            
            print("\n" + "=" * 80)
            print("✅ 效能预测模型训练完成！")
            print("=" * 80)
            print(f"\n输出目录: {self.output_dir}/")
            print("  - prediction_results.png: 可视化结果")
            print("  - efficacy_predictions.csv: 预测数据")
            print("  - model_metrics.csv: 模型指标")
            
        except Exception as e:
            print(f"\n❌ 错误: {str(e)}")
            import traceback
            traceback.print_exc()


def main():
    """主函数"""
    predictor = EfficacyPredictor()
    predictor.run_pipeline()


if __name__ == '__main__':
    main()
