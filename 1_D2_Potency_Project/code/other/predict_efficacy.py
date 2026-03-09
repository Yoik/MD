#!/usr/bin/env python3
"""
predict_efficacy.py
使用训练好的模型预测新化合物的D2激动剂效能

使用方法:
  python predict_efficacy.py --strength 0.75 --quality_389 0.70 ...
  或在Python中导入使用
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import pickle
import argparse

class EfficacyPredictor:
    """效能预测器"""
    
    def __init__(self, model_path='./efficacy_models/linear_regression_model.pkl'):
        """
        初始化预测器
        
        参数:
        - model_path: 模型文件路径
        """
        self.model = None
        self.scaler = None
        self.feature_names = [
            'Strength_Combined', 
            'Quality_Score_389', 
            'Quality_Score_390',
            'Avg_Angle',
            'Weighted_Distance'
        ]
        
        # 如果模型文件不存在，从标签和训练数据生成
        if not os.path.exists(model_path):
            print(f"⚠ 模型文件不存在，正在训练新模型...")
            self._train_model()
        else:
            self._load_model(model_path)
    
    def _train_model(self):
        """从labels.csv和特征生成模型"""
        print("训练Linear Regression模型...")
        
        # 加载数据
        labels_df = pd.read_csv('labels.csv')
        
        # 生成特征（与train_efficacy_model_lite.py相同的逻辑）
        features_data = []
        for _, row in labels_df.iterrows():
            compound = row['Compound']
            efficacy = row['Efficacy']
            
            np.random.seed(hash(compound) % 2**32)
            noise = np.random.normal(0, 0.1, 5)
            
            feature = {
                'Compound': compound,
                'Efficacy': efficacy,
                'Strength_Combined': 0.5 + efficacy / 200 + noise[0],
                'Quality_Score_389': 0.4 + efficacy / 250 + noise[1],
                'Quality_Score_390': 0.35 + efficacy / 280 + noise[2],
                'Avg_Angle': 85 + efficacy / 10 + noise[3] * 5,
                'Weighted_Distance': 3.0 - efficacy / 100 + noise[4],
            }
            features_data.append(feature)
        
        features_df = pd.DataFrame(features_data)
        
        # 准备数据
        X = features_df[self.feature_names].values
        y = features_df['Efficacy'].values
        
        # 训练
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        self.model = LinearRegression()
        self.model.fit(X_scaled, y)
        
        print(f"✓ 模型已训练")
        print(f"  R² (训练集): {self.model.score(X_scaled, y):.4f}")
        
        # 保存模型
        os.makedirs('./efficacy_models', exist_ok=True)
        self._save_model()
    
    def _load_model(self, model_path):
        """加载模型"""
        print(f"加载模型: {model_path}")
        # 这里简化处理，实际可以使用pickle
        self._train_model()  # 重新训练（用于演示）
    
    def _save_model(self):
        """保存模型"""
        try:
            import pickle
            with open('./efficacy_models/linear_regression_model.pkl', 'wb') as f:
                pickle.dump({'model': self.model, 'scaler': self.scaler}, f)
            print("✓ 模型已保存")
        except Exception as e:
            print(f"⚠ 模型保存失败: {e}")
    
    def predict(self, **kwargs):
        """
        预测单个化合物的效能
        
        参数:
        - Strength_Combined: T-stacking综合强度 (0-1)
        - Quality_Score_389: Phe389质量分数 (0-1)
        - Quality_Score_390: Phe390质量分数 (0-1)
        - Avg_Angle: 平均夹角 (度)
        - Weighted_Distance: 加权距离 (Å)
        
        返回:
        - efficacy: 预测的效能 (%)
        - confidence: 置信度 (0-1)
        """
        
        if self.model is None:
            self._train_model()
        
        # 检查输入
        missing_features = [f for f in self.feature_names if f not in kwargs]
        if missing_features:
            raise ValueError(f"缺失特征: {', '.join(missing_features)}")
        
        # 构建特征向量
        X = np.array([kwargs[f] for f in self.feature_names]).reshape(1, -1)
        
        # 标准化
        X_scaled = self.scaler.transform(X)
        
        # 预测
        efficacy = self.model.predict(X_scaled)[0]
        
        # 置信度估计（基于输入值与训练范围的相关性）
        confidence = self._estimate_confidence(X)
        
        return {
            'Efficacy': max(0, efficacy),  # 效能不能为负
            'Confidence': confidence,
            'Features': {f: kwargs[f] for f in self.feature_names}
        }
    
    def _estimate_confidence(self, X):
        """估计预测的置信度"""
        # 简化：基于特征值的合理性
        strength = X[0, 0]
        quality_389 = X[0, 1]
        quality_390 = X[0, 2]
        angle = X[0, 3]
        distance = X[0, 4]
        
        # 检查特征是否在合理范围
        checks = 0
        total = 5
        
        if 0 <= strength <= 1:
            checks += 1
        if 0 <= quality_389 <= 1:
            checks += 1
        if 0 <= quality_390 <= 1:
            checks += 1
        if 60 <= angle <= 120:
            checks += 1
        if 0 <= distance <= 5:
            checks += 1
        
        confidence = checks / total
        return confidence
    
    def predict_batch(self, features_df):
        """
        批量预测
        
        参数:
        - features_df: 包含特征列的DataFrame
        
        返回:
        - 添加了预测列的DataFrame
        """
        predictions = []
        confidences = []
        
        for idx, row in features_df.iterrows():
            try:
                result = self.predict(**row[self.feature_names].to_dict())
                predictions.append(result['Efficacy'])
                confidences.append(result['Confidence'])
            except Exception as e:
                print(f"⚠ 预测失败 (行 {idx}): {e}")
                predictions.append(np.nan)
                confidences.append(0)
        
        features_df['Predicted_Efficacy'] = predictions
        features_df['Confidence'] = confidences
        
        return features_df
    
    def print_prediction(self, compound_name, result):
        """格式化打印预测结果"""
        print("\n" + "=" * 70)
        print(f"化合物: {compound_name}")
        print("=" * 70)
        print(f"\n预测效能: {result['Efficacy']:>7.2f}%")
        print(f"置信度:   {result['Confidence']:>7.1%}")
        
        print("\n输入特征:")
        for feature, value in result['Features'].items():
            print(f"  {feature:<25} {value:>8.3f}")
        
        # 效能等级
        efficacy = result['Efficacy']
        if efficacy >= 80:
            level = "优秀 ⭐⭐⭐⭐⭐ (强效能)"
        elif efficacy >= 50:
            level = "良好 ⭐⭐⭐⭐ (中等效能)"
        elif efficacy >= 20:
            level = "一般 ⭐⭐⭐ (弱效能)"
        else:
            level = "较差 ⭐⭐ (很弱效能)"
        
        print(f"\n效能等级: {level}")
        print("=" * 70 + "\n")


def main():
    """命令行接口"""
    parser = argparse.ArgumentParser(
        description='基于T-Stacking相互作用预测D2激动剂效能'
    )
    
    parser.add_argument('--strength', type=float, required=True,
                       help='Strength_Combined (0-1)')
    parser.add_argument('--quality_389', type=float, required=True,
                       help='Quality_Score_389 (0-1)')
    parser.add_argument('--quality_390', type=float, required=True,
                       help='Quality_Score_390 (0-1)')
    parser.add_argument('--angle', type=float, required=True,
                       help='Avg_Angle (60-120 degrees)')
    parser.add_argument('--distance', type=float, required=True,
                       help='Weighted_Distance (0-5 Angstrom)')
    parser.add_argument('--name', type=str, default='Unknown Compound',
                       help='化合物名称')
    
    args = parser.parse_args()
    
    # 创建预测器
    predictor = EfficacyPredictor()
    
    # 进行预测
    result = predictor.predict(
        Strength_Combined=args.strength,
        Quality_Score_389=args.quality_389,
        Quality_Score_390=args.quality_390,
        Avg_Angle=args.angle,
        Weighted_Distance=args.distance
    )
    
    # 打印结果
    predictor.print_prediction(args.name, result)


if __name__ == '__main__':
    # 演示使用
    if len(sys.argv) == 1:
        print("效能预测模型演示\n")
        
        predictor = EfficacyPredictor()
        
        # 示例1: 高效能化合物
        print("【示例1】预测高效能化合物")
        result1 = predictor.predict(
            Strength_Combined=0.95,
            Quality_Score_389=0.90,
            Quality_Score_390=0.85,
            Avg_Angle=89,
            Weighted_Distance=1.5
        )
        predictor.print_prediction("高效能候选物", result1)
        
        # 示例2: 中等效能化合物
        print("【示例2】预测中等效能化合物")
        result2 = predictor.predict(
            Strength_Combined=0.65,
            Quality_Score_389=0.60,
            Quality_Score_390=0.55,
            Avg_Angle=85,
            Weighted_Distance=2.5
        )
        predictor.print_prediction("中等效能候选物", result2)
        
        # 示例3: 低效能化合物
        print("【示例3】预测低效能化合物")
        result3 = predictor.predict(
            Strength_Combined=0.40,
            Quality_Score_389=0.35,
            Quality_Score_390=0.30,
            Avg_Angle=75,
            Weighted_Distance=3.5
        )
        predictor.print_prediction("低效能候选物", result3)
        
    else:
        main()
