#!/usr/bin/env python3
import os
import glob
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from src.analysis.feature_names import get_feature_names


# =========================
# 配置区
# =========================

FEATURE_ROOT = "data/features"   # 存放 *.npy 的目录
RANDOM_SEED = 42
N_TREES = 500
TOP_K = 30


# =========================
# 数据加载
# =========================

def load_npy_dataset(root_dir):
    X_list = []
    y_list = []

    npy_files = sorted(
        glob.glob(os.path.join(root_dir, "**", "*_features.npy"), recursive=True)
    )

    if not npy_files:
        raise RuntimeError(f"No *_features.npy found under {root_dir}")

    print(f"Found {len(npy_files)} feature files\n")

    for f in npy_files:
        fpath = Path(f)

        # compound 名 = features/下面的第一级目录
        compound = fpath.parents[1].name

        data = np.load(f)
        if data.ndim != 2:
            print(f"[Skip] {f} invalid shape {data.shape}")
            continue

        X_list.append(data)
        y_list.extend([compound] * data.shape[0])

        print(f"[Load] {compound:<35s} -> {data.shape[0]:4d} frames")

    X = np.vstack(X_list)
    y = np.array(y_list)

    print(f"\nTotal samples: {X.shape[0]}")
    print(f"Feature dim  : {X.shape[1]}")
    print(f"Compounds   : {len(set(y))}")

    return X, y


# =========================
# 主流程
# =========================

def main():
    # 1. 读取数据
    X, y = load_npy_dataset(FEATURE_ROOT)

    # 2. 特征名
    feature_names = get_feature_names()
    assert len(feature_names) == X.shape[1], \
        f"Feature name mismatch: {len(feature_names)} vs {X.shape[1]}"

    # 3. 划分训练 / 测试
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.25,
        random_state=RANDOM_SEED,
        stratify=y
    )

    # 4. 训练 RF
    clf = RandomForestClassifier(
        n_estimators=N_TREES,
        random_state=RANDOM_SEED,
        n_jobs=-1,
        class_weight="balanced"
    )

    print("\nTraining RandomForest...")
    clf.fit(X_train, y_train)

    # 5. 评估
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"\nClassification accuracy: {acc:.4f}")

    # 6. 特征重要性
    importances = clf.feature_importances_
    order = np.argsort(importances)[::-1]

    print("\n>>> Top Feature Importances (RF):")
    print(f"{'Rank':<5} | {'Feature':<35} | {'Importance':<10}")
    print("-" * 60)

    rows = []
    for i in range(min(TOP_K, len(order))):
        idx = order[i]
        rows.append({
            "Rank": i + 1,
            "Feature": feature_names[idx],
            "Importance": importances[idx]
        })
        print(f"{i+1:<5} | {feature_names[idx]:<35} | {importances[idx]:.6f}")

    # 7. 保存 CSV
    df = pd.DataFrame(rows)
    out_csv = "rf_feature_importance.csv"
    df.to_csv(out_csv, index=False)
    print(f"\n[Save] Feature importance saved to {out_csv}")


if __name__ == "__main__":
    main()
