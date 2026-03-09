import pandas as pd
import numpy as np
from pathlib import Path

# ================= 路径 =================
IN_CSV  = Path("data/Emax_rank_PPT.csv")
TEMPLATE_LABEL = Path("data/labels_template.csv")
OUT_CSV = Path("data/labels.csv")

# ================= 名称映射 =================
NAME_MAP = {
    "Dopa": "Dopamine",
    "ROT": "Rotigotine",
    "UNC": "UNC2458A",
    "R10": "(R)-IHCH-7010",
    "S10": "(S)-IHCH-7010",
    "S84": "(S)-IHCH-7084",
    "CAR": "Cariprazine",
    "ARI": "Aripiprazole",
    "BRE": "Brexpiprazole",
    "Lisu": "Lisuride",
    "LSD": "LSD",
    "Pramipexole": "Pramipexole",
    "(R)-IHCH-7084": "(R)-IHCH-7084",
    "(R)-IHCH-7041": "(R)-IHCH-7041",
    "(S)-IHCH-7041": "(S)-IHCH-7041",
}

# ================= 工具函数 =================
def clip_and_quantile(col, dopa_value):
    """
    1. 先把所有值 clip 到 dopamine
    2. 再在该体系内做 rank → quantile
    """
    x = col.astype(float)

    # --- 核心：clip ---
    x_clipped = np.minimum(x, dopa_value)

    # --- rank → quantile ---
    n = x_clipped.notna().sum()
    r = x_clipped.rank(method="average", ascending=True, na_option="keep")
    q = (r - 1) / (n - 1)

    return q

# ================= 主流程 =================
def main():
    # 1. 读取 8-system 表
    df = pd.read_csv(IN_CSV, index_col=0)
    df.index.name = "Compound"

    # 2. 对每个体系：clip at dopamine → quantile
    q_df = pd.DataFrame(index=df.index)

    for col in df.columns:
        dopa_val = df.loc["Dopamine", col]
        q_df[col] = clip_and_quantile(df[col], dopa_val)

    # 3. 跨体系取中位数（共识点火强度）
    consensus = q_df.median(axis=1, skipna=True)

    # 4. 读取模板 labels（只用名字和顺序）
    tmpl = pd.read_csv(TEMPLATE_LABEL)

    new_labels = []
    for _, row in tmpl.iterrows():
        cmpd = row["Compound"]
        src_name = NAME_MAP[cmpd]

        new_labels.append({
            "Compound": cmpd,
            "Efficacy": consensus.loc[src_name]
        })

    out_df = pd.DataFrame(new_labels)
    out_df.to_csv(OUT_CSV, index=False)

    print("[OK] labels.csv generated (clip + quantile consensus)")

if __name__ == "__main__":
    main()
