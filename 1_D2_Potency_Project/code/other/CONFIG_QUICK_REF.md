# 配置系统快速参考卡片

## 三步改造任何脚本

```python
# 1️⃣  导入配置（在脚本顶部）
from src.config import init_config
config = init_config()

# 2️⃣  替换硬编码参数（找到原有配置，替换成下面的格式）
OLD: LEARNING_RATE = 0.001
NEW: LEARNING_RATE = config.get_float("training.learning_rate")

# 3️⃣  修改参数只需编辑 config.yaml
learning_rate: 0.0005  # ← 改这里，脚本自动生效
```

## 配置键速查表

### 路径 (paths)
```yaml
paths.label_file         → "data/labels.csv"
paths.result_dir         → "data/features"
paths.qc_output_dir      → "data/qc_structures"
paths.reference_pdb      → "data/step5_input.pdb"
paths.model_path         → "saved_models/best_model_mccv.pth"
paths.scaler_path        → "saved_models/scaler.pkl"
```

### 数据 (data)
```yaml
data.integration_radius       → 1.5
data.pocket_atom_num          → 12
data.input_dim_features       → 151
data.input_dim_eval          → 19
data.standard_sequence       → "MDPLNLSWYDDD..."
```

### 残基 (residues)
```yaml
residues.phe_residues        → [389, 390]
residues.obp_residues        → [114, 115, 118, 119, ...]
residues.plane_residues      → [114, 193, 197, 393]
```

### 训练 (training)
```yaml
training.learning_rate       → 0.001
training.dropout_rate        → 0.2
training.weight_decay        → 1e-4
training.num_epochs          → 60
training.batch_size          → 32
training.l1_lambda           → 0.005
training.window_size         → 100
training.stride              → 20
```

### 评估 (evaluation)
```yaml
evaluation.samples_per_level  → 500
```

### 绘图 (plotting)
```yaml
plotting.use_agg_backend     → true
plotting.style               → "default"
plotting.dpi                 → 100
```

## 获取方法速查

```python
# 字符串 / 路径
config.get("key")              # 直接获取
config.get_path("key")         # 获取路径

# 数值
config.get_int("key")          # 整数
config.get_float("key")        # 浮点数

# 集合
config.get_list("key")         # 列表
config.get_bool("key")         # 布尔值

# 带默认值
config.get("key", default=10)  # 获取失败返回默认值
```

## 实际代码示例

### 脚本开头（固定模板）
```python
import sys
import os
import torch
import numpy as np

# 导入配置 ← 加这个
from src.config import init_config
config = init_config()

# 然后把下面的改了
# ❌ LABEL_FILE = "data/labels.csv"
# ✅ LABEL_FILE = config.get_path("paths.label_file")
```

### 具体替换示例

```python
# 路径配置
LABEL_FILE = config.get_path("paths.label_file")
RESULT_DIR = config.get_path("paths.result_dir")
MODEL_PATH = config.get_path("paths.model_path")

# 数值配置
BATCH_SIZE = config.get_int("training.batch_size")
LEARNING_RATE = config.get_float("training.learning_rate")
INPUT_DIM = config.get_int("data.input_dim_features")

# 列表配置
OBP_RESIDUES = config.get_list("residues.obp_residues")
```

## 验证配置

```python
# 打印所有配置
config.to_dict()

# 获取特定配置
print(config.get("training.batch_size"))

# 验证配置是否存在
if config.get("paths.model_path"):
    print("配置成功！")
```

## 常见参数类型判断

```python
# 数字型 (用 get_int / get_float)
pocket_atom_num = 12
integration_radius = 1.5
num_epochs = 60

# 字符串/路径 (用 get_path / get)
label_file = "data/labels.csv"
reference_pdb = "data/step5_input.pdb"

# 列表 (用 get_list)
obp_residues = [114, 115, 118, 119, ...]
phe_residues = [389, 390]

# 浮点数 (用 get_float)
learning_rate = 0.001
dropout_rate = 0.2
```

## 改造检查清单

- [ ] 导入了 config: `from src.config import init_config`
- [ ] 初始化了 config: `config = init_config()`
- [ ] 替换了所有硬编码的配置参数
- [ ] 脚本能正常运行
- [ ] 修改 config.yaml 后，脚本使用新参数
- [ ] 提交代码时，包含了修改后的脚本和 config.yaml

## 无需改造的脚本部分

- 导入语句
- 函数定义
- 类定义
- 业务逻辑代码
- 只改"= 值"这种配置行

## 需要改造的配置示例

```python
# ✅ 这些需要改
LABEL_FILE = "data/labels.csv"              # 路径
LEARNING_RATE = 0.001                       # 数值
BATCH_SIZE = 32                              # 数值
NUM_EPOCHS = 60                              # 数值
OBP_RESIDUES = [114, 115, 118, 119, ...]   # 列表

# ❌ 这些不需要改
def load_data():
    pass

class Model:
    pass

if __name__ == "__main__":
    pass
```

## 更新多个脚本的建议顺序

1. **2_train_model.py** - 核心训练脚本
2. **3_evaluate_all.py** - 评估脚本
3. **1_extract_features.py** - 特征提取脚本
4. **8_generate_efficacy_map.py** - 图谱生成脚本
5. **其他脚本** - 根据需要

## 遇到问题

- 检查 config.yaml 是否在项目根目录
- 检查 yaml 格式是否正确（缩进用空格，不用 Tab）
- 运行 `python -c "from src.config import init_config; config = init_config()"` 测试
- 查看 CONFIG_USAGE.md 获取更多帮助
