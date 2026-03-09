# 配置系统改造步骤

## 📋 概览

已为你创建了统一的配置系统。现在可以通过编辑单一的 `config.yaml` 文件来管理所有脚本的参数，而不是逐个修改脚本。

## 📁 新增文件

1. **config.yaml** - 统一配置文件（项目根目录）
2. **src/config.py** - 配置加载模块
3. **CONFIG_USAGE.md** - 详细使用说明
4. **scripts/migrate_to_config.py** - 自动分析工具（可选）

## 🚀 快速开始（三步）

### 步骤 1: 在脚本顶部添加配置导入

在每个需要改造的脚本开头（导入部分之后）添加：

```python
from src.config import init_config

# 初始化配置
config = init_config()
```

### 步骤 2: 替换硬编码的配置变量

将原有的硬编码配置改为从 config 读取。例如：

**2_train_model.py 中的改造示例：**

```python
# ❌ 原来的方式（硬编码）
LABEL_FILE = "data/labels.csv"
RESULT_DIR = "data/features"
MODEL_SAVE_PATH = "saved_models/best_model_mccv.pth"
SCALER_SAVE_PATH = "saved_models/scaler.pkl"
POCKET_ATOM_NUM = 12
INPUT_DIM = 151
LEARNING_RATE = 0.001
DROPOUT_RATE = 0.2
WEIGHT_DECAY = 1e-4
NUM_EPOCHS = 60
BATCH_SIZE = 32
L1_LAMBDA = 0.005

# ✅ 改成这样
LABEL_FILE = config.get_path("paths.label_file")
RESULT_DIR = config.get_path("paths.result_dir")
MODEL_SAVE_PATH = config.get_path("paths.model_path")
SCALER_SAVE_PATH = config.get_path("paths.scaler_path")
POCKET_ATOM_NUM = config.get_int("data.pocket_atom_num")
INPUT_DIM = config.get_int("data.input_dim_features")
LEARNING_RATE = config.get_float("training.learning_rate")
DROPOUT_RATE = config.get_float("training.dropout_rate")
WEIGHT_DECAY = config.get_float("training.weight_decay")
NUM_EPOCHS = config.get_int("training.num_epochs")
BATCH_SIZE = config.get_int("training.batch_size")
L1_LAMBDA = config.get_float("training.l1_lambda")
```

### 步骤 3: 修改参数只需编辑 config.yaml

现在，要改学习率、批大小等任何参数，直接编辑 `config.yaml` 即可，无需改代码。

## 📊 主要脚本的改造清单

### 2_train_model.py
需改造参数：
- 路径: LABEL_FILE, RESULT_DIR, MODEL_SAVE_PATH, SCALER_SAVE_PATH
- 数据: POCKET_ATOM_NUM, INPUT_DIM
- 训练: LEARNING_RATE, DROPOUT_RATE, WEIGHT_DECAY, NUM_EPOCHS, BATCH_SIZE, L1_LAMBDA

### 1_extract_features.py
需改造参数：
- 路径: OUTPUT_BASE_DIR, QC_OUTPUT_DIR
- 数据: INTEGRATION_RADIUS
- 残基: PHE_RESIDUES_STD, OBP_RESIDUES_STD, PLANE_RESIDUES_STD (可选)
- 其他: STANDARD_SEQUENCE (可选)

### 3_evaluate_all.py
需改造参数：
- 路径: LABEL_FILE, RESULT_DIR, MODEL_PATH, SCALER_PATH
- 数据: POCKET_ATOM_NUM, INPUT_DIM

### 8_generate_efficacy_map.py
需改造参数：
- 路径: REFERENCE_PDB, DATA_DIR, MODEL_PATH, SCALER_PATH
- 数据: INPUT_DIM
- 评估: SAMPLES_PER_LEVEL
- 残基: OBP_RESIDUES (已在config中)

### 其他脚本
可根据需要类似改造：
- 4_predict_single.py
- 5_global_interpretation.py
- 6_3d_atom_cloud.py
- 7_reconstruct_pocket_cloud.py
- 9_benchmark_trajectory_duration.py
- 10_sanity_check_rf.py
- 等等...

## 💡 改造的好处

| 问题 | 原来的方式 | 现在的方式 |
|------|----------|---------|
| 修改学习率 | 改 2_train_model.py | 改 config.yaml |
| 修改多个参数 | 改多个脚本 | 改一个文件 |
| 参数一致性 | 容易出错 | 统一管理 |
| 版本控制 | 参数分散 | 参数集中 |
| 配置复用 | 困难 | 简单 |

## 📝 常见改造场景

### 场景 1: 只想改学习率和批大小

```yaml
# config.yaml 中只需改这里
training:
  learning_rate: 0.0005  # 改这个
  batch_size: 64         # 改这个
```

### 场景 2: 想切换到不同的模型

```yaml
# config.yaml 中只需改这里
paths:
  model_path: "saved_models/best_model_v2.pth"  # 改这个
  scaler_path: "saved_models/scaler_v2.pkl"     # 改这个
```

### 场景 3: 使用不同的特征目录

```yaml
# config.yaml 中只需改这里
paths:
  result_dir: "data/features_v2"  # 改这个
  qc_output_dir: "data/qc_v2"    # 改这个
```

## 🔧 API 速查

```python
# 字符串/路径
config.get_path("paths.model_path")
config.get("data.standard_sequence")

# 整数
config.get_int("training.batch_size")
config.get_int("data.pocket_atom_num")

# 浮点数
config.get_float("training.learning_rate")
config.get_float("data.integration_radius")

# 列表
config.get_list("residues.obp_residues")
config.get_list("residues.phe_residues")

# 布尔值
config.get_bool("plotting.use_agg_backend")

# 带默认值
config.get("some.key", default=10)
```

## ✅ 验证配置是否正确

```python
# 在脚本中运行，检查配置是否加载成功
from src.config import init_config
config = init_config()

print(config.to_dict())  # 打印所有配置
```

## 🎯 下一步

1. 根据上面的清单，逐个改造脚本
2. 在脚本开头添加 `from src.config import init_config; config = init_config()`
3. 替换硬编码的参数为 `config.get_*("配置.键")`
4. 测试脚本是否正常运行
5. 通过编辑 `config.yaml` 验证参数修改是否生效

## 📚 更多信息

详见 `CONFIG_USAGE.md` 获取更详细的文档和示例。

## 🆘 常见问题

**Q: 脚本报错 "配置文件未找到"**
A: 确保 `config.yaml` 在项目根目录，与脚本同级。

**Q: 改了 config.yaml 但脚本没生效**
A: 脚本需要重新运行才能加载新的配置。

**Q: 可以在运行时修改配置吗？**
A: 可以，但建议直接改 config.yaml，然后重新运行脚本。

**Q: 有多个项目需要不同的配置？**
A: 可以创建多个 config 文件，如 `config_prod.yaml`、`config_dev.yaml`，通过 `init_config("config_dev.yaml")` 加载。
