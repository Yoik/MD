# 配置系统使用指南

## 快速开始

### 1. 加载配置

在你的脚本开头添加：

```python
from src.config import init_config

# 初始化配置（自动加载 config.yaml）
config = init_config()
```

### 2. 获取配置值

```python
# 获取路径
label_file = config.get_path("paths.label_file")
model_path = config.get_path("paths.model_path")

# 获取数值
learning_rate = config.get_float("training.learning_rate")
batch_size = config.get_int("training.batch_size")
dropout_rate = config.get_float("training.dropout_rate")

# 获取列表
obp_residues = config.get_list("residues.obp_residues")
phe_residues = config.get_list("residues.phe_residues")

# 获取字符串
ref_sequence = config.get("data.standard_sequence")

# 获取布尔值
use_agg = config.get_bool("plotting.use_agg_backend")

# 带默认值的获取
custom_value = config.get("some.key", default="default_value")
```

## 配置结构

### 文件路径 (`paths`)
- `label_file`: 标签数据文件
- `result_dir`: 特征文件目录
- `qc_output_dir`: QC 结构输出目录
- `reference_pdb`: 参考 PDB 文件
- `model_path`: 训练好的模型路径
- `scaler_path`: 数据标准化器路径

### 数据处理 (`data`)
- `integration_radius`: 积分半径（Ångström）
- `pocket_atom_num`: 口袋原子数
- `input_dim_features`: 特征输入维度
- `input_dim_eval`: 评估输入维度
- `standard_sequence`: 标准蛋白序列

### 蛋白质残基 (`residues`)
- `phe_residues`: Phe 残基列表
- `obp_residues`: OBP（结合口袋）残基列表
- `plane_residues`: 平面残基列表

### 训练配置 (`training`)
- `learning_rate`: 学习率
- `dropout_rate`: Dropout 率
- `weight_decay`: 权重衰减
- `num_epochs`: 训练轮数
- `batch_size`: 批大小
- `l1_lambda`: L1 正则化系数
- `window_size`: 滑动窗口大小
- `stride`: 滑动步长

### 评估配置 (`evaluation`)
- `samples_per_level`: 每个等级的采样数

## 使用示例

### 示例 1: 在 `2_train_model.py` 中使用

**修改前：**
```python
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
```

**修改后：**
```python
from src.config import init_config

config = init_config()

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

### 示例 2: 在 `1_extract_features.py` 中使用

```python
from src.config import init_config

config = init_config()

INTEGRATION_RADIUS = config.get_float("data.integration_radius")
OUTPUT_BASE_DIR = config.get_path("paths.result_dir")
QC_OUTPUT_DIR = config.get_path("paths.qc_output_dir")
STANDARD_SEQUENCE = config.get("data.standard_sequence")
PHE_RESIDUES_STD = config.get_list("residues.phe_residues")
OBP_RESIDUES_STD = config.get_list("residues.obp_residues")
PLANE_RESIDUES_STD = config.get_list("residues.plane_residues")
```

### 示例 3: 在 `3_evaluate_all.py` 中使用

```python
from src.config import init_config

config = init_config()

LABEL_FILE = config.get_path("paths.label_file")
RESULT_DIR = config.get_path("paths.result_dir")
MODEL_PATH = config.get_path("paths.model_path")
SCALER_PATH = config.get_path("paths.scaler_path")
POCKET_ATOM_NUM = config.get_int("data.pocket_atom_num")
INPUT_DIM = config.get_int("data.input_dim_eval")
```

## 修改配置

只需编辑 `config.yaml` 文件即可修改所有脚本的参数。无需修改任何代码！

例如，要修改学习率：

```yaml
training:
  learning_rate: 0.0005  # 改这里
```

所有使用 `config.get_float("training.learning_rate")` 的脚本都会立即生效。

## API 参考

### `init_config(config_path: str = "config.yaml") -> Config`
初始化并加载配置文件。

### `Config.get(key: str, default: Any = None) -> Any`
获取配置值，支持嵌套键。

### `Config.get_path(key: str) -> str`
获取路径配置值。

### `Config.get_list(key: str, default: list = None) -> list`
获取列表配置值。

### `Config.get_int(key: str, default: int = None) -> int`
获取整数配置值。

### `Config.get_float(key: str, default: float = None) -> float`
获取浮点数配置值。

### `Config.get_bool(key: str, default: bool = False) -> bool`
获取布尔配置值。

## 常见问题

**Q: 配置文件找不到怎么办？**
A: 确保 `config.yaml` 在项目根目录（与脚本同级）。

**Q: 可以有多个配置文件吗？**
A: 可以，通过 `init_config("config_prod.yaml")` 加载不同的配置文件。

**Q: 如何在运行时修改配置？**
A: 直接修改 `config` 对象：
```python
config._config_data["training"]["batch_size"] = 64
```

**Q: YAML 格式不对怎么办？**
A: 确保使用空格（不是 Tab）缩进，格式类似 JSON 的嵌套结构。
