"""
快速迁移脚本：自动将脚本中的硬编码配置改为使用 config.yaml
用法：python scripts/migrate_to_config.py <script_path>
"""

import sys
import re
from pathlib import Path


def suggest_config_replacements(file_path):
    """
    分析脚本文件，建议可能的配置替换
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 常见的配置变量模式
    patterns = {
        # 路径配置
        r'LABEL_FILE\s*=\s*["\']([^"\']+)["\']': 'paths.label_file',
        r'RESULT_DIR\s*=\s*["\']([^"\']+)["\']': 'paths.result_dir',
        r'QC_OUTPUT_DIR\s*=\s*["\']([^"\']+)["\']': 'paths.qc_output_dir',
        r'MODEL_PATH\s*=\s*["\']([^"\']+)["\']': 'paths.model_path',
        r'SCALER_PATH\s*=\s*["\']([^"\']+)["\']': 'paths.scaler_path',
        r'REFERENCE_PDB\s*=\s*["\']([^"\']+)["\']': 'paths.reference_pdb',
        r'OUTPUT_BASE_DIR\s*=\s*["\']([^"\']+)["\']': 'paths.result_dir',
        
        # 数值配置
        r'POCKET_ATOM_NUM\s*=\s*(\d+)': 'data.pocket_atom_num',
        r'INPUT_DIM\s*=\s*(\d+)': 'data.input_dim_features',
        r'INTEGRATION_RADIUS\s*=\s*([\d.]+)': 'data.integration_radius',
        
        # 训练参数
        r'LEARNING_RATE\s*=\s*([\d.e-]+)': 'training.learning_rate',
        r'DROPOUT_RATE\s*=\s*([\d.]+)': 'training.dropout_rate',
        r'WEIGHT_DECAY\s*=\s*([\d.e-]+)': 'training.weight_decay',
        r'NUM_EPOCHS\s*=\s*(\d+)': 'training.num_epochs',
        r'BATCH_SIZE\s*=\s*(\d+)': 'training.batch_size',
        r'L1_LAMBDA\s*=\s*([\d.]+)': 'training.l1_lambda',
        
        # 其他
        r'SAMPLES_PER_LEVEL\s*=\s*(\d+)': 'evaluation.samples_per_level',
    }
    
    suggestions = []
    for pattern, config_key in patterns.items():
        matches = re.finditer(pattern, content)
        for match in matches:
            suggestions.append({
                'line': content[:match.start()].count('\n') + 1,
                'variable': match.group(0).split('=')[0].strip(),
                'value': match.group(1),
                'config_key': config_key,
                'full_match': match.group(0)
            })
    
    return suggestions


def print_suggestions(file_path):
    """打印改造建议"""
    suggestions = suggest_config_replacements(file_path)
    
    if not suggestions:
        print(f"✓ {file_path}: 未找到可改造的配置")
        return
    
    print(f"\n📄 {file_path}")
    print("=" * 80)
    print("\n【建议的改造内容】\n")
    
    # 打印导入语句
    print("在文件顶部添加：")
    print("-" * 40)
    print("from src.config import init_config")
    print("config = init_config()")
    print()
    
    # 按配置类型分组
    from collections import defaultdict
    by_type = defaultdict(list)
    for s in suggestions:
        type_key = s['config_key'].split('.')[0]
        by_type[type_key].append(s)
    
    print("【具体替换建议】\n")
    for type_key in sorted(by_type.keys()):
        print(f"--- {type_key.upper()} ---")
        for s in by_type[type_key]:
            print(f"Line {s['line']}: {s['variable']}")
            
            # 根据类型推荐获取方法
            if 'dim' in s['config_key'] or 'num' in s['config_key'] or 'epochs' in s['config_key'] or 'batch' in s['config_key'] or 'level' in s['config_key']:
                method = "get_int"
            elif 'rate' in s['config_key'] or 'lambda' in s['config_key'] or 'radius' in s['config_key']:
                method = "get_float"
            elif 'residues' in s['config_key']:
                method = "get_list"
            elif any(x in s['config_key'] for x in ['path', 'file', 'dir']):
                method = "get_path"
            else:
                method = "get"
            
            print(f"  → {s['variable']} = config.{method}(\"{s['config_key']}\")")
        print()


def main():
    if len(sys.argv) < 2:
        print("用法: python scripts/migrate_to_config.py <script_path> [<script_path> ...]")
        print("\n示例:")
        print("  python scripts/migrate_to_config.py 2_train_model.py")
        print("  python scripts/migrate_to_config.py *.py")
        return
    
    for pattern in sys.argv[1:]:
        from glob import glob
        files = glob(pattern) if '*' in pattern else [pattern]
        for file_path in files:
            if file_path.endswith('.py'):
                print_suggestions(file_path)


if __name__ == "__main__":
    main()
