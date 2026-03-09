#!/usr/bin/env python3
"""
verify_setup.py
验证分析系统是否准备就绪
"""

import os
import sys

def check_file(path, description):
    """检查文件是否存在"""
    if os.path.exists(path):
        size = os.path.getsize(path)
        print(f"✓ {description:<40} ({size:,} bytes)")
        return True
    else:
        print(f"✗ {description:<40} (缺失)")
        return False

def check_dir(path, description):
    """检查目录是否存在"""
    if os.path.isdir(path):
        print(f"✓ {description:<40} (目录)")
        return True
    else:
        print(f"✗ {description:<40} (缺失)")
        return False

print("=" * 80)
print("分析系统设置验证")
print("=" * 80)
print()

base_dir = "/home/hongyu/MD/1_partial"
os.chdir(base_dir)

all_ok = True

# 检查核心脚本
print("【核心脚本】")
print("-" * 80)
all_ok &= check_file("run_analysis_v2.py", "主分析脚本 (run_analysis_v2.py)")
print()

# 检查模块
print("【模块文件】")
print("-" * 80)
all_ok &= check_dir("modules", "modules/ 目录")
all_ok &= check_file("modules/__init__.py", "  - 包初始化 (__init__.py)")
all_ok &= check_file("modules/geometry.py", "  - 几何计算 (geometry.py)")
all_ok &= check_file("modules/output_handler.py", "  - 输出处理 (output_handler.py)")
print()

# 检查输出目录
print("【输出目录】")
print("-" * 80)
all_ok &= check_dir("results", "results/ 目录（存放输出）")
print()

# 检查文档
print("【文档文件】")
print("-" * 80)
all_ok &= check_file("README_V2.md", "完整使用文档 (README_V2.md)")
all_ok &= check_file("USAGE_GUIDE.md", "快速使用指南 (USAGE_GUIDE.md)")
all_ok &= check_file("RESTRUCTURING_SUMMARY.md", "重构总结 (RESTRUCTURING_SUMMARY.md)")
all_ok &= check_file("WEIGHTING_MECHANISM_EXPLANATION.md", "权重机制说明 (WEIGHTING_MECHANISM_EXPLANATION.md)")
all_ok &= check_file("ANGLE_CALCULATION_CORRECTION.md", "角度计算说明 (ANGLE_CALCULATION_CORRECTION.md)")
print()

# 检查示例数据文件
print("【输入数据文件】")
print("-" * 80)
csv_files = [f for f in os.listdir(".") if f.endswith(".csv")]
if csv_files:
    print(f"✓ 发现 {len(csv_files)} 个CSV文件")
    # 列出前5个和最后5个
    for csv in sorted(csv_files)[:5]:
        size = os.path.getsize(csv)
        print(f"  - {csv:<60} ({size:,} bytes)")
    if len(csv_files) > 10:
        print(f"  ... ({len(csv_files) - 10} 更多文件)")
else:
    print(f"✗ 未找到CSV数据文件")
    all_ok = False
print()

# 检查Python环境
print("【Python环境】")
print("-" * 80)
print(f"✓ Python 版本: {sys.version.split()[0]}")
print(f"✓ 工作目录: {os.getcwd()}")

# 尝试导入必要的包
try:
    import numpy as np
    print(f"✓ NumPy 已安装 (v{np.__version__})")
except ImportError:
    print("✗ NumPy 未安装")
    all_ok = False

try:
    import pandas as pd
    print(f"✓ Pandas 已安装 (v{pd.__version__})")
except ImportError:
    print("✗ Pandas 未安装")
    all_ok = False

try:
    import MDAnalysis as mda
    print(f"✓ MDAnalysis 已安装 (v{mda.__version__})")
except ImportError:
    print("✗ MDAnalysis 未安装")
    all_ok = False

try:
    from scipy import spatial
    import scipy
    print(f"✓ SciPy 已安装 (v{scipy.__version__})")
except ImportError:
    print("✗ SciPy 未安装")
    all_ok = False

try:
    import matplotlib
    print(f"✓ Matplotlib 已安装 (v{matplotlib.__version__})")
except ImportError:
    print("✗ Matplotlib 未安装")
    all_ok = False

print()

# 尝试导入自定义模块
print("【自定义模块导入测试】")
print("-" * 80)
try:
    from modules import geometry
    print("✓ modules.geometry 导入成功")
except Exception as e:
    print(f"✗ modules.geometry 导入失败: {e}")
    all_ok = False

try:
    from modules import output_handler
    print("✓ modules.output_handler 导入成功")
except Exception as e:
    print(f"✗ modules.output_handler 导入失败: {e}")
    all_ok = False

print()
print("=" * 80)

if all_ok:
    print("✓ 系统设置完整，可以运行分析")
    print()
    print("快速开始命令:")
    print("  python run_analysis_v2.py")
    print()
    print("更多信息请查看:")
    print("  - USAGE_GUIDE.md (快速使用指南)")
    print("  - README_V2.md (完整文档)")
    sys.exit(0)
else:
    print("✗ 系统设置不完整，请检查上述错误")
    sys.exit(1)
