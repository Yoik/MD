#!/bin/bash
set -e  # 出错立即退出

# 定义一个函数，调用指定 replicate 下的 README
run_replicate() {
    local replicate=$1
    local base_dir=~/2_MD/20251115_D2_ARI_cryoEM_rebuild/charmm-gui-6633139040/gromacs_${replicate}
    local readme=~/2_MD/20251115_D2_ARI_cryoEM_rebuild/charmm-gui-6633139040/README

    echo ">>> 正在运行 ${replicate}，输入目录: ${base_dir}"

    if [ -f "$readme" ]; then
        chmod +x "$readme"
        "$readme" "$base_dir"
    else
        echo "README 文件不存在于 $(dirname "$readme")"
        exit 1
    fi
}

# 依次执行 replicate_1、2、3
for r in replicate_1 replicate_2 replicate_3; do
    run_replicate "$r"
done
