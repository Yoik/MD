#!/bin/bash

# ================= 配置区 =================
# 想要模拟的时长 (例如 200 ns)
# 假设 step 为 0.002 ps (2fs)
# 200 ns = 200,000 ps = 100,000,000 steps
NSTEPS=5000000

# GROMACS 命令 (根据你的服务器环境修改，如 gmx_mpi 或 gmx)
GMX="gmx"
# ========================================

echo ">>> 1. 修改 MDP 文件时长..."
# 使用 sed 将 nsteps 修改为你想要的长度
sed -i "s/^nsteps.*/nsteps = ${NSTEPS}/g" bias.mdp

echo ">>> 2. 生成新的 TPR 文件..."
# 修改点：添加了 -n index.ndx
$GMX grompp -f bias.mdp -c start.gro -t start.cpt -p topol.top -n index.ndx -o bias.tpr -maxwarn 2
if [ ! -f "bias.tpr" ]; then
    echo "Error: TPR generation failed!"
    exit 1
fi

echo ">>> 3. 启动 OPES 偏置模拟..."
# -plumed 指定我们刚才写的 plumed.dat
# -ntomp 指定 OpenMP 线程数 (根据你的核数调整)
$GMX mdrun -deffnm bias -plumed plumed.dat -ntomp 8 -v

echo ">>> 模拟结束！"