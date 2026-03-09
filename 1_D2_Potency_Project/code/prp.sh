# 1. 创建偏置模拟文件夹
mkdir bias_sim
cd bias_sim

# 2. 复制必要文件
# 假设原来的路径是 ../charmm-gui-6321110432/gromacs_replicate_1/
SOURCE_DIR="../20251115_D2_Dopa_cryoEM_rebuild/charmm-gui-6321110432/gromacs_replicate_1"

# 我们需要 .mdp (参数), .gro (坐标), .top (拓扑), .cpt (速度/状态)
# 注意：如果有 index.ndx 也最好带上
cp ${SOURCE_DIR}/step7_production.mdp ./bias.mdp
cp ${SOURCE_DIR}/step7_3.gro ./start.gro
cp ${SOURCE_DIR}/step7_3.cpt ./start.cpt
cp ${SOURCE_DIR}/topol.top .
cp ${SOURCE_DIR}/*.itp . 2>/dev/null || :  # 如果有单独的itp也复制