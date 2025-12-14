"""
ring_matcher.py
环形匹配模块：支持苯环、吲哚(Indole)、呋喃(Furan)
"""

import numpy as np
import itertools
from scipy.spatial.distance import cdist


class RingMatcher:
    """
    识别参考分子中的芳香环，并将其与 MD 轨迹中的环匹配
    支持：
    - 6元环（苯环）
    - 5元环（吲哚杂环、呋喃）
    """
    
    def __init__(self, ref_coords, ref_elements):
        """
        初始化环匹配器
        
        Args:
            ref_coords: 参考分子的原子坐标 (N x 3)
            ref_elements: 参考分子的元素列表 ['C', 'C', 'N', ...]
        """
        self.ref_coords = ref_coords
        self.ref_elements = ref_elements
        self.n_ref = len(ref_coords)
        
        # 检测所有可能的环
        self.rings = self._detect_all_rings()
        
        if not self.rings:
            raise ValueError("No aromatic rings (6-membered or 5-membered) found")
        
        # 使用找到的第一个（或最大的）环作为参考
        self.ref_ring_idx = self.rings[0]['indices']
        self.ring_type = self.rings[0]['type']
        self.ring_elements = [ref_elements[i] for i in self.ref_ring_idx]  # 新增：环原子的元素类型
        
        # 查找环附近的邻近原子
        self._find_neighboring_atoms()

    def _align_by_fingerprint(self, candidate_atoms, full_md_atoms):
        """
        根据参考环的【元素类型】+【几何指纹】+【取代基连接情况】，
        重新排列找到的 MD 原子。
        (优化版：优先匹配特征独特的原子)
        """
        from scipy.spatial.distance import cdist
        import numpy as np
        
        # --- 1. 准备参考环特征 ---
        ref_indices = self.ref_ring_idx
        ref_coords = self.ref_coords[ref_indices]
        ref_elems = self.ring_elements
        
        # 计算参考环邻居数
        ref_neighbor_counts = []
        ref_dmat_all = cdist(self.ref_coords, self.ref_coords)
        for r_idx in ref_indices:
            neighbors = [n for n in range(self.n_ref) 
                         if n not in ref_indices and ref_dmat_all[r_idx, n] < 2.0]
            ref_neighbor_counts.append(len(neighbors))

        # --- 2. 准备 MD 候选原子特征 ---
        cand_coords = candidate_atoms.positions
        cand_elems = [a.name[0] for a in candidate_atoms]
        
        # 计算 MD 邻居数 (包含氢过滤)
        md_neighbor_counts = []
        cand_global_indices = [a.index for a in candidate_atoms]
        dmat_cand_full = cdist(cand_coords, full_md_atoms.positions)
        
        for i in range(len(candidate_atoms)):
            nearby_indices = np.where(dmat_cand_full[i] < 2.2)[0]
            count = 0
            for n_idx in nearby_indices:
                atom = full_md_atoms[n_idx]
                if atom.index in cand_global_indices: continue
                # 过滤氢和非重原子
                name_upper = atom.name.upper()
                if name_upper.startswith('H') or (name_upper[0].isdigit() and 'H' in name_upper): continue
                if name_upper[0] not in ['C', 'N', 'O', 'S', 'F', 'P', 'I', 'B']: continue
                count += 1
            md_neighbor_counts.append(count)

        # --- 3. 进行指纹匹配 (优化顺序) ---
        dmat_ref_internal = cdist(ref_coords, ref_coords)
        dmat_cand_internal = cdist(cand_coords, cand_coords)
        
        mapping_dict = {} # ref_idx -> md_idx
        used_indices = set()
        
        # 【关键优化】计算“独特性分数”，优先匹配独特的原子
        # 独特性 = (是不是杂原子 * 10) + (邻居数量 * 5)
        priorities = []
        for i in range(len(ref_indices)):
            score = 0
            if ref_elems[i] != 'C': score += 10
            score += ref_neighbor_counts[i] * 5
            priorities.append((score, i))
        
        # 按独特性从高到低排序，高分的先去挑 MD 原子
        priorities.sort(key=lambda x: x[0], reverse=True)
        sorted_ref_indices = [p[1] for p in priorities]
        
        for i in sorted_ref_indices:
            r_e = ref_elems[i]
            r_dists = np.sort(dmat_ref_internal[i])
            r_n_count = ref_neighbor_counts[i]
            
            best_j = -1
            min_score = float('inf')
            
            for j in range(len(candidate_atoms)):
                if j in used_indices: continue
                
                # A. 元素匹配
                if cand_elems[j] != r_e: continue
                
                # B. 取代基数量匹配
                diff = abs(md_neighbor_counts[j] - r_n_count)
                penalty = 5.0 * diff
                
                # C. 几何指纹匹配
                c_dists = np.sort(dmat_cand_internal[j])
                dist_score = np.sum(np.abs(r_dists - c_dists))
                
                total_score = dist_score + penalty
                
                if total_score < min_score:
                    min_score = total_score
                    best_j = j
            
            # 这里的阈值可以根据实际情况调整，80.0 比较宽松
            if best_j == -1 or min_score > 80.0: 
                return None
            
            mapping_dict[i] = best_j
            used_indices.add(best_j)
            
        # 按原始 ref 顺序还原列表
        final_mapping = [mapping_dict[i] for i in range(len(ref_indices))]
        return final_mapping
    
    def _detect_all_rings(self):
        """
        检测芳香环的优先级策略：
        1. 优先检测吲哚和呋喃（6元苯环+5元杂环共享原子）
        2. 如果不是复杂环，回退到单独的苯环（6元环）
        
        Returns:
            list: 环的列表，包含 {'indices': [...], 'type': 'indole'/'furan'/'benzene', ...}
        """
        rings = []
        
        # 首先尝试找 6 元环
        benzenes = self._detect_benzene_rings()
        if not benzenes:
            return rings
        
        # 尝试检测吲哚和呋喃（6+5 融合环）
        # 但如果没有找到，仍然返回苯环结果
        fused = self._detect_fused_rings()
        if fused:
            return fused
        
        # 回退到苯环
        return benzenes

    def _detect_fused_rings(self):
        """
        检测融合环系统：吲哚（苯环+吡咯）、呋喃等
        吲哚/呋喃特征：6元环和5元环共享2个原子
        
        Returns:
            list: 融合环列表或空列表
        """
        fused_rings = []
        
        try:
            # 先找所有 6 元环
            six_rings = self._find_all_6rings()
            
            # 再找所有 5 元环
            five_rings = self._find_all_5rings()
            
            if not six_rings or not five_rings:
                return fused_rings
            
            # 检查是否存在共享原子的 6+5 组合
            for six_ring in six_rings:
                six_set = set(six_ring['indices'])
                
                for five_ring in five_rings:
                    five_set = set(five_ring['indices'])
                    shared = six_set & five_set
                    
                    # 吲哚/呋喃的典型特征：共享 2 个原子
                    if len(shared) == 2:
                        # 合并为融合环
                        fused_indices = list(six_set | five_set)
                        fused_coords = self.ref_coords[fused_indices]
                        fused_elems = [self.ref_elements[i] for i in fused_indices]
                        
                        # 判断类型（根据5元环中的杂原子）
                        five_ring_elems = [self.ref_elements[i] for i in five_ring['indices']]
                        if 'S' in five_ring_elems:
                            ring_type = 'thiophene'  # 含硫融合环
                        elif 'O' in five_ring_elems:
                            ring_type = 'furan'  # 含氧融合环
                        elif 'N' in five_ring_elems:
                            ring_type = 'indole'  # 含氮融合环
                        else:
                            ring_type = None
                        
                        if ring_type:
                            fused_rings.append({
                                'indices': fused_indices,
                                'type': ring_type,
                                'size': len(fused_indices),  # 融合环通常是 6+5-2 = 9 个原子
                                'six_ring': six_ring['indices'],
                                'five_ring': five_ring['indices'],
                                'shared_atoms': list(shared),
                                'elements': fused_elems,
                                'coords': fused_coords
                            })
        except Exception as e:
            pass
        
        return fused_rings

    def _find_all_6rings(self):
        """查找所有有效的6元环（苯环）"""
        rings = []
        c_indices = [i for i, e in enumerate(self.ref_elements) if e == 'C']
        
        if len(c_indices) < 6:
            return rings
        
        c_coords = self.ref_coords[c_indices]
        dmat = cdist(c_coords, c_coords)
        adj = np.logical_and(dmat > 1.1, dmat < 1.7)
        
        for comb in itertools.combinations(range(len(c_indices)), 6):
            sub_idx = list(comb)
            curr_coords = c_coords[sub_idx]
            
            # 检查共平面性
            centered = curr_coords - curr_coords.mean(0)
            _, s, _ = np.linalg.svd(centered)
            if s[2] > 0.3:
                continue
            
            # 检查连接性
            sub_adj = adj[np.ix_(sub_idx, sub_idx)]
            if not np.all(np.sum(sub_adj, axis=1) >= 2):
                continue
            
            ordered_indices = self._order_ring_indices(sub_idx, sub_adj)
            global_indices = [c_indices[i] for i in ordered_indices]
            # ordered_indices 是本地索引（在 sub_idx 中的位置），需要用于索引 curr_coords
            ordered_local_indices = [sub_idx.index(i) for i in ordered_indices]
            rings.append({
                'indices': global_indices,
                'type': 'benzene',
                'size': 6,
                'coords': curr_coords[ordered_local_indices]
            })
        
        return rings

    def _find_all_5rings(self):
        """查找所有有效的5元环（含杂原子如N、O、S）"""
        rings = []
        all_indices = list(range(self.n_ref))
        
        for comb in itertools.combinations(all_indices, 5):
            ring_idx = list(comb)
            ring_coords = self.ref_coords[ring_idx]
            ring_elems = [self.ref_elements[i] for i in ring_idx]
            
            # 必须包含杂原子（N、O 或 S）
            has_hetero = any(elem in ring_elems for elem in ['N', 'O', 'S'])
            if not has_hetero:
                continue
            
            # 构建邻接矩阵
            sub_dmat = cdist(ring_coords, ring_coords)
            sub_adj = np.logical_and(sub_dmat > 1.1, sub_dmat < 1.9)
            
            # 检查连接性：每个原子至少连接到1个其他原子
            degrees = np.sum(sub_adj, axis=1)
            if not np.all(degrees >= 1):
                continue
            
            # 检查共平面性
            centered = ring_coords - ring_coords.mean(0)
            try:
                _, s, _ = np.linalg.svd(centered)
                if s[2] > 0.3:
                    continue
            except:
                continue
            
            # 确定5元环类型（根据杂原子类型）
            if 'S' in ring_elems:
                ring_type = 'thiophene'  # 硫杂环
            elif 'O' in ring_elems:
                ring_type = 'furan'  # 氧杂环
            elif 'N' in ring_elems:
                ring_type = 'pyrrole'  # 氮杂环（用于吲哚的5元部分）
            else:
                continue
            
            # 尝试找到环拓扑
            try:
                ordered_indices = self._order_ring_indices_5_flexible(ring_idx, sub_adj)
                if ordered_indices is None:
                    continue
                    
                rings.append({
                    'indices': [ring_idx[i] for i in ordered_indices],
                    'type': ring_type,
                    'size': 5,
                    'elements': ring_elems,
                    'coords': ring_coords[ordered_indices]
                })
            except:
                continue
        
        return rings

    def _detect_benzene_rings(self):
        """检测单独的苯环（6元环）"""
        return self._find_all_6rings()

    def _order_ring_indices(self, indices, sub_adj):
        """
        按拓扑顺序排列 6 元环的原子索引
        """
        ordered = [indices[0]]
        current = 0
        used = {0}
        
        for _ in range(5):
            for n in np.where(sub_adj[current])[0]:
                if n not in used:
                    ordered.append(indices[n])
                    used.add(n)
                    current = n
                    break
        
        return ordered

    def _order_ring_indices_5(self, indices, sub_adj):
        """
        按拓扑顺序排列 5 元环的原子索引
        """
        ordered = [0]
        current = 0
        used = {0}
        
        for _ in range(4):
            for n in np.where(sub_adj[current])[0]:
                if n not in used:
                    ordered.append(n)
                    used.add(n)
                    current = n
                    break
        
        return ordered

    def _order_ring_indices_5_flexible(self, indices, sub_adj):
        """
        按拓扑顺序排列 5 元环的原子索引（灵活版本，支持融合环）
        
        Args:
            indices: 5个原子的索引列表
            sub_adj: 5x5 邻接矩阵
            
        Returns:
            按顺序排列的索引列表，如果无法形成环则返回 None
        """
        # 尝试从每个原子开始
        for start in range(len(indices)):
            ordered = [start]
            current = start
            used = {start}
            
            # 贪心遍历
            for _ in range(4):
                neighbors = np.where(sub_adj[current])[0]
                found_next = False
                
                for n in neighbors:
                    if n not in used:
                        ordered.append(n)
                        used.add(n)
                        current = n
                        found_next = True
                        break
                
                if not found_next:
                    break
            
            # 检查是否形成了环：最后一个原子应该连接回第一个
            if len(ordered) == 5 and sub_adj[current, start]:
                return ordered
        
        return None

    def _find_neighboring_atoms(self):
        """找到环周围的邻近原子"""
        ring_set = set(self.ref_ring_idx)
        self.ref_neigh_idx = []
        dmat = cdist(self.ref_coords, self.ref_coords)
        
        for i in range(self.n_ref):
            if i not in ring_set and np.min(dmat[i, self.ref_ring_idx]) < 2.0:
                self.ref_neigh_idx.append(i)

    def match(self, md_atoms, anchor_com):
        """
        将参考环与 MD 轨迹中的环匹配
        
        Args:
            md_atoms: MDAnalysis AtomGroup（通常是配体原子）
            anchor_com: 锚点的质心（用于选择最接近的环）
            
        Returns:
            tuple: (匹配的环原子, 立方体文件中的原子索引（包括所有环原子，不只是碳）, MD中环原子的索引)
        """
        if self.ring_type == 'benzene':
            return self._match_benzene(md_atoms, anchor_com)
        # 这里的 indole/furan 实际上是指稠环系统（如吲哚、苯并呋喃）
        elif self.ring_type in ['indole', 'furan', 'benzofuran', 'thiophene']:
            return self._match_fused_system(md_atoms, anchor_com)            
        else:
            return None, None, None
        
    def _match_benzene(self, md_atoms, anchor_com):
        """匹配 6 元环（苯环） - 使用指纹对齐防止旋转"""
        md_coords = md_atoms.positions
        # 注意：这里我们用 name 筛选碳原子，确保只在碳骨架里找环
        md_c_indices = [i for i, a in enumerate(md_atoms) if a.name.startswith('C')]
        
        if len(md_c_indices) < 6:
            return None, None, None
        
        # 1. 构建碳原子的邻接矩阵
        md_c_coords = md_coords[md_c_indices]
        dmat = cdist(md_c_coords, md_c_coords)
        adj = np.logical_and(dmat > 1.1, dmat < 1.70) # 苯环 C-C 键长比较固定，1.7 够了
        
        # 2. 查找所有 6 元环 (DFS)
        found_rings = []
        seen = set()
        
        def dfs(s, c, p):
            if len(p) == 6:
                return p if adj[c, s] else None
            for n in np.where(adj[c])[0]:
                if n == s and len(p) < 5:
                    continue
                if n not in p:
                    r = dfs(s, n, p + [n])
                    if r: return r
            return None
        
        for i in range(len(md_c_indices)):
            res = dfs(i, i, [i])
            if res:
                s = tuple(sorted(res))
                if s not in seen:
                    found_rings.append(list(res))
                    seen.add(s)
        
        if not found_rings:
            return None, None, None
        
        # 3. 选择最接近锚点的环
        best_ring_local = None # 这是在 md_c_indices 里的局部索引
        min_dist = float('inf')
        
        for r in found_rings:
            # r 是局部索引，转换回 md_coords 的索引
            g = [md_c_indices[k] for k in r]
            cent = md_coords[g].mean(0)
            d = np.linalg.norm(cent - anchor_com)
            if d < min_dist:
                min_dist = d
                best_ring_local = g # 保存全局索引 (md_atoms 中的索引)
        
        if best_ring_local is None:
            return None, None, None

        # =========================================================
        # 【核心修改】 使用 _align_by_fingerprint 替代旧的旋转尝试
        # =========================================================
        
        # 4. 提取候选原子 (无序)
        candidate_atoms = md_atoms[best_ring_local]
        
        # 5. 指纹对齐 (强制一一对应，利用取代基指纹锁死旋转)
        # 传入 md_atoms (包含所有原子) 以便计算环外邻居
        sorted_order = self._align_by_fingerprint(candidate_atoms, md_atoms)
        
        if sorted_order is None:
            # 如果指纹对不上 (比如取代基数量不对)，说明匹配失败
            return None, None, None
            
        # 6. 根据对齐结果，重新排列 MD 索引
        md_ring_indices = [best_ring_local[i] for i in sorted_order]
        matched_atoms = md_atoms[md_ring_indices]
        
        # 7. 映射 Cube 索引
        ref_c_idxs = [i for i, e in enumerate(self.ref_elements) if e == 'C']
        cube_idxs = [{idx: rank for rank, idx in enumerate(ref_c_idxs)}[i] 
                     for i in self.ref_ring_idx if self.ref_elements[i] == 'C']
        
        # 兼容旧逻辑的返回格式：(matched_atoms, cube_idxs, md_indices)
        # 注意：中间那个返回值原本是 all_heavy_atom_idxs，但在苯环这通常只关心 C
        # 我们可以复用 fused system 的逻辑返回所有重原子索引，或者保持现状。
        # 这里保持现状，只返回 C 的 cube 索引即可。
        
        return matched_atoms, cube_idxs, md_ring_indices

    def _match_fused_system(self, md_atoms, anchor_com):
        """
        匹配稠环系统（如吲哚 9 原子）：
        逻辑：寻找共享 2 个原子的 [6元环] 和 [5元环] 组合。
        (修复版：遍历所有候选环，寻找指纹匹配成功的一个)
        """
        from scipy.spatial.distance import cdist
        import numpy as np
        
        md_coords = md_atoms.positions
        heavy_mask = [a.name[0] in ['C', 'N', 'O', 'S'] for a in md_atoms]
        heavy_indices_local = [i for i, x in enumerate(heavy_mask) if x]
        
        if len(heavy_indices_local) < 9: return None, None, None
            
        heavy_coords = md_coords[heavy_indices_local]
        # 保持 1.9 阈值
        dmat = cdist(heavy_coords, heavy_coords)
        adj = np.logical_and(dmat > 1.1, dmat < 1.9) 
        
        # --- 内部辅助函数：找环 ---
        def find_rings(target_len):
            found = []
            seen = set()
            def dfs(s, c, p):
                if len(p) == target_len: return p if adj[c, s] else None
                for n in np.where(adj[c])[0]:
                    if n == s and len(p) < target_len - 1: continue
                    if n not in p:
                        r = dfs(s, n, p + [n])
                        if r: return r
                return None
            for i in range(len(heavy_indices_local)):
                if np.sum(adj[i]) >= 2:
                    res = dfs(i, i, [i])
                    if res:
                        s = tuple(sorted(res))
                        if s not in seen: found.append(set(res)); seen.add(s)
            return found
        # ------------------------

        rings_6 = find_rings(6)
        rings_5 = find_rings(5)
        
        if not rings_6 or not rings_5: return None, None, None
            
        # 收集所有符合拓扑条件的候选环
        valid_candidates = []
        
        for r6 in rings_6:
            for r5 in rings_5:
                shared = r6.intersection(r5)
                if len(shared) == 2:
                    fused_set = r6.union(r5)
                    if len(fused_set) != 9: continue
                        
                    current_group_local = list(fused_set)
                    real_indices = [heavy_indices_local[i] for i in current_group_local]
                    
                    # 只有拓扑符合还不够，先存下来，后面去验指纹
                    cent = md_coords[real_indices].mean(0)
                    dist = np.linalg.norm(cent - anchor_com)
                    valid_candidates.append((dist, real_indices))

        # 按距离排序，优先尝试离锚点近的，但如果指纹不对就试下一个
        valid_candidates.sort(key=lambda x: x[0])
        
        for dist, best_indices_local in valid_candidates:
            # 1. 提取无序候选原子
            candidate_atoms = md_atoms[best_indices_local]
            
            # 2. 尝试指纹对齐
            # 注意：如果这个环是“李鬼”（元素不对），这里会直接返回 None
            sorted_local_order = self._align_by_fingerprint(candidate_atoms, md_atoms)
            
            if sorted_local_order is not None:
                # 3. 匹配成功！构造返回值
                final_md_indices = [best_indices_local[i] for i in sorted_local_order]
                matched_atoms = md_atoms[final_md_indices]
                
                ref_heavy_idxs = [i for i, e in enumerate(self.ref_elements) if e in ['C', 'N', 'O', 'S']]
                cube_idxs = [{idx: rank for rank, idx in enumerate(ref_heavy_idxs)}[i] 
                             for i in self.ref_ring_idx if i in ref_heavy_idxs]

                return matched_atoms, cube_idxs, final_md_indices

        # 如果所有候选环都试过了，还是没匹配上
        return None, None, None