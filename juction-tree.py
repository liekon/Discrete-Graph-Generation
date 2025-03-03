import numpy as np
from rdkit import Chem
from rdkit.Chem import EditableMol
import os
import random
from tqdm import tqdm
from collections import defaultdict

# 设置随机种子以确保可重复性
random.seed(42)
np.random.seed(42)

# 路径配置
base_dir = '/data/shared/vishal/junction_trees/icml18-jtnn/data/moses/graphs_0226/'
output_dir = '/data/shared/vishal/junction_trees/icml18-jtnn/data/moses/'
os.makedirs(output_dir, exist_ok=True)
all_txt_path = os.path.join(output_dir, 'all.txt')


# 确定总文件数
with open(all_txt_path, 'r') as f:
    total_files = len(f.readlines())

all_smiles = []
node_counts = []
total_bonds = 0


for i in tqdm(range(total_files)):
    file_path = os.path.join(base_dir, f'{i}.npz')
    try:
        data = np.load(file_path)
        node_labels = data['node_labels']
        edge_list = data['edge_list']
        
        # 统计信息
        node_count = len(node_labels)
        edge_count = edge_list.shape[0]
        node_counts.append(node_count)
        total_bonds += edge_count

        """ # 计算节点度数
        node_degrees = defaultdict(int)
        for edge in edge_list:
            u = int(edge[0]-1)
            v = int(edge[1]-1)
            if u != v:  # 忽略自环
                node_degrees[u] += 1
                node_degrees[v] += 1 """

        # 创建可编辑分子
        mol = EditableMol(Chem.Mol())
        
        # 添加所有碳原子
        for _ in range(node_count):
            mol.AddAtom(Chem.Atom('S'))
        
        """ # 添加原子（根据度数选择类型）
        atom_map = {}
        for idx in range(node_count):
            degree = node_degrees[idx]
            # 根据度数选择原子类型
            if degree <= 4:
                atom = Chem.Atom('C')
            else:
                atom = Chem.Atom('S')  # 硫原子支持更多连接
                # 设置允许的化合价（重要！）
                atom.SetNumExplicitHs(0)
                atom.SetNoImplicit(True)
            mol.AddAtom(atom)
            atom_map[idx] = idx  # 维持原始索引 """
        
        # 处理边并添加单键
        seen_bonds = set()
        for edge in edge_list:
            u = int(edge[0]-1)
            v = int(edge[1]-1)
            if u == v:  # 跳过自环
                continue
            # 标准化键方向
            if u > v:
                u, v = v, u
            if (u, v) not in seen_bonds:
                mol.AddBond(u, v, Chem.BondType.SINGLE)
                seen_bonds.add((u, v))
        
        # 转换为分子并生成SMILES
        rdkit_mol = mol.GetMol()
        try:
            Chem.SanitizeMol(rdkit_mol)
            smiles = Chem.MolToSmiles(rdkit_mol)
            all_smiles.append(smiles)
        except:
            pass
    except Exception as e:
        print(f'Error processing file {i}: {str(e)}')

# 保存所有有效的SMILES
with open(os.path.join(output_dir, 'all_smiles.smiles'), 'w') as f:
    f.write('\n'.join(all_smiles))

# 随机分割数据集
n = len(all_smiles)
indices = np.random.permutation(n)
train_end = int(0.85 * n)
test_end = train_end + int(0.05 * n)

train = [all_smiles[i] for i in indices[:train_end]]
test = [all_smiles[i] for i in indices[train_end:test_end]]
val = [all_smiles[i] for i in indices[test_end:]]

# 保存分割后的数据集
for name, data in [('train', train), ('test', test), ('val', val)]:
    with open(os.path.join(output_dir, f'{name}.smiles'), 'w') as f:
        f.write('\n'.join(data))

# 统计节点分布
max_nodes = max(node_counts) if node_counts else 0
node_dist = np.zeros(max_nodes + 1, dtype=int)
for count in node_counts:
    node_dist[count] += 1

print("节点数量分布：")
for i, count in enumerate(node_dist):
    if count > 0:
        print(f"{i}个节点: {count}个分子")

print(f"\n总单键数量：{total_bonds}")


