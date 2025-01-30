import os
import networkx as nx
import torch
import torch.nn.functional as F

from rdkit import Chem
from rdkit.Chem import BondType as BT
from torch_geometric.data import Data
from torch_geometric.utils import subgraph

from rdkit.Chem import Descriptors
from tqdm import tqdm
import pickle
import numpy as np
import matplotlib.pyplot as plt
import os
import torch
import numpy as np

########################################################
# 1) 全局设置
########################################################

# 普通原子 -> label
atom_types = {"H": 0, "C": 1, "N": 2, "S": 3, "O": 4, "F": 5, "Cl": 6, "Br": 7}

# 键类型 -> label
bond_types = {BT.SINGLE:1, BT.DOUBLE:2, BT.TRIPLE:3, BT.AROMATIC:4}

# 10种目标环: (环SMILES, label)
# index小 => 优先检测
ring_types = [
    ("C1=CC=CC=C1", 8),
    ("C1=CC=CN=C1", 9),
    ("C1=C[NH]N=C1", 10),
    ("C1CCNCC1", 11),
    ("C1CCNC1", 12),
    ("C1=CSC=C1", 13),
    ("C1=CSC=N1", 14),
    ("C1COCCN1", 15),
    ("C1CNCCN1", 16),
    ("C1=COC=C1", 17),
    ("C1=CN=CN=C1", 18),
    ("C1=C[NH]C=C1", 19),
    ("C1=CON=C1", 20),
    ("C1=N[NH]C=C1", 21),
    ("C1CCCCC1", 22)
]


########################################################
# 2) 目标环解析: 构建 ring_nx_list: 每个环的 graph
########################################################
def build_ring_graphs():
    """
    将 ring_types 中的 每个 (ringsmi, label) 构建为 (ring_graph, label)
    ring_graph: networkx.Graph, 用于之后同构检测
    """
    ring_graphs = []
    for (r_smi, r_label) in ring_types:
        g = smiles_to_nx(r_smi, as_ring=True)
        ring_graphs.append((g, r_label, r_smi))
    return ring_graphs


########################################################
# 3) SMILES -> NetworkX图
########################################################
def smiles_to_nx(smi, as_ring=False):
    """
    解析 SMILES => RDKit Mol => 构建 networkx.Graph
    如果 as_ring=True, 说明它本身就是个环, 不需要再环检测, 直接全部节点
    """
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        # 说明该smiles不合法
        return None

    G = nx.Graph()

    # 加节点
    for a in mol.GetAtoms():
        idx = a.GetIdx()
        sym = a.GetSymbol()
        G.add_node(idx, symbol=sym)

    # 加边
    for b in mol.GetBonds():
        s = b.GetBeginAtomIdx()
        e = b.GetEndAtomIdx()
        btype = b.GetBondType()
        b_label = bond_types.get(btype, 0)
        G.add_edge(s, e, bond_label=b_label)

    if as_ring:
        # 这个函数是给 ring_smi使用的
        # 不必做别的
        pass

    return G


########################################################
# 4) 图同构检测
########################################################
def find_subgraph_isomorphism(main_g, pattern_g):
    """
    在 main_g 中搜索 pattern_g 的同构子图.
    如果找到, 返回 (True, mapping), 否则 (False, None)
    仅检测 "symbol" 和 "bond_label" 的匹配.

    这里使用 networkx.algorithms.isomorphism, 需要构造 GraphMatcher
    """
    from networkx.algorithms import isomorphism

    nm = isomorphism.GraphMatcher(
        main_g, pattern_g,
        node_match=node_eq,
        edge_match=edge_eq
    )
    # 返回第一个match
    if nm.subgraph_is_isomorphic():
        for sub_mapping in nm.subgraph_isomorphisms_iter():
            # sub_mapping: main_node -> pattern_node
            return True, sub_mapping
    return False, None

def node_eq(n1, n2):
    # 要求 symbol相同
    return n1.get("symbol","?") == n2.get("symbol","?")

def edge_eq(e1, e2):
    # 要求 bond_label 相同
    return e1.get("bond_label",0) == e2.get("bond_label",0)


########################################################
# 5) 收缩环
########################################################
def contract_ring(main_g, sub_mapping, pattern_g, ring_label):
    """
    将 main_g 中 sub_mapping对应的那些节点(=pattern_g的全部节点)
    收缩为一个超点(label=ring_label).
    保留对外连接.

    步骤:
    1) 找到子图节点 set: ring_nodes
    2) 收集 ring_nodes 跟外部节点之间的边 => external_edges
    3) main_g中删除 ring_nodes
    4) 新增一个 supernode => label= ring_label
    5) 在 supernode 与 external_nodes之间加边( bond_label=1 ?? ) => 这里采用"保留原bond_label"
       但如果多个 ring节点 => 其bond_label 可能不同, 
       这里简化 => 只保留( ring节点, outside )的 bond_label中 "最小"或“第一个”?
    """
    ring_nodes = set(sub_mapping.keys())

    # 2) 收集外部连边
    external_edges = []  # list of (outside_node, bond_label)
    # 这里可能多个 ring_node连到同一个 outside => 需要决定哪条bond_label?

    for rn in ring_nodes:
        for neighbor in list(main_g[rn]):
            if neighbor not in ring_nodes:
                # ring->outside
                b_label = main_g[rn][neighbor]['bond_label']
                external_edges.append((rn, neighbor, b_label))

    # 3) 删除 ring_nodes
    #   先记录 ring_nodes list
    #   逆序可能不需要,  networkx只要remove_node(x)即可
    for rn in ring_nodes:
        main_g.remove_node(rn)

    # 4) 新增supernode => idx可用一个新的 int, 
    #   这里 simplest: super_idx = negative or a large int
    #   也可用 n+1
    new_super_idx = get_new_node_id(main_g)
    main_g.add_node(new_super_idx, symbol=f"RING_{ring_label}")

    # 5) 恢复外部连边 => supernode <-> outside
    #   可能出现同一个 outside 多次, 这里只连一次, bond_label 可保留第一次
    outside_map = {}  # outside->bond_label
    for (rnode, onode, blbl) in external_edges:
        if onode not in outside_map:
            outside_map[onode] = blbl
        else:
            # 多条 => 取第一个 or min
            # 这里简单保留第一个
            pass

    for out_n, b_l in outside_map.items():
        main_g.add_edge(new_super_idx, out_n, bond_label=b_l)


def get_new_node_id(G):
    """ 找一个不在G的节点id, simplest: max+1 """
    if len(G.nodes)==0:
        return 0
    else:
        return max(G.nodes)+1


########################################################
# 6) 循环收缩: 按 ring_types顺序, 只收缩一个 => break => 重新来
########################################################
def contract_rings_in_order(G, ring_graphs):
    """
    ring_graphs: [(pattern_g, ring_label, ring_smi), ...]  (与 ring_types对应)
    逻辑:
    while True:
       found_ring = False
       for each pattern in ring_graphs:
         => find_subgraph_isomorphism
         if found => contract_ring => found_ring=True => break
       if not found_ring => break
    """
    while True:
        ring_found = False
        for (rg, rlbl, r_smi) in ring_graphs:
            ok, mapping = find_subgraph_isomorphism(G, rg)
            if ok:
                # 收缩
                contract_ring(G, mapping, rg, rlbl)
                ring_found = True
                break  # 只收缩一个
        if not ring_found:
            break
    return G


########################################################
# 7) 将最终图 => PyG.Data
########################################################
def graph_to_pyg_data(G, idx):
    """
    G中:
     - 普通原子: symbol in {H,C,N,O,F}
     - 超点: symbol="RING_{ring_label}"
    构建 node_label_idx / edges
    """
    node_label_idx = []
    # 保证节点顺序 => 排序
    sorted_nodes = sorted(G.nodes())
    node_map = {}  # old-> new

    for i, n in enumerate(sorted_nodes):
        node_map[n] = i

    for n in sorted_nodes:
        sym = G.nodes[n].get("symbol","?")
        if sym.startswith("RING_"):
            # ring_label
            ring_lbl_str = sym.split("_")[1]  # e.g. "RING_5" => "5"
            ring_lbl = int(ring_lbl_str)
            node_label_idx.append(ring_lbl)  # 5..23
        else:
            # 普通原子
            # e.g. C=1, N=2, ...
            node_label_idx.append(atom_types.get(sym, 0))  # default H=0

    edges = []
    for (u,v) in G.edges():
        # main => new
        nu = node_map[u]
        nv = node_map[v]
        b_label = G[u][v].get("bond_label",0)
        # 无向 => 双向
        edges.append((nu,nv,b_label))
        edges.append((nv,nu,b_label))

    # 构建
    row, col, e_label = [],[],[]
    for (uu,vv,bb) in edges:
        row.append(uu); col.append(vv); e_label.append(bb)

    edge_index = torch.tensor([row,col], dtype=torch.long)
    edge_label = torch.tensor(e_label, dtype=torch.long)
    edge_attr = F.one_hot(edge_label, num_classes=5).float()  # 0..4

    node_label_idx = torch.tensor(node_label_idx).long()  # 0..14
    x = F.one_hot(node_label_idx, num_classes=23).float()  # shape(n,23)

    # 移除氢原子 => label=0
    to_keep = (node_label_idx>0)
    edge_index, edge_attr = subgraph(to_keep, edge_index, edge_attr, relabel_nodes=True,
                                     num_nodes=node_label_idx.size(0))
    x = x[to_keep]

    # shift => x[:,1:]
    x = x[:,1:]  # => (n',22)

    y = torch.zeros((1,0), dtype=torch.float)
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, idx=torch.tensor([idx]))
    return data


########################################################
# 8) 主流程： 解析 -> 收缩 -> PyG
########################################################

def process_single_smiles(smi, idx, ring_graphs):
    """
    1) smiles -> graph G
    2) 按顺序收缩
    3) G->Data
    """
    # 1) 解析
    rd_mol = Chem.MolFromSmiles(smi)
    if rd_mol is None:
        print(f"Warning: 无法解析SMILES: {smi}")
        return None

    # RDKit -> Nx
    G = smiles_to_nx(smi)
    if G is None or len(G)==0:
        return None

    # 2) 按顺序收缩
    G = contract_rings_in_order(G, ring_graphs)

    # 3) 转成 Data
    data = graph_to_pyg_data(G, idx)
    return data


def process_smiles_list(smiles_list):
    """
    对外接口: 读取 ring_graphs, 然后对每个 smiles 构造 Data
    """
    ring_graphs = build_ring_graphs()  # [(pattern_g, ring_label, ring_smi), ...]
    data_list = []
    for i, smi in enumerate(tqdm(smiles_list)):
        d = process_single_smiles(smi, i, ring_graphs)
        if d is not None:
            data_list.append(d)
    gather_stats(data_list)
    return data_list

def gather_stats(data_list):
    """
    需求：
    1) n_max: 所有 Data中节点数量的最大值
    2) node_count_dist: 长度 (n_max+1) 的列表, node_count_dist[i]表示节点数为i的分子数
       (注: sum(node_count_dist)=分子总数)
    3) node_type_freq: 长度14, 累计所有Data的节点类型出现次数
    4) edge_type_freq: 长度4, 累计所有Data的边类型(1..4)出现次数
    """
    if not data_list:
        print("No data to gather stats.")
        return

    # 1) 先找 n_max
    max_n = 0
    for d in data_list:
        n = d.x.size(0)  # 节点数
        if n>max_n:
            max_n = n

    # 2) 构建 node_count_dist
    node_count_dist = [0]*(max_n+1)

    # 3) node_type_freq => 23
    node_type_freq = [0]*23

    # 4) edge_type_freq => 4
    edge_type_freq = [0]*5

    for d in data_list:
        if d % 5000 == 0:
            print(d)
        n = d.x.size(0)
        node_count_dist[n]+=1

        # 节点类型: d.x.argmax(dim=1) => [0..21]
        if n>0:
            labels = d.x.argmax(dim=1)  # shape(n,)
            for lbl in labels.tolist():
                node_type_freq[lbl]+=1

        # 边类型: d.edge_attr.argmax(dim=1) => [0..4], 0表示无边(但实际上已保留?), 1..4是4种
        eattr_label = d.edge_attr.argmax(dim=1)  # shape(e,)
        for lbl_e in eattr_label.tolist():
            edge_type_freq[lbl_e]+=1

    # 打印结果
    print("\n===== Statistics =====")
    print(f"(1) n_max = {max_n}")
    print("(2) node_count_dist (index=节点数, value=分子数量):")
    print(node_count_dist)
    # sum(node_count_dist) = len(data_list)

    print("(3) node_type_freq (长度14): [C(0),N(1),O(2),F(3),RING_5(4),..., RING_14(13)] 的出现总次数")
    print(node_type_freq)
    # sum(node_type_freq)= 所有分子节点数(移除氢后)

    print("(4) edge_type_freq (长度4): single, double, triple, aromatic")
    print(edge_type_freq)
    # sum(edge_type_freq) = 所有分子的边总数(仅label=1..4)

def main():
    smiles_file_path = "/data/shared/peizhi/data/moses/new_train.smiles"
    if not os.path.exists(smiles_file_path):
        print(f"Error: file {smiles_file_path} not found.")
        return

    with open(smiles_file_path,"r") as f:
        smiles_list = [ln.strip() for ln in f]


    data_list = process_smiles_list(smiles_list)
    print(f"Done. Got {len(data_list)} Data objects.")

if __name__=="__main__":
    main()