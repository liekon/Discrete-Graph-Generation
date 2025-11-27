from rdkit import Chem, RDLogger
from rdkit.Chem.rdchem import BondType as BT

import os
import os.path as osp
import pathlib
from typing import Any, Sequence

import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
from torch_geometric.data import Data, InMemoryDataset, download_url
from torch_geometric.utils import subgraph
import pandas as pd
import networkx as nx
import json
from omegaconf import OmegaConf

import utils as utils
from datasets.abstract_dataset import AbstractDatasetInfos, MolecularDataModule
from analysis.rdkit_functions import mol2smiles, build_molecule_with_partial_charges, compute_molecular_metrics
from rdkit.Chem import rdchem


def to_list(value: Any) -> Sequence:
    if isinstance(value, Sequence) and not isinstance(value, str):
        return value
    else:
        return [value]


# moses 不包含 H，所以原子类型只有 7 种
atom_decoder = ['C', 'N', 'S', 'O', 'F', 'Cl', 'Br']


class MOSESDataset(InMemoryDataset):
    train_url = 'https://media.githubusercontent.com/media/molecularsets/moses/master/data/train.csv'
    val_url = 'https://media.githubusercontent.com/media/molecularsets/moses/master/data/test.csv'
    test_url = 'https://media.githubusercontent.com/media/molecularsets/moses/master/data/test_scaffolds.csv'

    def __init__(self, stage, root, filter_dataset: bool, transform=None, pre_transform=None, pre_filter=None, 
                 ring_types=None, ring_weights=None):
        self.stage = stage
        self.atom_decoder = atom_decoder
        self.filter_dataset = filter_dataset
        
        # moses 不包含 H，原子类型从 0 开始
        self.atom_types = {"C": 0, "N": 1, "S": 2, "O": 3, "F": 4, "Cl": 5, "Br": 6}
        self.bond_types = {BT.SINGLE: 1, BT.DOUBLE: 2, BT.TRIPLE: 3, BT.AROMATIC: 4}
        
        # 从config读取ring_types和ring_weights
        if ring_types is None:
            ring_types = ['C1CCC1', 'C1CC1', 'N1CCC1']  # 默认值（与qm9一致）
        if ring_weights is None:
            ring_weights = [56, 42, 57]  # 默认值（与qm9一致）
        # 保存权重以便质量计算
        self.ring_weights = ring_weights
        
        # 确保ring_types和ring_weights长度一致
        assert len(ring_types) == len(ring_weights), "ring_types和ring_weights长度必须一致"
        
        # 构建ring_types列表，格式为[(smiles, label), ...]
        self.ring_types = []
        base_label = len(self.atom_types)  # 从7开始（moses有7种原子类型）
        for i, (smiles, weight) in enumerate(zip(ring_types, ring_weights)):
            self.ring_types.append((smiles, base_label + i))
        
        self.ring_dict = {}
        
        if self.stage == 'train':
            self.file_idx = 0
        elif self.stage == 'val':
            self.file_idx = 1
        else:
            self.file_idx = 2
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[self.file_idx])

    @property
    def raw_file_names(self):
        return ['train_moses.csv', 'val_moses.csv', 'test_moses.csv']

    @property
    def split_file_name(self):
        return ['train_moses.csv', 'val_moses.csv', 'test_moses.csv']

    @property
    def split_paths(self):
        r"""The absolute filepaths that must be present in order to skip
        splitting."""
        files = to_list(self.split_file_name)
        return [osp.join(self.raw_dir, f) for f in files]

    @property
    def processed_file_names(self):
        # 根据是否有环压缩使用不同的文件名
        ring_suffix = '_ring' if len(self.ring_types) > 0 else ''
        if self.filter_dataset:
            return [f'train_filtered{ring_suffix}.pt', f'test_filtered{ring_suffix}.pt', f'test_scaffold_filtered{ring_suffix}.pt']
        else:
            return [f'train{ring_suffix}.pt', f'test{ring_suffix}.pt', f'test_scaffold{ring_suffix}.pt']

    def download(self):
        import rdkit  # noqa
        train_path = download_url(self.train_url, self.raw_dir)
        os.rename(train_path, osp.join(self.raw_dir, 'train_moses.csv'))

        test_path = download_url(self.test_url, self.raw_dir)
        os.rename(test_path, osp.join(self.raw_dir, 'val_moses.csv'))

        valid_path = download_url(self.val_url, self.raw_dir)
        os.rename(valid_path, osp.join(self.raw_dir, 'test_moses.csv'))

    def build_ring_graphs(self):
        ring_graphs = []
        for (r_smi, r_label) in self.ring_types:
            g = self.smiles_to_nx(r_smi, as_ring=True)
            if g is not None:
                ring_graphs.append((g, r_label, r_smi))
        return ring_graphs

    def smiles_to_nx(self, smi, as_ring=False):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return None
        
        G = nx.Graph()
        
        for a in mol.GetAtoms():
            idx = a.GetIdx()
            sym = a.GetSymbol()
            G.add_node(idx, symbol=sym)
        
        for b in mol.GetBonds():
            s = b.GetBeginAtomIdx()
            e = b.GetEndAtomIdx()
            btype = b.GetBondType()
            b_label = self.bond_types.get(btype, 0)
            G.add_edge(s, e, bond_label=b_label)
        
        return G

    def find_subgraph_isomorphism(self, main_g, pattern_g):
        from networkx.algorithms import isomorphism
        
        nm = isomorphism.GraphMatcher(
            main_g, pattern_g,
            node_match=self.node_eq,
            edge_match=self.edge_eq
        )

        if nm.subgraph_is_isomorphic():
            for sub_mapping in nm.subgraph_isomorphisms_iter():
                return True, sub_mapping
        return False, None

    def node_eq(self, n1, n2):
        return n1.get("symbol", "?") == n2.get("symbol", "?")

    def edge_eq(self, e1, e2):
        return e1.get("bond_label", 0) == e2.get("bond_label", 0)

    def contract_ring(self, main_g, sub_mapping, pattern_g, ring_label):
        ring_nodes = set(sub_mapping.keys())
        
        external_edges = []
        
        for rn in ring_nodes:
            for neighbor in list(main_g[rn]):
                if neighbor not in ring_nodes:
                    b_label = main_g[rn][neighbor]['bond_label']
                    external_edges.append((rn, neighbor, b_label))
        
        for rn in ring_nodes:
            main_g.remove_node(rn)
        
        new_super_idx = self.get_new_node_id(main_g)
        main_g.add_node(new_super_idx, symbol=f"RING_{ring_label}")
        
        outside_map = {}
        for (rnode, onode, blbl) in external_edges:
            if onode not in outside_map:
                outside_map[onode] = blbl
        
        for out_n, b_l in outside_map.items():
            main_g.add_edge(new_super_idx, out_n, bond_label=b_l)

    def get_new_node_id(self, G):
        if len(G.nodes) == 0:
            return 0
        else:
            return max(G.nodes) + 1

    def contract_rings_in_order(self, G, ring_graphs):
        while True:
            ring_found = False
            for (rg, rlbl, r_smi) in ring_graphs:
                ok, mapping = self.find_subgraph_isomorphism(G, rg)
                if ok:
                    self.contract_ring(G, mapping, rg, rlbl)
                    ring_found = True
                    break
            if not ring_found:
                break
        return G

    def process(self):
        RDLogger.DisableLog('rdApp.*')
        
        # moses 不包含 H，所以类型映射直接从 0 开始
        types = {atom: i for i, atom in enumerate(self.atom_decoder)}
        bonds = {BT.SINGLE: 0, BT.DOUBLE: 1, BT.TRIPLE: 2, BT.AROMATIC: 3}

        path = self.split_paths[self.file_idx]
        smiles_list = pd.read_csv(path)['SMILES'].values

        # 预构建环模板
        ring_graphs = self.build_ring_graphs()
        
        # 超点类型映射：将RING_<label>映射到新增的原子类型索引
        base_type_offset = len(self.atom_types)  # 7 (C, N, S, O, F, Cl, Br)
        supernode_label_to_type = {}
        for i, (_, r_label) in enumerate(self.ring_types):
            supernode_label_to_type[r_label] = base_type_offset + i

        data_list = []
        smiles_kept = []

        for i, smile in enumerate(tqdm(smiles_list)):
            mol = Chem.MolFromSmiles(smile)
            if mol is None:
                continue
            
            # 在压缩之前保存原始 SMILES
            original_smiles = mol2smiles(mol)
            if original_smiles is None:
                continue

            # 从RDKit分子构建NX图
            G = nx.Graph()
            for a in mol.GetAtoms():
                idx = a.GetIdx()
                sym = a.GetSymbol()
                G.add_node(idx, symbol=sym)
            for b in mol.GetBonds():
                s = b.GetBeginAtomIdx()
                e = b.GetEndAtomIdx()
                btype = b.GetBondType()
                G.add_edge(s, e, bond_label=(bonds.get(btype, 0) + 1))  # 与官方一致：写入时+1

            # 环压缩
            if len(ring_graphs) > 0:
                G = self.contract_rings_in_order(G, ring_graphs)

            # 将压缩后的图转换为PyG张量
            node_type_idx = []
            sorted_nodes = sorted(G.nodes())
            node_map = {n: idx for idx, n in enumerate(sorted_nodes)}
            for n in sorted_nodes:
                sym = G.nodes[n].get('symbol', '?')
                if sym.startswith('RING_'):
                    r_lbl = int(sym.split('_')[1])
                    node_type_idx.append(supernode_label_to_type[r_lbl])
                else:
                    mapped = self.atom_types.get(sym, 0)
                    node_type_idx.append(mapped)

            # 边（双向）与类型（已是+1）
            row, col, edge_type = [], [], []
            for (u, v) in G.edges():
                et = int(G[u][v].get('bond_label', 0))
                row += [node_map[u], node_map[v]]
                col += [node_map[v], node_map[u]]
                edge_type += [et, et]

            if len(row) == 0:
                continue

            edge_index = torch.tensor([row, col], dtype=torch.long)
            edge_type = torch.tensor(edge_type, dtype=torch.long)
            edge_attr = F.one_hot(edge_type, num_classes=len(bonds) + 1).to(torch.float)

            node_type_idx = torch.tensor(node_type_idx).long()
            
            # moses 不包含 H，无需左移处理
            x = F.one_hot(
                node_type_idx,
                num_classes=len(types) + len(self.ring_types)
            ).float()

            # 与官方一致的排序
            N = x.size(0)
            if edge_index.numel() > 0:
                perm = (edge_index[0] * N + edge_index[1]).argsort()
                edge_index = edge_index[:, perm]
                edge_attr = edge_attr[perm]

            y = torch.zeros((1, 0), dtype=torch.float)
            data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, idx=i)
            
            # 保存原始 SMILES（压缩前）
            data.smiles = original_smiles
            
            # 计算并存储分子质量：基础原子权重 + 超点权重
            try:
                base_weights = [12, 14, 32, 16, 19, 35.4, 79.9]  # C, N, S, O, F, Cl, Br
                ring_weights = list(self.ring_weights) if hasattr(self, 'ring_weights') else [1] * len(self.ring_types)
                weights = torch.tensor(base_weights + ring_weights, dtype=torch.float)
                type_counts = node_type_idx.bincount(minlength=len(base_weights) + len(self.ring_types)).to(torch.float)
                total_mass = float((type_counts * weights).sum().item())
                data.mol_weight = torch.tensor([total_mass], dtype=torch.float)
            except Exception:
                pass

            if self.filter_dataset:
                if len(self.ring_types) == 0:
                    dense_data, node_mask = utils.to_dense(data.x, data.edge_index, data.edge_attr, data.batch)
                    dense_data = dense_data.mask(node_mask, collapse=True)
                    X, E = dense_data.X, dense_data.E

                    assert X.size(0) == 1
                    atom_types = X[0]
                    edge_types = E[0]
                    mol = build_molecule_with_partial_charges(atom_types, edge_types, self.atom_decoder)
                    smiles = mol2smiles(mol)
                    if smiles is not None:
                        try:
                            mol_frags = Chem.rdmolops.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
                            if len(mol_frags) == 1:
                                data_list.append(data)
                                smiles_kept.append(smiles)

                        except Chem.rdchem.AtomValenceException:
                            print("Valence error in GetmolFrags")
                        except Chem.rdchem.KekulizeException:
                            print("Can't kekulize molecule")
                else:
                    # 启用环压缩后，无法在RDKit中重建RING_*原子，因此直接保留原始样本
                    data_list.append(data)
                    smiles_kept.append(original_smiles)
            else:
                if self.pre_filter is not None and not self.pre_filter(data):
                    continue
                if self.pre_transform is not None:
                    data = self.pre_transform(data)
                data_list.append(data)

        torch.save(self.collate(data_list), self.processed_paths[self.file_idx])

        if self.filter_dataset:
            smiles_save_path = osp.join(pathlib.Path(self.raw_paths[0]).parent, f'new_{self.stage}.smiles')
            print(smiles_save_path)
            with open(smiles_save_path, 'w') as f:
                f.writelines('%s\n' % s for s in smiles_kept)
            print(f"Number of molecules kept: {len(smiles_kept)} / {len(smiles_list)}")



class MosesDataModule(MolecularDataModule):
    def __init__(self, cfg):
        self.remove_h = False
        self.datadir = cfg.dataset.datadir
        self.filter_dataset = cfg.dataset.filter
        
        # 从config读取ring_types和ring_weights
        self.ring_types = getattr(cfg.dataset, 'ring_types', ['C1CCC1', 'C1CC1', 'N1CCC1'])
        self.ring_weights = getattr(cfg.dataset, 'ring_weights', [56, 42, 57])
        
        base_path = pathlib.Path(os.path.realpath(__file__)).parents[2]
        root_path = os.path.join(base_path, self.datadir)
        datasets = {'train': MOSESDataset(stage='train', root=root_path, filter_dataset=self.filter_dataset,
                                          ring_types=self.ring_types, ring_weights=self.ring_weights),
                    'val': MOSESDataset(stage='val', root=root_path, filter_dataset=self.filter_dataset,
                                       ring_types=self.ring_types, ring_weights=self.ring_weights),
                    'test': MOSESDataset(stage='test', root=root_path, filter_dataset=self.filter_dataset,
                                        ring_types=self.ring_types, ring_weights=self.ring_weights)}
        # 供MOSESinfos访问统计用样本
        self.datasets = datasets
        super().__init__(cfg, datasets)




class MOSESinfos(AbstractDatasetInfos):
    def __init__(self, datamodule, cfg):
        self.name = 'MOSES'
        self.input_dims = None
        self.output_dims = None
        self.remove_h = False
        self.need_to_strip = False

        # 从config读取ring_types和ring_weights
        self.ring_types_list = getattr(cfg.dataset, 'ring_types', ['C1CCC1', 'C1CC1', 'N1CCC1'])
        self.ring_weights_list = getattr(cfg.dataset, 'ring_weights', [56, 42, 57])
        
        # 确保长度一致
        assert len(self.ring_types_list) == len(self.ring_weights_list), "ring_types和ring_weights长度必须一致"
        
        # 基础原子类型（moses不包含H）
        self.atom_encoder = {'C': 0, 'N': 1, 'S': 2, 'O': 3, 'F': 4, 'Cl': 5, 'Br': 6}
        base_atom_decoder = ['C', 'N', 'S', 'O', 'F', 'Cl', 'Br']
        base_valencies = [4, 3, 4, 2, 1, 1, 1]
        base_atom_weights = {0: 12, 1: 14, 2: 32, 3: 16, 4: 19, 5: 35.4, 6: 79.9}
        base_num_atom_types = 7
        
        # 添加超点类型
        self.atom_decoder = base_atom_decoder + [f"RING_{i}" for i in range(len(self.ring_types_list))]
        self.num_atom_types = base_num_atom_types + len(self.ring_types_list)
        
        # 扩展valencies（超点的valency都是1）
        self.valencies = base_valencies + [1] * len(self.ring_types_list)
        
        # 扩展atom_weights
        self.atom_weights = base_atom_weights.copy()
        for i, weight in enumerate(self.ring_weights_list):
            self.atom_weights[base_num_atom_types + i] = weight
        
        # 动态生成label_to_ring映射
        self.label_to_ring = {}
        base_label = 7  # moses有7种原子类型，超点从7开始
        for i, ring_smiles in enumerate(self.ring_types_list):
            self.label_to_ring[base_label + i] = ring_smiles
        
        # 固定映射
        self.label_to_symbol = {0: "C", 1: "N", 2: "S", 3: "O", 4: "F", 5: "Cl", 6: "Br"}
        self.label_to_bondtype = {1: rdchem.BondType.SINGLE, 2: rdchem.BondType.DOUBLE, 
                                  3: rdchem.BondType.TRIPLE, 4: rdchem.BondType.AROMATIC}

        # 从config读取统计信息文件路径
        self.statistics_file = getattr(cfg.dataset, 'statistics_after_4', 'data/moses/moses_pyg/statistics_after_compression.json')
        
        if not self._load_statistics():
            print("统计信息文件不存在或内容不充足，开始重新计算...")
            self._compute_and_save_statistics(datamodule)

        # 如果从文件加载成功或重新计算后，仍缺少 valency_distribution，则重新计算一次
        if not hasattr(self, 'valency_distribution') or self.valency_distribution is None:
            print("计算 valency_distribution...")
            self.valency_distribution = datamodule.valency_count(self.max_n_nodes)
            # 更新统计文件
            try:
                with open(self.statistics_file, 'r') as f:
                    stats = json.load(f)
                stats['valency_distribution'] = self.valency_distribution.tolist()
                with open(self.statistics_file, 'w') as f:
                    json.dump(stats, f, indent=2)
            except Exception as e:
                print(f"更新统计文件失败: {e}")

        super().complete_infos(n_nodes=self.n_nodes, node_types=self.node_types)
            
    def _load_statistics(self):
        """尝试从文件加载统计信息"""
        if not os.path.exists(self.statistics_file):
            return False
        
        try:
            with open(self.statistics_file, 'r') as f:
                stats = json.load(f)
            
            # 检查必需的统计信息是否存在
            required_keys = ['max_n_nodes', 'max_weight', 'n_nodes', 'node_types', 'edge_types']
            if not all(key in stats for key in required_keys):
                print(f"统计信息文件缺少必需的键: {required_keys}")
                return False
            
            # 检查统计信息是否与当前配置匹配
            expected_num_node_types = 7 + len(self.ring_types_list)  # 基础7个 + 超点数量
            if len(stats['node_types']) != expected_num_node_types:
                print(f"节点类型数量不匹配: 期望{expected_num_node_types}, 实际{len(stats['node_types'])}")
                return False
            
            # 加载统计信息
            self.max_n_nodes = stats['max_n_nodes']
            self.max_weight = stats['max_weight']
            self.n_nodes = torch.tensor(stats['n_nodes'])
            self.node_types = torch.tensor(stats['node_types'])
            self.edge_types = torch.tensor(stats['edge_types'])
            
            # 如果文件中有 valency_distribution 就加载，否则稍后计算
            if 'valency_distribution' in stats:
                self.valency_distribution = torch.tensor(stats['valency_distribution'])
            else:
                self.valency_distribution = None
            
            print(f"成功从 {self.statistics_file} 加载统计信息")
            return True
            
        except Exception as e:
            print(f"加载统计信息失败: {e}")
            return False
    
    def _compute_and_save_statistics(self, datamodule):
        """计算并保存统计信息"""
        print("开始计算环压缩后的数据集统计信息...")
        
        # 计算统计信息
        self.n_nodes = datamodule.node_counts()
        self.node_types = datamodule.node_types()
        self.edge_types = datamodule.edge_counts()
            
        # 计算最大节点数
        self.max_n_nodes = len(self.n_nodes) - 1
        
        # 计算 valency_distribution（如果还没有计算）
        if not hasattr(self, 'valency_distribution') or self.valency_distribution is None:
            self.valency_distribution = datamodule.valency_count(self.max_n_nodes)
            
        # 基于实际数据观测最大分子质量
        observed_max_weight = 0.0
        datasets = getattr(datamodule, 'datasets', {})
        if isinstance(datasets, dict):
            for split_name, ds in datasets.items():
                if ds is None:
                    continue
                try:
                    length = len(ds)
                except Exception:
                    continue
                for i in range(length):
                    try:
                        d = ds[i]
                        if hasattr(d, 'mol_weight') and d.mol_weight is not None:
                            w = float(d.mol_weight.item())
                            if w > observed_max_weight:
                                observed_max_weight = w
                    except Exception:
                        continue
        
        if observed_max_weight > 0:
            self.max_weight = observed_max_weight
        else:
            # 上界估计：假设所有节点都是最重的原子类型（Br=79.9）加上所有超点
            max_ring_weight = max(self.ring_weights_list) if self.ring_weights_list else 100
            estimated_max = self.max_n_nodes * 79.9 + max_ring_weight * 10  # 假设最多10个超点
            self.max_weight = estimated_max
            print(f"无法从数据中获取最大质量，使用估计值: {self.max_weight}")
        
        # 保存统计信息到JSON文件
        stats = {
            'max_n_nodes': int(self.max_n_nodes),
            'max_weight': float(self.max_weight),
            'n_nodes': self.n_nodes.tolist(),
            'node_types': self.node_types.tolist(),
            'edge_types': self.edge_types.tolist(),
            'valency_distribution': self.valency_distribution.tolist(),
            'ring_types': OmegaConf.to_container(self.ring_types_list) if hasattr(OmegaConf, 'to_container') else list(self.ring_types_list),
            'ring_weights': OmegaConf.to_container(self.ring_weights_list) if hasattr(OmegaConf, 'to_container') else list(self.ring_weights_list)
        }
        
        os.makedirs(os.path.dirname(self.statistics_file), exist_ok=True)
        with open(self.statistics_file, 'w') as f:
            json.dump(stats, f, indent=2)
        
        print(f"统计信息已保存到 {self.statistics_file}")
        print(f"最大节点数: {self.max_n_nodes}, 最大分子质量: {self.max_weight}")


def get_train_smiles(cfg, train_dataloader, dataset_infos, evaluate_dataset=False):
    """从数据集的smiles属性中获取训练集SMILES"""
    train_smiles = []
    for data in train_dataloader:
        if hasattr(data, 'smiles'):
            if isinstance(data.smiles, (list, tuple)):
                train_smiles.extend(data.smiles)
            else:
                train_smiles.append(data.smiles)
    
    if len(train_smiles) == 0:
        print("警告: 无法从数据集中获取SMILES，尝试从文件读取...")
        base_path = pathlib.Path(os.path.realpath(__file__)).parents[2]
        smiles_path = os.path.join(base_path, cfg.dataset.datadir, 'train_moses.csv')
        if os.path.exists(smiles_path):
            df = pd.read_csv(smiles_path)
            train_smiles = df['SMILES'].tolist()
            print(f"从文件读取了 {len(train_smiles)} 个SMILES")
        else:
            print("无法找到SMILES文件")
            return None
    train_smiles = [s.strip() if isinstance(s, str) else str(s) for s in train_smiles]
    print(f"获取了 {len(train_smiles)} 个训练集SMILES")

    if evaluate_dataset:
        all_molecules = []
        for i, data in enumerate(tqdm(train_dataloader)):
            dense_data, node_mask = utils.to_dense(data.x, data.edge_index, data.edge_attr, data.batch)
            dense_data = dense_data.mask(node_mask, collapse=True)
            X, E = dense_data.X, dense_data.E

            for k in range(X.size(0)):
                n = int(torch.sum((X != -1)[k, :]))
                atom_types = X[k, :n].cpu()
                edge_types = E[k, :n, :n].cpu()
                all_molecules.append([atom_types, edge_types])

        print("Evaluating the dataset -- number of molecules to evaluate", len(all_molecules))
        metrics = compute_molecular_metrics(molecule_list=all_molecules, train_smiles=train_smiles,
                                            dataset_info=dataset_infos)
        print(metrics[0])

    return train_smiles


if __name__ == "__main__":
    # 测试代码已移除，请使用main.py进行训练
    pass