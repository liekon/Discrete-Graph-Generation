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


# Moses does not contain hydrogen atoms, so there are only 7 atom types
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
        
        # Moses has no hydrogen atoms, so indices start at 0
        self.atom_types = {"C": 0, "N": 1, "S": 2, "O": 3, "F": 4, "Cl": 5, "Br": 6}
        self.bond_types = {BT.SINGLE: 1, BT.DOUBLE: 2, BT.TRIPLE: 3, BT.AROMATIC: 4}
        
        # Read ring configuration from the config
        if ring_types is None:
            ring_types = ['C1CCC1', 'C1CC1', 'N1CCC1']  # default values matching QM9
        if ring_weights is None:
            ring_weights = [56, 42, 57]  # default values matching QM9
        # Store weights so we can compute molecular masses
        self.ring_weights = ring_weights
        
        # Ensure lengths match
        assert len(ring_types) == len(ring_weights), "ring_types and ring_weights must have identical lengths"
        
        # Build (smiles, label) tuples for each ring
        self.ring_types = []
        base_label = len(self.atom_types)  # start at 7 (seven atom types)
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
        # Use distinct filenames depending on whether ring compression is enabled
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
        
        # Moses has no hydrogen atoms, so the type mapping starts at 0
        types = {atom: i for i, atom in enumerate(self.atom_decoder)}
        bonds = {BT.SINGLE: 0, BT.DOUBLE: 1, BT.TRIPLE: 2, BT.AROMATIC: 3}

        path = self.split_paths[self.file_idx]
        smiles_list = pd.read_csv(path)['SMILES'].values

        # Prebuild ring templates
        ring_graphs = self.build_ring_graphs()
        
        # Map each RING_<label> to a new atom index
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
            
            # Save original SMILES before compression
            original_smiles = mol2smiles(mol)
            if original_smiles is None:
                continue

            # Build a NetworkX graph from the RDKit molecule
            G = nx.Graph()
            for a in mol.GetAtoms():
                idx = a.GetIdx()
                sym = a.GetSymbol()
                G.add_node(idx, symbol=sym)
            for b in mol.GetBonds():
                s = b.GetBeginAtomIdx()
                e = b.GetEndAtomIdx()
                btype = b.GetBondType()
                G.add_edge(s, e, bond_label=(bonds.get(btype, 0) + 1))  # +1 to match the official format

            # Apply ring compression
            if len(ring_graphs) > 0:
                G = self.contract_rings_in_order(G, ring_graphs)

            # Convert the compressed graph into PyG tensors
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

            # Bidirectional edges using the +1 encoding
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
            
            # Moses has no hydrogens, so no shift is required
            x = F.one_hot(
                node_type_idx,
                num_classes=len(types) + len(self.ring_types)
            ).float()

            # Sort edges as in the official release
            N = x.size(0)
            if edge_index.numel() > 0:
            perm = (edge_index[0] * N + edge_index[1]).argsort()
            edge_index = edge_index[:, perm]
            edge_attr = edge_attr[perm]

            y = torch.zeros((1, 0), dtype=torch.float)
            data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, idx=i)
            
            # Store the original SMILES (before compression)
            data.smiles = original_smiles
            
            # Compute molecular mass using base atoms plus supernodes
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
                    # When ring compression is enabled, RDKit cannot rebuild RING_* atoms, so keep the original sample
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
        
        # Read ring configuration from the config
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
        # Share datasets so MOSESinfos can reuse them for statistics
        self.datasets = datasets
        super().__init__(cfg, datasets)




class MOSESinfos(AbstractDatasetInfos):
    def __init__(self, datamodule, cfg):
        self.name = 'MOSES'
        self.input_dims = None
        self.output_dims = None
        self.remove_h = False
        self.need_to_strip = False

        # Read ring configuration from the config
        self.ring_types_list = getattr(cfg.dataset, 'ring_types', ['C1CCC1', 'C1CC1', 'N1CCC1'])
        self.ring_weights_list = getattr(cfg.dataset, 'ring_weights', [56, 42, 57])
        
        # Ensure lengths match
        assert len(self.ring_types_list) == len(self.ring_weights_list), "ring_types and ring_weights must have identical lengths"
        
        # Base atom types (Moses does not include hydrogen)
        self.atom_encoder = {'C': 0, 'N': 1, 'S': 2, 'O': 3, 'F': 4, 'Cl': 5, 'Br': 6}
        base_atom_decoder = ['C', 'N', 'S', 'O', 'F', 'Cl', 'Br']
        base_valencies = [4, 3, 4, 2, 1, 1, 1]
        base_atom_weights = {0: 12, 1: 14, 2: 32, 3: 16, 4: 19, 5: 35.4, 6: 79.9}
        base_num_atom_types = 7
        
        # Append supernode types
        self.atom_decoder = base_atom_decoder + [f"RING_{i}" for i in range(len(self.ring_types_list))]
        self.num_atom_types = base_num_atom_types + len(self.ring_types_list)
        
        # Extend valencies (supernodes have valency 1)
        self.valencies = base_valencies + [1] * len(self.ring_types_list)
        
        # Extend atom weights
        self.atom_weights = base_atom_weights.copy()
        for i, weight in enumerate(self.ring_weights_list):
            self.atom_weights[base_num_atom_types + i] = weight
        
        # Build the label_to_ring mapping
        self.label_to_ring = {}
        base_label = 7  # Moses has 7 atom types, so rings start at 7
        for i, ring_smiles in enumerate(self.ring_types_list):
            self.label_to_ring[base_label + i] = ring_smiles
        
        # Fixed mappings
        self.label_to_symbol = {0: "C", 1: "N", 2: "S", 3: "O", 4: "F", 5: "Cl", 6: "Br"}
        self.label_to_bondtype = {1: rdchem.BondType.SINGLE, 2: rdchem.BondType.DOUBLE, 
                                  3: rdchem.BondType.TRIPLE, 4: rdchem.BondType.AROMATIC}

        # Path to the cached statistics file
        self.statistics_file = getattr(cfg.dataset, 'statistics_after_4', 'data/moses/moses_pyg/statistics_after_compression.json')
        
        if not self._load_statistics():
            print("Statistics file missing or incomplete, recomputing...")
            self._compute_and_save_statistics(datamodule)

        # Recompute valency_distribution if it is still missing after loading
        if not hasattr(self, 'valency_distribution') or self.valency_distribution is None:
            print("Computing valency_distribution...")
            self.valency_distribution = datamodule.valency_count(self.max_n_nodes)
            # Update the statistics file
            try:
                with open(self.statistics_file, 'r') as f:
                    stats = json.load(f)
                stats['valency_distribution'] = self.valency_distribution.tolist()
                with open(self.statistics_file, 'w') as f:
                    json.dump(stats, f, indent=2)
            except Exception as e:
                print(f"Failed to update statistics file: {e}")

        super().complete_infos(n_nodes=self.n_nodes, node_types=self.node_types)
            
    def _load_statistics(self):
        """Attempt to load statistics from disk."""
        if not os.path.exists(self.statistics_file):
            return False
        
        try:
            with open(self.statistics_file, 'r') as f:
                stats = json.load(f)
            
            # Ensure required statistics are present
            required_keys = ['max_n_nodes', 'max_weight', 'n_nodes', 'node_types', 'edge_types']
            if not all(key in stats for key in required_keys):
                print(f"Statistics file missing required keys: {required_keys}")
                return False
            
            # Ensure the statistics match the current configuration
            expected_num_node_types = 7 + len(self.ring_types_list)  # base 7 types + rings
            if len(stats['node_types']) != expected_num_node_types:
                print(f"Node type count mismatch: expected {expected_num_node_types}, got {len(stats['node_types'])}")
                return False
            
            # Load statistics values
            self.max_n_nodes = stats['max_n_nodes']
            self.max_weight = stats['max_weight']
            self.n_nodes = torch.tensor(stats['n_nodes'])
            self.node_types = torch.tensor(stats['node_types'])
            self.edge_types = torch.tensor(stats['edge_types'])
            
            # Load valency_distribution if present, otherwise compute later
            if 'valency_distribution' in stats:
                self.valency_distribution = torch.tensor(stats['valency_distribution'])
            else:
                self.valency_distribution = None
            
            print(f"Loaded statistics from {self.statistics_file}")
            return True
            
        except Exception as e:
            print(f"Failed to load statistics: {e}")
            return False
    
    def _compute_and_save_statistics(self, datamodule):
        """Compute and save dataset statistics."""
        print("Computing statistics for the ring-compressed dataset...")
        
        # Aggregate statistics
            self.n_nodes = datamodule.node_counts()
        self.node_types = datamodule.node_types()
        self.edge_types = datamodule.edge_counts()
            
        # Determine maximum node count
            self.max_n_nodes = len(self.n_nodes) - 1
        
        # Compute valency distribution if needed
        if not hasattr(self, 'valency_distribution') or self.valency_distribution is None:
            self.valency_distribution = datamodule.valency_count(self.max_n_nodes)
            
        # Observe maximum molecular weight
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
            # Upper bound approximation: assume all nodes are the heaviest atom (Br=79.9) plus supernodes
            max_ring_weight = max(self.ring_weights_list) if self.ring_weights_list else 100
            estimated_max = self.max_n_nodes * 79.9 + max_ring_weight * 10  # assume up to 10 supernodes
            self.max_weight = estimated_max
            print(f"Could not observe max weight directly; using upper bound {self.max_weight}")
        
        # Persist statistics to JSON
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
        
        print(f"Saved statistics to {self.statistics_file}")
        print(f"Max nodes: {self.max_n_nodes}, max weight: {self.max_weight}")


def get_train_smiles(cfg, train_dataloader, dataset_infos, evaluate_dataset=False):
    """Collect training SMILES strings either from saved attributes or from file."""
    train_smiles = []
    for data in train_dataloader:
        if hasattr(data, 'smiles'):
            if isinstance(data.smiles, (list, tuple)):
                train_smiles.extend(data.smiles)
            else:
                train_smiles.append(data.smiles)
    
    if len(train_smiles) == 0:
        print("Warning: could not gather SMILES from the dataset, falling back to file input...")
    base_path = pathlib.Path(os.path.realpath(__file__)).parents[2]
        smiles_path = os.path.join(base_path, cfg.dataset.datadir, 'train_moses.csv')
    if os.path.exists(smiles_path):
            df = pd.read_csv(smiles_path)
            train_smiles = df['SMILES'].tolist()
            print(f"Loaded {len(train_smiles)} SMILES from file")
        else:
            print("SMILES file not found")
            return None
    train_smiles = [s.strip() if isinstance(s, str) else str(s) for s in train_smiles]
    print(f"Collected {len(train_smiles)} training SMILES")

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
    # Test harness removed; use main.py for training
    pass