import os
import os.path as osp
import pathlib
from typing import Any, Sequence

import torch
import torch.nn.functional as F
from rdkit import Chem, RDLogger
from rdkit.Chem.rdchem import BondType as BT
from tqdm import tqdm
import numpy as np
import pandas as pd
from torch_geometric.data import Data, InMemoryDataset, download_url, extract_zip
from torch_geometric.utils import subgraph

import utils as utils
from datasets.abstract_dataset import MolecularDataModule, AbstractDatasetInfos
from analysis.rdkit_functions import mol2smiles, build_molecule_with_partial_charges
from analysis.rdkit_functions import compute_molecular_metrics

from rdkit.Chem import Descriptors, rdchem, AllChem, rdmolops
import pickle
import matplotlib.pyplot as plt
from rdkit.Chem import BondType as BT

import networkx as nx
import random
from omegaconf import OmegaConf


def files_exist(files) -> bool:
    # NOTE: We return `False` in case `files` is empty, leading to a
    # re-processing of files on every instantiation.
    return len(files) != 0 and all([osp.exists(f) for f in files])


def to_list(value: Any) -> Sequence:
    if isinstance(value, Sequence) and not isinstance(value, str):
        return value
    else:
        return [value]


class RemoveYTransform:
    def __call__(self, data):
        data.y = torch.zeros((1, 0), dtype=torch.float)
        return data


class SelectMuTransform:
    def __call__(self, data):
        data.y = data.y[..., :1]
        return data


class SelectHOMOTransform:
    def __call__(self, data):
        data.y = data.y[..., 1:]
        return data


class QM9Dataset(InMemoryDataset):
    raw_url = ('https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/'
               'molnet_publish/qm9.zip')
    raw_url2 = 'https://ndownloader.figshare.com/files/3195404'
    processed_url = 'https://data.pyg.org/datasets/qm9_v3.zip'

    def __init__(self, stage, root, remove_h: bool = True, target_prop=None,
                 transform=None, pre_transform=None, pre_filter=None, ring_types=None, ring_weights=None):
        self.target_prop = target_prop
        self.stage = stage
        if self.stage == 'train':
            self.file_idx = 0
        elif self.stage == 'val':
            self.file_idx = 1
        else:
            self.file_idx = 2
        
        # Always enforce remove_h=True
        self.remove_h = True
        self.ring_dict = {}

        # Keep hydrogens for compatibility with the official dataset; they are filtered later
        self.atom_types = {"H": 0, "C": 1, "N": 2, "O": 3, "F": 4}

        self.bond_types = {BT.SINGLE:1, BT.DOUBLE:2, BT.TRIPLE:3, BT.AROMATIC:4}

        # Read ring_types and ring_weights from the config
        if ring_types is None:
            ring_types = ['C1CCC1', 'C1CC1', 'C1CNC1']  # default values
        if ring_weights is None:
            ring_weights = [48, 36, 50]  # default values
        # Store weights so we can compute molecular masses
        self.ring_weights = ring_weights
        
        # Ensure ring_types and ring_weights have the same length
        assert len(ring_types) == len(ring_weights), "ring_types and ring_weights must have identical lengths"
        
        # Build the ring list as (smiles, label) tuples
        self.ring_types = []
        base_label = 5  # start from 5 to avoid collisions with atom labels
        for i, (smiles, weight) in enumerate(zip(ring_types, ring_weights)):
            self.ring_types.append((smiles, base_label + i))

        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[self.file_idx])


    @property
    def raw_file_names(self):
        return ['gdb9.sdf', 'gdb9.sdf.csv', 'uncharacterized.txt']

    @property
    def split_file_name(self):
        return ['train.csv', 'val.csv', 'test.csv']

    @property
    def split_paths(self):

        files = to_list(self.split_file_name)
        return [osp.join(self.raw_dir, f) for f in files]

    @property
    def processed_file_names(self):
        if self.remove_h:
            return ['proc_tr_no_h.pt', 'proc_val_no_h.pt', 'proc_test_no_h.pt']
        else:
            return ['proc_tr_h.pt', 'proc_val_h.pt', 'proc_test_h.pt']

    def download(self):

        try:
            import rdkit  # noqa
            file_path = download_url(self.raw_url, self.raw_dir)
            extract_zip(file_path, self.raw_dir)
            os.unlink(file_path)

            file_path = download_url(self.raw_url2, self.raw_dir)
            os.rename(osp.join(self.raw_dir, '3195404'),
                      osp.join(self.raw_dir, 'uncharacterized.txt'))
        except ImportError:
            path = download_url(self.processed_url, self.raw_dir)
            extract_zip(path, self.raw_dir)
            os.unlink(path)
    
        # Generate train/val/test splits identical to the official release
        if files_exist(self.split_paths):
            return

        dataset = pd.read_csv(self.raw_paths[1])
        n_samples = len(dataset)
        n_train = 100000
        n_test = int(0.1 * n_samples)
        n_val = n_samples - (n_train + n_test)
        train, val, test = np.split(dataset.sample(frac=1, random_state=42), [n_train, n_val + n_train])
        train.to_csv(os.path.join(self.raw_dir, 'train.csv'))
        val.to_csv(os.path.join(self.raw_dir, 'val.csv'))
        test.to_csv(os.path.join(self.raw_dir, 'test.csv'))
        
    def build_ring_graphs(self):
        ring_graphs = []
        for (r_smi, r_label) in self.ring_types:
            g = self.smiles_to_nx(r_smi, as_ring=True)
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
        
        if as_ring:
            pass
        
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
        return n1.get("symbol","?") == n2.get("symbol","?")

    def edge_eq(self, e1, e2):
        return e1.get("bond_label",0) == e2.get("bond_label",0)


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
            else:
                pass
        
        for out_n, b_l in outside_map.items():
            main_g.add_edge(new_super_idx, out_n, bond_label=b_l)


    def get_new_node_id(self, G):
    
        if len(G.nodes)==0:
            return 0
        else:
            return max(G.nodes)+1



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

    def graph_to_pyg_data(self, G, idx):
        
        node_label_idx = []
        
        sorted_nodes = sorted(G.nodes())
        node_map = {} 
        
        for i, n in enumerate(sorted_nodes):
            node_map[n] = i
        
        for n in sorted_nodes:
            sym = G.nodes[n].get("symbol","?")
            if sym.startswith("RING_"):
                
                ring_lbl_str = sym.split("_")[1] 
                ring_lbl = int(ring_lbl_str)
                node_label_idx.append(ring_lbl)  
            else:
               
                node_label_idx.append(self.atom_types.get(sym, 0))  
        
        edges = []
        for (u,v) in G.edges():
       
            nu = node_map[u]
            nv = node_map[v]
            b_label = G[u][v].get("bond_label",0)
        
            edges.append((nu,nv,b_label))
            edges.append((nv,nu,b_label))
        

        row, col, e_label = [],[],[]
        for (uu,vv,bb) in edges:
            row.append(uu); col.append(vv); e_label.append(bb)
        
        edge_index = torch.tensor([row,col], dtype=torch.long)
        edge_label = torch.tensor(e_label, dtype=torch.long)
        edge_attr = F.one_hot(edge_label, num_classes=5).float()  
        
        node_label_idx = torch.tensor(node_label_idx).long()  
        x = F.one_hot(node_label_idx, num_classes=8).float()  
        
        to_keep = (node_label_idx>0)
        edge_index, edge_attr = subgraph(to_keep, edge_index, edge_attr, relabel_nodes=True,
                                        num_nodes=node_label_idx.size(0))
        x = x[to_keep]
        
        x = x[:,1:]  
        
        y = torch.zeros((1,0), dtype=torch.float)
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, idx=torch.tensor([idx]))
        return data

    def process_single_smiles(self, smi, idx, ring_graphs):
       
        rd_mol = Chem.MolFromSmiles(smi)
        if rd_mol is None:
            print(f"Warning: failed to parse SMILES: {smi}")
            return None
        
        
        G = self.smiles_to_nx(smi)
        if G is None or len(G)==0:
            return None
        
      
        G = self.contract_rings_in_order(G, ring_graphs)
        
 
        data = self.graph_to_pyg_data(G, idx)
        return data

    def process_smiles_list(self, smiles_list):
        
        ring_graphs = self.build_ring_graphs()  
        data_list = []
        for i, smi in enumerate(smiles_list):
            d = self.process_single_smiles(smi, i, ring_graphs)
            if d is not None:
                data_list.append(d)
        return data_list

    def process(self):
        RDLogger.DisableLog('rdApp.*')

        # Match the official type/bond mapping (remove_h enforced)
        types_remove_h = {'C': 0, 'N': 1, 'O': 2, 'F': 3}
        bonds = {BT.SINGLE: 0, BT.DOUBLE: 1, BT.TRIPLE: 2, BT.AROMATIC: 3}

        # Load split CSV and uncharacterized list
        target_df = pd.read_csv(self.split_paths[self.file_idx], index_col=0)
        # Drop mol_id if present
        if 'mol_id' in target_df.columns:
            target_df.drop(columns=['mol_id'], inplace=True)
        with open(self.raw_paths[-1], 'r') as f:
            skip = [int(x.split()[0]) - 1 for x in f.read().split('\n')[9:-2]]

        # Prebuild ring templates
        ring_graphs = self.build_ring_graphs()

        # Map each RING_<label> to a new atom index (taking the remove_h shift into account)
        # Before remove_h the indices start at 5; afterwards they must start at 4
        base_type_offset = len(self.atom_types)  # 5 (H, C, N, O, F)
        supernode_label_to_type = {}
        for i, (_, r_label) in enumerate(self.ring_types):
            supernode_label_to_type[r_label] = base_type_offset + i

        # RDKit supplier
        suppl = Chem.SDMolSupplier(self.raw_paths[0], removeHs=False, sanitize=False)

        data_list = []
        for i, mol in enumerate(tqdm(suppl)):
            if i in skip or i not in target_df.index:
                continue
            if mol is None:
                continue

            # Save original SMILES before compression
            original_smiles = mol2smiles(mol)
            if original_smiles is None:
                continue

            # Build a NetworkX graph (with hydrogens) from the RDKit molecule
            G = nx.Graph()
            for a in mol.GetAtoms():
                idx = a.GetIdx()
                sym = a.GetSymbol()
                G.add_node(idx, symbol=sym)
            for b in mol.GetBonds():
                s = b.GetBeginAtomIdx(); e = b.GetEndAtomIdx()
                btype = b.GetBondType()
                G.add_edge(s, e, bond_label=(bonds.get(btype, 0) + 1))  # +1 to match official format

            # Apply ring compression
            G = self.contract_rings_in_order(G, ring_graphs)

            # Convert the compressed graph to PyG tensors (matching the official encoding)
            # Node type indices
            node_type_idx = []
            sorted_nodes = sorted(G.nodes())
            node_map = {n: idx for idx, n in enumerate(sorted_nodes)}
            for n in sorted_nodes:
                sym = G.nodes[n].get('symbol', '?')
                if sym.startswith('RING_'):
                    r_lbl = int(sym.split('_')[1])
                    node_type_idx.append(supernode_label_to_type[r_lbl])
                else:
                    # Use the full atom map (including hydrogen)
                    mapped = self.atom_types.get(sym, 0)
                    node_type_idx.append(mapped)

            # Bidirectional edges with +1 encoding
            row, col, edge_type = [], [], []
            for (u, v) in G.edges():
                et = int(G[u][v].get('bond_label', 0))
                row += [node_map[u], node_map[v]]
                col += [node_map[v], node_map[u]]
                edge_type += [et, et]

            edge_index = torch.tensor([row, col], dtype=torch.long)
            edge_type = torch.tensor(edge_type, dtype=torch.long)
            edge_attr = F.one_hot(edge_type, num_classes=len(bonds) + 1).to(torch.float)

            node_type_idx = torch.tensor(node_type_idx).long()

            # remove_h: drop hydrogens (index 0) and shift the remaining types down by one
            to_keep = node_type_idx > 0  # Keep C, N, O, F and supernodes
            edge_index, edge_attr = subgraph(to_keep, edge_index, edge_attr, relabel_nodes=True,
                                             num_nodes=len(to_keep))

            node_type_idx_filtered = node_type_idx[to_keep] - 1  # Shift after removing hydrogen
            x = F.one_hot(
                node_type_idx_filtered,
                num_classes=len(types_remove_h) + len(self.ring_types)
            ).float()

            # Sort edges the same way as the official implementation
            N = x.size(0)
            if edge_index.numel() > 0:
                perm = (edge_index[0] * N + edge_index[1]).argsort()
                edge_index = edge_index[:, perm]
                edge_attr = edge_attr[perm]

            y = torch.zeros((1, 0), dtype=torch.float)
            data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, idx=i)
            # Store the original SMILES (before compression)
            data.smiles = original_smiles
            # Compute and store molecular mass: base atoms + supernodes
            try:
                base_weights = [12, 14, 16, 19]
                ring_weights = list(self.ring_weights) if hasattr(self, 'ring_weights') else [1] * len(self.ring_types)
                weights = torch.tensor(base_weights + ring_weights, dtype=torch.float)
                type_counts = node_type_idx_filtered.bincount(minlength=len(base_weights) + len(self.ring_types)).to(torch.float)
                total_mass = float((type_counts * weights).sum().item())
                data.mol_weight = torch.tensor([total_mass], dtype=torch.float)
            except Exception:
                pass
            data_list.append(data)

        torch.save(self.collate(data_list), self.processed_paths[self.file_idx])



class QM9DataModule(MolecularDataModule):
    def __init__(self, cfg):
        self.datadir = cfg.dataset.datadir
        
        # Always enforce remove_h=True
        self.remove_h = True
        
        # Read ring configuration from the config
        self.ring_types = getattr(cfg.dataset, 'ring_types', ['C1CCC1', 'C1CC1', 'C1CNC1'])
        self.ring_weights = getattr(cfg.dataset, 'ring_weights', [48, 36, 50])

        target = getattr(cfg.general, 'guidance_target', None)
        regressor = getattr(self, 'regressor', None)
        if regressor and target == 'mu':
            transform = SelectMuTransform()
        elif regressor and target == 'homo':
            transform = SelectHOMOTransform()
        elif regressor and target == 'both':
            transform = None
        else:
            transform = RemoveYTransform()

        base_path = pathlib.Path(os.path.realpath(__file__)).parents[2]
        root_path = os.path.join(base_path, self.datadir)
        datasets = {'train': QM9Dataset(stage='train', root=root_path, remove_h=True,
                                        target_prop=target, transform=RemoveYTransform(),
                                        ring_types=self.ring_types, ring_weights=self.ring_weights),
                    'val': QM9Dataset(stage='val', root=root_path, remove_h=True,
                                      target_prop=target, transform=RemoveYTransform(),
                                      ring_types=self.ring_types, ring_weights=self.ring_weights),
                    'test': QM9Dataset(stage='test', root=root_path, remove_h=True,
                                       target_prop=target, transform=transform,
                                       ring_types=self.ring_types, ring_weights=self.ring_weights)}
        # Expose datasets so QM9infos can reuse them for statistics
        self.datasets = datasets
        super().__init__(cfg, datasets)


class QM9infos(AbstractDatasetInfos):
    def __init__(self, datamodule, cfg):
        # Always enforce remove_h=True
        self.remove_h = True
        self.need_to_strip = False        # to indicate whether we need to ignore one output from the model

        self.name = 'qm9'
        
        # Read ring configuration from the config
        self.ring_types_list = getattr(cfg.dataset, 'ring_types', ['C1CCC1', 'C1CC1', 'C1CNC1'])
        self.ring_weights_list = getattr(cfg.dataset, 'ring_weights', [48, 36, 50])
        
        # Ensure matching lengths
        assert len(self.ring_types_list) == len(self.ring_weights_list), "ring_types and ring_weights must have identical lengths"
        
        # Base atom types (without hydrogen)
            self.atom_encoder = {'C': 0, 'N': 1, 'O': 2, 'F': 3}
        base_atom_decoder = ['C', 'N', 'O', 'F']
        base_valencies = [4, 3, 2, 1]
        base_atom_weights = {0: 12, 1: 14, 2: 16, 3: 19}
        base_num_atom_types = 4
        
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
        base_label = 4  # After removing H, atoms occupy 0..3 so rings start at 4
        for i, ring_smiles in enumerate(self.ring_types_list):
            self.label_to_ring[base_label + i] = ring_smiles
        
        # Fixed symbol/bond mappings
                self.label_to_symbol = {0: "C", 1: "N", 2: "O", 3: "F"}
                self.label_to_bondtype = {1: rdchem.BondType.SINGLE, 2: rdchem.BondType.DOUBLE, 3: rdchem.BondType.TRIPLE, 4: rdchem.BondType.AROMATIC}

        # Path to cached statistics
        self.statistics_file = getattr(cfg.dataset, 'statistics_after_4', 'data/qm9/qm9_pyg/statistics_after_compression.json')
        
        if not self._load_statistics():
            print("Statistics file missing or incomplete, recomputing...")
            self._compute_and_save_statistics(datamodule)

        # Recompute valency_distribution if it is still missing after loading
        if not hasattr(self, 'valency_distribution') or self.valency_distribution is None:
            print("Computing valency_distribution...")
            self.valency_distribution = datamodule.valency_count(self.max_n_nodes)
            # Update the statistics file
            try:
                import json
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
        import json
        
        if not os.path.exists(self.statistics_file):
            return False
        
        try:
            with open(self.statistics_file, 'r') as f:
                stats = json.load(f)
            
            # Ensure all required keys are present
            required_keys = ['max_n_nodes', 'max_weight', 'n_nodes', 'node_types', 'edge_types']
            if not all(key in stats for key in required_keys):
                print(f"Statistics file missing required keys: {required_keys}")
                return False
            
            # Ensure the statistics match the current configuration
            expected_num_node_types = 4 + len(self.ring_types_list)  # base atoms + supernodes
            if len(stats['node_types']) != expected_num_node_types:
                print(f"Node type count mismatch: expected {expected_num_node_types}, got {len(stats['node_types'])}")
                return False
            
            # Load statistics into memory
            self.max_n_nodes = stats['max_n_nodes']
            self.max_weight = stats['max_weight']
            self.n_nodes = torch.tensor(stats['n_nodes'])
            self.node_types = torch.tensor(stats['node_types'])
            self.edge_types = torch.tensor(stats['edge_types'])
            
            # Load valency_distribution if present, otherwise compute later
            if 'valency_distribution' in stats:
                self.valency_distribution = torch.tensor(stats['valency_distribution'])
            else:
                # Mark as missing so it can be recomputed
                self.valency_distribution = None
            
            print(f"Loaded statistics from {self.statistics_file}")
            return True
            
        except Exception as e:
            print(f"Failed to load statistics: {e}")
            return False
    
    def _compute_and_save_statistics(self, datamodule):
        """Compute and save dataset statistics."""
        import json
        
        print("Computing statistics for the ring-compressed dataset...")
        
        # Aggregate statistics
        self.n_nodes = datamodule.node_counts()
        self.node_types = datamodule.node_types()
        self.edge_types = datamodule.edge_counts()
            
        # Determine maximum node count and molecular weight
        self.max_n_nodes = len(self.n_nodes) - 1
        
        # Compute valency_distribution if necessary
        if not hasattr(self, 'valency_distribution') or self.valency_distribution is None:
            self.valency_distribution = datamodule.valency_count(self.max_n_nodes)
            
        # Observe the maximum molecular weight; fall back to an upper bound if necessary
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
                    except Exception:
                        continue
                    # Prefer previously computed mol_weight
                    if hasattr(d, 'mol_weight') and d.mol_weight is not None:
                        total_mass = float(d.mol_weight.item())
                    else:
                        # Fallback: derive weight from x
                        type_counts = d.x.sum(dim=0)
                        total_mass = 0.0
                        for idx_w, cnt in enumerate(type_counts.tolist()):
                            if cnt > 0:
                                w = self.atom_weights.get(idx_w, 0)
                                total_mass += w * cnt
                    if total_mass > observed_max_weight:
                        observed_max_weight = total_mass

        if observed_max_weight > 0:
            self.max_weight = observed_max_weight
        else:
            # Fallback: loose upper bound
            self.max_weight = max(self.atom_weights.values()) * self.max_n_nodes
        
        # Persist statistics
        # Convert ListConfig to plain lists for JSON serialization
        try:
            ring_types_list = OmegaConf.to_container(self.ring_types_list, resolve=True)
        except (TypeError, AttributeError):
            ring_types_list = list(self.ring_types_list) if hasattr(self.ring_types_list, '__iter__') else self.ring_types_list
        
        try:
            ring_weights_list = OmegaConf.to_container(self.ring_weights_list, resolve=True)
        except (TypeError, AttributeError):
            ring_weights_list = list(self.ring_weights_list) if hasattr(self.ring_weights_list, '__iter__') else self.ring_weights_list
        
        stats = {
            'max_n_nodes': self.max_n_nodes,
            'max_weight': self.max_weight,
            'n_nodes': self.n_nodes.tolist(),
            'node_types': self.node_types.tolist(),
            'edge_types': self.edge_types.tolist(),
            'valency_distribution': self.valency_distribution.tolist(),
            'ring_types_list': ring_types_list,
            'ring_weights_list': ring_weights_list,
            'atom_decoder': self.atom_decoder,
            'label_to_ring': self.label_to_ring
        }
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(self.statistics_file), exist_ok=True)
        
        with open(self.statistics_file, 'w') as f:
            json.dump(stats, f, indent=2)
        
        print(f"Saved statistics to {self.statistics_file}")
        print(f"Node-count distribution: {self.n_nodes}")
        print(f"Node-type distribution: {self.node_types}")
        print(f"Edge-type distribution: {self.edge_types}")


def get_train_smiles(cfg, train_dataloader, dataset_infos, evaluate_dataset=False):
    if evaluate_dataset:
        assert dataset_infos is not None, "If wanting to evaluate dataset, need to pass dataset_infos"
    datadir = cfg.dataset.datadir
    remove_h = cfg.dataset.remove_h
    atom_decoder = dataset_infos.atom_decoder
    root_dir = pathlib.Path(os.path.realpath(__file__)).parents[2]
    smiles_file_name = 'train_smiles_no_h.npy' if remove_h else 'train_smiles_h.npy'
    smiles_path = os.path.join(root_dir, datadir, smiles_file_name)
    if os.path.exists(smiles_path):
        print("Dataset smiles were found.")
        train_smiles = np.load(smiles_path)
    else:
        print("Computing dataset smiles...")
        train_smiles = compute_qm9_smiles(atom_decoder, train_dataloader, remove_h)
        np.save(smiles_path, np.array(train_smiles))

    if evaluate_dataset:
        train_dataloader = train_dataloader
        all_molecules = []
        for i, data in enumerate(train_dataloader):
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


def get_val_smiles(cfg, val_dataloader, dataset_infos, evaluate_dataset=False):
    if evaluate_dataset:
        assert dataset_infos is not None, "If wanting to evaluate dataset, need to pass dataset_infos"
    datadir = cfg.dataset.datadir
    remove_h = cfg.dataset.remove_h
    atom_decoder = dataset_infos.atom_decoder
    root_dir = pathlib.Path(os.path.realpath(__file__)).parents[2]
    smiles_file_name = 'val_smiles_no_h.npy' if remove_h else 'val_smiles_h.npy'
    smiles_path = os.path.join(root_dir, datadir, smiles_file_name)
    if os.path.exists(smiles_path):
        print("Dataset smiles were found.")
        val_smiles = np.load(smiles_path)
    else:
        print("Computing dataset smiles...")
        val_smiles = compute_qm9_smiles(atom_decoder, val_dataloader, remove_h)
        np.save(smiles_path, np.array(val_smiles))

    if evaluate_dataset:
        val_dataloader = val_dataloader
        all_molecules = []
        for i, data in enumerate(val_dataloader):
            dense_data, node_mask = utils.to_dense(data.x, data.edge_index, data.edge_attr, data.batch)
            dense_data = dense_data.mask(node_mask, collapse=True)
            X, E = dense_data.X, dense_data.E

            for k in range(X.size(0)):
                n = int(torch.sum((X != -1)[k, :]))
                atom_types = X[k, :n].cpu()
                edge_types = E[k, :n, :n].cpu()
                all_molecules.append([atom_types, edge_types])

        print("Evaluating the dataset -- number of molecules to evaluate", len(all_molecules))
        metrics = compute_molecular_metrics(molecule_list=all_molecules, val_smiles=val_smiles,
                                            dataset_info=dataset_infos)
        print(metrics[0])

    return val_smiles

def get_test_smiles(cfg, test_dataloader, dataset_infos, evaluate_dataset=False):
    if evaluate_dataset:
        assert dataset_infos is not None, "If wanting to evaluate dataset, need to pass dataset_infos"
    datadir = cfg.dataset.datadir
    remove_h = cfg.dataset.remove_h
    atom_decoder = dataset_infos.atom_decoder
    root_dir = pathlib.Path(os.path.realpath(__file__)).parents[2]
    smiles_file_name = 'test_smiles_no_h.npy' if remove_h else 'test_smiles_h.npy'
    smiles_path = os.path.join(root_dir, datadir, smiles_file_name)
    if os.path.exists(smiles_path):
        print("Dataset smiles were found.")
        test_smiles = np.load(smiles_path)
    else:
        print("Computing dataset smiles...")
        test_smiles = compute_qm9_smiles(atom_decoder, test_dataloader, remove_h)
        np.save(smiles_path, np.array(test_smiles))

    if evaluate_dataset:
        test_dataloader = test_dataloader
        all_molecules = []
        for i, data in enumerate(test_dataloader):
            dense_data, node_mask = utils.to_dense(data.x, data.edge_index, data.edge_attr, data.batch)
            dense_data = dense_data.mask(node_mask, collapse=True)
            X, E = dense_data.X, dense_data.E

            for k in range(X.size(0)):
                n = int(torch.sum((X != -1)[k, :]))
                atom_types = X[k, :n].cpu()
                edge_types = E[k, :n, :n].cpu()
                all_molecules.append([atom_types, edge_types])

        print("Evaluating the dataset -- number of molecules to evaluate", len(all_molecules))
        metrics = compute_molecular_metrics(molecule_list=all_molecules, test_smiles=test_smiles,
                                            dataset_info=dataset_infos)
        print(metrics[0])

    return test_smiles


def compute_qm9_smiles(atom_decoder, train_dataloader, remove_h):
    '''

    :param dataset_name: qm9 or qm9_second_half
    :return:
    '''
    print(f"\tConverting QM9 dataset to SMILES for remove_h={remove_h}...")

    # Read directly from the dataset instead of the dataloader to avoid batching issues
    dataset = train_dataloader.dataset
    mols_smiles = []
    len_train = len(dataset)
    invalid = 0
    
    for i in range(len_train):
        data = dataset[i]
        # Pull the original SMILES stored on each data instance (captured before compression)
        if hasattr(data, 'smiles') and data.smiles is not None:
            if isinstance(data.smiles, str):
                mols_smiles.append(data.smiles)
            else:
                invalid += 1
        else:
            # If a sample is missing SMILES, fall back to decompression (should not happen)
            print(f"Warning: data at index {i} does not have smiles attribute, falling back to decompression")
            # Decompression would be required here; skip the sample for now
                invalid += 1

        if i % 1000 == 0:
            print("\tConverting QM9 dataset to SMILES {0:.2%}".format(float(i) / len_train))
    print("Number of invalid molecules", invalid)
    return mols_smiles

