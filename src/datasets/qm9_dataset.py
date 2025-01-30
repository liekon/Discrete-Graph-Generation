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

import src.utils as utils
from src.datasets.abstract_dataset import MolecularDataModule, AbstractDatasetInfos
from src.analysis.rdkit_functions import mol2smiles, build_molecule_with_partial_charges
from src.analysis.rdkit_functions import compute_molecular_metrics

from rdkit.Chem import Descriptors, rdchem, AllChem, rdmolops
import pickle
import matplotlib.pyplot as plt
from rdkit.Chem import BondType as BT

import networkx as nx
import random


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

    def __init__(self, stage, root, remove_h: bool, target_prop=None,
                 transform=None, pre_transform=None, pre_filter=None):
        self.target_prop = target_prop
        self.stage = stage
        if self.stage == 'train':
            self.file_idx = 0
        elif self.stage == 'val':
            self.file_idx = 1
        else:
            self.file_idx = 2
        self.remove_h = remove_h
        self.ring_dict = {}

        self.atom_types = {"H": 0, "C": 1, "N": 2, "O": 3, "F": 4}

        self.bond_types = {BT.SINGLE:1, BT.DOUBLE:2, BT.TRIPLE:3, BT.AROMATIC:4}

        self.ring_types = [
            ("C1CCC1", 5),
            ("C1CC1", 6),
            ("C1CNC1", 7)
        ]

        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[self.file_idx])


    @property
    def raw_file_names(self):
        return ['qm9_v1_train.smiles', 'qm9_v1_valid.smiles', 'qm9_v1_test.smiles']

    @property
    def split_file_name(self):
        return ['qm9_v1_train.smiles', 'qm9_v1_valid.smiles', 'qm9_v1_test.smiles']

    @property
    def split_paths(self):
        r"""The absolute filepaths that must be present in order to skip
        splitting."""
        files = to_list(self.split_file_name)
        return [osp.join(self.raw_dir, f) for f in files]

    @property
    def processed_file_names(self):
        if self.remove_h:
            return ['proc_tr_no_h.pt', 'proc_val_no_h.pt', 'proc_test_no_h.pt']
        else:
            return ['proc_tr_h.pt', 'proc_val_h.pt', 'proc_test_h.pt']

    def download(self):
        """
        Download raw qm9 files. Taken from PyG QM9 class
        """
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
    
        if files_exist(self.split_paths):
            return
        
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
            print(f"Warning: 无法解析SMILES: {smi}")
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
        smile_list = open(self.split_paths[self.file_idx]).readlines()
        data_list = self.process_smiles_list(smile_list)

        torch.save(self.collate(data_list), self.processed_paths[self.file_idx])


class QM9DataModule(MolecularDataModule):
    def __init__(self, cfg):
        self.datadir = cfg.dataset.datadir
        self.remove_h = cfg.dataset.remove_h

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
        datasets = {'train': QM9Dataset(stage='train', root=root_path, remove_h=cfg.dataset.remove_h,
                                        target_prop=target, transform=RemoveYTransform()),
                    'val': QM9Dataset(stage='val', root=root_path, remove_h=cfg.dataset.remove_h,
                                      target_prop=target, transform=RemoveYTransform()),
                    'test': QM9Dataset(stage='test', root=root_path, remove_h=cfg.dataset.remove_h,
                                       target_prop=target, transform=transform)}
        super().__init__(cfg, datasets)


""" class QM9infos(AbstractDatasetInfos):
    def __init__(self, datamodule, cfg, recompute_statistics=False):
        self.remove_h = cfg.dataset.remove_h
        self.need_to_strip = False        # to indicate whether we need to ignore one output from the model

        self.name = 'qm9'
        if self.remove_h:
            self.atom_encoder = {'C': 0, 'N': 1, 'O': 2, 'F': 3}
            self.atom_decoder = ['C', 'N', 'O', 'F']
            self.num_atom_types = 4
            self.valencies = [4, 3, 2, 1]
            self.atom_weights = {0: 12, 1: 14, 2: 16, 3: 19}
            self.max_n_nodes = 9
            self.max_weight = 150
            self.n_nodes = torch.tensor([0, 2.2930e-05, 3.8217e-05, 6.8791e-05, 2.3695e-04, 9.7072e-04,
                                        0.0046472, 0.023985, 0.13666, 0.83337])
            self.node_types = torch.tensor([0.7230, 0.1151, 0.1593, 0.0026])
            self.edge_types = torch.tensor([0.7261, 0.2384, 0.0274, 0.0081, 0.0])

            self.edge_consume = torch.tensor([0, 1, 2, 3, 2])

            super().complete_infos(n_nodes=self.n_nodes, node_types=self.node_types)
            self.valency_distribution = torch.zeros(3 * self.max_n_nodes - 2)
            self.valency_distribution[0: 6] = torch.tensor([2.6071e-06, 0.163, 0.352, 0.320, 0.16313, 0.00073])
        else:
            self.atom_encoder = {'H': 0, 'C': 1, 'N': 2, 'O': 3, 'F': 4}
            self.atom_decoder = ['H', 'C', 'N', 'O', 'F']
            self.valencies = [1, 4, 3, 2, 1]
            self.num_atom_types = 5
            self.max_n_nodes = 29
            self.max_weight = 390
            self.atom_weights = {0: 1, 1: 12, 2: 14, 3: 16, 4: 19}
            self.n_nodes = torch.tensor([0, 0, 0, 1.5287e-05, 3.0574e-05, 3.8217e-05,
                                         9.1721e-05, 1.5287e-04, 4.9682e-04, 1.3147e-03, 3.6918e-03, 8.0486e-03,
                                         1.6732e-02, 3.0780e-02, 5.1654e-02, 7.8085e-02, 1.0566e-01, 1.2970e-01,
                                         1.3332e-01, 1.3870e-01, 9.4802e-02, 1.0063e-01, 3.3845e-02, 4.8628e-02,
                                         5.4421e-03, 1.4698e-02, 4.5096e-04, 2.7211e-03, 0.0000e+00, 2.6752e-04])

            self.node_types = torch.tensor([0.5122, 0.3526, 0.0562, 0.0777, 0.0013])
            self.edge_types = torch.tensor([0.88162,  0.11062,  5.9875e-03,  1.7758e-03, 0])

            super().complete_infos(n_nodes=self.n_nodes, node_types=self.node_types)
            self.valency_distribution = torch.zeros(3 * self.max_n_nodes - 2)
            self.valency_distribution[0:6] = torch.tensor([0, 0.5136, 0.0840, 0.0554, 0.3456, 0.0012])

        if recompute_statistics:
            np.set_printoptions(suppress=True, precision=5)
            self.n_nodes = datamodule.node_counts()
            print("Distribution of number of nodes", self.n_nodes)
            np.savetxt('n_counts.txt', self.n_nodes.numpy())
            self.node_types = datamodule.node_types()                                     # There are no node types
            print("Distribution of node types", self.node_types)
            np.savetxt('atom_types.txt', self.node_types.numpy())

            self.edge_types = datamodule.edge_counts()
            print("Distribution of edge types", self.edge_types)
            np.savetxt('edge_types.txt', self.edge_types.numpy())

            valencies = datamodule.valency_count(self.max_n_nodes)
            print("Distribution of the valencies", valencies)
            np.savetxt('valencies.txt', valencies.numpy())
            self.valency_distribution = valencies
            assert False """

class QM9infos(AbstractDatasetInfos):
    def __init__(self, datamodule, cfg, recompute_statistics=False):
        self.remove_h = cfg.dataset.remove_h
        self.need_to_strip = False        # to indicate whether we need to ignore one output from the model

        self.name = 'qm9'
        if self.remove_h:
            self.atom_encoder = {'C': 0, 'N': 1, 'O': 2, 'F': 3}
            self.atom_decoder = ['C', 'N', 'O', 'F']
            self.num_atom_types = 4
            self.valencies = [4, 3, 2, 1, 1, 1, 1] 
            self.atom_weights = {0: 12, 1: 14, 2: 16, 3: 19, 4: 48, 5: 36, 6: 50}
            self.max_n_nodes = 9
            self.max_weight = 150

            self.n_nodes = torch.tensor([0.0, 3.085435714946879e-05, 0.0002776892143452191, 0.0038362250722506195, 0.015468317717600355, 0.04283613250917917, 0.25470271826886487, 0.15281134617560244, 0.08154806594604601, 0.4484886507389619]) 
            self.node_types = torch.tensor([0.6247562823485964, 0.1255464977977157, 0.1846184970156468, 0.003079396286258941, 0.028743747522090732, 0.024717993558793658, 0.008537585470897804])
            self.edge_types = torch.tensor([0.4185789123595221, 0.44687886918882214, 0.046334704692231686, 0.021738880972721735, 0.06646863278670231]) 

            self.ring_types = {"C1CCC1", "C1CC1", "C1CNC1"}
            self.label_to_ring = {4:  "C1CCC1", 5:  "C1CC1", 6:  "C1CNC1"} 
            self.label_to_symbol = {0: "C", 1: "N", 2: "O", 3: "F"}
            self.label_to_bondtype = {1: rdchem.BondType.SINGLE, 2: rdchem.BondType.DOUBLE, 3: rdchem.BondType.TRIPLE, 4: rdchem.BondType.AROMATIC}

            self.norm_node_type = torch.tensor([0.650019170321977, 0.13862422959796372, 0.20752591371497195, 0.0038306863650873816, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
            self.super_node_type = torch.tensor([0, 0, 0, 0, 0.29178669386517436, 0.2509200170909542, 0.08666767742202253, 0.13330944275219495, 0.07768114343996801, 0.02166691935550563, 0.07364271635907543, 0.029302716634735985, 0.02509889322288534, 0.009923779857483495])


            super().complete_infos(n_nodes=self.n_nodes, node_types=self.node_types)
            self.valency_distribution = torch.zeros(3 * self.max_n_nodes - 2)
            self.valency_distribution[0: 6] = torch.tensor([2.6071e-06, 0.163, 0.352, 0.320, 0.16313, 0.00073])

        if recompute_statistics:
            np.set_printoptions(suppress=True, precision=5)
            self.n_nodes = datamodule.node_counts()
            print("Distribution of number of nodes", self.n_nodes)
            np.savetxt('n_counts.txt', self.n_nodes.numpy())
            self.node_types = datamodule.node_types()                                     # There are no node types
            print("Distribution of node types", self.node_types)
            np.savetxt('atom_types.txt', self.node_types.numpy())

            self.edge_types = datamodule.edge_counts()
            print("Distribution of edge types", self.edge_types)
            np.savetxt('edge_types.txt', self.edge_types.numpy())

            valencies = datamodule.valency_count(self.max_n_nodes)
            print("Distribution of the valencies", valencies)
            np.savetxt('valencies.txt', valencies.numpy())
            self.valency_distribution = valencies
            assert False


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

    mols_smiles = []
    len_train = len(train_dataloader)
    invalid = 0
    disconnected = 0
    for i, data in enumerate(train_dataloader):
        dense_data, node_mask = utils.to_dense(data.x, data.edge_index, data.edge_attr, data.batch)
        dense_data = dense_data.mask(node_mask, collapse=True)
        X, E = dense_data.X, dense_data.E

        n_nodes = [int(torch.sum((X != -1)[j, :])) for j in range(X.size(0))]

        molecule_list = []
        for k in range(X.size(0)):
            n = n_nodes[k]
            atom_types = X[k, :n].cpu()
            edge_types = E[k, :n, :n].cpu()
            molecule_list.append([atom_types, edge_types])

        for l, molecule in enumerate(molecule_list):
            mol = build_molecule_with_partial_charges(molecule[0], molecule[1], atom_decoder)
            smile = mol2smiles(mol)
            if smile is not None:
                mols_smiles.append(smile)
                mol_frags = Chem.rdmolops.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
                if len(mol_frags) > 1:
                    print("Disconnected molecule", mol, mol_frags)
                    disconnected += 1
            else:
                print("Invalid molecule obtained.")
                invalid += 1

        if i % 1000 == 0:
            print("\tConverting QM9 dataset to SMILES {0:.2%}".format(float(i) / len_train))
    print("Number of invalid molecules", invalid)
    print("Number of disconnected molecules", disconnected)
    return mols_smiles

