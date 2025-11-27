import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import time
import wandb
import os
import math
from models.transformer_model import GraphTransformer
from diffusion.noise_schedule import DiscreteUniformTransition, PredefinedNoiseScheduleDiscrete,\
    MarginalUniformTransition
from diffusion import diffusion_utils
from metrics.train_metrics import TrainLossDiscrete
from metrics.val_metrics import ValLossDiscrete
from metrics.test_metrics import TestLossDiscrete
from metrics.abstract_metrics import SumExceptBatchMetric, SumExceptBatchKL, NLL
import utils
import networkx as nx
import random
from rdkit.Chem import rdchem

class DiscreteDenoisingDiffusion(pl.LightningModule):
    def __init__(self, cfg, dataset_name, dataset_infos, train_metrics, sampling_metrics, visualization_tools, extra_features,
                 domain_features):
        super().__init__()

        input_dims = dataset_infos.input_dims
        output_dims = dataset_infos.output_dims
        nodes_dist = dataset_infos.nodes_dist

        input_dims['y'] += 1 

        self.cfg = cfg
        self.dataset_name = dataset_name
        self.name = cfg.general.name
        self.model_dtype = torch.float32
        self.T = cfg.model.diffusion_steps
        self.max_n_nodes = dataset_infos.max_n_nodes
        self.Xdim = input_dims['X']
        self.Edim = input_dims['E']
        self.ydim = input_dims['y']
        self.Xdim_output = output_dims['X']
        self.Edim_output = output_dims['E']
        self.ydim_output = output_dims['y']
        self.node_dist = nodes_dist

        self.dataset_info = dataset_infos

        self.train_loss = TrainLossDiscrete(self.cfg.model.lambda_train)
        self.val_loss = ValLossDiscrete(self.cfg.model.lambda_train)
        self.test_loss = TestLossDiscrete(self.cfg.model.lambda_train)

        self.train_metrics = train_metrics
        self.sampling_metrics = sampling_metrics

        self.visualization_tools = visualization_tools
        self.extra_features = extra_features
        self.domain_features = domain_features
        
        self.model = GraphTransformer(n_layers=cfg.model.n_layers,
                                      input_dims=input_dims,
                                      hidden_mlp_dims=cfg.model.hidden_mlp_dims,
                                      hidden_dims=cfg.model.hidden_dims,
                                      output_dims=output_dims,
                                      act_fn_in=nn.ReLU(),
                                      act_fn_out=nn.ReLU())

        self.noise_schedule = PredefinedNoiseScheduleDiscrete(cfg.model.diffusion_noise_schedule,
                                                              timesteps=cfg.model.diffusion_steps)

        if cfg.model.transition == 'uniform':
            self.transition_model = DiscreteUniformTransition(x_classes=self.Xdim_output, e_classes=self.Edim_output,
                                                              y_classes=self.ydim_output)
            x_limit = torch.ones(self.Xdim_output) / self.Xdim_output
            e_limit = torch.ones(self.Edim_output) / self.Edim_output
            y_limit = torch.ones(self.ydim_output) / self.ydim_output
            self.limit_dist = utils.PlaceHolder(X=x_limit, E=e_limit, y=y_limit)
        elif cfg.model.transition == 'marginal':

            node_types = self.dataset_info.node_types.float()
            x_marginals = node_types / torch.sum(node_types)

            edge_types = self.dataset_info.edge_types.float()
            e_marginals = edge_types / torch.sum(edge_types)
            print(f"Marginal distribution of the classes: {x_marginals} for nodes, {e_marginals} for edges")
            self.transition_model = MarginalUniformTransition(x_marginals=x_marginals, e_marginals=e_marginals,
                                                              y_classes=self.ydim_output)
            self.limit_dist = utils.PlaceHolder(X=x_marginals, E=e_marginals,
                                                y=torch.ones(self.ydim_output) / self.ydim_output)

        self.save_hyperparameters(ignore=['train_metrics', 'sampling_metrics'])
        self.start_epoch_time = None
        self.train_iterations = None
        self.val_iterations = None
        self.log_every_steps = cfg.general.log_every_steps
        self.number_chain_steps = cfg.general.number_chain_steps
        self.best_val_nll = 1e8
        self.best_val_validity = 0
        self.val_counter = 0

    def training_step(self, data, i):
        if data.edge_index.numel() == 0:
            self.print("Found a batch with no edges. Skipping.")
            return
        dense_data, node_mask = utils.to_dense(data.x, data.edge_index, data.edge_attr, data.batch)
        dense_data = dense_data.mask(node_mask)
        X, E = dense_data.X, dense_data.E
        noisy_data = self.apply_noise(X, E, data.y, node_mask)
        extra_data = self.compute_extra_data(noisy_data)
        pred = self.forward(noisy_data, extra_data, node_mask)
        loss = self.train_loss(masked_pred_X=pred.X, masked_pred_E=pred.E, pred_y=pred.y,
                               true_X=X, true_E=E, true_y=data.y, noisy_X_t=noisy_data["X_t"], noisy_E_t=noisy_data["E_t"], t=noisy_data["t_int"], 
                               t_e=noisy_data["t_e_int"], log=i % self.log_every_steps == 0)

        self.train_metrics(masked_pred_X=pred.X, masked_pred_E=pred.E, true_X=X, true_E=E,
                           log=i % self.log_every_steps == 0)

        return {'loss': loss}

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.cfg.train.lr, amsgrad=True,
                                 weight_decay=self.cfg.train.weight_decay)

    def on_fit_start(self) -> None:
        self.train_iterations = len(self.trainer.datamodule.train_dataloader())
        self.print("Size of the input features", self.Xdim, self.Edim, self.ydim)
        if self.local_rank == 0:
            utils.setup_wandb(self.cfg)

    def on_train_epoch_start(self) -> None:
        self.print("Starting train epoch...")
        self.start_epoch_time = time.time()
        self.train_loss.reset()
        self.train_metrics.reset()

    def on_train_epoch_end(self) -> None:
        to_log = self.train_loss.log_epoch_metrics()
        self.print(f"Epoch {self.current_epoch}: X_CE: {to_log['train_epoch/x_CE'] :.3f}"
                      f" -- E_CE: {to_log['train_epoch/E_CE'] :.3f} --"
                      f" y_CE: {to_log['train_epoch/y_CE'] :.3f}"
                      f" -- {time.time() - self.start_epoch_time:.1f}s ")
        epoch_at_metrics, epoch_bond_metrics = self.train_metrics.log_epoch_metrics()
        self.print(f"Epoch {self.current_epoch}: {epoch_at_metrics} -- {epoch_bond_metrics}")
        print(torch.cuda.memory_summary())

    def on_validation_epoch_start(self) -> None:
        self.val_loss.reset()
        self.start_val_epoch_time = time.time()
        self.sampling_metrics.reset()

    def validation_step(self, data, i):
        if data.edge_index.numel() == 0:
            self.print("Found a batch with no edges. Skipping.")
            return
        dense_data, node_mask = utils.to_dense(data.x, data.edge_index, data.edge_attr, data.batch)
        dense_data = dense_data.mask(node_mask)
        X, E = dense_data.X, dense_data.E
        noisy_data = self.apply_noise(X, E, data.y, node_mask)
        extra_data = self.compute_extra_data(noisy_data)
        pred = self.forward(noisy_data, extra_data, node_mask)
        loss = self.val_loss(masked_pred_X=pred.X, masked_pred_E=pred.E, pred_y=pred.y,
                               true_X=X, true_E=E, true_y=data.y, noisy_X_t=noisy_data["X_t"], noisy_E_t=noisy_data["E_t"], t=noisy_data["t_int"], 
                               t_e=noisy_data["t_e_int"], log=i % self.log_every_steps == 0)
        return {'loss': loss}

    def on_validation_epoch_end(self) -> None:
        to_log = self.val_loss.log_epoch_metrics()
        if wandb.run:
            wandb.log(to_log, commit=False)
        val_nll = to_log['val_epoch/x_CE'] + to_log['val_epoch/E_CE'] + to_log['val_epoch/y_CE']
        self.print(f"Epoch {self.current_epoch}: Val Loss {val_nll :.2f}")

        # Log val nll with default Lightning logger, so it can be monitored by checkpoint callback
        self.log("val/epoch_NLL", val_nll, sync_dist=True)

        if val_nll < self.best_val_nll:
            self.best_val_nll = val_nll
        self.print('Val loss: %.4f \t Best val loss:  %.4f\n' % (val_nll, self.best_val_nll))

        self.val_counter += 1
        if self.val_counter % self.cfg.general.sample_every_val == 0:
            start = time.time()
            samples_left_to_generate = self.cfg.general.samples_to_generate
            samples_left_to_save = self.cfg.general.samples_to_save
            chains_left_to_save = self.cfg.general.chains_to_save

            samples = []

            ident = 0
            while samples_left_to_generate > 0:
                bs = 2 * self.cfg.train.batch_size
                to_generate = min(samples_left_to_generate, bs)
                to_save = min(samples_left_to_save, bs)
                chains_save = min(chains_left_to_save, bs)
                samples.extend(self.sample_batch(batch_id=ident, batch_size=to_generate, num_nodes=None,
                                                 save_final=to_save,
                                                 keep_chain=chains_save,
                                                 number_chain_steps=self.number_chain_steps))
                ident += to_generate

                samples_left_to_save -= to_save
                samples_left_to_generate -= to_generate
                chains_left_to_save -= chains_save
            self.print("Computing sampling metrics...")
            validity = self.sampling_metrics.forward(samples, self.name, self.current_epoch, val_counter=-1, test=False,
                                          local_rank=self.local_rank)
            self.print(f'Done. Sampling took {time.time() - start:.2f} seconds\n')
            print("Validation epoch end ends...")
        else:
            validity = 0.0 

        if validity > self.best_val_validity:
            self.best_val_validity = validity
        self.print('Val validity: %.4f \t Best val validity:  %.4f\n' % (validity, self.best_val_validity))

        self.log("val/epoch_validity", validity, sync_dist=True)


    def on_test_epoch_start(self) -> None:
        self.print("Starting test...")
        self.start_test_epoch_time = time.time()
        if self.local_rank == 0:
            utils.setup_wandb(self.cfg)

    def test_step(self, data, i):
        if data.edge_index.numel() == 0:
            self.print("Found a batch with no edges. Skipping.")
            return
        dense_data, node_mask = utils.to_dense(data.x, data.edge_index, data.edge_attr, data.batch)
        dense_data = dense_data.mask(node_mask)
        X, E = dense_data.X, dense_data.E
        noisy_data = self.apply_noise(X, E, data.y, node_mask)
        extra_data = self.compute_extra_data(noisy_data)
        pred = self.forward(noisy_data, extra_data, node_mask)
        loss = self.test_loss(masked_pred_X=pred.X, masked_pred_E=pred.E, pred_y=pred.y,
                               true_X=X, true_E=E, true_y=data.y, noisy_X_t=noisy_data["X_t"], noisy_E_t=noisy_data["E_t"], t=noisy_data["t_int"], 
                               t_e=noisy_data["t_e_int"], log=i % self.log_every_steps == 0)
        return {'loss': loss}

    def on_test_epoch_end(self) -> None:
        """ Measure likelihood on a test set and compute stability metrics. """
        to_log = self.test_loss.log_epoch_metrics()
        if wandb.run:
            wandb.log(to_log, commit=False)
        test_nll = to_log['test_epoch/x_CE'] + to_log['test_epoch/E_CE'] + to_log['test_epoch/y_CE']
        self.print(f"Epoch {self.current_epoch}: Val Loss {test_nll :.2f}")

        if wandb.run:
            wandb.log({"test/epoch_NLL": test_nll}, commit=False)

        self.print(f'Test loss: {test_nll :.4f}')

        samples_left_to_generate = self.cfg.general.final_model_samples_to_generate
        samples_left_to_save = self.cfg.general.final_model_samples_to_save
        chains_left_to_save = self.cfg.general.final_model_chains_to_save

        samples = []
        id = 0
        while samples_left_to_generate > 0:
            self.print(f'Samples left to generate: {samples_left_to_generate}/'
                       f'{self.cfg.general.final_model_samples_to_generate}', end='', flush=True)
            bs = 2 * self.cfg.train.batch_size
            to_generate = min(samples_left_to_generate, bs)
            to_save = min(samples_left_to_save, bs)
            chains_save = min(chains_left_to_save, bs)
            samples.extend(self.sample_batch(id, to_generate, num_nodes=None, save_final=to_save,
                                             keep_chain=chains_save, number_chain_steps=self.number_chain_steps))
            id += to_generate
            samples_left_to_save -= to_save
            samples_left_to_generate -= to_generate
            chains_left_to_save -= chains_save
        self.print("Saving the generated graphs")
        filename = f'generated_samples1.txt'
        for i in range(2, 10):
            if os.path.exists(filename):
                filename = f'generated_samples{i}.txt'
            else:
                break
        with open(filename, 'w') as f:
            for item in samples:
                f.write(f"N={item[0].shape[0]}\n")
                atoms = item[0].tolist()
                f.write("X: \n")
                for at in atoms:
                    f.write(f"{at} ")
                f.write("\n")
                f.write("E: \n")
                for bond_list in item[1]:
                    for bond in bond_list:
                        f.write(f"{bond} ")
                    f.write("\n")
                f.write("\n")
        self.print("Generated graphs Saved. Computing sampling metrics...")
        validity = self.sampling_metrics(samples, self.name, self.current_epoch, self.val_counter, test=True, local_rank=self.local_rank)
        self.print("Done testing.")


    def apply_noise(self, X, E, y, node_mask):  # Sampling nodes and edges together, focus only on the subgraph between valid nodes, select edges proportionally, and change the scale

        batch_size, num_nodes, _ = X.size()
        device = X.device

        times = 2

        n_over_m = torch.rand(batch_size, device=device)

        ratio = (1 - torch.cos(0.5 * math.pi * ((n_over_m + 0.008) / (1 + 0.008))) ** 2)

        valid_nodes_per_graph = node_mask.sum(dim=1)  # (batch_size,)
        steps = times * valid_nodes_per_graph.float()  
        s = (ratio * steps).long() + 1
        t_nodes = ((s + (times - 1)) // times) 

        t_nodes = torch.clamp(t_nodes, min=torch.tensor(1, device=device), max=valid_nodes_per_graph)

        rand_nodes = torch.rand(batch_size, num_nodes, device=device)
        rand_nodes[~node_mask] = -float('inf')
        sorted_scores_nodes, sorted_indices_nodes = torch.sort(rand_nodes, dim=1, descending=True)
        range_tensor_nodes = torch.arange(num_nodes, device=device).unsqueeze(0).expand(batch_size, num_nodes)
        mask_nodes = range_tensor_nodes < t_nodes.unsqueeze(1)
        node_mask_noise = torch.zeros_like(mask_nodes, dtype=torch.bool)
        node_mask_noise.scatter_(1, sorted_indices_nodes, mask_nodes)
  

        node_mask_noise_row = node_mask_noise.unsqueeze(2)
        node_mask_noise_col = node_mask_noise.unsqueeze(1)
        potential_edge_mask = node_mask_noise_row & node_mask_noise_col

        diag_mask = ~torch.eye(num_nodes, dtype=torch.bool, device=device).unsqueeze(0)
        potential_edge_mask = potential_edge_mask & diag_mask

        node_mask_expanded = node_mask.unsqueeze(1) & node_mask.unsqueeze(2)
        potential_edge_mask = potential_edge_mask & node_mask_expanded

        triu_indices = torch.triu_indices(num_nodes, num_nodes, offset=1, device=device)
        potential_edge_mask_upper = potential_edge_mask[:, triu_indices[0], triu_indices[1]]


        r = s / steps
        r = (1 - torch.cos(0.5 * math.pi * ((r + 0.008) / (1 + 0.008))) ** 2)
        edge_noise_ratio = torch.full((batch_size,), 0.2, device=device) * r


        rand_edges = torch.rand(batch_size, triu_indices.size(1), device=device)
        rand_edges[~potential_edge_mask_upper] = 2.0 

 
        #edge_threshold = torch.quantile(rand_edges, edge_noise_ratio, dim=1, keepdim=True)
        Q = torch.quantile(rand_edges, edge_noise_ratio, dim=1)  # : (batch_size, batch_size)
     
        diag_idx = torch.arange(batch_size, device=device)
        edge_threshold = Q[diag_idx, diag_idx].unsqueeze(1)  # (batch_size, 1)

        edge_mask_noise_flat = rand_edges <= edge_threshold


        edge_mask_noise_flat = edge_mask_noise_flat & potential_edge_mask_upper

        # (batch_size, 2, num_edges)
        triu_indices_exp = triu_indices.unsqueeze(0).expand(batch_size, -1, -1)

        # (batch_size, num_edges)
        batch_indices = torch.arange(batch_size, device=device).unsqueeze(1).expand(-1, triu_indices.size(1))

        selected_rows = triu_indices_exp[:, 0, :][edge_mask_noise_flat]  # (num_selected_edges,)
        selected_cols = triu_indices_exp[:, 1, :][edge_mask_noise_flat]  # (num_selected_edges,)
        selected_batch = batch_indices[edge_mask_noise_flat]  # (num_selected_edges,) 

        # (batch_size, num_nodes, num_nodes)
        edge_mask_noise = torch.zeros((batch_size, num_nodes, num_nodes), dtype=torch.bool, device=device)
        edge_mask_noise[selected_batch, selected_rows, selected_cols] = True
        edge_mask_noise[selected_batch, selected_cols, selected_rows] = True
        # (batch_size,)
        t_edges = edge_mask_noise_flat.sum(dim=1)


        Qtb = self.transition_model.get_discrete_Qt_bar(device=device)
        probX = torch.matmul(X, Qtb.X)
        probE = torch.matmul(E, Qtb.E)


        current_X = X.argmax(dim=-1)
        current_E = E.argmax(dim=-1)

        probX_selected = probX.clone()
        if self.Xdim_output > 1:
            probX_selected[node_mask_noise] = probX_selected[node_mask_noise].scatter_(
                dim=-1,
                index=current_X[node_mask_noise].unsqueeze(-1),
                value=0
            )
            probX_selected[node_mask_noise] = probX_selected[node_mask_noise] / probX_selected[node_mask_noise].sum(dim=-1, keepdim=True)

        probE_selected = probE.clone()
        if self.Edim_output > 1:
            probE_selected[edge_mask_noise] = probE_selected[edge_mask_noise].scatter_(
                dim=-1,
                index=current_E[edge_mask_noise].unsqueeze(-1),
                value=0
            )
            probE_selected[edge_mask_noise] = probE_selected[edge_mask_noise] / probE_selected[edge_mask_noise].sum(dim=-1, keepdim=True)

        sampled = diffusion_utils.sample_discrete_features(self.limit_dist, probX_selected, probE_selected, node_mask)
        X_t = sampled.X
        E_t = sampled.E

        X_t_final = X.clone()
        E_t_final = E.clone()

        X_t_final[node_mask_noise] = F.one_hot(X_t[node_mask_noise], num_classes=self.Xdim_output).float()
        E_t_final[edge_mask_noise] = F.one_hot(E_t[edge_mask_noise], num_classes=self.Edim_output).float()

        z_t = utils.PlaceHolder(X=X_t_final, E=E_t_final, y=y).type_as(X_t_final).mask(node_mask)

        max_possible_edges = (valid_nodes_per_graph * (valid_nodes_per_graph - 1)) // 2


        noisy_data = {
            't_int': t_nodes,
            't_e_int': t_edges,
            't_nodes': t_nodes.unsqueeze(1).float() / valid_nodes_per_graph.unsqueeze(1).float(),
            't_edges': t_edges.unsqueeze(1).float() / (max_possible_edges.unsqueeze(1).float() + 1e-8),
            'X_t': z_t.X, 
            'E_t': z_t.E,
            'y_t': z_t.y,
            'node_mask': node_mask
        }

        return noisy_data



    def forward(self, noisy_data, extra_data, node_mask):
        X = torch.cat((noisy_data['X_t'], extra_data.X), dim=2).float()
        E = torch.cat((noisy_data['E_t'], extra_data.E), dim=3).float()
        y = torch.hstack((noisy_data['y_t'], extra_data.y)).float()
        return self.model(X, E, y, node_mask)



    def replace_supernode_with_ring(self, subX: torch.Tensor, subE: torch.Tensor):
        device = subX.device
        n = subX.size(0)

        base_count = len(getattr(self.dataset_info, 'atom_encoder', {'C':0,'N':1,'O':2,'F':3}))
        super_mask = (subX >= base_count).nonzero(as_tuple=False)
        if super_mask.numel() == 0:
            return subX, subE, n

        super_idx = super_mask[0].item()
        ring_lbl = subX[super_idx].item()  # >= base_count
        ring_smi = self.dataset_info.label_to_ring[ring_lbl]

        external_neighbors = []
        for j in range(n):
            if j == super_idx:
                continue
            btype = subE[super_idx, j].item()
            if btype > 0:
                external_neighbors.append(j)

        # 根据该超点的标签决定具体环
        ring_labels, ringE = self.parse_ring_smi(ring_smi)  # labels in [0..3], ringE in [r,r] with 1..4
        ring_size = len(ring_labels)
        if ring_size == 0:
            ring_labels = [0]
            ringE = torch.zeros((1,1), dtype=torch.long, device=device)
            ring_size = 1

        keep_idx = [x for x in range(n) if x != super_idx]
        subX_noSuper = subX[keep_idx]
        subE_noSuper = subE[keep_idx][:, keep_idx]

        ringX= torch.tensor(ring_labels, dtype=torch.long, device=device)
        # newX => cat(subX_noSuper, ringX)
        newX = torch.cat([subX_noSuper, ringX], dim=0)
        new_n = subX_noSuper.size(0) + ring_size

        newE = torch.zeros((new_n,new_n), dtype=torch.long, device=device)
        newE[:subX_noSuper.size(0), :subX_noSuper.size(0)] = subE_noSuper
        # place ringE => offset
        offset = subX_noSuper.size(0)
        newE[offset:offset+ring_size, offset:offset+ring_size] = ringE

        if len(external_neighbors) > 0:
            # 仅允许连接到环内的碳原子；若环内无碳，则不连外部
            carbon_rel_indices = [k for k, lbl in enumerate(ring_labels) if lbl == 0]
            if len(carbon_rel_indices) > 0:
                # 先进行一次随机打乱的不重复分配，超过后允许重复
                shuffled = carbon_rel_indices[:]
                random.shuffle(shuffled)
                c_count = len(shuffled)
                for idx_enb, enb in enumerate(external_neighbors):
                    real_j = keep_idx.index(enb)
                    if idx_enb < c_count:
                        rel_c = shuffled[idx_enb]
                    else:
                        rel_c = random.choice(carbon_rel_indices)
                    ringC_idx = offset + rel_c
                    # 外部连边统一单键
                    newE[real_j, ringC_idx] = 1
                    newE[ringC_idx, real_j] = 1

        return newX, newE, new_n

    def _sample_ring_smiles(self, ring_types):
        # 如果ring_types是集合，直接随机选择
        if isinstance(ring_types, set):
            return random.choice(list(ring_types))
        
        # 如果ring_types是字典，按权重采样
        keys = list(ring_types.keys())
        vals = list(ring_types.values())
        s = sum(vals)
        r = random.uniform(0, s)
        accum = 0
        for k, v in zip(keys, vals):
            accum += v
            if accum >= r:
                return k
        return keys[-1]

    def _parse_ring_smiles(self, ring_smi):
        # 使用数据集信息中的原子编码器
        if hasattr(self.dataset_info, 'atom_encoder'):
            label_map = self.dataset_info.atom_encoder
        else:
            # 默认QM9原子类型映射
            label_map = {'C':0, 'N':1, 'O':2, 'F':3}
        
        arr = []
        for ch in ring_smi:
            if ch in label_map:
                arr.append(label_map[ch])
        if len(arr) == 0:
            arr = [0]  # 默认为碳原子
        return arr

    def convert_feature_with_supernode(self, X, E, n_nodes):
        batch_size= X.size(0)
        converted_X = []  
        converted_E = []  
        molecule_list= []

        for i in range(batch_size):
            n = n_nodes[i].item()
            newX= X[i,:n]       # (n, feat_dim)
            newE= E[i,:n,:n]    # (n, n)

            base_count = len(getattr(self.dataset_info, 'atom_encoder', {'C':0,'N':1,'O':2,'F':3}))
            while (newX >= base_count).any():
                newX, newE, new_n = self.replace_supernode_with_ring(newX, newE)

            atom_types= newX.cpu()
            edge_types= newE.cpu()

            molecule_list.append([atom_types, edge_types])
            converted_X.append(atom_types) 
            converted_E.append(edge_types)  

        # 在所有分子处理完成后进行padding
            padded_X, padded_E = self.pad_features(converted_X, converted_E)

        return molecule_list, padded_X, padded_E

    def pad_tensor(self, tensor, max_length, pad_value=0):
        if tensor.dim() == 1:
            pad = (0, max_length - tensor.size(0))
            return F.pad(tensor, pad, "constant", pad_value)
        elif tensor.dim() == 2:
            pad = (0, max_length - tensor.size(1), 0, max_length - tensor.size(0))
            return F.pad(tensor, pad, "constant", pad_value)
        else:
            raise ValueError("Unsupported tensor dimensions for padding.")

    def pad_features(self, converted_X, converted_E):
     
        max_n = max([x.size(0) for x in converted_X])
        

        padded_X = []
        for x in converted_X:
            padded = self.pad_tensor(x, max_n, pad_value=0) 
            padded_X.append(padded)
        padded_X = torch.stack(padded_X, dim=0)  # shape (batch_size, max_n)
        
        padded_E = []
        for E in converted_E:
            padded = self.pad_tensor(E, max_n, pad_value=0)  # pad with 0 (noEdge)
            padded_E.append(padded)
        padded_E = torch.stack(padded_E, dim=0)  # shape (batch_size, max_n, max_n)
        
        return padded_X, padded_E
    

    def graph_to_smiles(self, origin_x: torch.Tensor, 
                        origin_e: torch.Tensor,
                        n_nodes: int) -> str:
     
        from rdkit import Chem, RDLogger
        from rdkit.Chem import RWMol
        # 禁用 RDKit 的警告输出
        RDLogger.DisableLog('rdApp.*')
        
        if n_nodes==0:
            
            return ""
        
        rwmol = RWMol()
        
        old_to_new = []
        for i in range(n_nodes):
            lbl = origin_x[i].item()
            sym = self.dataset_info.label_to_symbol.get(lbl, "C")  # fallback => "C"
            a = Chem.Atom(sym)

            new_idx = rwmol.AddAtom(a)
            old_to_new.append(new_idx)
        
        #   bond type => label_to_bondtype
        for i in range(n_nodes):
            for j in range(i+1, n_nodes):
                bt_lbl = origin_e[i,j].item()
                if bt_lbl>0:
                    # 1..4
                    bond_t = self.dataset_info.label_to_bondtype.get(bt_lbl, rdchem.BondType.SINGLE)
                    rwmol.AddBond(old_to_new[i], old_to_new[j], bond_t)
        
        mol = rwmol.GetMol()
        

        try:
            Chem.SanitizeMol(mol) 
            smi = Chem.MolToSmiles(mol, canonical=True)
        except Exception as e:
            smi = f"[InvalidMol_{str(e)}]"
        return smi

    def decode_origin_graphs_to_smiles(self, origin_X: torch.Tensor,
                                    origin_E: torch.Tensor,
                                    n_nodes: torch.Tensor):

        batch_size = origin_X.size(0)
        smiles_list = []
        for i in range(batch_size):
            n_i = n_nodes[i].item()
            subX = origin_X[i,:n_i]
            subE = origin_E[i,:n_i, :n_i]
            smi = self.graph_to_smiles(subX, subE, n_i)
            smiles_list.append(smi)
        return smiles_list

    def parse_ring_smi(self, ring_smi):

        from rdkit import Chem
        from rdkit.Chem import BondType as BT
        
        mol = Chem.MolFromSmiles(ring_smi)
        if not mol:
            return [], torch.zeros((0,0),dtype=torch.long)
        n = mol.GetNumAtoms()
        node_labels = []
        for a in mol.GetAtoms():
            sym = a.GetSymbol()
            node_labels.append(self.dataset_info.atom_encoder[sym])  # e.g. 'C'->0
        
        E = torch.zeros((n,n), dtype=torch.long)
        for b in mol.GetBonds():
            i = b.GetBeginAtomIdx()
            j = b.GetEndAtomIdx()
            bt = b.GetBondType()
            if bt==BT.SINGLE:
                E[i,j]=1; E[j,i]=1
            elif bt==BT.DOUBLE:
                E[i,j]=2; E[j,i]=2
            elif bt==BT.TRIPLE:
                E[i,j]=3; E[j,i]=3
            elif bt==BT.AROMATIC:
                E[i,j]=4; E[j,i]=4
            else:
                E[i,j]=1; E[j,i]=1
        return node_labels, E


  
    def decode_single_graph(self, subX, subE, n):
       
        node_info = [] 
        
        old2new = [-1]*n  # old node -> new idx
        new_idx_count = 0
        

        supernode_data = []
        
        for old_i in range(n):
            lbl = subX[old_i].item()  # int
            if lbl<4:
          
                node_info.append({'lbl': lbl, 'adj': {}}) 
                old2new[old_i]= new_idx_count
                new_idx_count+=1
            else:
                # ring supernode
                supernode_data.append(old_i)

        for old_i in range(n):
            for old_j in range(old_i+1, n):
                b = subE[old_i, old_j].item()
                if b>0:  # 1..4
                 
                    ni = old2new[old_i]
                    nj = old2new[old_j]
                    if ni>=0 and nj>=0:
                        # add to adjacency
                        node_info[ni]['adj'][nj]= b
                        node_info[nj]['adj'][ni]= b
        
        for snode in supernode_data:
            ring_lbl = subX[snode].item()  # in [4..13]
            # parse ring
            ring_smi = self.dataset_info.label_to_ring[ring_lbl]
            r_nodes, r_E = self.parse_ring_smi(ring_smi)  # r_nodes in [0..3], r_E in [r,r], r in [0..4]
            
            ring_base_idx = new_idx_count
            ring_size = len(r_nodes)
            # create them
            for rlbl in r_nodes:
                node_info.append({'lbl': rlbl, 'adj': {}})
            ring_new_indices = list(range(ring_base_idx, ring_base_idx+ring_size))
            new_idx_count += ring_size

            for rr_i in range(ring_size):
                for rr_j in range(rr_i+1, ring_size):
                    bb = r_E[rr_i, rr_j].item()
                    if bb>0:
                        # connect ring_new_indices[rr_i] <-> ring_new_indices[rr_j]
                        ni = ring_new_indices[rr_i]
                        nj = ring_new_indices[rr_j]
                        node_info[ni]['adj'][nj]= bb
                        node_info[nj]['adj'][ni]= bb
            
            ext_edges = []
            for other in range(n):
                if other!= snode:
                    b = subE[snode, other].item()
                    if b>0:  # supernode->other
                        ext_edges.append((other,b))
            
          
            ringC_list = []
            for ii, at_lbl in enumerate(r_nodes):
                if at_lbl==0:  # 0=>C
                    ringC_list.append(ring_new_indices[ii])
            # 若环内无碳原子，则不与外部相连
            if len(ringC_list)==0:
                ringC_list = []
               
            
            c_count = len(ringC_list)
            if c_count > 0:
                # 先不重复随机分配，再允许重复
                shuffled = ringC_list[:]
                random.shuffle(shuffled)
                for idx_e, (oth, _) in enumerate(ext_edges):
                    new_oth = old2new[oth]
                    if new_oth < 0:
                        continue
                    if idx_e < c_count:
                        targetC = shuffled[idx_e]
                    else:
                        targetC = random.choice(ringC_list)
                    # 外部连边统一单键
                    node_info[targetC]['adj'][new_oth]= 1
                    node_info[new_oth]['adj'][targetC]= 1
        

        new_n = len(node_info)
        origin_X = torch.zeros((new_n,), dtype=torch.long)
        origin_E = torch.zeros((new_n,new_n), dtype=torch.long)
        
        for i, info in enumerate(node_info):
            origin_X[i] = info['lbl']  # 0..3
        for i, info in enumerate(node_info):
            for j, bb in info['adj'].items():
                origin_E[i,j]= bb
        
        return origin_X, origin_E, new_n



    def decode_batch(self, X, E, n_nodes):
      
        batch_size = X.size(0)
        device = X.device
        
        origin_list_X = []
        origin_list_E = []
        new_n_list = []
        
        for i in range(batch_size):
            n = n_nodes[i].item()
            subX = X[i,:n]      # shape(n,)
            subE = E[i,:n,:n]   # shape(n,n)
            # decode
            oX,oE,nn = self.decode_single_graph(subX, subE, n)
            origin_list_X.append(oX)
            origin_list_E.append(oE)
            new_n_list.append(nn)
        
        new_max_n = max(new_n_list) if new_n_list else 0
        
        # padding => shape=(batch_size, new_max_n), (batch_size,new_max_n,new_max_n)
        origin_X_pad = torch.zeros((batch_size, new_max_n), dtype=torch.long, device=device)
        origin_E_pad = torch.zeros((batch_size, new_max_n, new_max_n), dtype=torch.long, device=device)
        
        for i in range(batch_size):
            nn = new_n_list[i]
            if nn>0:
                origin_X_pad[i,:nn] = origin_list_X[i]
                origin_E_pad[i,:nn,:nn] = origin_list_E[i]
        
        return origin_X_pad, origin_E_pad, torch.tensor(new_n_list, dtype=torch.long, device=device)



    def build_molecule_list(self, X, E, n_nodes):
        """
        molecule_list = []
        for i in range(batch_size):
            n = n_new[i]
            atom_types = origin_X_pad[i,:n]
            edge_types= origin_E_pad[i,:n,:n]
            molecule_list.append([atom_types, edge_types])
        """
        origin_X, origin_E, n_new = self.decode_batch(X,E,n_nodes)
        batch_size = X.size(0)
        
        molecule_list = []
        for i in range(batch_size):
            nn = n_new[i].item()
            at = origin_X[i,:nn].cpu()
            ed = origin_E[i,:nn,:nn].cpu()
            molecule_list.append([at, ed])
        try:
            from rdkit import Chem, RDLogger
            # 禁用 RDKit 的警告输出
            RDLogger.DisableLog('rdApp.*')
            
            def _is_valid_smiles(s):
                try:
                    m = Chem.MolFromSmiles(s)
                    if m is None:
                        return False
                    Chem.SanitizeMol(m)
                    return True
                except Exception:
                    return False
            
            train_ds = None
            try:
                if getattr(self, 'trainer', None) is not None and getattr(self.trainer, 'datamodule', None) is not None:
                    dm = self.trainer.datamodule
                    ds_map = getattr(dm, 'datasets', None)
                    if isinstance(ds_map, dict):
                        train_ds = ds_map.get('train', None)
            except Exception:
                train_ds = None
          
            for i in range(batch_size):
                nn = n_new[i].item()
                at = origin_X[i,:nn].cpu()
                ed = origin_E[i,:nn,:nn].cpu()
                smi = self.graph_to_smiles(at, ed, nn)
                if not _is_valid_smiles(smi):
                    
                    if train_ds is not None and len(train_ds) > 0:
                        import random
                        ridx = random.randrange(len(train_ds))
                        try:
                            d = train_ds[ridx]
                            n2 = d.x.size(0)
                            xl = d.x.argmax(dim=1).cpu()
                            ea = d.edge_attr.argmax(dim=1).cpu()
                            ei = d.edge_index.cpu()
                            E2 = torch.zeros((n2, n2), dtype=torch.long)
                            for k in range(ei.size(1)):
                                u = int(ei[0, k].item()); v = int(ei[1, k].item())
                                bt = int(ea[k].item())
                                E2[u, v] = bt
                            molecule_list[i] = [xl, E2]
                        except Exception:
                            pass
        except Exception:
            pass
        return origin_X, origin_E, molecule_list


    @torch.no_grad()
    def sample_batch(self, batch_id: int, batch_size: int, keep_chain: int, number_chain_steps: int,
                    save_final: int, num_nodes=None):  # focus only on the subgraph between the valid nodes
        """
        :param batch_id: int
        :param batch_size: int
        :param num_nodes: int, <int>tensor (batch_size) (optional) for specifying number of nodes
        :param save_final: int: number of predictions to save to file
        :param keep_chain: int: number of chains to save to file
        :param number_chain_steps: number of timesteps to save for each chain
        :return: molecule_list. Each element of this list is a tuple (atom_types, edge_types)
        """
        if num_nodes is None:
            n_nodes = self.node_dist.sample_n(batch_size, self.device)
        elif type(num_nodes) == int:
            n_nodes = num_nodes * torch.ones(batch_size, device=self.device, dtype=torch.int)
        else:
            assert isinstance(num_nodes, torch.Tensor)
            n_nodes = num_nodes

        n_max = torch.max(n_nodes).item()
        arange = torch.arange(n_max, device=self.device).unsqueeze(0).expand(batch_size, -1)
        node_mask = arange < n_nodes.unsqueeze(1)
        #   -- z_T  (batch_size, n_max, feature_dim)
        z_T = diffusion_utils.sample_discrete_feature_noise(dataset_name=self.dataset_name, limit_dist=self.limit_dist, node_mask=node_mask)
        X, E, y = z_T.X, z_T.E, z_T.y
        assert (E == torch.transpose(E, 1, 2)).all()

        chain_X_size = torch.Size((number_chain_steps + 1, keep_chain, X.size(1)))
        chain_E_size = torch.Size((number_chain_steps + 1, keep_chain, E.size(1), E.size(2)))

        chain_X = torch.zeros(chain_X_size, device=self.device)
        chain_E = torch.zeros(chain_E_size, device=self.device)


        valid_nodes_per_graph = n_nodes  # (batch_size,)
        max_node_steps = valid_nodes_per_graph.max().item() 

        times = 2
        t_nodes = (times * valid_nodes_per_graph).clone()  #  t_nodes 为 2*n
        valid_nodes = valid_nodes_per_graph.clone()  
        edge_noise_ratio = 0.2
        valid_edges = (valid_nodes * (valid_nodes - 1)) // 2 + 1e-8
        t_edges = (valid_edges.float() * edge_noise_ratio).floor()

        total_steps = (times * valid_nodes_per_graph).max().item()
        steps = (times * valid_nodes_per_graph).float()

        for step in reversed(range(total_steps + 1)):
            
            s_nodes = t_nodes - 1
            s_nodes = torch.clamp(s_nodes, min=0)

            t_nodes_real = (t_nodes + (times - 1)) // 2
            s_nodes_real = (s_nodes + (times - 1)) // 2

            t_norm_nodes = t_nodes_real.float() / valid_nodes.float()
            t_norm_edges = t_edges.float() / valid_edges.float()

            general_s_nodes = s_nodes
            s_nodes_tensor = s_nodes_real.unsqueeze(1).float()
            t_nodes_tensor = t_nodes_real.unsqueeze(1).float()
            t_norm_nodes_tensor = t_norm_nodes.unsqueeze(1)

            t_norm_edges_tensor = t_norm_edges.unsqueeze(1)

            
            sampled_s, discrete_sampled_s, num_edges = self.sample_p_zs_given_zt(
                s_nodes_tensor, t_nodes_tensor, t_norm_nodes_tensor, t_norm_edges_tensor, general_s_nodes, steps,
                X, E, y, node_mask)

            X, E, y = sampled_s.X, sampled_s.E, sampled_s.y


            write_index = t_nodes_real.min().item()
            if write_index < chain_X.size(0):
                chain_X[write_index] = discrete_sampled_s.X[:keep_chain]
                chain_E[write_index] = discrete_sampled_s.E[:keep_chain]


            t_nodes = s_nodes
            t_edges = num_edges 

        #X, E = self.connect_components(X, E, node_mask)
        sampled = utils.PlaceHolder(X=X, E=E, y=torch.zeros(y.shape[0], 0))
        sampled = sampled.mask(node_mask, collapse=True)
        X, E, y = sampled.X, sampled.E, sampled.y

        # 恢复超点还原功能
        molecule_list, X, E = self.convert_feature_with_supernode(X, E, n_nodes)

        chain_X_size = torch.Size((number_chain_steps + 1, keep_chain, X.size(1)))
        chain_E_size = torch.Size((number_chain_steps + 1, keep_chain, E.size(1), E.size(2)))

        chain_X = torch.zeros(chain_X_size, device=self.device)
        chain_E = torch.zeros(chain_E_size, device=self.device)

        if keep_chain > 0:
            final_X_chain = X[:keep_chain]
            final_E_chain = E[:keep_chain]

            chain_X[0] = final_X_chain  
            chain_E[0] = final_E_chain

            chain_X = diffusion_utils.reverse_tensor(chain_X)
            chain_E = diffusion_utils.reverse_tensor(chain_E)

            chain_X = torch.cat([chain_X, chain_X[-1:].repeat(10, 1, 1)], dim=0)
            chain_E = torch.cat([chain_E, chain_E[-1:].repeat(10, 1, 1, 1)], dim=0)
            assert chain_X.size(0) == (number_chain_steps + 11)


        
        if self.visualization_tools is not None:
            self.print('Visualizing chains...')
            current_path = os.getcwd()
            # 获取可视化目录，如果没有配置则使用当前目录
            vis_dir = getattr(self.cfg.general, 'visualization_dir', None)
            if vis_dir is None:
                vis_dir = os.getcwd()
            
            num_molecules = chain_X.size(1)  
            for i in range(num_molecules):
                result_path = os.path.join(vis_dir, f'chains/{self.cfg.general.name}/'
                                                        f'epoch{self.current_epoch}/'
                                                        f'chains/molecule_{batch_id + i}')
                # 使用 exist_ok=True 避免多进程创建目录时的竞态条件
                os.makedirs(result_path, exist_ok=True)
                # 只在主进程（local_rank == 0）时进行可视化，避免多进程同时写入
                if getattr(self, 'local_rank', 0) == 0:
                    _ = self.visualization_tools.visualize_chain(result_path,
                                                                chain_X[:, i, :].cpu().numpy(),
                                                                chain_E[:, i, :].cpu().numpy())
                self.print('\r{}/{} complete'.format(i+1, num_molecules), end='', flush=True)
            self.print('\nVisualizing molecules...')

            # 获取可视化目录，如果没有配置则使用当前目录
            vis_dir = getattr(self.cfg.general, 'visualization_dir', None)
            if vis_dir is None:
                vis_dir = os.getcwd()
           
            result_path = os.path.join(vis_dir,
                                    f'graphs/{self.name}/epoch{self.current_epoch}_b{batch_id}/')
            # 使用 exist_ok=True 避免多进程创建目录时的竞态条件
            os.makedirs(result_path, exist_ok=True)
            # 只在主进程（local_rank == 0）时进行可视化，避免多进程同时写入
            if getattr(self, 'local_rank', 0) == 0:
                self.visualization_tools.visualize(result_path, molecule_list, save_final)
                self.print("Done.")

        return molecule_list


    def sample_p_zs_given_zt(self, s_nodes, t_nodes, t_norm_nodes, t_norm_edges, general_s_nodes, steps, X_t, E_t, y_t, node_mask):
        #Samples from zs ~ p(zs | zt). Only used during sampling.
        #   if last_step, return the graph prediction as well
        bs, n, dxs = X_t.shape

        # Neural net predictions
        noisy_data = {'X_t': X_t, 'E_t': E_t, 'y_t': y_t, 't_nodes': t_norm_nodes, 't_edges': t_norm_edges, 'node_mask': node_mask}
        extra_data = self.compute_extra_data(noisy_data)
        pred = self.forward(noisy_data, extra_data, node_mask)

        pred_X = pred.X
        pred_E = pred.E

        X0 = pred_X.argmax(dim=-1)  # Shape: (bs, n)

        X0_onehot = F.one_hot(X0, num_classes=self.Xdim_output).float()

        
        triu_indices = torch.triu_indices(n, n, offset=1, device=self.device)  # (2, num_edges)
        num_edges = triu_indices.shape[1]

        
        pred_E_upper = pred_E[:, triu_indices[0], triu_indices[1], :]  # (bs, num_edges, de_out)

        
        E0_upper = pred_E_upper.argmax(dim=-1)  # (bs, num_edges)

        
        E0_upper_onehot = F.one_hot(E0_upper, num_classes=self.Edim_output).float()  # (bs, num_edges, de_out)

        
        E0 = torch.zeros(bs, n, n, device=self.device).long()  # (bs, n, n)

        
        E0[:, triu_indices[0], triu_indices[1]] = E0_upper

        
        E0[:, triu_indices[1], triu_indices[0]] = E0_upper

        
        E0_onehot = F.one_hot(E0, num_classes=self.Edim_output).float()  # (bs, n, n, de_out)

        z_s, num_edges = self.q_s_given_0(s_nodes, general_s_nodes, steps, X0_onehot, E0_onehot, y_t, node_mask)
        X_s = z_s.X
        E_s = z_s.E


        no_sampling_mask = (t_nodes <= 0).view(-1)  
        
        X_s[no_sampling_mask] = X_t[no_sampling_mask].float()
        E_s[no_sampling_mask] = E_t[no_sampling_mask].float()
        

        out_one_hot = utils.PlaceHolder(X=X_s, E=E_s, y=torch.zeros(y_t.shape[0], 0))
        out_discrete = utils.PlaceHolder(X=X_s, E=E_s, y=torch.zeros(y_t.shape[0], 0))

        return out_one_hot.mask(node_mask).type_as(y_t), out_discrete.mask(node_mask, collapse=True).type_as(y_t), num_edges


    def compute_extra_data(self, noisy_data):
        """ At every training step (after adding noise) and step in sampling, compute extra information and append to
            the network input. """

        extra_features = self.extra_features(noisy_data)
        extra_molecular_features = self.domain_features(noisy_data)

        extra_X = torch.cat((extra_features.X, extra_molecular_features.X), dim=-1)
        extra_E = torch.cat((extra_features.E, extra_molecular_features.E), dim=-1)
        extra_y = torch.cat((extra_features.y, extra_molecular_features.y), dim=-1)

        t_nodes = noisy_data['t_nodes']
        t_edges = noisy_data['t_edges']
        extra_y = torch.cat((extra_y, t_nodes, t_edges), dim=1)
        return utils.PlaceHolder(X=extra_X, E=extra_E, y=extra_y)



    def q_s_given_0(self, s_nodes, general_s_nodes, steps, X, E, y, node_mask):  # At the same time, focus only on the subgraph between valid nodes and select edges proportionally
 
        batch_size, num_nodes, _ = X.size()
        device = X.device

        valid_nodes_per_graph = node_mask.sum(dim=1)  

        
        rand_nodes = torch.rand(batch_size, num_nodes, device=device)
        rand_nodes[~node_mask] = -float('inf')
        sorted_scores_nodes, sorted_indices_nodes = torch.sort(rand_nodes, dim=1, descending=True)
        range_tensor_nodes = torch.arange(num_nodes, device=device).unsqueeze(0).expand(batch_size, num_nodes)
        
        mask_nodes = range_tensor_nodes < s_nodes 
        node_mask_noise = torch.zeros_like(mask_nodes, dtype=torch.bool)
        node_mask_noise.scatter_(1, sorted_indices_nodes, mask_nodes)

        node_mask_noise_row = node_mask_noise.unsqueeze(2)
        node_mask_noise_col = node_mask_noise.unsqueeze(1)
        potential_edge_mask = node_mask_noise_row & node_mask_noise_col
        diag_mask = ~torch.eye(num_nodes, dtype=torch.bool, device=device).unsqueeze(0)
        potential_edge_mask = potential_edge_mask & diag_mask
        node_mask_expanded = node_mask.unsqueeze(1) & node_mask.unsqueeze(2)
        potential_edge_mask = potential_edge_mask & node_mask_expanded

        triu_indices = torch.triu_indices(num_nodes, num_nodes, offset=1, device=device)
        potential_edge_mask_upper = potential_edge_mask[:, triu_indices[0], triu_indices[1]]

        s = general_s_nodes
        r = s / steps
        r = (1 - torch.cos(0.5 * math.pi * ((r + 0.008) / (1 + 0.008))) ** 2)
        edge_noise_ratio = torch.full((batch_size,), 0.2, device=device) * r

        rand_edges = torch.rand(batch_size, triu_indices.size(1), device=device)
        rand_edges[~potential_edge_mask_upper] = 2.0


        #edge_threshold = torch.quantile(rand_edges, edge_noise_ratio, dim=1, keepdim=True)
        Q = torch.quantile(rand_edges, edge_noise_ratio, dim=1)  # : (batch_size, batch_size)

        diag_idx = torch.arange(batch_size, device=device)
        edge_threshold = Q[diag_idx, diag_idx].unsqueeze(1)  # (batch_size, 1)
        edge_mask_noise_flat = rand_edges <= edge_threshold
        edge_mask_noise_flat = edge_mask_noise_flat & potential_edge_mask_upper

        triu_indices_exp = triu_indices.unsqueeze(0).expand(batch_size, -1, -1)
        batch_indices = torch.arange(batch_size, device=device).unsqueeze(1).expand(-1, triu_indices.size(1))

        selected_rows = triu_indices_exp[:, 0, :][edge_mask_noise_flat]
        selected_cols = triu_indices_exp[:, 1, :][edge_mask_noise_flat]
        selected_batch = batch_indices[edge_mask_noise_flat]

        edge_mask_noise = torch.zeros((batch_size, num_nodes, num_nodes), dtype=torch.bool, device=device)
        edge_mask_noise[selected_batch, selected_rows, selected_cols] = True
        edge_mask_noise[selected_batch, selected_cols, selected_rows] = True
        #  t_edges， (batch_size,)
        num_edges = edge_mask_noise_flat.sum(dim=1)

        Qtb = self.transition_model.get_discrete_Qt_bar(device=device)

        probX = torch.matmul(X, Qtb.X)
        probE = torch.matmul(E, Qtb.E)

        current_X = X.argmax(dim=-1)
        current_E = E.argmax(dim=-1)

        probX_selected = probX.clone()
        if self.Xdim_output > 1:
            probX_selected[node_mask_noise] = probX_selected[node_mask_noise].scatter_(
                dim=-1,
                index=current_X[node_mask_noise].unsqueeze(-1),
                value=0
            )
            probX_selected[node_mask_noise] = probX_selected[node_mask_noise] / probX_selected[node_mask_noise].sum(dim=-1, keepdim=True)

        probE_selected = probE.clone()
        if self.Edim_output > 1:
            probE_selected[edge_mask_noise] = probE_selected[edge_mask_noise].scatter_(
                dim=-1,
                index=current_E[edge_mask_noise].unsqueeze(-1),
                value=0
            )
            probE_selected[edge_mask_noise] = probE_selected[edge_mask_noise] / probE_selected[edge_mask_noise].sum(dim=-1, keepdim=True)

        probX_selected[~node_mask_noise] = X[~node_mask_noise]
        probE_selected[~edge_mask_noise] = E[~edge_mask_noise]

        sampled = diffusion_utils.sample_discrete_features(self.limit_dist, probX_selected, probE_selected, node_mask)

        X_s = sampled.X
        E_s = sampled.E

        X_s_final = X.argmax(dim=-1).clone()
        E_s_final = E.argmax(dim=-1).clone()

        X_s_final[node_mask_noise] = X_s[node_mask_noise]
        E_s_final[edge_mask_noise] = E_s[edge_mask_noise]

        X_s_onehot = F.one_hot(X_s_final, num_classes=self.Xdim_output).float()
        E_s_onehot = F.one_hot(E_s_final, num_classes=self.Edim_output).float()

        z_s = utils.PlaceHolder(X=X_s_onehot, E=E_s_onehot, y=y).type_as(X_s_onehot).mask(node_mask)

        return z_s, num_edges
