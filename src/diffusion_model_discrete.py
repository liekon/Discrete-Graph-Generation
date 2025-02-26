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
from src.diffusion import diffusion_utils
from metrics.train_metrics import TrainLossDiscrete
from metrics.val_metrics import ValLossDiscrete
from metrics.test_metrics import TestLossDiscrete
from metrics.abstract_metrics import SumExceptBatchMetric, SumExceptBatchKL, NLL
from src import utils
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

    

    def apply_noise1111111(self, X, E, y, node_mask): 
        #Sample noise and apply it to the data.

        # Sample a timestep t.
        # When evaluating, the loss for t=0 is computed separately
        lowest_t = 0 if self.training else 1
        t_int = torch.randint(lowest_t, self.T + 1, size=(X.size(0), 1), device=X.device).float()  # (bs, 1)

        s_int = t_int - 1

        t_float = t_int / self.T
        s_float = s_int / self.T

        # beta_t and alpha_s_bar are used for denoising/loss computation
        beta_t = self.noise_schedule(t_normalized=t_float)                         # (bs, 1)
        alpha_s_bar = self.noise_schedule.get_alpha_bar(t_normalized=s_float)      # (bs, 1)
        alpha_t_bar = self.noise_schedule.get_alpha_bar(t_normalized=t_float)      # (bs, 1)

        Qtb = self.transition_model.get_Qt_bar(alpha_t_bar, device=self.device)  # (bs, dx_in, dx_out), (bs, de_in, de_out)
        assert (abs(Qtb.X.sum(dim=2) - 1.) < 1e-4).all(), Qtb.X.sum(dim=2) - 1
        assert (abs(Qtb.E.sum(dim=2) - 1.) < 1e-4).all()

        # Compute transition probabilities
        probX = X @ Qtb.X  # (bs, n, dx_out)
        probE = E @ Qtb.E.unsqueeze(1)  # (bs, n, n, de_out)

        sampled_t = diffusion_utils.sample_discrete_features(self.limit_dist, probX=probX, probE=probE, node_mask=node_mask)

        X_t = F.one_hot(sampled_t.X, num_classes=self.Xdim_output)
        E_t = F.one_hot(sampled_t.E, num_classes=self.Edim_output)
        assert (X.shape == X_t.shape) and (E.shape == E_t.shape)

        z_t = utils.PlaceHolder(X=X_t, E=E_t, y=y).type_as(X_t).mask(node_mask)

        noisy_data = {'t_int': t_int, 't': t_float, 'beta_t': beta_t, 'alpha_s_bar': alpha_s_bar,
                      'alpha_t_bar': alpha_t_bar, 'X_t': z_t.X, 'E_t': z_t.E, 'y_t': z_t.y, 'node_mask': node_mask}
        return noisy_data

    
    def apply_noise1(self, X, E, y, node_mask): 
        """ 
        Sample noise and apply it to the graph data by randomly selecting 
        t nodes and t edges to perturb based on transition probabilities.

        Args:
            X (torch.Tensor): Node features of shape (batch_size, num_nodes, dx_in).
            E (torch.Tensor): Edge features of shape (batch_size, num_nodes, num_nodes, de_in).
            y (torch.Tensor): Labels or additional data.
            node_mask (torch.Tensor): Mask indicating valid nodes, shape (batch_size, num_nodes).

        Returns:
            dict: A dictionary containing noisy data and related information.
        """
        batch_size, num_nodes, _ = X.size()
        
        device = X.device
        num_edges = num_nodes * (num_nodes - 1) // 2  # C(num_nodes, 2)

        # Get upper triangular indices (excluding diagonal)
        triu_indices = torch.triu_indices(num_nodes, num_nodes, offset=1, device=device)  # (2, num_edges)

        # Sample t_nodes: integers between 1 and num_nodes (inclusive)
        t_nodes = torch.randint(1, num_nodes + 1, size=(batch_size, 1), device=device)  # Shape: (batch_size, 1)

        # Sample t_edges: integers between 1 and num_edges (inclusive)
        t_edges = torch.randint(1, num_edges + 1, size=(batch_size, 1), device=device)  # Shape: (batch_size, 1)

        # Generate random scores for nodes
        rand_nodes = torch.rand(batch_size, num_nodes, device=device)

        # Sort the scores in descending order and get sorted indices
        sorted_scores_nodes, sorted_indices_nodes = torch.sort(rand_nodes, dim=1, descending=True)

        # Create a range tensor for comparison
        range_tensor_nodes = torch.arange(num_nodes, device=device).unsqueeze(0).expand(batch_size, num_nodes)

        # Create a mask where positions less than t_nodes are True
        mask_nodes = range_tensor_nodes < t_nodes
 
        # Scatter the mask to the sorted indices to select top t_nodes
        node_mask_noise = torch.zeros_like(mask_nodes, dtype=torch.bool)
        node_mask_noise.scatter_(1, sorted_indices_nodes, mask_nodes)

        # Generate random scores for edges
        rand_edges = torch.rand(batch_size, num_edges, device=device)

        # Sort the scores in descending order and get sorted indices
        sorted_scores_edges, sorted_indices_edges = torch.sort(rand_edges, dim=1, descending=True)

        # Create a range tensor for comparison
        range_tensor_edges = torch.arange(num_edges, device=device).unsqueeze(0).expand(batch_size, num_edges)

        # Create a mask where positions less than t_edges are True
        mask_edges = range_tensor_edges < t_edges

        # Scatter the mask to the sorted indices to select top t_edges
        edge_mask_noise_flat = torch.zeros_like(mask_edges, dtype=torch.bool)
        edge_mask_noise_flat.scatter_(1, sorted_indices_edges, mask_edges)

        # Expand triu_indices to (1, 2, num_edges) and repeat for batch
        triu_indices_exp = triu_indices.unsqueeze(0).expand(batch_size, -1, -1)  # (batch_size, 2, num_edges)

        # Create batch indices
        batch_indices = torch.arange(batch_size, device=device).unsqueeze(1).expand(-1, num_edges)  # (batch_size, num_edges)

        # Gather selected edge indices
        selected_rows = triu_indices_exp[:, 0, :].masked_select(edge_mask_noise_flat)  # (num_selected,)
        selected_cols = triu_indices_exp[:, 1, :].masked_select(edge_mask_noise_flat)  # (num_selected,)
        selected_batch = batch_indices.masked_select(edge_mask_noise_flat)  # (num_selected,)

        # Initialize edge_mask_noise as all False
        edge_mask_noise = torch.zeros((batch_size, num_nodes, num_nodes), dtype=torch.bool, device=device)

        # Set selected edges in upper triangle
        edge_mask_noise[selected_batch, selected_rows, selected_cols] = True
        # Ensure symmetry by setting corresponding lower triangle edges
        edge_mask_noise[selected_batch, selected_cols, selected_rows] = True
      
        # Retrieve transition matrices for nodes and edges
        # Assuming Qtb.X has shape (dx_in, dx_out) and Qtb.E has shape (de_in, de_out)
        Qtb = self.transition_model.get_discrete_Qt_bar(device=device)  # Adjust as per your implementation

        # Compute transition probabilities for nodes
        # X: (batch_size, num_nodes, dx_in)
        # Qtb.X: (dx_in, dx_out)
        probX = torch.matmul(X, Qtb.X)  # Shape: (batch_size, num_nodes, dx_out)

        # Compute transition probabilities for edges
        # E: (batch_size, num_nodes, num_nodes, de_in)
        # Qtb.E: (de_in, de_out)
        probE = torch.matmul(E, Qtb.E)  # Shape: (batch_size, num_nodes, num_nodes, de_out)

        current_X = X.argmax(dim=-1)  # Shape: (batch_size, num_nodes)
        current_E = E.argmax(dim=-1)  # Shape: (batch_size, num_nodes, num_nodes)

        # make sure that the selected nodes will not be stable
        probX_selected = probX.clone()
        probX_selected[node_mask_noise] = probX_selected[node_mask_noise].scatter(
            dim=-1,
            index=current_X[node_mask_noise].unsqueeze(-1),
            value=0
        )
        # normalize
        probX_selected[node_mask_noise] = probX_selected[node_mask_noise] / probX_selected[node_mask_noise].sum(dim=-1, keepdim=True)

        # make sure that the selected edges will not be stable
        probE_selected = probE.clone()
        probE_selected[edge_mask_noise] = probE_selected[edge_mask_noise].scatter(
            dim=-1,
            index=current_E[edge_mask_noise].unsqueeze(-1),
            value=0
        )
        # normalize
        probE_selected[edge_mask_noise] = probE_selected[edge_mask_noise] / probE_selected[edge_mask_noise].sum(dim=-1, keepdim=True)
        

        sampled = diffusion_utils.sample_discrete_features(self.limit_dist, probX_selected, probE_selected, node_mask)

        X_t = sampled.X  # Shape: (batch_size, num_nodes)
        E_t = sampled.E  # Shape: (batch_size, num_nodes, num_nodes)


        X_t_final = X.clone()
        E_t_final = E.clone()

        
        X_t_final[node_mask_noise] = F.one_hot(X_t[node_mask_noise], num_classes=self.Xdim_output).float()


        E_t_final[edge_mask_noise] = F.one_hot(E_t[edge_mask_noise], num_classes=self.Edim_output).float()

        z_t = utils.PlaceHolder(X=X_t_final, E=E_t_final, y=y).type_as(X_t_final).mask(node_mask)

        noisy_data = {
        't_nodes': t_nodes.float() / num_nodes,          # Shape: (batch_size,)
        't_edges': t_edges.float() / num_edges,          # Shape: (batch_size,)
        'X_t': z_t.X,                        # Shape: (batch_size, num_nodes, dx_out)
        'E_t': z_t.E,                        # Shape: (batch_size, num_nodes, num_nodes, de_out)
        'y_t': z_t.y,                        
        'node_mask': node_mask               # Shape: (batch_size, num_nodes)
    }

        return noisy_data


    def apply_noise0101(self, X, E, y, node_mask): 
        """ 
        Sample noise and apply it to the graph data by randomly selecting 
        t nodes and t edges to perturb based on transition probabilities.

        Args:
            X (torch.Tensor): Node features of shape (batch_size, num_nodes, dx_in).
            E (torch.Tensor): Edge features of shape (batch_size, num_nodes, num_nodes, de_in).
            y (torch.Tensor): Labels or additional data.
            node_mask (torch.Tensor): Mask indicating valid nodes, shape (batch_size, num_nodes).

        Returns:
            dict: A dictionary containing noisy data and related information.
        """
        batch_size, num_nodes, _ = X.size()

        device = X.device
        num_edges = num_nodes * (num_nodes - 1) // 2  # C(num_nodes, 2)

        # Get upper triangular indices (excluding diagonal)
        triu_indices = torch.triu_indices(num_nodes, num_nodes, offset=1, device=device)  # (2, num_edges)

        # Sample t_nodes: integers between 1 and num_nodes (inclusive)
        t_nodes = torch.randint(1, num_nodes + 1, size=(batch_size, 1), device=device)  # Shape: (batch_size, 1)
        # Sample t_edges: integers between 1 and num_edges (inclusive)
        t_edges = ((num_nodes - 1) * t_nodes) // 2

        # Generate random scores for nodes
        rand_nodes = torch.rand(batch_size, num_nodes, device=device)

        # Sort the scores in descending order and get sorted indices
        sorted_scores_nodes, sorted_indices_nodes = torch.sort(rand_nodes, dim=1, descending=True)

        # Create a range tensor for comparison
        range_tensor_nodes = torch.arange(num_nodes, device=device).unsqueeze(0).expand(batch_size, num_nodes)

        # Create a mask where positions less than t_nodes are True
        mask_nodes = range_tensor_nodes < t_nodes
 
        # Scatter the mask to the sorted indices to select top t_nodes
        node_mask_noise = torch.zeros_like(mask_nodes, dtype=torch.bool)
        node_mask_noise.scatter_(1, sorted_indices_nodes, mask_nodes)

        # Generate random scores for edges
        rand_edges = torch.rand(batch_size, num_edges, device=device)

        # Sort the scores in descending order and get sorted indices
        sorted_scores_edges, sorted_indices_edges = torch.sort(rand_edges, dim=1, descending=True)

        # Create a range tensor for comparison
        range_tensor_edges = torch.arange(num_edges, device=device).unsqueeze(0).expand(batch_size, num_edges)

        # Create a mask where positions less than t_edges are True
        mask_edges = range_tensor_edges < t_edges

        # Scatter the mask to the sorted indices to select top t_edges
        edge_mask_noise_flat = torch.zeros_like(mask_edges, dtype=torch.bool)
        edge_mask_noise_flat.scatter_(1, sorted_indices_edges, mask_edges)

        # Expand triu_indices to (1, 2, num_edges) and repeat for batch
        triu_indices_exp = triu_indices.unsqueeze(0).expand(batch_size, -1, -1)  # (batch_size, 2, num_edges)

        # Create batch indices
        batch_indices = torch.arange(batch_size, device=device).unsqueeze(1).expand(-1, num_edges)  # (batch_size, num_edges)

        # Gather selected edge indices
        selected_rows = triu_indices_exp[:, 0, :].masked_select(edge_mask_noise_flat)  # (num_selected,)
        selected_cols = triu_indices_exp[:, 1, :].masked_select(edge_mask_noise_flat)  # (num_selected,)
        selected_batch = batch_indices.masked_select(edge_mask_noise_flat)  # (num_selected,)

        # Initialize edge_mask_noise as all False
        edge_mask_noise = torch.zeros((batch_size, num_nodes, num_nodes), dtype=torch.bool, device=device)

        # Set selected edges in upper triangle
        edge_mask_noise[selected_batch, selected_rows, selected_cols] = True
        # Ensure symmetry by setting corresponding lower triangle edges
        edge_mask_noise[selected_batch, selected_cols, selected_rows] = True
      
        # Retrieve transition matrices for nodes and edges
        # Assuming Qtb.X has shape (dx_in, dx_out) and Qtb.E has shape (de_in, de_out)
        Qtb = self.transition_model.get_discrete_Qt_bar(device=device)  # Adjust as per your implementation

        # Compute transition probabilities for nodes
        # X: (batch_size, num_nodes, dx_in)
        # Qtb.X: (dx_in, dx_out)
        probX = torch.matmul(X, Qtb.X)  # Shape: (batch_size, num_nodes, dx_out)

        # Compute transition probabilities for edges
        # E: (batch_size, num_nodes, num_nodes, de_in)
        # Qtb.E: (de_in, de_out)
        probE = torch.matmul(E, Qtb.E)  # Shape: (batch_size, num_nodes, num_nodes, de_out)

        current_X = X.argmax(dim=-1)  # Shape: (batch_size, num_nodes)
        current_E = E.argmax(dim=-1)  # Shape: (batch_size, num_nodes, num_nodes)

        # make sure that the selected nodes will not be stable
        probX_selected = probX.clone()

        probX_selected[node_mask_noise] = probX_selected[node_mask_noise].scatter(
            dim=-1,
            index=current_X[node_mask_noise].unsqueeze(-1),
            value=0
        )
        # normalize
        probX_selected[node_mask_noise] = probX_selected[node_mask_noise] / probX_selected[node_mask_noise].sum(dim=-1, keepdim=True)

        # make sure that the selected edges will not be stable
        probE_selected = probE.clone()
        probE_selected[edge_mask_noise] = probE_selected[edge_mask_noise].scatter(
            dim=-1,
            index=current_E[edge_mask_noise].unsqueeze(-1),
            value=0
        )
        # normalize
        probE_selected[edge_mask_noise] = probE_selected[edge_mask_noise] / probE_selected[edge_mask_noise].sum(dim=-1, keepdim=True)
        

        sampled = diffusion_utils.sample_discrete_features(self.limit_dist, probX_selected, probE_selected, node_mask)

        X_t = sampled.X  # Shape: (batch_size, num_nodes)
        E_t = sampled.E  # Shape: (batch_size, num_nodes, num_nodes)


        X_t_final = X.clone()
        E_t_final = E.clone()

        
        X_t_final[node_mask_noise] = F.one_hot(X_t[node_mask_noise], num_classes=self.Xdim_output).float()


        E_t_final[edge_mask_noise] = F.one_hot(E_t[edge_mask_noise], num_classes=self.Edim_output).float()

        z_t = utils.PlaceHolder(X=X_t_final, E=E_t_final, y=y).type_as(X_t_final).mask(node_mask)
        noisy_data = {
        't_int': t_nodes,
        't_nodes': t_nodes.float() / num_nodes,          # Shape: (batch_size,)
        't_edges': t_edges.float() / num_edges,          # Shape: (batch_size,)
        'X_t': z_t.X,                        # Shape: (batch_size, num_nodes, dx_out)
        'E_t': z_t.E,                        # Shape: (batch_size, num_nodes, num_nodes, de_out)
        'y_t': z_t.y,                        
        'node_mask': node_mask               # Shape: (batch_size, num_nodes)
    }

        return noisy_data

    def apply_noise010101(self, X, E, y, node_mask): 
      
        batch_size, num_nodes, _ = X.size()
        device = X.device

        valid_nodes_per_graph = node_mask.sum(dim=1)  
        rand_floats_nodes = torch.rand(batch_size, device=device)
        t_nodes = (rand_floats_nodes * valid_nodes_per_graph.float()).long() + 1  
        rand_nodes = torch.rand(batch_size, num_nodes, device=device)
        rand_nodes[~node_mask] = -float('inf')
        sorted_scores_nodes, sorted_indices_nodes = torch.sort(rand_nodes, dim=1, descending=True)
        range_tensor_nodes = torch.arange(num_nodes, device=device).unsqueeze(0).expand(batch_size, num_nodes)
        mask_nodes = range_tensor_nodes < t_nodes.unsqueeze(1)
        node_mask_noise = torch.zeros_like(mask_nodes, dtype=torch.bool)
        node_mask_noise.scatter_(1, sorted_indices_nodes, mask_nodes)
        node_mask_expanded = node_mask.unsqueeze(1) & node_mask.unsqueeze(2)  
        triu_indices = torch.triu_indices(num_nodes, num_nodes, offset=1, device=device)  
        valid_edge_mask_upper = node_mask_expanded[:, triu_indices[0], triu_indices[1]]
        valid_edges_per_graph = valid_edge_mask_upper.sum(dim=1)
        ratio = t_nodes.float() / valid_nodes_per_graph.float()  
        t_edges = (ratio * valid_edges_per_graph.float()).long()
        t_edges = torch.clamp(t_edges, min=torch.tensor(0, device=device), max=valid_edges_per_graph)  
        rand_edges = torch.rand(batch_size, triu_indices.size(1), device=device)
        rand_edges[~valid_edge_mask_upper] = -float('inf')
        sorted_scores_edges, sorted_indices_edges = torch.sort(rand_edges, dim=1, descending=True)
        range_tensor_edges = torch.arange(triu_indices.size(1), device=device).unsqueeze(0).expand(batch_size, triu_indices.size(1))

        # 创建掩码，选择前 t_edges 条边
        mask_edges = range_tensor_edges < t_edges.unsqueeze(1)

        # 根据排序后的索引和掩码，创建加噪边的掩码
        edge_mask_noise_flat = torch.zeros_like(mask_edges, dtype=torch.bool)
        edge_mask_noise_flat.scatter_(1, sorted_indices_edges, mask_edges)

        # 扩展 triu_indices，形状为 (batch_size, 2, num_edges)
        triu_indices_exp = triu_indices.unsqueeze(0).expand(batch_size, -1, -1)

        # 创建批次索引，形状为 (batch_size, num_edges)
        batch_indices = torch.arange(batch_size, device=device).unsqueeze(1).expand(-1, triu_indices.size(1))

        # 选取被加噪的边的索引
        selected_rows = triu_indices_exp[:, 0, :][edge_mask_noise_flat]  # (num_selected_edges,)
        selected_cols = triu_indices_exp[:, 1, :][edge_mask_noise_flat]  # (num_selected_edges,)
        selected_batch = batch_indices[edge_mask_noise_flat]  # (num_selected_edges,)

        # 初始化边的噪声掩码，形状为 (batch_size, num_nodes, num_nodes)
        edge_mask_noise = torch.zeros((batch_size, num_nodes, num_nodes), dtype=torch.bool, device=device)

        # 设置上三角形中被加噪的边
        edge_mask_noise[selected_batch, selected_rows, selected_cols] = True
        # 确保边的对称性，设置对应的下三角形边
        edge_mask_noise[selected_batch, selected_cols, selected_rows] = True

        # 获取节点和边的状态转移矩阵
        Qtb = self.transition_model.get_discrete_Qt_bar(device=device)  # 根据您的实现进行调整

        # 计算节点的转移概率
        probX = torch.matmul(X, Qtb.X)  # (batch_size, num_nodes, dx_out)

        # 计算边的转移概率
        probE = torch.matmul(E, Qtb.E)  # (batch_size, num_nodes, num_nodes, de_out)

        current_X = X.argmax(dim=-1)  # (batch_size, num_nodes)
        current_E = E.argmax(dim=-1)  # (batch_size, num_nodes, num_nodes)

        # 确保选中的节点不会保持不变
        probX_selected = probX.clone()
        probX_selected[node_mask_noise] = probX_selected[node_mask_noise].scatter_(
            dim=-1,
            index=current_X[node_mask_noise].unsqueeze(-1),
            value=0
        )
        # 归一化
        probX_selected[node_mask_noise] = probX_selected[node_mask_noise] / probX_selected[node_mask_noise].sum(dim=-1, keepdim=True)

        # 确保选中的边不会保持不变
        probE_selected = probE.clone()
        probE_selected[edge_mask_noise] = probE_selected[edge_mask_noise].scatter_(
            dim=-1,
            index=current_E[edge_mask_noise].unsqueeze(-1),
            value=0
        )
        # 归一化
        probE_selected[edge_mask_noise] = probE_selected[edge_mask_noise] / probE_selected[edge_mask_noise].sum(dim=-1, keepdim=True)

        # 采样离散特征
        sampled = diffusion_utils.sample_discrete_features(self.limit_dist, probX_selected, probE_selected, node_mask)

        X_t = sampled.X  # (batch_size, num_nodes)
        E_t = sampled.E  # (batch_size, num_nodes, num_nodes)

        X_t_final = X.clone()
        E_t_final = E.clone()

        # 仅对加噪的节点进行更新
        X_t_final[node_mask_noise] = F.one_hot(X_t[node_mask_noise], num_classes=self.Xdim_output).float()

        # 仅对加噪的边进行更新
        E_t_final[edge_mask_noise] = F.one_hot(E_t[edge_mask_noise], num_classes=self.Edim_output).float()

        z_t = utils.PlaceHolder(X=X_t_final, E=E_t_final, y=y).type_as(X_t_final).mask(node_mask)

        """ print(111)
        print(t_nodes[1])
        print(t_edges[1])
        print(X[1].argmax(dim=-1))
        print(E[1].argmax(dim=-1))
        print(node_mask_noise[1])
        print(edge_mask_noise[1])
        print(probX_selected[1])
        print(probE_selected[1])
        print(X_t[1])
        print(E_t[1])
        print(z_t.X[1].argmax(dim=-1))
        print(z_t.E[1].argmax(dim=-1))
        exit() """
        noisy_data = {
            't_int': t_nodes,
            't_e_int':t_edges,
            't_nodes': t_nodes.unsqueeze(1).float() / valid_nodes_per_graph.unsqueeze(1).float(),  # 形状: (batch_size, 1)
            't_edges': t_edges.unsqueeze(1).float() / (valid_edges_per_graph.unsqueeze(1).float() + 1e-8),  # 形状: (batch_size, 1)
            'X_t': z_t.X,                        # (batch_size, num_nodes, dx_out)
            'E_t': z_t.E,                        # (batch_size, num_nodes, num_nodes, de_out)
            'y_t': z_t.y,
            'node_mask': node_mask               # (batch_size, num_nodes)
        }

        return noisy_data



    def apply_noise10101010(self, X, E, y, node_mask):  # 点和边共同采样, 只focus有效节点构成的子图
        """
        通过随机选择 t 个节点并对这些节点之间的所有边添加噪声，基于节点和边的有效性。

        参数：
            X (torch.Tensor): 节点特征，形状为 (batch_size, num_nodes, dx_in)。
            E (torch.Tensor): 边特征，形状为 (batch_size, num_nodes, num_nodes, de_in)。
            y (torch.Tensor): 标签或其他附加数据。
            node_mask (torch.Tensor): 节点有效性掩码，形状为 (batch_size, num_nodes)。

        返回：
            dict: 包含加噪后的数据和相关信息的字典。
        """
        batch_size, num_nodes, _ = X.size()
        device = X.device

        # 计算每个图的有效节点数，形状为 (batch_size,)
        valid_nodes_per_graph = node_mask.sum(dim=1)  # 有效节点数量

        # 在每个图中，从 1 到 有效节点数 中随机选择 t_nodes，形状为 (batch_size,)
        rand_floats_nodes = torch.rand(batch_size, device=device)
        t_nodes = (rand_floats_nodes * valid_nodes_per_graph.float()).long() + 1
        t_nodes = torch.clamp(t_nodes, min=torch.tensor(1, device=device))  # 保证至少为1

        # 生成节点的随机分数，形状为 (batch_size, num_nodes)
        rand_nodes = torch.rand(batch_size, num_nodes, device=device)
        # 将无效节点的分数设为 -inf，确保排序时排在最后
        rand_nodes[~node_mask] = -float('inf')

        # 对分数进行降序排序，获取排序后的索引
        sorted_scores_nodes, sorted_indices_nodes = torch.sort(rand_nodes, dim=1, descending=True)

        # 创建用于比较的范围张量，形状为 (batch_size, num_nodes)
        range_tensor_nodes = torch.arange(num_nodes, device=device).unsqueeze(0).expand(batch_size, num_nodes)

        # 创建掩码，选择前 t_nodes 个节点，形状为 (batch_size, num_nodes)
        mask_nodes = range_tensor_nodes < t_nodes.unsqueeze(1)

        # 根据排序后的索引和掩码，创建加噪节点的掩码
        node_mask_noise = torch.zeros_like(mask_nodes, dtype=torch.bool)
        node_mask_noise.scatter_(1, sorted_indices_nodes, mask_nodes)

        # -------------------- 修改边的选择和加噪方式 --------------------

        # 计算每个图的 t_edges，形状为 (batch_size,)
        t_edges = t_nodes * (t_nodes - 1) // 2

        # 构建节点之间的连接关系，获取被选中节点之间的所有可能边
        # 首先，创建节点掩码，形状为 (batch_size, num_nodes, 1) 和 (batch_size, 1, num_nodes)
        node_mask_noise_row = node_mask_noise.unsqueeze(2)  # (batch_size, num_nodes, 1)
        node_mask_noise_col = node_mask_noise.unsqueeze(1)  # (batch_size, 1, num_nodes)

        # 计算节点之间的连接关系，形状为 (batch_size, num_nodes, num_nodes)
        edge_mask_noise = node_mask_noise_row & node_mask_noise_col

        # 排除自环边（如果不需要自环边）
        diag_mask = ~torch.eye(num_nodes, dtype=torch.bool, device=device).unsqueeze(0)  # (1, num_nodes, num_nodes)
        edge_mask_noise = edge_mask_noise & diag_mask  # (batch_size, num_nodes, num_nodes)

        # 确保只考虑有效节点之间的边
        node_mask_expanded = node_mask.unsqueeze(1) & node_mask.unsqueeze(2)  # (batch_size, num_nodes, num_nodes)
        edge_mask_noise = edge_mask_noise & node_mask_expanded

        # 计算每个图的有效边数（被选中节点之间的边数）
        valid_edges_per_graph = t_edges  # (batch_size,)

        # -------------------- 结束修改边的选择和加噪方式 --------------------

        # 获取状态转移矩阵
        Qtb = self.transition_model.get_discrete_Qt_bar(device=device)  # 根据您的实现进行调整

        # 计算节点的转移概率
        probX = torch.matmul(X, Qtb.X)  # (batch_size, num_nodes, dx_out)

        # 计算边的转移概率
        probE = torch.matmul(E, Qtb.E)  # (batch_size, num_nodes, num_nodes, de_out)

        current_X = X.argmax(dim=-1)  # (batch_size, num_nodes)
        current_E = E.argmax(dim=-1)  # (batch_size, num_nodes, num_nodes)

        # 确保选中的节点不会保持不变
        probX_selected = probX.clone()
        probX_selected[node_mask_noise] = probX_selected[node_mask_noise].scatter_(
            dim=-1,
            index=current_X[node_mask_noise].unsqueeze(-1),
            value=0
        )
        # 归一化
        probX_selected[node_mask_noise] = probX_selected[node_mask_noise] / probX_selected[node_mask_noise].sum(dim=-1, keepdim=True)

        # 确保选中的边不会保持不变
        probE_selected = probE.clone()
        """ probE_selected[edge_mask_noise] = probE_selected[edge_mask_noise].scatter_(
            dim=-1,
            index=current_E[edge_mask_noise].unsqueeze(-1),
            value=0
        )
        # 归一化
        probE_selected[edge_mask_noise] = probE_selected[edge_mask_noise] / probE_selected[edge_mask_noise].sum(dim=-1, keepdim=True) """

        # 采样离散特征
        sampled = diffusion_utils.sample_discrete_features(self.limit_dist, probX_selected, probE_selected, node_mask)

        X_t = sampled.X  # (batch_size, num_nodes)
        E_t = sampled.E  # (batch_size, num_nodes, num_nodes)

        X_t_final = X.clone()
        E_t_final = E.clone()

        # 仅对加噪的节点进行更新
        X_t_final[node_mask_noise] = F.one_hot(X_t[node_mask_noise], num_classes=self.Xdim_output).float()

        # 仅对加噪的边进行更新
        E_t_final[edge_mask_noise] = F.one_hot(E_t[edge_mask_noise], num_classes=self.Edim_output).float()

        z_t = utils.PlaceHolder(X=X_t_final, E=E_t_final, y=y).type_as(X_t_final).mask(node_mask)

        """ print(111)
        print(t_nodes[511])
        print(t_edges[511])
        #print(X[511].argmax(dim=-1))
        print(E[511])
        print(node_mask[511])
        print(node_mask_noise[511])
        print(edge_mask_noise[511])
        #print(probX_selected[511])
        print(probE_selected[511])
        #print(X_t[511])
        print(E_t[511])
        #print(z_t.X[511].argmax(dim=-1))
        print(z_t.E[511])
        exit() """

        noisy_data = {
            't_int': t_nodes,
            't_e_int': t_edges,
            't_nodes': t_nodes.unsqueeze(1).float() / valid_nodes_per_graph.unsqueeze(1).float(),  # 形状: (batch_size, 1)
            't_edges': t_edges.unsqueeze(1).float() / (valid_edges_per_graph.unsqueeze(1).float() + 1e-8),  # 形状: (batch_size, 1)
            'X_t': z_t.X,                        # (batch_size, num_nodes, dx_out)
            'E_t': z_t.E,                        # (batch_size, num_nodes, num_nodes, de_out)
            'y_t': z_t.y,
            'node_mask': node_mask               # (batch_size, num_nodes)
        }

        return noisy_data

    def apply_noise0101001(self, X, E, y, node_mask):  # 点和边共同采样, 只focus有效节点构成的子图，以cos形式选择需要改变的节点数量
        """
        通过根据指定的比例选择 t 个节点并对这些节点之间的所有边添加噪声，基于节点和边的有效性。

        参数：
            X (torch.Tensor): 节点特征，形状为 (batch_size, num_nodes, dx_in)。
            E (torch.Tensor): 边特征，形状为 (batch_size, num_nodes, num_nodes, de_in)。
            y (torch.Tensor): 标签或其他附加数据。
            node_mask (torch.Tensor): 节点有效性掩码，形状为 (batch_size, num_nodes)。

        返回：
            dict: 包含加噪后的数据和相关信息的字典。
        """
        batch_size, num_nodes, _ = X.size()
        device = X.device

        # -------------------- 计算节点选择比例 --------------------
        # 从 0 到 1 的均匀分布中随机采样 n_over_m，形状为 (batch_size,)
        n_over_m = torch.rand(batch_size, device=device)  # 每个图都有自己的 n_over_m

        # 计算 ratio = (1 - cos(0.5π * ((n/m + 0.008)/(1 + 0.008)))^2)
        ratio = (1 - torch.cos(0.5 * math.pi * ((n_over_m + 0.008) / (1 + 0.008))) ** 2)  # ratio 为一个标量

        # -------------------- 选择节点并添加噪声 --------------------

        # 计算每个图的有效节点数，形状为 (batch_size,)
        valid_nodes_per_graph = node_mask.sum(dim=1)  # 有效节点数量

        # 计算每个图需要选择的节点数量 t_nodes，确保在有效范围内
        t_nodes = (ratio * valid_nodes_per_graph.float()).long() + 1
        t_nodes = torch.clamp(t_nodes, min=torch.tensor(1, device=device), max=valid_nodes_per_graph)  # 保证至少为1，至多为有效节点数

        # 生成节点的随机分数，形状为 (batch_size, num_nodes)
        rand_nodes = torch.rand(batch_size, num_nodes, device=device)
        # 将无效节点的分数设为 -inf，确保排序时排在最后
        rand_nodes[~node_mask] = -float('inf')

        # 对分数进行降序排序，获取排序后的索引
        sorted_scores_nodes, sorted_indices_nodes = torch.sort(rand_nodes, dim=1, descending=True)

        # 创建用于比较的范围张量，形状为 (batch_size, num_nodes)
        range_tensor_nodes = torch.arange(num_nodes, device=device).unsqueeze(0).expand(batch_size, num_nodes)

        # 创建掩码，选择前 t_nodes 个节点，形状为 (batch_size, num_nodes)
        mask_nodes = range_tensor_nodes < t_nodes.unsqueeze(1)

        # 根据排序后的索引和掩码，创建加噪节点的掩码 node_mask_noise
        node_mask_noise = torch.zeros_like(mask_nodes, dtype=torch.bool)
        node_mask_noise.scatter_(1, sorted_indices_nodes, mask_nodes)

        # -------------------- 选择边并添加噪声 --------------------

        # 计算每个图的 t_edges，形状为 (batch_size,)
        t_edges = t_nodes * (t_nodes - 1) // 2

        # 构建被选中节点之间的连接关系，获取所有可能的边
        node_mask_noise_row = node_mask_noise.unsqueeze(2)  # (batch_size, num_nodes, 1)
        node_mask_noise_col = node_mask_noise.unsqueeze(1)  # (batch_size, 1, num_nodes)

        # 计算节点之间的连接关系 edge_mask_noise，形状为 (batch_size, num_nodes, num_nodes)
        edge_mask_noise = node_mask_noise_row & node_mask_noise_col

        # 排除自环边（如果不需要自环边）
        diag_mask = ~torch.eye(num_nodes, dtype=torch.bool, device=device).unsqueeze(0)  # (1, num_nodes, num_nodes)
        edge_mask_noise = edge_mask_noise & diag_mask  # (batch_size, num_nodes, num_nodes)

        # 确保只考虑有效节点之间的边
        node_mask_expanded = node_mask.unsqueeze(1) & node_mask.unsqueeze(2)  # (batch_size, num_nodes, num_nodes)
        edge_mask_noise = edge_mask_noise & node_mask_expanded

        # 计算每个图的有效边数（被选中节点之间的边数）
        valid_edges_per_graph = t_edges  # (batch_size,)

        # -------------------- 计算转移概率并添加噪声 --------------------
        # 获取状态转移矩阵
        Qtb = self.transition_model.get_discrete_Qt_bar(device=device)  # 根据您的实现进行调整

        # 计算节点的转移概率
        probX = torch.matmul(X, Qtb.X)  # (batch_size, num_nodes, dx_out)

        # 计算边的转移概率
        probE = torch.matmul(E, Qtb.E)  # (batch_size, num_nodes, num_nodes, de_out)

        current_X = X.argmax(dim=-1)  # (batch_size, num_nodes)
        current_E = E.argmax(dim=-1)  # (batch_size, num_nodes, num_nodes)

        # 确保选中的节点不会保持不变
        probX_selected = probX.clone()
        probX_selected[node_mask_noise] = probX_selected[node_mask_noise].scatter_(
            dim=-1,
            index=current_X[node_mask_noise].unsqueeze(-1),
            value=0
        )
        # 归一化
        probX_selected[node_mask_noise] = probX_selected[node_mask_noise] / probX_selected[node_mask_noise].sum(dim=-1, keepdim=True)

        # 确保选中的边不会保持不变
        probE_selected = probE.clone()
        probE_selected[edge_mask_noise] = probE_selected[edge_mask_noise].scatter_(
            dim=-1,
            index=current_E[edge_mask_noise].unsqueeze(-1),
            value=0
        )
        # 归一化
        probE_selected[edge_mask_noise] = probE_selected[edge_mask_noise] / probE_selected[edge_mask_noise].sum(dim=-1, keepdim=True)
        # -------------------- 采样并更新节点和边特征 --------------------

        # 采样离散特征
        sampled = diffusion_utils.sample_discrete_features(self.limit_dist, probX_selected, probE_selected, node_mask)

        X_t = sampled.X  # (batch_size, num_nodes)
        E_t = sampled.E  # (batch_size, num_nodes, num_nodes)

        X_t_final = X.clone()
        E_t_final = E.clone()

        # 仅对加噪的节点进行更新
        X_t_final[node_mask_noise] = F.one_hot(X_t[node_mask_noise], num_classes=self.Xdim_output).float()

        # 仅对加噪的边进行更新
        E_t_final[edge_mask_noise] = F.one_hot(E_t[edge_mask_noise], num_classes=self.Edim_output).float()

        # 创建占位符并应用节点掩码
        z_t = utils.PlaceHolder(X=X_t_final, E=E_t_final, y=y).type_as(X_t_final).mask(node_mask)

        # -------------------- 返回加噪后的数据 --------------------

        noisy_data = {
            't_int': t_nodes,
            't_e_int': t_edges,
            't_nodes': t_nodes.unsqueeze(1).float() / valid_nodes_per_graph.unsqueeze(1).float(),  # 形状: (batch_size, 1)
            't_edges': t_edges.unsqueeze(1).float() / (valid_edges_per_graph.unsqueeze(1).float() + 1e-8),  # 形状: (batch_size, 1)
            'X_t': z_t.X,                        # (batch_size, num_nodes, dx_out)
            'E_t': z_t.E,                        # (batch_size, num_nodes, num_nodes, de_out)
            'y_t': z_t.y,
            'node_mask': node_mask               # (batch_size, num_nodes)
        }

        return noisy_data

    def apply_noise452234(self, X, E, y, node_mask):  # 点和边共同采样, 只focus有效节点之间的子图，按比例选边
        """
        通过随机选择 t 个节点并对这些节点之间的部分边添加噪声，基于节点和边的有效性。

        参数：
            X (torch.Tensor): 节点特征，形状为 (batch_size, num_nodes, dx_in)。
            E (torch.Tensor): 边特征，形状为 (batch_size, num_nodes, num_nodes, de_in)。
            y (torch.Tensor): 标签或其他附加数据。
            node_mask (torch.Tensor): 节点有效性掩码，形状为 (batch_size, num_nodes)。

        返回：
            dict: 包含加噪后的数据和相关信息的字典。
        """
        batch_size, num_nodes, _ = X.size()
        device = X.device


        # -------------------- 计算节点选择比例 --------------------
        # 从 0 到 1 的均匀分布中随机采样 n_over_m，形状为 (batch_size,)
        n_over_m = torch.rand(batch_size, device=device)  # 每个图都有自己的 n_over_m

        # 计算 ratio = (1 - cos(0.5π * ((n/m + 0.008)/(1 + 0.008)))^2)
        ratio = (1 - torch.cos(0.5 * math.pi * ((n_over_m + 0.008) / (1 + 0.008))) ** 2)  # ratio 为一个标量 

        # -------------------- 选择节点并添加噪声 --------------------

        # 计算每个图的有效节点数，形状为 (batch_size,)
        valid_nodes_per_graph = node_mask.sum(dim=1)  # 有效节点数量

        # 计算每个图需要选择的节点数量 t_nodes，确保在有效范围内
        t_nodes = (ratio * valid_nodes_per_graph.float()).long() + 1
        t_nodes = torch.clamp(t_nodes, min=torch.tensor(1, device=device), max=valid_nodes_per_graph)  # 保证至少为1，至多为有效节点数



        """ # 计算每个图的有效节点数，形状为 (batch_size,)
        valid_nodes_per_graph = node_mask.sum(dim=1)  # 有效节点数量

        # 在每个图中，从 1 到 有效节点数 中随机选择 t_nodes，形状为 (batch_size,)
        rand_floats_nodes = torch.rand(batch_size, device=device)
        t_nodes = (rand_floats_nodes * valid_nodes_per_graph.float()).long() + 1
        t_nodes = torch.clamp(t_nodes, torch.tensor(1, device=device))  # 保证至少为1  """



        # 生成节点的随机分数，形状为 (batch_size, num_nodes)
        rand_nodes = torch.rand(batch_size, num_nodes, device=device)
        # 将无效节点的分数设为 -inf，确保排序时排在最后
        rand_nodes[~node_mask] = -float('inf')

        # 对分数进行降序排序，获取排序后的索引
        sorted_scores_nodes, sorted_indices_nodes = torch.sort(rand_nodes, dim=1, descending=True)

        # 创建用于比较的范围张量，形状为 (batch_size, num_nodes)
        range_tensor_nodes = torch.arange(num_nodes, device=device).unsqueeze(0).expand(batch_size, num_nodes)

        # 创建掩码，选择前 t_nodes 个节点，形状为 (batch_size, num_nodestimes = 3)
        mask_nodes = range_tensor_nodes < t_nodes.unsqueeze(1)

        # 根据排序后的索引和掩码，创建加噪节点的掩码
        node_mask_noise = torch.zeros_like(mask_nodes, dtype=torch.bool)
        node_mask_noise.scatter_(1, sorted_indices_nodes, mask_nodes)

        # -------------------- 修改边的选择和加噪方式 --------------------

        # 构建节点之间的连接关系，获取被选中节点之间的所有可能边
        node_mask_noise_row = node_mask_noise.unsqueeze(2)  # (batch_size, num_nodes, 1)
        node_mask_noise_col = node_mask_noise.unsqueeze(1)  # (batch_size, 1, num_nodes)

        # 计算节点之间的连接关系，形状为 (batch_size, num_nodes, num_nodes)
        potential_edge_mask = node_mask_noise_row & node_mask_noise_col

        # 排除自环边（如果不需要自环边）
        diag_mask = ~torch.eye(num_nodes, dtype=torch.bool, device=device).unsqueeze(0)  # (1, num_nodes, num_nodes)
        potential_edge_mask = potential_edge_mask & diag_mask  # (batch_size, num_nodes, num_nodes)

        # 确保只考虑有效节点之间的边
        node_mask_expanded = node_mask.unsqueeze(1) & node_mask.unsqueeze(2)  # (batch_size, num_nodes, num_nodes)
        potential_edge_mask = potential_edge_mask & node_mask_expanded

        # 获取上三角形（不包括对角线）的索引
        triu_indices = torch.triu_indices(num_nodes, num_nodes, offset=1, device=device)  # (2, num_edges)

        # 获取每个图中潜在加噪边的掩码，形状为 (batch_size, num_edges)
        potential_edge_mask_upper = potential_edge_mask[:, triu_indices[0], triu_indices[1]]  # (batch_size, num_edges)

        # 从潜在加噪边中随机选择一部分边来添加噪声
        edge_noise_ratio = 0.2  # 控制边噪声的比例，可根据需要调整（0到1之间）
        rand_edges = torch.rand(batch_size, triu_indices.size(1), device=device)
        rand_edges[~potential_edge_mask_upper] = 2.0  # 将不可用的边的随机数设为大于1的值，确保不会被选中

        # 根据 edge_noise_ratio 选择加噪边
        edge_threshold = torch.quantile(rand_edges, edge_noise_ratio, dim=1, keepdim=True)
        edge_mask_noise_flat = rand_edges <= edge_threshold

        # 确保只选择潜在的边
        edge_mask_noise_flat = edge_mask_noise_flat & potential_edge_mask_upper

        # 扩展 triu_indices，形状为 (batch_size, 2, num_edges)
        triu_indices_exp = triu_indices.unsqueeze(0).expand(batch_size, -1, -1)

        # 创建批次索引，形状为 (batch_size, num_edges)
        batch_indices = torch.arange(batch_size, device=device).unsqueeze(1).expand(-1, triu_indices.size(1))

        # 选取被加噪的边的索引
        selected_rows = triu_indices_exp[:, 0, :][edge_mask_noise_flat]  # (num_selected_edges,)
        selected_cols = triu_indices_exp[:, 1, :][edge_mask_noise_flat]  # (num_selected_edges,)
        selected_batch = batch_indices[edge_mask_noise_flat]  # (num_selected_edges,)

        # 初始化边的噪声掩码，形状为 (batch_size, num_nodes, num_nodes)
        edge_mask_noise = torch.zeros((batch_size, num_nodes, num_nodes), dtype=torch.bool, device=device)

        # 设置上三角形中被加噪的边
        edge_mask_noise[selected_batch, selected_rows, selected_cols] = True
        # 确保边的对称性，设置对应的下三角形边
        edge_mask_noise[selected_batch, selected_cols, selected_rows] = True

        # 计算每个图的 t_edges，形状为 (batch_size,)
        t_edges = edge_mask_noise_flat.sum(dim=1)

        # 计算每个图的有效边数（被选中节点之间的边数）
        valid_edges_per_graph = potential_edge_mask_upper.sum(dim=1)  # (batch_size,) 

        """ # 计算可能的边数
        possible_edges_count = potential_edge_mask_upper.sum(dim=1)  # (batch_size,)

        # 按0.1比例选边数量，并下取整
        num_selected_edges = (possible_edges_count.float() * edge_noise_ratio).floor().long()  # (batch_size,)

        # 对rand_edges进行排序，挑选出前num_selected_edges[i]个值最低的边
        idx = rand_edges.argsort(dim=1)  # (batch_size, num_edges)，排序后每行是从小到大

        # 构建一个mask，用于挑选每个图的前num_selected_edges[i]个边
        num_edges = rand_edges.size(1)

        take_mask = torch.arange(num_edges, device=device).unsqueeze(0) < num_selected_edges.unsqueeze(1)
        # take_mask shape: (batch_size, num_edges)

        # 利用 take_mask 从 idx 中选择对应的边的索引
        chosen_indices = idx[take_mask]  # 这将是一维张量，包含所有图的选中边索引

        # 同时需要确定这些选中边属于哪个图
        batch_arange = torch.arange(batch_size, device=device).unsqueeze(1).expand(-1, num_edges)
        selected_batch = batch_arange[take_mask]  # (num_selected_selected_edges,)

        # 为了正确映射回原图中的行列索引，需要利用 selected_batch 和 chosen_indices
        triu_indices_exp = triu_indices.unsqueeze(0).expand(batch_size, -1, -1)  # (batch_size, 2, num_edges)

        # 根据 selected_batch 和 chosen_indices 获取对应的 row 和 col
        selected_rows = triu_indices_exp[selected_batch, 0, chosen_indices]
        selected_cols = triu_indices_exp[selected_batch, 1, chosen_indices]

        # 初始化边的噪声掩码
        edge_mask_noise = torch.zeros((batch_size, num_nodes, num_nodes), dtype=torch.bool, device=device)

        # 设置被选中的边
        edge_mask_noise[selected_batch, selected_rows, selected_cols] = True
        edge_mask_noise[selected_batch, selected_cols, selected_rows] = True

        # 计算每个图的 t_edges，即选中边的数量
        t_edges = num_selected_edges  # (batch_size,)"""
        # -------------------- 结束修改边的选择和加噪方式 --------------------

        # 获取状态转移矩阵
        Qtb = self.transition_model.get_discrete_Qt_bar(device=device)  # 根据您的实现进行调整

        # 计算节点的转移概率
        probX = torch.matmul(X, Qtb.X)  # (batch_size, num_nodes, dx_out)

        # 计算边的转移概率
        probE = torch.matmul(E, Qtb.E)  # (batch_size, num_nodes, num_nodes, de_out)

        current_X = X.argmax(dim=-1)  # (batch_size, num_nodes)
        current_E = E.argmax(dim=-1)  # (batch_size, num_nodes, num_nodes)

        # 确保选中的节点不会保持不变
        probX_selected = probX.clone()
        probX_selected[node_mask_noise] = probX_selected[node_mask_noise].scatter_(
            dim=-1,
            index=current_X[node_mask_noise].unsqueeze(-1),
            value=0
        )
        # 归一化
        probX_selected[node_mask_noise] = probX_selected[node_mask_noise] / probX_selected[node_mask_noise].sum(dim=-1, keepdim=True)

        # 确保选中的边不会保持不变
        probE_selected = probE.clone()
        probE_selected[edge_mask_noise] = probE_selected[edge_mask_noise].scatter_(
            dim=-1,
            index=current_E[edge_mask_noise].unsqueeze(-1),
            value=0
        )
        # 归一化
        probE_selected[edge_mask_noise] = probE_selected[edge_mask_noise] / probE_selected[edge_mask_noise].sum(dim=-1, keepdim=True)

        # 采样离散特征
        sampled = diffusion_utils.sample_discrete_features(self.limit_dist, probX_selected, probE_selected, node_mask)

        X_t = sampled.X  # (batch_size, num_nodes)
        E_t = sampled.E  # (batch_size, num_nodes, num_nodes)

        X_t_final = X.clone()
        E_t_final = E.clone()

        # 仅对加噪的节点进行更新
        X_t_final[node_mask_noise] = F.one_hot(X_t[node_mask_noise], num_classes=self.Xdim_output).float()

        # 仅对加噪的边进行更新
        E_t_final[edge_mask_noise] = F.one_hot(E_t[edge_mask_noise], num_classes=self.Edim_output).float()

        z_t = utils.PlaceHolder(X=X_t_final, E=E_t_final, y=y).type_as(X_t_final).mask(node_mask)


        max_possible_edges = (valid_nodes_per_graph * (valid_nodes_per_graph - 1)) // 2

        noisy_data = {
            't_int': t_nodes,
            't_e_int': t_edges,
            't_nodes': t_nodes.unsqueeze(1).float() / valid_nodes_per_graph.unsqueeze(1).float(),  # 形状: (batch_size, 1)
            #'t_edges': t_edges.unsqueeze(1).float() / (valid_edges_per_graph.unsqueeze(1).float() + 1e-8),  # 形状: (batch_size, 1)
            't_edges': t_edges.unsqueeze(1).float() / (max_possible_edges.unsqueeze(1).float() + 1e-8),  # 形状: (batch_size, 1)
            'X_t': z_t.X,                        # (batch_size, num_nodes, dx_out)
            'E_t': z_t.E,                        # (batch_size, num_nodes, num_nodes, de_out)
            'y_t': z_t.y,
            'node_mask': node_mask               # (batch_size, num_nodes)
        }

        return noisy_data

    import torch

    def apply_noise(self, X, E, y, node_mask):  # 点和边共同采样, 只focus有效节点之间的子图，按比例选边，改变比例

        batch_size, num_nodes, _ = X.size()
        device = X.device

        # 参数times = 2
        times = 2

        n_over_m = torch.rand(batch_size, device=device)

        ratio = (1 - torch.cos(0.5 * math.pi * ((n_over_m + 0.008) / (1 + 0.008))) ** 2)

        valid_nodes_per_graph = node_mask.sum(dim=1)  # 有效节点数量 (batch_size,)
        steps = times * valid_nodes_per_graph.float()  # 扩大times倍的步数
        s = (ratio * steps).long() + 1
        # 真实t_nodes = (s+(times-1))//times，这里times=2，所以 t_nodes=(s+1)//2
        t_nodes = ((s + (times - 1)) // times) 
        
        """ r = s.float() / (steps + 1e-8)
        #cos_val = torch.cos(0.5* math.pi * (r + 0.008) / (1 + 0.008))**2  # (batch_size,)
        #cos_val = r**2
        cos_val = (1.0 - torch.exp(-2*r)) / (1.0 - math.e**(-2))
        t_nodes_float = 1.0 + (valid_nodes_per_graph - 1.0) * (1 - cos_val)
        t_nodes = t_nodes_float.floor().long()"""

        """ n_over_m = torch.rand(batch_size, device=device)
        valid_nodes_per_graph = node_mask.sum(dim=1) 
        cond1 = (valid_nodes_per_graph <= 4)
        cond2 = (valid_nodes_per_graph > 4) & (valid_nodes_per_graph <= 7)
        cond3 = (valid_nodes_per_graph > 7)
        steps = torch.zeros_like(valid_nodes_per_graph)  # (batch_size,)
        steps[cond1] = valid_nodes_per_graph[cond1]
        steps[cond2] = 4 + (valid_nodes_per_graph[cond2] - 4) * 2
        steps[cond3] = 10 + (valid_nodes_per_graph[cond3] - 7) * 4
        s = (n_over_m * steps.float()).long() + 1           # (batch_size,)
        t_nodes = torch.zeros_like(s)  # (batch_size,)
        segA_mask = (s <= 4)
        t_nodes[segA_mask] = s[segA_mask]
        segB_mask = (s > 4) & (s <= 10)
        offsetB = (s[segB_mask] - 4)
        t_nodes[segB_mask] = 4 + (offsetB + 1) // 2 
        segC_mask = (s > 10)
        offsetC = (s[segC_mask] - 10)
        t_nodes[segC_mask] = 7 + (offsetC + 3) // 4 """

        t_nodes = torch.clamp(t_nodes, min=torch.tensor(1, device=device), max=valid_nodes_per_graph)

        # 根据 t_nodes 选择节点（原逻辑不变）
        rand_nodes = torch.rand(batch_size, num_nodes, device=device)
        rand_nodes[~node_mask] = -float('inf')
        sorted_scores_nodes, sorted_indices_nodes = torch.sort(rand_nodes, dim=1, descending=True)
        range_tensor_nodes = torch.arange(num_nodes, device=device).unsqueeze(0).expand(batch_size, num_nodes)
        mask_nodes = range_tensor_nodes < t_nodes.unsqueeze(1)
        node_mask_noise = torch.zeros_like(mask_nodes, dtype=torch.bool)
        node_mask_noise.scatter_(1, sorted_indices_nodes, mask_nodes)

        # -------------------- 修改边的选择和加噪方式 --------------------

        node_mask_noise_row = node_mask_noise.unsqueeze(2)
        node_mask_noise_col = node_mask_noise.unsqueeze(1)
        potential_edge_mask = node_mask_noise_row & node_mask_noise_col

        diag_mask = ~torch.eye(num_nodes, dtype=torch.bool, device=device).unsqueeze(0)
        potential_edge_mask = potential_edge_mask & diag_mask

        node_mask_expanded = node_mask.unsqueeze(1) & node_mask.unsqueeze(2)
        potential_edge_mask = potential_edge_mask & node_mask_expanded

        triu_indices = torch.triu_indices(num_nodes, num_nodes, offset=1, device=device)
        potential_edge_mask_upper = potential_edge_mask[:, triu_indices[0], triu_indices[1]]

        #even_s_mask = (s % 2 == 0)
        #odd_s_mask = (s % 2 == 1)
        #edge_noise_ratio = torch.full((batch_size,), 0.2, device=device)
        #edge_noise_ratio[odd_s_mask] = 0.05

        r = s / steps
        r = (1 - torch.cos(0.5 * math.pi * ((r + 0.008) / (1 + 0.008))) ** 2)
        edge_noise_ratio = torch.full((batch_size,), 0.2, device=device) * r


        rand_edges = torch.rand(batch_size, triu_indices.size(1), device=device)
        rand_edges[~potential_edge_mask_upper] = 2.0  # 将不可用的边的随机数设为大于1的值，确保不会被选中

        # 根据 edge_noise_ratio 选择加噪边
        #edge_threshold = torch.quantile(rand_edges, edge_noise_ratio, dim=1, keepdim=True)
        # 使用 quantile 计算量化值，会返回 (batch_size, batch_size)
        Q = torch.quantile(rand_edges, edge_noise_ratio, dim=1)  # 形状: (batch_size, batch_size)
        # 提取对角线元素 Q[i,i]
        diag_idx = torch.arange(batch_size, device=device)
        edge_threshold = Q[diag_idx, diag_idx].unsqueeze(1)  # (batch_size, 1)

        edge_mask_noise_flat = rand_edges <= edge_threshold

        # 确保只选择潜在的边
        edge_mask_noise_flat = edge_mask_noise_flat & potential_edge_mask_upper

        # 扩展 triu_indices，形状为 (batch_size, 2, num_edges)
        triu_indices_exp = triu_indices.unsqueeze(0).expand(batch_size, -1, -1)

        # 创建批次索引，形状为 (batch_size, num_edges)
        batch_indices = torch.arange(batch_size, device=device).unsqueeze(1).expand(-1, triu_indices.size(1))

        
        # 选取被加噪的边的索引
        selected_rows = triu_indices_exp[:, 0, :][edge_mask_noise_flat]  # (num_selected_edges,)
        selected_cols = triu_indices_exp[:, 1, :][edge_mask_noise_flat]  # (num_selected_edges,)
        selected_batch = batch_indices[edge_mask_noise_flat]  # (num_selected_edges,) 

        # 初始化边的噪声掩码，形状为 (batch_size, num_nodes, num_nodes)
        edge_mask_noise = torch.zeros((batch_size, num_nodes, num_nodes), dtype=torch.bool, device=device)
        # 设置上三角形中被加噪的边
        edge_mask_noise[selected_batch, selected_rows, selected_cols] = True
        # 确保边的对称性，设置对应的下三角形边
        edge_mask_noise[selected_batch, selected_cols, selected_rows] = True
        # 计算每个图的 t_edges，形状为 (batch_size,)
        t_edges = edge_mask_noise_flat.sum(dim=1)





        """ # 2) 计算每个图可能的可用边数量
        possible_edges = potential_edge_mask_upper.sum(dim=1)  # (batch_size,)
        # 3) 根据 edge_noise_ratio 计算选中边的个数: floor(possible_edges * edge_noise_ratio)
        selected_edge_count = (possible_edges.float() * edge_noise_ratio).floor().long()  # (batch_size,)
        # 4) 对 rand_edges 每行从小到大排序
        vals, idx = rand_edges.sort(dim=1)  # vals, idx shape 同为 (batch_size, num_edges)
        # vals[i] 是第 i 个图的 排序后随机值, idx[i] 是其对应原列索引
        # 为了一次性在 batch 上处理，构造一个行索引:
        batch_arange = torch.arange(batch_size, device=device).unsqueeze(1).expand(-1, idx.size(1))
        # shape: (batch_size, num_edges)
        # 5) 为每个图选出前 selected_edge_count[i] 个最小值
        # 构造 mask: 
        take_mask = torch.arange(vals.size(1), device=device).unsqueeze(0) < selected_edge_count.unsqueeze(1)
        # take_mask shape: (batch_size, num_edges)
        # 当 take_mask[i,j] = True 表示: 对第 i 个图, 第 j 小的边要被选中
        # 6) 从 idx 中取出被选中的边索引
        chosen_indices = idx[take_mask]             # 一维张量，所有图选中边的列索引
        selected_batch = batch_arange[take_mask]    # 对应的图索引(一维张量), 与 chosen_indices 同长度
        # 7) 映射回 triu_indices_exp
        triu_indices_exp = triu_indices.unsqueeze(0).expand(batch_size, -1, -1)  # (batch_size, 2, num_edges)
        # chosen_indices 中存的是列索引 => row = triu_indices_exp[selected_batch, 0, chosen_indices]
        #                                   col = triu_indices_exp[selected_batch, 1, chosen_indices]
        selected_rows = triu_indices_exp[selected_batch, 0, chosen_indices]
        selected_cols = triu_indices_exp[selected_batch, 1, chosen_indices]
        # 8) 其余不变, 初始化 edge_mask_noise 并赋值
        edge_mask_noise = torch.zeros((batch_size, num_nodes, num_nodes), dtype=torch.bool, device=device)
        edge_mask_noise[selected_batch, selected_rows, selected_cols] = True
        edge_mask_noise[selected_batch, selected_cols, selected_rows] = True
        # 9) 计算选中边数
        t_edges = selected_edge_count  # (batch_size,) """


        Qtb = self.transition_model.get_discrete_Qt_bar(device=device)
        probX = torch.matmul(X, Qtb.X)
        probE = torch.matmul(E, Qtb.E)

        """ Qtnb, Qtsb = self.transition_model.get_discrete_Qtnb_Qtsb_bar(device=device)
        labels = X.argmax(dim=-1)  # [batch_size, n_node]
        mask1 = (labels >= 0) & (labels <= 3)   # 标签 0-3
        mask2 = (labels >= 4) & (labels <= 13)  # 标签 4-13
        mask1 = mask1.unsqueeze(-1).float()  # [batch_size, n_node, 1]
        mask2 = mask2.unsqueeze(-1).float()  # [batch_size, n_node, 1]
        X_mask1 = X * mask1  # [batch_size, n_node, ndim]
        X_mask2 = X * mask2  # [batch_size, n_node, ndim]
        probX1 = torch.matmul(X_mask1, Qtnb.X)  # [batch_size, n_node, ndim]
        probX2 = torch.matmul(X_mask2, Qtsb.X)  # [batch_size, n_node, ndim]
        probX = probX1 + probX2  # [batch_size, n_node, ndim]
        probE = torch.matmul(E, Qtnb.E)  """

        current_X = X.argmax(dim=-1)
        current_E = E.argmax(dim=-1)

        # 确保选中的节点不会保持不变
        probX_selected = probX.clone()
        if self.Xdim_output > 1:
            probX_selected[node_mask_noise] = probX_selected[node_mask_noise].scatter_(
                dim=-1,
                index=current_X[node_mask_noise].unsqueeze(-1),
                value=0
            )
            probX_selected[node_mask_noise] = probX_selected[node_mask_noise] / probX_selected[node_mask_noise].sum(dim=-1, keepdim=True)

        # 确保选中的边不会保持不变
        probE_selected = probE.clone()
        if self.Edim_output > 1:
            probE_selected[edge_mask_noise] = probE_selected[edge_mask_noise].scatter_(
                dim=-1,
                index=current_E[edge_mask_noise].unsqueeze(-1),
                value=0
            )
            # 归一化
            probE_selected[edge_mask_noise] = probE_selected[edge_mask_noise] / probE_selected[edge_mask_noise].sum(dim=-1, keepdim=True)

        sampled = diffusion_utils.sample_discrete_features(self.dataset_name, self.limit_dist, probX_selected, probE_selected, node_mask)
        X_t = sampled.X
        E_t = sampled.E

        X_t_final = X.clone()
        E_t_final = E.clone()

        X_t_final[node_mask_noise] = F.one_hot(X_t[node_mask_noise], num_classes=self.Xdim_output).float()
        E_t_final[edge_mask_noise] = F.one_hot(E_t[edge_mask_noise], num_classes=self.Edim_output).float()

        z_t = utils.PlaceHolder(X=X_t_final, E=E_t_final, y=y).type_as(X_t_final).mask(node_mask)

        max_possible_edges = (valid_nodes_per_graph * (valid_nodes_per_graph - 1)) // 2
        #max_possible_edges = (t_nodes * (t_nodes - 1)) // 2


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


    def compute_connected_components_batch(self, adjacency_matrix, node_mask):
        """
        计算批量图的连通分量。

        参数：
            adjacency_matrix: (batch_size, num_nodes, num_nodes) 的邻接矩阵。
            node_mask: (batch_size, num_nodes) 的节点掩码。

        返回：
            connected_components: (batch_size, num_nodes) 的张量，每个节点的连通分量标签。
        """
        batch_size, num_nodes, _ = adjacency_matrix.size()
        device = adjacency_matrix.device

        # 初始化连通分量标签，每个节点的初始标签为其索引
        labels = torch.arange(num_nodes, device=device).unsqueeze(0).expand(batch_size, -1).clone()

        adjacency_matrix = adjacency_matrix.clone()
        adjacency_matrix = adjacency_matrix * node_mask.unsqueeze(1) * node_mask.unsqueeze(2)

        for _ in range(num_nodes):
            # 传播标签：labels = min(labels, neighbor_labels)
            neighbor_labels = torch.bmm(adjacency_matrix, F.one_hot(labels, num_nodes).float())
            neighbor_labels = torch.where(neighbor_labels > 0, torch.arange(num_nodes, device=device).float(), float('inf'))
            min_neighbor_labels, _ = neighbor_labels.min(dim=2)
            labels = torch.min(labels, min_neighbor_labels.long())

        return labels

    def apply_noise342(self, X, E, y, node_mask):  # 点和边共同采样，只focus有效节点之间的子图，按照边的分数选，确保图的连通性
        """
        通过随机选择 t 个节点并对这些节点之间的部分边添加噪声，基于节点和边的有效性。
        同时，调整边的加噪策略，避免破坏图的连通性。

        参数：
            X (torch.Tensor): 节点特征，形状为 (batch_size, num_nodes, dx_in)。
            E (torch.Tensor): 边特征，形状为 (batch_size, num_nodes, num_nodes, de_in)。
            y (torch.Tensor): 标签或其他附加数据。
            node_mask (torch.Tensor): 节点有效性掩码，形状为 (batch_size, num_nodes)。

        返回：
            dict: 包含加噪后的数据和相关信息的字典。
        """
        batch_size, num_nodes, _ = X.size()
        device = X.device

        # -------------------- 计算节点选择比例 --------------------
        # 从 0 到 1 的均匀分布中随机采样 n_over_m，形状为 (batch_size,)
        n_over_m = torch.rand(batch_size, device=device)  # 每个图都有自己的 n_over_m
        # 计算 ratio = (1 - cos(0.5π * ((n/m + 0.008)/(1 + 0.008)))^2)
        ratio = (1 - torch.cos(0.5 * math.pi * ((n_over_m + 0.008) / (1 + 0.008))) ** 2)  # (batch_size,)
        # -------------------- 选择节点并添加噪声 --------------------
        # 计算每个图的有效节点数，形状为 (batch_size,)
        valid_nodes_per_graph = node_mask.sum(dim=1)  # 有效节点数量
        # 计算每个图需要选择的节点数量 t_nodes，确保在有效范围内
        t_nodes = (ratio * valid_nodes_per_graph.float()).long() + 1
        t_nodes = torch.clamp(t_nodes, min=torch.tensor(1, device=device), max=valid_nodes_per_graph)  # 保证至少为1，至多为有效节点数

        # 生成节点的随机分数，形状为 (batch_size, num_nodes)
        rand_nodes = torch.rand(batch_size, num_nodes, device=device)
        # 将无效节点的分数设为 -inf，确保排序时排在最后
        rand_nodes[~node_mask] = -float('inf')

        # 对分数进行降序排序，获取排序后的索引
        sorted_scores_nodes, sorted_indices_nodes = torch.sort(rand_nodes, dim=1, descending=True)

        # 创建用于比较的范围张量，形状为 (batch_size, num_nodes)
        range_tensor_nodes = torch.arange(num_nodes, device=device).unsqueeze(0).expand(batch_size, num_nodes)

        # 创建掩码，选择前 t_nodes 个节点，形状为 (batch_size, num_nodes)
        mask_nodes = range_tensor_nodes < t_nodes.unsqueeze(1)

        # 根据排序后的索引和掩码，创建加噪节点的掩码
        node_mask_noise = torch.zeros_like(mask_nodes, dtype=torch.bool)
        node_mask_noise.scatter_(1, sorted_indices_nodes, mask_nodes)

        # -------------------- 修改边的选择和加噪方式 --------------------

        # 构建节点之间的连接关系，获取被选中节点之间的所有可能边
        node_mask_noise_row = node_mask_noise.unsqueeze(2)  # (batch_size, num_nodes, 1)
        node_mask_noise_col = node_mask_noise.unsqueeze(1)  # (batch_size, 1, num_nodes)

        # 计算节点之间的连接关系，形状为 (batch_size, num_nodes, num_nodes)
        potential_edge_mask = node_mask_noise_row & node_mask_noise_col

        # 排除自环边（如果不需要自环边）
        diag_mask = ~torch.eye(num_nodes, dtype=torch.bool, device=device).unsqueeze(0)  # (1, num_nodes, num_nodes)
        potential_edge_mask = potential_edge_mask & diag_mask  # (batch_size, num_nodes, num_nodes)

        # 确保只考虑有效节点之间的边
        node_mask_expanded = node_mask.unsqueeze(1) & node_mask.unsqueeze(2)  # (batch_size, num_nodes, num_nodes)
        potential_edge_mask = potential_edge_mask & node_mask_expanded

        # -------------------- 基于边的重要性调整边的加噪概率 --------------------

        # 从 E 中获取当前的边类型
        current_E = E.argmax(dim=-1)  # (batch_size, num_nodes, num_nodes)
        adjacency_matrix = (current_E > 0).float()  # (batch_size, num_nodes, num_nodes)

        # 计算节点度数
        degrees = adjacency_matrix.sum(dim=-1)  # (batch_size, num_nodes)

        # 计算边的重要性评分
        degree_i = degrees.unsqueeze(2)  # (batch_size, num_nodes, 1)
        degree_j = degrees.unsqueeze(1)  # (batch_size, 1, num_nodes)
        epsilon = 1e-6  # 防止除以零
        edge_importance = 1 / (degree_i + degree_j - 2 + epsilon)  # (batch_size, num_nodes, num_nodes)

        # 对于潜在的边，计算加噪概率
        edge_noise_ratio = 0.2  # 控制边噪声的总体比例，可根据需要调整

        # 归一化重要性评分，使其最大值为1
        max_importance = torch.amax(edge_importance, dim=(1, 2), keepdim=True)  # (batch_size, 1, 1)
        edge_modify_prob = edge_importance / (max_importance + 1e-8)  # (batch_size, num_nodes, num_nodes)

        # 调整加噪概率
        edge_modify_prob = edge_modify_prob * edge_noise_ratio

        # 只考虑潜在的边
        edge_modify_prob = edge_modify_prob * potential_edge_mask.float()

        # 生成与边相同形状的随机数矩阵
        rand_edges = torch.rand(batch_size, num_nodes, num_nodes, device=device)

        # 确定哪些边需要添加噪声
        edge_mask_noise = rand_edges < edge_modify_prob

        # 确保对称性
        edge_mask_noise = edge_mask_noise.triu(1)
        edge_mask_noise = edge_mask_noise | edge_mask_noise.transpose(1, 2)

        # 计算每个图的 t_edges，形状为 (batch_size,)
        t_edges = edge_mask_noise.sum(dim=(1, 2)) // 2  # 每条边计算两次，需要除以2

        # 计算每个图的有效边数（被选中节点之间的边数）
        valid_edges_per_graph = potential_edge_mask.sum(dim=(1, 2)) // 2  # (batch_size,)

        # -------------------- 结束修改边的选择和加噪方式 --------------------



        """ current_E = E.argmax(dim=-1)
        adjacency_matrix = (current_E > 0).float()
        adjacency_matrix = adjacency_matrix * node_mask.unsqueeze(1) * node_mask.unsqueeze(2)

        connected_components = self.compute_connected_components_batch(adjacency_matrix, node_mask)

        comp_i = connected_components.unsqueeze(2)
        comp_j = connected_components.unsqueeze(1)
        different_component = comp_i != comp_j

        base_edge_prob = 0.2
        increased_edge_prob = base_edge_prob * 2  # 根据需要调整

        edge_modify_prob = torch.full((batch_size, num_nodes, num_nodes), base_edge_prob, device=device)
        edge_modify_prob = torch.where(different_component & potential_edge_mask, increased_edge_prob, edge_modify_prob)
        edge_modify_prob = torch.clamp(edge_modify_prob, 0.0, 1.0)

        rand_edges = torch.rand(batch_size, num_nodes, num_nodes, device=device)
        edge_mask_noise = rand_edges < edge_modify_prob
        edge_mask_noise = edge_mask_noise & potential_edge_mask
        edge_mask_noise = edge_mask_noise.triu(1)
        edge_mask_noise = edge_mask_noise | edge_mask_noise.transpose(1, 2)

        # 计算每个图的 t_edges，形状为 (batch_size,)
        t_edges = edge_mask_noise.sum(dim=(1, 2)) // 2  # 每条边计算两次，需要除以2

        # 计算每个图的有效边数（被选中节点之间的边数）
        valid_edges_per_graph = potential_edge_mask.sum(dim=(1, 2)) // 2  # (batch_size,) """



        # 获取状态转移矩阵
        Qtb = self.transition_model.get_discrete_Qt_bar(device=device)  # 根据您的实现进行调整

        # 计算节点的转移概率
        probX = torch.matmul(X, Qtb.X)  # (batch_size, num_nodes, dx_out)

        # 计算边的转移概率
        probE = torch.matmul(E, Qtb.E)  # (batch_size, num_nodes, num_nodes, de_out)

        current_X = X.argmax(dim=-1)  # (batch_size, num_nodes)

        # 确保选中的节点不会保持不变
        probX_selected = probX.clone()
        probX_selected[node_mask_noise] = probX_selected[node_mask_noise].scatter_(
            dim=-1,
            index=current_X[node_mask_noise].unsqueeze(-1),
            value=0
        )
        # 归一化
        probX_selected[node_mask_noise] = probX_selected[node_mask_noise] / probX_selected[node_mask_noise].sum(dim=-1, keepdim=True)

        # 确保选中的边不会保持不变
        probE_selected = probE.clone()
        probE_selected[edge_mask_noise] = probE_selected[edge_mask_noise].scatter_(
            dim=-1,
            index=current_E[edge_mask_noise].unsqueeze(-1),
            value=0
        )
        # 归一化
        probE_selected[edge_mask_noise] = probE_selected[edge_mask_noise] / probE_selected[edge_mask_noise].sum(dim=-1, keepdim=True)

        # 采样离散特征
        sampled = diffusion_utils.sample_discrete_features(self.limit_dist, probX_selected, probE_selected, node_mask)

        X_t = sampled.X  # (batch_size, num_nodes)
        E_t = sampled.E  # (batch_size, num_nodes, num_nodes)

        X_t_final = X.clone()
        E_t_final = E.clone()

        # 仅对加噪的节点进行更新
        X_t_final[node_mask_noise] = F.one_hot(X_t[node_mask_noise], num_classes=self.Xdim_output).float()

        # 仅对加噪的边进行更新
        E_t_final[edge_mask_noise] = F.one_hot(E_t[edge_mask_noise], num_classes=self.Edim_output).float()

        z_t = utils.PlaceHolder(X=X_t_final, E=E_t_final, y=y).type_as(X_t_final).mask(node_mask)

        noisy_data = {
            't_int': t_nodes,
            't_e_int': t_edges,
            't_nodes': t_nodes.unsqueeze(1).float() / valid_nodes_per_graph.unsqueeze(1).float(),  # 形状: (batch_size, 1)
            't_edges': t_edges.unsqueeze(1).float() / (valid_edges_per_graph.unsqueeze(1).float() + 1e-8),  # 形状: (batch_size, 1)
            'X_t': z_t.X,                        # (batch_size, num_nodes, dx_out)
            'E_t': z_t.E,                        # (batch_size, num_nodes, num_nodes, de_out)
            'y_t': z_t.y,
            'node_mask': node_mask               # (batch_size, num_nodes)
        }

        return noisy_data


    def apply_noise11111(self, X, E, y, node_mask): #边依赖点
        """ 
        Sample noise and apply it to the graph data by selecting edges 
        connected to the nodes selected for noise addition.

        Args:
            X (torch.Tensor): Node features of shape (batch_size, num_nodes, dx_in).
            E (torch.Tensor): Edge features of shape (batch_size, num_nodes, num_nodes, de_in).
            y (torch.Tensor): Labels or additional data.
            node_mask (torch.Tensor): Mask indicating valid nodes, shape (batch_size, num_nodes).

        Returns:
            dict: A dictionary containing noisy data and related information.
        """
        batch_size, num_nodes, _ = X.size()
        
        device = X.device
        num_edges = num_nodes * (num_nodes - 1) // 2  # Total possible edges

        # Get upper triangular indices (excluding diagonal)
        triu_indices = torch.triu_indices(num_nodes, num_nodes, offset=1, device=device)  # (2, num_edges)

        # Sample t_nodes: integers between 1 and num_nodes (inclusive)
        t_nodes = torch.randint(1, num_nodes + 1, size=(batch_size, 1), device=device)  # Shape: (batch_size, 1)

        # Compute t_edges based on t_nodes
        t_edges = ((num_nodes - 1) * t_nodes) // 2  # Shape: (batch_size, 1)

        # Generate random scores for nodes
        rand_nodes = torch.rand(batch_size, num_nodes, device=device)

        # Sort the scores in descending order and get sorted indices
        sorted_scores_nodes, sorted_indices_nodes = torch.sort(rand_nodes, dim=1, descending=True)

        # Create a range tensor for comparison
        range_tensor_nodes = torch.arange(num_nodes, device=device).unsqueeze(0).expand(batch_size, num_nodes)

        mask_nodes = range_tensor_nodes < t_nodes

        # Scatter the mask to the sorted indices to select top t_nodes
        node_mask_noise = torch.zeros_like(mask_nodes, dtype=torch.bool)
        node_mask_noise.scatter_(1, sorted_indices_nodes, mask_nodes)

        # Create edge priority mask: edges connecting selected nodes
        node_mask_noise_unsqueezed = node_mask_noise.unsqueeze(2) & node_mask_noise.unsqueeze(1)
        # Remove self-loops if any
        diag_indices = torch.arange(num_nodes, device=device)
        node_mask_noise_unsqueezed[:, diag_indices, diag_indices] = False

        # Get priority edges in upper triangle
        edge_mask_priority = node_mask_noise_unsqueezed[:, triu_indices[0], triu_indices[1]]  # (batch_size, num_edges)

        # Remaining edges (non-priority)
        edge_mask_non_priority = ~edge_mask_priority  # (batch_size, num_edges)

        scores = torch.zeros(batch_size, num_edges, device=device)

        num_priority_edges_total = edge_mask_priority.sum().item()
        num_non_priority_edges_total = edge_mask_non_priority.sum().item()
        # Calculate the number of priority edges for each sample
        num_priority_edges = edge_mask_priority.sum(dim=1)  # (batch_size,)

        # Generate random scores for priority and non-priority edges
        rand_edges_priority = torch.rand(batch_size, num_edges, device=device)
        rand_edges_priority[~edge_mask_priority] = 2.0  # Set non-priority edges to a higher value
        rand_edges_non_priority = torch.rand(batch_size, num_edges, device=device)
        rand_edges_non_priority[~edge_mask_non_priority] = 2.0  # Set priority edges to a higher value

        # Initialize edge_mask_noise_flat as all False
        edge_mask_noise_flat = torch.zeros((batch_size, num_edges), dtype=torch.bool, device=device)

        # Total number of edges to select per sample
        t_edges_expanded = t_edges.expand(-1, num_edges)  # (batch_size, num_edges)

        # For priority edges, set high scores to ensure selection
        scores_priority = rand_edges_priority.clone()
        scores_priority[~edge_mask_priority] = 2.0  # Non-priority edges get high scores

        # For non-priority edges, set high scores to avoid selection unless needed
        scores_non_priority = rand_edges_non_priority.clone()
        scores_non_priority[~edge_mask_non_priority] = 2.0  # Priority edges get high scores

        scores[edge_mask_priority] = torch.rand(num_priority_edges_total, device=device)

        # For non-priority edges
        scores[edge_mask_non_priority] = torch.rand(num_non_priority_edges_total, device=device) + 1.0  # [1,2)

        # Sort the scores and get indices
        sorted_scores, sorted_indices = torch.sort(scores, dim=1)

        # Create a range tensor for comparison
        range_edges = torch.arange(num_edges, device=device).unsqueeze(0).expand(batch_size, num_edges)

        edge_selection_mask = range_edges < t_edges # (batch_size, num_edges)

        # Initialize edge_mask_noise_flat as all False
        edge_mask_noise_flat = torch.zeros(batch_size, num_edges, dtype=torch.bool, device=device)

        # Scatter the mask to the sorted indices to select top t_edges
        edge_mask_noise_flat.scatter_(1, sorted_indices, edge_selection_mask)

        # Expand triu_indices to (1, 2, num_edges) and repeat for batch
        triu_indices_exp = triu_indices.unsqueeze(0).expand(batch_size, -1, -1)  # (batch_size, 2, num_edges)

        # Create batch indices
        batch_indices = torch.arange(batch_size, device=device).unsqueeze(1).expand(-1, num_edges)  # (batch_size, num_edges)

        # Gather selected edge indices
        selected_rows = triu_indices_exp[:, 0, :][edge_mask_noise_flat]  # (num_selected,)
        selected_cols = triu_indices_exp[:, 1, :][edge_mask_noise_flat]  # (num_selected,)
        selected_batch = batch_indices[edge_mask_noise_flat]  # (num_selected,)

        # Initialize edge_mask_noise as all False
        edge_mask_noise = torch.zeros((batch_size, num_nodes, num_nodes), dtype=torch.bool, device=device)

        # Set selected edges in upper triangle
        edge_mask_noise[selected_batch, selected_rows, selected_cols] = True
        # Ensure symmetry by setting corresponding lower triangle edges
        edge_mask_noise[selected_batch, selected_cols, selected_rows] = True

        # Retrieve transition matrices for nodes and edges
        Qtb = self.transition_model.get_discrete_Qt_bar(device=device)  # Adjust as per your implementation

        # Compute transition probabilities for nodes
        probX = torch.matmul(X, Qtb.X)  # Shape: (batch_size, num_nodes, dx_out)

        # Compute transition probabilities for edges
        probE = torch.matmul(E, Qtb.E)  # Shape: (batch_size, num_nodes, num_nodes, de_out)

        current_X = X.argmax(dim=-1)  # Shape: (batch_size, num_nodes)
        current_E = E.argmax(dim=-1)  # Shape: (batch_size, num_nodes, num_nodes)

        # Ensure that the selected nodes will not stay the same
        probX_selected = probX.clone()
        """ probX_selected[node_mask_noise] = probX_selected[node_mask_noise].scatter_(
            dim=-1,
            index=current_X[node_mask_noise].unsqueeze(-1),
            value=0
        )
        # Normalize
        probX_selected[node_mask_noise] = probX_selected[node_mask_noise] / probX_selected[node_mask_noise].sum(dim=-1, keepdim=True) """

        # Ensure that the selected edges will not stay the same
        probE_selected = probE.clone()
        """ probE_selected[edge_mask_noise] = probE_selected[edge_mask_noise].scatter_(
            dim=-1,
            index=current_E[edge_mask_noise].unsqueeze(-1),
            value=0
        )
        # Normalize
        probE_selected[edge_mask_noise] = probE_selected[edge_mask_noise] / probE_selected[edge_mask_noise].sum(dim=-1, keepdim=True) """
            
        # Sample new features for nodes and edges
        sampled = diffusion_utils.sample_discrete_features(self.limit_dist, probX_selected, probE_selected, node_mask)

        X_t = sampled.X  # Shape: (batch_size, num_nodes)
        E_t = sampled.E  # Shape: (batch_size, num_nodes, num_nodes)

        X_t_final = X.clone()
        E_t_final = E.clone()

        # Update selected nodes
        X_t_final[node_mask_noise] = F.one_hot(X_t[node_mask_noise], num_classes=self.Xdim_output).float()

        # Update selected edges
        E_t_final[edge_mask_noise] = F.one_hot(E_t[edge_mask_noise], num_classes=self.Edim_output).float()

        z_t = utils.PlaceHolder(X=X_t_final, E=E_t_final, y=y).type_as(X_t_final).mask(node_mask)

        # Normalize t_nodes and t_edges by their maximum values for scaling
        t_nodes_norm = t_nodes.float() / num_nodes  # Shape: (batch_size,)
        t_edges_norm = t_edges.float() / num_edges  # Shape: (batch_size,)

        noisy_data = {
            't_nodes': t_nodes_norm,    # Shape: (batch_size,)
            't_edges': t_edges_norm,    # Shape: (batch_size,)
            'X_t': z_t.X,                            # Shape: (batch_size, num_nodes, dx_out)
            'E_t': z_t.E,                            # Shape: (batch_size, num_nodes, num_nodes, de_out)
            'y_t': z_t.y,                        
            'node_mask': node_mask                   # Shape: (batch_size, num_nodes)
        }

        return noisy_data


    def apply_noise1(self, X, E, y, node_mask): #子图
        """ 
        Sample noise and apply it to the graph data by selecting edges 
        connected to the nodes selected for noise addition.

        Args:
            X (torch.Tensor): Node features of shape (batch_size, num_nodes, dx_in).
            E (torch.Tensor): Edge features of shape (batch_size, num_nodes, num_nodes, de_in).
            y (torch.Tensor): Labels or additional data.
            node_mask (torch.Tensor): Mask indicating valid nodes, shape (batch_size, num_nodes).

        Returns:
            dict: A dictionary containing noisy data and related information.
        """
        batch_size, num_nodes, _ = X.size()
        
        device = X.device
        num_edges = num_nodes * (num_nodes - 1) // 2  # Total possible edges

        # Get upper triangular indices (excluding diagonal)
        triu_indices = torch.triu_indices(num_nodes, num_nodes, offset=1, device=device)  # (2, num_edges)

        # Sample t_nodes: integers between 1 and num_nodes (inclusive)
        t_nodes = torch.randint(1, num_nodes + 1, size=(batch_size, 1), device=device)  # Shape: (batch_size, 1)

        # Generate random scores for nodes
        rand_nodes = torch.rand(batch_size, num_nodes, device=device)

        # Sort the scores in descending order and get sorted indices
        sorted_scores_nodes, sorted_indices_nodes = torch.sort(rand_nodes, dim=1, descending=True)

        # Create a range tensor for comparison
        range_tensor_nodes = torch.arange(num_nodes, device=device).unsqueeze(0).expand(batch_size, num_nodes)

        mask_nodes = range_tensor_nodes < t_nodes

        # Scatter the mask to the sorted indices to select top t_nodes
        node_mask_noise = torch.zeros_like(mask_nodes, dtype=torch.bool)
        node_mask_noise.scatter_(1, sorted_indices_nodes, mask_nodes)

        node_mask_noise_expanded_1 = node_mask_noise.unsqueeze(2)  # (batch_size, num_nodes, 1)
        node_mask_noise_expanded_2 = node_mask_noise.unsqueeze(1)  # (batch_size, 1, num_nodes)

        # 计算节点之间的连接掩码
        edge_mask_noise_full = node_mask_noise_expanded_1 & node_mask_noise_expanded_2  # (batch_size, num_nodes, num_nodes)

        # 移除对角线元素（自环）
        diag_mask = torch.eye(num_nodes, dtype=torch.bool, device=device).unsqueeze(0)  # (1, num_nodes, num_nodes)
        edge_mask_noise = edge_mask_noise_full & (~diag_mask)

        # Retrieve transition matrices for nodes and edges
        Qtb = self.transition_model.get_discrete_Qt_bar(device=device)  # Adjust as per your implementation

        # Compute transition probabilities for nodes
        probX = torch.matmul(X, Qtb.X)  # Shape: (batch_size, num_nodes, dx_out)

        # Compute transition probabilities for edges
        probE = torch.matmul(E, Qtb.E)  # Shape: (batch_size, num_nodes, num_nodes, de_out)

        current_X = X.argmax(dim=-1)  # Shape: (batch_size, num_nodes)
        current_E = E.argmax(dim=-1)  # Shape: (batch_size, num_nodes, num_nodes)

        # Ensure that the selected nodes will not stay the same
        probX_selected = probX.clone()
        probX_selected[node_mask_noise] = probX_selected[node_mask_noise].scatter_(
            dim=-1,
            index=current_X[node_mask_noise].unsqueeze(-1),
            value=0
        )
        # Normalize
        probX_selected[node_mask_noise] = probX_selected[node_mask_noise] / probX_selected[node_mask_noise].sum(dim=-1, keepdim=True)

        # Ensure that the selected edges will not stay the same
        probE_selected = probE.clone()
        #probE_selected[edge_mask_noise] = probE_selected[edge_mask_noise].scatter_(
        #    dim=-1,
        #    index=current_E[edge_mask_noise].unsqueeze(-1),
        #    value=0
        #)
        # Normalize
        #probE_selected[edge_mask_noise] = probE_selected[edge_mask_noise] / probE_selected[edge_mask_noise].sum(dim=-1, keepdim=True)
            
        # Sample new features for nodes and edges
        sampled = diffusion_utils.sample_discrete_features(self.limit_dist, probX_selected, probE_selected, node_mask)

        X_t = sampled.X  # Shape: (batch_size, num_nodes)
        E_t = sampled.E  # Shape: (batch_size, num_nodes, num_nodes)

        X_t_final = X.clone()
        E_t_final = E.clone()

        # Update selected nodes
        X_t_final[node_mask_noise] = F.one_hot(X_t[node_mask_noise], num_classes=self.Xdim_output).float()

        # Update selected edges
        E_t_final[edge_mask_noise] = F.one_hot(E_t[edge_mask_noise], num_classes=self.Edim_output).float()

        """ print(111)
        print(t_nodes[0])
        print(X[0].argmax(dim=-1))
        print(E[0].argmax(dim=-1))
        print(node_mask_noise[0])
        print(edge_mask_noise[0])
        print(probX_selected[0])
        print(probE_selected[0])
        print(node_mask[0])
        print(X_t[0])
        print(E_t[0])
        print(X_t_final[0].argmax(dim=-1))
        print(E_t_final[0].argmax(dim=-1))
        exit() """

        z_t = utils.PlaceHolder(X=X_t_final, E=E_t_final, y=y).type_as(X_t_final).mask(node_mask)

        # Normalize t_nodes and t_edges by their maximum values for scaling
        t_nodes_norm = t_nodes.float() / num_nodes  # Shape: (batch_size, 1)

        noisy_data = {
            't_nodes': t_nodes_norm,    # Shape: (batch_size, 1)
            'X_t': z_t.X,                            # Shape: (batch_size, num_nodes, dx_out)
            'E_t': z_t.E,                            # Shape: (batch_size, num_nodes, num_nodes, de_out)
            'y_t': z_t.y,                        
            'node_mask': node_mask                   # Shape: (batch_size, num_nodes)
        }

        return noisy_data

    def forward(self, noisy_data, extra_data, node_mask):
        X = torch.cat((noisy_data['X_t'], extra_data.X), dim=2).float()
        E = torch.cat((noisy_data['E_t'], extra_data.E), dim=3).float()
        y = torch.hstack((noisy_data['y_t'], extra_data.y)).float()
        return self.model(X, E, y, node_mask)

    @torch.no_grad()
    def sample_batch1(self, batch_id: int, batch_size: int, keep_chain: int, number_chain_steps: int,
                     save_final: int, num_nodes=None):   #点和边同时
        """
        :param batch_id: int
        :param batch_size: int
        :param num_nodes: int, <int>tensor (batch_size) (optional) for specifying number of nodes
        :param save_final: int: number of predictions to save to file
        :param keep_chain: int: number of chains to save to file
        :param keep_chain_steps: number of timesteps to save for each chain
        :return: molecule_list. Each element of this list is a tuple (atom_types, charges, positions)
        """
        if num_nodes is None:
            n_nodes = self.node_dist.sample_n(batch_size, self.device)
        elif type(num_nodes) == int:
            n_nodes = num_nodes * torch.ones(batch_size, device=self.device, dtype=torch.int)
        else:
            assert isinstance(num_nodes, torch.Tensor)
            n_nodes = num_nodes
        n_max = torch.max(n_nodes).item()
        # Build the masks
        arange = torch.arange(n_max, device=self.device).unsqueeze(0).expand(batch_size, -1)
        node_mask = arange < n_nodes.unsqueeze(1)
        # Sample noise  -- z has size (n_samples, n_nodes, n_features)
        z_T = diffusion_utils.sample_discrete_feature_noise(limit_dist=self.limit_dist, node_mask=node_mask)
        X, E, y = z_T.X, z_T.E, z_T.y

        assert (E == torch.transpose(E, 1, 2)).all()
        assert number_chain_steps < self.T
        chain_X_size = torch.Size((number_chain_steps, keep_chain, X.size(1)))
        chain_E_size = torch.Size((number_chain_steps, keep_chain, E.size(1), E.size(2)))

        chain_X = torch.zeros(chain_X_size)
        chain_E = torch.zeros(chain_E_size)



        # Iteratively sample p(z_s | z_t) for t = 1, ..., T, with s = t - 1.
        """ for s_int in reversed(range(0, self.T)):
            s_array = s_int * torch.ones((batch_size, 1)).type_as(y)
            t_array = s_array + 1
            s_norm = s_array / self.T
            t_norm = t_array / self.T

            # Sample z_s
            sampled_s, discrete_sampled_s = self.sample_p_zs_given_zt(s_norm, t_norm, X, E, y, node_mask)
            X, E, y = sampled_s.X, sampled_s.E, sampled_s.y

            # Save the first keep_chain graphs
            write_index = (s_int * number_chain_steps) // self.T
            chain_X[write_index] = discrete_sampled_s.X[:keep_chain]
            chain_E[write_index] = discrete_sampled_s.E[:keep_chain]  """



        # Define generation steps for nodes and edges
        total_node_steps = n_max
        total_edge_steps = n_max * (n_max - 1) // 2

        # Define total generation steps
        total_steps = total_edge_steps  

        # Define interval steps of nodes generation
        node_update_interval = max(1, total_edge_steps // total_node_steps) if total_node_steps > 0 else total_edge_steps

        current_node_step = total_node_steps
        current_edge_step = total_edge_steps

        for step in reversed(range(total_steps)):
            current_step = step + 1  

            # Nodes generation
            if current_node_step > 0 and (current_step % node_update_interval == 0):
                # Nodes need generation
                s_nodes = current_node_step - 1
                t_nodes = current_node_step
                s_norm_nodes = s_nodes / total_node_steps
                t_norm_nodes = t_nodes / total_node_steps
                s_nodes_tensor = s_nodes * torch.ones((batch_size, 1), device=self.device, dtype=torch.float)
                t_nodes_tensor = t_nodes * torch.ones((batch_size, 1), device=self.device, dtype=torch.float)
                s_norm_nodes_tensor = s_norm_nodes * torch.ones((batch_size, 1), device=self.device, dtype=torch.float)
                t_norm_nodes_tensor = t_norm_nodes * torch.ones((batch_size, 1), device=self.device, dtype=torch.float)
                nodes_to_update = True
                current_node_step -= 1
            else:
                # Nodes do not need generation
                s_nodes = current_node_step - 1
                t_nodes = current_node_step
                s_norm_nodes = s_nodes / total_node_steps if total_node_steps > 0 else 0.0
                t_norm_nodes = t_nodes / total_node_steps if total_node_steps > 0 else 0.0
                s_nodes_tensor = s_nodes * torch.ones((batch_size, 1), device=self.device, dtype=torch.float)
                t_nodes_tensor = t_nodes * torch.ones((batch_size, 1), device=self.device, dtype=torch.float)
                s_norm_nodes_tensor = s_norm_nodes * torch.ones((batch_size, 1), device=self.device, dtype=torch.float)
                t_norm_nodes_tensor = t_norm_nodes * torch.ones((batch_size, 1), device=self.device, dtype=torch.float)
                nodes_to_update = False

            # Edges generation
            s_edges = current_edge_step - 1
            t_edges = current_edge_step
            s_norm_edges = s_edges / total_edge_steps
            t_norm_edges = t_edges / total_edge_steps
            s_edges_tensor = s_edges * torch.ones((batch_size, 1), device=self.device, dtype=torch.float)
            t_edges_tensor = t_edges * torch.ones((batch_size, 1), device=self.device, dtype=torch.float)
            s_norm_edges_tensor = s_norm_edges * torch.ones((batch_size, 1), device=self.device, dtype=torch.float)
            t_norm_edges_tensor = t_norm_edges * torch.ones((batch_size, 1), device=self.device, dtype=torch.float)
            current_edge_step -= 1

            # Sample nodes and edges
            sampled_s, discrete_sampled_s = self.sample_p_zs_given_zt(s_nodes_tensor, t_nodes_tensor, s_edges_tensor, t_edges_tensor, 
            s_norm_nodes_tensor, t_norm_nodes_tensor, s_norm_edges_tensor, t_norm_edges_tensor, X, E, y, node_mask)
    
            if nodes_to_update:
                X, E, y = sampled_s.X, sampled_s.E, sampled_s.y
            else:
                E, y = sampled_s.E, sampled_s.y

            # Save the first keep_chain graphs
            write_index = s_nodes 
            chain_X[write_index] = discrete_sampled_s.X[:keep_chain]
            chain_E[write_index] = discrete_sampled_s.E[:keep_chain]

        # Sample
        sampled_s = sampled_s.mask(node_mask, collapse=True)
        X, E, y = sampled_s.X, sampled_s.E, sampled_s.y

        # Prepare the chain for saving
        if keep_chain > 0:
            final_X_chain = X[:keep_chain]
            final_E_chain = E[:keep_chain]

            chain_X[0] = final_X_chain                  # Overwrite last frame with the resulting X, E
            chain_E[0] = final_E_chain

            chain_X = diffusion_utils.reverse_tensor(chain_X)
            chain_E = diffusion_utils.reverse_tensor(chain_E)

            # Repeat last frame to see final sample better
            chain_X = torch.cat([chain_X, chain_X[-1:].repeat(10, 1, 1)], dim=0)
            chain_E = torch.cat([chain_E, chain_E[-1:].repeat(10, 1, 1, 1)], dim=0)
            assert chain_X.size(0) == (number_chain_steps + 10)

        molecule_list = []
        for i in range(batch_size):
            n = n_nodes[i]
            atom_types = X[i, :n].cpu()
            edge_types = E[i, :n, :n].cpu()
            molecule_list.append([atom_types, edge_types])

        # Visualize chains
        if self.visualization_tools is not None:
            self.print('Visualizing chains...')
            current_path = os.getcwd()
            num_molecules = chain_X.size(1)       # number of molecules
            for i in range(num_molecules):
                result_path = os.path.join(current_path, f'chains/{self.cfg.general.name}/'
                                                         f'epoch{self.current_epoch}/'
                                                         f'chains/molecule_{batch_id + i}')
                if not os.path.exists(result_path):
                    os.makedirs(result_path)
                    _ = self.visualization_tools.visualize_chain(result_path,
                                                                 chain_X[:, i, :].numpy(),
                                                                 chain_E[:, i, :].numpy())
                self.print('\r{}/{} complete'.format(i+1, num_molecules), end='', flush=True)
            self.print('\nVisualizing molecules...')

            # Visualize the final molecules
            current_path = os.getcwd()
            result_path = os.path.join(current_path,
                                       f'graphs/{self.name}/epoch{self.current_epoch}_b{batch_id}/')
            self.visualization_tools.visualize(result_path, molecule_list, save_final)
            self.print("Done.")

        return molecule_list



    @torch.no_grad()
    def sample_batch111(self, batch_id: int, batch_size: int, keep_chain: int, number_chain_steps: int,
                     save_final: int, num_nodes=None):  #先点后边
        """
        :param batch_id: int
        :param batch_size: int
        :param num_nodes: int, <int>tensor (batch_size) (optional) for specifying number of nodes
        :param save_final: int: number of predictions to save to file
        :param keep_chain: int: number of chains to save to file
        :param keep_chain_steps: number of timesteps to save for each chain
        :return: molecule_list. Each element of this list is a tuple (atom_types, charges, positions)
        """
        if num_nodes is None:
            n_nodes = self.node_dist.sample_n(batch_size, self.device)
        elif type(num_nodes) == int:
            n_nodes = num_nodes * torch.ones(batch_size, device=self.device, dtype=torch.int)
        else:
            assert isinstance(num_nodes, torch.Tensor)
            n_nodes = num_nodes
        n_max = torch.max(n_nodes).item()
        # Build the masks
        arange = torch.arange(n_max, device=self.device).unsqueeze(0).expand(batch_size, -1)
        node_mask = arange < n_nodes.unsqueeze(1)
        # Sample noise  -- z has size (n_samples, n_nodes, n_features)
        z_T = diffusion_utils.sample_discrete_feature_noise(limit_dist=self.limit_dist, node_mask=node_mask)
        X, E, y = z_T.X, z_T.E, z_T.y

        assert (E == torch.transpose(E, 1, 2)).all()
        number_chain_steps = n_max * (n_max - 1) // 2
        chain_X_size = torch.Size((number_chain_steps, keep_chain, X.size(1)))
        chain_E_size = torch.Size((number_chain_steps, keep_chain, E.size(1), E.size(2)))

        chain_X = torch.zeros(chain_X_size)
        chain_E = torch.zeros(chain_E_size)

        # Define generation steps for nodes and edges
        total_node_steps = n_max
        total_edge_steps = n_max * (n_max - 1) // 2

        current_node_step = total_node_steps
        current_edge_step = total_edge_steps

        # 定义最大时间步张量
        max_node_steps_tensor = total_node_steps * torch.ones((batch_size, 1)).type_as(y)
        max_edge_steps_tensor = total_edge_steps * torch.ones((batch_size, 1)).type_as(y)

        # 首先生成节点
        for step in reversed(range(total_node_steps)):
            current_step = step + 1

            # 节点生成
            s_nodes = current_node_step - 1
            t_nodes = current_node_step
            s_norm_nodes = s_nodes / total_node_steps
            t_norm_nodes = t_nodes / total_node_steps
            s_nodes_tensor = s_nodes * torch.ones((batch_size, 1)).type_as(y)
            t_nodes_tensor = t_nodes * torch.ones((batch_size, 1)).type_as(y)
            s_norm_nodes_tensor = s_norm_nodes * torch.ones((batch_size, 1)).type_as(y)
            t_norm_nodes_tensor = t_norm_nodes * torch.ones((batch_size, 1)).type_as(y)
            current_node_step -= 1

            # 边不需要更新，设置为最大时间步
            s_edges_tensor = max_edge_steps_tensor
            t_edges_tensor = max_edge_steps_tensor
            s_norm_edges_tensor = torch.ones((batch_size, 1)).type_as(y)
            t_norm_edges_tensor = torch.ones((batch_size, 1)).type_as(y)

            # 采样节点
            sampled_s, discrete_sampled_s = self.sample_p_zs_given_zt(
                s_nodes_tensor, t_nodes_tensor, s_edges_tensor, t_edges_tensor,
                s_norm_nodes_tensor, t_norm_nodes_tensor, s_norm_edges_tensor, t_norm_edges_tensor,
                X, E, y, node_mask
            )

            # 只更新节点特征，边特征保持不变
            X = sampled_s.X

        # 然后生成边
        for step in reversed(range(total_edge_steps)):
            current_step = step + 1

            # 边生成
            s_edges = current_edge_step - 1
            t_edges = current_edge_step
            s_norm_edges = s_edges / total_edge_steps
            t_norm_edges = t_edges / total_edge_steps
            s_edges_tensor = s_edges * torch.ones((batch_size, 1)).type_as(y)
            t_edges_tensor = t_edges * torch.ones((batch_size, 1)).type_as(y)
            s_norm_edges_tensor = s_norm_edges * torch.ones((batch_size, 1)).type_as(y)
            t_norm_edges_tensor = t_norm_edges * torch.ones((batch_size, 1)).type_as(y)
            current_edge_step -= 1

            # 节点不需要更新，设置为1
            s_nodes_tensor = 0
            t_nodes_tensor = 1
            s_norm_nodes_tensor = torch.zeros((batch_size, 1)).type_as(y)
            t_norm_nodes_tensor = torch.zeros((batch_size, 1)).type_as(y)

            # 采样边
            sampled_s, discrete_sampled_s = self.sample_p_zs_given_zt(
                s_nodes_tensor, t_nodes_tensor, s_edges_tensor, t_edges_tensor,
                s_norm_nodes_tensor, t_norm_nodes_tensor, s_norm_edges_tensor, t_norm_edges_tensor,
                X, E, y, node_mask
            )

            # 只更新边特征，节点特征保持不变
            E = sampled_s.E

            # 保存采样结果（如果需要）
            write_index = s_edges
            chain_X[write_index] = discrete_sampled_s.X[:keep_chain]
            chain_E[write_index] = discrete_sampled_s.E[:keep_chain]

        # Sample
        sampled = utils.PlaceHolder(X=X, E=E, y=torch.zeros(y.shape[0], 0))
        sampled = sampled.mask(node_mask, collapse=True)
        X, E, y = sampled.X, sampled.E, sampled.y

        # Prepare the chain for saving
        if keep_chain > 0:
            final_X_chain = X[:keep_chain]
            final_E_chain = E[:keep_chain]

            chain_X[0] = final_X_chain                  # Overwrite last frame with the resulting X, E
            chain_E[0] = final_E_chain

            chain_X = diffusion_utils.reverse_tensor(chain_X)
            chain_E = diffusion_utils.reverse_tensor(chain_E)

            # Repeat last frame to see final sample better
            chain_X = torch.cat([chain_X, chain_X[-1:].repeat(10, 1, 1)], dim=0)
            chain_E = torch.cat([chain_E, chain_E[-1:].repeat(10, 1, 1, 1)], dim=0)
            assert chain_X.size(0) == (number_chain_steps + 10)

        molecule_list = []
        for i in range(batch_size):
            n = n_nodes[i]
            atom_types = X[i, :n].cpu()
            edge_types = E[i, :n, :n].cpu()
            molecule_list.append([atom_types, edge_types])

        # Visualize chains
        if self.visualization_tools is not None:
            self.print('Visualizing chains...')
            current_path = os.getcwd()
            num_molecules = chain_X.size(1)       # number of molecules
            for i in range(num_molecules):
                result_path = os.path.join(current_path, f'chains/{self.cfg.general.name}/'
                                                         f'epoch{self.current_epoch}/'
                                                         f'chains/molecule_{batch_id + i}')
                if not os.path.exists(result_path):
                    os.makedirs(result_path)
                    _ = self.visualization_tools.visualize_chain(result_path,
                                                                 chain_X[:, i, :].numpy(),
                                                                 chain_E[:, i, :].numpy())
                self.print('\r{}/{} complete'.format(i+1, num_molecules), end='', flush=True)
            self.print('\nVisualizing molecules...')

            # Visualize the final molecules
            current_path = os.getcwd()
            result_path = os.path.join(current_path,
                                       f'graphs/{self.name}/epoch{self.current_epoch}_b{batch_id}/')
            self.visualization_tools.visualize(result_path, molecule_list, save_final)
            self.print("Done.")

        return molecule_list


    @torch.no_grad()
    def sample_batch0101(self, batch_id: int, batch_size: int, keep_chain: int, number_chain_steps: int,
                     save_final: int, num_nodes=None):  #点和边真正的同时
        """
        :param batch_id: int
        :param batch_size: int
        :param num_nodes: int, <int>tensor (batch_size) (optional) for specifying number of nodes
        :param save_final: int: number of predictions to save to file
        :param keep_chain: int: number of chains to save to file
        :param keep_chain_steps: number of timesteps to save for each chain
        :return: molecule_list. Each element of this list is a tuple (atom_types, charges, positions)
        """
        if num_nodes is None:
            n_nodes = self.node_dist.sample_n(batch_size, self.device)
        elif type(num_nodes) == int:
            n_nodes = num_nodes * torch.ones(batch_size, device=self.device, dtype=torch.int)
        else:
            assert isinstance(num_nodes, torch.Tensor)
            n_nodes = num_nodes

        n_max = torch.max(n_nodes).item()
        # Build the masks
        arange = torch.arange(n_max, device=self.device).unsqueeze(0).expand(batch_size, -1)
        node_mask = arange < n_nodes.unsqueeze(1)
        # Sample noise  -- z has size (n_samples, n_nodes, n_features)
        z_T = diffusion_utils.sample_discrete_feature_noise(limit_dist=self.limit_dist, node_mask=node_mask)
        X, E, y = z_T.X, z_T.E, z_T.y
        assert (E == torch.transpose(E, 1, 2)).all()

        number_chain_steps = n_max + 1
        chain_X_size = torch.Size((number_chain_steps, keep_chain, X.size(1)))
        chain_E_size = torch.Size((number_chain_steps, keep_chain, E.size(1), E.size(2)))

        total_steps = n_max

        chain_X = torch.zeros(chain_X_size)
        chain_E = torch.zeros(chain_E_size)

        # Define generation steps for nodes and edges
        total_node_steps = n_max
        total_edge_steps = n_max * (n_max - 1) // 2

        for step in reversed(range(total_steps)):
            s = step  
            s_nodes = s
            t_nodes = s + 1

            s_edges = ((n_max - 1) * s_nodes) // 2
            t_edges = ((n_max - 1) * t_nodes) // 2

            s_norm_nodes = s_nodes / total_node_steps
            t_norm_nodes = t_nodes / total_node_steps

            s_norm_edges = s_edges / total_edge_steps if total_edge_steps > 0 else 0.0
            t_norm_edges = t_edges / total_edge_steps if total_edge_steps > 0 else 0.0
           
            s_nodes_tensor = s_nodes * torch.ones((batch_size, 1)).type_as(y)#, device=self.device, dtype=torch.float)
            t_nodes_tensor = t_nodes * torch.ones((batch_size, 1)).type_as(y)#, device=self.device, dtype=torch.float)
            s_norm_nodes_tensor = s_norm_nodes * torch.ones((batch_size, 1)).type_as(y)#, device=self.device, dtype=torch.float)
            t_norm_nodes_tensor = t_norm_nodes * torch.ones((batch_size, 1)).type_as(y)#, device=self.device, dtype=torch.float)

            s_edges_tensor = s_edges * torch.ones((batch_size, 1)).type_as(y)#, device=self.device, dtype=torch.float)
            t_edges_tensor = t_edges * torch.ones((batch_size, 1)).type_as(y)#, device=self.device, dtype=torch.float)
            s_norm_edges_tensor = s_norm_edges * torch.ones((batch_size, 1)).type_as(y)#, device=self.device, dtype=torch.float)
            t_norm_edges_tensor = t_norm_edges * torch.ones((batch_size, 1)).type_as(y)#, device=self.device, dtype=torch.float)

            sampled_s, discrete_sampled_s = self.sample_p_zs_given_zt(
            s_nodes_tensor, t_nodes_tensor, s_edges_tensor, t_edges_tensor,
            s_norm_nodes_tensor, t_norm_nodes_tensor, s_norm_edges_tensor, t_norm_edges_tensor,
            X, E, y, node_mask)

            X, E, y = sampled_s.X, sampled_s.E, sampled_s.y


            write_index = t_nodes
            chain_X[write_index] = discrete_sampled_s.X[:keep_chain]
            chain_E[write_index] = discrete_sampled_s.E[:keep_chain]
        # Sample
        sampled = utils.PlaceHolder(X=X, E=E, y=torch.zeros(y.shape[0], 0))
        sampled = sampled.mask(node_mask, collapse=True)
        X, E, y = sampled.X, sampled.E, sampled.y

        # Prepare the chain for saving
        if keep_chain > 0:
            final_X_chain = X[:keep_chain]
            final_E_chain = E[:keep_chain]

            chain_X[0] = final_X_chain                  # Overwrite last frame with the resulting X, E
            chain_E[0] = final_E_chain

            chain_X = diffusion_utils.reverse_tensor(chain_X)
            chain_E = diffusion_utils.reverse_tensor(chain_E)

            # Repeat last frame to see final sample better
            chain_X = torch.cat([chain_X, chain_X[-1:].repeat(10, 1, 1)], dim=0)
            chain_E = torch.cat([chain_E, chain_E[-1:].repeat(10, 1, 1, 1)], dim=0)
            assert chain_X.size(0) == (number_chain_steps + 10)

        molecule_list = []
        for i in range(batch_size):
            n = n_nodes[i]
            atom_types = X[i, :n].cpu()
            edge_types = E[i, :n, :n].cpu()
            molecule_list.append([atom_types, edge_types])

        # Visualize chains
        if self.visualization_tools is not None:
            self.print('Visualizing chains...')
            current_path = os.getcwd()
            num_molecules = chain_X.size(1)       # number of molecules
            for i in range(num_molecules):
                result_path = os.path.join(current_path, f'chains/{self.cfg.general.name}/'
                                                         f'epoch{self.current_epoch}/'
                                                         f'chains/molecule_{batch_id + i}')
                if not os.path.exists(result_path):
                    os.makedirs(result_path)
                    _ = self.visualization_tools.visualize_chain(result_path,
                                                                 chain_X[:, i, :].numpy(),
                                                                 chain_E[:, i, :].numpy())
                self.print('\r{}/{} complete'.format(i+1, num_molecules), end='', flush=True)
            self.print('\nVisualizing molecules...')

            # Visualize the final molecules
            current_path = os.getcwd()
            result_path = os.path.join(current_path,
                                       f'graphs/{self.name}/epoch{self.current_epoch}_b{batch_id}/')
            self.visualization_tools.visualize(result_path, molecule_list, save_final)
            self.print("Done.")

        return molecule_list

    @torch.no_grad()
    def sample_batch0101001(self, batch_id: int, batch_size: int, keep_chain: int, number_chain_steps: int,
                    save_final: int, num_nodes=None):  # 点和边真正的同时，只focus有效节点
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
        # 构建节点掩码
        arange = torch.arange(n_max, device=self.device).unsqueeze(0).expand(batch_size, -1)
        node_mask = arange < n_nodes.unsqueeze(1)
        # 采样初始噪声  -- z_T 形状为 (batch_size, n_max, feature_dim)
        z_T = diffusion_utils.sample_discrete_feature_noise(limit_dist=self.limit_dist, node_mask=node_mask)
        X, E, y = z_T.X, z_T.E, z_T.y
        assert (E == torch.transpose(E, 1, 2)).all()

        # 定义链的尺寸，用于保存中间结果
        chain_X_size = torch.Size((number_chain_steps + 1, keep_chain, X.size(1)))
        chain_E_size = torch.Size((number_chain_steps + 1, keep_chain, E.size(1), E.size(2)))

        chain_X = torch.zeros(chain_X_size, device=self.device)
        chain_E = torch.zeros(chain_E_size, device=self.device)

        # 计算每个图的有效节点数
        valid_nodes_per_graph = n_nodes  # 形状为 (batch_size,)
        max_node_steps = valid_nodes_per_graph.max().item()  # 最大的有效节点数

        # 初始化 t_nodes 为有效节点数的张量，形状为 (batch_size,)
        t_nodes = valid_nodes_per_graph.clone()  # 初始化为有效节点数

        # 计算每个图的有效边数
        valid_edges_per_graph = (valid_nodes_per_graph * (valid_nodes_per_graph - 1)) // 2  # 形状为 (batch_size,)

        # 初始化 t_edges 为有效边数的张量，形状为 (batch_size,)
        t_edges = valid_edges_per_graph.clone()

        # 计算总的时间步数（以最大节点数为准）
        total_steps = max_node_steps

        for step in reversed(range(total_steps + 1)):
            # 对于每个时间步，计算对应的 s_nodes 和 t_nodes
            s_nodes = t_nodes - 1  # 形状为 (batch_size,)
            s_nodes = torch.clamp(s_nodes, min=0)  # 保证不小于0

            # 对于边，同样计算 s_edges 和 t_edges
            s_edges = (valid_nodes_per_graph - 1) * s_nodes // 2
            s_edges = torch.clamp(s_edges, min=0)

            # 计算归一化的 s 和 t
            s_norm_nodes = s_nodes.float() / valid_nodes_per_graph.float()
            t_norm_nodes = t_nodes.float() / valid_nodes_per_graph.float()

            s_norm_edges = s_edges.float() / valid_edges_per_graph.float()
            t_norm_edges = t_edges.float() / valid_edges_per_graph.float()

            # 将 s 和 t 转换为张量，形状为 (batch_size, 1)
            s_nodes_tensor = s_nodes.unsqueeze(1).float()
            t_nodes_tensor = t_nodes.unsqueeze(1).float()
            s_norm_nodes_tensor = s_norm_nodes.unsqueeze(1)
            t_norm_nodes_tensor = t_norm_nodes.unsqueeze(1)

            s_edges_tensor = s_edges.unsqueeze(1).float()
            t_edges_tensor = t_edges.unsqueeze(1).float()
            s_norm_edges_tensor = s_norm_edges.unsqueeze(1)
            t_norm_edges_tensor = t_norm_edges.unsqueeze(1)

            # 调用采样函数，从 z_t 采样 z_s
            sampled_s, discrete_sampled_s = self.sample_p_zs_given_zt(
                s_nodes_tensor, t_nodes_tensor, s_edges_tensor, t_edges_tensor,
                s_norm_nodes_tensor, t_norm_nodes_tensor, s_norm_edges_tensor, t_norm_edges_tensor,
                X, E, y, node_mask)

            X, E, y = sampled_s.X, sampled_s.E, sampled_s.y

            # 保存中间结果
            write_index = t_nodes.min().item()  # 使用最小的 t_nodes 作为索引
            if write_index < chain_X.size(0):
                chain_X[write_index] = discrete_sampled_s.X[:keep_chain]
                chain_E[write_index] = discrete_sampled_s.E[:keep_chain]

            # 更新 t_nodes 和 t_edges，为下一时间步做准备
            t_nodes = s_nodes
            t_edges = s_edges

        # 最终的采样结果
        sampled = utils.PlaceHolder(X=X, E=E, y=torch.zeros(y.shape[0], 0))
        sampled = sampled.mask(node_mask, collapse=True)
        X, E, y = sampled.X, sampled.E, sampled.y

        # 准备保存链的结果
        if keep_chain > 0:
            final_X_chain = X[:keep_chain]
            final_E_chain = E[:keep_chain]

            chain_X[0] = final_X_chain  # 将最终的 X, E 保存到链的起始位置
            chain_E[0] = final_E_chain

            chain_X = diffusion_utils.reverse_tensor(chain_X)
            chain_E = diffusion_utils.reverse_tensor(chain_E)

            # 重复最后一帧以更好地查看最终样本
            chain_X = torch.cat([chain_X, chain_X[-1:].repeat(10, 1, 1)], dim=0)
            chain_E = torch.cat([chain_E, chain_E[-1:].repeat(10, 1, 1, 1)], dim=0)
            assert chain_X.size(0) == (number_chain_steps + 11)

        molecule_list = []
        for i in range(batch_size):
            n = n_nodes[i]
            atom_types = X[i, :n].cpu()
            edge_types = E[i, :n, :n].cpu()
            molecule_list.append([atom_types, edge_types])

        # 可视化链
        if self.visualization_tools is not None:
            self.print('Visualizing chains...')
            current_path = os.getcwd()
            num_molecules = chain_X.size(1)  # 分子数量
            for i in range(num_molecules):
                result_path = os.path.join(current_path, f'chains/{self.cfg.general.name}/'
                                                        f'epoch{self.current_epoch}/'
                                                        f'chains/molecule_{batch_id + i}')
                if not os.path.exists(result_path):
                    os.makedirs(result_path)
                    _ = self.visualization_tools.visualize_chain(result_path,
                                                                chain_X[:, i, :].cpu().numpy(),
                                                                chain_E[:, i, :].cpu().numpy())
                self.print('\r{}/{} complete'.format(i+1, num_molecules), end='', flush=True)
            self.print('\nVisualizing molecules...')

            # 可视化最终的分子
            current_path = os.getcwd()
            result_path = os.path.join(current_path,
                                    f'graphs/{self.name}/epoch{self.current_epoch}_b{batch_id}/')
            self.visualization_tools.visualize(result_path, molecule_list, save_final)
            self.print("Done.")

        return molecule_list

    def connect_components543453(self, X, E, node_mask): #origin
        """
        在批量中的每个图中连接不相连的连通分量。

        参数：
            X: 形状为 (batch_size, num_nodes, num_node_features) 的张量
                节点特征张量。
            E: 形状为 (batch_size, num_nodes, num_nodes, num_edge_features) 的张量
                边特征张量。
            node_mask: 形状为 (batch_size, num_nodes) 的张量
                指示每个图中有效节点的掩码。

        返回：
            X_connected: 更新后的节点特征张量。
            E_connected: 添加连接后的边特征张量。
        """
        batch_size, num_nodes, node_classes = X.size()
        device = X.device

        """  batch_size=1
        num_nodes=9
        X = torch.zeros((1, 9, 4), device=device)
        X[0,0,0] = 1  # node0: C
        X[0,1,1] = 1  # node1: N
        X[0,2,2] = 1  # node2: O
        X[0,3,3] = 1  # node3: F
        X[0,4,2] = 1  # node4: O
        X[0,5,3] = 1  # node5: F
        X[0,6,2] = 1  # node6: O

        node_mask=torch.tensor([[True, True, True, True, True, True, True, False, False]], device=device)
        print(node_mask.shape)
        # 定义边类型的one-hot编码
        # 无边, 单键, 双键, 三键, 芳香键
        E = torch.zeros((1, 9, 9, 5), device=device)

        # 添加连通分量：
        # component1: node0（单独一个节点）
        # component2: node1（单独一个节点）
        # component3: node2-node6（连通，单键连接）
        # 边类型为单键（索引1）
        E[0,2,3,1] = 1
        E[0,3,2,1] = 1
        E[0,3,4,1] = 1
        E[0,4,3,1] = 1
        E[0,4,5,1] = 1
        E[0,5,4,1] = 1
        E[0,5,6,1] = 1
        E[0,6,5,1] = 1

        print(X[0].argmax(dim=-1))
        print(E[0].argmax(dim=-1)) """

        # -------------------- 步骤 1：识别连通分量 --------------------
        # 从边特征中计算邻接矩阵
        edge_types = E.argmax(dim=-1)  # (batch_size, num_nodes, num_nodes)
        edge_exists = edge_types > 0  # (batch_size, num_nodes, num_nodes)
        edge_exists = edge_exists & node_mask.unsqueeze(1) & node_mask.unsqueeze(2)
        edge_exists = edge_exists.to(device)

        # 初始化每个节点的连通分量标签
        component_labels = torch.arange(num_nodes, device=device).unsqueeze(0).expand(batch_size, -1).clone()
        component_labels = component_labels * node_mask.long() + (~node_mask).long() * num_nodes  # 无效节点标签为 num_nodes

        # 使用并行的标签传播算法识别连通分量
        adjacency = edge_exists  # (batch_size, num_nodes, num_nodes)
        for _ in range(num_nodes):
            # 获取邻居的标签最小值
            neighbor_labels = torch.where(adjacency, component_labels.unsqueeze(2), num_nodes)
            min_neighbor_labels, _ = neighbor_labels.min(dim=1)
            # 更新标签
            component_labels = torch.min(component_labels, min_neighbor_labels)
        component_labels = component_labels * node_mask.long() + (~node_mask).long() * num_nodes  # 无效节点标签为 num_nodes

        # -------------------- 步骤 2：计算每个图的连通分量数量 --------------------
        # 为了避免循环，我们对标签进行偏移，使得每个图的标签不重叠
        offsets = (torch.arange(batch_size, device=device) * (num_nodes + 1)).unsqueeze(1)  # (batch_size, 1)
        component_labels_offset = component_labels + offsets  # (batch_size, num_nodes)

        # 获取每个图的有效节点的标签
        valid_labels = component_labels_offset.view(-1)[node_mask.view(-1)]
        # 计算所有有效标签的唯一值
        unique_labels, inverse_indices = torch.unique(valid_labels, return_inverse=True)
        # 计算每个图的连通分量数量
        graph_indices = (unique_labels // (num_nodes + 1)).long()  # (num_unique_components,)
        num_components_per_graph = torch.zeros(batch_size, device=device, dtype=torch.long)
        num_components_per_graph.scatter_add_(0, graph_indices, torch.ones_like(graph_indices))

        # 标记需要连接的图
        graphs_to_connect = num_components_per_graph > 1  # (batch_size,)

        if graphs_to_connect.sum() == 0:
            # 所有图已经是连通的
            return X, E

        # -------------------- 步骤 3：选择连接节点 --------------------
        # 计算每个节点的可用价键数
        atom_types = X.argmax(dim=-1)  # (batch_size, num_nodes)
        atom_max_valence = torch.tensor(self.dataset_info.valencies, device=device)  # [4, 3, 2, 1] self.valencies
        max_valence = atom_max_valence[atom_types]  # (batch_size, num_nodes)

        bond_orders =  self.dataset_info.edge_consume.to(device=device)  # 对应于边类型索引
        edge_bond_orders = bond_orders[edge_types]  # (batch_size, num_nodes, num_nodes)
        current_degree = edge_bond_orders.sum(dim=2)  # (batch_size, num_nodes)

        available_valence = max_valence - current_degree  # (batch_size, num_nodes)
        available_valence = torch.clamp(available_valence, min=0)  # 确保非负
        available_nodes_mask = (available_valence > 0) & node_mask  # (batch_size, num_nodes)

        batch_indices, node_indices_in_batch = torch.nonzero(available_nodes_mask, as_tuple=True)
        global_node_indices = batch_indices * num_nodes + node_indices_in_batch  # (num_available_nodes,)

        # 获取可用节点的连通分量标签和原子类型
        component_labels_available = component_labels_offset.view(-1)[global_node_indices]  # (num_available_nodes,)
        atom_types_available = atom_types.view(-1)[global_node_indices]  # (num_available_nodes,)

        # 记录可用节点的数量
        num_available_nodes = component_labels_available.size(0)

        if num_available_nodes == 0:
            # 没有可用的节点，无法连接
            return X, E

        # 获取每个连通分量的节点
        unique_comp_labels, inverse_indices = torch.unique(component_labels_available, return_inverse=True)  # unique_comp_labels: (num_comps,)
        num_comps = unique_comp_labels.size(0)

        # 计算每个连通分量的节点数
        #comp_sizes = torch.bincount(inverse_indices, minlength=num_comps)  # (num_comps,)

        # 按照节点类型概率，计算节点选择的权重
        node_type_probs = self.dataset_info.node_types.to(device)  # (num_node_types,) self.node_types
        node_type_probs = node_type_probs / node_type_probs.sum()  # 归一化
        node_probs = node_type_probs[atom_types_available]  # (num_available_nodes,)

        # 计算每个连通分量的节点选择概率
        #node_probs_normalized = node_probs / comp_sizes[inverse_indices].float()  # (num_available_nodes,)

        # 从每个连通分量中随机选择一个节点
        # 使用 scatter 来构建一个二维概率分布
        comp_node_probs = torch.zeros((num_comps, num_available_nodes), device=device)
        #comp_node_probs.scatter_(1, inverse_indices.unsqueeze(0), node_probs_normalized.unsqueeze(0))
        #comp_node_probs = torch.where(comp_node_probs > 0, comp_node_probs, torch.tensor(0.0, device=device))
        # 对每个连通分量的概率进行归一化
        #comp_node_probs = comp_node_probs / comp_node_probs.sum(dim=1, keepdim=True)
        comp_node_probs[inverse_indices, torch.arange(num_available_nodes, device=device)] = node_probs

        comp_probs_sum = comp_node_probs.sum(dim=1, keepdim=True)

        # 处理零值以避免除以零
        comp_probs_sum[comp_probs_sum == 0] = 1.0

        comp_node_probs = comp_node_probs / comp_probs_sum

        # 确保概率值合法
        comp_node_probs = torch.clamp(comp_node_probs, min=torch.tensor(0.0, device=device), max=torch.tensor(1.0, device=device))

        sampled_indices_in_comp = torch.multinomial(comp_node_probs, num_samples=1).squeeze(1)
        selected_indices = sampled_indices_in_comp

        selected_node_indices = node_indices_in_batch[selected_indices]
        selected_batch_indices = batch_indices[selected_indices]

        # -------------------- 步骤 4：构建需要连接的节点对 --------------------
        selected_nodes = torch.stack([selected_batch_indices, selected_node_indices], dim=1)
        selected_nodes = selected_nodes[graphs_to_connect[selected_nodes[:, 0]]]
        if selected_nodes.size(0) == 0:
            return X, E

        # 获取每个图的索引
        graph_indices = selected_nodes[:, 0]
        unique_graphs, graph_positions = torch.unique(graph_indices, return_inverse=True)
        num_unique_graphs = unique_graphs.size(0)

        # 计算每个图的连通分量数量
        comps_per_graph = torch.bincount(graph_positions, minlength=num_unique_graphs)

        # 最大的连通分量数量
        max_comps = comps_per_graph.max().item()

        # 按图索引对节点进行排序
        sorted_indices = torch.argsort(selected_nodes[:, 0])
        selected_nodes_sorted = selected_nodes[sorted_indices]
        graph_positions_sorted = graph_positions[sorted_indices]

        # 计算每个图在 selected_nodes_sorted 中的起始索引
        cumulative_counts = torch.cumsum(torch.cat([torch.tensor([0], device=device), comps_per_graph[:-1]]), dim=0)

        # 计算在每个图内的位置
        positions_in_graph = torch.arange(selected_nodes.size(0), device=device) - cumulative_counts[graph_positions_sorted]

        # 创建填充的节点矩阵
        node_indices_padded = torch.full((num_unique_graphs, max_comps), -1, device=device, dtype=torch.long)
        node_indices_padded[graph_positions_sorted, positions_in_graph] = selected_nodes_sorted[:, 1]

        # 构建需要连接的节点对
        node_u = node_indices_padded[:, :-1].reshape(-1)
        node_v = node_indices_padded[:, 1:].reshape(-1)
        batch_indices_final = unique_graphs.unsqueeze(1).expand(-1, max_comps - 1).reshape(-1)

        # 去除无效的节点对
        valid_mask = (node_u != -1) & (node_v != -1)
        node_u = node_u[valid_mask]
        node_v = node_v[valid_mask]
        batch_indices_final = batch_indices_final[valid_mask]

        if node_u.size(0) == 0:
            return X, E

        # -------------------- 步骤 5：为每条边随机选择边类型 --------------------
        edge_types_prob = torch.tensor([1, 0, 0, 0]).to(device)#self.dataset_info.edge_types[1:].to(device)#self.dataset_info.edge_types[1:].to(device)  # (4,) self.edge_types
        edge_types_prob = edge_types_prob / edge_types_prob.sum()
        num_edges_to_add = node_u.size(0)
        edge_types_sampled = torch.multinomial(edge_types_prob, num_edges_to_add, replacement=True) + 1  # 边类型索引（1到3）

        # -------------------- 步骤 6：更新边特征张量 --------------------
        E_connected = E.clone()
        # 使用one-hot编码
        edge_types_onehot = F.one_hot(edge_types_sampled, num_classes=self.Edim_output).long()  # (num_edges_to_add, 5)

        # 更新E_connected
        E_connected[batch_indices_final, node_u, node_v, :] = edge_types_onehot
        E_connected[batch_indices_final, node_v, node_u, :] = edge_types_onehot  # 保持对称性
        return X, E_connected


    def connect_components234(self, X, E, node_mask): #两两
        batch_size, num_nodes, _ = X.size()
        device = X.device

        # 初始化边特征张量
        E_connected = E.clone()

        # -------------------- 循环，直到所有连通分量连接成一个 --------------------
        while True:
            # -------------------- 步骤 1：识别连通分量 --------------------
            edge_types = E_connected.argmax(dim=-1)
            edge_exists = edge_types > 0
            edge_exists = edge_exists & node_mask.unsqueeze(1) & node_mask.unsqueeze(2)
            edge_exists = edge_exists.to(device)

            component_labels = torch.arange(num_nodes, device=device).unsqueeze(0).expand(batch_size, -1).clone()
            component_labels = component_labels * node_mask.long() + (~node_mask).long() * num_nodes

            for _ in range(num_nodes):
                neighbor_labels = torch.where(edge_exists, component_labels.unsqueeze(2), num_nodes)
                min_neighbor_labels, _ = neighbor_labels.min(dim=1)
                component_labels = torch.min(component_labels, min_neighbor_labels)
            component_labels = component_labels * node_mask.long() + (~node_mask).long() * num_nodes

            # -------------------- 步骤 2：计算每个图的连通分量数量 --------------------
            offsets = (torch.arange(batch_size, device=device) * (num_nodes + 1)).unsqueeze(1)
            component_labels_offset = component_labels + offsets

            valid_labels = component_labels_offset.view(-1)[node_mask.view(-1)]
            unique_labels, inverse_indices = torch.unique(valid_labels, return_inverse=True)
            graph_indices = (unique_labels // (num_nodes + 1)).long()
            num_components_per_graph = torch.zeros(batch_size, device=device, dtype=torch.long)
            num_components_per_graph.scatter_add_(0, graph_indices, torch.ones_like(graph_indices))

            # 标记需要连接的图
            graphs_to_connect = num_components_per_graph > 1

            if graphs_to_connect.sum() == 0:
                # 所有图已经是连通的
                break  # 退出循环

            # -------------------- 步骤 3：选择连接节点 --------------------
            atom_types = X.argmax(dim=-1)
            atom_max_valence = torch.tensor(self.dataset_info.valencies, device=device)
            max_valence = atom_max_valence[atom_types]

            bond_orders = self.dataset_info.edge_consume.to(device=device)

            edge_bond_orders = bond_orders[edge_types]

            current_degree = edge_bond_orders.sum(dim=2)

            available_valence = max_valence - current_degree
            available_valence = torch.clamp(available_valence, min=0)

            available_nodes_mask = (available_valence > 0) & node_mask

            batch_indices, node_indices_in_batch = torch.nonzero(available_nodes_mask, as_tuple=True)
            global_node_indices = batch_indices * num_nodes + node_indices_in_batch

            component_labels_available = component_labels_offset.view(-1)[global_node_indices]
            atom_types_available = atom_types.view(-1)[global_node_indices]

            num_available_nodes = component_labels_available.size(0)

            if num_available_nodes == 0:
                # 没有可用的节点，无法连接
                break  # 退出循环

            unique_comp_labels, inverse_indices = torch.unique(component_labels_available, return_inverse=True)
            num_comps = unique_comp_labels.size(0)

            if num_comps == batch_size:
                # 每个图只有一个可用的连通分量，无需继续连接
                break

            node_type_probs = self.dataset_info.node_types.to(device)
            node_type_probs = node_type_probs / node_type_probs.sum()
            node_probs = node_type_probs[atom_types_available]

            comp_node_probs = torch.zeros(num_comps, num_available_nodes, device=device)
            comp_node_probs[inverse_indices, torch.arange(num_available_nodes, device=device)] = node_probs

            comp_probs_sum = comp_node_probs.sum(dim=1, keepdim=True)
            comp_probs_sum[comp_probs_sum == 0] = 1.0
            comp_node_probs = comp_node_probs / comp_probs_sum
            comp_node_probs = torch.clamp(comp_node_probs, min=0.0, max=1.0)

            sampled_indices_in_comp = torch.multinomial(comp_node_probs, num_samples=1).squeeze(1)
            selected_indices = sampled_indices_in_comp

            selected_node_indices = node_indices_in_batch[selected_indices]
            selected_batch_indices = batch_indices[selected_indices]

            # -------------------- 步骤 4：构建需要连接的节点对 --------------------
            selected_nodes = torch.stack([selected_batch_indices, selected_node_indices], dim=1)
            selected_nodes = selected_nodes[graphs_to_connect[selected_nodes[:, 0]]]
            if selected_nodes.size(0) == 0:
                break

            # 获取每个图的索引
            graph_indices = selected_nodes[:, 0]
            unique_graphs, graph_positions = torch.unique(graph_indices, return_inverse=True)
            num_unique_graphs = unique_graphs.size(0)

            # 计算每个图的连通分量数量
            comps_per_graph = torch.bincount(graph_positions, minlength=num_unique_graphs)

            # 按照连通分量数量降序排列
            sorted_graph_indices = torch.argsort(-comps_per_graph)
            unique_graphs = unique_graphs[sorted_graph_indices]
            comps_per_graph = comps_per_graph[sorted_graph_indices]

            # 最大的连通分量数量
            max_comps = comps_per_graph.max().item()

            # 创建填充的节点矩阵
            node_indices_padded = torch.full((num_unique_graphs, max_comps), -1, device=device, dtype=torch.long)

            cumulative_counts = torch.cumsum(torch.cat([torch.tensor([0], device=device), comps_per_graph[:-1]]), dim=0)
            positions_in_graph = torch.arange(selected_nodes.size(0), device=device) - cumulative_counts[graph_positions]

            node_indices_padded[graph_positions, positions_in_graph] = selected_nodes[:, 1]

            # 构建需要连接的节点对
            node_u = node_indices_padded[:, :-1].reshape(-1)
            node_v = node_indices_padded[:, 1:].reshape(-1)
            batch_indices_final = unique_graphs.unsqueeze(1).expand(-1, max_comps - 1).reshape(-1)

            # 去除无效的节点对
            valid_mask = (node_u != -1) & (node_v != -1)

            # 检查 valid_mask 是否包含至少一个 True 值
            if not valid_mask.any().item():
                break

            node_u = node_u[valid_mask]
            node_v = node_v[valid_mask]
            batch_indices_final = batch_indices_final[valid_mask]

            # 检查索引是否在合法范围内
            if node_u.min() < 0 or node_u.max() >= num_nodes or node_v.min() < 0 or node_v.max() >= num_nodes:
                print("索引超出范围，node_u 或 node_v 包含非法值。")
                break

            # -------------------- 步骤 5：为每条边随机选择边类型（只允许单键） --------------------
            edge_types_sampled = torch.ones(node_u.size(0), device=device, dtype=torch.long)  # 边类型索引为1（单键）

            # -------------------- 步骤 6：更新边特征张量 --------------------
            edge_types_onehot = F.one_hot(edge_types_sampled, num_classes=E.size(-1)).long()

            # 确保索引操作不会越界
            try:
                E_connected[batch_indices_final, node_u, node_v, :] = edge_types_onehot
                E_connected[batch_indices_final, node_v, node_u, :] = edge_types_onehot  # 保持对称性
            except IndexError as e:
                print("索引错误：", e)
                break

        return X, E_connected

    def connect_components753(self, X, E, node_mask): #两两
        batch_size, num_nodes, _ = X.size()
        device = X.device

        E_connected = E.clone()

        for batch_idx in range(batch_size):
            # 获取当前图的节点掩码
            node_mask_b = node_mask[batch_idx]
            valid_nodes = node_mask_b.nonzero(as_tuple=False).squeeze(1)
            if valid_nodes.numel() == 0:
                continue

            # 提取当前图的节点和边特征
            X_b = X[batch_idx, valid_nodes]
            E_b = E_connected[batch_idx, valid_nodes][:, valid_nodes]

            # 构建 NetworkX 图
            G = nx.Graph()
            num_valid_nodes = valid_nodes.size(0)
            G.add_nodes_from(range(num_valid_nodes))

            # 添加边
            edge_indices = (E_b.argmax(dim=-1) > 0).nonzero(as_tuple=False)
            for edge in edge_indices:
                u, v = edge[:2]
                G.add_edge(u.item(), v.item())

            # 获取连通分量
            components = list(nx.connected_components(G))
            if len(components) <= 1:
                continue  # 图已经是连通的

            # 计算每个节点的可用价键数
            atom_types = X_b.argmax(dim=-1)
            atom_max_valence = torch.tensor(self.dataset_info.valencies, device=device)
            max_valence = atom_max_valence[atom_types]
            bond_orders = self.dataset_info.edge_consume.to(device=device)
            edge_types = E_b.argmax(dim=-1)
            edge_bond_orders = bond_orders[edge_types]
            current_degree = edge_bond_orders.sum(dim=1)
            available_valence = max_valence - current_degree
            available_valence = available_valence.cpu()  # 转移到 CPU 进行处理
            available_valence = torch.clamp(available_valence, min=0)

            # 从每个连通分量中选择一个可用节点
            component_nodes = []
            for comp in components:
                comp_nodes = list(comp)
                # 筛选有可用价键的节点
                comp_available_nodes = [n for n in comp_nodes if available_valence[n] > 0]
                if not comp_available_nodes:
                    continue  # 该连通分量没有可用节点
                # 随机选择一个节点
                selected_node = random.choice(comp_available_nodes)
                component_nodes.append(selected_node)

            # 如果可用节点不足以连接，跳过
            if len(component_nodes) <= 1:
                continue

            # 成对连接节点
            for i in range(len(component_nodes) - 1):
                u = component_nodes[i]
                v = component_nodes[i + 1]
                # 检查节点的可用价键数
                if available_valence[u] > 0 and available_valence[v] > 0:
                    # 添加单键
                    E_connected[batch_idx, valid_nodes[u], valid_nodes[v], :] = F.one_hot(torch.tensor(1, device=device), num_classes=E.size(-1))
                    E_connected[batch_idx, valid_nodes[v], valid_nodes[u], :] = F.one_hot(torch.tensor(1, device=device), num_classes=E.size(-1))
                    # 更新可用价键数
                    available_valence[u] -= 1
                    available_valence[v] -= 1
                else:
                    # 如果任何一个节点没有可用价键，跳过连接
                    continue

        return X, E_connected

    def connect_components(self, X, E, node_mask): #选最大的
        """
        在批量中的每个图中连接不相连的连通分量。

        参数：
            X: 形状为 (batch_size, num_nodes, num_node_features) 的张量
                节点特征张量。
            E: 形状为 (batch_size, num_nodes, num_nodes, num_edge_features) 的张量
                边特征张量。
            node_mask: 形状为 (batch_size, num_nodes) 的张量
                指示每个图中有效节点的掩码。

        返回：
            X_connected: 更新后的节点特征张量。
            E_connected: 添加连接后的边特征张量。
        """
        batch_size, num_nodes, node_classes = X.size()
        device = X.device
        # -------------------- 步骤 1：识别连通分量 --------------------
        # 从边特征中计算邻接矩阵
        edge_types = E.argmax(dim=-1)  # (batch_size, num_nodes, num_nodes)
        edge_exists = edge_types > 0  # (batch_size, num_nodes, num_nodes)
        edge_exists = edge_exists & node_mask.unsqueeze(1) & node_mask.unsqueeze(2)
        edge_exists = edge_exists.to(device)

        # 初始化每个节点的连通分量标签
        component_labels = torch.arange(num_nodes, device=device).unsqueeze(0).expand(batch_size, -1).clone()
        component_labels = component_labels * node_mask.long() + (~node_mask).long() * num_nodes  # 无效节点标签为 num_nodes

        # 使用并行的标签传播算法识别连通分量
        adjacency = edge_exists  # (batch_size, num_nodes, num_nodes)
        for _ in range(num_nodes):
            # 获取邻居的标签最小值
            neighbor_labels = torch.where(adjacency, component_labels.unsqueeze(2), num_nodes)
            min_neighbor_labels, _ = neighbor_labels.min(dim=1)
            # 更新标签
            component_labels = torch.min(component_labels, min_neighbor_labels)
        component_labels = component_labels * node_mask.long() + (~node_mask).long() * num_nodes  # 无效节点标签为 num_nodes

        # -------------------- 步骤 2：计算每个图的连通分量数量 --------------------
        # 为了避免循环，我们对标签进行偏移，使得每个图的标签不重叠
        offsets = (torch.arange(batch_size, device=device) * (num_nodes + 1)).unsqueeze(1)  # (batch_size, 1)
        component_labels_offset = component_labels + offsets  # (batch_size, num_nodes)

        # 获取每个图的有效节点的标签
        valid_labels = component_labels_offset.view(-1)[node_mask.view(-1)]
        # 计算所有有效标签的唯一值
        unique_labels, inverse_indices = torch.unique(valid_labels, return_inverse=True)
        # 计算每个图的连通分量数量
        graph_indices = (unique_labels // (num_nodes + 1)).long()  # (num_unique_components,)
        num_components_per_graph = torch.zeros(batch_size, device=device, dtype=torch.long)
        num_components_per_graph.scatter_add_(0, graph_indices, torch.ones_like(graph_indices))

        # 标记需要连接的图
        graphs_to_connect = num_components_per_graph > 1  # (batch_size,)

        if graphs_to_connect.sum() == 0:
            # 所有图已经是连通的
            return X, E

        # -------------------- 步骤 3：选择连接节点 --------------------
        # 计算每个节点的可用价键数
        atom_types = X.argmax(dim=-1)  # (batch_size, num_nodes)
        atom_max_valence = torch.tensor(self.dataset_info.valencies, device=device)  # [4, 3, 2, 1]
        max_valence = atom_max_valence[atom_types]  # (batch_size, num_nodes)

        bond_orders = self.dataset_info.edge_consume.to(device=device)  # 对应于边类型索引
        edge_bond_orders = bond_orders[edge_types]  # (batch_size, num_nodes, num_nodes)
        current_degree = edge_bond_orders.sum(dim=2)  # (batch_size, num_nodes)

        available_valence = max_valence - current_degree  # (batch_size, num_nodes)
        available_valence = torch.clamp(available_valence, min=0)  # 确保非负
        available_nodes_mask = (available_valence > 0) & node_mask  # (batch_size, num_nodes)

        batch_indices, node_indices_in_batch = torch.nonzero(available_nodes_mask, as_tuple=True)
        global_node_indices = batch_indices * num_nodes + node_indices_in_batch  # (num_available_nodes,)

        # 获取可用节点的连通分量标签和原子类型
        component_labels_available = component_labels_offset.view(-1)[global_node_indices]  # (num_available_nodes,)
        atom_types_available = atom_types.view(-1)[global_node_indices]  # (num_available_nodes,)
        available_valence_available_nodes = available_valence.view(-1)[global_node_indices]  # (num_available_nodes,)

        # 记录可用节点的数量
        num_available_nodes = component_labels_available.size(0)

        if num_available_nodes == 0:
            # 没有可用的节点，无法连接
            return X, E

        # 获取每个连通分量的节点
        unique_comp_labels, inverse_indices = torch.unique(component_labels_available, return_inverse=True)  # unique_comp_labels: (num_comps,)
        num_comps = unique_comp_labels.size(0)

        # -------------------- 修改的部分：选择每个连通分量中可用价键数最高的节点 --------------------
        selected_indices = []
        for comp_idx in range(num_comps):
            # 获取当前连通分量中的节点索引
            nodes_in_comp = (inverse_indices == comp_idx).nonzero(as_tuple=False).view(-1)
            if nodes_in_comp.numel() == 0:
                continue
            # 获取该连通分量中节点的可用价键数
            valence_in_comp = available_valence_available_nodes[nodes_in_comp]
            max_valence = valence_in_comp.max()
            # 找到具有最大可用价键数的节点
            max_valence_nodes = nodes_in_comp[valence_in_comp == max_valence]
            # 如果有多个节点，随机选择一个
            rand_idx = torch.randint(len(max_valence_nodes), (), device=device)
            selected_node_idx = max_valence_nodes[rand_idx]
            selected_indices.append(selected_node_idx)

        if len(selected_indices) == 0:
            # 没有可用的节点，无法连接
            return X, E

        selected_indices = torch.tensor(selected_indices, device=device)

        selected_node_indices = node_indices_in_batch[selected_indices]
        selected_batch_indices = batch_indices[selected_indices]

        # -------------------- 步骤 4：构建需要连接的节点对 --------------------
        selected_nodes = torch.stack([selected_batch_indices, selected_node_indices], dim=1)  # (num_selected_nodes, 2)

        # 确保 graphs_to_connect 是一维张量
        graphs_to_connect = graphs_to_connect.view(-1)  # (batch_size,)

        # 获取 selected_nodes 的批次索引，并转换为长整型
        selected_nodes_batch_indices = selected_nodes[:, 0].long()

        # 创建掩码，形状为 (num_selected_nodes,)
        graphs_to_connect_mask = graphs_to_connect[selected_nodes_batch_indices]

        # 使用掩码索引 selected_nodes
        selected_nodes = selected_nodes[graphs_to_connect_mask]

        if selected_nodes.size(0) == 0:
            return X, E


        # 获取每个图的索引
        graph_indices = selected_nodes[:, 0]
        unique_graphs, graph_positions = torch.unique(graph_indices, return_inverse=True)
        num_unique_graphs = unique_graphs.size(0)

        # 计算每个图的连通分量数量
        comps_per_graph = torch.bincount(graph_positions, minlength=num_unique_graphs)

        # 最大的连通分量数量
        max_comps = comps_per_graph.max().item()

        # 按图索引对节点进行排序
        sorted_indices = torch.argsort(selected_nodes[:, 0])
        selected_nodes_sorted = selected_nodes[sorted_indices]
        graph_positions_sorted = graph_positions[sorted_indices]

        # 计算每个图在 selected_nodes_sorted 中的起始索引
        cumulative_counts = torch.cumsum(torch.cat([torch.tensor([0], device=device), comps_per_graph[:-1]]), dim=0)

        # 计算在每个图内的位置
        positions_in_graph = torch.arange(selected_nodes.size(0), device=device) - cumulative_counts[graph_positions_sorted]

        # 创建填充的节点矩阵
        node_indices_padded = torch.full((num_unique_graphs, max_comps), -1, device=device, dtype=torch.long)
        node_indices_padded[graph_positions_sorted, positions_in_graph] = selected_nodes_sorted[:, 1]

        # 构建需要连接的节点对
        node_u = node_indices_padded[:, :-1].reshape(-1)
        node_v = node_indices_padded[:, 1:].reshape(-1)
        batch_indices_final = unique_graphs.unsqueeze(1).expand(-1, max_comps - 1).reshape(-1)

        # 去除无效的节点对
        valid_mask = (node_u != -1) & (node_v != -1)
        node_u = node_u[valid_mask]
        node_v = node_v[valid_mask]
        batch_indices_final = batch_indices_final[valid_mask]

        if node_u.size(0) == 0:
            return X, E

        """ # -------------------- 修改的步骤 4：构建需要连接的节点对 --------------------
        selected_comp_labels = component_labels_available[selected_indices]

        # 构建连通分量标签到选定节点索引的映射
        comp_label_to_node = {}
        for idx, comp_label in zip(range(len(selected_indices)), selected_comp_labels.tolist()):
            comp_label_to_node[comp_label] = idx

        # 获取连通分量及其对应的选定节点
        components = list(comp_label_to_node.keys())
        nodes = [comp_label_to_node[comp_label] for comp_label in components]

        if len(nodes) <= 1:
            # 只有一个连通分量或没有可用节点，无需连接
            return X, E

        edges_to_add = []
        for i in range(len(nodes) - 1):
            u_idx = nodes[i]
            v_idx = nodes[i + 1]
            u_node = selected_node_indices[u_idx]
            v_node = selected_node_indices[v_idx]
            batch_idx = selected_batch_indices[u_idx]  # 所有节点都属于同一个图
            edges_to_add.append((batch_idx.item(), u_node.item(), v_node.item()))

        if len(edges_to_add) == 0:
            return X, E

        batch_indices_final = torch.tensor([edge[0] for edge in edges_to_add], device=device, dtype=torch.long)
        node_u = torch.tensor([edge[1] for edge in edges_to_add], device=device, dtype=torch.long)
        node_v = torch.tensor([edge[2] for edge in edges_to_add], device=device, dtype=torch.long) """

        # -------------------- 步骤 5：为每条边随机选择边类型 --------------------
        edge_types_prob = torch.tensor([1, 0, 0, 0], device=device).float()
        edge_types_prob = edge_types_prob / edge_types_prob.sum()
        num_edges_to_add = node_u.size(0)
        edge_types_sampled = torch.multinomial(edge_types_prob, num_edges_to_add, replacement=True) + 1  # 边类型索引（1到4）

        # -------------------- 步骤 5：为每条边随机选择边类型（只允许单键） --------------------
        #edge_types_sampled = torch.ones(len(edges_to_add), device=device, dtype=torch.long)  # 边类型索引为1（单键）

        # -------------------- 步骤 6：更新边特征张量 --------------------
        E_connected = E.clone()
        # 使用one-hot编码
        edge_types_onehot = F.one_hot(edge_types_sampled, num_classes=self.Edim_output).long()  # (num_edges_to_add, num_edge_features)

        # 更新E_connected
        E_connected[batch_indices_final, node_u, node_v, :] = edge_types_onehot
        E_connected[batch_indices_final, node_v, node_u, :] = edge_types_onehot  # 保持对称性
        return X, E_connected


    """ def replace_supernode_with_ring(self, subX: torch.Tensor, subE: torch.Tensor, ring_types: dict):
        device = subX.device
        n = subX.size(0)

        super_mask = (subX == 4).nonzero(as_tuple=False)
        if super_mask.numel() == 0:
            # 无超点 => 原样返回
            return subX, subE, n

        # 只处理单个超点: super_idx = super_mask[0][0].item()  (如果多个，需要更完整逻辑)
        super_idx = super_mask[0].item()

        # 2) 找到超点与外部节点
        external_neighbors = []
        for j in range(n):
            if j == super_idx:
                continue
            # 若 subE[super_idx,j] > 0 表示存在一条边(可视自己定义)
            if subE[super_idx,j] > 0:
                external_neighbors.append(j)

        # 3) 按照 ring_types 的分布采样一个环 SMILES
        chosen_smi = self._sample_ring_smiles(ring_types)

        # 4) 根据 chosen_smi => 构建 ring 的节点与边(极简示例)
        ring_labels = self._parse_ring_smiles(chosen_smi)  # e.g. [1,1,3,1,1]
        ring_size = len(ring_labels)
        if ring_size == 0:
            ring_labels = [0]  # 0-> C
            ring_size=1

        # ringE => (ring_size, ring_size), 全 SINGLE=1
        ringE = torch.zeros((ring_size, ring_size), dtype=torch.long, device=device)
        for k in range(ring_size):
            nxt = (k+1) % ring_size
            ringE[k,nxt] = 1
            ringE[nxt,k] = 1

        # 5) 移除原超点 => keep others
        keep_idx = [x for x in range(n) if x != super_idx]
        subX_noSuper = subX[keep_idx]
        subE_noSuper = subE[keep_idx][:, keep_idx]

        ringX= torch.tensor(ring_labels, dtype=torch.long, device=device)  # shape(ring_size,)
        # 6) 合并 ring => newX, newE
        # newX => cat(subX_noSuper, ringX)
        newX = torch.cat([subX_noSuper, ringX], dim=0)
        new_n = subX_noSuper.size(0) + ring_size

        newE = torch.zeros((new_n,new_n), dtype=torch.long, device=device)
        newE[:subX_noSuper.size(0), :subX_noSuper.size(0)] = subE_noSuper
        # place ringE => offset
        offset = subX_noSuper.size(0)
        newE[offset:offset+ring_size, offset:offset+ring_size] = ringE

        # 7) 连接外部 neighbors => ring(0) => SINGLE=1
        if len(external_neighbors)==0:
            # 超点无边 => 直接是一个独立环
            pass
        else:
            ring0_idx = offset  # ring第0个节点
            for enb in external_neighbors:
                real_j = keep_idx.index(enb)
                newE[real_j, ring0_idx] = 1
                newE[ring0_idx, real_j] = 1

        return newX, newE, new_n

    def _sample_ring_smiles(self, ring_types):
        keys = list(ring_types.keys())
        vals = list(ring_types.values())
        s = sum(vals)
        r = random.uniform(0,s)
        accum=0
        for k,v in zip(keys,vals):
            accum+=v
            if accum>=r:
                return k
        return keys[-1]

    def _parse_ring_smiles(self, ring_smi):
        label_map = {'C':0, 'N':1, 'O':2, 'F':3}
        arr=[]
        for ch in ring_smi:
            if ch in label_map:
                arr.append(label_map[ch])
        # 如果解析结果 arr=[] => 说明该环Smi里没有 'C','N','O','F'
        if len(arr)==0:
            # 防御处理,比如：
            # 1) 默认填一个C: arr=[0]
            # 2) raise Exception("Parsed empty ring!")
            arr = [0]  # 默认让它至少有一个 C
        return arr

    def convert_feature_with_supernode(self, X, E, n_nodes, ring_types):
        batch_size= X.size(0)
        converted_X = []  
        converted_E = []  
        molecule_list= []

        for i in range(batch_size):
            n = n_nodes[i].item()
            newX= X[i,:n]       # (n, feat_dim)
            newE= E[i,:n,:n]    # (n, n)

            while (newX == 4).any():
                newX, newE, new_n = self.replace_supernode_with_ring(newX, newE, ring_types)
                #newX,newE,new_n= self.replace_supernode_with_ring(subX, subE, ring_types)
            #else:
                #newX,newE,new_n= subX, subE, n

            # 最后 => atom_types => newX.argmax => shape(new_n,)
            #       edge_types => newE => shape(new_n,new_n)
            #   e.g. atom_types: size(new_n)
            atom_types= newX.cpu()
            edge_types= newE.cpu()

            molecule_list.append([atom_types, edge_types])
            converted_X.append(atom_types) 
            converted_E.append(edge_types)  

           
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
        # 找到转换后最大的节点数
        max_n = max([x.size(0) for x in converted_X])
        
        # 填充 atom_types
        padded_X = []
        for x in converted_X:
            padded = self.pad_tensor(x, max_n, pad_value=0)  # 假设 pad_value=0 对应 'C'
            padded_X.append(padded)
        padded_X = torch.stack(padded_X, dim=0)  # shape (batch_size, max_n)
        
        # 填充 edge_types
        padded_E = []
        for E in converted_E:
            padded = self.pad_tensor(E, max_n, pad_value=0)  # pad with 0 (noEdge)
            padded_E.append(padded)
        padded_E = torch.stack(padded_E, dim=0)  # shape (batch_size, max_n, max_n)
        
        return padded_X, padded_E """
    

    def graph_to_smiles(self, origin_x: torch.Tensor, 
                        origin_e: torch.Tensor,
                        n_nodes: int) -> str:
        """
        origin_x: (n_nodes,) in [0..3]
        origin_e: (n_nodes, n_nodes) in [0..4]
        返回: SMILES 字符串
        
        注意:
        - 不检查图的有效性或连通性
        - 如果结构无效, 会在 Sanitization 时报错
        """
        from rdkit import Chem
        from rdkit.Chem import RWMol
        
        if n_nodes==0:
            # 空分子 => 返回空字符串或其他占位
            return ""
        
        # 1) 构造 RWMol
        rwmol = RWMol()
        
        # 记录 old->new atom idx
        old_to_new = []
        for i in range(n_nodes):
            lbl = origin_x[i].item()
            sym = self.dataset_info.label_to_symbol.get(lbl, "C")  # fallback => "C"
            a = Chem.Atom(sym)
            # RDKit 中可能要设置初始价等, 这里省略
            new_idx = rwmol.AddAtom(a)
            old_to_new.append(new_idx)
        
        # 2) 添加键
        #   遍历上三角 i<j, 若 origin_e[i,j] in [1..4], 则 addBond
        #   bond type => label_to_bondtype
        for i in range(n_nodes):
            for j in range(i+1, n_nodes):
                bt_lbl = origin_e[i,j].item()
                if bt_lbl>0:
                    # 1..4
                    bond_t = self.dataset_info.label_to_bondtype.get(bt_lbl, rdchem.BondType.SINGLE)
                    rwmol.AddBond(old_to_new[i], old_to_new[j], bond_t)
        
        # 3) 生成 mol
        mol = rwmol.GetMol()
        
        # 4) Sanitization / SMILES
        #   如果结构无效或价态不对, 可能抛异常
        try:
            Chem.SanitizeMol(mol)  # 可能产生异常
            smi = Chem.MolToSmiles(mol, canonical=True)
        except Exception as e:
            # 如果失败, 返回空或异常提示
            smi = f"[InvalidMol_{str(e)}]"
        return smi


    #####################################################
    # 3) 批量处理 => 返回 [smiles1, smiles2, ...]
    #####################################################

    def decode_origin_graphs_to_smiles(self, origin_X: torch.Tensor,
                                    origin_E: torch.Tensor,
                                    n_nodes: torch.Tensor):
        """
        origin_X: (batch_size, max_n), label in [0..3]
        origin_E: (batch_size, max_n, max_n), label in [0..4]
        n_nodes: (batch_size,)
        返回: list of smiles (batch_size,)
        """
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
        """
        返回 (node_labels, edge_labels) 
        - node_labels: [n], each in [0..3]
        - edge_labels: (n,n), each in [0..4], 0=无边,1=单,etc.
        示例仅做极简: 读 RDKit => parse
        """
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
        
        # 构造 edge matrix
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


    ########################################################
    # 2) 解码单个图 => origin_X, origin_E
    ########################################################
    def decode_single_graph(self, subX, subE, n):
        """
        subX: (n,) in [0..13]
        subE: (n,n) in [0..4]
        返回: (origin_X, origin_E) 
        - origin_X: (new_n,) in [0..3]  => 仅 C=0, N=1, O=2, F=3
        - origin_E: (new_n,new_n) in [0..4]
        思路:
        - 遍历节点, 如果 label in [0..3], 直接保留
                    如果 label in [4..13], => ring => parse ring smi => ring graph
        - 对于与 supernode相连的外部边 => 需分配给 ring中若干C
        - 组装新的 adjacency
        """
        # 先构建一个 “动态” graph( adjacency ), python list
        # node_info => (label, list_of_edges)
        # 先把普通原子保留
        node_info = []  # 每个条目: { 'lbl': int, 'adj': {} } 
        # 以便后面可插入 ring
        
        old2new = [-1]*n  # old node -> new idx
        new_idx_count = 0
        
        # 记录每个超点 => ring info => 之后再添加
        supernode_data = []
        
        for old_i in range(n):
            lbl = subX[old_i].item()  # int
            if lbl<4:
                # 普通原子 => new node
                node_info.append({'lbl': lbl, 'adj': {}}) 
                old2new[old_i]= new_idx_count
                new_idx_count+=1
            else:
                # ring supernode
                supernode_data.append(old_i)
        
        # 先处理已有的普通节点之间的边
        for old_i in range(n):
            for old_j in range(old_i+1, n):
                b = subE[old_i, old_j].item()
                if b>0:  # 1..4
                    # 两端必须是普通原子 => both old2new != -1
                    ni = old2new[old_i]
                    nj = old2new[old_j]
                    if ni>=0 and nj>=0:
                        # add to adjacency
                        node_info[ni]['adj'][nj]= b
                        node_info[nj]['adj'][ni]= b
        
        # 处理 supernode => ring
        for snode in supernode_data:
            ring_lbl = subX[snode].item()  # in [4..13]
            # parse ring
            ring_smi = self.dataset_info.label_to_ring[ring_lbl]
            r_nodes, r_E = self.parse_ring_smi(ring_smi)  # r_nodes in [0..3], r_E in [r,r], r in [0..4]
            
            # 在新图中新增 ring的多个节点
            ring_base_idx = new_idx_count
            ring_size = len(r_nodes)
            # create them
            for rlbl in r_nodes:
                node_info.append({'lbl': rlbl, 'adj': {}})
            ring_new_indices = list(range(ring_base_idx, ring_base_idx+ring_size))
            new_idx_count += ring_size
            
            # 建立 ring内部边
            for rr_i in range(ring_size):
                for rr_j in range(rr_i+1, ring_size):
                    bb = r_E[rr_i, rr_j].item()
                    if bb>0:
                        # connect ring_new_indices[rr_i] <-> ring_new_indices[rr_j]
                        ni = ring_new_indices[rr_i]
                        nj = ring_new_indices[rr_j]
                        node_info[ni]['adj'][nj]= bb
                        node_info[nj]['adj'][ni]= bb
            
            # supernode 与外部节点 => subE[snode, x] in [1..4]
            # 需将 edges “分配”给 ring中的C
            # gather external edges
            ext_edges = []
            for other in range(n):
                if other!= snode:
                    b = subE[snode, other].item()
                    if b>0:  # supernode->other
                        ext_edges.append((other,b))
            
            # ring中的C(=0 in parse result?), or you define "C=0 => user"
            # 先收集 ring内部 idx of 0 => ring_new_indices[..] where r_nodes[..]=0
            ringC_list = []
            for ii, at_lbl in enumerate(r_nodes):
                if at_lbl==0:  # 0=>C
                    ringC_list.append(ring_new_indices[ii])
            if len(ringC_list)==0:
                # 假设环中没有C => fallback: use ring_new_indices[0]
                ringC_list = [ring_new_indices[0]]
            
            c_count = len(ringC_list)
            # 分配
            used_count = 0
            for (oth,b_lbl) in ext_edges:
                new_oth = old2new[oth]
                if new_oth<0:
                    # 说明 other是supernode => ignore or ring ?
                    # 可能出现 supernode->supernode
                    # 这里演示: 跳过
                    continue
                # pick ringC_list[ used_count % c_count ]
                targetC = ringC_list[ used_count % c_count ]
                used_count+=1
                # add edge
                node_info[targetC]['adj'][new_oth]= b_lbl
                node_info[new_oth]['adj'][targetC]= b_lbl
        
        # 至此，node_info记录了所有普通节点&环节点及其边
        # 构造 origin_X, origin_E
        new_n = len(node_info)
        origin_X = torch.zeros((new_n,), dtype=torch.long)
        origin_E = torch.zeros((new_n,new_n), dtype=torch.long)
        
        for i, info in enumerate(node_info):
            origin_X[i] = info['lbl']  # 0..3
        for i, info in enumerate(node_info):
            for j, bb in info['adj'].items():
                origin_E[i,j]= bb
        
        return origin_X, origin_E, new_n


    ########################################################
    # 3) 对 batch 处理
    ########################################################
    def decode_batch(self, X, E, n_nodes):
        """
        X: (batch_size, max_n)
        E: (batch_size, max_n, max_n)
        n_nodes: (batch_size,)
        返回 origin_X, origin_E(维度=?), or molecule_list => [[atom_types, edge_types], ...]
        题意: 
        - 先解码每个图 => (origin_X_i, origin_E_i)
        - 记录到 molecule_list
        - “最终得到的origin_X和origin_E的第一个维度仍然是batch_size”，
        说明我们需要再padding到同一维度, new_max_n
        """
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


    ########################################################
    # 4) molecule_list
    ########################################################
    def build_molecule_list(self, X, E, n_nodes):
        """
        返回:
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
        return origin_X, origin_E, molecule_list


    @torch.no_grad()
    def sample_batch(self, batch_id: int, batch_size: int, keep_chain: int, number_chain_steps: int,
                    save_final: int, num_nodes=None):  # 点和边真正的同时，只focus有效节点之间的子图
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
        # 构建节点掩码
        arange = torch.arange(n_max, device=self.device).unsqueeze(0).expand(batch_size, -1)
        node_mask = arange < n_nodes.unsqueeze(1)
        # 采样初始噪声  -- z_T 形状为 (batch_size, n_max, feature_dim)
        z_T = diffusion_utils.sample_discrete_feature_noise(dataset_name=self.dataset_name, limit_dist=self.limit_dist, node_mask=node_mask)
        X, E, y = z_T.X, z_T.E, z_T.y
        assert (E == torch.transpose(E, 1, 2)).all()

        # 定义链的尺寸，用于保存中间结果
        chain_X_size = torch.Size((number_chain_steps + 1, keep_chain, X.size(1)))
        chain_E_size = torch.Size((number_chain_steps + 1, keep_chain, E.size(1), E.size(2)))

        chain_X = torch.zeros(chain_X_size, device=self.device)
        chain_E = torch.zeros(chain_E_size, device=self.device)

        """ # 计算每个图的有效节点数
        valid_nodes_per_graph = n_nodes  # 形状为 (batch_size,)
        max_node_steps = valid_nodes_per_graph.max().item()  # 最大的有效节点数

        # 初始化 t_nodes 为有效节点数的张量，形状为 (batch_size,)
        t_nodes = valid_nodes_per_graph.clone()  # 初始化为有效节点数
        edge_noise_ratio = 0.2
        max_possible_subgraph_edges = (t_nodes * (t_nodes - 1)) // 2
        t_edges = (max_possible_subgraph_edges.float() * edge_noise_ratio).floor()
        # 计算每个图的有效边数
        # 在新的边添加噪声方式下，valid_edges_per_graph 不再是 (n_nodes * (n_nodes - 1)) // 2
        # 而是根据当前的 t_nodes 计算的 t_edges，因此我们在循环中动态计算 t_edges

        # 计算总的时间步数（以最大节点数为准）
        total_steps = max_node_steps

        for step in reversed(range(total_steps + 1)):
            # 对于每个时间步，计算对应的 s_nodes 和 t_nodes
            s_nodes = t_nodes - 1  # 形状为 (batch_size,)
            s_nodes = torch.clamp(s_nodes, min=0)  # 保证至少为0

            # 计算 t_edges 和 s_edges，使用新的计算方式
            #t_edges = t_nodes * (t_nodes - 1) // 2  # t_edges = t_nodes * (t_nodes - 1) / 2
            #s_edges = s_nodes * (s_nodes - 1) // 2
            #s_edges = torch.clamp(s_edges, min=0)  # 保证不小于0 

            # 计算每个图的有效节点数和有效边数
            valid_nodes = valid_nodes_per_graph  # (batch_size,)
            valid_edges = valid_nodes * (valid_nodes - 1) // 2 + 1e-8  # 避免除以零

            # 计算归一化的 s 和 t
            #s_norm_nodes = s_nodes.float() / valid_nodes.float()
            t_norm_nodes = t_nodes.float() / valid_nodes.float()

            #s_norm_edges = s_edges.float() / valid_edges.float()
            t_norm_edges = t_edges.float() / valid_edges.float()

            # 将 s 和 t 转换为张量，形状为 (batch_size, 1)
            s_nodes_tensor = s_nodes.unsqueeze(1).float()
            t_nodes_tensor = t_nodes.unsqueeze(1).float()
            #s_norm_nodes_tensor = s_norm_nodes.unsqueeze(1)
            t_norm_nodes_tensor = t_norm_nodes.unsqueeze(1)

            #s_edges_tensor = s_edges.unsqueeze(1).float()
            #t_edges_tensor = t_edges.unsqueeze(1).float()
            #s_norm_edges_tensor = s_norm_edges.unsqueeze(1)
            t_norm_edges_tensor = t_norm_edges.unsqueeze(1)

            # 调用采样函数，从 z_t 采样 z_s
            sampled_s, discrete_sampled_s, num_edges = self.sample_p_zs_given_zt(
                s_nodes_tensor, t_nodes_tensor, t_norm_nodes_tensor, t_norm_edges_tensor,
                X, E, y, node_mask)

            X, E, y = sampled_s.X, sampled_s.E, sampled_s.y

            # 保存中间结果
            write_index = t_nodes.min().item()  # 使用最小的 t_nodes 作为索引
            if write_index < chain_X.size(0):
                chain_X[write_index] = discrete_sampled_s.X[:keep_chain]
                chain_E[write_index] = discrete_sampled_s.E[:keep_chain]

            # 更新 t_nodes 和 t_edges，为下一时间步做准备
            t_nodes = s_nodes
            t_edges = num_edges

        # 最终的采样结果
        #X, E = self.connect_components(X, E, node_mask)
        sampled = utils.PlaceHolder(X=X, E=E, y=torch.zeros(y.shape[0], 0))
        sampled = sampled.mask(node_mask, collapse=True)
        X, E, y = sampled.X, sampled.E, sampled.y """




        # 计算每个图的有效节点数
        valid_nodes_per_graph = n_nodes  # (batch_size,)
        max_node_steps = valid_nodes_per_graph.max().item()  # 原始最大有效节点数

        times = 2
        t_nodes = (times * valid_nodes_per_graph).clone()  # 初始化 t_nodes 为 2*n
        valid_nodes = valid_nodes_per_graph.clone()  # 初始化为有效节点数
        edge_noise_ratio = 0.2
        valid_edges = (valid_nodes * (valid_nodes - 1)) // 2 + 1e-8
        t_edges = (valid_edges.float() * edge_noise_ratio).floor()

        # total_steps 使用扩展后的最大值
        total_steps = (times * valid_nodes_per_graph).max().item()
        steps = (times * valid_nodes_per_graph).float()

        """ valid_nodes_per_graph = n_nodes
        valid_nodes = valid_nodes_per_graph.clone()
        edge_noise_ratio = 0.2
        valid_edges = (valid_nodes * (valid_nodes - 1)) // 2 + 1e-8
        t_edges = (valid_edges.float() * edge_noise_ratio).floor()

        cond1 = (valid_nodes_per_graph <= 4)
        cond2 = (valid_nodes_per_graph > 4) & (valid_nodes_per_graph <= 7)
        cond3 = (valid_nodes_per_graph > 7)
        t_nodes = torch.zeros_like(valid_nodes_per_graph)  # (batch_size,)
        t_nodes[cond1] = valid_nodes_per_graph[cond1]
        t_nodes[cond2] = 4 + (valid_nodes_per_graph[cond2] - 4) * 2
        t_nodes[cond3] = 10 + (valid_nodes_per_graph[cond3] - 7) * 4
        
        total_steps = t_nodes.max().item()
        steps = t_nodes.clone().float() """

        for step in reversed(range(total_steps + 1)):
            
            s_nodes = t_nodes - 1
            s_nodes = torch.clamp(s_nodes, min=0)


            """ r_t = t_nodes.float() / (steps + 1e-8)
            r_s = s_nodes.float() / (steps + 1e-8)
            cos_val_t = (1.0 - torch.exp(-2*r_t)) / (1.0 - math.e**(-2))#r_t**2#torch.cos(0.5* math.pi * (r_t + 0.008) / (1 + 0.008))**2  # (batch_size,)
            cos_val_s = (1.0 - torch.exp(-2*r_s)) / (1.0 - math.e**(-2))#r_s**2#torch.cos(0.5* math.pi * (r_s + 0.008) / (1 + 0.008))**2  # (batch_size,)
            t_nodes_float = 1.0 + (valid_nodes_per_graph - 1.0) * (1 - cos_val_t)
            s_nodes_float = 1.0 + (valid_nodes_per_graph - 1.0) * (1 - cos_val_s)
            t_nodes_real = t_nodes_float.floor().long()
            s_nodes_real = s_nodes_float.floor().long() """

            # 将 s_nodes, t_nodes 映射回真实步数
            # 真实步数 = (扩展步数值 + 1) // 2
            t_nodes_real = (t_nodes + (times - 1)) // 2
            s_nodes_real = (s_nodes + (times - 1)) // 2
            """ t_nodes_real = torch.zeros_like(t_nodes)  # (batch_size,)
            segA_mask_t = (t_nodes <= 4)
            t_nodes_real[segA_mask_t] = t_nodes[segA_mask_t]
            segB_mask_t = (t_nodes > 4) & (t_nodes <= 10)
            offsetB_t = (t_nodes[segB_mask_t] - 4)
            t_nodes_real[segB_mask_t] = 4 + (offsetB_t + 1) // 2 
            segC_mask_t = (t_nodes > 10)
            offsetC_t = (t_nodes[segC_mask_t] - 10)
            t_nodes_real[segC_mask_t] = 7 + (offsetC_t + 3) // 4

            s_nodes_real = torch.zeros_like(s_nodes)  # (batch_size,)
            segA_mask_s = (s_nodes <= 4)
            s_nodes_real[segA_mask_s] = s_nodes[segA_mask_s]
            segB_mask_s = (s_nodes > 4) & (s_nodes <= 10)
            offsetB_s = (s_nodes[segB_mask_s] - 4)
            s_nodes_real[segB_mask_s] = 4 + (offsetB_s + 1) // 2 
            segC_mask_s = (s_nodes > 10)
            offsetC_s = (s_nodes[segC_mask_s] - 10)
            s_nodes_real[segC_mask_s] = 7 + (offsetC_s + 3) // 4 """
            #max_possible_edges = (t_nodes_real * (t_nodes_real - 1)) // 2 + 1e-8

            # 使用真实步数计算归一化值
            t_norm_nodes = t_nodes_real.float() / valid_nodes.float()
            """ r_t = t_nodes / steps
            r_t = (1 - torch.cos(0.5 * math.pi * ((r_t + 0.008) / (1 + 0.008))) ** 2) """
            t_norm_edges = t_edges.float() / valid_edges.float()
            #t_norm_edges = t_edges.float() / max_possible_edges.float()

            # 将 s 和 t 转换为 (batch_size, 1) 张量
            general_s_nodes = s_nodes
            s_nodes_tensor = s_nodes_real.unsqueeze(1).float()
            t_nodes_tensor = t_nodes_real.unsqueeze(1).float()
            t_norm_nodes_tensor = t_norm_nodes.unsqueeze(1)

            t_norm_edges_tensor = t_norm_edges.unsqueeze(1)

            # 调用采样函数
            sampled_s, discrete_sampled_s, num_edges = self.sample_p_zs_given_zt(
                s_nodes_tensor, t_nodes_tensor, t_norm_nodes_tensor, t_norm_edges_tensor, general_s_nodes, steps,
                X, E, y, node_mask)

            X, E, y = sampled_s.X, sampled_s.E, sampled_s.y

            # 保存中间结果
            # 使用真实步数索引，而非扩展步数索引
            # write_index 应该使用 t_nodes_real 的最小值，而非 t_nodes
            write_index = t_nodes_real.min().item()
            if write_index < chain_X.size(0):
                chain_X[write_index] = discrete_sampled_s.X[:keep_chain]
                chain_E[write_index] = discrete_sampled_s.E[:keep_chain]

            # 更新 t_nodes, t_edges (扩展坐标)
            t_nodes = s_nodes
            t_edges = num_edges 

        # 最终的采样结果
        #X, E = self.connect_components(X, E, node_mask)
        sampled = utils.PlaceHolder(X=X, E=E, y=torch.zeros(y.shape[0], 0))
        sampled = sampled.mask(node_mask, collapse=True)
        X, E, y = sampled.X, sampled.E, sampled.y

        #molecule_list, X, E= self.convert_feature_with_supernode(X,E,n_nodes,self.dataset_info.ring_types)
        #X, E, molecule_list = self.build_molecule_list(X, E, n_nodes)

        # 定义链的尺寸，用于保存中间结果
        chain_X_size = torch.Size((number_chain_steps + 1, keep_chain, X.size(1)))
        chain_E_size = torch.Size((number_chain_steps + 1, keep_chain, E.size(1), E.size(2)))

        chain_X = torch.zeros(chain_X_size, device=self.device)
        chain_E = torch.zeros(chain_E_size, device=self.device)

        # 准备保存链的结果
        if keep_chain > 0:
            final_X_chain = X[:keep_chain]
            final_E_chain = E[:keep_chain]

            chain_X[0] = final_X_chain  # 将最终的 X, E 保存到链的起始位置
            chain_E[0] = final_E_chain

            chain_X = diffusion_utils.reverse_tensor(chain_X)
            chain_E = diffusion_utils.reverse_tensor(chain_E)

            # 重复最后一帧以更好地查看最终样本
            chain_X = torch.cat([chain_X, chain_X[-1:].repeat(10, 1, 1)], dim=0)
            chain_E = torch.cat([chain_E, chain_E[-1:].repeat(10, 1, 1, 1)], dim=0)
            assert chain_X.size(0) == (number_chain_steps + 11)

        molecule_list = []
        for i in range(batch_size):
            n = n_nodes[i]
            atom_types = X[i, :n].cpu()
            edge_types = E[i, :n, :n].cpu()
            molecule_list.append([atom_types, edge_types])

        # 可视化链
        if self.visualization_tools is not None:
            self.print('Visualizing chains...')
            current_path = os.getcwd()
            num_molecules = chain_X.size(1)  # 分子数量
            for i in range(num_molecules):
                result_path = os.path.join(current_path, f'chains/{self.cfg.general.name}/'
                                                        f'epoch{self.current_epoch}/'
                                                        f'chains/molecule_{batch_id + i}')
                if not os.path.exists(result_path):
                    os.makedirs(result_path)
                    _ = self.visualization_tools.visualize_chain(result_path,
                                                                chain_X[:, i, :].cpu().numpy(),
                                                                chain_E[:, i, :].cpu().numpy())
                self.print('\r{}/{} complete'.format(i+1, num_molecules), end='', flush=True)
            self.print('\nVisualizing molecules...')

            # 可视化最终的分子
            current_path = os.getcwd()
            result_path = os.path.join(current_path,
                                    f'graphs/{self.name}/epoch{self.current_epoch}_b{batch_id}/')
            self.visualization_tools.visualize(result_path, molecule_list, save_final)
            self.print("Done.")

        return molecule_list



    def sample_p_zs_given_zt1111111(self, s, t, X_t, E_t, y_t, node_mask): #原始
        #Samples from zs ~ p(zs | zt). Only used during sampling.
        #   if last_step, return the graph prediction as well
        bs, n, dxs = X_t.shape
        beta_t = self.noise_schedule(t_normalized=t)  # (bs, 1)
        alpha_s_bar = self.noise_schedule.get_alpha_bar(t_normalized=s)
        alpha_t_bar = self.noise_schedule.get_alpha_bar(t_normalized=t)

        # Retrieve transitions matrix
        Qtb = self.transition_model.get_Qt_bar(alpha_t_bar, self.device)
        Qsb = self.transition_model.get_Qt_bar(alpha_s_bar, self.device)
        Qt = self.transition_model.get_Qt(beta_t, self.device)

        # Neural net predictions
        noisy_data = {'X_t': X_t, 'E_t': E_t, 'y_t': y_t, 't': t, 'node_mask': node_mask}
        extra_data = self.compute_extra_data(noisy_data)
        pred = self.forward(noisy_data, extra_data, node_mask)

        # Normalize predictions
        pred_X = F.softmax(pred.X, dim=-1)               # bs, n, d0
        pred_E = F.softmax(pred.E, dim=-1)               # bs, n, n, d0

        p_s_and_t_given_0_X = diffusion_utils.compute_batched_over0_posterior_distribution(X_t=X_t,
                                                                                           Qt=Qt.X,
                                                                                           Qsb=Qsb.X,
                                                                                           Qtb=Qtb.X)

        p_s_and_t_given_0_E = diffusion_utils.compute_batched_over0_posterior_distribution(X_t=E_t,
                                                                                           Qt=Qt.E,
                                                                                           Qsb=Qsb.E,
                                                                                           Qtb=Qtb.E) 
        # Dim of these two tensors: bs, N, d0, d_t-1
        weighted_X = pred_X.unsqueeze(-1) * p_s_and_t_given_0_X         # bs, n, d0, d_t-1
        unnormalized_prob_X = weighted_X.sum(dim=2)                     # bs, n, d_t-1
        unnormalized_prob_X[torch.sum(unnormalized_prob_X, dim=-1) == 0] = 1e-5
        prob_X = unnormalized_prob_X / torch.sum(unnormalized_prob_X, dim=-1, keepdim=True)  # bs, n, d_t-1

        pred_E = pred_E.reshape((bs, -1, pred_E.shape[-1]))
        weighted_E = pred_E.unsqueeze(-1) * p_s_and_t_given_0_E        # bs, N, d0, d_t-1
        unnormalized_prob_E = weighted_E.sum(dim=-2)
        unnormalized_prob_E[torch.sum(unnormalized_prob_E, dim=-1) == 0] = 1e-5
        prob_E = unnormalized_prob_E / torch.sum(unnormalized_prob_E, dim=-1, keepdim=True)
        prob_E = prob_E.reshape(bs, n, n, pred_E.shape[-1])
      
        assert ((prob_X.sum(dim=-1) - 1).abs() < 1e-4).all()
        assert ((prob_E.sum(dim=-1) - 1).abs() < 1e-4).all()

        prob_X = pred_X @ Qsb.X
        prob_E = pred_E @ Qsb.E.unsqueeze(1)

        sampled_s = diffusion_utils.sample_discrete_features(self.limit_dist, prob_X, prob_E, node_mask=node_mask)

        X_s = F.one_hot(sampled_s.X, num_classes=self.Xdim_output).float()
        E_s = F.one_hot(sampled_s.E, num_classes=self.Edim_output).float()

        """ print(t[0])
        print(X_t[0])
        print(E_t[0])
        print(pred_X[0])
        print(pred_E[0])
        print(prob_X[0])
        print(prob_E[0])
        print(X_s[0])
        print(E_s[0])
        exit() """

        assert (E_s == torch.transpose(E_s, 1, 2)).all()
        assert (X_t.shape == X_s.shape) and (E_t.shape == E_s.shape)

        out_one_hot = utils.PlaceHolder(X=X_s, E=E_s, y=torch.zeros(y_t.shape[0], 0))
        out_discrete = utils.PlaceHolder(X=X_s, E=E_s, y=torch.zeros(y_t.shape[0], 0))

        return out_one_hot.mask(node_mask).type_as(y_t), out_discrete.mask(node_mask, collapse=True).type_as(y_t)



    def sample_p_zs_given_zt11(self, s_nodes, t_nodes, s_edges, t_edges, s_norm_nodes, t_norm_nodes, s_norm_edges, t_norm_edges, X_t, E_t, y_t, node_mask):
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

        # 获取上三角边的索引（不包括对角线）
        triu_indices = torch.triu_indices(n, n, offset=1, device=self.device)  # (2, num_edges)
        num_edges = triu_indices.shape[1]

        # 提取上三角边的预测
        pred_E_upper = pred_E[:, triu_indices[0], triu_indices[1], :]  # (bs, num_edges, de_out)

        # 将边特征转换为离散标签
        E0_upper = pred_E_upper.argmax(dim=-1)  # (bs, num_edges)

        # 对上三角边进行 one-hot 编码
        E0_upper_onehot = F.one_hot(E0_upper, num_classes=self.Edim_output).float()  # (bs, num_edges, de_out)

        # 初始化 E0 为零张量
        E0 = torch.zeros(bs, n, n, device=self.device).long()  # (bs, n, n)

        # 设置上三角边
        E0[:, triu_indices[0], triu_indices[1]] = E0_upper

        # 镜像上三角边到下三角边，确保对称性
        E0[:, triu_indices[1], triu_indices[0]] = E0_upper

        # 对整个 E0 进行 one-hot 编码
        E0_onehot = F.one_hot(E0, num_classes=self.Edim_output).float()  # (bs, n, n, de_out)

        """ sampled_s = diffusion_utils.sample_discrete_features(self.limit_dist, pred_X, pred_E, node_mask=node_mask)
        X0 = sampled_s.X
        E0 = sampled_s.E
        X0_onehot = F.one_hot(X0, num_classes=self.Xdim_output).float()
        E0_onehot = F.one_hot(E0, num_classes=self.Edim_output).float() """

        #Compute the nodes difference
        current_X = X_t.argmax(dim=-1)  # Shape: (bs, n)
        diff_nodes = (X0 != current_X) & node_mask # Shape: (bs, n)
        
        rand_nodes = torch.rand_like(diff_nodes, dtype=torch.float)
        rand_nodes[~diff_nodes] = -1.0  # 仅在diff_nodes为True的地方保留随机数
        selected_node = rand_nodes.argmax(dim=1)

        diff_nodes_selected = diff_nodes[torch.arange(bs, device=self.device), selected_node]
        mask_nodes = torch.zeros_like(diff_nodes, dtype=torch.bool)
        valid_samples = diff_nodes_selected
        valid_batch_indices = torch.nonzero(valid_samples).squeeze(-1)
        if valid_batch_indices.numel() > 0:
            mask_nodes[valid_batch_indices, selected_node[valid_batch_indices]] = True

        #Compute the edges difference
        current_E = E_t.argmax(dim=-1)  # Shape: (bs, n, n)
        valid_edge_mask = node_mask.unsqueeze(1) & node_mask.unsqueeze(2)
        diff_edges = (E0 != current_E) & valid_edge_mask # Shape: (bs, n, n)

        diff_edges_upper = diff_edges[:, triu_indices[0], triu_indices[1]]

        # 为满足条件的边赋予随机分数，其余赋值为-1
        rand_edges = torch.rand_like(diff_edges_upper, dtype=torch.float)
        rand_edges[~diff_edges_upper] = -1.0  # 仅在diff_edges_flat为True的地方保留随机数

        selected_edge_flat = rand_edges.argmax(dim=1)  # Shape: (bs,)

        diff_edges_selected = diff_edges_upper[torch.arange(bs), selected_edge_flat]  # Shape: (bs,)
        valid_edge_samples = diff_edges_selected  # (bs,), bool
        valid_edge_batch_indices = torch.nonzero(valid_edge_samples).squeeze(-1)  # (num_valid_samples,)
        if valid_edge_batch_indices.numel() > 0:
            i = triu_indices[0, selected_edge_flat[valid_edge_batch_indices]]
            j = triu_indices[1, selected_edge_flat[valid_edge_batch_indices]]
            mask_edges = torch.zeros_like(diff_edges, dtype=torch.bool)
            mask_edges[valid_edge_batch_indices.unsqueeze(1), i.unsqueeze(1), j.unsqueeze(1)] = True
            mask_edges[valid_edge_batch_indices.unsqueeze(1), j.unsqueeze(1), i.unsqueeze(1)] = True  # Ensure symmetry
        else:
            mask_edges = torch.zeros_like(diff_edges, dtype=torch.bool)

        X_s = X_t.clone().float()
        if mask_nodes.any():
            X_s[mask_nodes] = X0_onehot[mask_nodes]  

        E_s = E_t.clone().float()
        if mask_edges.any():
            E_s[mask_edges] = E0_onehot[mask_edges]  

        """ print(t_nodes[0])
        print(X_t[0])
        print(current_X[0])
        print(current_E[0])
        print(pred_X[0])
        print(pred_E[0])
        print(X0[0])
        print(E0[0])
        print(mask_nodes[0])
        print(mask_edges[0])
        print(X_s[0].argmax(dim=-1))
        print(E_s[0].argmax(dim=-1))
        exit() """

        z_t = utils.PlaceHolder(X=X_s, E=E_s, y=y_t).type_as(X_s).mask(node_mask)

        out_one_hot = utils.PlaceHolder(X=X_s, E=E_s, y=torch.zeros(y_t.shape[0], 0))
        out_discrete = utils.PlaceHolder(X=X_s, E=E_s, y=torch.zeros(y_t.shape[0], 0))

        return out_one_hot.mask(node_mask).type_as(y_t), out_discrete.mask(node_mask, collapse=True).type_as(y_t)

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

        # 获取上三角边的索引（不包括对角线）
        triu_indices = torch.triu_indices(n, n, offset=1, device=self.device)  # (2, num_edges)
        num_edges = triu_indices.shape[1]

        # 提取上三角边的预测
        pred_E_upper = pred_E[:, triu_indices[0], triu_indices[1], :]  # (bs, num_edges, de_out)

        # 将边特征转换为离散标签
        E0_upper = pred_E_upper.argmax(dim=-1)  # (bs, num_edges)

        # 对上三角边进行 one-hot 编码
        E0_upper_onehot = F.one_hot(E0_upper, num_classes=self.Edim_output).float()  # (bs, num_edges, de_out)

        # 初始化 E0 为零张量
        E0 = torch.zeros(bs, n, n, device=self.device).long()  # (bs, n, n)

        # 设置上三角边
        E0[:, triu_indices[0], triu_indices[1]] = E0_upper

        # 镜像上三角边到下三角边，确保对称性
        E0[:, triu_indices[1], triu_indices[0]] = E0_upper

        # 对整个 E0 进行 one-hot 编码
        E0_onehot = F.one_hot(E0, num_classes=self.Edim_output).float()  # (bs, n, n, de_out)

        z_s, num_edges = self.q_s_given_0(s_nodes, general_s_nodes, steps, X0_onehot, E0_onehot, y_t, node_mask)
        X_s = z_s.X
        E_s = z_s.E

        # -------------------- 新增的逻辑 --------------------
        # 对于 t_nodes <= 0 的图不进行变换，直接使用 X_t、E_t 替换 X_s、E_s
        no_sampling_mask = (t_nodes <= 0).view(-1)  # (bs,)的布尔张量
        # 使用布尔索引在批维度上进行替换
        X_s[no_sampling_mask] = X_t[no_sampling_mask].float()
        E_s[no_sampling_mask] = E_t[no_sampling_mask].float()
        # ----------------------------------------------

        out_one_hot = utils.PlaceHolder(X=X_s, E=E_s, y=torch.zeros(y_t.shape[0], 0))
        out_discrete = utils.PlaceHolder(X=X_s, E=E_s, y=torch.zeros(y_t.shape[0], 0))

        return out_one_hot.mask(node_mask).type_as(y_t), out_discrete.mask(node_mask, collapse=True).type_as(y_t), num_edges


    """ def compute_extra_data(self, noisy_data):
            #At every training step (after adding noise) and step in sampling, compute extra information and append to
            #the network input. 

        extra_features = self.extra_features(noisy_data)
        extra_molecular_features = self.domain_features(noisy_data)

        extra_X = torch.cat((extra_features.X, extra_molecular_features.X), dim=-1)
        extra_E = torch.cat((extra_features.E, extra_molecular_features.E), dim=-1)
        extra_y = torch.cat((extra_features.y, extra_molecular_features.y), dim=-1)

        t = noisy_data['t']
        extra_y = torch.cat((extra_y, t), dim=1)

        return utils.PlaceHolder(X=extra_X, E=extra_E, y=extra_y) """

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
        #extra_y = torch.cat((extra_y, t_nodes), dim=1)
        return utils.PlaceHolder(X=extra_X, E=extra_E, y=extra_y)


    def q_s_given_01111111(self, s_nodes, s_edges, X, E, y, node_mask):
    
        batch_size, num_nodes, _ = X.size()
        
        device = X.device
        num_edges = num_nodes * (num_nodes - 1) // 2  # C(num_nodes, 2)

        # Get upper triangular indices (excluding diagonal)
        triu_indices = torch.triu_indices(num_nodes, num_nodes, offset=1, device=device)  # (2, num_edges)

        # Generate random scores for nodes
        rand_nodes = torch.rand(batch_size, num_nodes, device=device)

        # Sort the scores in descending order and get sorted indices
        sorted_scores_nodes, sorted_indices_nodes = torch.sort(rand_nodes, dim=1, descending=True)

        # Create a range tensor for comparison
        range_tensor_nodes = torch.arange(num_nodes, device=device).unsqueeze(0).expand(batch_size, num_nodes)

        # Create a mask where positions less than s_nodes are True
        mask_nodes = range_tensor_nodes < s_nodes
 
        # Scatter the mask to the sorted indices to select top s_nodes
        node_mask_noise = torch.zeros_like(mask_nodes, dtype=torch.bool)
        node_mask_noise.scatter_(1, sorted_indices_nodes, mask_nodes)

        # Generate random scores for edges
        rand_edges = torch.rand(batch_size, num_edges, device=device)

        # Sort the scores in descending order and get sorted indices
        sorted_scores_edges, sorted_indices_edges = torch.sort(rand_edges, dim=1, descending=True)

        # Create a range tensor for comparison
        range_tensor_edges = torch.arange(num_edges, device=device).unsqueeze(0).expand(batch_size, num_edges)

        # Create a mask where positions less than s_edges are True
        mask_edges = range_tensor_edges < s_edges

        # Scatter the mask to the sorted indices to select top s_edges
        edge_mask_noise_flat = torch.zeros_like(mask_edges, dtype=torch.bool)
        edge_mask_noise_flat.scatter_(1, sorted_indices_edges, mask_edges)

        # Expand triu_indices to (1, 2, num_edges) and repeat for batch
        triu_indices_exp = triu_indices.unsqueeze(0).expand(batch_size, -1, -1)  # (batch_size, 2, num_edges)

        # Create batch indices
        batch_indices = torch.arange(batch_size, device=device).unsqueeze(1).expand(-1, num_edges)  # (batch_size, num_edges)

        # Gather selected edge indices
        selected_rows = triu_indices_exp[:, 0, :].masked_select(edge_mask_noise_flat)  # (num_selected,)
        selected_cols = triu_indices_exp[:, 1, :].masked_select(edge_mask_noise_flat)  # (num_selected,)
        selected_batch = batch_indices.masked_select(edge_mask_noise_flat)  # (num_selected,)

        # Initialize edge_mask_noise as all False
        edge_mask_noise = torch.zeros((batch_size, num_nodes, num_nodes), dtype=torch.bool, device=device)

        # Set selected edges in upper triangle
        edge_mask_noise[selected_batch, selected_rows, selected_cols] = True
        # Ensure symmetry by setting corresponding lower triangle edges
        edge_mask_noise[selected_batch, selected_cols, selected_rows] = True
      
        # Retrieve transition matrices for nodes and edges
        # Assuming Qtb.X has shape (dx_in, dx_out) and Qtb.E has shape (de_in, de_out)
        Qtb = self.transition_model.get_discrete_Qt_bar(device=device)  # Adjust as per your implementation

        # Compute transition probabilities for nodes
        # X: (batch_size, num_nodes, dx_in)
        # Qtb.X: (dx_in, dx_out)
        probX = torch.matmul(X, Qtb.X)  # Shape: (batch_size, num_nodes, dx_out)

        # Compute transition probabilities for edges
        # E: (batch_size, num_nodes, num_nodes, de_in)
        # Qtb.E: (de_in, de_out)
        probE = torch.matmul(E, Qtb.E)  # Shape: (batch_size, num_nodes, num_nodes, de_out)

        current_X = X.argmax(dim=-1)  # Shape: (batch_size, num_nodes)
        current_E = E.argmax(dim=-1)  # Shape: (batch_size, num_nodes, num_nodes)

        # make sure that the selected nodes will not be stable
        probX_selected = probX.clone()
        probX_selected[node_mask_noise] = probX_selected[node_mask_noise].scatter(
            dim=-1,
            index=current_X[node_mask_noise].unsqueeze(-1),
            value=0
        )
        # normalize
        probX_selected[node_mask_noise] = probX_selected[node_mask_noise] / probX_selected[node_mask_noise].sum(dim=-1, keepdim=True)

        # make sure that the selected edges will not be stable
        probE_selected = probE.clone()
        probE_selected[edge_mask_noise] = probE_selected[edge_mask_noise].scatter(
            dim=-1,
            index=current_E[edge_mask_noise].unsqueeze(-1),
            value=0
        )
        # normalize
        probE_selected[edge_mask_noise] = probE_selected[edge_mask_noise] / probE_selected[edge_mask_noise].sum(dim=-1, keepdim=True)
        

        sampled = diffusion_utils.sample_discrete_features(self.limit_dist, probX_selected, probE_selected, node_mask)

        X_s = sampled.X  # Shape: (batch_size, num_nodes)
        E_s = sampled.E  # Shape: (batch_size, num_nodes, num_nodes)


        X_s_final = X.clone()
        E_s_final = E.clone()

        
        X_s_final[node_mask_noise] = F.one_hot(X_s[node_mask_noise], num_classes=self.Xdim_output).float()


        E_s_final[edge_mask_noise] = F.one_hot(E_s[edge_mask_noise], num_classes=self.Edim_output).float()

        z_s = utils.PlaceHolder(X=X_s_final, E=E_s_final, y=y).type_as(X_s_final).mask(node_mask)

        return z_s


    def q_s_given_00101(self, s_nodes, s_edges, X, E, y, node_mask): #点和边同时
    
        batch_size, num_nodes, _ = X.size()
        
        device = X.device
        num_edges = num_nodes * (num_nodes - 1) // 2  # C(num_nodes, 2)

        # Get upper triangular indices (excluding diagonal)
        triu_indices = torch.triu_indices(num_nodes, num_nodes, offset=1, device=device)  # (2, num_edges)

        # Generate random scores for nodes
        rand_nodes = torch.rand(batch_size, num_nodes, device=device)

        # Sort the scores in descending order and get sorted indices
        sorted_scores_nodes, sorted_indices_nodes = torch.sort(rand_nodes, dim=1, descending=True)

        # Create a range tensor for comparison
        range_tensor_nodes = torch.arange(num_nodes, device=device).unsqueeze(0).expand(batch_size, num_nodes)

        # Create a mask where positions less than s_nodes are True
        mask_nodes = range_tensor_nodes < s_nodes
 
        # Scatter the mask to the sorted indices to select top s_nodes
        node_mask_noise = torch.zeros_like(mask_nodes, dtype=torch.bool)
        node_mask_noise.scatter_(1, sorted_indices_nodes, mask_nodes)

        # Generate random scores for edges
        rand_edges = torch.rand(batch_size, num_edges, device=device)

        # Sort the scores in descending order and get sorted indices
        sorted_scores_edges, sorted_indices_edges = torch.sort(rand_edges, dim=1, descending=True)

        # Create a range tensor for comparison
        range_tensor_edges = torch.arange(num_edges, device=device).unsqueeze(0).expand(batch_size, num_edges)

        # Create a mask where positions less than s_edges are True
        mask_edges = range_tensor_edges < s_edges

        # Scatter the mask to the sorted indices to select top s_edges
        edge_mask_noise_flat = torch.zeros_like(mask_edges, dtype=torch.bool)
        edge_mask_noise_flat.scatter_(1, sorted_indices_edges, mask_edges)

        # Expand triu_indices to (1, 2, num_edges) and repeat for batch
        triu_indices_exp = triu_indices.unsqueeze(0).expand(batch_size, -1, -1)  # (batch_size, 2, num_edges)

        # Create batch indices
        batch_indices = torch.arange(batch_size, device=device).unsqueeze(1).expand(-1, num_edges)  # (batch_size, num_edges)

        # Gather selected edge indices
        selected_rows = triu_indices_exp[:, 0, :].masked_select(edge_mask_noise_flat)  # (num_selected,)
        selected_cols = triu_indices_exp[:, 1, :].masked_select(edge_mask_noise_flat)  # (num_selected,)
        selected_batch = batch_indices.masked_select(edge_mask_noise_flat)  # (num_selected,)

        # Initialize edge_mask_noise as all False
        edge_mask_noise = torch.zeros((batch_size, num_nodes, num_nodes), dtype=torch.bool, device=device)

        # Set selected edges in upper triangle
        edge_mask_noise[selected_batch, selected_rows, selected_cols] = True
        # Ensure symmetry by setting corresponding lower triangle edges
        edge_mask_noise[selected_batch, selected_cols, selected_rows] = True
      
        # Retrieve transition matrices for nodes and edges
        # Assuming Qtb.X has shape (dx_in, dx_out) and Qtb.E has shape (de_in, de_out)
        Qtb = self.transition_model.get_discrete_Qt_bar(device=device)  # Adjust as per your implementation

        # Compute transition probabilities for nodes
        # X: (batch_size, num_nodes, dx_in)
        # Qtb.X: (dx_in, dx_out)
        probX = torch.matmul(X, Qtb.X)  # Shape: (batch_size, num_nodes, dx_out)

        # Compute transition probabilities for edges
        # E: (batch_size, num_nodes, num_nodes, de_in)
        # Qtb.E: (de_in, de_out)
        probE = torch.matmul(E, Qtb.E)  # Shape: (batch_size, num_nodes, num_nodes, de_out)

        current_X = X.argmax(dim=-1)  # Shape: (batch_size, num_nodes)
        current_E = E.argmax(dim=-1)  # Shape: (batch_size, num_nodes, num_nodes)

        # make sure that the selected nodes will not be stable
        probX_selected = probX.clone()
        probX_selected[node_mask_noise] = probX_selected[node_mask_noise].scatter(
            dim=-1,
            index=current_X[node_mask_noise].unsqueeze(-1),
            value=0
        )
        # normalize
        probX_selected[node_mask_noise] = probX_selected[node_mask_noise] / probX_selected[node_mask_noise].sum(dim=-1, keepdim=True)

        # make sure that the selected edges will not be stable
        probE_selected = probE.clone()
        """ probE_selected[edge_mask_noise] = probE_selected[edge_mask_noise].scatter(
            dim=-1,
            index=current_E[edge_mask_noise].unsqueeze(-1),
            value=0
        )
        # normalize
        probE_selected[edge_mask_noise] = probE_selected[edge_mask_noise] / probE_selected[edge_mask_noise].sum(dim=-1, keepdim=True) """
        

        sampled = diffusion_utils.sample_discrete_features(self.limit_dist, probX_selected, probE_selected, node_mask)

        X_s = sampled.X  # Shape: (batch_size, num_nodes)
        E_s = sampled.E  # Shape: (batch_size, num_nodes, num_nodes)


        X_s_final = X.clone()
        E_s_final = E.clone()

        
        X_s_final[node_mask_noise] = F.one_hot(X_s[node_mask_noise], num_classes=self.Xdim_output).float()


        E_s_final[edge_mask_noise] = F.one_hot(E_s[edge_mask_noise], num_classes=self.Edim_output).float()

        z_s = utils.PlaceHolder(X=X_s_final, E=E_s_final, y=y).type_as(X_s_final).mask(node_mask)

        return z_s


    def q_s_given_0010101(self, s_nodes, s_edges, X, E, y, node_mask): #点和边同时，只focus有效节点
        """
        给原始图添加噪声，得到 s 时刻的图。这个函数与 apply_noise 的功能类似，
        需要根据每个图的有效节点和边数进行动态处理。

        参数：
            s_nodes: (batch_size, 1) 张量，每个图在 s 时刻的节点数。
            s_edges: (batch_size, 1) 张量，每个图在 s 时刻的边数。
            X: (batch_size, num_nodes, dx_in) 张量，原始节点特征的 one-hot 编码。
            E: (batch_size, num_nodes, num_nodes, de_in) 张量，原始边特征的 one-hot 编码。
            y: (batch_size, *) 张量，标签或其他附加数据。
            node_mask: (batch_size, num_nodes) 张量，指示有效节点的掩码。

        返回：
            z_s: 包含加噪后的节点和边特征的占位符。
        """
        batch_size, num_nodes, _ = X.size()
        device = X.device

        # 计算每个图的有效节点数，形状为 (batch_size,)
        valid_nodes_per_graph = node_mask.sum(dim=1)  # 有效节点数量

        # 在每个图中，从有效节点中随机选择 s_nodes 个节点来添加噪声
        rand_nodes = torch.rand(batch_size, num_nodes, device=device)
        # 将无效节点的分数设为 -inf，确保排序时排在最后
        rand_nodes[~node_mask] = -float('inf')

        # 对分数进行降序排序，获取排序后的索引
        sorted_scores_nodes, sorted_indices_nodes = torch.sort(rand_nodes, dim=1, descending=True)

        # 创建用于比较的范围张量，形状为 (batch_size, num_nodes)
        range_tensor_nodes = torch.arange(num_nodes, device=device).unsqueeze(0).expand(batch_size, num_nodes)

        # 创建掩码，选择前 s_nodes 个节点
        mask_nodes = range_tensor_nodes < s_nodes

        # 根据排序后的索引和掩码，创建加噪节点的掩码
        node_mask_noise = torch.zeros_like(mask_nodes, dtype=torch.bool)
        node_mask_noise.scatter_(1, sorted_indices_nodes, mask_nodes)

        # 对于边，首先计算每个图的有效边数
        node_mask_expanded = node_mask.unsqueeze(1) & node_mask.unsqueeze(2)  # 形状为 (batch_size, num_nodes, num_nodes)

        # 获取上三角形（不包括对角线）的索引
        triu_indices = torch.triu_indices(num_nodes, num_nodes, offset=1, device=device)  # (2, num_edges)
        num_edges = triu_indices.shape[1]

        # 获取每个图的有效边掩码，形状为 (batch_size, num_edges)
        valid_edge_mask_upper = node_mask_expanded[:, triu_indices[0], triu_indices[1]]

        # 在每个图中，从有效边中随机选择 s_edges 个边来添加噪声
        rand_edges = torch.rand(batch_size, num_edges, device=device)
        # 将无效边的分数设为 -inf，确保排序时排在最后
        rand_edges[~valid_edge_mask_upper] = -float('inf')

        # 对边的分数进行降序排序
        sorted_scores_edges, sorted_indices_edges = torch.sort(rand_edges, dim=1, descending=True)

        # 创建用于比较的范围张量，形状为 (batch_size, num_edges)
        range_tensor_edges = torch.arange(num_edges, device=device).unsqueeze(0).expand(batch_size, num_edges)

        # 创建掩码，选择前 s_edges 个边
        mask_edges = range_tensor_edges < s_edges

        # 根据排序后的索引和掩码，创建加噪边的掩码
        edge_mask_noise_flat = torch.zeros_like(mask_edges, dtype=torch.bool)
        edge_mask_noise_flat.scatter_(1, sorted_indices_edges, mask_edges)

        # 扩展 triu_indices，形状为 (batch_size, 2, num_edges)
        triu_indices_exp = triu_indices.unsqueeze(0).expand(batch_size, -1, -1)

        # 创建批次索引，形状为 (batch_size, num_edges)
        batch_indices = torch.arange(batch_size, device=device).unsqueeze(1).expand(-1, num_edges)

        # 选取被加噪的边的索引
        selected_rows = triu_indices_exp[:, 0, :][edge_mask_noise_flat]  # (num_selected_edges,)
        selected_cols = triu_indices_exp[:, 1, :][edge_mask_noise_flat]  # (num_selected_edges,)
        selected_batch = batch_indices[edge_mask_noise_flat]  # (num_selected_edges,)

        # 初始化边的噪声掩码，形状为 (batch_size, num_nodes, num_nodes)
        edge_mask_noise = torch.zeros((batch_size, num_nodes, num_nodes), dtype=torch.bool, device=device)

        # 设置上三角形中被加噪的边
        edge_mask_noise[selected_batch, selected_rows, selected_cols] = True
        # 确保边的对称性，设置对应的下三角形边
        edge_mask_noise[selected_batch, selected_cols, selected_rows] = True

        # 获取状态转移矩阵
        Qtb = self.transition_model.get_discrete_Qt_bar(device=device)

        # 计算节点的转移概率
        probX = torch.matmul(X, Qtb.X)  # (batch_size, num_nodes, dx_out)

        # 计算边的转移概率
        probE = torch.matmul(E, Qtb.E)  # (batch_size, num_nodes, num_nodes, de_out)

        current_X = X.argmax(dim=-1)  # (batch_size, num_nodes)
        current_E = E.argmax(dim=-1)  # (batch_size, num_nodes, num_nodes)

        # 确保选中的节点不会保持不变
        probX_selected = probX.clone()
        probX_selected[node_mask_noise] = probX_selected[node_mask_noise].scatter_(
            dim=-1,
            index=current_X[node_mask_noise].unsqueeze(-1),
            value=0
        )
        # 归一化
        probX_selected[node_mask_noise] = probX_selected[node_mask_noise] / probX_selected[node_mask_noise].sum(dim=-1, keepdim=True)

        # 确保选中的边不会保持不变
        probE_selected = probE.clone()
        probE_selected[edge_mask_noise] = probE_selected[edge_mask_noise].scatter_(
            dim=-1,
            index=current_E[edge_mask_noise].unsqueeze(-1),
            value=0
        )
        # 归一化
        probE_selected[edge_mask_noise] = probE_selected[edge_mask_noise] / probE_selected[edge_mask_noise].sum(dim=-1, keepdim=True)

        # 采样离散特征
        sampled = diffusion_utils.sample_discrete_features(self.limit_dist, probX_selected, probE_selected, node_mask)

        X_s = sampled.X  # (batch_size, num_nodes)
        E_s = sampled.E  # (batch_size, num_nodes, num_nodes)

        X_s_final = X.clone()
        E_s_final = E.clone()

            # 仅对加噪的节点进行更新
        X_s_final[node_mask_noise] = F.one_hot(X_s[node_mask_noise], num_classes=self.Xdim_output).float()

        # 仅对加噪的边进行更新
        E_s_final[edge_mask_noise] = F.one_hot(E_s[edge_mask_noise], num_classes=self.Edim_output).float()
        

        z_s = utils.PlaceHolder(X=X_s_final, E=E_s_final, y=y).type_as(X_s_final).mask(node_mask)

        return z_s

    def q_s_given_001010101(self, s_nodes, s_edges, X, E, y, node_mask):  # 点和边同时，只focus有效节点之间的子图
        """
        给原始图添加噪声，得到 s 时刻的图。这个函数与 apply_noise 的功能类似，
        需要根据每个图的有效节点和边数进行动态处理。

        参数：
            s_nodes: (batch_size, 1) 张量，每个图在 s 时刻的节点数。
            s_edges: (batch_size, 1) 张量，每个图在 s 时刻的边数。
            X: (batch_size, num_nodes, dx_in) 张量，原始节点特征的 one-hot 编码。
            E: (batch_size, num_nodes, num_nodes, de_in) 张量，原始边特征的 one-hot 编码。
            y: (batch_size, *) 张量，标签或其他附加数据。
            node_mask: (batch_size, num_nodes) 张量，指示有效节点的掩码。

        返回：
            z_s: 包含加噪后的节点和边特征的占位符。
        """
        batch_size, num_nodes, _ = X.size()
        device = X.device
        
        # 计算每个图的有效节点数，形状为 (batch_size,)
        valid_nodes_per_graph = node_mask.sum(dim=1)  # 有效节点数量

        # 在每个图中，从有效节点中随机选择 s_nodes 个节点来添加噪声
        rand_nodes = torch.rand(batch_size, num_nodes, device=device)
        # 将无效节点的分数设为 -inf，确保排序时排在最后
        rand_nodes[~node_mask] = -float('inf')

        # 对分数进行降序排序，获取排序后的索引
        sorted_scores_nodes, sorted_indices_nodes = torch.sort(rand_nodes, dim=1, descending=True)

        # 创建用于比较的范围张量，形状为 (batch_size, num_nodes)
        range_tensor_nodes = torch.arange(num_nodes, device=device).unsqueeze(0).expand(batch_size, num_nodes)

        # 创建掩码，选择前 s_nodes 个节点
        mask_nodes = range_tensor_nodes < s_nodes

        # 根据排序后的索引和掩码，创建加噪节点的掩码
        node_mask_noise = torch.zeros_like(mask_nodes, dtype=torch.bool)
        node_mask_noise.scatter_(1, sorted_indices_nodes, mask_nodes)

        # -------------------- 修改边的选择和加噪方式 --------------------

        # 计算每个图的 s_edges，形状为 (batch_size,)
        # s_edges = s_nodes * (s_nodes - 1) // 2  # 已经在外部计算，这里可不再重复

        # 构建节点之间的连接关系，获取被选中节点之间的所有可能边
        # 首先，创建节点掩码，形状为 (batch_size, num_nodes, 1) 和 (batch_size, 1, num_nodes)
        node_mask_noise_row = node_mask_noise.unsqueeze(2)  # (batch_size, num_nodes, 1)
        node_mask_noise_col = node_mask_noise.unsqueeze(1)  # (batch_size, 1, num_nodes)

        # 计算节点之间的连接关系，形状为 (batch_size, num_nodes, num_nodes)
        edge_mask_noise = node_mask_noise_row & node_mask_noise_col

        # 排除自环边（如果不需要自环边）
        diag_mask = ~torch.eye(num_nodes, dtype=torch.bool, device=device).unsqueeze(0)  # (1, num_nodes, num_nodes)
        edge_mask_noise = edge_mask_noise & diag_mask  # (batch_size, num_nodes, num_nodes)

        # 确保只考虑有效节点之间的边
        node_mask_expanded = node_mask.unsqueeze(1) & node_mask.unsqueeze(2)  # (batch_size, num_nodes, num_nodes)
        edge_mask_noise = edge_mask_noise & node_mask_expanded

        # -------------------- 结束修改边的选择和加噪方式 --------------------

        # 获取状态转移矩阵
        Qtb = self.transition_model.get_discrete_Qt_bar(device=device)

        # 计算节点的转移概率
        probX = torch.matmul(X, Qtb.X)  # (batch_size, num_nodes, dx_out)

        # 计算边的转移概率
        probE = torch.matmul(E, Qtb.E)  # (batch_size, num_nodes, num_nodes, de_out)

        current_X = X.argmax(dim=-1)  # (batch_size, num_nodes)
        current_E = E.argmax(dim=-1)  # (batch_size, num_nodes, num_nodes)

        # 确保选中的节点不会保持不变
        probX_selected = probX.clone()
        probX_selected[node_mask_noise] = probX_selected[node_mask_noise].scatter_(
            dim=-1,
            index=current_X[node_mask_noise].unsqueeze(-1),
            value=0
        )
        # 归一化
        probX_selected[node_mask_noise] = probX_selected[node_mask_noise] / probX_selected[node_mask_noise].sum(dim=-1, keepdim=True)

        # 确保选中的边不会保持不变
        probE_selected = probE.clone()
        probE_selected[edge_mask_noise] = probE_selected[edge_mask_noise].scatter_(
            dim=-1,
            index=current_E[edge_mask_noise].unsqueeze(-1),
            value=0
        )
        # 归一化
        probE_selected[edge_mask_noise] = probE_selected[edge_mask_noise] / probE_selected[edge_mask_noise].sum(dim=-1, keepdim=True)

        # 对未选中的节点和边，保持原始状态
        probX_selected[~node_mask_noise] = X[~node_mask_noise]
        probE_selected[~edge_mask_noise] = E[~edge_mask_noise]

        # 采样离散特征
        sampled = diffusion_utils.sample_discrete_features(self.limit_dist, probX_selected, probE_selected, node_mask)
        
        X_s = sampled.X  # (batch_size, num_nodes)
        E_s = sampled.E  # (batch_size, num_nodes, num_nodes)

        X_s_final = X.clone()
        E_s_final = E.clone()

            # 仅对加噪的节点进行更新
        X_s_final[node_mask_noise] = F.one_hot(X_s[node_mask_noise], num_classes=self.Xdim_output).float()

        # 仅对加噪的边进行更新
        E_s_final[edge_mask_noise] = F.one_hot(E_s[edge_mask_noise], num_classes=self.Edim_output).float()
        

        z_s = utils.PlaceHolder(X=X_s_final, E=E_s_final, y=y).type_as(X_s_final).mask(node_mask)

        return z_s


    def q_s_given_0(self, s_nodes, general_s_nodes, steps, X, E, y, node_mask):  # 点和边同时，只focus有效节点之间的子图，按比例选边
        """
        给原始图添加噪声，得到 s 时刻的图。这个函数与 apply_noise 的功能类似，
        需要根据每个图的有效节点和边数进行动态处理。

        参数：
            s_nodes: (batch_size, 1) 张量，每个图在 s 时刻的节点数。
            s_edges: (batch_size, 1) 张量，每个图在 s 时刻的边数。
            X: (batch_size, num_nodes, dx_in) 张量，原始节点特征的 one-hot 编码。
            E: (batch_size, num_nodes, num_nodes, de_in) 张量，原始边特征的 one-hot 编码。
            y: (batch_size, *) 张量，标签或其他附加数据。
            node_mask: (batch_size, num_nodes) 张量，指示有效节点的掩码。

        返回：
            z_s: 包含加噪后的节点和边特征的占位符。
        """
        batch_size, num_nodes, _ = X.size()
        device = X.device

        # 计算每个图的有效节点数，形状为 (batch_size,)
        valid_nodes_per_graph = node_mask.sum(dim=1)  # 有效节点数量

        # 在每个图中，从有效节点中随机选择 s_nodes 个节点来添加噪声
        rand_nodes = torch.rand(batch_size, num_nodes, device=device)
        rand_nodes[~node_mask] = -float('inf')
        sorted_scores_nodes, sorted_indices_nodes = torch.sort(rand_nodes, dim=1, descending=True)
        range_tensor_nodes = torch.arange(num_nodes, device=device).unsqueeze(0).expand(batch_size, num_nodes)
        
        mask_nodes = range_tensor_nodes < s_nodes 
        node_mask_noise = torch.zeros_like(mask_nodes, dtype=torch.bool)
        node_mask_noise.scatter_(1, sorted_indices_nodes, mask_nodes)

        # -------------------- 修改边的选择和加噪方式 --------------------

        # 构建节点之间的连接关系，获取被选中节点之间的所有可能边
        node_mask_noise_row = node_mask_noise.unsqueeze(2)
        node_mask_noise_col = node_mask_noise.unsqueeze(1)
        potential_edge_mask = node_mask_noise_row & node_mask_noise_col
        diag_mask = ~torch.eye(num_nodes, dtype=torch.bool, device=device).unsqueeze(0)
        potential_edge_mask = potential_edge_mask & diag_mask
        node_mask_expanded = node_mask.unsqueeze(1) & node_mask.unsqueeze(2)
        potential_edge_mask = potential_edge_mask & node_mask_expanded

        triu_indices = torch.triu_indices(num_nodes, num_nodes, offset=1, device=device)
        potential_edge_mask_upper = potential_edge_mask[:, triu_indices[0], triu_indices[1]]

        # 从潜在加噪边中随机选择一部分边来添加噪声
        """ edge_noise_ratio = 0.2  # 与 apply_noise 中的值一致
        s = general_s_nodes
        even_s_mask = (s % 2 == 0)
        odd_s_mask = (s % 2 == 1)
        edge_noise_ratio = torch.full((batch_size,), 0.2, device=device)
        edge_noise_ratio[odd_s_mask] = 0.05 """

        s = general_s_nodes
        r = s / steps
        r = (1 - torch.cos(0.5 * math.pi * ((r + 0.008) / (1 + 0.008))) ** 2)
        edge_noise_ratio = torch.full((batch_size,), 0.2, device=device) * r

        rand_edges = torch.rand(batch_size, triu_indices.size(1), device=device)
        rand_edges[~potential_edge_mask_upper] = 2.0


        #edge_threshold = torch.quantile(rand_edges, edge_noise_ratio, dim=1, keepdim=True)
        Q = torch.quantile(rand_edges, edge_noise_ratio, dim=1)  # 形状: (batch_size, batch_size)
        # 提取对角线元素 Q[i,i]
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
        # 计算每个图的 t_edges，形状为 (batch_size,)
        num_edges = edge_mask_noise_flat.sum(dim=1)



        """ # 2) 计算每个图可能的可用边数量
        possible_edges = potential_edge_mask_upper.sum(dim=1)  # (batch_size,)
        # 3) 根据 edge_noise_ratio 计算选中边的个数: floor(possible_edges * edge_noise_ratio)
        selected_edge_count = (possible_edges.float() * edge_noise_ratio).floor().long()  # (batch_size,)
        # 4) 对 rand_edges 每行从小到大排序
        vals, idx = rand_edges.sort(dim=1)  # vals, idx shape 同为 (batch_size, num_edges)
        # vals[i] 是第 i 个图的 排序后随机值, idx[i] 是其对应原列索引
        # 为了一次性在 batch 上处理，构造一个行索引:
        batch_arange = torch.arange(batch_size, device=device).unsqueeze(1).expand(-1, idx.size(1))
        # shape: (batch_size, num_edges)
        # 5) 为每个图选出前 selected_edge_count[i] 个最小值
        # 构造 mask: 
        take_mask = torch.arange(vals.size(1), device=device).unsqueeze(0) < selected_edge_count.unsqueeze(1)
        # take_mask shape: (batch_size, num_edges)
        # 当 take_mask[i,j] = True 表示: 对第 i 个图, 第 j 小的边要被选中
        # 6) 从 idx 中取出被选中的边索引
        chosen_indices = idx[take_mask]             # 一维张量，所有图选中边的列索引
        selected_batch = batch_arange[take_mask]    # 对应的图索引(一维张量), 与 chosen_indices 同长度
        # 7) 映射回 triu_indices_exp
        triu_indices_exp = triu_indices.unsqueeze(0).expand(batch_size, -1, -1)  # (batch_size, 2, num_edges)
        # chosen_indices 中存的是列索引 => row = triu_indices_exp[selected_batch, 0, chosen_indices]
        #                                   col = triu_indices_exp[selected_batch, 1, chosen_indices]
        selected_rows = triu_indices_exp[selected_batch, 0, chosen_indices]
        selected_cols = triu_indices_exp[selected_batch, 1, chosen_indices]
        # 8) 其余不变, 初始化 edge_mask_noise 并赋值
        edge_mask_noise = torch.zeros((batch_size, num_nodes, num_nodes), dtype=torch.bool, device=device)
        edge_mask_noise[selected_batch, selected_rows, selected_cols] = True
        edge_mask_noise[selected_batch, selected_cols, selected_rows] = True
        # 9) 计算选中边数
        t_edges = selected_edge_count  # (batch_size,)
        num_edges = t_edges """


        # -------------------- 结束修改边的选择和加噪方式 --------------------

        # 获取状态转移矩阵
        Qtb = self.transition_model.get_discrete_Qt_bar(device=device)

        # 计算节点的转移概率
        probX = torch.matmul(X, Qtb.X)
        probE = torch.matmul(E, Qtb.E)
        
        """ Qtnb, Qtsb = self.transition_model.get_discrete_Qtnb_Qtsb_bar(device=device)
        # 获取节点标签（从 one-hot 编码转换为索引）
        labels = X.argmax(dim=-1)  # [batch_size, n_node]
        mask1 = (labels >= 0) & (labels <= 3)   # 标签 0-3
        mask2 = (labels >= 4) & (labels <= 13)  # 标签 4-13
        mask1 = mask1.unsqueeze(-1).float()  # [batch_size, n_node, 1]
        mask2 = mask2.unsqueeze(-1).float()  # [batch_size, n_node, 1]
        X_mask1 = X * mask1  # [batch_size, n_node, ndim]
        X_mask2 = X * mask2  # [batch_size, n_node, ndim]
        probX1 = torch.matmul(X_mask1, Qtnb.X)  # [batch_size, n_node, ndim]
        probX2 = torch.matmul(X_mask2, Qtsb.X)  # [batch_size, n_node, ndim]
        probX = probX1 + probX2  # [batch_size, n_node, ndim]
        probE = torch.matmul(E, Qtnb.E) """


        current_X = X.argmax(dim=-1)
        current_E = E.argmax(dim=-1)

        # 确保选中的节点不会保持不变
        probX_selected = probX.clone()
        if self.Xdim_output > 1:
            probX_selected[node_mask_noise] = probX_selected[node_mask_noise].scatter_(
                dim=-1,
                index=current_X[node_mask_noise].unsqueeze(-1),
                value=0
            )
            probX_selected[node_mask_noise] = probX_selected[node_mask_noise] / probX_selected[node_mask_noise].sum(dim=-1, keepdim=True)

        # 确保选中的边不会保持不变
        probE_selected = probE.clone()
        if self.Edim_output > 1:
            probE_selected[edge_mask_noise] = probE_selected[edge_mask_noise].scatter_(
                dim=-1,
                index=current_E[edge_mask_noise].unsqueeze(-1),
                value=0
            )
            # 归一化
            probE_selected[edge_mask_noise] = probE_selected[edge_mask_noise] / probE_selected[edge_mask_noise].sum(dim=-1, keepdim=True)

        probX_selected[~node_mask_noise] = X[~node_mask_noise]
        probE_selected[~edge_mask_noise] = E[~edge_mask_noise]

        sampled = diffusion_utils.sample_discrete_features(self.dataset_name, self.limit_dist, probX_selected, probE_selected, node_mask)

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

    def q_s_given_0234(self, s_nodes, s_edges, X, E, y, node_mask): # 点和边同时，只focus有效节点之间的子图，按边的重要性分数选边
        """
        给原始图添加噪声，得到 s 时刻的图。这个函数与 apply_noise 的功能类似，
        需要根据每个图的有效节点和边数进行动态处理。

        参数：
            s_nodes: (batch_size, 1) 张量，每个图在 s 时刻的节点数。
            s_edges: (batch_size, 1) 张量，每个图在 s 时刻的边数。
            X: (batch_size, num_nodes, dx_in) 张量，原始节点特征的 one-hot 编码。
            E: (batch_size, num_nodes, num_nodes, de_in) 张量，原始边特征的 one-hot 编码。
            y: (batch_size, *) 张量，标签或其他附加数据。
            node_mask: (batch_size, num_nodes) 张量，指示有效节点的掩码。

        返回：
            z_s: 包含加噪后的节点和边特征的占位符。
        """
        batch_size, num_nodes, _ = X.size()
        device = X.device

        # 计算每个图的有效节点数，形状为 (batch_size,)
        valid_nodes_per_graph = node_mask.sum(dim=1)  # 有效节点数量

        # 在每个图中，从有效节点中随机选择 s_nodes 个节点来添加噪声
        rand_nodes = torch.rand(batch_size, num_nodes, device=device)
        rand_nodes[~node_mask] = -float('inf')
        sorted_scores_nodes, sorted_indices_nodes = torch.sort(rand_nodes, dim=1, descending=True)
        range_tensor_nodes = torch.arange(num_nodes, device=device).unsqueeze(0).expand(batch_size, num_nodes)
        
        mask_nodes = range_tensor_nodes < s_nodes
        node_mask_noise = torch.zeros_like(mask_nodes, dtype=torch.bool)
        node_mask_noise.scatter_(1, sorted_indices_nodes, mask_nodes)

        # -------------------- 修改边的选择和加噪方式 --------------------

        # 构建节点之间的连接关系，获取被选中节点之间的所有可能边
        node_mask_noise_row = node_mask_noise.unsqueeze(2)
        node_mask_noise_col = node_mask_noise.unsqueeze(1)
        potential_edge_mask = node_mask_noise_row & node_mask_noise_col
        diag_mask = ~torch.eye(num_nodes, dtype=torch.bool, device=device).unsqueeze(0)
        potential_edge_mask = potential_edge_mask & diag_mask
        node_mask_expanded = node_mask.unsqueeze(1) & node_mask.unsqueeze(2)
        potential_edge_mask = potential_edge_mask & node_mask_expanded

        # -------------------- 基于边的重要性调整边的加噪概率 --------------------

        # 从 E 中获取当前的边类型
        current_E = E.argmax(dim=-1)  # (batch_size, num_nodes, num_nodes)
        adjacency_matrix = (current_E > 0).float()  # (batch_size, num_nodes, num_nodes)

        # 计算节点度数
        degrees = adjacency_matrix.sum(dim=-1)  # (batch_size, num_nodes)

        # 计算边的重要性评分
        degree_i = degrees.unsqueeze(2)  # (batch_size, num_nodes, 1)
        degree_j = degrees.unsqueeze(1)  # (batch_size, 1, num_nodes)
        epsilon = 1e-6  # 防止除以零
        edge_importance = 1 / (degree_i + degree_j - 2 + epsilon)  # (batch_size, num_nodes, num_nodes)

        # 对于潜在的边，计算加噪概率
        edge_noise_ratio = 0.2  # 与 apply_noise 中的值一致

        # 归一化重要性评分，使其最大值为1
        max_importance = torch.amax(edge_importance, dim=(1, 2), keepdim=True)  # (batch_size, 1, 1)
        edge_modify_prob = edge_importance / (max_importance + 1e-8)  # (batch_size, num_nodes, num_nodes)

        # 调整加噪概率
        edge_modify_prob = edge_modify_prob * edge_noise_ratio

        # 只考虑潜在的边
        edge_modify_prob = edge_modify_prob * potential_edge_mask.float()

        # 生成与边相同形状的随机数矩阵
        rand_edges = torch.rand(batch_size, num_nodes, num_nodes, device=device)

        # 确定哪些边需要添加噪声
        edge_mask_noise = rand_edges < edge_modify_prob

        # 确保对称性
        edge_mask_noise = edge_mask_noise.triu(1)
        edge_mask_noise = edge_mask_noise | edge_mask_noise.transpose(1, 2)

        # -------------------- 结束修改边的选择和加噪方式 --------------------




        """ current_E = E.argmax(dim=-1)
        adjacency_matrix = (current_E > 0).float()
        adjacency_matrix = adjacency_matrix * node_mask.unsqueeze(1) * node_mask.unsqueeze(2)

        connected_components = self.compute_connected_components_batch(adjacency_matrix, node_mask)

        comp_i = connected_components.unsqueeze(2)
        comp_j = connected_components.unsqueeze(1)
        different_component = comp_i != comp_j

        base_edge_prob = 0.2
        increased_edge_prob = base_edge_prob * 2  # 根据需要调整

        edge_modify_prob = torch.full((batch_size, num_nodes, num_nodes), base_edge_prob, device=device)
        edge_modify_prob = torch.where(different_component & potential_edge_mask, increased_edge_prob, edge_modify_prob)
        edge_modify_prob = torch.clamp(edge_modify_prob, 0.0, 1.0)

        rand_edges = torch.rand(batch_size, num_nodes, num_nodes, device=device)
        edge_mask_noise = rand_edges < edge_modify_prob
        edge_mask_noise = edge_mask_noise & potential_edge_mask
        edge_mask_noise = edge_mask_noise.triu(1)
        edge_mask_noise = edge_mask_noise | edge_mask_noise.transpose(1, 2) """



        # 获取状态转移矩阵
        Qtb = self.transition_model.get_discrete_Qt_bar(device=device)

        # 计算节点的转移概率
        probX = torch.matmul(X, Qtb.X)
        probE = torch.matmul(E, Qtb.E)

        current_X = X.argmax(dim=-1)
        current_E = E.argmax(dim=-1)

        probX_selected = probX.clone()
        probX_selected[node_mask_noise] = probX_selected[node_mask_noise].scatter_(
            dim=-1,
            index=current_X[node_mask_noise].unsqueeze(-1),
            value=0
        )
        probX_selected[node_mask_noise] = probX_selected[node_mask_noise] / probX_selected[node_mask_noise].sum(dim=-1, keepdim=True)

        probE_selected = probE.clone()
        probE_selected[edge_mask_noise] = probE_selected[edge_mask_noise].scatter_(
            dim=-1,
            index=current_E[edge_mask_noise].unsqueeze(-1),
            value=0
        )
        probE_selected[edge_mask_noise] = probE_selected[edge_mask_noise] / probE_selected[edge_mask_noise].sum(dim=-1, keepdim=True)

        # 对未被加噪的节点和边，保持原始状态
        probX_selected[~node_mask_noise] = X[~node_mask_noise]
        probE_selected[~edge_mask_noise] = E[~edge_mask_noise]

        # 对节点和边的概率分布进行采样，得到离散的特征
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

        return z_s


    def q_s_given_011(self, s_nodes, s_edges, X, E, y, node_mask): #边依赖点

        batch_size, num_nodes, _ = X.size()
        
        device = X.device
        num_edges = num_nodes * (num_nodes - 1) // 2  # Total possible edges

        # Get upper triangular indices (excluding diagonal)
        triu_indices = torch.triu_indices(num_nodes, num_nodes, offset=1, device=device)  # (2, num_edges)

        # Generate random scores for nodes
        rand_nodes = torch.rand(batch_size, num_nodes, device=device)

        # Sort the scores in descending order and get sorted indices
        sorted_scores_nodes, sorted_indices_nodes = torch.sort(rand_nodes, dim=1, descending=True)

        # Create a range tensor for comparison
        range_tensor_nodes = torch.arange(num_nodes, device=device).unsqueeze(0).expand(batch_size, num_nodes)

        mask_nodes = range_tensor_nodes < s_nodes

        # Scatter the mask to the sorted indices to select top s_nodes
        node_mask_noise = torch.zeros_like(mask_nodes, dtype=torch.bool)
        node_mask_noise.scatter_(1, sorted_indices_nodes, mask_nodes)

        # Create edge priority mask: edges connecting selected nodes
        node_mask_noise_unsqueezed = node_mask_noise.unsqueeze(2) & node_mask_noise.unsqueeze(1)
        # Remove self-loops if any
        diag_indices = torch.arange(num_nodes, device=device)
        node_mask_noise_unsqueezed[:, diag_indices, diag_indices] = False

        # Get priority edges in upper triangle
        edge_mask_priority = node_mask_noise_unsqueezed[:, triu_indices[0], triu_indices[1]]  # (batch_size, num_edges)

        # Remaining edges (non-priority)
        edge_mask_non_priority = ~edge_mask_priority  # (batch_size, num_edges)

        scores = torch.zeros(batch_size, num_edges, device=device)

        num_priority_edges_total = edge_mask_priority.sum().item()
        num_non_priority_edges_total = edge_mask_non_priority.sum().item()
        # Calculate the number of priority edges for each sample
        num_priority_edges = edge_mask_priority.sum(dim=1)  # (batch_size,)

        # Generate random scores for priority and non-priority edges
        rand_edges_priority = torch.rand(batch_size, num_edges, device=device)
        rand_edges_priority[~edge_mask_priority] = 2.0  # Set non-priority edges to a higher value
        rand_edges_non_priority = torch.rand(batch_size, num_edges, device=device)
        rand_edges_non_priority[~edge_mask_non_priority] = 2.0  # Set priority edges to a higher value

        # Initialize edge_mask_noise_flat as all False
        edge_mask_noise_flat = torch.zeros((batch_size, num_edges), dtype=torch.bool, device=device)

        # Total number of edges to select per sample
        t_edges_expanded = s_edges.expand(-1, num_edges)  # (batch_size, num_edges)

        # For priority edges, set high scores to ensure selection
        scores_priority = rand_edges_priority.clone()
        scores_priority[~edge_mask_priority] = 2.0  # Non-priority edges get high scores

        # For non-priority edges, set high scores to avoid selection unless needed
        scores_non_priority = rand_edges_non_priority.clone()
        scores_non_priority[~edge_mask_non_priority] = 2.0  # Priority edges get high scores

        scores[edge_mask_priority] = torch.rand(num_priority_edges_total, device=device)

        # For non-priority edges
        scores[edge_mask_non_priority] = torch.rand(num_non_priority_edges_total, device=device) + 1.0  # [1,2)

        # Sort the scores and get indices
        sorted_scores, sorted_indices = torch.sort(scores, dim=1)

        # Create a range tensor for comparison
        range_edges = torch.arange(num_edges, device=device).unsqueeze(0).expand(batch_size, num_edges)

        edge_selection_mask = range_edges < s_edges # (batch_size, num_edges)

        # Initialize edge_mask_noise_flat as all False
        edge_mask_noise_flat = torch.zeros(batch_size, num_edges, dtype=torch.bool, device=device)

        # Scatter the mask to the sorted indices to select top s_edges
        edge_mask_noise_flat.scatter_(1, sorted_indices, edge_selection_mask)

        # Expand triu_indices to (1, 2, num_edges) and repeat for batch
        triu_indices_exp = triu_indices.unsqueeze(0).expand(batch_size, -1, -1)  # (batch_size, 2, num_edges)

        # Create batch indices
        batch_indices = torch.arange(batch_size, device=device).unsqueeze(1).expand(-1, num_edges)  # (batch_size, num_edges)

        # Gather selected edge indices
        selected_rows = triu_indices_exp[:, 0, :][edge_mask_noise_flat]  # (num_selected,)
        selected_cols = triu_indices_exp[:, 1, :][edge_mask_noise_flat]  # (num_selected,)
        selected_batch = batch_indices[edge_mask_noise_flat]  # (num_selected,)

        # Initialize edge_mask_noise as all False
        edge_mask_noise = torch.zeros((batch_size, num_nodes, num_nodes), dtype=torch.bool, device=device)

        # Set selected edges in upper triangle
        edge_mask_noise[selected_batch, selected_rows, selected_cols] = True
        # Ensure symmetry by setting corresponding lower triangle edges
        edge_mask_noise[selected_batch, selected_cols, selected_rows] = True

        # Retrieve transition matrices for nodes and edges
        Qtb = self.transition_model.get_discrete_Qt_bar(device=device)  # Adjust as per your implementation

        # Compute transition probabilities for nodes
        probX = torch.matmul(X, Qtb.X)  # Shape: (batch_size, num_nodes, dx_out)

        # Compute transition probabilities for edges
        probE = torch.matmul(E, Qtb.E)  # Shape: (batch_size, num_nodes, num_nodes, de_out)

        current_X = X.argmax(dim=-1)  # Shape: (batch_size, num_nodes)
        current_E = E.argmax(dim=-1)  # Shape: (batch_size, num_nodes, num_nodes)

        # Ensure that the selected nodes will not stay the same
        probX_selected = probX.clone()
        """ probX_selected[node_mask_noise] = probX_selected[node_mask_noise].scatter_(
            dim=-1,
            index=current_X[node_mask_noise].unsqueeze(-1),
            value=0
        )
        # Normalize
        probX_selected[node_mask_noise] = probX_selected[node_mask_noise] / probX_selected[node_mask_noise].sum(dim=-1, keepdim=True)"""
        # Ensure that the selected edges will not stay the same
        probE_selected = probE.clone()
        """ probE_selected[edge_mask_noise] = probE_selected[edge_mask_noise].scatter_(
            dim=-1,
            index=current_E[edge_mask_noise].unsqueeze(-1),
            value=0
        )
        # Normalize
        probE_selected[edge_mask_noise] = probE_selected[edge_mask_noise] / probE_selected[edge_mask_noise].sum(dim=-1, keepdim=True) """
            
        # Sample new features for nodes and edges
        sampled = diffusion_utils.sample_discrete_features(self.limit_dist, probX_selected, probE_selected, node_mask)

        X_s = sampled.X  # Shape: (batch_size, num_nodes)
        E_s = sampled.E  # Shape: (batch_size, num_nodes, num_nodes)

        X_s_final = X.clone()
        E_s_final = E.clone()

        # Update selected nodes
        X_s_final[node_mask_noise] = F.one_hot(X_s[node_mask_noise], num_classes=self.Xdim_output).float()

        # Update selected edges
        E_s_final[edge_mask_noise] = F.one_hot(E_s[edge_mask_noise], num_classes=self.Edim_output).float()

        z_s = utils.PlaceHolder(X=X_s_final, E=E_s_final, y=y).type_as(X_s_final).mask(node_mask)

        return z_s

    def q_s_given_0111(self, s_nodes, s_edges, X, E, y, node_mask): #子图
 
        batch_size, num_nodes, _ = X.size()
        
        device = X.device
        num_edges = num_nodes * (num_nodes - 1) // 2  # Total possible edges

        # Get upper triangular indices (excluding diagonal)
        triu_indices = torch.triu_indices(num_nodes, num_nodes, offset=1, device=device)  # (2, num_edges)

        # Sample s_nodes: integers between 1 and num_nodes (inclusive)
        s_nodes = torch.randint(1, num_nodes + 1, size=(batch_size, 1), device=device)  # Shape: (batch_size, 1)

        # Generate random scores for nodes
        rand_nodes = torch.rand(batch_size, num_nodes, device=device)

        # Sort the scores in descending order and get sorted indices
        sorted_scores_nodes, sorted_indices_nodes = torch.sort(rand_nodes, dim=1, descending=True)

        # Create a range tensor for comparison
        range_tensor_nodes = torch.arange(num_nodes, device=device).unsqueeze(0).expand(batch_size, num_nodes)

        mask_nodes = range_tensor_nodes < s_nodes

        # Scatter the mask to the sorted indices to select top s_nodes
        node_mask_noise = torch.zeros_like(mask_nodes, dtype=torch.bool)
        node_mask_noise.scatter_(1, sorted_indices_nodes, mask_nodes)

        node_mask_noise_expanded_1 = node_mask_noise.unsqueeze(2)  # (batch_size, num_nodes, 1)
        node_mask_noise_expanded_2 = node_mask_noise.unsqueeze(1)  # (batch_size, 1, num_nodes)

        # 计算节点之间的连接掩码
        edge_mask_noise_full = node_mask_noise_expanded_1 & node_mask_noise_expanded_2  # (batch_size, num_nodes, num_nodes)

        # 移除对角线元素（自环）
        diag_mask = torch.eye(num_nodes, dtype=torch.bool, device=device).unsqueeze(0)  # (1, num_nodes, num_nodes)
        edge_mask_noise = edge_mask_noise_full & (~diag_mask)

        # Retrieve transition matrices for nodes and edges
        Qtb = self.transition_model.get_discrete_Qt_bar(device=device)  # Adjust as per your implementation

        # Compute transition probabilities for nodes
        probX = torch.matmul(X, Qtb.X)  # Shape: (batch_size, num_nodes, dx_out)

        # Compute transition probabilities for edges
        probE = torch.matmul(E, Qtb.E)  # Shape: (batch_size, num_nodes, num_nodes, de_out)

        current_X = X.argmax(dim=-1)  # Shape: (batch_size, num_nodes)
        current_E = E.argmax(dim=-1)  # Shape: (batch_size, num_nodes, num_nodes)

        # Ensure that the selected nodes will not stay the same
        probX_selected = probX.clone()
        probX_selected[node_mask_noise] = probX_selected[node_mask_noise].scatter_(
            dim=-1,
            index=current_X[node_mask_noise].unsqueeze(-1),
            value=0
        )
        # Normalize
        probX_selected[node_mask_noise] = probX_selected[node_mask_noise] / probX_selected[node_mask_noise].sum(dim=-1, keepdim=True)

        # Ensure that the selected edges will not stay the same
        probE_selected = probE.clone()
        #probE_selected[edge_mask_noise] = probE_selected[edge_mask_noise].scatter_(
        #    dim=-1,
        #    index=current_E[edge_mask_noise].unsqueeze(-1),
        #    value=0
        #)
        # Normalize
        #probE_selected[edge_mask_noise] = probE_selected[edge_mask_noise] / probE_selected[edge_mask_noise].sum(dim=-1, keepdim=True)
            
        # Sample new features for nodes and edges
        sampled = diffusion_utils.sample_discrete_features(self.limit_dist, probX_selected, probE_selected, node_mask)

        X_s = sampled.X  # Shape: (batch_size, num_nodes)
        E_s = sampled.E  # Shape: (batch_size, num_nodes, num_nodes)

        X_s_final = X.clone()
        E_s_final = E.clone()

        # Update selected nodes
        X_s_final[node_mask_noise] = F.one_hot(X_s[node_mask_noise], num_classes=self.Xdim_output).float()

        # Update selected edges
        E_s_final[edge_mask_noise] = F.one_hot(E_s[edge_mask_noise], num_classes=self.Edim_output).float()


        z_s = utils.PlaceHolder(X=X_s_final, E=E_s_final, y=y).type_as(X_s_final).mask(node_mask)

        return z_s

