import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import time
import wandb
import os

from models.transformer_model import GraphTransformer
from diffusion.noise_schedule import DiscreteUniformTransition, PredefinedNoiseScheduleDiscrete,\
    MarginalUniformTransition
from src.diffusion import diffusion_utils
from metrics.train_metrics import TrainLossDiscrete
from metrics.abstract_metrics import SumExceptBatchMetric, SumExceptBatchKL, NLL
from src import utils


class DiscreteDenoisingDiffusion(pl.LightningModule):
    def __init__(self, cfg, dataset_infos, train_metrics, sampling_metrics, visualization_tools, extra_features,
                 domain_features):
        super().__init__()

        input_dims = dataset_infos.input_dims
        output_dims = dataset_infos.output_dims
        nodes_dist = dataset_infos.nodes_dist

        #input_dims['y'] += 1 #change

        self.cfg = cfg
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

        self.val_nll = NLL()
        self.val_X_kl = SumExceptBatchKL()
        self.val_E_kl = SumExceptBatchKL()
        self.val_X_logp = SumExceptBatchMetric()
        self.val_E_logp = SumExceptBatchMetric()

        self.test_nll = NLL()
        self.test_X_kl = SumExceptBatchKL()
        self.test_E_kl = SumExceptBatchKL()
        self.test_X_logp = SumExceptBatchMetric()
        self.test_E_logp = SumExceptBatchMetric()

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
                               log=i % self.log_every_steps == 0)

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
        self.val_nll.reset()
        self.val_X_kl.reset()
        self.val_E_kl.reset()
        self.val_X_logp.reset()
        self.val_E_logp.reset()
        self.sampling_metrics.reset()

    def validation_step(self, data, i):
        dense_data, node_mask = utils.to_dense(data.x, data.edge_index, data.edge_attr, data.batch)
        dense_data = dense_data.mask(node_mask)
        noisy_data = self.apply_noise(dense_data.X, dense_data.E, data.y, node_mask)
        extra_data = self.compute_extra_data(noisy_data)
        pred = self.forward(noisy_data, extra_data, node_mask)
        #nll = self.compute_val_loss(pred, noisy_data, dense_data.X, dense_data.E, data.y,  node_mask, test=False)
        nll = self.kl_prior(dense_data.X, dense_data.E, node_mask)
        return {'loss': nll}

    def on_validation_epoch_end(self) -> None:
        metrics = [self.val_nll.compute(), self.val_X_kl.compute() * self.T, self.val_E_kl.compute() * self.T,
                   self.val_X_logp.compute(), self.val_E_logp.compute()]
        if wandb.run:
            wandb.log({"val/epoch_NLL": metrics[0],
                       "val/X_kl": metrics[1],
                       "val/E_kl": metrics[2],
                       "val/X_logp": metrics[3],
                       "val/E_logp": metrics[4]}, commit=False)

        self.print(f"Epoch {self.current_epoch}: Val NLL {metrics[0] :.2f} -- Val Atom type KL {metrics[1] :.2f} -- ",
                   f"Val Edge type KL: {metrics[2] :.2f}")

        # Log val nll with default Lightning logger, so it can be monitored by checkpoint callback
        val_nll = metrics[0]
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
            self.sampling_metrics.forward(samples, self.name, self.current_epoch, val_counter=-1, test=False,
                                          local_rank=self.local_rank)
            self.print(f'Done. Sampling took {time.time() - start:.2f} seconds\n')
            print("Validation epoch end ends...")

    def on_test_epoch_start(self) -> None:
        self.print("Starting test...")
        self.test_nll.reset()
        self.test_X_kl.reset()
        self.test_E_kl.reset()
        self.test_X_logp.reset()
        self.test_E_logp.reset()
        if self.local_rank == 0:
            utils.setup_wandb(self.cfg)

    def test_step(self, data, i):
        dense_data, node_mask = utils.to_dense(data.x, data.edge_index, data.edge_attr, data.batch)
        dense_data = dense_data.mask(node_mask)
        noisy_data = self.apply_noise(dense_data.X, dense_data.E, data.y, node_mask)
        extra_data = self.compute_extra_data(noisy_data)
        pred = self.forward(noisy_data, extra_data, node_mask)
        #nll = self.compute_val_loss(pred, noisy_data, dense_data.X, dense_data.E, data.y, node_mask, test=True)
        nll = self.kl_prior(dense_data.X, dense_data.E, node_mask)
        return {'loss': nll}

    def on_test_epoch_end(self) -> None:
        """ Measure likelihood on a test set and compute stability metrics. """
        metrics = [self.test_nll.compute(), self.test_X_kl.compute(), self.test_E_kl.compute(),
                   self.test_X_logp.compute(), self.test_E_logp.compute()]
        if wandb.run:
            wandb.log({"test/epoch_NLL": metrics[0],
                       "test/X_kl": metrics[1],
                       "test/E_kl": metrics[2],
                       "test/X_logp": metrics[3],
                       "test/E_logp": metrics[4]}, commit=False)

        self.print(f"Epoch {self.current_epoch}: Test NLL {metrics[0] :.2f} -- Test Atom type KL {metrics[1] :.2f} -- ",
                   f"Test Edge type KL: {metrics[2] :.2f}")

        test_nll = metrics[0]
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
        self.sampling_metrics(samples, self.name, self.current_epoch, self.val_counter, test=True, local_rank=self.local_rank)
        self.print("Done testing.")


    def kl_prior(self, X, E, node_mask):
        """Computes the KL between q(z1 | x) and the prior p(z1) = Normal(0, 1).

        This is essentially a lot of work for something that is in practice negligible in the loss. However, you
        compute it so that you see it when you've made a mistake in your noise schedule.
        """
        # Compute the last alpha value, alpha_T.
        ones = torch.ones((X.size(0), 1), device=X.device)
        Ts = self.T * ones
        alpha_t_bar = self.noise_schedule.get_alpha_bar(t_int=Ts)  # (bs, 1)

        Qtb = self.transition_model.get_Qt_bar(alpha_t_bar, self.device)

        # Compute transition probabilities
        probX = X @ Qtb.X  # (bs, n, dx_out)
        probE = E @ Qtb.E.unsqueeze(1)  # (bs, n, n, de_out)
        assert probX.shape == X.shape

        bs, n, _ = probX.shape

        limit_X = self.limit_dist.X[None, None, :].expand(bs, n, -1).type_as(probX)
        limit_E = self.limit_dist.E[None, None, None, :].expand(bs, n, n, -1).type_as(probE)

        # Make sure that masked rows do not contribute to the loss
        limit_dist_X, limit_dist_E, probX, probE = diffusion_utils.mask_distributions(true_X=limit_X.clone(),
                                                                                      true_E=limit_E.clone(),
                                                                                      pred_X=probX,
                                                                                      pred_E=probE,
                                                                                      node_mask=node_mask)

        kl_distance_X = F.kl_div(input=probX.log(), target=limit_dist_X, reduction='none')
        kl_distance_E = F.kl_div(input=probE.log(), target=limit_dist_E, reduction='none')

        return diffusion_utils.sum_except_batch(kl_distance_X) + \
               diffusion_utils.sum_except_batch(kl_distance_E)

    def compute_Lt(self, X, E, y, pred, noisy_data, node_mask, test):
        pred_probs_X = F.softmax(pred.X, dim=-1)
        pred_probs_E = F.softmax(pred.E, dim=-1)
        pred_probs_y = F.softmax(pred.y, dim=-1)

        Qtb = self.transition_model.get_Qt_bar(noisy_data['alpha_t_bar'], self.device)
        Qsb = self.transition_model.get_Qt_bar(noisy_data['alpha_s_bar'], self.device)
        Qt = self.transition_model.get_Qt(noisy_data['beta_t'], self.device)

        # Compute distributions to compare with KL
        bs, n, d = X.shape
        prob_true = diffusion_utils.posterior_distributions(X=X, E=E, y=y, X_t=noisy_data['X_t'], E_t=noisy_data['E_t'],
                                                            y_t=noisy_data['y_t'], Qt=Qt, Qsb=Qsb, Qtb=Qtb)
        prob_true.E = prob_true.E.reshape((bs, n, n, -1))
        prob_pred = diffusion_utils.posterior_distributions(X=pred_probs_X, E=pred_probs_E, y=pred_probs_y,
                                                            X_t=noisy_data['X_t'], E_t=noisy_data['E_t'],
                                                            y_t=noisy_data['y_t'], Qt=Qt, Qsb=Qsb, Qtb=Qtb)
        prob_pred.E = prob_pred.E.reshape((bs, n, n, -1))

        # Reshape and filter masked rows
        prob_true_X, prob_true_E, prob_pred.X, prob_pred.E = diffusion_utils.mask_distributions(true_X=prob_true.X,
                                                                                                true_E=prob_true.E,
                                                                                                pred_X=prob_pred.X,
                                                                                                pred_E=prob_pred.E,
                                                                                                node_mask=node_mask)
        kl_x = (self.test_X_kl if test else self.val_X_kl)(prob_true.X, torch.log(prob_pred.X))
        kl_e = (self.test_E_kl if test else self.val_E_kl)(prob_true.E, torch.log(prob_pred.E))
        return self.T * (kl_x + kl_e)

    def reconstruction_logp(self, t, X, E, node_mask):
        # Compute noise values for t = 0.
        t_zeros = torch.zeros_like(t)
        beta_0 = self.noise_schedule(t_zeros)
        Q0 = self.transition_model.get_Qt(beta_t=beta_0, device=self.device)

        probX0 = X @ Q0.X  # (bs, n, dx_out)
        probE0 = E @ Q0.E.unsqueeze(1)  # (bs, n, n, de_out)

        sampled0 = diffusion_utils.sample_discrete_features(self.limit_dist, probX=probX0, probE=probE0, node_mask=node_mask)

        X0 = F.one_hot(sampled0.X, num_classes=self.Xdim_output).float()
        E0 = F.one_hot(sampled0.E, num_classes=self.Edim_output).float()
        y0 = sampled0.y
        assert (X.shape == X0.shape) and (E.shape == E0.shape)

        sampled_0 = utils.PlaceHolder(X=X0, E=E0, y=y0).mask(node_mask)

        # Predictions
        noisy_data = {'X_t': sampled_0.X, 'E_t': sampled_0.E, 'y_t': sampled_0.y, 'node_mask': node_mask,
                      't': torch.zeros(X0.shape[0], 1).type_as(y0)}
        extra_data = self.compute_extra_data(noisy_data)
        pred0 = self.forward(noisy_data, extra_data, node_mask)

        # Normalize predictions
        probX0 = F.softmax(pred0.X, dim=-1)
        probE0 = F.softmax(pred0.E, dim=-1)
        proby0 = F.softmax(pred0.y, dim=-1)

        # Set masked rows to arbitrary values that don't contribute to loss
        probX0[~node_mask] = torch.ones(self.Xdim_output).type_as(probX0)
        probE0[~(node_mask.unsqueeze(1) * node_mask.unsqueeze(2))] = torch.ones(self.Edim_output).type_as(probE0)

        diag_mask = torch.eye(probE0.size(1)).type_as(probE0).bool()
        diag_mask = diag_mask.unsqueeze(0).expand(probE0.size(0), -1, -1)
        probE0[diag_mask] = torch.ones(self.Edim_output).type_as(probE0)

        return utils.PlaceHolder(X=probX0, E=probE0, y=proby0)

    def apply_noise1111111(self, X, E, y, node_mask): #最原始的采样
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

        """ print(111)
        print(t_int[0])
        print(X[0])
        print(E[0])
        print(Qtb.X[0])
        print(Qtb.E[0])
        print(probX[0])
        print(probE[0])
        print(X_t[0])
        print(E_t[0])
        exit() """

        z_t = utils.PlaceHolder(X=X_t, E=E_t, y=y).type_as(X_t).mask(node_mask)

        noisy_data = {'t_int': t_int, 't': t_float, 'beta_t': beta_t, 'alpha_s_bar': alpha_s_bar,
                      'alpha_t_bar': alpha_t_bar, 'X_t': z_t.X, 'E_t': z_t.E, 'y_t': z_t.y, 'node_mask': node_mask}
        return noisy_data

    
    def apply_noise1(self, X, E, y, node_mask): #点和边分别独立采样
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

        """ print(111)
        print(t_nodes[0])
        print(t_edges[0])
        print(X[0])
        print(E[0])
        print(node_mask_noise[0])
        print(edge_mask_noise[0])
        print(Qtb.X[0])
        print(Qtb.E[0])
        print(probX_selected[0])
        print(probE_selected[0])
        print(X_t[0])
        print(E_t[0])
        print(X_t_final[0])
        print(E_t_final[0])
        exit() """

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


    def apply_noise(self, X, E, y, node_mask): #点和边共同采样
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
        """ probE_selected[edge_mask_noise] = probE_selected[edge_mask_noise].scatter(
            dim=-1,
            index=current_E[edge_mask_noise].unsqueeze(-1),
            value=0
        )
        # normalize
        probE_selected[edge_mask_noise] = probE_selected[edge_mask_noise] / probE_selected[edge_mask_noise].sum(dim=-1, keepdim=True) """
        

        sampled = diffusion_utils.sample_discrete_features(self.limit_dist, probX_selected, probE_selected, node_mask)

        X_t = sampled.X  # Shape: (batch_size, num_nodes)
        E_t = sampled.E  # Shape: (batch_size, num_nodes, num_nodes)


        X_t_final = X.clone()
        E_t_final = E.clone()

        
        X_t_final[node_mask_noise] = F.one_hot(X_t[node_mask_noise], num_classes=self.Xdim_output).float()


        E_t_final[edge_mask_noise] = F.one_hot(E_t[edge_mask_noise], num_classes=self.Edim_output).float()

        """ print(111)
        print(t_nodes[0])
        print(t_edges[0])
        print(X[0].argmax(dim=-1))
        print(E[0].argmax(dim=-1))
        print(node_mask_noise[0])
        print(edge_mask_noise[0])
        print(Qtb.X[0])
        print(Qtb.E[0])
        print(X_t[0])
        print(E_t[0])
        print(X_t_final[0].argmax(dim=-1))
        print(E_t_final[0].argmax(dim=-1))
        exit()
        """
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


    def compute_val_loss(self, pred, noisy_data, X, E, y, node_mask, test=False):
        """Computes an estimator for the variational lower bound.
           pred: (batch_size, n, total_features)
           noisy_data: dict
           X, E, y : (bs, n, dx),  (bs, n, n, de), (bs, dy)
           node_mask : (bs, n)
           Output: nll (size 1)
       """
        t = noisy_data['t']

        # 1.
        N = node_mask.sum(1).long()
        log_pN = self.node_dist.log_prob(N)

        # 2. The KL between q(z_T | x) and p(z_T) = Uniform(1/num_classes). Should be close to zero.
        kl_prior = self.kl_prior(X, E, node_mask)

        # 3. Diffusion loss
        loss_all_t = self.compute_Lt(X, E, y, pred, noisy_data, node_mask, test)

        # 4. Reconstruction loss
        # Compute L0 term : -log p (X, E, y | z_0) = reconstruction loss
        prob0 = self.reconstruction_logp(t, X, E, node_mask)

        loss_term_0 = self.val_X_logp(X * prob0.X.log()) + self.val_E_logp(E * prob0.E.log())

        # Combine terms
        nlls = - log_pN + kl_prior + loss_all_t - loss_term_0
        assert len(nlls.shape) == 1, f'{nlls.shape} has more than only batch dim.'

        # Update NLL metric object and return batch nll
        nll = (self.test_nll if test else self.val_nll)(nlls)        # Average over the batch

        if wandb.run:
            wandb.log({"kl prior": kl_prior.mean(),
                       "Estimator loss terms": loss_all_t.mean(),
                       "log_pn": log_pN.mean(),
                       "loss_term_0": loss_term_0,
                       'batch_test_nll' if test else 'val_nll': nll}, commit=False)
        return nll

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
    def sample_batch(self, batch_id: int, batch_size: int, keep_chain: int, number_chain_steps: int,
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

        for step in reversed(range(total_steps + 1)):
            t = step  
            s_nodes = t - 1
            t_nodes = t

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

    def sample_p_zs_given_zt(self, s_nodes, t_nodes, s_edges, t_edges, s_norm_nodes, t_norm_nodes, s_norm_edges, t_norm_edges, X_t, E_t, y_t, node_mask):
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

        z_s = self.q_s_given_0(s_nodes, s_edges, X0_onehot, E0_onehot, y_t, node_mask)
        X_s = z_s.X
        E_s = z_s.E

        out_one_hot = utils.PlaceHolder(X=X_s, E=E_s, y=torch.zeros(y_t.shape[0], 0))
        out_discrete = utils.PlaceHolder(X=X_s, E=E_s, y=torch.zeros(y_t.shape[0], 0))

        return out_one_hot.mask(node_mask).type_as(y_t), out_discrete.mask(node_mask, collapse=True).type_as(y_t)


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
        # t_edges = noisy_data['t_edges']
        # extra_y = torch.cat((extra_y, t_nodes, t_edges), dim=1)
        extra_y = torch.cat((extra_y, t_nodes), dim=1)
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


    def q_s_given_0(self, s_nodes, s_edges, X, E, y, node_mask): 
    
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

