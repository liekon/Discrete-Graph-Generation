import torch
from torch import Tensor
import torch.nn as nn
from torchmetrics import Metric, MeanSquaredError, MetricCollection
import time
import wandb
from src.metrics.abstract_metrics import SumExceptBatchMetric, SumExceptBatchMSE, SumExceptBatchKL, CrossEntropyMetric, \
    ProbabilityMetric, NLL


class NodeMSE(MeanSquaredError):
    def __init__(self, *args):
        super().__init__(*args)


class EdgeMSE(MeanSquaredError):
    def __init__(self, *args):
        super().__init__(*args)


class TestLoss(nn.Module):
    def __init__(self):
        super(TestLoss, self).__init__()
        self.test_node_mse = NodeMSE()
        self.test_edge_mse = EdgeMSE()
        self.test_y_mse = MeanSquaredError()

    def forward(self, masked_pred_epsX, masked_pred_epsE, pred_y, true_epsX, true_epsE, true_y, log: bool):
        mse_X = self.test_node_mse(masked_pred_epsX, true_epsX) if true_epsX.numel() > 0 else 0.0
        mse_E = self.test_edge_mse(masked_pred_epsE, true_epsE) if true_epsE.numel() > 0 else 0.0
        mse_y = self.test_y_mse(pred_y, true_y) if true_y.numel() > 0 else 0.0
        mse = mse_X + mse_E + mse_y

        if log:
            to_log = {'test_loss/batch_mse': mse.detach(),
                      'test_loss/node_MSE': self.test_node_mse.compute(),
                      'test_loss/edge_MSE': self.test_edge_mse.compute(),
                      'test_loss/y_mse': self.test_y_mse.compute()}
            if wandb.run:
                wandb.log(to_log, commit=True)

        return mse

    def reset(self):
        for metric in (self.test_node_mse, self.test_edge_mse, self.test_y_mse):
            metric.reset()

    def log_epoch_metrics(self):
        epoch_node_mse = self.test_node_mse.compute() if self.test_node_mse.total > 0 else -1
        epoch_edge_mse = self.test_edge_mse.compute() if self.test_edge_mse.total > 0 else -1
        epoch_y_mse = self.test_y_mse.compute() if self.test_y_mse.total > 0 else -1

        to_log = {"test_epoch/epoch_X_mse": epoch_node_mse,
                  "test_epoch/epoch_E_mse": epoch_edge_mse,
                  "test_epoch/epoch_y_mse": epoch_y_mse}
        if wandb.run:
            wandb.log(to_log)
        return to_log



""" class TestLossDiscrete(nn.Module):
    #Test with Cross entropy
    def __init__(self, lambda_test):
        super().__init__()
        self.node_loss = CrossEntropyMetric()
        self.edge_loss = CrossEntropyMetric()
        self.y_loss = CrossEntropyMetric()
        self.lambda_test = lambda_test

    def forward(self, masked_pred_X, masked_pred_E, pred_y, true_X, true_E, true_y, log: bool):
        #Compute test metrics
        #masked_pred_X : tensor -- (bs, n, dx)
        #masked_pred_E : tensor -- (bs, n, n, de)
        #pred_y : tensor -- (bs, )
        #true_X : tensor -- (bs, n, dx)
        #true_E : tensor -- (bs, n, n, de)
        #true_y : tensor -- (bs, )
        #log : boolean. 
        true_X = torch.reshape(true_X, (-1, true_X.size(-1)))  # (bs * n, dx)
        true_E = torch.reshape(true_E, (-1, true_E.size(-1)))  # (bs * n * n, de)
        masked_pred_X = torch.reshape(masked_pred_X, (-1, masked_pred_X.size(-1)))  # (bs * n, dx)
        masked_pred_E = torch.reshape(masked_pred_E, (-1, masked_pred_E.size(-1)))   # (bs * n * n, de)

        # Remove masked rows
        mask_X = (true_X != 0.).any(dim=-1)
        mask_E = (true_E != 0.).any(dim=-1)

        flat_true_X = true_X[mask_X, :]
        flat_pred_X = masked_pred_X[mask_X, :]

        flat_true_E = true_E[mask_E, :]
        flat_pred_E = masked_pred_E[mask_E, :]

        loss_X = self.node_loss(flat_pred_X, flat_true_X) if true_X.numel() > 0 else 0.0
        loss_E = self.edge_loss(flat_pred_E, flat_true_E) if true_E.numel() > 0 else 0.0
        loss_y = self.y_loss(pred_y, true_y) if true_y.numel() > 0 else 0.0

        if log:
            to_log = {"test_loss/batch_CE": (loss_X + loss_E + loss_y).detach(),
                      "test_loss/X_CE": self.node_loss.compute() if true_X.numel() > 0 else -1,
                      "test_loss/E_CE": self.edge_loss.compute() if true_E.numel() > 0 else -1,
                      "test_loss/y_CE": self.y_loss.compute() if true_y.numel() > 0 else -1}
            if wandb.run:
                wandb.log(to_log, commit=True)
        return loss_X + self.lambda_test[0] * loss_E + self.lambda_test[1] * loss_y

    def reset(self):
        for metric in [self.node_loss, self.edge_loss, self.y_loss]:
            metric.reset()

    def log_epoch_metrics(self):
        epoch_node_loss = self.node_loss.compute() if self.node_loss.total_samples > 0 else -1
        epoch_edge_loss = self.edge_loss.compute() if self.edge_loss.total_samples > 0 else -1
        epoch_y_loss = self.test_y_loss.compute() if self.y_loss.total_samples > 0 else -1

        to_log = {"test_epoch/x_CE": epoch_node_loss,
                  "test_epoch/E_CE": epoch_edge_loss,
                  "test_epoch/y_CE": epoch_y_loss}
        if wandb.run:
            wandb.log(to_log, commit=False)

        return to_log """



""" class TestLossDiscrete(nn.Module):
    def __init__(self, lambda_test):
        super().__init__()
        self.node_loss = CrossEntropyMetric()
        self.edge_loss = CrossEntropyMetric()
        self.y_loss = CrossEntropyMetric()
        self.lambda_test = lambda_test

    def forward(self, masked_pred_X, masked_pred_E, pred_y, true_X, true_E, true_y, noisy_X_t, noisy_E_t, t, log: bool):
        #Compute test metrics
        #masked_pred_X : tensor -- (bs, n, dx)
        #masked_pred_E : tensor -- (bs, n, n, de)
        #pred_y : tensor -- (bs, )
        #true_X : tensor -- (bs, n, dx)
        #true_E : tensor -- (bs, n, n, de)
        #true_y : tensor -- (bs, )
        #log : boolean.

        true_X = torch.reshape(true_X, (-1, true_X.size(-1)))  # (bs * n, dx)
        true_E = torch.reshape(true_E, (-1, true_E.size(-1)))  # (bs * n * n, de)
        masked_pred_X = torch.reshape(masked_pred_X, (-1, masked_pred_X.size(-1)))  # (bs * n, dx)
        masked_pred_E = torch.reshape(masked_pred_E, (-1, masked_pred_E.size(-1)))   # (bs * n * n, de)

        # Remove masked rows
        mask_X = (true_X != 0.).any(dim=-1)
        mask_E = (true_E != 0.).any(dim=-1)

        flat_true_X = true_X[mask_X, :]
        flat_pred_X = masked_pred_X[mask_X, :]

        flat_true_E = true_E[mask_E, :]
        flat_pred_E = masked_pred_E[mask_E, :]

        loss_X = self.node_loss(flat_pred_X, flat_true_X) if true_X.numel() > 0 else 0.0
        loss_E = self.edge_loss(flat_pred_E, flat_true_E) if true_E.numel() > 0 else 0.0
        loss_y = self.y_loss(pred_y, true_y) if true_y.numel() > 0 else 0.0


        noisy_X = torch.reshape(noisy_X_t, (-1, noisy_X_t.size(-1)))  # (bs * n, dx)
        #noisy_E = torch.reshape(noisy_E_t, (-1, noisy_E_t.size(-1)))  # (bs * n * n, de)
 
        noisy_X = noisy_X[mask_X, :]
        #noisy_E = noisy_E[mask_E, :]

        pred_labels_X = torch.argmax(flat_pred_X, dim=-1)
        noisy_labels_X = torch.argmax(noisy_X, dim=-1)

        #pred_labels_E = torch.argmax(flat_pred_E, dim=-1)
        #noisy_labels_E = torch.argmax(noisy_E, dim=-1)

        diff_nodes = (pred_labels_X != noisy_labels_X).sum()
        #diff_edges = (pred_labels_E != noisy_labels_E).sum()
        
        diff_nodes_loss = (diff_nodes - t) ** 2
        #t_edges = ((masked_pred_X.shape[1] - 1) * t) // 2
        #diff_edges_loss = (diff_edges - t_edges) ** 2

        diff_nodes_loss = torch.mean(diff_nodes_loss.float())
        #diff_edges_loss = torch.mean(diff_edges_loss.float())
        diff_loss =  diff_nodes_loss #+ 5 * diff_edges_loss
        total_loss = loss_X + self.lambda_test[0] * loss_E + self.lambda_test[1] * loss_y + diff_loss

        if log:
            to_log = {"test_loss/batch_CE": (loss_X + loss_E + loss_y).detach(),
                      "test_loss/X_CE": self.node_loss.compute() if true_X.numel() > 0 else -1,
                      "test_loss/E_CE": self.edge_loss.compute() if true_E.numel() > 0 else -1,
                      "test_loss/y_CE": self.y_loss.compute() if true_y.numel() > 0 else -1,
                      "test_loss/diff_loss": diff_loss.detach(),
                      "test_loss/diff_nodes_loss": diff_nodes_loss.detach()}
            if wandb.run:
                wandb.log(to_log, commit=True)
        return loss_X + self.lambda_test[0] * loss_E + self.lambda_test[1] * loss_y + diff_loss

    def reset(self):
        for metric in [self.node_loss, self.edge_loss, self.y_loss]:
            metric.reset()

    def log_epoch_metrics(self):
        epoch_node_loss = self.node_loss.compute() if self.node_loss.total_samples > 0 else -1
        epoch_edge_loss = self.edge_loss.compute() if self.edge_loss.total_samples > 0 else -1
        epoch_y_loss = self.test_y_loss.compute() if self.y_loss.total_samples > 0 else -1

        to_log = {"test_epoch/x_CE": epoch_node_loss,
                  "test_epoch/E_CE": epoch_edge_loss,
                  "test_epoch/y_CE": epoch_y_loss}
        if wandb.run:
            wandb.log(to_log, commit=False)

        return to_log """


class TestLossDiscrete(nn.Module):
    # Test with Cross entropy
    def __init__(self, lambda_test):
        super().__init__()
        self.node_loss = CrossEntropyMetric()
        self.edge_loss = CrossEntropyMetric()
        self.y_loss = CrossEntropyMetric()
        self.lambda_test = lambda_test

    def forward(self, masked_pred_X, masked_pred_E, pred_y, true_X, true_E, true_y, noisy_X_t, noisy_E_t, t, t_e, log: bool):
        # Compute test metrics
        # masked_pred_X : tensor -- (bs, n, dx)
        # masked_pred_E : tensor -- (bs, n, n, de)
        # pred_y : tensor -- (bs, )
        # true_X : tensor -- (bs, n, dx)
        # true_E : tensor -- (bs, n, n, de)
        # true_y : tensor -- (bs, )
        # log : boolean. 

        true_X = torch.reshape(true_X, (-1, true_X.size(-1)))  # (bs * n, dx)
        true_E = torch.reshape(true_E, (-1, true_E.size(-1)))  # (bs * n * n, de)
        masked_pred_X = torch.reshape(masked_pred_X, (-1, masked_pred_X.size(-1)))  # (bs * n, dx)
        masked_pred_E = torch.reshape(masked_pred_E, (-1, masked_pred_E.size(-1)))   # (bs * n * n, de)

        # Remove masked rows
        mask_X = (true_X != 0.).any(dim=-1)
        mask_E = (true_E != 0.).any(dim=-1)

        flat_true_X = true_X[mask_X, :]
        flat_pred_X = masked_pred_X[mask_X, :]

        flat_true_E = true_E[mask_E, :]
        flat_pred_E = masked_pred_E[mask_E, :]

        loss_X = self.node_loss(flat_pred_X, flat_true_X) if true_X.numel() > 0 else 0.0
        loss_E = self.edge_loss(flat_pred_E, flat_true_E) if true_E.numel() > 0 else 0.0
        loss_y = self.y_loss(pred_y, true_y) if true_y.numel() > 0 else 0.0

        

        noisy_X = torch.reshape(noisy_X_t, (-1, noisy_X_t.size(-1)))  # (bs * n, dx)
        noisy_E = torch.reshape(noisy_E_t, (-1, noisy_E_t.size(-1)))  # (bs * n * n, de)

        noisy_X = noisy_X[mask_X, :]
        noisy_E = noisy_E[mask_E, :]

     
        pred_labels_X = torch.argmax(flat_pred_X, dim=-1)
        noisy_labels_X = torch.argmax(noisy_X, dim=-1)

        pred_labels_E = torch.argmax(flat_pred_E, dim=-1)
        noisy_labels_E = torch.argmax(noisy_E, dim=-1)

 
        diff_nodes = (pred_labels_X != noisy_labels_X).sum()
        diff_edges = (pred_labels_E != noisy_labels_E).sum()
        
        diff_nodes_loss = (diff_nodes - t) ** 2
        #t_edges = ((masked_pred_X.shape[1] - 1) * t) // 2
        diff_edges_loss = (diff_edges - t_e) ** 2

        diff_nodes_loss = torch.mean(diff_nodes_loss.float()) 
        diff_edges_loss = torch.mean(diff_edges_loss.float())
        diff_loss =  diff_nodes_loss + diff_edges_loss
        total_loss = loss_X + self.lambda_test[0] * loss_E + self.lambda_test[1] * loss_y + diff_loss

        if log:
            to_log = {"test_loss/batch_CE": (loss_X + loss_E + loss_y).detach(),
                      "test_loss/X_CE": self.node_loss.compute() if true_X.numel() > 0 else -1,
                      "test_loss/E_CE": self.edge_loss.compute() if true_E.numel() > 0 else -1,
                      "test_loss/y_CE": self.y_loss.compute() if true_y.numel() > 0 else -1,
                      "test_loss/diff_loss": diff_loss.detach(),
                      "test_loss/diff_nodes_loss": diff_nodes_loss.detach(),
                      "test_loss/diff_edges_loss": diff_edges_loss.detach()}
            if wandb.run:
                wandb.log(to_log, commit=True)
        return loss_X + self.lambda_test[0] * loss_E + self.lambda_test[1] * loss_y + diff_loss

    def reset(self):
        for metric in [self.node_loss, self.edge_loss, self.y_loss]:
            metric.reset()

    def log_epoch_metrics(self):
        epoch_node_loss = self.node_loss.compute() if self.node_loss.total_samples > 0 else -1
        epoch_edge_loss = self.edge_loss.compute() if self.edge_loss.total_samples > 0 else -1
        epoch_y_loss = self.test_y_loss.compute() if self.y_loss.total_samples > 0 else -1

        to_log = {"test_epoch/x_CE": epoch_node_loss,
                  "test_epoch/E_CE": epoch_edge_loss,
                  "test_epoch/y_CE": epoch_y_loss}

        return to_log