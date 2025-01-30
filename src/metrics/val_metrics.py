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


class ValLoss(nn.Module):
    def __init__(self):
        super(ValLoss, self).__init__()
        self.val_node_mse = NodeMSE()
        self.val_edge_mse = EdgeMSE()
        self.val_y_mse = MeanSquaredError()

    def forward(self, masked_pred_epsX, masked_pred_epsE, pred_y, true_epsX, true_epsE, true_y, log: bool):
        mse_X = self.val_node_mse(masked_pred_epsX, true_epsX) if true_epsX.numel() > 0 else 0.0
        mse_E = self.val_edge_mse(masked_pred_epsE, true_epsE) if true_epsE.numel() > 0 else 0.0
        mse_y = self.val_y_mse(pred_y, true_y) if true_y.numel() > 0 else 0.0
        mse = mse_X + mse_E + mse_y

        if log:
            to_log = {'val_loss/batch_mse': mse.detach(),
                      'val_loss/node_MSE': self.val_node_mse.compute(),
                      'val_loss/edge_MSE': self.val_edge_mse.compute(),
                      'val_loss/y_mse': self.val_y_mse.compute()}
            if wandb.run:
                wandb.log(to_log, commit=True)

        return mse

    def reset(self):
        for metric in (self.val_node_mse, self.val_edge_mse, self.val_y_mse):
            metric.reset()

    def log_epoch_metrics(self):
        epoch_node_mse = self.val_node_mse.compute() if self.val_node_mse.total > 0 else -1
        epoch_edge_mse = self.val_edge_mse.compute() if self.val_edge_mse.total > 0 else -1
        epoch_y_mse = self.val_y_mse.compute() if self.val_y_mse.total > 0 else -1

        to_log = {"val_epoch/epoch_X_mse": epoch_node_mse,
                  "val_epoch/epoch_E_mse": epoch_edge_mse,
                  "val_epoch/epoch_y_mse": epoch_y_mse}
        if wandb.run:
            wandb.log(to_log)
        return to_log



""" clas ValLossDiscrete(nn.Module):
    #val with Cross entropy
    def __init__(self, lambda_val):
        super().__init__()
        self.node_loss = CrossEntropyMetric()
        self.edge_loss = CrossEntropyMetric()
        self.y_loss = CrossEntropyMetric()
        self.lambda_val = lambda_val

    def forward(self, masked_pred_X, masked_pred_E, pred_y, true_X, true_E, true_y, log: bool):
        #Compute val metrics
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
            to_log = {"val_loss/batch_CE": (loss_X + loss_E + loss_y).detach(),
                      "val_loss/X_CE": self.node_loss.compute() if true_X.numel() > 0 else -1,
                      "val_loss/E_CE": self.edge_loss.compute() if true_E.numel() > 0 else -1,
                      "val_loss/y_CE": self.y_loss.compute() if true_y.numel() > 0 else -1}
            if wandb.run:
                wandb.log(to_log, commit=True)
        return loss_X + self.lambda_val[0] * loss_E + self.lambda_val[1] * loss_y

    def reset(self):
        for metric in [self.node_loss, self.edge_loss, self.y_loss]:
            metric.reset()

    def log_epoch_metrics(self):
        epoch_node_loss = self.node_loss.compute() if self.node_loss.total_samples > 0 else -1
        epoch_edge_loss = self.edge_loss.compute() if self.edge_loss.total_samples > 0 else -1
        epoch_y_loss = self.val_y_loss.compute() if self.y_loss.total_samples > 0 else -1

        to_log = {"val_epoch/x_CE": epoch_node_loss,
                  "val_epoch/E_CE": epoch_edge_loss,
                  "val_epoch/y_CE": epoch_y_loss}
        if wandb.run:
            wandb.log(to_log, commit=False)

        return to_log """



""" class ValLossDiscrete(nn.Module):
    def __init__(self, lambda_val):
        super().__init__()
        self.node_loss = CrossEntropyMetric()
        self.edge_loss = CrossEntropyMetric()
        self.y_loss = CrossEntropyMetric()
        self.lambda_val = lambda_val

    def forward(self, masked_pred_X, masked_pred_E, pred_y, true_X, true_E, true_y, noisy_X_t, noisy_E_t, t, log: bool):
        #Compute val metrics
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

        
        # 将噪声图和预测图展平
        noisy_X = torch.reshape(noisy_X_t, (-1, noisy_X_t.size(-1)))  # (bs * n, dx)
        #noisy_E = torch.reshape(noisy_E_t, (-1, noisy_E_t.size(-1)))  # (bs * n * n, de)
        # 只保留未被掩码的行
        noisy_X = noisy_X[mask_X, :]
        #noisy_E = noisy_E[mask_E, :]

        # 获取预测图和噪声图的标签
        pred_labels_X = torch.argmax(flat_pred_X, dim=-1)
        noisy_labels_X = torch.argmax(noisy_X, dim=-1)

        #pred_labels_E = torch.argmax(flat_pred_E, dim=-1)
        #noisy_labels_E = torch.argmax(noisy_E, dim=-1)

        # 计算预测图和噪声图之间标签不同的节点和边的数量
        diff_nodes = (pred_labels_X != noisy_labels_X).sum()
        #diff_edges = (pred_labels_E != noisy_labels_E).sum()
        
        # 计算差异数量的损失
        diff_nodes_loss = (diff_nodes - t) ** 2
        #t_edges = ((masked_pred_X.shape[1] - 1) * t) // 2
        #diff_edges_loss = (diff_edges - t_edges) ** 2

        diff_nodes_loss = torch.mean(diff_nodes_loss.float())
        #diff_edges_loss = torch.mean(diff_edges_loss.float())
        # 总的差异损失，使用一个新的lambda参数进行加权
        diff_loss =  diff_nodes_loss #+ 5 * diff_edges_loss
        total_loss = loss_X + self.lambda_val[0] * loss_E + self.lambda_val[1] * loss_y + diff_loss

        if log:
            to_log = {"val_loss/batch_CE": (loss_X + loss_E + loss_y).detach(),
                      "val_loss/X_CE": self.node_loss.compute() if true_X.numel() > 0 else -1,
                      "val_loss/E_CE": self.edge_loss.compute() if true_E.numel() > 0 else -1,
                      "val_loss/y_CE": self.y_loss.compute() if true_y.numel() > 0 else -1,
                      "val_loss/diff_loss": diff_loss.detach(),
                      "val_loss/diff_nodes_loss": diff_nodes_loss.detach()}
            if wandb.run:
                wandb.log(to_log, commit=True)
        return loss_X + self.lambda_val[0] * loss_E + self.lambda_val[1] * loss_y + diff_loss

    def reset(self):
        for metric in [self.node_loss, self.edge_loss, self.y_loss]:
            metric.reset()

    def log_epoch_metrics(self):
        epoch_node_loss = self.node_loss.compute() if self.node_loss.total_samples > 0 else -1
        epoch_edge_loss = self.edge_loss.compute() if self.edge_loss.total_samples > 0 else -1
        epoch_y_loss = self.val_y_loss.compute() if self.y_loss.total_samples > 0 else -1

        to_log = {"val_epoch/x_CE": epoch_node_loss,
                  "val_epoch/E_CE": epoch_edge_loss,
                  "val_epoch/y_CE": epoch_y_loss}
        if wandb.run:
            wandb.log(to_log, commit=False)

        return to_log """


class ValLossDiscrete(nn.Module):
    # val with Cross entropy
    def __init__(self, lambda_val):
        super().__init__()
        self.node_loss = CrossEntropyMetric()
        self.edge_loss = CrossEntropyMetric()
        self.y_loss = CrossEntropyMetric()
        self.lambda_val = lambda_val

    def forward(self, masked_pred_X, masked_pred_E, pred_y, true_X, true_E, true_y, noisy_X_t, noisy_E_t, t, t_e, log: bool):
        # Compute val metrics
        # masked_pred_X : tensor -- (bs, n, dx)
        # masked_pred_E : tensor -- (bs, n, n, de)
        # pred_y : tensor -- (bs, )
        # true_X : tensor -- (bs, n, dx)
        # true_E : tensor -- (bs, n, n, de)
        # true_y : tensor -- (bs, )
        # log : boolean. 

        """ # -------------------- 添加新的连通性损失 --------------------

        # 1. 获取有效节点掩码
        batch_size, num_nodes, dx = masked_pred_X.size()
        device = masked_pred_X.device
        valid_node_mask = (masked_pred_X.sum(dim=-1) != 0)  # (bs, n)
        # 从 masked_pred_E 中获取预测的边类型
        pred_edge_types = torch.argmax(masked_pred_E, dim=-1)  # (bs, n, n) """



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

        
        # 将噪声图和预测图展平
        noisy_X = torch.reshape(noisy_X_t, (-1, noisy_X_t.size(-1)))  # (bs * n, dx)
        noisy_E = torch.reshape(noisy_E_t, (-1, noisy_E_t.size(-1)))  # (bs * n * n, de)
        # 只保留未被掩码的行
        noisy_X = noisy_X[mask_X, :]
        noisy_E = noisy_E[mask_E, :]

        # 获取预测图和噪声图的标签
        pred_labels_X = torch.argmax(flat_pred_X, dim=-1)
        noisy_labels_X = torch.argmax(noisy_X, dim=-1)

        pred_labels_E = torch.argmax(flat_pred_E, dim=-1)
        noisy_labels_E = torch.argmax(noisy_E, dim=-1)

        # 计算预测图和噪声图之间标签不同的节点和边的数量
        diff_nodes = (pred_labels_X != noisy_labels_X).sum()
        diff_edges = (pred_labels_E != noisy_labels_E).sum()
        
        # 计算差异数量的损失
        diff_nodes_loss = (diff_nodes - t) ** 2
        #t_edges = ((masked_pred_X.shape[1] - 1) * t) // 2
        diff_edges_loss = (diff_edges - t_e) ** 2

        diff_nodes_loss = torch.mean(diff_nodes_loss.float()) 
        diff_edges_loss = torch.mean(diff_edges_loss.float())
        # 总的差异损失，使用一个新的lambda参数进行加权
        diff_loss =  diff_nodes_loss + diff_edges_loss



        """ # -------------------- 添加新的连通性损失 --------------------
        # 2. 构建邻接矩阵 A
        # 边类型大于 0 的表示存在边（类型 1-4），等于 0 的表示无边
        A = (pred_edge_types > 0).float()  # (bs, n, n)
        # 3. 掩码无效节点对应的边
        # 构建节点掩码矩阵
        node_mask = valid_node_mask.unsqueeze(1) * valid_node_mask.unsqueeze(2)  # (bs, n, n)
        # 掩码无效节点的边
        A = A * node_mask  # (bs, n, n)

        # 4. 构建度矩阵 D
        degrees = A.sum(dim=-1)  # (bs, n)
        D = torch.diag_embed(degrees)  # (bs, n, n)

        # 5. 计算拉普拉斯矩阵 L
        L = D - A  # (bs, n, n)

        # 6. 计算拉普拉斯矩阵的特征值
        eigvals = torch.linalg.eigvalsh(L)  # (bs, n)

        # 7. 忽略无效节点对应的特征值
        # 将无效节点的特征值设为一个较大的数
        large_value = 1e6
        valid_node_mask_float = valid_node_mask.float()
        eigvals = eigvals * valid_node_mask_float + large_value * (1 - valid_node_mask_float)

        # 8. 计算每个图的零特征值个数（即特征值小于 epsilon 的个数）
        epsilon = 0
        num_zero_eigvals = (eigvals == epsilon).sum(dim=1)  # (bs,)

        # 9. 计算连通性损失
        # 对于连通图，零特征值的个数应为 1，因此损失为 num_zero_eigvals - 1
        connectivity_loss = num_zero_eigvals - 1  # (bs,)
        # 确保损失非负
        connectivity_loss = torch.clamp(connectivity_loss, min=0).float()
        # 计算平均损失
        connectivity_loss = connectivity_loss.mean() """



        total_loss = loss_X + self.lambda_val[0] * loss_E + self.lambda_val[1] * loss_y + diff_loss 

        if log:
            to_log = {"val_loss/batch_CE": (loss_X + loss_E + loss_y).detach(),
                      "val_loss/X_CE": self.node_loss.compute() if true_X.numel() > 0 else -1,
                      "val_loss/E_CE": self.edge_loss.compute() if true_E.numel() > 0 else -1,
                      "val_loss/y_CE": self.y_loss.compute() if true_y.numel() > 0 else -1,
                      "val_loss/diff_loss": diff_loss.detach(),
                      "val_loss/diff_nodes_loss": diff_nodes_loss.detach(),
                      "val_loss/diff_edges_loss": diff_edges_loss.detach()}
            if wandb.run:
                wandb.log(to_log, commit=True)
        return loss_X + self.lambda_val[0] * loss_E + self.lambda_val[1] * loss_y + diff_loss 

    def reset(self):
        for metric in [self.node_loss, self.edge_loss, self.y_loss]:
            metric.reset()

    def log_epoch_metrics(self):
        epoch_node_loss = self.node_loss.compute() if self.node_loss.total_samples > 0 else -1
        epoch_edge_loss = self.edge_loss.compute() if self.edge_loss.total_samples > 0 else -1
        epoch_y_loss = self.val_y_loss.compute() if self.y_loss.total_samples > 0 else -1

        to_log = {"val_epoch/x_CE": epoch_node_loss,
                  "val_epoch/E_CE": epoch_edge_loss,
                  "val_epoch/y_CE": epoch_y_loss}

        return to_log