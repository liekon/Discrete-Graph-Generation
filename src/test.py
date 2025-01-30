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

import torch


# 参数times = 2
times = 2
batch_size = 1024
n_over_m = torch.rand(batch_size)
print(n_over_m)
ratio = (1 - torch.cos(0.5 * math.pi * ((n_over_m + 0.008) / (1 + 0.008))) ** 2)
print(ratio)
valid_nodes_per_graph = 9 * torch.ones((batch_size,))
steps = times * valid_nodes_per_graph.float()  # 扩大times倍的步数
s = (ratio * steps).long() + 1
print(s)
""" t_nodes = ((s + (times - 1)) // times)
print(t_nodes) """

""" denom = torch.max(steps, torch.ones_like(steps))  # 避免0
print(denom)
r = s.float() / denom  # (batch_size,)
# 5) 定义一个非线性函数 f(ratio) 来得到 t_nodes
#   示例: t_nodes(ratio) = 1 + (v-1)*(1 - cos(0.5*pi*ratio)^2)
#   当 ratio=0 -> t_nodes=1
#   当 ratio=1 -> t_nodes=v(=valid_nodes_per_graph)
#   也可根据需要自定义其他函数
cos_val = torch.cos(0.5* math.pi * (r + 0.008) / (1 + 0.008))**2  # (batch_size,)
# non_lin_factor = (1 - cos_val)
# t_nodes_float = 1 + (valid_nodes_per_graph - 1)*non_lin_factor
# 也可写成一行:
t_nodes_float = 1.0 + (valid_nodes_per_graph - 1.0) * cos_val

# 再对 t_nodes 浮点数做下取整并clamp到[1, valid_nodes]
t_nodes = t_nodes_float.floor().long()
#t_nodes = torch.clamp(t_nodes, min=torch.tensor(1), max=valid_nodes_per_graph.long()) """

cond1 = (valid_nodes_per_graph <= 4)
cond2 = (valid_nodes_per_graph > 4) & (valid_nodes_per_graph <= 7)
cond3 = (valid_nodes_per_graph > 7)
total_steps = torch.zeros_like(valid_nodes_per_graph)  # (batch_size,)
total_steps[cond1] = valid_nodes_per_graph[cond1]
total_steps[cond2] = 4 + (valid_nodes_per_graph[cond2] - 4)*2
total_steps[cond3] = 10 + (valid_nodes_per_graph[cond3] - 7)*3
s = (n_over_m * total_steps.float()).long() + 1           # (batch_size,)
t_nodes = torch.zeros_like(s)  # (batch_size,)
segA_mask = (s <= 4)
t_nodes[segA_mask] = s[segA_mask]
segB_mask = (s > 4) & (s <= 10)
offsetB = (s[segB_mask] - 4)
t_nodes[segB_mask] = 4 + (offsetB + 1) // 2 
segC_mask = (s > 10)
offsetC = (s[segC_mask] - 10)
t_nodes[segC_mask] = 7 + (offsetC + 2) // 3

t_nodes = torch.clamp(t_nodes, min=torch.tensor(1), max=valid_nodes_per_graph.long()) 

unique_vals, counts = torch.unique(t_nodes, return_counts=True)
print("t_nodes分布: (值, 频数)")
for val, c in zip(unique_vals.tolist(), counts.tolist()):
    print(f"{val}, {c}")

unique_vals, counts = torch.unique(s, return_counts=True)
print("s: (值, 频数)")
for val, c in zip(unique_vals.tolist(), counts.tolist()):
    print(f"{val}, {c}")

 # 根据 t_nodes 选择节点（原逻辑不变）
rand_nodes = torch.rand(batch_size, 9)
sorted_scores_nodes, sorted_indices_nodes = torch.sort(rand_nodes, dim=1, descending=True)
range_tensor_nodes = torch.arange(9).unsqueeze(0).expand(batch_size, 9)
mask_nodes = range_tensor_nodes < t_nodes.unsqueeze(1)
node_mask_noise = torch.zeros_like(mask_nodes, dtype=torch.bool)
node_mask_noise.scatter_(1, sorted_indices_nodes, mask_nodes)
print(t_nodes)
print(node_mask_noise)