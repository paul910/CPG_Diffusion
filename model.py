import math

import torch
from torch import Tensor
from torch import nn
from torch_geometric.nn import GCNConv, TopKPooling
from torch_geometric.typing import PairTensor
from torch_geometric.utils import (
    add_self_loops,
    remove_self_loops,
    to_torch_csr_tensor,
)

torch.autograd.set_detect_anomaly(True)


class GraphUNet(torch.nn.Module):
    def __init__(
            self,
            in_channels: int,
            hidden_channels: int,
            out_channels: int,
            depth: int,
    ):
        super().__init__()
        assert depth >= 1
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.depth = depth
        self.pool_ratios = 0.5

        self.time_emb_dim = 32
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(self.time_emb_dim),
            nn.Linear(self.time_emb_dim, self.time_emb_dim),
            nn.ReLU()
        )

        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()

        self.conv_in = GCNConv(in_channels, hidden_channels, improved=True)
        for i in range(depth):
            self.downs.append(DownBlock(hidden_channels, self.time_emb_dim, self.pool_ratios))
            self.ups.append(UpBlock(hidden_channels, self.time_emb_dim))
        self.conv_out = GCNConv(hidden_channels, out_channels, improved=True)

    def forward(self, x: Tensor, edge_index: Tensor, timestep: Tensor) -> Tensor:
        """"""
        t_mlp = self.time_mlp(timestep)

        edge_weight = x.new_ones(edge_index.size(1))

        x = self.conv_in(x, edge_index, edge_weight)

        xs = [x]
        edge_indices = [edge_index]
        edge_weights = [edge_weight]
        perms = []

        for i in range(self.depth):
            x, edge_index, edge_weight, perm = self.downs[i](x, edge_index, edge_weight, t_mlp)

            if (i + 1) < self.depth:
                xs += [x]
                edge_indices += [edge_index]
                edge_weights += [edge_weight]
            perms += [perm]

        for i in range(self.depth):
            j = self.depth - 1 - i

            res = xs[j]
            edge_index = edge_indices[j]
            edge_weight = edge_weights[j]
            perm = perms[j]

            up = torch.zeros_like(res)
            up[perm] = x
            x = res + up
            x = self.ups[i](x, edge_index, edge_weight, t_mlp)

        x = self.conv_out(x, edge_index, edge_weight)

        return x


class DownBlock(torch.nn.Module):
    def __init__(self, channels, time_emb_dim, pool_ratio):
        super().__init__()
        self.act = nn.ReLU()

        self.conv1 = GCNConv(channels, channels, improved=True)
        self.n1 = nn.LayerNorm(channels)
        self.conv2 = GCNConv(channels, channels, improved=True)
        self.n2 = nn.LayerNorm(channels)

        self.pool = TopKPooling(channels, pool_ratio)
        self.time = nn.Linear(time_emb_dim, channels)

    def forward(self, x, edge_index, edge_weight, t):
        edge_index, edge_weight = self.augment_adj(edge_index, edge_weight, x.size(0))
        x, edge_index, edge_weight, _, perm, _ = self.pool(
            x, edge_index, edge_weight)

        x = self.n1(self.act(self.conv1(x, edge_index, edge_weight)))
        x = x + self.act(self.time(t))
        x = self.n2(self.act(self.conv2(x, edge_index, edge_weight)))
        return x, edge_index, edge_weight, perm

    def augment_adj(self, edge_index: Tensor, edge_weight: Tensor,
                    num_nodes: int) -> PairTensor:
        edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)
        edge_index, edge_weight = add_self_loops(edge_index, edge_weight, num_nodes=num_nodes)
        adj = to_torch_csr_tensor(edge_index, edge_weight, size=(num_nodes, num_nodes))
        # adj = adj.to_dense()  # TODO: remove on AWS
        adj = (adj @ adj).to_sparse_coo()
        edge_index, edge_weight = adj.indices(), adj.values()
        edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)
        return edge_index, edge_weight


class UpBlock(torch.nn.Module):
    def __init__(self, channels, time_emb_dim):
        super().__init__()
        self.act = nn.ReLU()

        self.conv1 = GCNConv(channels, channels, improved=True)
        self.n1 = nn.LayerNorm(channels)
        self.conv2 = GCNConv(channels, channels, improved=True)
        self.n2 = nn.LayerNorm(channels)

        self.time = nn.Linear(time_emb_dim, channels)

    def forward(self, x, edge_index, edge_weight, t):
        x = self.n1(self.act(self.conv1(x, edge_index, edge_weight)))
        x = x + self.act(self.time(t))
        x = self.n2(self.act(self.conv2(x, edge_index, edge_weight)))
        return x


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :].to(time.device)
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings
