import math

import torch
from torch import Tensor
from torch import nn
from torch_geometric.nn import GCNConv, TopKPooling, GATConv
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
            time_emb_dim: int,
    ):
        super().__init__()
        assert depth >= 1
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.depth = depth
        self.pool_ratios = 0.5

        self.time_emb_dim = time_emb_dim
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(self.time_emb_dim),
            nn.Linear(self.time_emb_dim, self.time_emb_dim),
            nn.ReLU()
        )

        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()

        self.conv_in = GATConv(in_channels, hidden_channels, improved=True)
        for i in range(depth):
            self.downs.append(DownBlock(hidden_channels, self.time_emb_dim, self.pool_ratios))
            self.ups.append(UpBlock(hidden_channels, self.time_emb_dim))
        self.conv_out = GATConv(hidden_channels, out_channels, improved=True)

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

        self.conv1 = GATConv(channels, channels, improved=True)
        self.n1 = nn.LayerNorm(channels)
        self.conv2 = GATConv(channels, channels, improved=True)
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
        adj = adj.to_dense()  # TODO: remove on AWS
        adj = (adj @ adj).to_sparse_coo()
        edge_index, edge_weight = adj.indices(), adj.values()
        edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)
        return edge_index, edge_weight


class UpBlock(torch.nn.Module):
    def __init__(self, channels, time_emb_dim):
        super().__init__()
        self.act = nn.ReLU()

        self.conv1 = GATConv(channels, channels, improved=True)
        self.n1 = nn.LayerNorm(channels)
        self.conv2 = GATConv(channels, channels, improved=True)
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


class Block(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim, up=False):
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, out_ch)
        if up:
            self.conv1 = nn.Conv2d(2 * in_ch, out_ch, 3, padding=1)
            self.transform = nn.ConvTranspose2d(out_ch, out_ch, 4, 2, 1)
        else:
            self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
            self.transform = nn.Conv2d(out_ch, out_ch, 4, 2, 1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.bnorm1 = nn.BatchNorm2d(out_ch)
        self.bnorm2 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU()

    def forward(self, x, t):
        h = self.bnorm1(self.relu(self.conv1(x)))
        time_emb = self.relu(self.time_mlp(t))
        time_emb = time_emb[(...,) + (None,) * 2]
        h = h + time_emb
        h = self.bnorm2(self.relu(self.conv2(h)))
        return self.transform(h)


class Unet(nn.Module):

    def __init__(self, model_depth, model_start_channels=32, time_emb_dim=32):
        super().__init__()
        in_channels = 1
        start = model_start_channels
        down_channels = tuple(start * 2 ** i for i in range(model_depth))
        up_channels = tuple(reversed(down_channels))
        out_dim = 1
        time_emb_dim = time_emb_dim

        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.ReLU()
        )

        self.conv_in = nn.Conv2d(in_channels, down_channels[0], 3, padding=1)
        self.downs = nn.ModuleList([Block(down_channels[i], down_channels[i + 1], \
                                          time_emb_dim) \
                                    for i in range(len(down_channels) - 1)])
        self.ups = nn.ModuleList([Block(up_channels[i], up_channels[i + 1], \
                                        time_emb_dim, up=True) \
                                  for i in range(len(up_channels) - 1)])
        self.conv_out = nn.Conv2d(up_channels[-1], out_dim, 1)

    def forward(self, x, timestep):
        t = self.time_mlp(timestep)
        x = self.conv_in(x)
        residual_inputs = []
        for down in self.downs:
            x = down(x, t)
            residual_inputs.append(x)
        for up in self.ups:
            residual_x = residual_inputs.pop()
            x = torch.cat((x, residual_x), dim=1)
            x = up(x, t)
        return self.conv_out(x)
