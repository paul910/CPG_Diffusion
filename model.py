import math
from typing import Callable, List, Union

import torch
from torch import Tensor
from torch import nn
from torch_geometric.nn import GCNConv, TopKPooling
from torch_geometric.nn.resolver import activation_resolver
from torch_geometric.typing import OptTensor, PairTensor
from torch_geometric.utils import (
    add_self_loops,
    remove_self_loops,
    to_torch_csr_tensor,
)
from torch_geometric.utils.repeat import repeat


class GDNN(nn.Module):
    def __init__(self, in_channels: int, time_emb_dim: int, model_depth: int = 2, model_mult_factor: int = 3):
        super().__init__()

        self.depth_x = model_depth
        self.mult_factor = model_mult_factor
        self.time_emb_dim = time_emb_dim

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

        self.bn_in = nn.BatchNorm1d(in_channels * self.mult_factor)
        self.bn_out = nn.BatchNorm1d(in_channels)

        self.conv_x_in = GCNConv(in_channels, in_channels * self.mult_factor)

        self.conv_x = nn.ModuleList(
            [Block(int(in_channels * math.pow(self.mult_factor, i)),
                   int(in_channels * (math.pow(self.mult_factor, i + 1))), time_emb_dim) for i in
             range(1, self.depth_x)] +
            [Block(int(in_channels * (math.pow(self.mult_factor, i + 1))),
                   int(in_channels * math.pow(self.mult_factor, i)), time_emb_dim) for i in
             reversed(range(1, self.depth_x))]
        )
        self.conv_x_out = GCNConv(in_channels * self.mult_factor, in_channels)

        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.ReLU(),
        )

    def forward(self, x: Tensor, edge_index: Tensor, timestep: Tensor):
        t = self.time_mlp(timestep)

        x = self.relu(self.bn_in(self.conv_x_in(x, edge_index)))
        for conv in self.conv_x:
            x = conv(x, edge_index, t)
        x = self.relu(self.bn_out(self.conv_x_out(x, edge_index)))
        return x


class Block(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim):
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, out_ch)
        self.conv1 = GCNConv(in_ch, out_ch)
        # self.conv2 = GCNConv(out_ch, out_ch)
        self.bn1 = nn.BatchNorm1d(out_ch)
        # self.bn2 = nn.BatchNorm1d(out_ch)
        self.relu = nn.ReLU()

    def forward(self, x, edge_index, t):
        h = self.conv1(x, edge_index)  # self.bn1(self.relu(self.conv1(x, edge_index)))
        time_emb = self.relu(self.time_mlp(t))
        time_emb = time_emb.repeat(h.shape[0], 1)
        return self.bn1(self.relu(h + time_emb))  # self.bn2(self.relu(self.conv2(h + time_emb, edge_index)))


class GraphUNet(torch.nn.Module):
    def __init__(
            self,
            in_channels: int,
            hidden_channels: int,
            out_channels: int,
            depth: int,
            pool_ratios: Union[float, List[float]] = 0.5,
            sum_res: bool = True,
            act: Union[str, Callable] = 'relu',
    ):
        super().__init__()
        assert depth >= 1
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.depth = depth
        self.pool_ratios = repeat(pool_ratios, depth)
        self.act = activation_resolver(act)
        self.sum_res = sum_res

        self.time_emb_dim = 32

        channels = hidden_channels

        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(self.time_emb_dim),
            nn.Linear(self.time_emb_dim, self.time_emb_dim),
            nn.ReLU()
        )

        self.down_convs = torch.nn.ModuleList()
        self.down_time = torch.nn.ModuleList()
        self.pools = torch.nn.ModuleList()
        self.down_convs.append(GCNConv(in_channels, channels, improved=True))
        for i in range(depth):
            self.pools.append(TopKPooling(channels, self.pool_ratios[i]))
            self.down_convs.append(GCNConv(channels, channels, improved=True))
            self.down_time.append(nn.Linear(self.time_emb_dim, channels))

        in_channels = channels if sum_res else 2 * channels

        self.up_convs = torch.nn.ModuleList()
        self.up_time = torch.nn.ModuleList()
        for i in range(depth - 1):
            self.up_convs.append(GCNConv(in_channels, channels, improved=True))
            self.up_time.append(nn.Linear(self.time_emb_dim, channels))
        self.up_convs.append(GCNConv(in_channels, out_channels, improved=True))

        self.reset_parameters()

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        for conv in self.down_convs:
            conv.reset_parameters()
        for pool in self.pools:
            pool.reset_parameters()
        for conv in self.up_convs:
            conv.reset_parameters()

    def forward(self, x: Tensor, edge_index: Tensor, timestep: Tensor,
                batch: OptTensor = None) -> Tensor:
        """"""
        t_mlp = self.time_mlp(timestep)

        if batch is None:
            batch = edge_index.new_zeros(x.size(0))
        edge_weight = x.new_ones(edge_index.size(1))

        x = self.down_convs[0](x, edge_index, edge_weight)
        x = self.act(x)

        xs = [x]
        edge_indices = [edge_index]
        edge_weights = [edge_weight]
        perms = []

        for i in range(1, self.depth + 1):
            edge_index, edge_weight = self.augment_adj(edge_index, edge_weight,
                                                       x.size(0))
            x, edge_index, edge_weight, batch, perm, _ = self.pools[i - 1](
                x, edge_index, edge_weight, batch)

            t = self.down_time[i - 1](t_mlp)
            x = x + t
            x = self.down_convs[i](x, edge_index, edge_weight)
            x = self.act(x)

            if i < self.depth:
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
            x = res + up if self.sum_res else torch.cat((res, up), dim=-1)

            t = self.down_time[i](t_mlp)
            x = x + t
            x = self.up_convs[i](x, edge_index, edge_weight)
            x = self.act(x) if i < self.depth - 1 else x

        return x

    def augment_adj(self, edge_index: Tensor, edge_weight: Tensor,
                    num_nodes: int) -> PairTensor:
        edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)
        edge_index, edge_weight = add_self_loops(edge_index, edge_weight,
                                                 num_nodes=num_nodes)
        adj = to_torch_csr_tensor(edge_index, edge_weight,
                                  size=(num_nodes, num_nodes))
        #adj = adj.to_dense()  # TODO: remove on AWS
        adj = (adj @ adj).to_sparse_coo()
        edge_index, edge_weight = adj.indices(), adj.values()
        edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)
        return edge_index, edge_weight

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.hidden_channels}, {self.out_channels}, '
                f'depth={self.depth}, pool_ratios={self.pool_ratios})')


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
