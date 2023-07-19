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


class TestModel(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_emb_dim: int):
        super().__init__()
        self.img = 32

        self.time_emb_dim = time_emb_dim
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(self.time_emb_dim),
            nn.Linear(self.time_emb_dim, self.time_emb_dim),
            nn.ReLU()
        )

        self.act = nn.ReLU()
        self.conv_in = GCNConv(in_channels, in_channels)

        self.conv1 = GCNConv(in_channels, in_channels * 2)
        self.n1 = nn.LayerNorm(in_channels * 2)
        self.time1 = nn.Linear(time_emb_dim, in_channels * 2)
        self.conv1_2 = GCNConv(in_channels * 2, in_channels * 2)
        self.n1_2 = nn.LayerNorm(in_channels * 2)

        self.conv2 = GCNConv(in_channels * 2, in_channels * 4)
        self.n2 = nn.LayerNorm(in_channels * 4)
        self.time2 = nn.Linear(time_emb_dim, in_channels * 4)
        self.conv2_2 = GCNConv(in_channels * 4, in_channels * 4)
        self.n2_2 = nn.LayerNorm(in_channels * 4)

        self.conv3 = GCNConv(in_channels * 4, in_channels * 2)
        self.n3 = nn.LayerNorm(in_channels * 2)
        self.time3 = nn.Linear(time_emb_dim, in_channels * 2)
        self.conv3_2 = GCNConv(in_channels * 2, in_channels * 2)
        self.n3_2 = nn.LayerNorm(in_channels * 2)

        self.conv4 = GCNConv(in_channels * 2, in_channels)
        self.n4 = nn.LayerNorm(in_channels)
        self.time4 = nn.Linear(time_emb_dim, in_channels)
        self.conv4_2 = GCNConv(in_channels, in_channels)
        self.n4_2 = nn.LayerNorm(in_channels)

        self.conv_out = GCNConv(in_channels, out_channels)

        self.conv_im_in = nn.Conv2d(1, self.img, 3, padding=1)

        self.conv_im_1 = nn.Conv2d(self.img, self.img * 2, 3, padding=1)
        self.n_im_1 = nn.BatchNorm2d(self.img * 2)
        self.time_im_1 = nn.Linear(time_emb_dim, self.img * 2)
        self.conv_im_1_2 = nn.Conv2d(self.img * 2, self.img * 2, 3, padding=1)
        self.n_im_1_2 = nn.BatchNorm2d(self.img * 2)

        self.conv_im_2 = nn.Conv2d(self.img * 2, self.img * 4, 3, padding=1)
        self.n_im_2 = nn.BatchNorm2d(self.img * 4)
        self.time_im_2 = nn.Linear(time_emb_dim, self.img * 4)
        self.conv_im_2_2 = nn.Conv2d(self.img * 4, self.img * 4, 3, padding=1)
        self.n_im_2_2 = nn.BatchNorm2d(self.img * 4)

        self.conv_im_3 = nn.Conv2d(self.img * 4, self.img * 2, 3, padding=1)
        self.n_im_3 = nn.BatchNorm2d(self.img * 2)
        self.time_im_3 = nn.Linear(time_emb_dim, self.img * 2)
        self.conv_im_3_2 = nn.Conv2d(self.img * 2, self.img * 2, 3, padding=1)
        self.n_im_3_2 = nn.BatchNorm2d(self.img * 2)

        self.conv_im_4 = nn.Conv2d(self.img * 2, self.img, 3, padding=1)
        self.n_im_4 = nn.BatchNorm2d(self.img)
        self.time_im_4 = nn.Linear(time_emb_dim, self.img)
        self.conv_im_4_2 = nn.Conv2d(self.img, self.img, 3, padding=1)
        self.n_im_4_2 = nn.BatchNorm2d(self.img)

        self.conv_im_out = nn.Conv2d(self.img, out_channels, 3, padding=1)

    def forward(self, x: Tensor, edge_index: Tensor, t: Tensor) -> Tensor:
        t_mlp = self.time_mlp(t)

        x = self.act(self.conv_in(x, edge_index))

        x = self.n1(self.act(self.conv1(x, edge_index)))
        x = x + self.act(self.time1(t_mlp))
        x = self.n1_2(self.act(self.conv1_2(x, edge_index)))

        x = self.n2(self.act(self.conv2(x, edge_index)))
        x = x + self.act(self.time2(t_mlp))
        x = self.n2_2(self.act(self.conv2_2(x, edge_index)))

        x = self.n3(self.act(self.conv3(x, edge_index)))
        x = x + self.act(self.time3(t_mlp))
        x = self.n3_2(self.act(self.conv3_2(x, edge_index)))

        x = self.n4(self.act(self.conv4(x, edge_index)))
        x = x + self.act(self.time4(t_mlp))
        x = self.n4_2(self.act(self.conv4_2(x, edge_index)))

        x = self.act(self.conv_out(x, edge_index))

        x = x.unsqueeze(0).unsqueeze(0)

        x = self.act(self.conv_im_in(x))

        x = self.n_im_1(self.act(self.conv_im_1(x)))
        time_emb = self.act(self.time_im_1(t_mlp))
        time_emb = time_emb[(...,) + (None,) * 2]
        x = x + time_emb
        x = self.n_im_1_2(self.act(self.conv_im_1_2(x)))

        x = self.n_im_2(self.act(self.conv_im_2(x)))
        time_emb = self.act(self.time_im_2(t_mlp))
        time_emb = time_emb[(...,) + (None,) * 2]
        x = x + time_emb
        x = self.n_im_2_2(self.act(self.conv_im_2_2(x)))

        x = self.n_im_3(self.act(self.conv_im_3(x)))
        time_emb = self.act(self.time_im_3(t_mlp))
        time_emb = time_emb[(...,) + (None,) * 2]
        x = x + time_emb
        x = self.n_im_3_2(self.act(self.conv_im_3_2(x)))

        x = self.n_im_4(self.act(self.conv_im_4(x)))
        time_emb = self.act(self.time_im_4(t_mlp))
        time_emb = time_emb[(...,) + (None,) * 2]
        x = x + time_emb
        x = self.n_im_4_2(self.act(self.conv_im_4_2(x)))

        x = self.conv_im_out(x)

        return x.squeeze(0).squeeze(0)


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
        adj = adj.to_dense()  # TODO: remove on AWS
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
