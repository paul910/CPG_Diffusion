import math

import torch
from torch import Tensor
from torch import nn
from torch_geometric.nn.conv import GCNConv


class GDNN(nn.Module):
    def __init__(self, in_channels: int, time_emb_dim: int, model_depth: int = 2, model_mult_factor: int = 3):
        super().__init__()

        self.depth_x = model_depth
        self.mult_factor = model_mult_factor
        self.time_emb_dim = time_emb_dim

        self.conv_x_in = GCNConv(in_channels, in_channels * self.mult_factor)

        self.conv_x = nn.ModuleList(
            [Block(int(in_channels * math.pow(self.mult_factor, i)), int(in_channels * (math.pow(self.mult_factor, i + 1))), time_emb_dim) for i in
             range(1, self.depth_x)] +
            [Block(int(in_channels * (math.pow(self.mult_factor, i + 1))), int(in_channels * math.pow(self.mult_factor, i)), time_emb_dim) for i in
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

        x = self.conv_x_in(x, edge_index)
        for conv in self.conv_x:
            x = conv(x, edge_index, t)
        x = self.conv_x_out(x, edge_index)

        return x


class Block(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim):
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, out_ch)
        self.conv1 = GCNConv(in_ch, out_ch)
        self.conv2 = GCNConv(out_ch, out_ch)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.relu = nn.ReLU()

    def forward(self, x, edge_index, t):
        h = self.bn1(self.relu(self.conv1(x, edge_index)))
        time_emb = self.relu(self.time_mlp(t))
        time_emb = time_emb.repeat(h.shape[0], 1)
        return self.bn2(self.relu(self.conv2(h + time_emb, edge_index)))


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
