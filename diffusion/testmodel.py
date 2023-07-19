from typing import Any, Callable, Dict, Optional, Tuple, Union

import torch
from torch import Tensor, nn
from torch.nn import ModuleList
from torch_geometric.nn.conv import (
    GATConv,
    GATv2Conv,
    GCNConv,
    GINConv,
    MessagePassing,
    SAGEConv,
)
from torch_geometric.nn.models import MLP

from diffusion.model import SinusoidalPositionEmbeddings


class BasicGNN(torch.nn.Module):
    def __init__(
            self,
            in_channels: int = 178,
            hidden_channels: int = 178 * 4,
            num_layers: int = 4,
            out_channels: Optional[int] = 178,
            time_emb_dim: int = 32,
            act_first: bool = False,
            norm: Union[str, Callable, None] = None,
            norm_kwargs: Optional[Dict[str, Any]] = None,
            **kwargs,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.time_emb_dim = time_emb_dim

        self.act = nn.ReLU()
        self.act_first = act_first
        self.norm = norm if isinstance(norm, str) else None
        self.norm_kwargs = norm_kwargs

        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(self.time_emb_dim),
            nn.Linear(self.time_emb_dim, self.time_emb_dim),
            nn.ReLU()
        )

        self.convs = ModuleList()
        self.convs2 = ModuleList()
        self.times = ModuleList()

        self.convs.append(self.init_conv(in_channels, hidden_channels, **kwargs))

        for _ in range(num_layers - 2):
            self.convs.append(self.init_conv(hidden_channels, hidden_channels, **kwargs))
            self.times.append(nn.Linear(self.time_emb_dim, self.hidden_channels))
            self.convs2.append(self.init_conv(hidden_channels, hidden_channels, **kwargs))

        self.convs.append(self.init_conv(hidden_channels, out_channels, **kwargs))

        self.norms = ModuleList()
        self.norms2 = ModuleList()
        for _ in range(num_layers - 2):
            self.norms.append(nn.LayerNorm(hidden_channels))
            self.norms2.append(nn.LayerNorm(hidden_channels))

    def init_conv(self, in_channels: Union[int, Tuple[int, int]],
                  out_channels: int, **kwargs) -> MessagePassing:
        raise NotImplementedError

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        for conv in self.convs:
            conv.reset_parameters()
        for norm in self.norms or []:
            norm.reset_parameters()

    def forward(
            self,
            x: Tensor,
            edge_index: Tensor,
            timestep: Tensor,
    ) -> Tensor:

        t_mlp = self.time_mlp(timestep)

        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            x = self.act(x) if i < self.num_layers - 1 else x
            if 0 < i < self.num_layers - 1:
                x = self.norms[i - 1](x)
                x = x + self.times[i - 1](t_mlp)
                x = self.convs2[i - 1](x, edge_index)
                x = self.act(x)
                x = self.norms2[i - 1](x)
        return x


class GCN(BasicGNN):
    def init_conv(self, in_channels: int, out_channels: int,
                  **kwargs) -> MessagePassing:
        return GCNConv(in_channels, out_channels, **kwargs)


class GraphSAGE(BasicGNN):
    def init_conv(self, in_channels: Union[int, Tuple[int, int]],
                  out_channels: int, **kwargs) -> MessagePassing:
        return SAGEConv(in_channels, out_channels, **kwargs)


class GIN(BasicGNN):
    def init_conv(self, in_channels: int, out_channels: int,
                  **kwargs) -> MessagePassing:
        mlp = MLP(
            [in_channels, out_channels, out_channels],
            act=self.act,
            act_first=self.act_first,
            norm=self.norm,
            norm_kwargs=self.norm_kwargs,
        )
        return GINConv(mlp, **kwargs)


class GAT(BasicGNN):
    def init_conv(self, in_channels: Union[int, Tuple[int, int]],
                  out_channels: int, **kwargs) -> MessagePassing:

        v2 = kwargs.pop('v2', False)
        heads = kwargs.pop('heads', 1)
        concat = kwargs.pop('concat', True)

        # Do not use concatenation in case the layer `GATConv` layer maps to
        # the desired output channels (out_channels != None and jk != None):
        if getattr(self, '_is_conv_to_out', False):
            concat = False

        if concat and out_channels % heads != 0:
            raise ValueError(f"Ensure that the number of output channels of "
                             f"'GATConv' (got '{out_channels}') is divisible "
                             f"by the number of heads (got '{heads}')")

        if concat:
            out_channels = out_channels // heads

        Conv = GATConv if not v2 else GATv2Conv
        return Conv(in_channels, out_channels, heads=heads, concat=concat,
                    dropout=self.dropout.p, **kwargs)
