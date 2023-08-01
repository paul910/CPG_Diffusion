import math

import torch
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.utils import dense_to_sparse

from utils.utils import to_adj


class Graph:
    def __init__(self, pyg_graph: Data = None, model_adj_depth: int = None, adj: Tensor = None,
                 features: Tensor = None):
        if pyg_graph is not None:
            self.adj = to_adj(pyg_graph.edge_index)
            self.features = pyg_graph.x

            self.adjust_features()
            self.adjust_adj()
        elif adj is not None and features is not None:
            self.adj = adj
            self.features = features
        else:
            raise ValueError("Invalid arguments")

        self.trim(model_adj_depth)

    def get_edge_index(self):
        edge_index, _ = dense_to_sparse(self.adj)
        return edge_index

    def get_pyg_graph(self) -> Data:
        edge_index = self.get_edge_index()
        return Data(x=self.features, edge_index=edge_index)

    def set_adj(self, adj):
        self.adj = adj

    def set_features(self, features):
        self.features = features

    def adjust_features(self):
        x_first_50 = self.features[:, :50] * 2 - 1
        x_last_50 = self.features[:, 50:]

        self.features = torch.cat((x_first_50, x_last_50), dim=1)

    def adjust_adj(self):
        self.adj[self.adj == 0.0] = -1.0

    def trim(self, model_depth):
        trim_val = int(self.adj.shape[-1] % (math.pow(2, model_depth)))
        if trim_val != 0:
            self.adj = self.adj[:-trim_val, :-trim_val]
            self.features = self.features[:-trim_val, :]
