import math
import shutil
from random import random

import torch
import torch.nn.functional as F
from torch_geometric.utils import to_dense_adj


def geometric_beta_schedule(timesteps, start=0.0001, end=0.02):
    decay_rate = (end / start) ** (1.0 / (timesteps - 1))
    return torch.tensor([start * decay_rate ** i for i in range(timesteps)])


def get_index_from_list(vals, t, x_shape):
    batch_size = t.shape[0]
    vals = vals.to(t.device)
    out = vals.gather(-1, t)
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))


def to_adj(edge_index):
    return to_dense_adj(edge_index).squeeze()


def adjust_feature_values(x):
    x_first_50 = x[:, :50] * 2 - 1
    x_last_50 = x[:, 50:]

    return torch.cat((x_first_50, x_last_50), dim=1)


def adjust_adj_values(adj):
    adj[adj == 0.0] = -1.0
    return adj


def console_log(message: str, header=True):
    terminal_width, _ = shutil.get_terminal_size()
    line_width = terminal_width - 2

    msg_length = len(message)
    msg_length += 1 if msg_length % 2 == 1 else 0

    horizontal_dash_count = (line_width - msg_length) // 2
    i = 1 if (line_width - msg_length) % 2 == 1 else 0

    output_line = f"{'-' * (horizontal_dash_count + i)}{message.center(msg_length + 2)}{'-' * horizontal_dash_count}"
    separator_line = '-' * terminal_width

    if header:
        print(separator_line + "\n" + output_line + "\n" + separator_line)
    else:
        print(output_line)


def pad(adj, model_depth):
    padding = get_pad_size(adj.shape[-1], model_depth)

    if padding != 0:
        if padding % 2 == 0:
            return F.pad(adj, (padding // 2, padding // 2, padding // 2, padding // 2), "constant", -1.0)
        else:
            return F.pad(adj, (padding // 2, padding // 2 + 1, padding // 2, padding // 2 + 1), "constant", -1.0)

    return adj


def get_pad_size(num_nodes, model_depth):
    padding_size = int(num_nodes % (math.pow(2, model_depth)))

    if padding_size != 0:
        pad = int(math.pow(2, model_depth) - padding_size)
        return pad
    return 0


def ensure_features(features):
    # ensure one hot encoding for first 50 features in last timestep
    max_values, _ = torch.max(features[:, :50], dim=1, keepdim=True)
    features[:, :50] = torch.where(features[:, :50] == max_values, torch.tensor(1.), torch.tensor(-1.))

    return features


def ensure_adj(adj):
    # thresholding by number of edges. num_edges = num_nodes * (avg_degree +- 0.3)
    num_edges = int(adj.shape[-1] * (3.68 + random.uniform(-0.3, 0.3)))
    values, _ = torch.sort(torch.flatten(adj), descending=True)
    threshold = values[num_edges - 1]
    # ensure 0/1 encoding for last timestep in adjacency by thresholding
    adj = torch.where(adj >= threshold, torch.tensor(1.), torch.tensor(0.))

    return adj
