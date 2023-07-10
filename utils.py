import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.sparse import coo_matrix
from torch_geometric.utils import dense_to_sparse, to_dense_adj


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


def to_edge_index(adj):
    return dense_to_sparse(adj)[0]


def convert_adj_to_edge_index_and_edge_attr(adj):
    coo = coo_matrix(adj)
    edge_index = torch.tensor(np.vstack((np.array(coo.row), np.array(coo.col))), dtype=torch.long)
    edge_attr = torch.tensor(coo.data, dtype=torch.float)
    edge_attr = normalize(edge_attr)
    return edge_index, edge_attr


def normalize(matrix):
    min_val = torch.min(matrix)
    max_val = torch.max(matrix)
    return (matrix - min_val) / (max_val - min_val)


def plot(value, x_axis=None, yaxis=None, title=None):
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    plt.figure(figsize=(10, 10))
    chart = plt.imshow(value)
    plt.colorbar(chart, label='Pixel Value')
    if x_axis is not None:
        plt.xlabel(x_axis)
    if yaxis is not None:
        plt.ylabel(yaxis)
    if title is not None:
        plt.title(title)
    plt.show()


def plot_array(values, x_axis=None, yaxis=None, title=None):
    ncols = len(values)  # Number of columns equals the number of values
    nrows = 1  # Only one row

    # Adjust figure size according to the aspect ratio of the images
    aspect_ratio = values[0].shape[1] / values[0].shape[0]  # Assuming all images have the same shape
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(10 * ncols * aspect_ratio, 10))  # Adjust figure width according to aspect ratio

    # Identify global min and max
    global_min = np.inf
    global_max = -np.inf
    values_np = []
    for value in values:
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()
        global_min = min(global_min, value.min())
        global_max = max(global_max, value.max())
        values_np.append(value)

    # Create images
    imgs = [axes[i].imshow(value, vmin=global_min, vmax=global_max) for i, value in enumerate(values_np)]

    # Only set the x, y labels for the first subplot
    for i in range(ncols):
        if i == 0:
            if x_axis is not None:
                axes[i].set_xlabel(x_axis)
            if yaxis is not None:
                axes[i].set_ylabel(yaxis)

    # Adjust subplots to leave space for the colorbar
    fig.subplots_adjust(right=0.8, wspace=0.1)  # Adjust wspace to make gap as small as possible

    # Create a placeholder axes for colorbar
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(imgs[0], cax=cbar_ax, label='Pixel Value')

    # Set title for the whole plot
    if title is not None:
        plt.suptitle(title)

    plt.show()
