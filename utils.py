import shutil

import matplotlib.pyplot as plt
import numpy as np
import torch
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
    x_first_50 = (x[:, :50] - 0.5) * 2
    x_last_50 = x[:, 50:] * 2

    return torch.cat((x_first_50, x_last_50), dim=1)


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
    ncols = len(values)
    nrows = 1

    aspect_ratio = values[0].shape[1] / values[0].shape[0]
    fig, axes = plt.subplots(nrows, ncols, figsize=(10 * ncols * aspect_ratio, 10))

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


def console_log(message: str, header=True):
    terminal_width, _ = shutil.get_terminal_size()
    line_width = terminal_width - 2

    msg_length = len(message)
    if msg_length % 2 == 1:
        msg_length += 1

    horizontal_dash_count = (line_width - msg_length) // 2
    i = 1 if (line_width - msg_length) % 2 == 1 else 0

    output_line = f"{'-' * (horizontal_dash_count + i)}{message.center(msg_length + 2)}{'-' * horizontal_dash_count}"
    separator_line = '-' * terminal_width

    if header:
        print(separator_line + "\n" + output_line + "\n" + separator_line)
    else:
        print(output_line)
