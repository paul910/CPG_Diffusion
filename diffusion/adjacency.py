import configparser
import math

import torch
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.data import Data

from diffusion.diffusionmanager import DiffusionManager
from utils.utils import to_adj, get_index_from_list


class Adjacency(DiffusionManager):

    def __init__(self, config: configparser.ConfigParser):
        super().__init__(config, "MODEL_ADJ")

    @torch.no_grad()
    def sample_timestep(self, graph: Data, t: Tensor):
        adj = to_adj(graph.edge_index).unsqueeze(0)

        betas_t = get_index_from_list(self.betas, t, adj.shape)
        sqrt_one_minus_alphas_cumprod_t = get_index_from_list(
            self.sqrt_one_minus_alphas_cumprod, t, adj.shape
        )
        sqrt_recip_alphas_t = get_index_from_list(self.sqrt_recip_alphas, t, adj.shape)

        model_mean = sqrt_recip_alphas_t * (
                adj - betas_t * self.model(adj, t) / sqrt_one_minus_alphas_cumprod_t
        )
        posterior_variance_t = get_index_from_list(self.posterior_variance, t, adj.shape)

        if t == 0:
            return model_mean
        else:
            noise = torch.randn_like(adj)
            return model_mean + torch.sqrt(posterior_variance_t) * noise

    def loss(self, graph):
        t = torch.randint(0, self.T, (1,), device=self.device).long()

        adj = self.pad(to_adj(graph.edge_index))

        adj_t, adj_noise = self.forward_diffusion_sample(adj, t)

        adj_t = adj_t.unsqueeze(0)

        adj_noise_pred = self.model(adj_t.to(self.device), t)

        adj_noise_pred.to(self.device)
        adj_noise.to(self.device)

        smooth_l1_loss = F.smooth_l1_loss(adj_noise, adj_noise_pred)
        mse_loss = F.mse_loss(adj_noise, adj_noise_pred)

        loss = smooth_l1_loss + mse_loss

        return loss, smooth_l1_loss, mse_loss

    def pad(self, adj):
        padding_size = int(adj.shape[-1] % (math.pow(2, self.depth)))

        if padding_size != 0:
            pad = int(math.pow(2, self.depth) - padding_size)
            if pad % 2 == 0:
                return F.pad(adj, (pad // 2, pad // 2, pad // 2, pad // 2), "constant", 0)
            else:
                return F.pad(adj, (pad // 2, pad // 2 + 1, pad // 2, pad // 2 + 1), "constant", 0)

        return adj
