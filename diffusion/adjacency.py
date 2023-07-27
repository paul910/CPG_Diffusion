import configparser

import torch
import torch.nn.functional as F
from torch import Tensor

from diffusion.diffusionmanager import DiffusionManager
from utils.utils import get_index_from_list


class Adjacency(DiffusionManager):

    def __init__(self, config: configparser.ConfigParser):
        super().__init__(config, "MODEL_ADJ")

    @torch.no_grad()
    def sample_timestep(self, adj: Tensor, t: Tensor):
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

    def loss(self, adj: Tensor, t: Tensor):
        adj_t, adj_noise = self.forward_diffusion_sample(adj, t)
        adj_noise_pred = self.model(adj_t.to(self.device), t)

        adj_noise_pred.to(self.device)
        adj_noise.to(self.device)

        mse_loss = F.mse_loss(adj_noise, adj_noise_pred)

        return mse_loss
