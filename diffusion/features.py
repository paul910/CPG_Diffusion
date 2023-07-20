import configparser

import torch
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.data import Data

from diffusion.diffusionmanager import DiffusionManager
from utils.utils import adjust_feature_values, get_index_from_list


class Features(DiffusionManager):

    def __init__(self, config: configparser.ConfigParser):
        super().__init__(config, "MODEL_X")

    @torch.no_grad()
    def sample_timestep(self, graph: Data, t: Tensor):
        betas_t = get_index_from_list(self.betas, t, graph.x.shape)
        sqrt_one_minus_alphas_cumprod_t = get_index_from_list(
            self.sqrt_one_minus_alphas_cumprod, t, graph.x.shape
        )
        sqrt_recip_alphas_t = get_index_from_list(self.sqrt_recip_alphas, t, graph.x.shape)

        model_mean = sqrt_recip_alphas_t * (
                graph.x - betas_t * self.model(graph.x, graph.edge_index, t) / sqrt_one_minus_alphas_cumprod_t
        )
        posterior_variance_t = get_index_from_list(self.posterior_variance, t, graph.x.shape)

        if t == 0:
            return model_mean
        else:
            noise = torch.randn_like(graph.x)
            return model_mean + torch.sqrt(posterior_variance_t) * noise

    def loss(self, graph):
        # x = adjust_feature_values(graph.x)
        x = graph.x

        t = torch.randint(0, self.T, (1,), device=self.device).long()
        x_t, x_noise = self.forward_diffusion_sample(x, t)
        x_noise_pred = self.model(x_t.to(self.device), graph.edge_index.to(self.device), t)

        x_noise_pred.to(self.device)
        x_noise.to(self.device)

        mse_loss = F.mse_loss(x_noise, x_noise_pred)

        return mse_loss
