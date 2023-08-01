import configparser

import torch
import torch.nn.functional as F
from torch import Tensor

from datatype import Graph
from diffusion.diffusionmanager import DiffusionManager
from utils.utils import get_index_from_list


class Features(DiffusionManager):

    def __init__(self, config: configparser.ConfigParser):
        super().__init__(config, "MODEL_X")

    def forward_diffusion_sample(self, graph: Graph, t: Tensor) -> (Tensor, Tensor):
        features = graph.features

        noise = torch.randn_like(features)
        sqrt_alphas_cumprod_t = get_index_from_list(self.sqrt_alphas_cumprod, t, features.shape)
        sqrt_one_minus_alphas_cumprod_t = get_index_from_list(self.sqrt_one_minus_alphas_cumprod, t, features.shape)

        features = sqrt_alphas_cumprod_t.to(self.device) * features.to(
            self.device) + sqrt_one_minus_alphas_cumprod_t.to(self.device) * noise.to(self.device)
        noise = noise.to(self.device)

        features = torch.clamp(features, -1., 1.)

        return features, noise

    @torch.no_grad()
    def sample_timestep(self, graph: Graph, t: Tensor) -> Graph:
        features = graph.features

        betas_t = get_index_from_list(self.betas, t, features.shape)
        sqrt_one_minus_alphas_cumprod_t = get_index_from_list(self.sqrt_one_minus_alphas_cumprod, t, features.shape)
        sqrt_recip_alphas_t = get_index_from_list(self.sqrt_recip_alphas, t, features.shape)

        model_mean = sqrt_recip_alphas_t * (
                features - betas_t * self.model(features, graph.get_edge_index(), t) / sqrt_one_minus_alphas_cumprod_t)
        posterior_variance_t = get_index_from_list(self.posterior_variance, t, features.shape)

        features = model_mean if t == 0 else model_mean + torch.sqrt(posterior_variance_t) * torch.randn_like(features)
        features = torch.clamp(features, -1.0, 1.0)

        graph.set_features(features)

        return graph

    def loss(self, graph: Graph, t: Tensor) -> Tensor:
        x_t, x_noise = self.forward_diffusion_sample(graph, t)
        x_noise_pred = self.model(x_t.to(self.device), graph.get_edge_index().to(self.device), t)

        x_noise_pred.to(self.device)
        x_noise.to(self.device)

        mse_loss = F.mse_loss(x_noise, x_noise_pred)

        return mse_loss

    def ensure_valid(self, graph: Graph) -> Graph:
        features = graph.features

        # ensure one hot encoding for first 50 features in last timestep
        max_values, _ = torch.max(features[:, :50], dim=1, keepdim=True)
        features[:, :50] = torch.where(features[:, :50] == max_values, torch.tensor(1.), torch.tensor(-1.))

        graph.set_features(features)

        return graph

    def get_tensor(self, graph: Graph) -> Tensor:
        return graph.features
