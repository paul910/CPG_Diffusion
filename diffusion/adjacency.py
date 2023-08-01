import configparser
import random

import torch
import torch.nn.functional as F
from torch import Tensor

from datatype import Graph
from diffusion.diffusionmanager import DiffusionManager
from utils.utils import get_index_from_list


class Adjacency(DiffusionManager):

    def __init__(self, config: configparser.ConfigParser):
        super().__init__(config, "MODEL_ADJ")

    def forward_diffusion_sample(self, graph: Graph, t: Tensor) -> (Tensor, Tensor):
        adj = graph.adj

        noise = torch.randn_like(adj)
        sqrt_alphas_cumprod_t = get_index_from_list(self.sqrt_alphas_cumprod, t, adj.shape)
        sqrt_one_minus_alphas_cumprod_t = get_index_from_list(self.sqrt_one_minus_alphas_cumprod, t, adj.shape)

        adj = sqrt_alphas_cumprod_t.to(self.device) * adj.to(self.device) + sqrt_one_minus_alphas_cumprod_t.to(
            self.device) * noise.to(self.device)
        noise = noise.to(self.device)

        adj = torch.clamp(adj, -1., 1.)

        return adj, noise

    @torch.no_grad()
    def sample_timestep(self, graph: Graph, t: Tensor) -> Graph:
        adj = graph.adj

        betas_t = get_index_from_list(self.betas, t, adj.shape)
        sqrt_one_minus_alphas_cumprod_t = get_index_from_list(self.sqrt_one_minus_alphas_cumprod, t, adj.shape)
        sqrt_recip_alphas_t = get_index_from_list(self.sqrt_recip_alphas, t, adj.shape)

        model_mean = sqrt_recip_alphas_t * (adj - betas_t * self.model(adj, t) / sqrt_one_minus_alphas_cumprod_t)
        posterior_variance_t = get_index_from_list(self.posterior_variance, t, adj.shape)

        adj = model_mean if t == 0 else model_mean + torch.sqrt(posterior_variance_t) * torch.randn_like(adj)
        adj = torch.clamp(adj, -1.0, 1.0)

        graph.set_adj(adj)

        return graph

    def loss(self, graph: Graph, t: Tensor) -> Tensor:
        adj_t, adj_noise = self.forward_diffusion_sample(graph, t)
        adj_noise_pred = self.model(adj_t.to(self.device), t)

        adj_noise_pred.to(self.device)
        adj_noise.to(self.device)

        mse_loss = F.mse_loss(adj_noise, adj_noise_pred)

        return mse_loss

    def ensure_valid(self, graph: Graph) -> Graph:
        adj = graph.adj

        # thresholding by number of edges. num_edges = num_nodes * (avg_degree +- 0.3)
        num_edges = int(adj.shape[-1] * (3.68 + random.uniform(-0.3, 0.3)))
        values, _ = torch.sort(torch.flatten(adj), descending=True)
        threshold = values[num_edges - 1]
        # ensure 0/1 encoding for last timestep in adjacency by thresholding
        adj = torch.where(adj >= threshold, torch.tensor(1.), torch.tensor(0.))

        graph.set_adj(adj)

        return graph

    def get_tensor(self, graph: Graph) -> Tensor:
        return graph.adj
