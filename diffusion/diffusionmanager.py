import configparser
import os
from abc import ABC, abstractmethod

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.optim import Adam

from datatype import Graph
from diffusion.model import GraphUNet, Unet
from utils.utils import geometric_beta_schedule


class DiffusionManager(ABC):

    def __init__(self, config: configparser.ConfigParser, model_name: str):
        self.config = config
        self.model_name = model_name
        self.model_config = config[model_name]
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.depth = self.model_config.getint('depth')
        self.start_units = self.model_config.getint('start_units')
        self.hidden_units = self.model_config.getint('hidden_units')
        self.time_emb_dim = self.model_config.getint('time_emb_dim')
        self.learning_rate = self.model_config.getfloat('learning_rate')

        self.num_node_features = config.getint('DATASET', 'num_node_features')

        if not os.path.exists("model"):
            os.makedirs("model")

        self.model_path = "model/" + model_name + "_depth" + str(self.depth) + "_start" + str(
            self.start_units) + "_hidden" + str(self.hidden_units) + "_time" + str(self.time_emb_dim) + ".pth"

        if model_name == 'MODEL_X':
            self.model = GraphUNet(self.start_units, self.hidden_units, self.start_units, self.depth,
                                   self.time_emb_dim).to(self.device)
        elif model_name == 'MODEL_ADJ':
            self.model = Unet(self.depth, self.start_units, self.time_emb_dim).to(self.device)

        self.load_model()

        self.optimizer = Adam(self.model.parameters(), lr=self.learning_rate)

        self.T = config.getint('DEFAULT', 'T')
        self.betas = geometric_beta_schedule(timesteps=self.T)
        alphas = 1. - self.betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
        self.posterior_variance = self.betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

    def save_model(self):
        torch.save(self.model.state_dict(), self.model_path)

    def load_model(self):
        if os.path.exists(self.model_path):
            self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))

    @abstractmethod
    def forward_diffusion_sample(self, graph: Graph, t: Tensor) -> Tensor:
        pass

    @abstractmethod
    def sample_timestep(self, graph: Graph, t: Tensor) -> Graph:
        pass

    @abstractmethod
    def loss(self, graph: Graph, t: Tensor):
        pass

    @abstractmethod
    def ensure_valid(self, graph: Graph):
        pass

    @abstractmethod
    def get_tensor(self, graph: Graph) -> Tensor:
        pass

    def get_noisy_graph(self, num_nodes: int, num_node_features: int) -> Graph:
        adj = torch.randn(num_nodes, num_nodes).to(self.device)
        features = torch.randn(num_nodes, num_node_features).to(self.device)

        return Graph(adj=adj, features=features, model_adj_depth=self.config.getint('MODEL_ADJ', 'depth'))
