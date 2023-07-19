import configparser
import os
from abc import ABC, abstractmethod

import torch
from torch import Tensor
import torch.nn.functional as F
from torch.optim import Adam

from diffusion.model import GraphUNet, Unet, TestModel
from utils.utils import get_index_from_list, geometric_beta_schedule


class DiffusionManager(ABC):

    def __init__(self, config: configparser.ConfigParser, model_name: str):
        self.config = config
        self.model_config = config[model_name]
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.depth = self.model_config.getint('depth')
        self.start_units = self.model_config.getint('start_units')
        self.hidden_units = self.model_config.getint('hidden_units')
        self.time_emb_dim = self.model_config.getint('time_emb_dim')
        self.learning_rate = self.model_config.getfloat('learning_rate')

        self.model_path = "model/store/" + model_name + "_depth" + str(self.depth) + "_start" + \
                          str(self.start_units) + "_hidden" + str(self.hidden_units) + "_time" + \
                          str(self.time_emb_dim) + ".pth"

        if model_name == 'MODEL_X':
            # self.model = GraphUNet(self.start_units, self.hidden_units, self.start_units, self.depth, self.time_emb_dim).to(self.device)
            self.model = TestModel(self.start_units, self.start_units, self.time_emb_dim).to(self.device)
        elif model_name == 'MODEL_ADJ':
            self.model = Unet(self.depth, self.start_units, self.time_emb_dim).to(self.device)

        self.load_model()

        self.optimizer = Adam(self.model.parameters(), lr=self.learning_rate)

        self.T = config.getint('TRAINING', 'T')
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

    def forward_diffusion_sample(self, input: Tensor, t: Tensor):
        noise = torch.randn_like(input)
        sqrt_alphas_cumprod_t = get_index_from_list(self.sqrt_alphas_cumprod, t, input.shape)
        sqrt_one_minus_alphas_cumprod_t = get_index_from_list(
            self.sqrt_one_minus_alphas_cumprod, t, input.shape
        )
        return sqrt_alphas_cumprod_t.to(self.device) * input.to(self.device) \
               + sqrt_one_minus_alphas_cumprod_t.to(self.device) * noise.to(self.device), noise.to(self.device)

    @abstractmethod
    def sample_timestep(self, graph, t):
        pass

    @abstractmethod
    def loss(self, graph):
        pass
