import configparser
import os
import warnings

warnings.filterwarnings("ignore")

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.optim import Adam
from torch_geometric.loader import DataLoader
from tqdm import tqdm

import wandb
from dataset import CPGDataset
from model import GraphUNet
from utils import geometric_beta_schedule, get_index_from_list, to_adj, plot_array, adjust_feature_values, console_log, \
    plot


class Diffusion:
    def __init__(self, config: configparser.ConfigParser):
        self.config = config
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.epochs = config.getint('TRAINING', 'epochs')
        self.learning_rate = config.getfloat('TRAINING', 'learning_rate')

        self.dataset = CPGDataset(config.get('DATASET', 'dataset_path'), config.getint('DATASET', 'num_node_features'))
        self.num_node_features = config.getint('DATASET', 'num_node_features')
        self.train_dataset, self.test_dataset = self.dataset.train_test_split()
        self.train_loader = DataLoader(self.train_dataset, shuffle=True)
        self.test_loader = DataLoader(self.test_dataset, shuffle=False)

        self.model_path = config.get('MODEL', 'model_path')
        self.model_depth = config.getint('MODEL', 'model_depth')
        self.time_embedding_size = config.getint('MODEL', 'time_embedding_size')
        self.model = GraphUNet(self.num_node_features, config.getint('MODEL', 'hidden_size'), self.num_node_features,
                               self.model_depth).to(self.device)

        if os.path.exists(self.model_path):
            self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))

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

        self.val_losses = {}

        self.log_wandb = config.getboolean('LOGGING', 'log_wandb')
        if self.log_wandb:
            console_log('Configure wandb')
            self.wandb = wandb
            self.config_wandb()

    def config_wandb(self):
        self.wandb.init(
            # set the wandb project where this run will be logged
            project=f"CPG_Diffusion_All_{self.model.__class__.__name__}",

            # track hyperparameters and run metadata
            config={
                "learning_rate": self.learning_rate,
                "model_depth": self.model_depth,
                "time_embedding_size": self.time_embedding_size,
                "epochs": self.epochs,
            }
        )

    def save_model(self):
        torch.save(self.model.state_dict(), self.model_path)

    def forward_diffusion_sample(self, input: Tensor, t: Tensor):
        noise = torch.randn_like(input)
        sqrt_alphas_cumprod_t = get_index_from_list(self.sqrt_alphas_cumprod, t, input.shape)
        sqrt_one_minus_alphas_cumprod_t = get_index_from_list(
            self.sqrt_one_minus_alphas_cumprod, t, input.shape
        )
        return sqrt_alphas_cumprod_t.to(self.device) * input.to(self.device) \
               + sqrt_one_minus_alphas_cumprod_t.to(self.device) * noise.to(self.device), noise.to(self.device)

    @torch.no_grad()
    def sample_timestep(self, x, edge_index, t):
        betas_t = get_index_from_list(self.betas, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = get_index_from_list(
            self.sqrt_one_minus_alphas_cumprod, t, x.shape
        )
        sqrt_recip_alphas_t = get_index_from_list(self.sqrt_recip_alphas, t, x.shape)

        model_mean = sqrt_recip_alphas_t * (
                x - betas_t * self.model(x, edge_index, t) / sqrt_one_minus_alphas_cumprod_t
        )
        posterior_variance_t = get_index_from_list(self.posterior_variance, t, x.shape)

        if t == 0:
            return model_mean
        else:
            noise = torch.randn_like(x)
            return model_mean + torch.sqrt(posterior_variance_t) * noise

    def calculate_loss(self, graph):
        x = adjust_feature_values(graph.x)

        t = torch.randint(0, self.T, (1,), device=self.device).long()
        x_t, x_noise = self.forward_diffusion_sample(x, t)
        x_noise_pred = self.model(x_t.to(self.device), graph.edge_index.to(self.device), t)

        x_noise_pred.to(self.device)
        x_noise.to(self.device)

        smooth_l1_loss = F.smooth_l1_loss(x_noise, x_noise_pred)
        mse_loss = F.mse_loss(x_noise, x_noise_pred)

        loss = smooth_l1_loss + mse_loss

        return loss, smooth_l1_loss, mse_loss

    def train_loss(self, graph):
        loss, smooth_l1_loss, mse_loss = self.calculate_loss(graph)
        loss.backward()

        if self.log_wandb:
            self.wandb.log(
                {"loss": loss.item(), "smooth_l1_loss": smooth_l1_loss.item(), "mse_loss": mse_loss.item()})

    def val_loss(self, graph):
        loss, smooth_l1_loss, mse_loss = self.calculate_loss(graph)

        self.val_losses["total_loss"] += loss.item()
        self.val_losses["total_smooth_l1_loss"] += smooth_l1_loss.item()
        self.val_losses["total_mse_loss"] += mse_loss.item()

    def train(self):
        console_log('Model Training')

        for epoch in range(self.epochs):

            self.model.train()
            print((40 * '-') + f'Epoch: {epoch}'.center(20) + (40 * '-'))

            for step, graph in enumerate(tqdm(self.train_loader, total=len(self.train_loader), desc="Training")):
                self.optimizer.zero_grad()
                self.train_loss(graph)
                self.optimizer.step()

            self.save_model()
            self.validate()

    @torch.no_grad()
    def validate(self):
        self.model.eval()
        self.val_losses = {"total_loss": 0, "total_smooth_l1_loss": 0, "total_mse_loss": 0}

        for graph in tqdm(self.test_loader, total=len(self.test_loader), desc="Validating"):
            self.val_loss(graph)

        mean_losses = {key: val / len(self.test_loader) for key, val in self.val_losses.items()}
        if self.log_wandb:
            self.wandb.log({
                "val_loss": mean_losses["total_loss"],
                "val_smooth_l1_loss": mean_losses["total_smooth_l1_loss"],
                "val_mse_loss": mean_losses["total_mse_loss"]
            })
        else:
            console_log(f"Loss: {mean_losses['total_loss']}")

    def show_forward_diff(self, num_show=6):
        console_log('Show forward diffusion')

        graph = next(iter(self.train_loader))
        x = adjust_feature_values(graph[0].x)

        out = []
        for t in range(0, self.T, (self.T - 1) // (num_show - 1)):
            t = torch.full((1,), t, dtype=torch.long, device=self.device)
            x, noise = self.forward_diffusion_sample(x, t)
            x = x.clamp(-1, 1)
            out.append(x)

        plot_array(out, "Features", "Nodes", "Forward Diffusion")

    @torch.no_grad()
    def sample(self):
        self.model.eval()

        edge_index = self.test_loader.dataset[5].edge_index.to(self.device)
        x = torch.randn(to_adj(edge_index).shape[0], self.num_node_features).to(self.device)

        x_out = []

        for i in tqdm(reversed(range(self.T)), total=self.T, desc='Sampling'):
            t = torch.full((1,), i, dtype=torch.long, device=self.device)
            x = self.sample_timestep(x, edge_index, t)
            x = x if i == 0 else x.clamp(-1, 1)
            x_out.append(x)

        return x_out

    def show_sample(self, num_show=6):
        console_log('Show sample')

        x = self.sample()

        out = []
        for i in range(0, self.T, (self.T - 1) // (num_show - 1)):
            if i > self.T - (self.T - 1) // (num_show - 1):
                out.append(x[-1])
            else:
                out.append(x[i])

        plot_array(out, "Features", "Nodes", "Sample")

        # ensure one hot encoding for first 50 features in last timestep
        max_values, _ = torch.max(out[-1][:, :50], dim=1, keepdim=True)
        out[-1][:, :50] = torch.where(out[-1][:, :50] == max_values, torch.tensor(1.), torch.tensor(-1.))
        plot(out[-1], "Features", "Nodes", "Thresholded Sample")

    def start(self):
        if self.config.get('DEFAULT', 'mode') == 'train':
            self.train()
        elif self.config.get('DEFAULT', 'mode') == 'sample':
            self.show_sample()
        elif self.config.get('DEFAULT', 'mode') == 'forward_diffusion':
            self.show_forward_diff()