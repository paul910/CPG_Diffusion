import os
from datetime import datetime

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.optim import Adam
from torch_geometric.data import Dataset
from torch_geometric.loader import DataLoader
from tqdm import tqdm

import wandb
from dataset import CPGDataset
from model import GDNN, GraphUNet
from utils import geometric_beta_schedule, get_index_from_list, plot, to_adj


class Diffusion:
    def __init__(self, dataset: Dataset, model_path=None):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.batch_size = 1
        self.epochs = 1000
        self.learning_rate = 0.00001

        self.dataset = dataset
        self.num_node_features = 178
        self.train_dataset, self.test_dataset = self.dataset.train_test_split()
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
        self.test_loader = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)

        self.model = "GraphUNet"

        self.model_path = model_path
        self.model_depth = 3
        self.model_mult_factor = 2
        self.time_embedding_size = 64

        if self.model == "GDNN":
            self.model = GDNN(self.num_node_features, self.time_embedding_size, self.model_depth,
                              self.model_mult_factor).to(self.device)
        elif self.model == "GraphUNet":
            self.model = GraphUNet(self.num_node_features, 1028, self.num_node_features, self.model_depth).to(self.device)

        if os.path.exists(self.model_path):
            self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))

        self.optimizer = Adam(self.model.parameters(), lr=self.learning_rate)

        self.T = 1000
        self.betas = geometric_beta_schedule(timesteps=self.T)
        alphas = 1. - self.betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
        self.posterior_variance = self.betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

        self.wandb = wandb
        self.config_wandb()

    def config_wandb(self):
        self.wandb.init(
            # set the wandb project where this run will be logged
            project="CPG_Diffusion_UNet",

            # track hyperparameters and run metadata
            config={
                "learning_rate": self.learning_rate,
                "model_depth": self.model_depth,
                "model_mult_factor": self.model_mult_factor,
                "time_embedding_size": self.time_embedding_size,
                "batch_size": self.batch_size,
                "epochs": self.epochs,
            }
        )

    def forward_diffusion_sample_x(self, x_0: Tensor, t: Tensor):
        # Initialize noise term for each node
        noise = torch.randn_like(x_0)

        # Get diffusion parameters for current time step
        sqrt_alphas_cumprod_t = get_index_from_list(self.sqrt_alphas_cumprod, t, x_0.shape)
        sqrt_one_minus_alphas_cumprod_t = get_index_from_list(self.sqrt_one_minus_alphas_cumprod, t, x_0.shape)

        # Compute diffused state for each node
        x_t = sqrt_alphas_cumprod_t.to(self.device) * x_0.to(self.device) + sqrt_one_minus_alphas_cumprod_t.to(
            self.device) * noise.to(self.device)

        return x_t, noise

    def forward_diffusion_sample_adj(self, adj_matrix: Tensor, t: Tensor):
        # Initialize noise term for each edge
        noise = torch.randn_like(adj_matrix)

        # Get diffusion parameters for current time step
        sqrt_alphas_cumprod_t = get_index_from_list(self.sqrt_alphas_cumprod, t, adj_matrix.shape)
        sqrt_one_minus_alphas_cumprod_t = get_index_from_list(self.sqrt_one_minus_alphas_cumprod, t, adj_matrix.shape)

        # Compute diffused state for each edge
        adj_t = sqrt_alphas_cumprod_t.to(self.device) * adj_matrix.to(self.device) + sqrt_one_minus_alphas_cumprod_t.to(
            self.device) * noise.to(self.device)

        return adj_t, noise

    @staticmethod
    def calculate_loss(x_pred: Tensor, x_target: Tensor):

        smooth_l1_loss = F.smooth_l1_loss(x_target, x_pred)
        mse_loss = F.mse_loss(x_target, x_pred)

        loss = smooth_l1_loss + mse_loss

        return loss, smooth_l1_loss, mse_loss

    def train(self):
        self.model.train()

        for epoch in range(self.epochs):
            print((100 * '-') + f'\n- Epoch: {epoch}\n' + (100 * '-'))
            for step, graph in enumerate(self.train_loader):
                self.optimizer.zero_grad()

                t = torch.randint(0, self.T, (1,), device=self.device).long()

                x_t, x_noise = self.forward_diffusion_sample_x(graph.x, t)

                x_noise_pred = self.model(x_t, graph.edge_index.to(self.device), t)

                loss, smooth_l1_loss, mse_loss = self.calculate_loss(x_noise_pred.to(self.device),
                                                                     x_noise.to(self.device))

                loss.backward()
                self.optimizer.step()

                self.wandb.log({"loss": loss.item(), "smooth_l1_loss": smooth_l1_loss.item(), "mse_loss": mse_loss.item()})

            if self.model_path is not None:
                self.save_model()

    @torch.no_grad()
    def sample_timestep(self, x, edge_index, t):
        x_betas_t = get_index_from_list(self.betas, t, x.shape)
        x_sqrt_one_minus_alphas_cumprod_t = get_index_from_list(
            self.sqrt_one_minus_alphas_cumprod, t, x.shape
        )
        x_sqrt_recip_alphas_t = get_index_from_list(self.sqrt_recip_alphas, t, x.shape)

        x_noise_pred = self.model(x, edge_index, t)

        x_model_mean = x_sqrt_recip_alphas_t * (
                x - x_betas_t * x_noise_pred / x_sqrt_one_minus_alphas_cumprod_t
        )
        x_posterior_variance_t = get_index_from_list(self.posterior_variance, t, x.shape)

        if t == 0:
            return x_model_mean
        else:
            x_noise = torch.randn_like(x)
            return x_model_mean + torch.sqrt(x_posterior_variance_t) * x_noise

    @torch.no_grad()
    def sample(self, num_nodes):
        self.model.eval()

        x = torch.randn(num_nodes, self.num_node_features).to(self.device)
        edge_index = self.test_loader.dataset[0].edge_index.to(self.device)

        x_out = []

        for i in tqdm(reversed(range(self.T)), total=self.T, desc='Sampling'):
            t = torch.full((1,), i, dtype=torch.long, device=self.device)
            x = self.sample_timestep(x, edge_index, t)
            x_out.append(x)

        return x_out

    def show_sample(self, num_show=10):
        x = self.sample(128)

        for step, i in enumerate(x):
            if step % (self.T / num_show) == 0:
                plot(i)

    def show_forward_diff(self, show_adj=False, show_x=False):
        for graph in self.train_loader:
            for step, t in enumerate(range(0, self.T, 100)):
                t = torch.full((1,), t, dtype=torch.long, device=self.device)
                if show_adj:
                    adj, noise = self.forward_diffusion_sample_adj(to_adj(graph.edge_index), t)
                    plot(adj, "Nodes", "Nodes", f"Step {step}")
                    plot(noise, "Nodes", "Nodes", f"Step {step}")
                if show_x:
                    x, noise = self.forward_diffusion_sample_x(graph.x, t)
                    plot(x, "Features", "Nodes", f"Step {step}")
                    plot(noise, "Features", "Nodes", f"Step {step}")
            break

    def save_model(self):
        torch.save(self.model.state_dict(), self.model_path)


def main():
    data_path = "data/reveal/"
    model_path = f"models/model_{datetime.now().time()}.pth"
    dataset = CPGDataset(data_path, model_path)

    diffusion = Diffusion(dataset, model_path)
    diffusion.train()
    diffusion.show_sample()


if __name__ == '__main__':
    main()
