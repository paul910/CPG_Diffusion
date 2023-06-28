import os
import warnings

warnings.filterwarnings("ignore")
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
    def __init__(self, dataset: Dataset, model_path=None, ):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.batch_size = 1
        self.epochs = 1000
        self.learning_rate = 0.0001
        self.weight_decay = 0.0001

        self.dataset = dataset
        self.first_features = True
        self.num_node_features = 50 if self.first_features else 128
        self.train_dataset, self.test_dataset = self.dataset.train_test_split()
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
        self.test_loader = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)

        self.model = "GraphUNet"

        self.model_path = model_path
        self.model_depth = 2
        self.time_embedding_size = 32

        if self.model == "GDNN":
            self.model_mult_factor = 2
            self.model = GDNN(self.num_node_features, self.time_embedding_size, self.model_depth,
                              self.model_mult_factor).to(self.device)
        elif self.model == "GraphUNet":
            self.model = GraphUNet(self.num_node_features, 512, self.num_node_features, self.model_depth).to(
                self.device)

        if os.path.exists(self.model_path):
            self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))

        self.optimizer = Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

        self.T = 1000
        self.betas = geometric_beta_schedule(timesteps=self.T)
        alphas = 1. - self.betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
        self.posterior_variance = self.betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

        self.log_wandb = True
        if self.log_wandb:
            self.wandb = wandb
            self.config_wandb()

    def config_wandb(self):
        self.wandb.init(
            # set the wandb project where this run will be logged
            project=f"CPG_Diffusion_{self.model.__class__.__name__}",

            # track hyperparameters and run metadata
            config={
                "learning_rate": self.learning_rate,
                "model_depth": self.model_depth,
                "time_embedding_size": self.time_embedding_size,
                "batch_size": self.batch_size,
                "epochs": self.epochs,
            }
        )

    def forward_diffusion_sample(self, input: Tensor, t: Tensor):
        noise = torch.randn_like(input)
        sqrt_alphas_cumprod_t = get_index_from_list(self.sqrt_alphas_cumprod, t, input.shape)
        sqrt_one_minus_alphas_cumprod_t = get_index_from_list(
            self.sqrt_one_minus_alphas_cumprod, t, input.shape
        )
        return sqrt_alphas_cumprod_t.to(self.device) * input.to(self.device) \
               + sqrt_one_minus_alphas_cumprod_t.to(self.device) * noise.to(self.device), noise.to(self.device)

    @staticmethod
    def calculate_loss(x_pred: Tensor, x_target: Tensor):

        smooth_l1_loss = F.smooth_l1_loss(x_target, x_pred)
        mse_loss = F.mse_loss(x_target, x_pred)

        loss = smooth_l1_loss + mse_loss

        return loss, smooth_l1_loss, mse_loss

    def train(self):
        print((100 * '-') + '\n' + (40 * '-') + 'Model Training'.center(20) + (40 * '-') + '\n' + (100 * '-'))
        self.model.train()

        for epoch in range(self.epochs):
            print((40 * '-') + f'Epoch: {epoch}'.center(20) + (40 * '-'))
            for step, graph in enumerate(tqdm(self.train_loader, total=len(self.train_loader), desc="Training")):
                self.optimizer.zero_grad()

                x = (graph.x[:, :50] - 0.5 if self.first_features else graph.x[:, 50:]) * 2

                t = torch.randint(0, self.T, (1,), device=self.device).long()

                x_t, x_noise = self.forward_diffusion_sample(x, t)

                x_noise_pred = self.model(x_t, graph.edge_index.to(self.device), t)

                loss, smooth_l1_loss, mse_loss = self.calculate_loss(x_noise_pred.to(self.device),
                                                                     x_noise.to(self.device))

                loss.backward()
                self.optimizer.step()

                if self.log_wandb:
                    self.wandb.log(
                        {"loss": loss.item(), "smooth_l1_loss": smooth_l1_loss.item(), "mse_loss": mse_loss.item()})

            if self.model_path is not None:
                self.save_model()

            self.validate()

            out = self.sample()
            torch.save(out, f"out_{epoch}.pt")

    def validate(self):
        self.model.eval()
        total_loss = 0
        total_smooth_l1_loss = 0
        total_mse_loss = 0
        with torch.no_grad():
            for graph in tqdm(self.test_loader, total=len(self.test_loader), desc="Validating"):
                x = (graph.x[:, :50] - 0.5 if self.first_features else graph.x[:, 50:]) * 2
                t = torch.randint(0, self.T, (1,), device=self.device).long()
                x_t, x_noise = self.forward_diffusion_sample(x, t)
                x_noise_pred = self.model(x_t, graph.edge_index.to(self.device), t)
                loss, smooth_l1_loss, mse_loss = self.calculate_loss(x_noise_pred.to(self.device),
                                                                     x_noise.to(self.device))
                total_loss += loss.item()
                total_smooth_l1_loss += smooth_l1_loss.item()
                total_mse_loss += mse_loss.item()
        mean_loss = total_loss / len(self.test_loader)
        mean_smooth_l1_loss = total_smooth_l1_loss / len(self.test_loader)
        mean_mse_loss = total_mse_loss / len(self.test_loader)
        if self.log_wandb:
            self.wandb.log(
                {"val_loss": mean_loss, "val_smooth_l1_loss": mean_smooth_l1_loss, "val_mse_loss": mean_mse_loss})
        self.model.train()

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

    @torch.no_grad()
    def sample(self):
        self.model.eval()

        edge_index = self.test_loader.dataset[0].edge_index.to(self.device)
        x = torch.randn(to_adj(edge_index).shape[0], self.num_node_features).to(self.device)

        x_out = []

        for i in tqdm(reversed(range(self.T)), total=self.T, desc='Sampling'):
            t = torch.full((1,), i, dtype=torch.long, device=self.device)
            x = self.sample_timestep(x, edge_index, t)
            x = x.clamp(-1, 1)
            x_out.append(x)

        # ensure one hot encoding for last timestep
        max_values, _ = torch.max(x_out[-1], dim=1, keepdim=True)
        x_out[-1] = torch.where(x_out[-1] == max_values, 1., -1.)

        return x_out

    def show_sample(self, num_show=10, pre_processed=None):
        print("Showing samples")

        x = self.sample() if pre_processed is None else pre_processed

        for step, i in enumerate(x):
            if step % (self.T / num_show) == 0:
                plot(i)

        plot(x[-1], "out")

    def show_forward_diff(self):
        print("Showing forward diffusion")

        for graph in self.train_loader:
            x = (graph.x[:, :50] - 0.5 if self.first_features else graph.x[:, 50:]) * 2

            for step, t in enumerate(range(0, self.T, 100)):
                t = torch.full((1,), t, dtype=torch.long, device=self.device)
                x, noise = self.forward_diffusion_sample(x, t)
                x = x.clamp(-1, 1)
                plot(x, "Features", "Nodes", f"Step {step}")
            break

    def save_model(self):
        torch.save(self.model.state_dict(), self.model_path)


def main():
    data_path = "data/reveal/"
    model_path = f"models/model_{datetime.now().time()}.pth"
    # model_path = 'models/model.pth'
    dataset = CPGDataset(data_path)
    diffusion = Diffusion(dataset, model_path)

    # diffusion.show_forward_diff()
    # diffusion.show_sample()

    diffusion.train()


if __name__ == '__main__':
    main()
