import configparser
import random
import warnings

import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.utils import dense_to_sparse
from tqdm import tqdm

from adjacency import Adjacency
from dataset import CPGDataset
from features import Features
from logger import Logger
from utils import console_log

warnings.filterwarnings("ignore")


class Diffusion:
    def __init__(self, config: configparser.ConfigParser):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config = config

        self.epochs = config.getint('TRAINING', 'epochs')

        self.dataset = CPGDataset(config.get('DATASET', 'dataset_path'), config.getint('DATASET', 'num_node_features'))
        self.num_node_features = config.getint('DATASET', 'num_node_features')
        self.train_dataset, self.test_dataset = self.dataset.train_test_split()
        self.train_loader = DataLoader(self.train_dataset, shuffle=True)
        self.test_loader = DataLoader(self.test_dataset, shuffle=False)

        self.adjacency = Adjacency(config)
        self.features = Features(config)

        self.logger = Logger(config)

    def train(self):
        console_log('Model Training')

        for epoch in range(self.epochs):

            self.adjacency.model.train()
            self.features.model.train()

            console_log(f'Epoch: {epoch}', False)

            for step, graph in enumerate(tqdm(self.train_loader, total=len(self.train_loader), desc="Training")):
                self.adjacency.optimizer.zero_grad()
                loss, smooth_l1_loss, mse_loss = self.adjacency.loss(graph)
                self.logger.train_log(loss, smooth_l1_loss, mse_loss, "adjacency")
                loss.backward()
                self.adjacency.optimizer.step()

                self.features.optimizer.zero_grad()
                loss, smooth_l1_loss, mse_loss = self.features.loss(graph)
                self.logger.train_log(loss, smooth_l1_loss, mse_loss, "features")
                loss.backward()
                self.features.optimizer.step()

            self.adjacency.save_model()
            self.features.save_model()

            self.validate()

    @torch.no_grad()
    def validate(self):
        loss_adj = {"total_loss": 0, "total_smooth_l1_loss": 0, "total_mse_loss": 0}
        loss_features = {"total_loss": 0, "total_smooth_l1_loss": 0, "total_mse_loss": 0}

        self.adjacency.model.eval()
        self.features.model.eval()

        for graph in tqdm(self.test_loader, total=len(self.test_loader), desc="Validating"):
            loss, smooth_l1_loss, mse_loss = self.adjacency.loss(graph)
            loss_adj["total_loss"] += loss
            loss_adj["total_smooth_l1_loss"] += smooth_l1_loss
            loss_adj["total_mse_loss"] += mse_loss

            loss, smooth_l1_loss, mse_loss = self.features.loss(graph)
            loss_features["total_loss"] += loss
            loss_features["total_smooth_l1_loss"] += smooth_l1_loss
            loss_features["total_mse_loss"] += mse_loss

        mean_losses_adj = {key: val / len(self.test_loader) for key, val in loss_adj.items()}
        mean_losses_features = {key: val / len(self.test_loader) for key, val in loss_features.items()}

        self.logger.val_log(mean_losses_adj, "adjacency")
        self.logger.val_log(mean_losses_features, "features")

    @torch.no_grad()
    def sample(self):
        self.adjacency.model.eval()
        self.features.model.eval()

        data = []

        for _ in tqdm(range(self.config.getint("SAMPLING", "num_samples")), desc='Sampling'):
            num_nodes = random.randint(self.config.getint('DATASET', 'min_nodes'),
                                       self.config.getint('DATASET', 'max_nodes'))
            x = torch.randn(num_nodes, self.num_node_features).to(self.device)
            adj = torch.randint(2, (num_nodes, num_nodes)).to(self.device)

            for i in reversed(range(self.config.getint("TRAINING", "T"))):
                t = torch.full((1,), i, dtype=torch.long, device=self.device)
                adj = self.adjacency.sample_timestep(Data(x=x, edge_index=dense_to_sparse(adj)), t)

            edge_index = dense_to_sparse(adj).to(self.device)
            for i in reversed(range(self.config.getint("TRAINING", "T"))):
                t = torch.full((1,), i, dtype=torch.long, device=self.device)
                x = self.features.sample_timestep(Data(x=x, edge_index=edge_index), t)
                x = x if i == 0 else x.clamp(-1, 1)

            data.append(Data(x=x, edge_index=edge_index))

        return data

    '''
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
    '''

    def start(self):
        if self.config.get('DEFAULT', 'mode') == 'train':
            self.train()
        elif self.config.get('DEFAULT', 'mode') == 'sample':
            _ = self.sample()
        elif self.config.get('DEFAULT', 'mode') == 'forward_diffusion':
            # self.show_forward_diff()
            pass
