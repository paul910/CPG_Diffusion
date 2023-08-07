import configparser
import random
import warnings
from datetime import datetime
from os import makedirs
from os.path import exists

import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.utils import dense_to_sparse
from tqdm import tqdm

from cpg_reconstruction.reconstruction import process
from data.dataset import CPGDataset
from diffusion.adjacency import Adjacency
from diffusion.features import Features
from utils.logger import Logger
from utils.utils import adjust_feature_values, to_adj, get_pad_size, ensure_features, ensure_adj, adjust_adj_values

warnings.filterwarnings("ignore")


class Diffusion:
    def __init__(self, configuration: configparser.ConfigParser):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config = configuration

        self.T = self.config.getint('DEFAULT', 'T')
        self.mode = self.config.get("DEFAULT", "mode")

        self.vulnerability = self.config.getboolean('DEFAULT', 'vulnerability')

        self.log_step = self.config.getint('LOGGING', 'log_step')

        self.num_samples = self.config.getint("SAMPLING", "num_samples")
        self.epochs = self.config.getint('TRAINING', 'epochs')
        self.model_adj_depth = self.config.getint("MODEL_ADJ", "depth")

        self.flag_adj = self.config.getboolean('DEBUG', 'adj')
        self.flag_features = self.config.getboolean('DEBUG', 'features')

        self.min_nodes = self.config.getint('DATASET', 'min_nodes')
        self.max_nodes = self.config.getint('DATASET', 'max_nodes')
        self.num_node_features = self.config.getint('DATASET', 'num_node_features')
        self.dataset_path = self.config.get('DATASET', 'dataset_path')
        self.dataset = CPGDataset(self.dataset_path, self.num_node_features)
        self.train_dataset, self.test_dataset = self.dataset.train_test_split()
        self.train_loader = DataLoader(self.train_dataset, shuffle=True)
        self.test_loader = DataLoader(self.test_dataset, shuffle=False)

        self.adjacency = Adjacency(configuration)
        self.features = Features(configuration)

        self.logger = Logger(configuration)

    def train(self):
        for epoch in range(self.epochs):
            for graph in tqdm(self.train_loader, total=len(self.train_loader), desc="Training"):
                if self.vulnerability and graph[0].y == 0:
                    continue
                elif not self.vulnerability and graph[0].y == 1:
                    continue

                t = torch.randint(0, self.T, (1,), device=self.device).long()

                if self.flag_adj:
                    self.adjacency.optimizer.zero_grad()
                    train_loss_adj = self.adjacency.loss(graph.edge_index, t)
                    train_loss_adj.backward()
                    self.adjacency.optimizer.step()
                else:
                    train_loss_adj = 0

                if self.flag_features:
                    self.features.optimizer.zero_grad()
                    train_loss_features = self.features.loss(graph, t)
                    train_loss_features.backward()
                    self.features.optimizer.step()
                else:
                    train_loss_features = 0

                self.logger.train_log(train_loss_adj, train_loss_features)

            self.adjacency.save_model()
            self.features.save_model()

            self.validate()

    @torch.no_grad()
    def validate(self):
        loss_adj = 0
        loss_features = 0
        for graph in tqdm(self.test_loader, total=len(self.test_loader), desc="Validating"):
            if self.vulnerability and graph[0].y == 0:
                continue
            elif not self.vulnerability and graph[0].y == 1:
                continue

            t = torch.randint(0, self.T, (1,), device=self.device).long()

            loss_adj += self.adjacency.loss(graph.edge_index, t) if self.flag_adj else 0
            loss_features += self.features.loss(graph, t) if self.flag_features else 0

        mean_losses_adj = loss_adj / len(self.test_loader)
        mean_losses_features = loss_features / len(self.test_loader)

        self.logger.val_log(mean_losses_adj, mean_losses_features)

    @torch.no_grad()
    def sample(self):
        if not exists('data/generated'):
            makedirs('data/generated')

        for _ in tqdm(range(self.num_samples), desc='Sampling'):
            num_nodes = random.randint(self.min_nodes, self.max_nodes)
            num_nodes = num_nodes + get_pad_size(num_nodes, self.model_adj_depth)

            if self.flag_adj:
                adj = torch.randn((1, 1, num_nodes, num_nodes), device=self.device)

                for i in tqdm(reversed(range(self.T))):
                    t = torch.full((1,), i, device=self.device, dtype=torch.long)
                    adj = self.adjacency.sample_timestep(adj, t)
                    adj = torch.clamp(adj, -1.0, 1.0)
                    if self.mode == "show" and i % self.log_step == 0:
                        self.logger.log_img(adj, "Sampled Adjacency")

                adj = ensure_adj(adj.squeeze())
                edge_index, _ = dense_to_sparse(adj)
            else:
                graph = next(iter(self.train_loader))
                edge_index = graph.edge_index

            if self.flag_features:
                x = torch.randn(num_nodes, self.num_node_features).to(self.device)

                for i in tqdm(reversed(range(self.T)), total=self.T, desc="Features Sampling"):
                    t = torch.full((1,), i, dtype=torch.long, device=self.device)
                    x = self.features.sample_timestep(Data(x=x, edge_index=edge_index), t)
                    x = torch.clamp(x, -1.0, 1.0)

                    if self.mode == "show" and i % self.log_step == 0:
                        self.logger.log_img(x, "Sampled Features")

                x = ensure_features(x)
            else:
                x = torch.randn(num_nodes, self.num_node_features).to(self.device)

            graph = Data(x=x, edge_index=edge_index)

            self.logger.log_img(to_adj(edge_index), f"Adjacency {num_nodes}")
            self.logger.log_img(x, f"Features {num_nodes}")

            filename = datetime.now().strftime("cpg_%Y_%m_%d_%H_%M_%S.pt")
            torch.save(graph, f'data/generated/{filename}')

    @torch.no_grad()
    def show_forward_diff(self):
        graph = next(iter(self.train_loader))

        if self.flag_adj:
            adj = adjust_adj_values(to_adj(graph.edge_index))
            for t in range(self.T):
                t = torch.full((1,), t, dtype=torch.long, device=self.device)
                adj, noise = self.adjacency.forward_diffusion_sample(adj, t)
                adj = torch.clamp(adj, -1., 1.)

                if t % self.log_step == 0:
                    self.logger.log_img(adj, "Forward Diffusion Adjacency")

        if self.flag_features:
            x = adjust_feature_values(graph.x)
            for t in range(self.T):
                t = torch.full((1,), t, dtype=torch.long, device=self.device)
                x, noise = self.features.forward_diffusion_sample(x, t)
                x = torch.clamp(x, -1., 1.)

                if t % self.log_step == 0:
                    self.logger.log_img(x, "Forward Diffusion Features")

    def start(self):
        if self.config.get('DEFAULT', 'mode') == 'train':
            self.train()
        elif self.config.get('DEFAULT', 'mode') == 'sample':
            self.sample()
        elif self.config.get('DEFAULT', 'mode') == 'show':
            # self.show_forward_diff()
            self.sample()
        elif self.config.get('DEFAULT', 'mode') == 'code':
            data = self.sample()
            for i in range(len(data)):
                self.logger.log_code(process(data[i]), "Code Reconstruction")

        self.logger.close()


if __name__ == '__main__':
    config = configparser.ConfigParser()
    config.read('config.ini')

    Diffusion(config).start()
