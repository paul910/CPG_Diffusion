import configparser
import math
import random
import warnings
from datetime import datetime
from os import makedirs
from os.path import exists

import torch
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from cpg_reconstruction.reconstruction import process
from data.dataset import CPGDataset
from datatype import Graph
from diffusion.adjacency import Adjacency
from diffusion.diffusionmanager import DiffusionManager
from diffusion.features import Features
from utils.logger import Logger
from utils.utils import to_adj, get_pad_size, adjust_adj_values

warnings.filterwarnings("ignore")


class Diffusion:
    def __init__(self, configuration: configparser.ConfigParser):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config = configuration

        self.T = self.config.getint('DEFAULT', 'T')
        self.mode = self.config.get("DEFAULT", "mode")

        self.log_step = self.config.getint('LOGGING', 'log_step')

        self.num_samples = self.config.getint("SAMPLING", "num_samples")
        self.epochs = self.config.getint('TRAINING', 'epochs')
        self.model_adj_depth = self.config.getint("MODEL_ADJ", "depth")

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

    def train(self, dff_manager: DiffusionManager):
        for epoch in range(self.epochs):
            dff_manager.model.train()

            for step, pyg_graph in enumerate(tqdm(self.train_loader, total=len(self.train_loader), desc="Training")):
                graph = Graph(pyg_graph, self.model_adj_depth)
                if graph.adj.shape[0] < math.pow(2, self.model_adj_depth):
                    continue

                t = torch.randint(0, self.T, (1,), device=self.device).long()

                dff_manager.optimizer.zero_grad()
                loss = dff_manager.loss(graph, t)
                loss.backward()
                dff_manager.optimizer.step()

                self.logger.log(f"{str(dff_manager.model_name)}: Training", loss)

            dff_manager.save_model()
            self.validate(dff_manager)

    @torch.no_grad()
    def validate(self, dff_manager: DiffusionManager):
        dff_manager.model.eval()

        loss = 0
        for pyg_graph in tqdm(self.test_loader, total=len(self.test_loader), desc="Validating"):
            graph = Graph(pyg_graph)
            if graph.adj.shape[0] < math.pow(2, self.model_adj_depth):
                continue

            t = torch.randint(0, self.T, (1,), device=self.device).long()
            loss += dff_manager.loss(graph, t)

        self.logger.log(f"{str(dff_manager.model_name)}: Validation", loss / len(self.test_loader))

    @torch.no_grad()
    def sample(self, adj_manager: DiffusionManager, feature_manager: DiffusionManager):
        if not exists('data/generated'):
            makedirs('data/generated')

        num_nodes = 0
        for _ in tqdm(range(self.num_samples), desc=f'Sampling graph with {num_nodes} nodes'):
            num_nodes = random.randint(self.min_nodes, self.max_nodes)
            num_nodes = num_nodes + get_pad_size(num_nodes, self.model_adj_depth)

            graph = adj_manager.get_noisy_graph(num_nodes, self.num_node_features)

            graph = self.sample_part(adj_manager, graph)
            graph = self.sample_part(feature_manager, graph)

            filename = datetime.now().strftime("cpg_%Y_%m_%d_%H_%M_%S.pt")
            torch.save(graph.get_pyg_graph(), f'data/generated/{filename}')

    def sample_part(self, dff_manager: DiffusionManager, graph):
        for i in tqdm(reversed(range(self.T)), total=self.T,
                      desc=f"{str(dff_manager.model_name)}: Sampling graph with {graph.adj.shape[0]} nodes"):

            t = torch.full((1,), i, device=self.device, dtype=torch.long)
            graph = dff_manager.sample_timestep(graph, t)

            if i % self.log_step == 0:
                self.logger.log_img(dff_manager.get_tensor(graph), "Sampled " + str(dff_manager.model_name))

        return dff_manager.ensure_valid(graph)

    @torch.no_grad()
    def show_forward_diff(self, dff_manager: DiffusionManager):
        pyg_graph = next(iter(self.test_loader))
        graph = Graph(pyg_graph)

        for t in range(self.T):
            t = torch.full((1,), t, dtype=torch.long, device=self.device)
            adj, noise = dff_manager.forward_diffusion_sample(dff_manager.get_tensor(graph), t)
            adj = torch.clamp(adj, -1., 1.)

            if t % self.log_step == 0:
                self.logger.log_img(adj, "Forward Diffusion Adjacency")

    def start(self):
        if self.mode == 'train':
            self.train(self.adjacency)
            self.train(self.features)
        elif self.mode == 'train_adj':
            self.train(self.adjacency)
        elif self.mode == 'train_features':
            self.train(self.features)
        elif self.mode == 'validate':
            self.validate(self.adjacency)
            self.validate(self.features)
        elif self.mode == 'sample':
            self.sample(self.adjacency, self.features)
        elif self.mode == 'show':
            self.show_forward_diff(self.adjacency)
            self.show_forward_diff(self.features)
        elif self.mode == 'code':
            data = self.sample()  # TODO: sample code
            for i in range(len(data)):
                self.logger.log_code(process(data[i]), "Code Reconstruction")

        self.logger.close()


if __name__ == '__main__':
    config = configparser.ConfigParser()
    config.read('config.ini')

    Diffusion(config).start()
