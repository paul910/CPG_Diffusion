import configparser

import wandb

from utils.utils import console_log


class Logger:
    def __init__(self, config: configparser.ConfigParser):
        self.config = config
        self.log_wandb = config.getboolean('LOGGING', 'log_wandb')

        if self.log_wandb and self.config.get("DEFAULT", "mode") == 'train':
            console_log('Configure wandb')
            self.wandb = wandb
            self.config_wandb()

    def config_wandb(self):
        self.wandb.init(project=self.config.get("DEFAULT", "project_name"))

    def val_log(self, mean_losses_adj, mean_losses_features):
        if self.log_wandb:
            self.wandb.log({
                "val_adjacency_loss": mean_losses_adj["total_loss"],
                "val_adjacency_smooth_l1_loss": mean_losses_adj["total_smooth_l1_loss"],
                "val_adjacency_mse_loss": mean_losses_adj["total_mse_loss"],
                "val_features_loss": mean_losses_features["total_loss"],
                "val_features_smooth_l1_loss": mean_losses_features["total_smooth_l1_loss"],
                "val_features_mse_loss": mean_losses_features["total_mse_loss"]
            })

    def train_log(self, train_loss_adj, train_loss_features):
        if self.log_wandb:
            self.wandb.log({
                "adjacency_loss": train_loss_adj[0],
                "adjacency_smooth_l1_loss": train_loss_adj[1],
                "adjacency_mse_loss": train_loss_adj[2],
                "features_loss": train_loss_features[0],
                "features_smooth_l1_loss": train_loss_features[1],
                "features_mse_loss": train_loss_features[2]
            })
