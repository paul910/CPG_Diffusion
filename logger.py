import configparser

import wandb
from utils import console_log


class Logger:
    def __init__(self, config: configparser.ConfigParser):
        self.log_wandb = config.getboolean('LOGGING', 'log_wandb')

        if self.log_wandb:
            console_log('Configure wandb')
            self.wandb = wandb
            self.config_wandb()

    def config_wandb(self):
        self.wandb.init(
            # set the wandb project where this run will be logged
            project=self.config.get("DEFAULT", "project_name"),

            # track hyperparameters and run metadata
            config={
                "learning_rate": self.config.getfloat('TRAINING', 'learning_rate'),
                "model_depth": self.config.getint('MODEL', 'depth'),
                "time_embedding_size": self.config.getint('MODEL', 'time_emb_dim'),
                "epochs": self.config.getint('TRAINING', 'epochs'),
            }
        )

    def val_log(self, mean_losses, model_name):
        if model_name == "adjacency":
            if self.log_wandb:
                self.wandb.log({
                    "val_adjacency_loss": mean_losses["total_loss"],
                    "val_adjacency_smooth_l1_loss": mean_losses["total_smooth_l1_loss"],
                    "val_adjacency_mse_loss": mean_losses["total_mse_loss"],
                })
        elif model_name == "features":
            if self.log_wandb:
                self.wandb.log({
                    "val_features_loss": mean_losses["total_loss"],
                    "val_features_smooth_l1_loss": mean_losses["total_smooth_l1_loss"],
                    "val_features_mse_loss": mean_losses["total_mse_loss"],
                })

    def train_log(self, loss, smooth_l1_loss, mse_loss, model_name):
        if model_name == "adjacency":
            if self.log_wandb:
                self.wandb.log({
                    "adjacency_loss": loss,
                    "adjacency_smooth_l1_loss": smooth_l1_loss,
                    "adjacency_mse_loss": mse_loss,
                })
        elif model_name == "features":
            if self.log_wandb:
                self.wandb.log({
                    "features_loss": loss,
                    "features_smooth_l1_loss": smooth_l1_loss,
                    "features_mse_loss": mse_loss,
                })
