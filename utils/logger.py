import configparser

import wandb

from utils.utils import console_log


class Logger:
    def __init__(self, config: configparser.ConfigParser):
        self.config = config
        self.log_wandb = config.getboolean('LOGGING', 'log_wandb')

        if self.log_wandb:
            console_log('Configure wandb')
            self.wandb = wandb
            self.wandb.init(project=self.config.get("DEFAULT", "project_name"))

    def val_log(self, mean_losses_adj, mean_losses_features):
        if self.log_wandb:
            self.wandb.log({"val_adjacency_loss": mean_losses_adj, "val_features_loss": mean_losses_features})

    def train_log(self, train_loss_adj, train_loss_features):
        if self.log_wandb:
            self.wandb.log({"adjacency_loss": train_loss_adj, "features_loss": train_loss_features})

    def log_img(self, img, name):
        img = img.float().squeeze().detach().cpu().numpy()
        if self.log_wandb:
            self.wandb.log({name: [self.wandb.Image(img, caption=name)]})

    def log_code(self, code, name):
        if self.log_wandb:
            self.wandb.log({name: self.wandb.Html(code, inject=False)})

    def close(self):
        self.wandb.finish()
