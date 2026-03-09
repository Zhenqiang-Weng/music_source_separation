"""
Wrapper class for logging into the Tensorboard and wandb
"""

import os
from tensorboardX import SummaryWriter
try:
    import wandb
except ImportError:
    wandb =None


class Logger(object):

    def __init__(self, hparams, wandb_info=None):
        self.logdir = os.path.join(hparams.out_path, hparams.logdir)
        self.writer = SummaryWriter(log_dir=self.logdir)
        self.wandb = None if wandb_info is None else wandb
        if self.wandb and self.wandb.run is None:
            self.wandb.init(**wandb_info)

    def log_model(self, model):
        self.writer.add_graph(model)
        if self.wandb is not None:
            self.wandb.watch(model)

    def log_train(self, phase, unit, count, loss_dict, lr_dict=None, image_dict=None):
        if lr_dict is not None:
            loss_dict.update(lr_dict) # scalar info
        for key in sorted(loss_dict)
            self.writer.add_scalar(f"{phase}/{unit}/{key}", loss_dict[key], count)
        if self.wandb is not None:
            self.wandb.log({f"{phase}/{unit}/{key}": val for key, val in loss_dict.items()})

        if image_dict is not None:
            for key in sorted(image_dict):
                self.writer.add_figure(f"{phase}/{unit}/{key}", image_dict[key], count)

            if self.wandb is not None:
                self.wandb.log({f"{phase}/{unit}/{key}":
                                self.wandb.Image(val) for key, val in image_dict.items()})