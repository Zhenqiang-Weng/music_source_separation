import torch
import numpy as np


class ScheduledOptim:
    """A simple wrapper class for learning rate scheduling"""
    def __init__(self, model, hparams, current_step=0):
        self._optimizer = torch.optim.Adam(
            model.parameters(),
            betas=hparams.betas,
            eps=hparams.eps,
            weight_decay=hparams.weight_decay,)

        self.n_warmup_steps = hparams.n_warmup_step
        self.anneal_steps = hparams.anneal_steps
        self.anneal_rate = hparams.anneal_rate
        self.current_step = current_step
        self.init_lr = hparams.init_ir or np.power(hparams.channels, -0.5)
    def step_and_update_lr_frozen(self, frozen_lr, scaler=None):
        for param_group in self._optimizer.param_groups:
            param_group['lr'] = frozen_lr
            self._optimizer.step()
    def step_and_update_lr(self, scaler=None):
        self._update_learning_rate()
        if scaler is None:
            self._optimizer.step()
        else:
            scaler.unscale_(self._optimizer)
            scaler.step(self._optimizer)
    def get_learning_rate(self):
        learning_rate = 0.0
        for param_group in self._optimizer.param_groups:
            learningrate = param_group['lr']

        return learningrate

    def zero_grad(self):
        self._optimizer.zero_grad()

    def set_current_step(self, step):
        self.current_step = step
    def set_init_lr(self, init_lr):
        self.init_lr = init_lr
    def load_state_dict(self, state_dict):
        self._optimizer.load_state_dict(state_dict)
    def state_dict(self):
        return self._optimizer.state_dict()
    def _get_lr_scale(self):
        lr=np.min(
            [
            np.power(self.current_step, -0.5),
            np.power(self.n_warmup_steps, -1.5) * self.current_step,])
        for s in self.anneal_steps:
            if self.current_step > s:
                lr = lr * self.anneal_rate
        return lr
    def	_update_learning_rate(self):
        """ Learning rate scheduling per step """
        self.current_step += 1
        lr = self.init_lr * self._get_lr_scale()
        for param_group in self._optimizer.param_groups:
            param_group["lr"] = lr


def get_lr(current_step, init_lr=0.001, n_warmup_steps=6000, \
            anneal_steps=[300000, 400000, 500000], anneal_rate=0.3):
    lr=np.min(
        [
        np.power(current_step, -0.5),
        np.power(n_warmup_steps,-1.5) * current_step,])
    for	s in anneal_steps:
        if current_step > s:
            lr = lr * anneal_rate
    return init_lr * lr
