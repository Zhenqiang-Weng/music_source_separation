from torch.optim.lr_scheduler import _LRScheduler


class NoamScheduler(_LRScheduler):
    """Set adam lr to the max desired lr - it will be the peak lr"""

    def __init__(self, optimizer, n_warmup, init_scale, min_lr=0):
        """
        true_lr = init_scale * lr in first step

        :param optimizer:
        :param warmup_steps:
        :param init_scale:
        :param min: minimum value of lr
        """
        self.n_warmup = n_warmup
        self.min_lr = min_lr
        self.init_scale = init_scale
        super().__init__(optimizer)

    def get_lr(self):
        last_epoch = self.last_epoch

        if last_epoch <= self.n_warmup:
            c = (1 - self.init_scale) / self.n_warmup
            scale = c * last_epoch + self.init_scale
        else:
            scale = max(1, self.n_warmup) ** 0.5 * last_epoch ** (-0.5)

            # scale = self.warmup_steps ** 0.5 * min(last_epoch ** (-0.5), last_epoch * self.warmup_steps ** (-1.5))
        return [max(base_lr * scale, self.min_lr) for base_lr in self.base_lrs]
