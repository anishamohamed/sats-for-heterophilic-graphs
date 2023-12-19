from torch import optim

class ZincLRScheduler(optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, lr, warmup):
        self.warmup = warmup
        if warmup is None:
            self.lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=0.5,
                patience=15,
                min_lr=1e-05,
                verbose=False,
            )
        else:
            self.lr_steps = (lr - 1e-6) / warmup
            self.decay_factor = lr * warmup**0.5
            super().__init__(optimizer)

    def get_lr(self):
        if self.warmup is None:
            return [group["lr"] for group in self.optimizer.param_groups]
        else:
            return [
                self._get_lr(group["initial_lr"], self.last_epoch)
                for group in self.optimizer.param_groups
            ]

    def _get_lr(self, initial_lr, s):
        if s < self.warmup:
            lr = 1e-6 + s * self.lr_steps
        else:
            lr = self.decay_factor * s**-0.5
        return lr
