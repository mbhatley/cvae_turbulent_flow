import torch
from torch import nn


class CVAEScheduler(nn.Module):
    def __init__(self):
        super().__init__()

    @staticmethod
    def get_scheduler(optimizer, epochs, steps_per_epoch):
        """
        Cosine warmup learning rate implementation
        """
        warmup_epochs = 10 # max(5, int(0.1 * epochs))
        cos_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs - warmup_epochs, eta_min=1e-6
        )
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cos_scheduler],
            milestones=[warmup_epochs]
        )

        return scheduler, 'epoch'


# Module-level convenience function so train.py can call get_scheduler() directly
def get_scheduler(optimizer, epochs, steps_per_epoch):
    return CVAEScheduler.get_scheduler(optimizer, epochs, steps_per_epoch)
