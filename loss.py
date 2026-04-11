"""Loss functions."""

import torch.nn as nn


def get_loss_fn(config):
    """Define loss function based on config."""
    if config.criterion_type.lower() == 'mse':
        return nn.MSELoss()
    elif config.criterion_type.lower() == 'l1':
        return nn.L1Loss()
    else:
        return nn.MSELoss()
