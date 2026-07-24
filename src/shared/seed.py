import random

import numpy as np
import torch


def set_seed(seed: int):
    """
    Seed all RNGs (python, numpy, torch CPU/CUDA) so a training run is
    reproducible, and disable cudnn's nondeterministic kernel selection.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False