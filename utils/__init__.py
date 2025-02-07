import os
import random

from loguru import logger


def seed_everything(seed: int) -> int:
    """Stolen from https://pytorch-lightning.readthedocs.io/en/1.7.7/_modules/pytorch_lightning/utilities/seed.html#seed_everything

    Function that sets seed for pseudo-random number generators in: pytorch, numpy, python.random In addition,
    sets the following environment variables:

    - `PL_GLOBAL_SEED`: will be passed to spawned subprocesses (e.g. ddp_spawn backend).
    - `PL_SEED_WORKERS`: (optional) is set to 1 if ``workers=True``.

    Args:
        seed: the integer value seed for global random state in Lightning.
    """
    if seed is None:
        raise ValueError(
            "seed_everything() called without a seed. A seed must be provided or set as an environment variable PL_GLOBAL_SEED")
    elif not isinstance(seed, int):
        seed = int(seed)

    logger.info(f"Global seed set to {seed}")
    os.environ["PL_GLOBAL_SEED"] = str(seed)
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
        logger.info(f"numpy seed set to {seed}")
    except ImportError:
        logger.warning("Could not set numpy seed.")
        pass
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        logger.info(f"PyTorch seed set to {seed}")
    except ImportError:
        logger.warning("Could not set torch seed.")
        pass

    return seed