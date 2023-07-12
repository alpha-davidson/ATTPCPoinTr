from utils import registry
import numpy as np
import torch
from torch.utils.data import TensorDataset


DATASETS = registry.Registry('dataset')


def build_dataset_from_cfg(cfg, default_args = None):
    """
    Build a dataset, defined by `dataset_name`.
    Args:
        cfg (eDICT): 
    Returns:
        Dataset: a constructed dataset specified by dataset_name.
    """
    return DATASETS.build(cfg, default_args = default_args)


def build_my_dataset(cfg):
    """
    Author: Ben Wagner
    
    """

    feats = np.load(cfg.partial.path)
    labels = np.load(cfg.complete.path)

    # assert feats.shape == (len(feats), cfg.partial.npoints, 3)
    # assert labels.shape == (len(labels), cfg.complete.npoints, 3)

    feats = torch.Tensor(feats)
    labels = torch.Tensor(labels)

    dataset = TensorDataset(feats, labels)


    return dataset


