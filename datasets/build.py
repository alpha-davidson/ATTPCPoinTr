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

    feats = np.load(cfg.partial.path, allow_pickle=True)

    if feats.shape != (len(feats), len(feats[0]), 3):
        feats = feats[:, :, :3]
    labels = np.load(cfg.complete.path)
    if labels.shape != (len(labels), len(labels[0]), 3):
        labels = labels[:, :, :3]

    # feats = feats[:60, :, :]
    # labels = labels[:60, :, :]

    # assert len(feats) == len(labels), f'len(feats) == {len(feats)} -- len(labels) == {len(labels)}'
    # assert feats.shape == (len(feats), cfg.partial.npoints, 3), f'Expected {(len(feats), cfg.partial.npoints, 3)}, got {feats.shape}'
    # assert labels.shape == (len(labels), cfg.complete.npoints, 3), f'Expected {(len(labels), cfg.complete.npoints, 3)}, got {labels.shape}'

    feats = torch.Tensor(feats)
    labels = torch.Tensor(labels)

    # print(feats.size())
    # print(labels.size())

    dataset = TensorDataset(feats, labels)


    return dataset


