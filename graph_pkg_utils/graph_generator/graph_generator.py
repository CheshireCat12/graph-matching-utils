from argparse import ArgumentParser
from torch import randperm
from torch_geometric.datasets import TUDataset
import torch
import numpy as np
import random

__STANDARD_SEED = 42
__UPPERBOUND = 99999


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_permutations(len_dataset: int, n_permutations: int=None) -> List[int]:
    """
    Create dataset permutations

    :param len_dataset:
    :param n_permutations:
    :return:
    """
    set_seeds(__STANDARD_SEED)

    seeds = torch.randint(
        high=__UPPERBOUND,
        size=(n_permutations if n_permutations else 1,)
    )

    permutations = []
    for seed in seeds:
        set_seeds(seed)
        permutations.append(randperm(len_dataset))

    return permutations

from graph_pkg_utils.graph_generator.graph_convertor import convert


def generate(dataset: str, n_permutations: int, size_train: int=0.6, format: str='graphml') -> Tuple:
    """
    Generate data with split
    1) Shuffle the dataset by generating a permutation
    2) Split the data given the generated permutation
    """

    dataset = TUDataset(root=f'data/{args.dataset.upper()}',
                        name=args.dataset.upper())
    permutations = get_permutations(len_dataset=len(dataset),
                                    n_permutations=args.n_permutations)

    size_tr = args.size_train
    size_va = (1-args.size_train)/2

    sets = []
    indices = []

    for permutation in permutations:
        shuffled_dataset = dataset[permutation]

        tr_set = shuffled_dataset[:size_tr]
        va_set = shuffled_dataset[size_tr:size_tr + size_va]
        te_set = shuffled_dataset[size_tr + size_val:]

        idx_tr_set = permutation[:size_tr]
        idx_va_set = permutation[size_tr:size_tr + size_va]
        idx_te_set = permutation[size_tr + size_va:]

        sets.append((tr_set, va_set, te_set))
        indices.append((idx_tr_set, idx_va, idx_te))

    return sets, indices, permutations

