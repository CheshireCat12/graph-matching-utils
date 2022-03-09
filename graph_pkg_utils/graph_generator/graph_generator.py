import torch

from torch import randperm
from torch_geometric import seed_everything
from torch_geometric.datasets import TUDataset
from typing import List, Tuple

__STANDARD_SEED = 42
__UPPERBOUND = 99999


def get_permutations(len_dataset: int,
                     n_permutations: int = None) -> List[Tuple[int, List]]:
    """
    Create dataset permutations

    :param len_dataset:
    :param n_permutations:
    :return:
    """
    seed_everything(__STANDARD_SEED)

    seeds = torch.randint(
        high=__UPPERBOUND,
        size=(n_permutations if n_permutations else 1,)
    )

    permutations = []
    for seed in seeds:
        seed_everything(seed)
        permutations.append((seed, randperm(len_dataset)))

    return permutations

def split(dataset, permutation, percentage_train: int = 0.6):
    size_tr, size_va, _ = define_size_sets(dataset, percentage_train)
    shuffled_dataset = dataset[permutation]

    tr_set = shuffled_dataset[:size_tr]
    va_set = shuffled_dataset[size_tr:size_tr + size_va]
    te_set = shuffled_dataset[size_tr + size_va:]

    idx_tr_set = permutation[:size_tr]
    idx_va_set = permutation[size_tr:size_tr + size_va]
    idx_te_set = permutation[size_tr + size_va:]

    return (tr_set, va_set, te_set), (idx_tr_set, idx_va_set, idx_te_set)

def define_size_sets(dataset, percentage_train: int = 0.6):
    """Return the size of the dataset for the training, validation and test sets"""
    size_tr = int(percentage_train * len(dataset))
    size_va = int(((1 - percentage_train) / 2) * len(dataset))

    return size_tr, size_va, size_va

def generate(dataset_name: str,
             n_permutations: int,
             percentage_train: int = 0.6) -> Tuple:
    """
    Generate data with split
    1) Shuffle the dataset by generating a permutation
    2) Split the data given the generated permutation
    """

    dataset = TUDataset(root=f'data/{dataset_name.upper()}',
                        name=dataset_name.upper())
    permutations = get_permutations(len_dataset=len(dataset),
                                    n_permutations=n_permutations)

    sets = []
    indices = []

    for seed, permutation in permutations:
        sets_data, indices_data = split(dataset,
                                        permutation,
                                        percentage_train)

        sets.append(sets_data)
        indices.append(indices_data)

    return sets, indices, permutations
