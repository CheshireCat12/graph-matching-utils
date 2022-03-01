import torch

from torch import randperm
from torch_geometric import seed_everything
from torch_geometric.datasets import TUDataset
from typing import List, Tuple

__STANDARD_SEED = 42
__UPPERBOUND = 99999


def get_permutations(len_dataset: int,
                     n_permutations: int=None) -> List[Tuple[int, List]]:
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


def generate(dataset_name: str,
             n_permutations: int,
             size_train: int=0.6,
             format: str='graphml') -> Tuple:
    """
    Generate data with split
    1) Shuffle the dataset by generating a permutation
    2) Split the data given the generated permutation
    """

    dataset = TUDataset(root=f'data/{dataset_name.upper()}',
                        name=dataset_name.upper())
    permutations = get_permutations(len_dataset=len(dataset),
                                    n_permutations=n_permutations)

    size_tr = int(size_train * len(dataset))
    size_va = int(((1 - size_train) / 2 ) * len(dataset))

    sets = []
    indices = []

    for seed, permutation in permutations:
        shuffled_dataset = dataset[permutation]

        tr_set = shuffled_dataset[:size_tr]
        va_set = shuffled_dataset[size_tr:size_tr + size_va]
        te_set = shuffled_dataset[size_tr + size_va:]

        idx_tr_set = permutation[:size_tr]
        idx_va_set = permutation[size_tr:size_tr + size_va]
        idx_te_set = permutation[size_tr + size_va:]

        sets.append((tr_set, va_set, te_set))
        indices.append((idx_tr_set, idx_va_set, idx_te_set))

    return sets, indices, permutations
