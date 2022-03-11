from argparse import ArgumentParser
from graph_pkg_utils.graph_generator.graph_generator import generate
from graph_pkg_utils.graph_generator.graph_converter import convert_and_save

import os
import numpy as np
from pathlib import Path

def main(args):
    """"""
    data = generate(args.dataset,
                    args.n_permutations,
                    args.size_train)

    splits = ('train', 'validation', 'test')

    for set_split, index_split, (seed, permutation) in zip(*data):
        folder = os.path.join(args.folder,
                              args.dataset,
                              args.level,
                              str(int(seed)))
        Path(folder).mkdir(parents=True, exist_ok=True)

        np.save(os.path.join(folder, 'permutations.npy'), permutation)

        for set_, indices, name_set in zip(set_split, index_split, splits):
            convert_and_save(name_set=name_set,
                             dataset=set_,
                             indices=indices,
                             folder=folder)


if __name__ == '__main__':
    parser = ArgumentParser(description='Run Experiments')

    # Parameters for the data generation
    parser.add_argument('--dataset',
                        type=str,
                        required=True,
                        help='Specify the dataset to generate')
    parser.add_argument('--format',
                        type=str,
                        default='graphml',
                        help='Specify the format of the output files')
    parser.add_argument('--n-permutations',
                        type=int,
                        help='Specify the number of splits to generate with different seeds')
    parser.add_argument('--size-train',
                        type=float,
                        default=0.6,
                        help='Specify the percentage in the training set.\n'
                             'The remaining graphs are equally split into the validation and test sets.')
    parser.add_argument('--folder',
                        type=str,
                        required=True,
                        help='Specify the folder where the generated graphs have to be saved.')
    parser.add_argument('--level',
                        type=str,
                        default='100',
                        help='Specify the level of the generated graphs.')
    args = parser.parse_args()
    main(args)
