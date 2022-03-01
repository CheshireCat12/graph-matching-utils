from graph_pkg_utils.graph_generator.graph_generator import generate


def main(args):
    """"""
     = generate(args.dataset, args.format, args.n_permutations, args.size_train)



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
                        help='Specify the folder where the generated graphs have to be saved.')
    args = parser.parse_args()
    main(args)

