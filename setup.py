from setuptools import setup, find_packages

setup(
    name='graph-matching-utils',
    version='0.1.0',
    packages=find_packages(include=['data_generator', 'data_generator.*']),
    install_requires=[
        'argparse',
        'torch',
        'numpy',
        'torch_scatter',
        'torch_sparse',
        'torch_geometric',
    ]
)
