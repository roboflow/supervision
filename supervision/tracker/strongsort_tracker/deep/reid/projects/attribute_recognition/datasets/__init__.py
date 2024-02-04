from __future__ import division, print_function, absolute_import

from .pa100k import PA100K

__datasets = {'pa100k': PA100K}


def init_dataset(name, **kwargs):
    avai_datasets = list(__datasets.keys())
    if name not in avai_datasets:
        raise ValueError(
            'Invalid dataset name. Received "{}", '
            'but expected to be one of {}'.format(name, avai_datasets)
        )
    return __datasets[name](**kwargs)
