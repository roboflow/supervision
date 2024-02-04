"""
Compute channel-wise mean and standard deviation of a dataset.

Usage:
$ python compute_mean_std.py DATASET_ROOT DATASET_KEY

- The first argument points to the root path where you put the datasets.
- The second argument means the specific dataset key.

For instance, your datasets are put under $DATA and you wanna
compute the statistics of Market1501, do
$ python compute_mean_std.py $DATA market1501
"""
import argparse

import torchreid


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=str)
    parser.add_argument('sources', type=str)
    args = parser.parse_args()

    datamanager = torchreid.data.ImageDataManager(
        root=args.root,
        sources=args.sources,
        targets=None,
        height=256,
        width=128,
        batch_size_train=100,
        batch_size_test=100,
        transforms=None,
        norm_mean=[0., 0., 0.],
        norm_std=[1., 1., 1.],
        train_sampler='SequentialSampler'
    )
    train_loader = datamanager.train_loader

    print('Computing mean and std ...')
    mean = 0.
    std = 0.
    n_samples = 0.
    for data in train_loader:
        data = data['img']
        batch_size = data.size(0)
        data = data.view(batch_size, data.size(1), -1)
        mean += data.mean(2).sum(0)
        std += data.std(2).sum(0)
        n_samples += batch_size

    mean /= n_samples
    std /= n_samples
    print('Mean: {}'.format(mean))
    print('Std: {}'.format(std))


if __name__ == '__main__':
    main()
