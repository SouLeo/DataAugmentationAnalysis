"""
This file is responsible for generating the data-sparse IMDB dataset from
the torchnlp.dataset package.
"""

import torch
from torch.utils.data import DataLoader
from torchnlp.datasets import imdb_dataset


def get_low_resource_imdb_dataset(ds_size: int) -> DataLoader:
    """
    The function pulls the full IMDB sentiment analysis training set.
    Based on the user inputted dataset size, the number of positive and
    negative labels in the newly created, data-sparse data loader will be equal.
    This function also uses a random method to sample pos/neg examples
    from the original dataset.

    :param ds_size: Specifies the imdb dataset size which includes positive and negative examples.
    Positive and negative sentiment examples are equally represented by construction.

    :return: DataLoader of IMDB dataset
    """
    assert ds_size % 2 == 0, "Dataset size must be an even number!"

    # Generate random indices to gather random examples from the original dataset
    indx_probabs = torch.rand(ds_size)

    # Separate examples by sentiment to ensure consistent/representative label distribution
    # even when sampling from smaller data set
    pos_ds = imdb_dataset(train=True, sentiments=['pos'])
    neg_ds = imdb_dataset(train=True, sentiments=['neg'])

    pos_ex_len = len(pos_ds)
    neg_ex_len = len(neg_ds)

    pos_indices = (indx_probabs[:int(len(indx_probabs)/2)]*pos_ex_len).type(torch.int32)
    neg_indices = (indx_probabs[int(len(indx_probabs)/2):]*neg_ex_len).type(torch.int32)

    # Construct dataset with positive and negative examples
    sampled_ds = []
    for p_ex, n_ex in zip(pos_indices, neg_indices):
        sampled_ds.append(pos_ds[p_ex])
        sampled_ds.append(neg_ds[n_ex])

    return DataLoader(sampled_ds)


if __name__ == '__main__':
    ds = get_low_resource_imdb_dataset(ds_size=10)
